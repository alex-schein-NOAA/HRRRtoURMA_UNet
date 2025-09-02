import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import os
import time
import datetime as dt
from netCDF4 import Dataset as nc_Dataset
import pandas as pd
import numpy as np
import xarray as xr

from SR_UNet_simple import SR_UNet_simple
from DefineModelAttributes import DefineModelAttributes
from HRRR_URMA_Datasets_AllVars import HRRR_URMA_Dataset_AllVars

torch.manual_seed(42)

################################################################################

TRAINING_LOG_FILEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Training_logs/training_log.txt"

NUM_GPUS_TO_USE = 3

BATCH_SIZE = 256*NUM_GPUS_TO_USE #Be careful when initializing with this; should use BATCH_SIZE in DataLoader but use BATCH_SIZE/NUM_GPUS in model setup call
NUM_EPOCHS = 1000

PREDICTOR_VAR_LIST = ["t2m", "d2m", "pressurf", "u10m", "v10m"]

for i in range(len(PREDICTOR_VAR_LIST)-1):

    #Not bothering with BATCH_SIZE since default = 256 which is fine for now
    current_model = DefineModelAttributes(predictor_vars=PREDICTOR_VAR_LIST,
                                          target_vars=["t2m"])

    if not os.path.exists(f"/scratch/RTMA/alex.schein/CNN_Main/Trained_models/{current_model.savename}.pt"):
        appendation = ""
    else: #model under that name already exists
        print(f"A model under {current_model.savename} already exists in the save directory! Change that model's name, or otherwise ensure non overlapping names")
        appendation = "_nonoverlapping"
        print(f"Running with model name {current_model.savename}{appendation}")
    
    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Starting {current_model.savename} | Current time = {current_time} \n")
    
    current_model.create_dataset()
    current_model_dataloader = DataLoader(current_model.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4*NUM_GPUS_TO_USE, pin_memory=True)

    with open(TRAINING_LOG_FILEPATH, "a") as file:
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Data loaded | Current time = {current_time} \n")
    

    model = SR_UNet_simple(n_channels_in=current_model.num_channels_in)
    device = torch.device("cuda")
    model = nn.DataParallel(model, device_ids=[i for i in range(NUM_GPUS_TO_USE)])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=[0.5,0.999]) #torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_function = torch.nn.L1Loss()
    lowest_loss = 999

    model.train()
    
    for epoch in range(1,NUM_EPOCHS+1):
        epoch_loss = 0.0
        start = time.time()
        for i, (inputs,labels) in enumerate(current_model_dataloader):    
            
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = model(inputs.float()) #weird datatype mismatching... for some reason it's seeing HRRR data as double
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
            
        end2 = time.time() 
        
        if epoch_loss <= lowest_loss:
            lowest_loss = epoch_loss
            #Note need to use model.module when doing DataParallel, else saved model needs to be loaded on the same # of GPUs as training
            torch.save(model.module.state_dict(), f"/scratch/RTMA/alex.schein/CNN_Main/Trained_models/{current_model.savename}{appendation}_TEMP.pt")
        
        # write log at every interval so we can read back all the history if needed
        with open(TRAINING_LOG_FILEPATH, "a") as file: 
            file.write(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(current_model_dataloader):.5f} | Time for epoch = {end2-start:.1f} sec \n")
    
    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Training finished | Current time = {current_time} \n")
        
    os.rename(f"/scratch/RTMA/alex.schein/CNN_Main/Trained_models/{current_model.savename}{appendation}_TEMP.pt", 
              f"/scratch/RTMA/alex.schein/CNN_Main/Trained_models/{current_model.savename}{appendation}.pt") #so if training is interrupted, previously saved model under the same name isn't wiped out

    
    PREDICTOR_VAR_LIST.pop() #starts w/ all predictors, loses one each time until we're down to just t2m and d2m (have already done t2m only)
