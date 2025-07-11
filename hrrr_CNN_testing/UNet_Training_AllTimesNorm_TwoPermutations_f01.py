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

from HRRR_URMA_Datasets import *
from SR_UNet_simple import SR_UNet_simple
from utils import *

torch.manual_seed(42)

################################################################################

NUM_GPUS_TO_USE = 3

BATCH_SIZE = 256*NUM_GPUS_TO_USE 
NUM_EPOCHS = 1000 #only training 2 models this time, so probably ok to go a little further

################################################################################

# while not os.path.exists(f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/train_hrrr_alltimes_t2m_f01.nc"):
#     print("File doesn't exist yet. Sleeping for 10 seconds")
#     time.sleep(10)
    

for TERRAIN_LIST in [["diff"]]: #["hrrr","urma","diff"]
    W_HRRR_TERR = 0
    W_URMA_TERR = 0
    if "hrrr" in TERRAIN_LIST:
        W_HRRR_TERR=1
    if "urma" in TERRAIN_LIST:
        W_URMA_TERR=1

    
    for W_SIG_YR, W_SIG_HR in [[True,True]]: #,[False,False]]:
        for TIME_SIG_SCHEME in ["Direct", "Frac"]:
            savename = f"UNSim_BS{int(BATCH_SIZE/NUM_GPUS_TO_USE)}_NE{NUM_EPOCHS}_tH{int(W_HRRR_TERR)}_tU{int(W_URMA_TERR)}_tD1_sigScheme{TIME_SIG_SCHEME}_sigYr{int(W_SIG_YR)}_sigHr{int(W_SIG_HR)}"
            if not os.path.exists(f"/scratch/RTMA/alex.schein/hrrr_CNN_testing/Trained models/{savename}.pt"):
                appendation = ""
            else: #model under that name already exists
                print(f"A model under {savename} already exists in the save directory! Change that model's name, or otherwise ensure non overlapping names")
                appendation = "_nonoverlapping"
                print(f"Running with model name {savename}{appendation}")
                    
            with open("/scratch/RTMA/alex.schein/hrrr_CNN_testing/training_log.txt", "a") as file: 
                now = dt.datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"Starting {savename} | Current time = {current_time} \n")
            
            train_ds = HRRR_URMA_Dataset(is_train=True,
                                          months=[1,12],  
                                          hours="all",
                                          forecast_lead_time=1,
                                          normalization_scheme="all times",
                                          with_terrains=TERRAIN_LIST, 
                                          with_yearly_time_sig=W_SIG_YR, 
                                          with_hourly_time_sig=W_SIG_HR,
                                          time_sig_scheme=TIME_SIG_SCHEME)
        
            train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4*NUM_GPUS_TO_USE, pin_memory=True)
            with open("/scratch/RTMA/alex.schein/hrrr_CNN_testing/training_log.txt", "a") as file: 
                now = dt.datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"Data loaded | Current time = {current_time} \n")
        
            model = SR_UNet_simple(n_channels_in=np.shape(train_ds[0][0])[0])
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
                for i, (inputs,labels) in enumerate(train_dataloader):    
                    
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
                    torch.save(model.module.state_dict(), f"/scratch/RTMA/alex.schein/hrrr_CNN_testing/Trained models/{savename}{appendation}_TEMP.pt")
                
                # write log at every interval so we can read back all the history if needed
                with open("/scratch/RTMA/alex.schein/hrrr_CNN_testing/training_log.txt", "a") as file: 
                    file.write(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(train_dataloader):.5f} | Time for epoch = {end2-start:.1f} sec \n")
            
            with open("/scratch/RTMA/alex.schein/hrrr_CNN_testing/training_log.txt", "a") as file: 
                now = dt.datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"Training finished | Current time = {current_time} \n")
                
            os.rename(f"/scratch/RTMA/alex.schein/hrrr_CNN_testing/Trained models/{savename}{appendation}_TEMP.pt", 
                      f"/scratch/RTMA/alex.schein/hrrr_CNN_testing/Trained models/{savename}{appendation}.pt") #so if training is interrupted, previously saved model under the same name isn't wiped out
            