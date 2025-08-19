import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchinfo import summary

import os
import glob
import time
import datetime as dt
from netCDF4 import Dataset as nc_Dataset
from netCDF4 import date2num, num2date
import pandas as pd
import numpy as np
import math
import xarray as xr

from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.SR_UNet_simple import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.StatObjectConstructor import *
from FunctionsAndClasses.utils import *
from FunctionsAndClasses.CONSTANTS import *

from FunctionsAndClasses.Dataset_TESTING import *

#######

TRAINED_MODEL_SAVEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Trained_models"
TRAINING_LOG_FILEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Training_logs/training_log.txt"

test_model_attrs = DefineModelAttributes(is_train=False,
                                         with_terrains=['diff'],
                                         predictor_vars=['t2m'],
                                         target_vars=['t2m'],
                                         BATCH_SIZE=18,
                                         NUM_EPOCHS=100)

NUM_GPUS_TO_USE=3
MULTIGPU_BATCH_SIZE = test_model_attrs.BATCH_SIZE*NUM_GPUS_TO_USE

ds_test = Dataset_TEST(load_xr_into_memory=False, with_terrain_diff=True)

dl_test = DataLoader(ds_test, 
                     batch_size=MULTIGPU_BATCH_SIZE, 
                     shuffle=True, 
                     num_workers=4*NUM_GPUS_TO_USE, 
                     pin_memory=True)

model = SR_UNet_simple(n_channels_in=2, 
                        n_channels_out=1)
device = torch.device("cuda")
model = nn.DataParallel(model, device_ids=[i for i in range(NUM_GPUS_TO_USE)])
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=[0.5,0.999])
loss_function = torch.nn.L1Loss()

model.train()

lowest_loss = 999
for epoch in range(1,test_model_attrs.NUM_EPOCHS+1):
    epoch_loss = 0.0
    start = time.time()
    for i, (inputs,labels) in enumerate(dl_test):    
        start_dl = time.time()
        print(f"Starting batch {i+1}/{len(dl_test)} in epoch {epoch}")
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.float()) #weird datatype mismatching... for some reason it's seeing HRRR data as double
        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        #print(f"Done with batch {i+1}/{len(dl_test)}. Time taken = {time.time()-start_dl:.1f} sec. Running loss = {epoch_loss/(i+1):.5f}")
        
    end = time.time() 
    
    if epoch_loss <= lowest_loss: #only save models that have lower loss than previous best
        lowest_loss = epoch_loss
        torch.save(model.module.state_dict(), f"{TRAINED_MODEL_SAVEPATH}/{test_model_attrs.savename}.pt")

    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        file.write(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(dl_test):.5f} | Time for epoch = {end-start:.1f} sec \n")