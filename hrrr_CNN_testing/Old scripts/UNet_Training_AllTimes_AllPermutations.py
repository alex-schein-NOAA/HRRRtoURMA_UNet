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
NUM_EPOCHS = 750

################################################################################

for TERRAIN_LIST in [["hrrr","urma","diff"], ["diff"]]:
    for W_YEAR_SIG in [True, False]:
        for W_HOUR_SIG in [True, False]:

            W_HRRR_TERR = 0
            W_URMA_TERR = 0
            if "hrrr" in TERRAIN_LIST:
                W_HRRR_TERR=1
            if "urma" in TERRAIN_LIST:
                W_URMA_TERR=1

            savename = f"UNSim_BS{int(BATCH_SIZE/NUM_GPUS_TO_USE)}_NE{NUM_EPOCHS}_months1-12_tHRRR{int(W_HRRR_TERR)}_tURMA{int(W_URMA_TERR)}_tDIFF1_sigYr{int(W_YEAR_SIG)}_sigHr{int(W_HOUR_SIG)}"
            with open("/scratch/RTMA/alex.schein/hrrr_CNN_testing/training_log.txt", "a") as file: 
                now = dt.datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"Starting {savename} | Current time = {current_time} \n")
            
            train_ds = HRRR_URMA_Dataset_AllTimes_AnyDates_AnyTerrains(is_train=True,
                                                                      months=[1,12],  
                                                                      hours="all", 
                                                                      with_terrains=TERRAIN_LIST, 
                                                                      with_yearly_time_sig=W_YEAR_SIG, 
                                                                      with_hourly_time_sig=W_HOUR_SIG)

            train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
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

            log_epoch_interval = 50
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
                    torch.save(model.module.state_dict(), f"/scratch/RTMA/alex.schein/hrrr_CNN_testing/Trained models/{savename}_TEMP.pt")
                
                #if epoch % log_epoch_interval == 0:
                with open("/scratch/RTMA/alex.schein/hrrr_CNN_testing/training_log.txt", "a") as file: 
                    file.write(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(train_dataloader):.3f} | Time for epoch = {end2-start:.1f} sec \n")
            
            with open("/scratch/RTMA/alex.schein/hrrr_CNN_testing/training_log.txt", "a") as file: 
                now = dt.datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"Training finished | Current time = {current_time} \n")
                
            os.rename(f"/scratch/RTMA/alex.schein/hrrr_CNN_testing/Trained models/{savename}_TEMP.pt", 
                      f"/scratch/RTMA/alex.schein/hrrr_CNN_testing/Trained models/{savename}.pt") #so if training is interrupted, previously saved model under the same name isn't wiped out







            