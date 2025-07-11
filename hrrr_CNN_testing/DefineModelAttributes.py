import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import time
import datetime as dt
from netCDF4 import Dataset as nc_Dataset
import pandas as pd
import numpy as np
import xarray as xr

from HRRR_URMA_Datasets_AllVars import *

######################################################################################################################################################

class DefineModelAttributes():
    def __init__(self,
                 BATCH_SIZE=256,
                 NUM_EPOCHS=1000,
                 is_train=True,
                 months=[1,12],  
                 hours="all", 
                 forecast_lead_time=1, 
                 normalization_scheme="all times",
                 with_terrains=["diff"], 
                 with_yearly_time_sig=True, 
                 with_hourly_time_sig=True,
                 predictor_vars=["t2m", "d2m", "pressurf", "u10m", "v10m"],
                 target_vars=["t2m", "d2m", "pressurf", "u10m", "v10m"]):
        
        """
            See HRRR_URMA_Datasets_AllVars for variable definitions/restrictions
        """

        #########################################
    
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.is_train = is_train
        self.months = months
        self.hours = hours
        self.forecast_lead_time = forecast_lead_time
        self.normalization_scheme = normalization_scheme
        self.with_terrains = with_terrains
        self.with_yearly_time_sig = with_yearly_time_sig
        self.with_hourly_time_sig = with_hourly_time_sig
        self.predictor_vars = predictor_vars
        self.target_vars = target_vars

        self.savename = self.create_save_name()
        
    ######################################### FUNCTIONS #########################################
    
    def create_save_name(self):
        
        ## Define all optional/constructive arguments
        month_str = ""
        hours_str = ""
        forecast_str = ""
        terrain_str = ""
        time_sig_list = []
        if self.months != [1,12]:
            month_str = f"months{self.months[0]}-{self.months[1]}"
        if self.hours != "all":
            hours_str = f"hours{'-'.join([str(hour) for hour in self.hours])}"
        if self.forecast_lead_time != 1:
            forecast_str = f"f{str(forecast_lead_time).zfill(2)}"
        terrain_str = "_".join([f"t{x[0].capitalize()}" for x in self.with_terrains])
        if self.with_yearly_time_sig:
            time_sig_list.append("sY")
        if self.with_hourly_time_sig:
            time_sig_list.append("sH")
        time_sig_str = "_".join(time_sig_list)

        optional_str_list = [month_str, hours_str, forecast_str, terrain_str, time_sig_str]
        optional_str = "_".join([x for x in optional_str_list if x != ""])
        
        # Doesn't play nice if defined within the f-string
        pred_str = "-".join(self.predictor_vars)
        targ_str = "-".join(self.target_vars)
        
        savename = f"BS{self.BATCH_SIZE}" \
                   f"_NE{self.NUM_EPOCHS}" \
                   f"_{optional_str}" \
                   f"_pred({pred_str})" \
                   f"_targ({targ_str})"
        
        return savename

    #########################################

    def create_dataset(self):
        # Creates the requisite Dataset for use in Pytorch Dataloader
        # NOT called by default (due to calculation expense) - must be invoked by calling function 
        self.dataset = HRRR_URMA_Dataset_AllVars(is_train = self.is_train,
                                                 months = self.months,  
                                                 hours = self.hours, 
                                                 forecast_lead_time = self.forecast_lead_time, 
                                                 normalization_scheme = self.normalization_scheme,
                                                 with_terrains = self.with_terrains, 
                                                 with_yearly_time_sig = self.with_yearly_time_sig, 
                                                 with_hourly_time_sig = self.with_hourly_time_sig,
                                                 predictor_vars = self.predictor_vars,
                                                 target_vars = self.target_vars)

        self.num_channels = np.shape(self.dataset[0][0])[0]
        return