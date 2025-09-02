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

from FunctionsAndClasses.CONSTANTS import *

######################################################################################################################################################

class Dataset_TEST(Dataset):
    def __init__(self, load_xr_into_memory=False, with_terrain_diff=False):

        self.CONSTS = CONSTANTS()
        
        HRRR_PATH = self.CONSTS.DIR_TRAIN_TEST
        URMA_PATH = self.CONSTS.DIR_TRAIN_TEST
        TERRAIN_PATH = self.CONSTS.DIR_TRAIN_TEST
        var_name = 't2m'
        
        
        self.load_xr_into_memory = load_xr_into_memory
        self.with_terrain_diff = with_terrain_diff
        
        self.xr_hrrr = xr.open_dataarray(f"{HRRR_PATH}/train_hrrr_alltimes_{var_name}_f01.grib2", 
                                         decode_timedelta=True, 
                                         engine="cfgrib")
        self.xr_urma = xr.open_dataarray(f"{URMA_PATH}/train_urma_alltimes_{var_name}.grib2", 
                                         decode_timedelta=True, 
                                         engine='cfgrib')
        
        if self.load_xr_into_memory:
            print(f"Loading {var_name} HRRR xarray dataset into memory")
            start = time.time()
            self.xr_hrrr_loaded = self.xr_hrrr.data
            print(f"Data loaded. Time taken = {time.time() - start:.1f} sec")

            print(f"Loading {var_name} URMA xarray dataset into memory")
            start = time.time()
            self.xr_urma_loaded = self.xr_urma.data
            print(f"Data loaded. Time taken = {time.time() - start:.1f} sec")
        else:
            print(f"{var_name} xarray datasets loaded (NOT in memory)")
        
        if self.with_terrain_diff:
            print(f"Loading terrain datasets")
            self.xr_terrain_hrrr = xr.open_dataarray(f"{TERRAIN_PATH}/terrain_subset_HRRR_2p5km.grib2",
                                                     decode_timedelta=True,
                                                     engine="cfgrib")
            self.xr_terrain_urma = xr.open_dataarray(f"{TERRAIN_PATH}/terrain_subset_URMA_2p5km.grib2",
                                                     decode_timedelta=True,
                                                     engine="cfgrib")
            print(f"Terrain datasets loaded. Calculating difference")
            self.terrain_diff = self.xr_terrain_hrrr.data - self.xr_terrain_urma.data
            self.terrain_diff_mean = np.mean(self.terrain_diff)
            self.terrain_diff_stddev = np.std(self.terrain_diff)
            self.terrain_diff_normed = (self.terrain_diff - self.terrain_diff_mean)/self.terrain_diff_stddev
            print(f"Terrain difference DONE")
        
        self.hrrr_mean = self.CONSTS.hrrr_means_dict['test'][f"{var_name}"]
        self.hrrr_stddev = self.CONSTS.hrrr_stddevs_dict['test'][f"{var_name}"]

        self.urma_mean = self.CONSTS.urma_means_dict['test'][f"{var_name}"]
        self.urma_stddev = self.CONSTS.urma_stddevs_dict['test'][f"{var_name}"]

        
        self.predictor_indices = np.arange(len(self.xr_hrrr))
    #########################################
    
    def __len__(self):
        return len(self.predictor_indices)

    
    #########################################

    def __getitem__(self, idx):
        
        if self.load_xr_into_memory:
            predictor = ((self.xr_hrrr_loaded[idx] - self.hrrr_mean)/self.hrrr_stddev)[np.newaxis,:,:]
            target = ((self.xr_urma_loaded[idx] - self.urma_mean)/self.urma_stddev)[np.newaxis,:,:]
            if self.with_terrain_diff:
                predictor = np.concatenate((predictor, self.terrain_diff_normed[np.newaxis,:,:]), axis=0)
        else:
            predictor = ((self.xr_hrrr[idx].data - self.hrrr_mean)/self.hrrr_stddev)[np.newaxis,:,:]
            target = ((self.xr_urma[idx].data - self.urma_mean)/self.urma_stddev)[np.newaxis,:,:]
            if self.with_terrain_diff:
                predictor = np.concatenate((predictor, self.terrain_diff_normed[np.newaxis,:,:]), axis=0)
        
        return (predictor), (target)