# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# import os
# import time
# import datetime as dt
# from netCDF4 import Dataset as nc_Dataset
# import pandas as pd
# import numpy as np
# import xarray as xr

from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.CONSTANTS import *

######################################################################################################################################################

class HRRR_URMA_Dataset_AllVars(Dataset):
    def __init__(self, 
                 is_train=True,
                 months=[1,12],  
                 hours="all", 
                 forecast_lead_time = 1, 
                 normalization_scheme = "all times",
                 with_terrains=["hrrr","urma","diff"], 
                 with_yearly_time_sig = False, 
                 with_hourly_time_sig = False,
                 predictor_vars = ["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"],
                 target_vars = ["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"]
                ):
        
        """
        - is_train --> bool to load either training or testing datasets
        - months --> 2-tuple of ints 1-12; first entry = start month, second entry = end month. (AS OF 2025-08) SHOULD ONLY BE [1,12]
            - Note this means only a continuous range of months can be selected, and all days in those months will be selected
            - Note also this means only dates WITHIN a year can be selected - cross-year selection currently not allowed (causes issues with the fact we don't have 2020 data, so any cross-year times in the training set can only contain 2 years - and we don't have any non-2024 data in the testing set...)
            - if is_train=True, all years 2021/22/23 will have data selected from this range
            - if is_train=False, only 2024 data will be selected
        - hours --> either str of "all" to include all times, or a subset of list of ints [0,1,..23]. Selects valid hours to include.
            - Can be discontinuous, but almost certainly won't be (why bother?)
        - forecast_lead_time --> int from 0 to 23, specifying HRRR forecast lead time
            - Only used when loading the data - HRRR files need to have forecast lead time specified in the filename
            - URMA not affected by this
            - Currently only a value of 1 works
        - normalization_scheme --> string to select how the data is normalized 
            - (REMOVED AS OF 2025-08-12) "per hour" = each hour of data is normalized w.r.t the mean and stddev of only that hour across the dataset. Note that this will return lists containing each hour's mean and stddev, in the order given in "hours".
            - "all times" = takes the mean and stddev over the entire dataset and normalizes w.r.t that. Note that this only returns lists containing only ONE mean and stddev for the whole dataset! 
        - with_terrains --> list of terrains to include as separate, normalized channels (can be None to not include terrain):
            - "hrrr" --> include NORMALIZED 2.5 km downscaled HRRR terrain field as a separate channel
            - "urma" --> include NORMALIZED 2.5 km nam_smarttopconus2p5 terrain field as a separate channel
            - "diff" --> include NORMALIZED map of the difference between HRRR and URMA terrain as a separate channel
            - !!!! NOTE: if both "hrrr" and "urma" are included, then HRRR terrain field will be normalized with respect to the mean/stddev of the URMA terrain! According to Ryan Lagerquist, best to use one norm for both terrains in this case
        - (REMOVED AS OF 2025-08) with_yearly_time_sig --> bool to include yearly time signatures. Changed from sinusoidal to linear as of 7/7
        - (REMOVED AS OF 2025-08) with_hourly_time_sig --> bool to include time-of-day signatures. Changed from sinusoidal to linear as of 7/7
        """

        #########################################
        ## Initialize vars

        self.C = CONSTANTS()
        
        self.is_train = is_train
        print(f"is_train = {self.is_train}")
        self.months = months
        if self.months != [1,12]:
            print(f"Not all months selected - MEANS AND STDDEVS FOR DATA MAY NOT MATCH! Be careful!")
        self.hours = hours
        if self.hours == "all": #we always need a list of ints 
            self.hours = [i for i in range(24)]
        else:
            print(f"Not all hours selected - MEANS AND STDDEVS FOR DATA MAY NOT MATCH! Be careful!")
        
        self.forecast_lead_time = forecast_lead_time

        if normalization_scheme == "per hour":
            print(f"''per hour'' normalization has been removed. Using ''all times'' instead")
            self.normalization_scheme = "all times"
        elif normalization_scheme == "all times":
            self.normalization_scheme = normalization_scheme
        else:
            print(f"Must enter ''all times'' for normalization_scheme. Defaulting to ''all times''")
            self.normalization_scheme = "all times"
    
        self.with_terrains = with_terrains
        self.with_hrrr_terrain = False
        self.with_urma_terrain = False
        self.with_diff_terrain = False
        if self.with_terrains is not None:
            if "hrrr".casefold() in [x.casefold() for x in self.with_terrains]: #more complex check to allow for whatever casing, as all lowercase might not necessarily be the best
                self.with_hrrr_terrain = True
            if "urma".casefold() in [x.casefold() for x in self.with_terrains]:
                self.with_urma_terrain = True
            if "diff".casefold() in [x.casefold() for x in self.with_terrains]:
                self.with_diff_terrain = True
        
        self.with_yearly_time_sig = with_yearly_time_sig
        self.with_hourly_time_sig = with_hourly_time_sig

        self.predictor_vars = predictor_vars
        self.target_vars = target_vars

        ######################################
        ## Initialize arrays to contain each variable's data/attributes
        
        self.xr_datasets_pred = [] #list of length == len(self.predictor_vars), containing the raw xarray dataset for each predictor variable, in order
        self.xr_datasets_targ = [] #same, but for target vars

        #These will be a list of lists; these sublists will be of length 1 (if normalization_scheme == "all times")
        self.datasets_pred_normed_means = [] 
        self.datasets_targ_normed_means = []
        self.datasets_pred_normed_stddevs = [] 
        self.datasets_targ_normed_stddevs = []

        
        #########################################
        ## Normalize terrain field

        terrain_path_hrrr, terrain_path_urma = self.get_var_filepath("terrain")
        if self.with_hrrr_terrain:
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True, engine='cfgrib')
        if self.with_urma_terrain:
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True, engine='cfgrib')
        if self.with_diff_terrain: #needed in case one or both of HRRR or/and URMA terrain aren't included, but their difference is
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True, engine='cfgrib')
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True, engine='cfgrib')
                
        self.normalize_terrain()

        #########################################
        ## Load predictor and target datasets
        # Unlike previous dataloader, data is NOT loaded into memory, as it takes too long for the larger western domain and doing so doesn't speed up training
        # Also unlike previous, normalization of the data does NOT happen here, but rather in __getitem__
        # Might want to explore ways of caching the data, but doesn't seem to be needed for now

        if is_train:
            train_test_str = 'train'
        else:
            train_test_str = 'test'
        
        for var_name in self.predictor_vars:
            start = time.time()
            data_save_path = self.get_var_filepath(var_name, is_pred=True)
            self.xr_datasets_pred.append(xr.open_dataarray(data_save_path, decode_timedelta=True, engine='cfgrib'))
            self.datasets_pred_normed_means.append(self.C.hrrr_means_dict[train_test_str][var_name])
            self.datasets_pred_normed_stddevs.append(self.C.hrrr_stddevs_dict[train_test_str][var_name])
            print(f"Predictor data for {var_name} loaded. Time taken = {time.time()-start:.1f} sec")

        for var_name in self.target_vars:
            start = time.time()
            data_save_path = self.get_var_filepath(var_name, is_pred=False)
            self.xr_datasets_targ.append(xr.open_dataarray(data_save_path, decode_timedelta=True, engine='cfgrib'))
            self.datasets_targ_normed_means.append(self.C.urma_means_dict[train_test_str][var_name]) #(8/27) was mistakenly appending HRRR data here... whoops... models trained before this date are no longer valid with this dataset, unless manually tweaking targ means and stddevs to match pred means/stddevs
            self.datasets_targ_normed_stddevs.append(self.C.urma_stddevs_dict[train_test_str][var_name])
            print(f"Target data for {var_name} loaded. Time taken = {time.time()-start:.1f} sec")
            
        if self.with_hourly_time_sig or self.with_yearly_time_sig:
            print(f"Time signatures removed as of 2025-08")

        self.predictor_indices = np.arange(len(self.xr_datasets_pred[0]))
        self.target_indices = np.arange(len(self.xr_datasets_targ[0]))
        assert len(self.predictor_indices) == len(self.target_indices), "Predictor indices array should be of the same length as the target indices array"

        print("DATASET CONSTRUCTION DONE")
        
    ######################################### FUNCTIONS #########################################
    
    def __len__(self):
        return len(self.predictor_indices)

    #########################################
    
    def normalize_terrain(self): #Put into a separate method just to clean up main
        if self.with_hrrr_terrain and not self.with_urma_terrain:
            terrain_hrrr = self.xr_terrain_hrrr.data
            self.terrain_hrrr_mean = np.mean(terrain_hrrr)
            self.terrain_hrrr_std = np.std(terrain_hrrr)
            self.terrain_hrrr_normed = (terrain_hrrr - self.terrain_hrrr_mean)/self.terrain_hrrr_std
        if self.with_urma_terrain:
            terrain_urma = self.xr_terrain_urma.data
            self.terrain_urma_mean = np.mean(terrain_urma)
            self.terrain_urma_std = np.std(terrain_urma)
            self.terrain_urma_normed = (terrain_urma - self.terrain_urma_mean)/self.terrain_urma_std
        if self.with_diff_terrain:
            terrain_hrrr = self.xr_terrain_hrrr.data
            terrain_urma = self.xr_terrain_urma.data
            terrain_diff = terrain_hrrr-terrain_urma
            self.terrain_diff_mean = np.mean(terrain_diff)
            self.terrain_diff_std = np.std(terrain_diff)
            self.terrain_diff_normed = (terrain_diff - self.terrain_diff_mean)/self.terrain_diff_std
        if self.with_hrrr_terrain and self.with_urma_terrain: #use the same mean/std to norm both. Using URMA at the moment
            # Note in this case the URMA terrain stuff has already been done
            terrain_hrrr = self.xr_terrain_hrrr.data
            self.terrain_hrrr_mean = np.mean(terrain_hrrr) #for diagnostics only
            self.terrain_hrrr_std = np.std(terrain_hrrr) #for diagnostics only
            self.terrain_hrrr_normed = (terrain_hrrr - self.terrain_urma_mean)/self.terrain_urma_std
        if self.with_terrains is not None:
            print(f"Terrain normalization done for {self.with_terrains}") 
        return

    #########################################

    def get_var_filepath(self, var_name, is_pred=True):
        """
        Inputs: 
            - var_name as string (e.g. "t2m", "d2m", etc). Can be "terrain" to load terrain data as well
            - is_pred = bool to control if predictor or target filepath is returned
        Output: relevant filepath(s) for that var
        """
        
        if var_name == "terrain":
            terrain_path_hrrr = f"{self.C.DIR_TRAIN_TEST}/terrain_subset_HRRR_2p5km.grib2"
            terrain_path_urma = f"{self.C.DIR_TRAIN_TEST}/terrain_subset_URMA_2p5km.grib2"
            return terrain_path_hrrr, terrain_path_urma
        else:
            if self.is_train:
                if is_pred:
                    data_save_path = f"{self.C.DIR_TRAIN_TEST}/train_hrrr_alltimes_{var_name}_f{str(self.forecast_lead_time).zfill(2)}.grib2"
                else:
                    data_save_path = f"{self.C.DIR_TRAIN_TEST}/train_urma_alltimes_{var_name}.grib2"
            else:
                if is_pred:
                    data_save_path = f"{self.C.DIR_TRAIN_TEST}/test_hrrr_alltimes_{var_name}_f{str(self.forecast_lead_time).zfill(2)}.grib2"
                else:
                    data_save_path = f"{self.C.DIR_TRAIN_TEST}/test_urma_alltimes_{var_name}.grib2"
            
            return data_save_path

    #########################################

    def get_normed_data_at_idx(self, i, idx, is_pred=True):
        """
        Helper function to be used in __getitem__. 
        Note: relies on datasets/means/stddevs in list to be ordered the same as the order in predictor_vars or target_vars, but the lists are constructed this way in the main function, so not that big of a deal for use in __getitem__. Be careful if calling this in an outside script, though!
        Note: only works for hours = "all", months=[1,12], but this is default behavior so not much of a restiction
        Inputs:
            - i = index of current variable in relation to predictor_vars or target_vars
            - idx = actual index to select
            - is_pred = bool to select from the correct list

        Outputs:
            - output of variable @ i, index=idx, normed appropriately, and appended with newaxis for concat purposes
        """
        if is_pred: #select from pred data
            return ((self.xr_datasets_pred[i][idx].data - self.datasets_pred_normed_means[i])/self.datasets_pred_normed_stddevs[i])[np.newaxis,:,:]
        else: #select from targ data
            return ((self.xr_datasets_targ[i][idx].data - self.datasets_targ_normed_means[i])/self.datasets_targ_normed_stddevs[i])[np.newaxis,:,:]

    #########################################

    def __getitem__(self, idx):
        ## Start with the first variable for each of predictor and target
        predictor = self.get_normed_data_at_idx(0, idx, is_pred=True)
        target = self.get_normed_data_at_idx(0, idx, is_pred=False)

        ## Add new channels for as many variables as we have
        if len(self.predictor_vars) > 1:
            for i, var_name in enumerate(self.predictor_vars[1:]): #don't double up on index 0
                ds = self.get_normed_data_at_idx(i, idx, is_pred=True)
                predictor = np.concatenate((predictor, ds), axis=0)
        if len(self.target_vars) > 1:
            for i, var_name in enumerate(self.target_vars[1:]): #don't double up on index 0
                ds = self.get_normed_data_at_idx(i, idx, is_pred=False)
                target = np.concatenate((target, ds), axis=0)

        ## Add terrain layers as last channels
        if self.with_hrrr_terrain:
            predictor = np.concatenate((predictor, self.terrain_hrrr_normed[np.newaxis,:,:]), axis=0)
        if self.with_urma_terrain:
            predictor = np.concatenate((predictor, self.terrain_urma_normed[np.newaxis,:,:]), axis=0)
        if self.with_diff_terrain:
            predictor = np.concatenate((predictor, self.terrain_diff_normed[np.newaxis,:,:]), axis=0)

        
        return (predictor), (target)
