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
                 predictor_vars = ["t2m", "d2m", "pressurf", "u10m", "v10m"],
                 target_vars = ["t2m", "d2m", "pressurf", "u10m", "v10m"]):
        
        """
        - is_train --> bool to load either training or testing datasets
        - months --> 2-tuple of ints 1-12; first entry = start month, second entry = end month. 
            - Note this means only a continuous range of months can be selected, and all days in those months will be selected
            - Note also this means only dates WITHIN a year can be selected - cross-year selection currently not allowed (causes issues with the fact we don't have 2020 data, so any cross-year times in the training set can only contain 2 years - and we don't have any non-2024 data in the testing set...)
            - if is_train=True, all years 2021/22/23 will have data selected from this range
            - if is_train=False, only 2024 data will be selected
        - hours --> either str of "all" to include all times, or a subset of list of ints [0,1,..23]. Selects valid hours to include.
            - Can be discontinuous, but almost certainly won't be (why bother?)
        - forecast_lead_time --> int from 0 to 23, specifying HRRR forecast lead time
            - Only used when loading the data - HRRR files need to have forecast lead time specified in the filename
            - URMA not affected by this
        - normalization_scheme = string to select how the data is normalized 
            - "per hour" = each hour of data is normalized w.r.t the mean and stddev of only that hour across the dataset. Note that this will return lists containing each hour's mean and stddev, in the order given in "hours"
            - "all times" = takes the mean and stddev over the entire dataset and normalizes w.r.t that. Note that this only returns lists containing only ONE mean and stddev for the whole dataset! 
        - with_terrains --> list of terrains to include as separate, normalized channels (can be None to not include terrain):
            - "hrrr" --> include NORMALIZED 2.5 km downscaled HRRR terrain field as a separate channel
            - "urma" --> include NORMALIZED 2.5 km nam_smarttopconus2p5 terrain field as a separate channel
            - "diff" --> include NORMALIZED map of the difference between HRRR and URMA terrain as a separate channel
            - !!!! NOTE: if both "hrrr" and "urma" are included, then HRRR terrain field will be normalized with respect to the mean/stddev of the URMA terrain! According to Ryan Lagerquist, best to use one norm for both terrains in this case
        - with_yearly_time_sig --> bool to include yearly time signatures. Changed from sinusoidal to linear as of 7/7
        - with_hourly_time_sig --> bool to include time-of-day signatures. Changed from sinusoidal to linear as of 7/7
        """

        #########################################
        ## Initialize vars

        self.is_train = is_train
        self.months = months
        self.hours = hours
        if self.hours == "all": #we always need a list of ints 
            self.hours = [i for i in range(24)]
        
        self.forecast_lead_time = forecast_lead_time

        if normalization_scheme == "per hour":
            self.normalization_scheme = normalization_scheme
        elif normalization_scheme == "all times":
            self.normalization_scheme = normalization_scheme
        else:
            print(f"Must enter ''per hour'' or ''all times'' for normalization_scheme. Defaulting to ''all times''")
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
        self.xr_datasets_targ = [] #same idea but for target vars

        self.datasets_pred_normed = []
        self.datasets_targ_normed = []

        #These will be a list of lists; these sublists will be of length 1 (if normalization_scheme == "all times") or length == len(hours)
        self.datasets_pred_normed_means = [] 
        self.datasets_targ_normed_means = []
        self.datasets_pred_normed_stddevs = [] 
        self.datasets_targ_normed_stddevs = []

        
        #########################################
        ## Do requisite normalizations

        terrain_path_hrrr, terrain_path_urma = self.get_var_filepath("terrain")
        if self.with_hrrr_terrain:
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True)
        if self.with_urma_terrain:
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True)
        if self.with_diff_terrain: #needed in case one or both of HRRR or/and URMA terrain aren't included, but their difference is
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True)
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True)
                
        self.normalize_terrain()

        # Process predictor vars
        for var_name in self.predictor_vars:
            tmp_xr, tmp_normed_data, tmp_means, tmp_stddevs = self.process_variable(var_name, is_pred=True)
            self.xr_datasets_pred.append(tmp_xr)
            self.datasets_pred_normed.append(tmp_normed_data)
            self.datasets_pred_normed_means.append(tmp_means)
            self.datasets_pred_normed_stddevs.append(tmp_stddevs)
            del tmp_xr, tmp_normed_data, tmp_means, tmp_stddevs #just in case - almost certain this is not needed

        # Process target vars
        for var_name in self.target_vars:
            tmp_xr, tmp_normed_data, tmp_means, tmp_stddevs = self.process_variable(var_name, is_pred=False)
            self.xr_datasets_targ.append(tmp_xr)
            self.datasets_targ_normed.append(tmp_normed_data)
            self.datasets_targ_normed_means.append(tmp_means)
            self.datasets_targ_normed_stddevs.append(tmp_stddevs)
            del tmp_xr, tmp_normed_data, tmp_means, tmp_stddevs #just in case - almost certain this is not needed
            
        if self.with_hourly_time_sig or self.with_yearly_time_sig:
            self.make_time_signatures()

        self.predictor_indices = self.xr_datasets_pred[0].sample_idx.data
        self.target_indices = self.xr_datasets_targ[0].sample_idx.data
        assert len(self.predictor_indices) == len(self.target_indices), "Predictor indices array should be of the same length as the target indices array"

        print("DATASET CONSTRUCTION DONE")
        
    ######################################### FUNCTIONS #########################################
    
    def __len__(self):
        return len(self.predictor_indices)

    def normalize_data(self, data):
        """
        Inputs: 
            - data = NUMPY ARRAY (i.e. feed in only .data) of subset of main HRRR or URMA dataarray (which itself has already been subselected down to the desired hours) containing all data for whatever hour is desired, e.g. data for hour 0 for all days in the dataarray
                - NOTE: can also be the entire dataset, if normalization_scheme = "all times"
                
        Outputs:
            - normed_data = NUMPY ARRAY of normalized data 
                - Normalization is done w.r.t ALL days included in the input, i.e. for training, done w.r.t 2021/22/23 data, all months and days
            - mean_at_hour = float of the mean of the data used for normalization. Should be appended to a list; if normalization_scheme == "per hour" then this list will be of length == len(self.hours), but if normalization_scheme == "all times" then the list will only have one entry.
            - stddev_at_hour = float of the stddev of the data used for normalization. Should be appended to a list; if normalization_scheme == "per hour" then this list will be of length == len(self.hours), but if normalization_scheme == "all times" then the list will only have one entry.
        """
        mean = np.mean(data)
        stddev = np.std(data)
        normed_data = (data - mean)/stddev

        return normed_data, mean, stddev

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

    def make_time_signatures(self):
        # These are 1D arrays; in __getitem__ they should be multiplied by the appropriate matrix to serve as an additional input channel for the predictor
        # For simplicity of selection in __getitem__, these are the same length as the datasets so they can share an idx
        if self.with_yearly_time_sig:
            self.date_linear_frac_list = []
            for date in self.xr_datasets_pred[0].valid_time.data:
                dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
                dt_delta = dt_current - dt.datetime(dt_current.year, 1, 1) 
                numdays_currentyear = (dt.datetime(dt_current.year+1, 1,1) - dt.datetime(dt_current.year, 1, 1)).days 
                # +1 to use 1-365 encoding instead of 0-364; probably doesn't matter but w/e
                self.date_linear_frac_list.append((dt_delta.days+1)/numdays_currentyear) 
            print("Yearly time signatures done")

        if self.with_hourly_time_sig:
            self.hour_linear_frac_list = []
            for date in self.xr_datasets_pred[0].valid_time.data:
                dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
                # No +1 here; want to retain 0-23 indexing
                self.hour_linear_frac_list.append((dt_current.hour)/23) #using 23 so the full 0-1 range is covered
            print("Hourly time signatures done")

        return

    #########################################

    def get_var_filepath(self, var_name, is_pred=True):
        """
        Inputs: 
            - var_name as string (e.g. "t2m", "d2m", etc). Can be "terrain" to load terrain data as well
            - is_pred = bool to control if predictor or target filepath is returned
        Output: relevant filepath(s) for that var
        """
        
        path_root = "/data1/projects/RTMA/alex.schein" #os.path.dirname(os.getcwd())
        if var_name == "terrain":
            terrain_path_hrrr = os.path.join(path_root, "Terrain_Maps", "terrain_subset_HRRR_2p5km.nc")
            terrain_path_urma = os.path.join(path_root, "Terrain_Maps", "terrain_subset_namsmarttopconus2p5.nc")
            return terrain_path_hrrr, terrain_path_urma
        else:
            # if self.is_train:
            #     if is_pred:
            #         data_save_path = os.path.join(path_root,"Regridded_HRRR_train_test", f"train_hrrr_alltimes_{var_name}_f{str(self.forecast_lead_time).zfill(2)}.nc")
            #     else:
            #         data_save_path = os.path.join(path_root,"URMA_train_test", f"train_urma_alltimes_{var_name}.nc")
            # else:
            #     if is_pred:
            #         data_save_path = os.path.join(path_root,"Regridded_HRRR_train_test", f"test_hrrr_alltimes_{var_name}_f{str(self.forecast_lead_time).zfill(2)}.nc")
            #     else:
            #         data_save_path = os.path.join(path_root,"URMA_train_test", f"test_urma_alltimes_{var_name}.nc")
            
            
            if self.is_train:
                if is_pred:
                    data_save_path = os.path.join(path_root,"Regridded_HRRR_train_test", "OLD_DATASETS","master_netcdfs_co_domain_radius_6371950",f"train_hrrr_alltimes_{var_name}_f{str(self.forecast_lead_time).zfill(2)}.nc")
                else:
                    data_save_path = os.path.join(path_root,"URMA_train_test", "OLD_DATASETS", "master_netcdfs_co_domain", f"train_urma_alltimes_{var_name}.nc")
            else:
                if is_pred:
                    data_save_path = os.path.join(path_root,"Regridded_HRRR_train_test","OLD_DATASETS","master_netcdfs_co_domain_radius_6371950", f"test_hrrr_alltimes_{var_name}_f{str(self.forecast_lead_time).zfill(2)}.nc")
                else:
                    data_save_path = os.path.join(path_root,"URMA_train_test", "OLD_DATASETS", "master_netcdfs_co_domain", f"test_urma_alltimes_{var_name}.nc")
            return data_save_path

    #########################################

    def process_variable(self, var_name, is_pred=True):
        """
        Processes relevant variable (could be predictor or target) to return the following:
            - xr_dataset: the raw dataset 
            - dataset_normed: dataset normed with respect to normalization_scheme
            - dataset_normed_means: list of mean(s) of the data
                - If self.normalization_scheme == "all times", will be a list of length 1
                - If self.normalization_scheme == "per hour", will be a list of length == len(self.hours) containing each hour's mean
            - dataset_normed_stddevs: list of stddev(s) of the data
                - If self.normalization_scheme == "all times", will be a list of length 1
                - If self.normalization_scheme == "per hour", will be a list of length == len(self.hours) containing each hour's mean

        Outputs should be appended to their relevant list in the main function
        """
        
        data_save_path = self.get_var_filepath(var_name, is_pred)
        ## Open xarray dataarrays
            # Note using dataarrays rather than datasets because then we don't have to worry about variable names when subselecting
            # However, this assumes we are dealing with datasets containing only one data variable!
            # !!!! DATASETS MUST ALSO CONTAIN A "sample_idx" DIMENSION, ONE INDEX PER TIME, MONOTONICALLY INCREASING, ALIGNED BETWEEN PRED AND TARG DATASETS !!!!
        xr_dataset = xr.open_dataarray(data_save_path, decode_timedelta=True)

        ## Subset dataarrays to only valid times
            # Returns an xarray dataset where the first entry is the first instance of data at self.hours[0], second entry is the first instance of data at self.hours[1], and so on; every len(self.hours) it wraps around to the next date's data
            # Builds a list of the sample idxs fulfilling the month and hour conditions, then subsets the dataarrays (again, currently not subsetting days within a month, i.e. if a month is included, all its days are too) 
            # !!!! SELECTS BASED ON valid_time I.E. THE ACTUAL TIME WE CARE ABOUT - so make sure any file with forecast lead time gets this correct, and the valid_time coordinate actually lines up with the right time! So the first valid_time in the training set should be 2021-01-01 00 UTC, etc
        does_date_fulfill_conditions = []
        for i, date in enumerate(xr_dataset.valid_time.data):
            date_as_dt = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
            if ((date_as_dt.month >= self.months[0])&(date_as_dt.month <= self.months[1]))&(date_as_dt.hour in self.hours):
                does_date_fulfill_conditions.append(i) #Can use i here as we are looping over all indices, so i == sample_idx, but this is NOT true when dealing with temporally restricted subsets! Be careful!
        valid_date_idxs = np.array(does_date_fulfill_conditions)

        ## Normalize datasets
            # Done here instead of in __getitem__ so it's only done once per Dataset call instead of once every item call
            # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs/xarray!!! So be careful in __getitem__
            # EXTREMELY IMPORTANT: this assumes xr_dataset_[pred/targ] have already been subselected down to whatever months and hours we desire, but they remain xarray datasets
        
        # Restrict the datasets, so idx works as a selector
        xr_dataset = xr_dataset[valid_date_idxs]
        dataset_normed = np.empty(np.shape(xr_dataset))
        dataset_normed_means = []
        dataset_normed_stddevs = []

        if self.is_train:
            year_str = "Years = 2021/22/23"
        else:
            year_str = "Year = 2024"

        if is_pred:
            pred_targ_str = "predictor"
        else:
            pred_targ_str = "target"

        print_hours = ("all" if self.hours==[i for i in range(24)] else self.hours)
        
        print(f"Loading {pred_targ_str} dataset for {var_name} ({year_str}, months = {self.months[0]} to {self.months[1]}, hours = {print_hours})")
        start = time.time()
        tmp_dataset_data = xr_dataset.data
        print(f"{pred_targ_str.capitalize()} dataset data loaded. Time taken = {(time.time()- start):.1f} sec")

        if self.normalization_scheme == "per hour":
            print(f"Normalizing per hour")
            for i, hr in enumerate(self.hours):
                start = time.time()
                #This indexing method is fast BUT relies on perfect data ordering in the non-normalized set! So input must conform to prior dataset description
                tmp_data, tmp_mean, tmp_stddev = self.normalize_data(tmp_dataset_data[i::len(self.hours)])
                dataset_normed[i::len(self.hours)] = tmp_data
                dataset_normed_means.append(tmp_mean)
                dataset_normed_stddevs.append(tmp_stddev) 
                print(f"[{i+1}/{len(self.hours)}] | Normalization for hour {str(hr).zfill(2)}'s data done. Time taken = {(time.time()- start):.1f} sec")
        else: #norm over the whole dataset
            print(f"Normalizing over all times")
            start = time.time()
            tmp_data, tmp_mean, tmp_stddev = self.normalize_data(tmp_dataset_data)
            dataset_normed = tmp_data
            dataset_normed_means.append(tmp_mean)
            dataset_normed_stddevs.append(tmp_stddev)
            print(f"Normalization done. Time taken = {(time.time()- start):.1f} sec")

        return xr_dataset, dataset_normed, dataset_normed_means, dataset_normed_stddevs

    #########################################

    def __getitem__(self, idx):
        ## Start with the first variable for each of predictor and target
        predictor = self.datasets_pred_normed[0][idx][np.newaxis,:,:]
        target = self.datasets_targ_normed[0][idx][np.newaxis,:,:]

        ## Add new channels for as many variables as we have
        # Do need to do a check for the case of variable list being 1 entry, as this might very well be the case, especially for target
        if len(self.predictor_vars) > 1:
            for i, var_name in enumerate(self.predictor_vars[1:]): #don't double up on index 0
                predictor = np.concatenate((predictor, self.datasets_pred_normed[i+1][idx][np.newaxis,:]), axis=0)
        if len(self.target_vars) > 1:
            for i, var_name in enumerate(self.target_vars[1:]): #don't double up on index 0
                target = np.concatenate((target, self.datasets_targ_normed[i+1][idx][np.newaxis,:]), axis=0)

        ## Add terrain layers as new channels
        if self.with_hrrr_terrain:
            predictor = np.concatenate((predictor, self.terrain_hrrr_normed[np.newaxis,:,:]), axis=0)
        if self.with_urma_terrain:
            predictor = np.concatenate((predictor, self.terrain_urma_normed[np.newaxis,:,:]), axis=0)
        if self.with_diff_terrain:
            predictor = np.concatenate((predictor, self.terrain_diff_normed[np.newaxis,:,:]), axis=0)

        ## Add time signature layers as new channels
        # (7/7) updated to linear signatures - note this changes the # of channels, so other code will have to be updated to match this change
        if self.with_yearly_time_sig:
            date_layer = (self.date_linear_frac_list[idx]*np.ones(np.shape(self.datasets_pred_normed[0][idx])))[np.newaxis,:,:]
            predictor = np.concatenate((predictor, date_layer), axis=0)
        if self.with_hourly_time_sig:
            hour_layer = (self.hour_linear_frac_list[idx]*np.ones(np.shape(self.datasets_pred_normed[0][idx])))[np.newaxis,:,:]
            predictor = np.concatenate((predictor, hour_layer), axis=0)

        
        return (predictor), (target)