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

class HRRR_URMA_Dataset(Dataset):
    def __init__(self, 
                 is_train=True,
                 months=[1,12],  
                 hours="all", 
                 forecast_lead_time = 0, 
                 normalization_scheme = "per hour",
                 with_terrains=["hrrr","urma","diff"], 
                 with_yearly_time_sig = True, 
                 with_hourly_time_sig = True,
                 time_sig_scheme = "frac"):
        
        """
        - is_train --> bool to load either training or testing datasets
        - months --> 2-tuple of ints 1-12; first entry = start month, second entry = end month. 
            - Note this means only a continuous range of months can be selected, and all days in those months will be selected
            - Note also this means only dates WITHIN a year can be selected - cross-year selection currently not allowed (causes issues with the fact we don't have 2020 data, so any cross-year times in the training set can only contain 2years - and we don't have any non-2024 data in the testing set...)
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
        - with_terrains --> list of terrains to include as separate, normalized channels (can be empty to not include terrain):
            - "hrrr" --> include NORMALIZED 2.5 km downscaled HRRR terrain field as a separate channel
            - "urma" --> include NORMALIZED 2.5 km nam_smarttopconus2p5 terrain field as a separate channel
            - "diff" --> include NORMALIZED map of the difference between HRRR and URMA terrain as a separate channel
            - !!!! NOTE: if both "hrrr" and "urma" are included, then HRRR terrain field will be normalized with respect to the mean/stddev of the URMA terrain!
        - with_yearly_time_sig --> bool to include yearly time signatures. Changed from sinusoidal to linear as of 7/7
        - with_hourly_time_sig --> bool to include time-of-day signatures. Changed from sinusoidal to linear as of 7/7
        - time_sig_scheme = str to control what scheme to use
            - "direct" --> use integer indices, e.g. 1,2,...,365 for day of year, and 0,1,...,23 for hour of day
            - "frac" --> use proportional indices, e.g. n/365 for nth day of year, m/23 for hour of day (note use of 23 instead of 24 to get 0 and 1 included)
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
    
        self.with_hrrr_terrain = False
        self.with_urma_terrain = False
        self.with_diff_terrain = False
        if "hrrr".casefold() in [x.casefold() for x in with_terrains]: #more complex check to allow for whatever casing, as all lowercase might not necessarily be the best
            self.with_hrrr_terrain = True
        if "urma".casefold() in [x.casefold() for x in with_terrains]:
            self.with_urma_terrain = True
        if "diff".casefold() in [x.casefold() for x in with_terrains]:
            self.with_diff_terrain = True
        
        self.with_yearly_time_sig = with_yearly_time_sig
        self.with_hourly_time_sig = with_hourly_time_sig
        self.time_sig_scheme = time_sig_scheme.lower() #will likely be calling this with capitalized "Direct" or "Frac" but don't want this in below code, so enforce lowercase
        
        #########################################
        ## Establish paths

        #(6/23) These need to be rewritten/expanded to include new vars
        RADIUS = 6371200 #Should be 6371950
        
        path_root = "/data1/projects/RTMA/alex.schein" #os.path.dirname(os.getcwd())
        terrain_path_hrrr = os.path.join(path_root, "Terrain_Maps", "terrain_subset_HRRR_2p5km.nc")
        terrain_path_urma = os.path.join(path_root, "Terrain_Maps", "terrain_subset_namsmarttopconus2p5.nc")
        if self.is_train:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", f"train_hrrr_alltimes_t2m_f{str(forecast_lead_time).zfill(2)}.nc")#_{RADIUS}.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", f"train_urma_alltimes_t2m.nc")
        else:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", f"test_hrrr_alltimes_t2m_f{str(forecast_lead_time).zfill(2)}.nc")#_{RADIUS}.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", f"test_urma_alltimes_t2m.nc")

        #########################################
        ## Open xarray dataarrays
            # Note using dataarrays rather than datasets because then we don't have to worry about variable names when subselecting
            # However, this assumes we are dealing with datasets containing only one data variable!
            # !!!! DATASETS MUST ALSO CONTAIN A "sample_idx" DIMENSION, ONE INDEX PER TIME, MONOTONICALLY INCREASING, ALIGNED BETWEEN PRED AND TARG DATASETS !!!!
            
        self.xr_dataset_pred = xr.open_dataarray(data_save_path_pred, decode_timedelta=True)
        self.xr_dataset_targ = xr.open_dataarray(data_save_path_targ, decode_timedelta=True)
        if self.with_hrrr_terrain:
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True)
        if self.with_urma_terrain:
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True)
        if self.with_diff_terrain: #needed in case one or both of HRRR or/and URMA terrain aren't included, but their difference is
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True)
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True)

        #########################################
        ## Subset dataarrays to only valid times

        # Returns an xarray dataset where the first entry is the first instance of data at self.hours[0], second entry is the first instance of data at self.hours[1], and so on; every len(self.hours) it wraps around to the next date's data
        # Builds a list of the sample idxs fulfilling the month and hour conditions, then subsets the dataarrays (again, currently not subsetting days within a month, i.e. if a month is included, all its days are too) 
        # List is only made over predictor times/indices, but these must necessarily match with the target's, so it shouldn't be an issue
        # !!!! SELECTS BASED ON valid_time I.E. THE ACTUAL TIME WE CARE ABOUT - so make sure any file with forecast lead time gets this correct, and the valid_time coordinate actually lines up with the right time! So the first valid_time in the training set should be 2021-01-01 00 UTC, etc
        
        does_date_fulfill_conditions = []
        for i, date in enumerate(self.xr_dataset_pred.valid_time.data):
            date_as_dt = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
            if ((date_as_dt.month >= self.months[0])&(date_as_dt.month <= self.months[1]))&(date_as_dt.hour in self.hours):
                does_date_fulfill_conditions.append(i) #Can use i here as we are looping over all indices, so i == sample_idx, but this is NOT true when dealing with temporally restricted subsets! Be careful!
        valid_date_idxs = np.array(does_date_fulfill_conditions)

        #########################################
        ## Normalize datasets
            # Done here instead of in __getitem__ so it's only done once per Dataset call instead of once every item call
            # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs/xarray!!! So be careful in __getitem__
            # EXTREMELY IMPORTANT: this assumes xr_dataset_[pred/targ] have already been subselected down to whatever months and hours we desire, but they remain xarray datasets


        # (6/24) Should be rewritten to be functionalized so looping over all vars works easily

        
        # Restrict the datasets, so idx works as a selector
        self.xr_dataset_pred = self.xr_dataset_pred[valid_date_idxs]
        self.xr_dataset_targ = self.xr_dataset_targ[valid_date_idxs]
        
        self.dataset_pred_normed = np.empty(np.shape(self.xr_dataset_pred)) 
        self.dataset_targ_normed = np.empty(np.shape(self.xr_dataset_targ))
        self.dataset_pred_normed_means = []
        self.dataset_targ_normed_means = []
        self.dataset_pred_normed_stddevs = []
        self.dataset_targ_normed_stddevs = []

        if self.is_train:
            year_str = "Years = 2021/22/23"
        else:
            year_str = "Year = 2024"
        
        # For speed purposes, worth preloading all data as it's considerably faster than loading each hour in the below loop
        print(f"Loading predictor dataset ({year_str}, months = {self.months[0]} to {self.months[1]}, hours = {self.hours})")
        start = time.time()
        tmp_dataset_pred_data = self.xr_dataset_pred.data
        print(f"Predictor dataset data loaded. Time taken = {(time.time()- start):.1f} sec")

        print(f"Loading target dataset (Same time span)") #({year_str}, months = {self.months[0]} to {self.months[1]}, hours = {self.hours})")
        start = time.time()
        tmp_dataset_targ_data = self.xr_dataset_targ.data
        print(f"Target dataset data loaded. Time taken = {(time.time()- start):.1f} sec")
        
        # Once the data is in memory, calculation is fast
        if self.normalization_scheme == "per hour":
            for i, hr in enumerate(self.hours):
                #This indexing method is fast BUT relies on perfect data ordering in the non-normalized set! So input must conform to prior dataset description
                tmp_data, tmp_mean, tmp_stddev = self.normalize_one_hour(tmp_dataset_pred_data[i::len(self.hours)]) 
                self.dataset_pred_normed[i::len(self.hours)] = tmp_data
                self.dataset_pred_normed_means.append(tmp_mean)
                self.dataset_pred_normed_stddevs.append(tmp_stddev)
    
                tmp_data, tmp_mean, tmp_stddev = self.normalize_one_hour(tmp_dataset_targ_data[i::len(self.hours)])
                self.dataset_targ_normed[i::len(self.hours)] = tmp_data
                self.dataset_targ_normed_means.append(tmp_mean)
                self.dataset_targ_normed_stddevs.append(tmp_stddev)
    
                print_flag = True
                if print_flag: 
                    print(f"[{i+1}/{len(self.hours)}] | Normalization for hour {str(hr).zfill(2)}'s data done")
        else: #norm over the whole dataset
            print_flag = True
            if print_flag:
                print(f"Normalizing over all times")
                start = time.time()
            
            tmp_data, tmp_mean, tmp_stddev = self.normalize_one_hour(tmp_dataset_pred_data) #can reuse the same function, though it's not well named now... 
            self.dataset_pred_normed = tmp_data
            self.dataset_pred_normed_means.append(tmp_mean) 
            self.dataset_pred_normed_stddevs.append(tmp_stddev)

            tmp_data, tmp_mean, tmp_stddev = self.normalize_one_hour(tmp_dataset_targ_data)
            self.dataset_targ_normed = tmp_data
            self.dataset_targ_normed_means.append(tmp_mean)
            self.dataset_targ_normed_stddevs.append(tmp_stddev)

            if print_flag:
                print(f"Normalization done. Time taken = {(time.time()- start):.1f} sec")

        # These are memory hogs worth deleting manually in case garbage handling doesn't deal with them
        del tmp_dataset_pred_data, tmp_dataset_targ_data, tmp_data
        
        # Set terrain normalizations
        self.normalize_terrain()

        #########################################
        ## Calculate time index arrays
            # These are 1D arrays; in __getitem__ they should be multiplied by the appropriate matrix to serve as an additional input channel for the predictor
            # For simplicity of selection in __getitem__, these are the same length as the datasets so they can share an idx

        # Make sin/cos waves for date encoding
        # Takes number of days since start of current year, casts as a fraction of the current year, takes sin/cos
        # Only done for pred; targ necessarily must have the same times but this is not intended for use as a target function, only additional info in the prediction
        # (7/7) Updated to linear signature instead
        if self.with_yearly_time_sig:
            # self.date_sin_list = []
            # self.date_cos_list = []
            # for date in self.xr_dataset_pred.valid_time.data:
            #     dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
            #     dt_delta = dt_current - dt.datetime(dt_current.year, 1, 1) #no need for +1 here as it should start at 0
            #     numdays_currentyear = (dt.datetime(dt_current.year+1, 1,1) - dt.datetime(dt_current.year, 1, 1)).days #the difference between jan 1 and dec 31 of the same year is reported as 364 (non-leap year) while year-to-year jan 1 is actually correct. This only exists to handle leap years (of which 2024 is one!)
            #     self.date_sin_list.append(np.sin(2*np.pi*(dt_delta.days / numdays_currentyear)))
            #     self.date_cos_list.append(np.cos(2*np.pi*(dt_delta.days / numdays_currentyear)))
            self.date_linear_frac_list = []
            self.date_linear_direct_list = []
            for date in self.xr_dataset_pred.valid_time.data:
                dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
                dt_delta = dt_current - dt.datetime(dt_current.year, 1, 1) 
                numdays_currentyear = (dt.datetime(dt_current.year+1, 1,1) - dt.datetime(dt_current.year, 1, 1)).days 
                
                # +1 to use 1-365 encoding instead of 0-364; probably doesn't matter but w/e
                self.date_linear_frac_list.append((dt_delta.days+1)/numdays_currentyear) 
                self.date_linear_direct_list.append(dt_delta.days+1)
            
            print("Yearly time signatures done")


        # Make sin/cos waves for hour encoding
        # Just takes current hour, casts as a fraction of 24h day, takes sin/cos
        # (7/7) Updated to linear signature instead
        if self.with_hourly_time_sig:
            # self.hour_sin_list = []
            # self.hour_cos_list = []
            # for date in self.xr_dataset_pred.valid_time.data:
            #     dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
            #     self.hour_sin_list.append(np.sin(2*np.pi*(dt_current.hour / 24)))
            #     self.hour_cos_list.append(np.cos(2*np.pi*(dt_current.hour / 24)))
            self.hour_linear_frac_list = []
            self.hour_linear_direct_list = []
            for date in self.xr_dataset_pred.valid_time.data:
                dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
                # No +1 here; want to retain 0-23 indexing
                self.hour_linear_frac_list.append((dt_current.hour)/23) #using 23 so the full 0-1 range is covered
                self.hour_linear_direct_list.append(dt_current.hour)
            
            print("Hourly time signatures done")
        
        self.predictor_indices = self.xr_dataset_pred.sample_idx.data
        self.target_indices = self.xr_dataset_targ.sample_idx.data
        assert len(self.predictor_indices) == len(self.target_indices), "Predictor indices array should be of the same length as the target indices array"
        
    ######################################### FUNCTIONS #########################################
    
    def __len__(self):
        return len(self.predictor_indices)

    def normalize_one_hour(self, data_at_hour):
        #inputs: 
            #data_at_hour = NUMPY ARRAY (i.e. feed in only .data) of subset of main HRRR or URMA dataarray (which itself has already been subselected down to the desired hours) containing all data for whatever hour is desired, e.g. data for hour 0 for all days in the dataarray
        # outputs:
            # normed_data_at_hour = NUMPY ARRAY of normalized data for that hour
                # Normalization is done w.r.t ALL days included in the input, i.e. for training, done w.r.t 2021/22/23 data, all months and days
            # mean_at_hour = float of the mean of the data used for normalization. Should be appended to a list of the same length as self.hours
            # stddev_at_hour = float of the stddev of the data used for normalization. Should be appended to a list of the same length as self.hours

        mean = np.mean(data_at_hour)
        stddev = np.std(data_at_hour)
        normed_data_at_hour = (data_at_hour - mean)/stddev

        return normed_data_at_hour, mean, stddev

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
        print("Terrain normalization done") 
        return

    #########################################

    def __getitem__(self, idx):

        # get sample index for predictor and target
        p_idx = self.predictor_indices[idx]
        t_idx = self.target_indices[idx]
    
        ## Normed
        predictor = self.dataset_pred_normed[idx][np.newaxis,:,:]
        target = self.dataset_targ_normed[idx][np.newaxis,:,:]

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
            # date_sin_layer = (self.date_sin_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            # date_cos_layer = (self.date_cos_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            # predictor = np.concatenate((predictor, date_sin_layer, date_cos_layer), axis=0)
            if self.time_sig_scheme == "direct":
                date_layer = (self.date_linear_direct_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            elif self.time_sig_scheme == "frac":
                date_layer = (self.date_linear_frac_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            predictor = np.concatenate((predictor, date_layer), axis=0)
        if self.with_hourly_time_sig:
            # hour_sin_layer = (self.hour_sin_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            # hour_cos_layer = (self.hour_cos_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            # predictor = np.concatenate((predictor, hour_sin_layer, hour_cos_layer), axis=0)
            if self.time_sig_scheme == "direct":
                hour_layer = (self.hour_linear_direct_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            elif self.time_sig_scheme == "frac":
                hour_layer = (self.hour_linear_frac_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            predictor = np.concatenate((predictor, hour_layer), axis=0)
     
        return (predictor), (target)