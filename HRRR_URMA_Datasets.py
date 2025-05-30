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

class HRRR_URMA_Dataset_Anytime_Anydate_Anyterrain(Dataset):
    def __init__(self, 
                 is_train=True, 
                 with_hrrr_terrain=False,
                 with_urma_terrain=False,
                 with_terrain_difference=False,
                 months=[1,3], 
                 days=[1,31], 
                 hour=0):
        # is_train --> load either training or testing datasets
        # with_hrrr_terrain --> include NORMALIZED 2.5 km downscaled HRRR terrain field as a separate channel 
        # with_urma_terrain --> include NORMALIZED 2.5 km nam_smarttopconus2p5 terrain field as a separate channel 
        # with_terrain_difference --> include NORMALIZED map of the difference between HRRR and URMA terrain as a separate channel
        # months = 2-tuple of ints 1-12; first entry = start month, second entry = end month. All years 2021/22/23 will have data selected from this range
        # days = 2-tuple of ints 1-31; first entry = start day, second entry = end day
        # hour = int, 0 or 12 (currently). Determines if this uses 00z or 12z data
        self.with_hrrr_terrain = with_hrrr_terrain
        self.with_urma_terrain = with_urma_terrain
        self.with_terrain_difference = with_terrain_difference
        self.months = months
        self.days = days #note user is resposible for making sure days[1] matches actual number of days in months[1]...
        self.hour = hour
        
        # Establish paths
        # UNLIKE Marshall's code, the training and predictor indices align exactly and are contained in the sample_idx coordinate, so no need for separate path
        # Also, now we have separate .nc files for training and testing, so need multiple paths there
        path_root = os.path.dirname(os.getcwd())
        terrain_path_hrrr = os.path.join(path_root, "Terrain_Maps", "terrain_subset_HRRR_2p5km.nc")
        terrain_path_urma = os.path.join(path_root, "Terrain_Maps", "terrain_subset_namsmarttopconus2p5.nc")
        if is_train:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "train_hrrr.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "train_urma.nc")
        else:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "test_hrrr.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "test_urma.nc")

        # Open xarray dataarrays
        # Note using dataarrays rather than datasets because then we don't have to worry about variable names when subselecting
        self.xr_dataset_pred = xr.open_dataarray(data_save_path_pred, decode_timedelta=True)
        self.xr_dataset_targ = xr.open_dataarray(data_save_path_targ, decode_timedelta=True)
        if self.with_hrrr_terrain:
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True)
        if self.with_urma_terrain:
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True)
        if self.with_terrain_difference: #needed in case one or both of HRRR or/and URMA terrain aren't included, but their difference is
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True)
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True)
       
        # Restrict datasets to only 00z or 12z 
        if hour==0:
            self.xr_dataset_pred = self.xr_dataset_pred[0::2]
            self.xr_dataset_targ = self.xr_dataset_targ[0::2]
            print('00z data selected')
        elif hour==12:
            self.xr_dataset_pred = self.xr_dataset_pred[1::2]
            self.xr_dataset_targ = self.xr_dataset_targ[1::2]
            print('12z data selected')
        else:
            print('ERROR: hour should be 0 or 12 only')
        
        # Make date ranges. Should be ok to keep these as local var
        date_idxs, date_idxs0, date_idxs1, date_idxs2 = self.make_date_idxs()
        print(f'Done making date index list. Months = {self.months[0]} to {self.months[1]}, days = {self.days[0]} to {self.days[1]}')
        
        # Normalize datasets
        # Done here instead of in __getitem__ so it's only done once per Dataset call instead of once every item call
        # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs/xarray!!! So be careful in __getitem__
        
        #terribleness needed due to extreme slowness of selecting URMA data with master date index list that contains discontinuous entries
        data_hrrr = np.concatenate((self.xr_dataset_pred[date_idxs0,:].data, self.xr_dataset_pred[date_idxs1,:].data, self.xr_dataset_pred[date_idxs2,:].data))
        data_urma = np.concatenate((self.xr_dataset_targ[date_idxs0,:].data, self.xr_dataset_targ[date_idxs1,:].data, self.xr_dataset_targ[date_idxs2,:].data))
        
        self.hrrr_mean = np.mean(data_hrrr)
        self.hrrr_std = np.std(data_hrrr)
        self.urma_mean = np.mean(data_urma)
        self.urma_std = np.std(data_urma)
        
        self.dataset_pred_normed = (data_hrrr - self.hrrr_mean)/self.hrrr_std
        self.dataset_targ_normed = (data_urma - self.urma_mean)/self.urma_std

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
        if self.with_terrain_difference:
            terrain_hrrr = self.xr_terrain_hrrr.data
            terrain_urma = self.xr_terrain_urma.data
            terrain_diff = terrain_hrrr-terrain_urma
            self.terrain_diff_mean = np.mean(terrain_diff)
            self.terrain_diff_std = np.std(terrain_diff)
            self.terrain_diff_normed = (terrain_diff - self.terrain_diff_mean)/self.terrain_diff_std
        if self.with_hrrr_terrain and self.with_urma_terrain: #use the same mean/std to norm both. Using URMA at the moment
            # Note the URMA terrain stuff has already been done in this case
            terrain_hrrr = self.xr_terrain_hrrr.data
            self.terrain_hrrr_mean = np.mean(terrain_hrrr) #for diagnostics only
            self.terrain_hrrr_std = np.std(terrain_hrrr) #for diagnostics only
            self.terrain_hrrr_normed = (terrain_hrrr - self.terrain_urma_mean)/self.terrain_urma_std
        print("Done with normalization")

        # Restrict datasets (already only 00z or 12z) to only date indices
        # Needs to be done post-normalization as normalization needs to select from full (albiet 00z or 12z restricted) datasets
        # Normalization code only looks at the data, i.e. it ignores selector indices, hence we refer back to the original arrays with their metadata
        self.xr_dataset_pred = self.xr_dataset_pred[date_idxs,:]
        self.xr_dataset_targ = self.xr_dataset_targ[date_idxs,:]
        print('Done restricting xarrays to date indices')
        
        # Indices only over 00z times AND only date indices
        # Should be restricted already by prior operations
        self.predictor_indices = self.xr_dataset_pred.sample_idx.data
        self.target_indices = self.xr_dataset_targ.sample_idx.data
        assert len(self.predictor_indices) == len(self.target_indices), "Predictor indices array should be of the same length as the target indices array"

    def __len__(self):
        return len(self.predictor_indices)
        
    def make_date_idxs(self):
        start_date0 = dt.datetime(2021,self.months[0],self.days[0],self.hour)
        end_date0 = dt.datetime(2021,self.months[1],self.days[1],self.hour)
        start_date1 = dt.datetime(2022,self.months[0],self.days[0],self.hour)
        end_date1 = dt.datetime(2022,self.months[1],self.days[1],self.hour)
        start_date2 = dt.datetime(2023,self.months[0],self.days[0],self.hour)
        end_date2 = dt.datetime(2023,self.months[1],self.days[1],self.hour)
        
        idxs0 = np.where((np.datetime64(start_date0) <= self.xr_dataset_pred.valid_time.data)
        				 & (self.xr_dataset_pred.valid_time.data <= np.datetime64(end_date0)))
        idxs1 = np.where((np.datetime64(start_date1) <= self.xr_dataset_pred.valid_time.data) 
        				 & (self.xr_dataset_pred.valid_time.data <= np.datetime64(end_date1)))
        idxs2 = np.where((np.datetime64(start_date2) <= self.xr_dataset_pred.valid_time.data) 
        				 & (self.xr_dataset_pred.valid_time.data <= np.datetime64(end_date2)))
        
        # Need to return concated list as well as each individual one due to weirdness with URMA...   
        return np.concatenate((idxs0[0], idxs1[0], idxs2[0])), idxs0[0], idxs1[0], idxs2[0]

    def __getitem__(self, idx):

        # get sample index for predictor and target
        p_idx = self.predictor_indices[idx]
        t_idx = self.target_indices[idx]

        ## Normed
        predictor = self.dataset_pred_normed[idx][np.newaxis,:,:]
        target = self.dataset_targ_normed[idx][np.newaxis,:,:]

        if self.with_hrrr_terrain:
            predictor = np.concatenate((predictor, self.terrain_hrrr_normed[np.newaxis,:,:]), axis=0)
        if self.with_urma_terrain:
            predictor = np.concatenate((predictor, self.terrain_urma_normed[np.newaxis,:,:]), axis=0)
        if self.with_terrain_difference:
            predictor = np.concatenate((predictor, self.terrain_diff_normed[np.newaxis,:,:]), axis=0)
            
        return (predictor), (target) 


######################################################################################################################################################

class HRRR_URMA_Dataset_AllTimes_AnyDates_AnyTerrains(Dataset):
    def __init__(self, 
                 is_train=True,
                 months=[1,12],  
                 hours="all", 
                 with_terrains=["hrrr","urma","diff"], 
                 with_yearly_time_sig = True, 
                 with_hourly_time_sig = True):
        # is_train --> bool to load either training or testing datasets
        # months --> 2-tuple of ints 1-12; first entry = start month, second entry = end month. 
            # Note this means only a continuous range of months can be selected, and all days in those months will be selected
            # if is_train=True, all years 2021/22/23 will have data selected from this range
            # if is_train=False, only 2024 data will be selected
        # hours --> either str of "all" to include all times, or a subset of list of ints [0,1,..23]. Selects valid hours to include.
            # Can be discontinuous, but almost certainly won't be (why bother?)
        # with_terrains --> list of terrains to include as separate, normalized channels (can be empty to not include terrain):
            # "hrrr" --> include NORMALIZED 2.5 km downscaled HRRR terrain field as a separate channel
            # "urma" --> include NORMALIZED 2.5 km nam_smarttopconus2p5 terrain field as a separate channel
            # "diff" --> include NORMALIZED map of the difference between HRRR and URMA terrain as a separate channel
            # !!!! NOTE: if both "hrrr" and "urma" are included, then HRRR terrain field will be normalized with respect to the mean/stddev of the URMA terrain!
        # with_yearly_time_sig --> bool to include yearly time signatures. Only included as an option for testing; should train a model with/without time signature options
        # with_hourly_time_sig --> bool to include time-of-day signatures. Only included as an option for testing; should train a model with/without time signature options

        #########################################
        ## Initialize vars

        self.is_train = is_train
        self.months = months
        self.hours = hours
        if self.hours == "all": #we always need a list of ints 
            self.hours = [i for i in range(24)]
        
        self.with_hrrr_terrain = False
        self.with_urma_terrain = False
        self.with_diff_terrain = False
        if "hrrr".casefold() in [x.casefold() for x in with_terrains]: #more complex check to allow for whatever casing, as all lowercase might not necessarily be the best
            self.with_hrrr_terrain = True
        if "urma".casefold() in [x.casefold() for x in with_terrains]:
            self.with_urma_terrain = True
        if "diff".casefold() in [x.casefold() for x in with_terrains]:
            self.with_diff_terrain = True
            
        #########################################
        ## Establish paths
        
        path_root = os.path.dirname(os.getcwd())
        terrain_path_hrrr = os.path.join(path_root, "Terrain_Maps", "terrain_subset_HRRR_2p5km.nc")
        terrain_path_urma = os.path.join(path_root, "Terrain_Maps", "terrain_subset_namsmarttopconus2p5.nc")
        if is_train:
            #data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "train_hrrr_alltimes.nc")
            
            # !!!!!!!!!!!!!! FOR INITIAL TESTING ONLY WHILE HRRR ALLTIMES ISN'T READY
            data_save_path_pred = os.path.join(path_root,"URMA_train_test", "train_urma_alltimes.nc")
            
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "train_urma_alltimes.nc")
        else:
            #data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "test_hrrr_alltimes.nc")
            
            # !!!!!!!!!!!!!! FOR INITIAL TESTING ONLY WHILE HRRR ALLTIMES ISN'T READY
            data_save_path_pred = os.path.join(path_root,"URMA_train_test", "test_urma_alltimes.nc")
            
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "test_urma_alltimes.nc")

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
            # !!! This is only done if months != [1,12] and/or hours != [0,1,2,...23]
            
        # IMPLEMENT IF NEEDED - for current work (as of 2025/05/28) this is not needed, as all months and all times of day will be used
        # Implementation should probably use the time signatures from the valid_time coordinate rather than index-based methods, though index-based might be easier as it aligns with the list of ints comprising self.hours
        # Whatever the implementation, needs to return an xarray dataset where the first entry is the first instance of data at self.hours[0], second entry is the first instance of data at self.hours[1], and so on; every len(self.hours) it wraps around to the next day's data
        # This is also where splitting on is_train will likely be necessary

        #########################################
        ## Normalize datasets
            # Done here instead of in __getitem__ so it's only done once per Dataset call instead of once every item call
            # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs/xarray!!! So be careful in __getitem__
            # EXTREMELY IMPORTANT: this assumes xr_dataset_[pred/targ] have already been subselected down to whatever months and hours we desire, but they remain xarray datasets

        self.dataset_pred_normed = np.empty(np.shape(self.xr_dataset_pred))
        self.dataset_targ_normed = np.empty(np.shape(self.xr_dataset_targ))
        self.dataset_pred_normed_means = []
        self.dataset_targ_normed_means = []
        self.dataset_pred_normed_stddevs = []
        self.dataset_targ_normed_stddevs = []
        
        # For speed purposes, worth preloading all data as it's considerably faster than loading each hour in the below loop
        start = time.time()
        tmp_dataset_pred_data = self.xr_dataset_pred.data
        print(f"Predictor dataset data loaded. Time taken = {(time.time()- start):.1f} sec")
        
        start = time.time()
        tmp_dataset_targ_data = self.xr_dataset_targ.data
        print(f"Target dataset data loaded. Time taken = {(time.time()- start):.1f} sec")

        # Once the data is in memory, calculation is fast
        for i, hr in enumerate(self.hours):
            tmp_data, tmp_mean, tmp_stddev = self.normalize_one_hour(tmp_dataset_pred_data[i::len(self.hours)])
            self.dataset_pred_normed[i::len(self.hours)] = tmp_data
            self.dataset_pred_normed_means.append(tmp_mean)
            self.dataset_pred_normed_stddevs.append(tmp_stddev)

            tmp_data, tmp_mean, tmp_stddev = self.normalize_one_hour(tmp_dataset_targ_data[i::len(self.hours)])
            self.dataset_targ_normed[i::len(self.hours)] = tmp_data
            self.dataset_targ_normed_means.append(tmp_mean)
            self.dataset_targ_normed_stddevs.append(tmp_stddev)

            print(f"[{i}/{len(self.hours)}] | Normalization for hour {hr}'s data done")

        # These are memory hogs worth deleting manually in case garbage handling doesn't deal with them
        del tmp_dataset_pred_data, tmp_dataset_targ_data

        # Set terrain normalizations
        self.normalize_terrain()

        #########################################
        ## Calculate time index arrays
            # These are 1D arrays; in __getitem__ they should be multiplied by the appropriate matrix to serve as an additional input channel for the predictor
            # For simplicity of selection in __getitem__, these are the same length as the datasets so they can share an idx

        # Make sin/cos waves for date encoding
        # Takes number of days since start of current year, casts as a fraction of the current year, takes sin/cos
        # Only done for pred; targ necessarily must have the same times but this is not intended for use as a target player, only additional info in the prediction
        if with_yearly_time_sig:
            self.date_sin_list = []
            self.date_cos_list = []
            for date in self.xr_dataset_pred.valid_time.data:
                dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
                dt_delta = dt_current - dt.datetime(dt_current.year, 1, 1) #no need for +1 here as it should start at 0
                numdays_currentyear = (dt.datetime(dt_current.year+1, 1,1) - dt.datetime(dt_current.year, 1, 1)).days #the difference between jan 1 and dec 31 of the same year is reported as 364 (non-leap year) while year-to-year jan 1 is actually correct. This only exists to handle leap years (of which 2024 is one!)
                self.date_sin_list.append(np.sin(2*np.pi*(dt_delta.days / numdays_currentyear)))
                self.date_cos_list.append(np.cos(2*np.pi*(dt_delta.days / numdays_currentyear)))
            print("Yearly time signatures done")


        # Make sin/cos waves for hour encoding
        # Just takes current hour, casts as a fraction of 24h day, takes sin/cos
        if with_hourly_time_sig:
            self.hour_sin_list = []
            self.hour_cos_list = []
            for date in self.xr_dataset_pred.valid_time.data:
                dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
                self.hour_sin_list.append(np.sin(2*np.pi*(dt_current.hour / 24)))
                self.hour_cos_list.append(np.cos(2*np.pi*(dt_current.hour / 24)))
            print("Hourly time signatures done")
        
    ######################################### FUNCTIONS #########################################

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
        print("Terrain normalization done") #Note print is here rather than main, so if no terrain is called, this won't print
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
        if with_yearly_time_sig:
            date_sin_layer = (self.date_sin_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            date_cos_layer = (self.date_cos_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            predictor = np.concatenate((predictor, date_sin_layer, date_cos_layer), axis=0)
        if with_hourly_time_sig:
            hour_sin_layer = (self.hour_sin_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            hour_cos_layer = (self.hour_cos_list[idx]*np.ones(np.shape(self.dataset_pred_normed[idx])))[np.newaxis,:,:]
            predictor = np.concatenate((predictor, hour_sin_layer, hour_cos_layer), axis=0)
     
        return (predictor), (target)