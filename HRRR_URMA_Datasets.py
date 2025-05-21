import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import datetime as dt
from netCDF4 import Dataset as nc_Dataset
import pandas as pd
import numpy as np
import xarray as xr

class HRRR_URMA_Dataset(Dataset):
    
    def __init__(self, is_train=False):
        # is_train --> load either training or testing datasets
        
        # Establish paths
        # UNLIKE Marshall's code, the training and predictor indices align exactly and are contained in the sample_idx coordinate, so no need for separate path
        # Also, now we have separate .nc files for training and testing, so need multiple paths there
        path_root = os.path.dirname(os.getcwd())
        terrain_path = os.path.join(path_root, "terrain_subset.nc")
        if is_train:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "train_hrrr.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "train_urma.nc")
        else:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "test_hrrr.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "test_urma.nc")

        # open netCDF4 datasets
        self.nc_dataset_pred = nc_Dataset(data_save_path_pred)
        self.nc_dataset_targ = nc_Dataset(data_save_path_targ)
        self.nc_dataset_terrain = nc_Dataset(terrain_path)

        # Normalize datasets
        # Done here instead of in __getitem__ so it's only done once per Dataset call instead of once every item call
        # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs!!! So be careful in __getitem__
        self.dataset_pred_normed = self.normalize(dataset_type="hrrr")
        self.dataset_targ_normed = self.normalize(dataset_type="urma")
        self.dataset_terrain_normed = self.normalize(dataset_type="terrain")

        # open xarray datasets
        # necessary because coordinate indices store the mappings, and it's much easier to read those with xarray than netcdf4
        self.xr_dataset_pred = xr.open_dataset(data_save_path_pred)
        self.xr_dataset_targ = xr.open_dataset(data_save_path_targ)

        self.predictor_indices = self.xr_dataset_pred.sample_idx.data
        self.target_indices = self.xr_dataset_targ.sample_idx.data
        assert len(self.predictor_indices) == len(self.target_indices), "Predictor indices array should be of the same length as the target indices array"

    def __len__(self):
        return len(self.predictor_indices)
        
    def normalize(self, dataset_type="hrrr"):
        # normalize datasets (note normalization here is (data - mean)/stddev)
        # "dataset_type" --> one of "hrrr", "urma", or "terrain". Three cases are needed since HRRR and URMA are time-dependent but terrain is not, AND the former two have differently named temperature variables (whoopsie)
        # Note index 0 for both HRRR and URMA train/test datasets is 00z, and 00/12z cycle around, so can just do odd/even selection, then normalize appropriately, then return interlaced array
        # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs!!! So be careful! (Can't do in-place data replacement in netCDF unfortunately)
        if dataset_type=="terrain":
            data = self.nc_dataset_terrain['h'][:].data
            normed_data = (data - np.mean(data))/np.std(data)
        elif dataset_type=="hrrr":
            data_00z = self.nc_dataset_pred['t'][0::2].data
            data_12z = self.nc_dataset_pred['t'][1::2].data
            tmp_00z = (data_00z - np.mean(data_00z))/np.std(data_00z)
            tmp_12z = (data_12z - np.mean(data_12z))/np.std(data_12z)
            normed_data = np.empty(self.nc_dataset_pred['t'][:].data.shape, dtype=tmp_00z.dtype)
            normed_data[0::2] = tmp_00z
            normed_data[1::2] = tmp_12z
        elif dataset_type=="urma":
            data_00z = self.nc_dataset_targ['t2m'][0::2].data
            data_12z = self.nc_dataset_targ['t2m'][1::2].data
            tmp_00z = (data_00z - np.mean(data_00z))/np.std(data_00z)
            tmp_12z = (data_12z - np.mean(data_12z))/np.std(data_12z)
            normed_data = np.empty(self.nc_dataset_targ['t2m'][:].data.shape, dtype=tmp_00z.dtype)
            normed_data[0::2] = tmp_00z
            normed_data[1::2] = tmp_12z
        else:
            print("ERROR: dataset_type must be one of 'terrain', 'hrrr', or 'urma' ")

        return normed_data
    
    
    def __getitem__(self, idx):

        # Determine if terrain map is included as separate channel
        with_terrain = True
        
        # get sample index for predictor and target
        p_idx = self.predictor_indices[idx]
        t_idx = self.target_indices[idx]

        # extract 2m temp image 
        # awkward: forgot to rename HRRR temp from "t" to "t2m" as in URMA...

        ## Unnormed 
        #predictor = self.nc_dataset_pred["t"][p_idx,:,:].data[np.newaxis,:,:]
        #target = self.nc_dataset_targ["t2m"][t_idx,:,:].data[np.newaxis,:,:]

        ## Normed
        predictor = self.dataset_pred_normed[idx][np.newaxis,:,:]
        target = self.dataset_targ_normed[idx][np.newaxis,:,:]
        if with_terrain:
            #predictor = np.concatenate((predictor, self.dataset_terrain_normed[np.newaxis,:,:]), axis=0)
            ##### Try appending the negative version of terrain to introduce a direct correlation with height and temperature
            predictor = np.concatenate((predictor, -1*self.dataset_terrain_normed[np.newaxis,:,:]), axis=0)
            

        return (predictor), (target) #copying Marshall's syntax, see if it works

######################################################################

class HRRR_URMA_Dataset_00z(Dataset):
    
    def __init__(self, is_train=False):
        # is_train --> load either training or testing datasets
        
        # Establish paths
        # UNLIKE Marshall's code, the training and predictor indices align exactly and are contained in the sample_idx coordinate, so no need for separate path
        # Also, now we have separate .nc files for training and testing, so need multiple paths there
        path_root = os.path.dirname(os.getcwd())
        terrain_path = os.path.join(path_root, "terrain_subset.nc")
        if is_train:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "train_hrrr.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "train_urma.nc")
        else:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "test_hrrr.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "test_urma.nc")

        # open netCDF4 datasets
        self.nc_dataset_pred = nc_Dataset(data_save_path_pred)
        self.nc_dataset_targ = nc_Dataset(data_save_path_targ)
        self.nc_dataset_terrain = nc_Dataset(terrain_path)

        # Normalize datasets
        # Done here instead of in __getitem__ so it's only done once per Dataset call instead of once every item call
        # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs!!! So be careful in __getitem__
        self.dataset_pred_normed = self.normalize(dataset_type="hrrr")
        self.dataset_targ_normed = self.normalize(dataset_type="urma")
        self.dataset_terrain_normed = self.normalize(dataset_type="terrain")

        #Restrict datasets to only 00z (even indices)
        self.dataset_pred_normed = self.dataset_pred_normed[0::2]
        self.dataset_pred_normed = self.dataset_targ_normed[0::2]

        # open xarray datasets
        # necessary because coordinate indices store the mappings, and it's much easier to read those with xarray than netcdf4
        self.xr_dataset_pred = xr.open_dataset(data_save_path_pred)
        self.xr_dataset_targ = xr.open_dataset(data_save_path_targ)

        #Indices only over 00z times
        self.predictor_indices = self.xr_dataset_pred.sample_idx.data[0::2]
        self.target_indices = self.xr_dataset_targ.sample_idx.data[0::2]
        assert len(self.predictor_indices) == len(self.target_indices), "Predictor indices array should be of the same length as the target indices array"

    def __len__(self):
        return len(self.predictor_indices)
        
    def normalize(self, dataset_type="hrrr"):
        # normalize datasets (note normalization here is (data - mean)/stddev)
        # "dataset_type" --> one of "hrrr", "urma", or "terrain". Three cases are needed since HRRR and URMA are time-dependent but terrain is not, AND the former two have differently named temperature variables (whoopsie)
        # Note index 0 for both HRRR and URMA train/test datasets is 00z, and 00/12z cycle around, so can just do odd/even selection, then normalize appropriately, then return interlaced array
        # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs!!! So be careful! (Can't do in-place data replacement in netCDF unfortunately)
        if dataset_type=="terrain":
            data = self.nc_dataset_terrain['h'][:].data
            normed_data = (data - np.mean(data))/np.std(data)
        elif dataset_type=="hrrr":
            data_00z = self.nc_dataset_pred['t'][0::2].data
            data_12z = self.nc_dataset_pred['t'][1::2].data
            tmp_00z = (data_00z - np.mean(data_00z))/np.std(data_00z)
            tmp_12z = (data_12z - np.mean(data_12z))/np.std(data_12z)
            normed_data = np.empty(self.nc_dataset_pred['t'][:].data.shape, dtype=tmp_00z.dtype)
            normed_data[0::2] = tmp_00z
            normed_data[1::2] = tmp_12z
        elif dataset_type=="urma":
            data_00z = self.nc_dataset_targ['t2m'][0::2].data
            data_12z = self.nc_dataset_targ['t2m'][1::2].data
            tmp_00z = (data_00z - np.mean(data_00z))/np.std(data_00z)
            tmp_12z = (data_12z - np.mean(data_12z))/np.std(data_12z)
            normed_data = np.empty(self.nc_dataset_targ['t2m'][:].data.shape, dtype=tmp_00z.dtype)
            normed_data[0::2] = tmp_00z
            normed_data[1::2] = tmp_12z
        else:
            print("ERROR: dataset_type must be one of 'terrain', 'hrrr', or 'urma' ")

        return normed_data
    
    
    def __getitem__(self, idx):

        # Determine if terrain map is included as separate channel
        with_terrain = False
        
        # get sample index for predictor and target
        p_idx = self.predictor_indices[idx]
        t_idx = self.target_indices[idx]

        # extract 2m temp image 
        # awkward: forgot to rename HRRR temp from "t" to "t2m" as in URMA...

        ## Unnormed 
        #predictor = self.nc_dataset_pred["t"][p_idx,:,:].data[np.newaxis,:,:]
        #target = self.nc_dataset_targ["t2m"][t_idx,:,:].data[np.newaxis,:,:]

        ## Normed
        predictor = self.dataset_pred_normed[idx][np.newaxis,:,:]
        target = self.dataset_targ_normed[idx][np.newaxis,:,:]
        if with_terrain:
            #predictor = np.concatenate((predictor, self.dataset_terrain_normed[np.newaxis,:,:]), axis=0)
            ##### Try appending the negative version of terrain to introduce a direct correlation with height and temperature
            predictor = np.concatenate((predictor, -1*self.dataset_terrain_normed[np.newaxis,:,:]), axis=0)
            

        return (predictor), (target) #copying Marshall's syntax, see if it works

######################################################################

class HRRR_URMA_Dataset_Anytime_Anydate(Dataset):
    def __init__(self, 
                 is_train=False, 
                 months=[1,3], 
                 days=[1,31], 
                 hour=0):
        # is_train --> load either training or testing datasets
        # months = 2-tuple of ints 1-12; first entry = start month, second entry = end month. All years 2021/22/23 will have data selected from this range
        # days = 2-tuple of ints 1-31; first entry = start day, second entry = end day
        # hour = int, 0 or 12 (currently). Determines if this uses 00z or 12z data
        self.months = months
        self.days = days #note user is resposible for making sure days[1] matches actual number of days in months[1]...
        self.hour = hour
        
        # Establish paths
        # UNLIKE Marshall's code, the training and predictor indices align exactly and are contained in the sample_idx coordinate, so no need for separate path
        # Also, now we have separate .nc files for training and testing, so need multiple paths there
        path_root = os.path.dirname(os.getcwd())
        if is_train:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "train_hrrr.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "train_urma.nc")
        else:
            data_save_path_pred = os.path.join(path_root,"Regridded_HRRR_train_test", "test_hrrr.nc")
            data_save_path_targ = os.path.join(path_root,"URMA_train_test", "test_urma.nc")

        # Open xarray dataarrays
        # Note using dataarrays rather than datasets because then we don't have to worry about variable names when subselecting
        self.xr_dataset_pred = xr.open_dataarray(data_save_path_pred)
        self.xr_dataset_targ = xr.open_dataarray(data_save_path_targ)
       
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
        self.date_idxs = date_idxs #for reference
        print(f'Done making date index list. Months = {self.months[0]} to {self.months[1]}, days = {self.days[0]} to {self.days[1]}')
        
        # Normalize datasets
        # Done here instead of in __getitem__ so it's only done once per Dataset call instead of once every item call
        # EXTREMELY IMPORTANT: this returns numpy arrays, NOT netCDFs/xarray!!! So be careful in __getitem__
        
        #terribleness needed due to extreme slowness of selecting URMA data with master date index list that contains discontinuous entries
        self.data_hrrr = np.concatenate((self.xr_dataset_pred[date_idxs0,:].data, self.xr_dataset_pred[date_idxs1,:].data, self.xr_dataset_pred[date_idxs2,:].data))
        self.data_urma = np.concatenate((self.xr_dataset_targ[date_idxs0,:].data, self.xr_dataset_targ[date_idxs1,:].data, self.xr_dataset_targ[date_idxs2,:].data))
        
        self.hrrr_mean = np.mean(self.data_hrrr)
        self.hrrr_std = np.std(self.data_hrrr)
        self.urma_mean = np.mean(self.data_urma)
        self.urma_std = np.std(self.data_urma)
        
        self.dataset_pred_normed = (self.data_hrrr - self.hrrr_mean)/self.hrrr_std
        self.dataset_targ_normed = (self.data_urma - self.urma_mean)/self.urma_std

        print("Done with normalization")

        # Restrict unnormed datasets (already only 00z or 12z) to only date indices
        # Needs to be done post-normalization as normalization needs to select from full (albiet 00z or 12z restricted) datasets
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

        return (predictor), (target) 

######################################################################

class HRRR_URMA_Dataset_Anytime_Anydate_Anyterrain(Dataset):
    def __init__(self, 
                 is_train=False, 
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
        self.xr_dataset_pred = xr.open_dataarray(data_save_path_pred)
        self.xr_dataset_targ = xr.open_dataarray(data_save_path_targ)
        if self.with_hrrr_terrain:
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr)
        if self.with_urma_terrain:
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma)
        if self.with_terrain_difference: #needed in case one or both of HRRR or/and URMA terrain aren't included, but their difference is
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr)
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma)
       
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

        if self.with_hrrr_terrain:
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
