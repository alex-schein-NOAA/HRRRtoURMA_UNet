from pathlib import Path
import os
import shutil
from datetime import date, timedelta
import xarray as xr
import netCDF4 as nc
import numpy as np
import glob
import dask
import time

#####################################################

var_select_dict = {"t2m":r"TMP:2 m", 
                   "d2m":r"DPT:2 m", 
                   "pressurf":r"PRES:surface",
                   "u10m":r"UGRD:10 m",
                   "v10m":r"VGRD:10 m"}
    
for var_string in list(var_select_dict.keys()): #(7/30) doing all vars on new domain
    PATH_HRRR_TRAIN = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/LOOSE_FILES/train_spatiallyrestricted_f01/{var_string}"
    PATH_HRRR_TEST = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/LOOSE_FILES/test_spatiallyrestricted_f01/{var_string}"
    
    files_train_hrrr = sorted(glob.glob(PATH_HRRR_TRAIN+"/*.nc"))
    files_test_hrrr = sorted(glob.glob(PATH_HRRR_TEST+"/*.nc"))
    
    def check_files_for_problems(input_files):
        #input: sorted glob list of filepaths to directory of loose netcdfs
        #Checks if the coords from the first dataset (asssumed to be good) are in all files
        #Far from comprehensive but usually catches if a regridded file failed for whatever reason
        flag = 0
        ds0 = xr.open_dataset(input_files[0], decode_timedelta=True)
        for i, filepath in enumerate(input_files):
            ds1 = xr.open_dataset(filepath, decode_timedelta=True)
            if i % (int(len(input_files)/50))==0:
                print(f"{(i/len(input_files))*100:.0f}% of files checked")
            if list(ds0.coords) != list(ds1.coords):
                print(f"Failure in {filepath}")
                flag = 1
        return flag
    
    def concat_netcdfs(input_files, output_filepath):
        #input: sorted glob list of filepaths to directory of loose netcdfs
        #input: filepath, including name and .nc extension, of output master netcdf
    
        #Check for problems in the files before wasting time 
        # flag = check_files_for_problems(input_files) #Uncomment if running this for unchecked files
        flag = 0 #Comment if running this for unchecked files
    
        if flag:
            print(f"Problems found. Not concatenating until all problems fixed")
        else:
            if not os.path.exists(output_filepath): #prevents accidental overwrites if file was already written
                trunc_input_files = input_files[1:] #to fix indexing issues - very bad but whatever
            
                ds0 = xr.open_dataset(input_files[0], decode_timedelta=True)
                for i, filename in enumerate(trunc_input_files):
                    ds1 = xr.open_dataset(trunc_input_files[i], decode_timedelta=True)
                    ds0 = xr.concat([ds0,ds1], dim="valid_time_dim")
                    if i % (int(len(trunc_input_files)/100))==0:
                        print(f"{i}/{len(trunc_input_files)} concatenated")
                        with open(f"/scratch/RTMA/alex.schein/LOG_u10m_v10m_HRRR_concat_20250702.txt", "a") as file: ##### CHANGE THIS FILEPATH AS NEEDED
                          file.write(f"{var_string} | {i}/{len(trunc_input_files)} concatenated \n")
    
                    
                ds_concat = ds0.assign_coords(sample_idx=("valid_time_dim",[i for i in range(len(input_files))])) #uses input_files, not trunc version, as length should equal # of loose netcdf files in original directory
                ds_concat = ds_concat.swap_dims(dims_dict={"valid_time_dim":"sample_idx"})
            
                ds_concat.to_netcdf(output_filepath)
                with open(f"/scratch/RTMA/alex.schein/LOG_u10m_v10m_HRRR_concat_20250702.txt", "a") as file: ##### CHANGE THIS FILEPATH AS NEEDED
                    file.write(f"{output_filepath} written to disk \n")
    
        return
    
    
    #### POST 5/23: these files refer to ALL times
    ## NOTE (5/23): NEED TO REGRID HRRR FILES! Then spatially restrict, THEN concatenate...
    
    # path_test_urma = "/scratch/RTMA/alex.schein/URMA_train_test/test_urma_alltimes.nc"
    # path_train_urma = "/scratch/RTMA/alex.schein/URMA_train_test/train_urma_alltimes.nc"
    
    output_path_train_hrrr = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/train_hrrr_alltimes_{var_string}_f01.nc"
    output_path_test_hrrr = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/test_hrrr_alltimes_{var_string}_f01.nc"
    
    
    ## (6/10) Only need to make new URMA files if big changes are made (e.g. new variables, new domain). Forecast time change only --> only new HRRR files need to be made
    # if not os.path.exists(path_train_urma):
    #     concat_netcdfs(files_train_urma, path_train_urma)
    # else:
    #     print(f"{path_train_urma} already exists")
    
    # if not os.path.exists(path_test_urma):
    #     concat_netcdfs(files_test_urma, path_test_urma)
    # else:
    #     print(f"{path_test_urma} already exists")
        
    ## Run this before doing concatenation, if doing many vars at once, so any errors can be caught before investing a lot of time
#    print(f"Checking {var_string}")
#    check_files_for_problems(files_train_hrrr)
#    check_files_for_problems(files_test_hrrr)
    
    if not os.path.exists(output_path_train_hrrr):
      concat_netcdfs(files_train_hrrr, output_path_train_hrrr)
    else:
      print(f"{path_train_hrrr} already exists")
    
    if not os.path.exists(output_path_test_hrrr):
      concat_netcdfs(files_test_hrrr, output_path_test_hrrr)
    else:
      print(f"{path_test_hrrr} already exists")
