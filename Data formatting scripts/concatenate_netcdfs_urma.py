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
                   
#Translates from HRRR var names to URMA; files directory structure uses the HRRR format but need to use URMA to select data
#May not be needed here - only concatenating on valid_time dim which is present in all files regardless of varname - but might be needed in future
varname_translation_dict = {"t2m":"t2m",
                            "d2m":"d2m",
                            "pressurf":"sp",
                            "u10m":"u10",
                            "v10m":"v10"}
    
for var_string in list(var_select_dict.keys())[1:]: #list(var_select_dict.keys()): #(7/1) Doing all vars aside from t2m
    PATH_URMA_TRAIN = f"/data1/projects/RTMA/alex.schein/URMA_train_test/LOOSE_FILES/train/{var_string}"
    PATH_URMA_TEST = f"/data1/projects/RTMA/alex.schein/URMA_train_test/LOOSE_FILES/test/{var_string}"
    
    files_train_urma = sorted(glob.glob(PATH_URMA_TRAIN+"/*.nc"))
    files_test_urma = sorted(glob.glob(PATH_URMA_TEST+"/*.nc"))
    
    def check_files_for_problems(input_files):
        #input: sorted glob list of filepaths to directory of loose netcdfs
        #Checks if the coords from the first dataset (asssumed to be good) are in all files
        #Far from comprehensive but usually catches if a regridded file failed for whatever reason
        flag = 0
        ds0 = xr.open_dataset(input_files[0], decode_timedelta=True)
        for i, filepath in enumerate(input_files):
            ds1 = xr.open_dataset(filepath, decode_timedelta=True)
            if i % (int(len(input_files)/100))==0:
                print(f"{i}/{len(input_files)} files checked")
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
                        with open(f"/scratch/RTMA/alex.schein/LOG_allvars_URMA_concat_20250701.txt", "a") as file: ##### CHANGE THIS FILEPATH AS NEEDED
                          file.write(f"{var_string} | {i}/{len(trunc_input_files)} concatenated \n")
    
                    
                ds_concat = ds0.assign_coords(sample_idx=("valid_time_dim",[i for i in range(len(input_files))])) #uses input_files, not trunc version, as length should equal # of loose netcdf files in original directory
                ds_concat = ds_concat.swap_dims(dims_dict={"valid_time_dim":"sample_idx"})
            
                ds_concat.to_netcdf(output_filepath)
                print(f"{output_filepath} written to disk")
    
        return
    
    
    
    output_path_train_urma = f"/data1/projects/RTMA/alex.schein/URMA_train_test/train_urma_alltimes_{var_string}.nc"
    output_path_test_urma = f"/data1/projects/RTMA/alex.schein/URMA_train_test/test_urma_alltimes_{var_string}.nc"
    
    
    
    ## Run this before doing concatenation, if doing many vars at once, so any errors can be caught before investing a lot of time
    print(f"Checking {var_string}")
    check_files_for_problems(files_train_urma)
    check_files_for_problems(files_test_urma)
    
    # if not os.path.exists(path_train_urma):
    #     concat_netcdfs(files_train_urma, path_train_urma)
    # else:
    #     print(f"{path_train_urma} already exists")
    
    # if not os.path.exists(path_test_urma):
    #     concat_netcdfs(files_test_urma, path_test_urma)
    # else:
    #     print(f"{path_test_urma} already exists")
        
    
    
    