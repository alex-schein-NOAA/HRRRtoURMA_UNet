from pathlib import Path
import os
import shutil
from datetime import date, timedelta
import xarray as xr
import netCDF4 as nc
import numpy as np

####################################################

def restrict_files(START_DATE, END_DATE, TIME_LIST, PATH_ORIGINAL, PATH_NEW, IDX_MIN_LON=596, IDX_MIN_LAT=645, IMG_SIZE_LON=180, IMG_SIZE_LAT=180):
    # NOTE: default idx mins are for HRRR - need to change if using this for URMA!
    NUM_DAYS = END_DATE-START_DATE
    for i in range(NUM_DAYS.days+1):
        DATE_STR = date.strftime(START_DATE + timedelta(days=i), "%Y%m%d")
        filenames = os.listdir(f"{PATH_ORIGINAL}/{DATE_STR}")
        for time in TIME_LIST:
            filename = [x for x in filenames if f"t{time}z" in x and ".idx" not in x][0]
            new_filename = f"hrrr_regridded_spatiallyrestricted_{DATE_STR}_t{time}z.nc"
            if not os.path.exists(f"{PATH_NEW}/{new_filename}"):
                var = xr.open_dataset(f"{PATH_ORIGINAL}/{DATE_STR}/{filename}", engine="cfgrib", decode_timedelta=True)
                var_subset = var.isel(y=slice(IDX_MIN_LAT, IDX_MIN_LAT+IMG_SIZE_LAT),
                                      x=slice(IDX_MIN_LON, IDX_MIN_LON+IMG_SIZE_LON))
                var_subset.to_netcdf(f"{PATH_NEW}/{new_filename}")
                print(f"{new_filename} written to {PATH_NEW}")
            else:
                #print(f"{new_filename} already exists in {PATH_NEW}. No action taken")
                foo=1
                
    return

####################################################
var_select_dict = {"t2m":r"TMP:2 m", 
                   "d2m":r"DPT:2 m", 
                   "pressurf":r"PRES:surface",
                   "u10m":r"UGRD:10 m",
                   "v10m":r"VGRD:10 m"}
                   
                   
for var_string in [list(var_select_dict.keys())[1]]: #list(var_select_dict.keys()): #(6/24) don't need to do t2m, but include in the future

    PATH_HRRR_ORIGINAL = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR/{var_string}"
    PATH_TRAIN = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/LOOSE_FILES/train_spatiallyrestricted_f01/{var_string}"
    PATH_TEST = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/LOOSE_FILES/test_spatiallyrestricted_f01/{var_string}"
    
    START_DATE_TRAIN = date(2021,1,1) #should be jan 1, 2021
    END_DATE_TRAIN = date(2023,12,31) #should be dec 31, 2023
    
    START_DATE_TEST = date(2024,1,1) #should be jan 1, 2024
    END_DATE_TEST = date(2024,12,31) #should be dec 31, 2024
    
    TIME_LIST = [str(i).zfill(2) for i in range(24)]
    
    restrict_files(START_DATE=START_DATE_TRAIN,
                   END_DATE=END_DATE_TRAIN, 
                   TIME_LIST=TIME_LIST,
                   PATH_ORIGINAL=PATH_HRRR_ORIGINAL,
                   PATH_NEW=PATH_TRAIN)
    
    restrict_files(START_DATE=START_DATE_TEST,
                   END_DATE=END_DATE_TEST, 
                   TIME_LIST=TIME_LIST,
                   PATH_ORIGINAL=PATH_HRRR_ORIGINAL,
                   PATH_NEW=PATH_TEST)