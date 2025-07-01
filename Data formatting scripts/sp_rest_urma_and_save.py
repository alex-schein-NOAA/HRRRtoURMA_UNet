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

############################################

IDX_MIN_LON = 796 
IDX_MIN_LAT = 645 

IMG_SIZE_LON = 180
IMG_SIZE_LAT = 180

TIME_LIST = [str(i).zfill(2) for i in range(24)] 

PATH_URMA_ORIGINAL = "/data1/ai-datadepot/models/urma/2p5km/grib2"

#Translates from HRRR var names to URMA; files directory structure uses the HRRR format but need to use URMA to select data
varname_translation_dict = {"t2m":"t2m",
                            "d2m":"d2m",
                            "pressurf":"sp",
                            "u10m":"u10",
                            "v10m":"v10"}

urma_var_select_dict = {"t2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                        "d2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                        "sp":"",
                        "u10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}},
                        "v10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}}}

START_DATE_TRAIN = date(2021,1,1) #should be jan 1, 2021
END_DATE_TRAIN = date(2023,12,31) #should be dec 31, 2023
NUM_DAYS_TRAIN = END_DATE_TRAIN-START_DATE_TRAIN

START_DATE_TEST = date(2024,1,1) #should be jan 1, 2024
END_DATE_TEST = date(2024,12,31) #should be dec 31, 2024
NUM_DAYS_TEST = END_DATE_TEST-START_DATE_TEST

#############################

def restrict_files(START_DATE, END_DATE, TIME_LIST, PATH_ORIGINAL, PATH_NEW, var_string, IDX_MIN_LON=796, IDX_MIN_LAT=645, IMG_SIZE_LON=180, IMG_SIZE_LAT=180):
    NUM_DAYS = END_DATE-START_DATE
    for i in range(NUM_DAYS.days+1):
        DATE_STR = date.strftime(START_DATE + timedelta(days=i), "%Y%m%d")
        filenames = os.listdir(f"{PATH_ORIGINAL}/{DATE_STR}")
        for time in TIME_LIST:
            filename = [x for x in filenames if time in x and ".idx" not in x][0] #will only be one matching filename @ appropriate time
            new_filename = f"urma_{DATE_STR}_t{time}z.nc"
            if not os.path.exists(f"{PATH_NEW}/{new_filename}"):
            #throws a hissy fit as it can't write an index file on ai-datadepot, but it should still compute fine...
                urma_orig = xr.open_dataset(f"{PATH_ORIGINAL}/{DATE_STR}/{filename}", 
                                            engine='cfgrib', 
                                            backend_kwargs=urma_var_select_dict[varname_translation_dict[var_string]],
                                            decode_timedelta=True)
                var = urma_orig[varname_translation_dict[var_string]]
                var_subset = var.isel(y=slice(IDX_MIN_LAT, IDX_MIN_LAT+IMG_SIZE_LAT),
                                      x=slice(IDX_MIN_LON, IDX_MIN_LON+IMG_SIZE_LON))
                var_subset.to_netcdf(f"{PATH_NEW}/{new_filename}")
                with open(f"/scratch/RTMA/alex.schein/tmp_dump.txt",'a') as file: #CHANGE FILEPATH AS NEEDED 
                    file.write(f"{new_filename} written to {PATH_NEW} \n")
            else:
                foo=1
                #with open(f"/scratch/RTMA/alex.schein/tmp_dump.txt",'a') as file: #CHANGE FILEPATH AS NEEDED
                #    file.write(f"{new_filename} already exists in {PATH_NEW}. No action taken \n")

    return

#############################
    
for var_string in list(varname_translation_dict.keys())[1:]: #don't need t2m right now, but change this if we do later

    PATH_URMA_TRAIN = f"/data1/projects/RTMA/alex.schein/URMA_train_test/LOOSE_FILES/train/{var_string}"
    PATH_URMA_TEST = f"/data1/projects/RTMA/alex.schein/URMA_train_test/LOOSE_FILES/test/{var_string}"

    restrict_files(START_DATE=START_DATE_TRAIN,
                   END_DATE=END_DATE_TRAIN, 
                   TIME_LIST=TIME_LIST,
                   PATH_ORIGINAL=PATH_URMA_ORIGINAL,
                   PATH_NEW=PATH_URMA_TRAIN,
                   var_string=var_string)

    restrict_files(START_DATE=START_DATE_TEST,
                   END_DATE=END_DATE_TEST, 
                   TIME_LIST=TIME_LIST,
                   PATH_ORIGINAL=PATH_URMA_ORIGINAL,
                   PATH_NEW=PATH_URMA_TEST,
                   var_string=var_string)

    