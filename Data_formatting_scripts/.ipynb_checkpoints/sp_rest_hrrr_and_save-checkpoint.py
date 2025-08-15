from pathlib import Path
import os
import shutil
import datetime as dt #import date, timedelta
import xarray as xr
import netCDF4 as nc
import numpy as np
import glob
import dask
import time


####################################################

def restrict_files(START_DATE, END_DATE, FORECAST_LEAD_HOURS, PATH_ORIGINAL, PATH_NEW, IDX_MIN_LON=596, IDX_MIN_LAT=645, IMG_SIZE_LON=180, IMG_SIZE_LAT=180):
    ## DEFAULT INDEXES AND IMG SIZES ARE FOR OLD DOMAIN!! Keeping as a record though
    ## New domain is based on URMA grid indexing:
    # IDX_MIN_LON = 250
    # IDX_MIN_LAT = 400
    # IMG_SIZE_LON = 800
    # IMG_SIZE_LAT = 800

    NUM_HOURS = int((END_DATE-START_DATE).total_seconds()/3600) #surprisingly no built in hours function...
    for i in range(NUM_HOURS+1):
        DATE_STR = dt.date.strftime(START_DATE + dt.timedelta(hours=i), "%Y%m%d")
        filename_to_open = f"hrrr_regridded_{DATE_STR}_t{str((START_DATE.hour+i)%24).zfill(2)}z_f{str(FORECAST_LEAD_HOURS).zfill(2)}.grib2"
        new_filename = f"hrrr_regridded_spatiallyrestricted_{DATE_STR}_t{str((START_DATE.hour+i)%24).zfill(2)}z.nc"
        if not os.path.exists(f"{PATH_NEW}/{new_filename}"):
            var = xr.open_dataset(f"{PATH_ORIGINAL}/{DATE_STR}/{filename_to_open}", engine="cfgrib", decode_timedelta=True)
            var_subset = var.isel(y=slice(IDX_MIN_LAT, IDX_MIN_LAT+IMG_SIZE_LAT),
                                  x=slice(IDX_MIN_LON, IDX_MIN_LON+IMG_SIZE_LON))
            var_subset.to_netcdf(f"{PATH_NEW}/{new_filename}")
            print(f"{new_filename} written to {PATH_NEW}")
        else:
            print(f"{new_filename} already exists in {PATH_NEW}. No action taken")

####################################################
var_select_dict = {"t2m":r"TMP:2 m", 
                   "d2m":r"DPT:2 m", 
                   "pressurf":r"PRES:surface",
                   "u10m":r"UGRD:10 m",
                   "v10m":r"VGRD:10 m"}

# (7/28) New domain indices
IDX_MIN_LON = 250
IDX_MIN_LAT = 400
IMG_SIZE_LON = 800
IMG_SIZE_LAT = 800

for var_string in list(var_select_dict.keys()): #(7/28) Doing variables on new domain. Need to get spfh2m regridded then run this script again on that var

    PATH_HRRR_ORIGINAL = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR/{var_string}"
    PATH_TRAIN = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/LOOSE_FILES/train_spatiallyrestricted_f01/{var_string}"
    PATH_TEST = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/LOOSE_FILES/test_spatiallyrestricted_f01/{var_string}"

    FORECAST_LEAD_HOURS = 1

    #This enables naiive datetime starts and ends for the study period, with the proper restrictions and file movements done in the regrid function
    START_DATE_TRAIN = dt.datetime(2021,1,1,0)-dt.timedelta(hours=FORECAST_LEAD_HOURS) 
    END_DATE_TRAIN = dt.datetime(2023,12,31,23)-dt.timedelta(hours=FORECAST_LEAD_HOURS) 

    START_DATE_TEST = dt.datetime(2024,1,1,0)-dt.timedelta(hours=FORECAST_LEAD_HOURS) 
    END_DATE_TEST = dt.datetime(2024,12,31,23)-dt.timedelta(hours=FORECAST_LEAD_HOURS) 
    
    restrict_files(START_DATE=START_DATE_TRAIN,
                   END_DATE=END_DATE_TRAIN, 
                   FORECAST_LEAD_HOURS=FORECAST_LEAD_HOURS,
                   PATH_ORIGINAL=PATH_HRRR_ORIGINAL,
                   PATH_NEW=PATH_TRAIN, 
                   IDX_MIN_LAT=IDX_MIN_LAT, 
                   IDX_MIN_LON=IDX_MIN_LON,
                   IMG_SIZE_LAT=IMG_SIZE_LAT,
                   IMG_SIZE_LON=IMG_SIZE_LON)
    
    restrict_files(START_DATE=START_DATE_TEST,
                   END_DATE=END_DATE_TEST, 
                   FORECAST_LEAD_HOURS=FORECAST_LEAD_HOURS,
                   PATH_ORIGINAL=PATH_HRRR_ORIGINAL,
                   PATH_NEW=PATH_TEST,
                   IDX_MIN_LAT=IDX_MIN_LAT, 
                   IDX_MIN_LON=IDX_MIN_LON,
                   IMG_SIZE_LAT=IMG_SIZE_LAT,
                   IMG_SIZE_LON=IMG_SIZE_LON)