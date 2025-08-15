import os
import glob
import time
import datetime as dt
from netCDF4 import Dataset as nc_Dataset
from netCDF4 import date2num, num2date
import pandas as pd
import numpy as np
import math
import xarray as xr

from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.SR_UNet_simple import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.StatObjectConstructor import *
from FunctionsAndClasses.utils import *

###############################

varname_translation_dict = {"pressurf":"sp",
                            "t2m":"t2m",
                            "d2m":"d2m",
                            "spfh2m":"sh2",
                            "u10m":"u10",
                            "v10m":"v10"}

TRAIN_TEST_DIR = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/"
MEANS_FILE_TEST = "test_hrrr_means_stddevs_western_domain.txt"
MEANS_FILE_TRAIN = "train_hrrr_means_stddevs_western_domain.txt"

for var_name in list(varname_translation_dict.keys()):
    with xr.open_dataset(f"{TRAIN_TEST_DIR}/test_hrrr_alltimes_{var_name}_f01.grib2", engine="cfgrib", decode_timedelta=True) as xr_ds_test:
        var_mean_test = xr_ds_test[varname_translation_dict[var_name]].mean()
        var_stddev_test = xr_ds_test[varname_translation_dict[var_name]].std()
        with open(f"{TRAIN_TEST_DIR}/{MEANS_FILE_TEST}", "a") as file:
            file.write(f"TEST {var_name} | mean = {var_mean_test:.16f} | stddev = {var_stddev_test:.16f} \n")

    with xr.open_dataset(f"{TRAIN_TEST_DIR}/train_hrrr_alltimes_{var_name}_f01.grib2", engine="cfgrib", decode_timedelta=True) as xr_ds_train:
        var_mean_train = xr_ds_train[varname_translation_dict[var_name]].mean()
        var_stddev_train = xr_ds_train[varname_translation_dict[var_name]].std()
        with open(f"{TRAIN_TEST_DIR}/{MEANS_FILE_TRAIN}", "a") as file:
            file.write(f"TRAIN {var_name} | mean = {var_mean_train:.16f} | stddev = {var_stddev_train:.16f} \n")

