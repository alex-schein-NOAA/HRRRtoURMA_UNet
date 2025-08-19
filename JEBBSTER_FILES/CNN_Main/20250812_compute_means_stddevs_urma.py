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

TRAIN_TEST_DIR = f"/data1/projects/RTMA/alex.schein/URMA_train_test/"
MEANS_FILE_TEST = "test_urma_means_stddevs_western_domain.txt"
MEANS_FILE_TRAIN = "train_urma_means_stddevs_western_domain.txt"

for var_name in list(varname_translation_dict.keys()):
    with xr.open_dataset(f"{TRAIN_TEST_DIR}/test_urma_alltimes_{var_name}.grib2", engine="cfgrib", decode_timedelta=True) as xr_ds_test:
        tmp_means_list = []
        for idx in range(len(xr_ds_test.time)):
            tmp_means_list.append(np.mean(xr_ds_test[varname_translation_dict[var_name]][idx].data))
        var_mean_test = np.mean(tmp_means_list)
        var_stddev_test = np.std(xr_ds_test[varname_translation_dict[var_name]].data, mean=var_mean_test)
        with open(f"{TRAIN_TEST_DIR}/{MEANS_FILE_TEST}", "a") as file:
            file.write(f"TEST {var_name} | mean = {var_mean_test:.16f} | stddev = {var_stddev_test:.16f} \n")

    with xr.open_dataset(f"{TRAIN_TEST_DIR}/train_urma_alltimes_{var_name}.grib2", engine="cfgrib", decode_timedelta=True) as xr_ds_train:
        tmp_means_list = []
        for idx in range(len(xr_ds_train.time)):
            tmp_means_list.append(np.mean(xr_ds_train[varname_translation_dict[var_name]][idx].data))
        var_mean_train = np.mean(tmp_means_list)
        var_stddev_train = np.std(xr_ds_train[varname_translation_dict[var_name]].data, mean=var_mean_train)
        with open(f"{TRAIN_TEST_DIR}/{MEANS_FILE_TRAIN}", "a") as file:
            file.write(f"TRAIN {var_name} | mean = {var_mean_train:.16f} | stddev = {var_stddev_train:.16f} \n")

