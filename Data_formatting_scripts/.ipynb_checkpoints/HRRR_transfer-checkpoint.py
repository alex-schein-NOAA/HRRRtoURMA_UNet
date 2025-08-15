from herbie import Herbie
from pathlib import Path
import os
from datetime import date, timedelta
import numpy as np
import xarray as xr

#########

PATH_TO_DOWNLOAD_TO = r"/scratch/RTMA/alex.schein/Herbie_downloads"

INIT_TIME_LIST = [str(i).zfill(2) for i in range(24)]  #["00", "12"] #members will populate filename in the "t[INIT_TIME]z." portion
FORECAST_TIME_LIST = ["01"] #["00"] #members will populate filename in the "wrfsfc[FORECAST_TIME]." portion

START_DATE = date(2022,7,28) #should be jan 1, 2021
END_DATE = date(2024,12,31) #should be dec 31, 2024
NUM_DAYS = END_DATE-START_DATE

for i in range(NUM_DAYS.days + 1):
    DATE_STR = date.strftime(START_DATE + timedelta(days=i), "%Y-%m-%d")
    for INIT_TIME in INIT_TIME_LIST:
        for FORECAST_TIME in FORECAST_TIME_LIST:
            H_HRRR = Herbie(f"{DATE_STR} {INIT_TIME}:00", model="hrrr", fxx=int(FORECAST_TIME), save_dir=PATH_TO_DOWNLOAD_TO, verbose=False)
            H_HRRR.download(r":TMP:2 m") #MISTAKENLY DOWNLOADED SURFACE TEMP BEFORE....
            print(f"Done: {DATE_STR} | INIT_TIME={INIT_TIME} | fxx={FORECAST_TIME}")