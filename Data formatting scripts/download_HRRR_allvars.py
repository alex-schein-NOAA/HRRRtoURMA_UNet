from herbie import Herbie
from pathlib import Path
import os
from datetime import date, timedelta

############################################

var_select_dict = {"t2m":r"TMP:2 m", 
                   "d2m":r"DPT:2 m", 
                   "pressurf":r"PRES:surface",
                   "u10m":r"UGRD:10 m",
                   "v10m":r"VGRD:10 m"}

############################################

def download_one_hour_one_variable(DATE_STR, INIT_TIME, FORECAST_TIME, var_string, save_dir):
    H =  Herbie(f"{DATE_STR} {INIT_TIME}:00", model="hrrr", fxx=int(FORECAST_TIME), save_dir=PATH_TO_DOWNLOAD_TO, verbose=False)
    H.download(var_select_dict[var_string])
    return

############################################

INIT_TIME_LIST = [str(i).zfill(2) for i in range(24)]  #["00", "12"] #members will populate filename in the "t[INIT_TIME]z." portion
FORECAST_TIME = "01" # Note this is a string; needs to be cast into an int for Herbie. Will populate filename in the "wrfsfc[FORECAST_TIME]." portion

START_DATE = date(2021,1,1) #should be jan 1, 2021
END_DATE = date(2024,12,31) #should be dec 31, 2024
NUM_DAYS = END_DATE-START_DATE

for var_string in list(var_select_dict.keys())[1:]: #don't need t2m right now, but change this if we do later
    PATH_TO_DOWNLOAD_TO = f"/data1/projects/RTMA/alex.schein/Herbie_downloads/{var_string}"

    #download 2020/12/31 23z manually
    H = Herbie(f"2020-12-31 23:00", model="hrrr", fxx=1, save_dir=PATH_TO_DOWNLOAD_TO, verbose=False)
    H.download(var_select_dict[var_string])
    print(f"Done: {var_string} | 2020-12-31 23:00 | fxx={FORECAST_TIME}")

    for i in range(NUM_DAYS.days + 1):
        DATE_STR = date.strftime(START_DATE + timedelta(days=i), "%Y-%m-%d")
        
        for INIT_TIME in INIT_TIME_LIST:
            download_one_hour_one_variable(DATE_STR=DATE_STR, INIT_TIME=INIT_TIME, FORECAST_TIME=FORECAST_TIME, var_string=var_string, save_dir=PATH_TO_DOWNLOAD_TO)
            print(f"Done: {var_string} | {DATE_STR} | INIT_TIME={INIT_TIME} | fxx={FORECAST_TIME}")