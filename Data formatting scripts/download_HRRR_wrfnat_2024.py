from herbie import Herbie
from pathlib import Path
import os
from datetime import date, timedelta
import time

############################################

def download_one_hour(DATE_STR, INIT_TIME, FORECAST_TIME, save_dir):
    H =  Herbie(f"{DATE_STR} {INIT_TIME}:00", model="hrrr", fxx=FORECAST_TIME, product="nat", save_dir=PATH_TO_DOWNLOAD_TO, verbose=False)
    H.download()
    return

############################################

PATH_TO_DOWNLOAD_TO = f"/data1/projects/RTMA/alex.schein/Herbie_downloads_2024"

INIT_TIME_LIST = [str(i).zfill(2) for i in range(24)]  #["00", "12"] #members will populate filename in the "t[INIT_TIME]z." portion
FORECAST_TIME = 1 # Note this is an int for Herbie. Will populate filename in the "wrfnat[FORECAST_TIME]." portion

START_DATE = date(2024,1,1) 
END_DATE = date(2024,12,31) #should be dec 31, 2024
NUM_DAYS = END_DATE-START_DATE


#download 2023/12/31 23z manually
H = Herbie(f"2023-12-31 23:00", model="hrrr", fxx=1, product="nat", save_dir=PATH_TO_DOWNLOAD_TO, verbose=False)
H.download()
print(f"Done | 2023-12-31 23:00 | fxx={FORECAST_TIME}")

for i in range(NUM_DAYS.days + 1):
    DATE_STR = date.strftime(START_DATE + timedelta(days=i), "%Y-%m-%d")
    
    for INIT_TIME in INIT_TIME_LIST:
        start=time.time()
        download_one_hour(DATE_STR=DATE_STR, INIT_TIME=INIT_TIME, FORECAST_TIME=FORECAST_TIME, save_dir=PATH_TO_DOWNLOAD_TO)
        print(f"Done | {DATE_STR} | INIT_TIME={INIT_TIME} | fxx={FORECAST_TIME} | Time taken = {time.time() - start:.1f} sec")