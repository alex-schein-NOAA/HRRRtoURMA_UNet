import subprocess
import os
from time import sleep
from datetime import datetime as dt

MAX_CONCURRENT_PROCESSES = 25
SCRIPT = "/scratch/AIWP/Marshall.Baldwin/scripts/hrrr_emulator/grib_extraction/parallelized/per_date_grib2nc_hrrr.py"

def main():

    days = [day for day in sorted(os.listdir("/data1/ai-datadepot/models/hrrr/conus/grib2-subset")) if day.isdigit()]
    days = [day for day in days if (dt.strptime(day, "%Y%m%d") > dt(2024, 10, 1)) or (dt.strptime(day, "%Y%m%d") >= dt(2023, 1, 1) and dt.strptime(day, "%Y%m%d") < dt(2023, 5, 1))] #TEMPORARY!! Remove once DSG has all dates
    
    running_procs = []
    for day in days:

        #check if there's room to spin up the next process
        while len(running_procs) >= MAX_CONCURRENT_PROCESSES:
            running_procs = [p for p in running_procs if p.poll() is None]
            sleep(1)

        #start extracting data for the day
        p = subprocess.Popen(["python3", SCRIPT, "--day", day])
        running_procs.append(p)

    # Wait for all remaining processes to finish
    for p in running_procs:
        p.wait()

if __name__ == "__main__":
    main()
