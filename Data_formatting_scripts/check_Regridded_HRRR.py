# Checks Herbie downloads to make sure that one or more didn't fail
# RUN THIS BEFORE DOING REGRIDDING SO WE DON'T GET ERRORS

import os


HERBIE_DIR = f"/data1/projects/RTMA/alex.schein/Regridded_HRRR"

file_paths = []
for root, _, files in os.walk(f"{HERBIE_DIR}/spfh2m"): #(8/4) just need to check spfh2m now, but modify as needed
    for file in files:
        if ".idx" not in os.path.join(root, file):
          if os.path.getsize(os.path.join(root, file)) < 1000: #less than 1 MB --> something's probably gone wrong (subsetted variable over CONUS should be ~1.5 MB)
            print(f"{os.path.join(root, file)} is too small") 
