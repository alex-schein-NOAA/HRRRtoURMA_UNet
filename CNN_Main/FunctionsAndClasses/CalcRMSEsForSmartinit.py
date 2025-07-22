import os
import glob
import time
import datetime as dt
import numpy as np
import xarray as xr

import csv

from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.SR_UNet_simple import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.StatObjectConstructor import *
from FunctionsAndClasses.utils import *

## SHOULD ONLY BE DONE IF DATE OR SPATIAL RANGE CHANGES
# Also if adding in additional vars (though no need to repeat computation for already-completed vars)
for target_var in ["d2m", "pressurf", "u10m", "v10m"]: #t2m already done

    if not os.path.exists(f"/scratch/RTMA/alex.schein/CNN_Main/Smartinit_stats/smartinit_RMSE_alltimes_{target_var}.csv"):
        print(f"Starting on {target_var}")
        statobj_smartinit = ConstructStatObject(is_smartinit=True, target_var=target_var)
        statobj_smartinit.CalcDomainAvgRMSEAllTimes()
    
        with open(f"smartinit_RMSE_alltimes_{target_var}.csv", "w", newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(statobj_smartinit.domain_avg_rmse_alltimes_list)
        print(f"{target_var} csv written")