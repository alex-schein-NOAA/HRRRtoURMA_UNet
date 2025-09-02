# import os
# import glob
# import time
# import datetime as dt
# import numpy as np
# import xarray as xr

# import csv

# from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
# from FunctionsAndClasses.SR_UNet_simple import *
# from FunctionsAndClasses.DefineModelAttributes import *
# from FunctionsAndClasses.StatObjectConstructor import *
# from FunctionsAndClasses.utils import *
# from FunctionsAndClasses.CONSTANTS import *

from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.CONSTANTS import *
from FunctionsAndClasses.StatObjectConstructor import *

import csv

################################################################

## SHOULD ONLY BE DONE IF DATE OR SPATIAL RANGE CHANGES
# Also if adding in additional vars (though no need to repeat computation for already-completed vars)
# (2025-08-21) Redoing for new Western domain, and adding in spfh2m

C = CONSTANTS()

for target_var in list(C.hrrr_means_dict['train'].keys()):

    if not os.path.exists(f"{C.DIR_UNET_MAIN}/Smartinit_stats/smartinit_RMSE_alltimes_{target_var}.csv"):
        print(f"Starting on {target_var}")
        statobj_smartinit = ConstructStatObject(is_smartinit=True, target_var=target_var)
        statobj_smartinit.calc_domain_avg_RMSE_alltimes()
    
        # with open(f"smartinit_RMSE_alltimes_{target_var}.csv", "w", newline='') as file:
        #     csv_writer = csv.writer(file)
        #     csv_writer.writerow(statobj_smartinit.domain_avg_rmse_alltimes_list)
        # print(f"{target_var} csv written")