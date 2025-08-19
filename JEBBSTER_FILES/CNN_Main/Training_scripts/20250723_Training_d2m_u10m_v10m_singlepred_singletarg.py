import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import os
import time
import datetime as dt
from netCDF4 import Dataset as nc_Dataset
import pandas as pd
import numpy as np
import xarray as xr

from FunctionsAndClasses.SR_UNet_simple import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.utils import *

torch.manual_seed(42)

################################################################################

# Check if current training (pressurf --> pressurf is done yet)
while os.path.exists(f"/scratch/RTMA/alex.schein/CNN_Main/Trained_models/RMSELoss_BS256_NE1500_tD_pred(pressurf)_targ(pressurf)_TEMP.pt"):
    print("Current job not yet finished. Sleeping for 30 seconds")
    time.sleep(30)

os.rename(f"/scratch/RTMA/alex.schein/CNN_Main/Training_logs/training_log.txt",
          f"/scratch/RTMA/alex.schein/CNN_Main/Training_logs/training_log_20250722.txt")

### Train single pred --> single targ with 1000 epochs and MAE loss
# d2m
current_model = DefineModelAttributes(predictor_vars=["d2m"], target_vars=["d2m"])
TrainOneModel(current_model, loss_fcn="MAE", TRAINED_MODEL_SAVEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Trained_models/MAE_Loss")

# u10m
current_model = DefineModelAttributes(predictor_vars=["u10m"], target_vars=["u10m"])
TrainOneModel(current_model, loss_fcn="MAE", TRAINED_MODEL_SAVEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Trained_models/MAE_Loss")

# v10m
current_model = DefineModelAttributes(predictor_vars=["v10m"], target_vars=["v10m"])
TrainOneModel(current_model, loss_fcn="MAE", TRAINED_MODEL_SAVEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Trained_models/MAE_Loss")