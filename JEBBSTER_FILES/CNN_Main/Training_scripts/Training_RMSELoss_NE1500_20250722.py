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

#Train all preds --> all targs with 1000 epochs and RMSE loss
current_model = DefineModelAttributes()
current_model.savename = f"RMSELoss_{current_model.savename}"
TrainOneModel(current_model, loss_fcn="RMSE")

#Train all preds --> all targs with 1500 epochs and RMSE loss
current_model = DefineModelAttributes(NUM_EPOCHS=1500)
current_model.savename = f"RMSELoss_{current_model.savename}"
TrainOneModel(current_model, loss_fcn="RMSE")

#Train pressurf-->pressurf only model with 1500 epochs and RMSE loss
current_model = DefineModelAttributes(NUM_EPOCHS=1500, predictor_vars=["pressurf"], target_vars=["pressurf"])
current_model.savename = f"RMSELoss_{current_model.savename}"
TrainOneModel(current_model, loss_fcn="RMSE")