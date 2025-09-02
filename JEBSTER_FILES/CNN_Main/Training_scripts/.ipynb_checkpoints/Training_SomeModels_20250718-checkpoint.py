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

from SR_UNet_simple import SR_UNet_simple
from DefineModelAttributes import DefineModelAttributes
from HRRR_URMA_Datasets_AllVars import HRRR_URMA_Dataset_AllVars
from utils import *

torch.manual_seed(42)

################################################################################

#Train pressurf-->pressurf only model, to see if it performs better than all preds --> pressurf
current_model = DefineModelAttributes(predictor_vars=["pressurf"], target_vars=["pressurf"])
TrainOneModel(current_model)

#Train all preds --> [u10m, v10m]
current_model = DefineModelAttributes(target_vars=["u10m","v10m"])
TrainOneModel(current_model)

#Train all preds --> all targets
current_model = DefineModelAttributes()
TrainOneModel(current_model)
