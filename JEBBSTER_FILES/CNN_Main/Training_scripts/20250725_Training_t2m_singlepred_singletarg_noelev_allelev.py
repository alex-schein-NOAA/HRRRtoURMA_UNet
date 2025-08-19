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

# Train with no elevation data
current_model = DefineModelAttributes(with_terrains=None, predictor_vars=["t2m"], target_vars=["t2m"])
TrainOneModel(current_model, loss_fcn="MAE", TRAINED_MODEL_SAVEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Trained_models/MAE_Loss")

# Train with only HRRR and URMA elevation data
current_model = DefineModelAttributes(with_terrains=["hrrr","urma"], predictor_vars=["t2m"], target_vars=["t2m"])
TrainOneModel(current_model, loss_fcn="MAE", TRAINED_MODEL_SAVEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Trained_models/MAE_Loss")

# Train with all elevation data
current_model = DefineModelAttributes(with_terrains=["hrrr","urma","diff"], predictor_vars=["t2m"], target_vars=["t2m"])
TrainOneModel(current_model, loss_fcn="MAE", TRAINED_MODEL_SAVEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Trained_models/MAE_Loss")
