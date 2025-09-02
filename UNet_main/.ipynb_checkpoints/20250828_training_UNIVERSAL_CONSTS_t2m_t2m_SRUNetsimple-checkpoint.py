# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# # from torchvision import datasets
# # from torchvision import transforms
# # from torchvision.transforms import ToTensor

# import os
# import glob
# import time
# import datetime as dt
# import pandas as pd
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from matplotlib.markers import MarkerStyle
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
# from FunctionsAndClasses.SR_UNet_simple import *
# from FunctionsAndClasses.DefineModelAttributes import *
# from FunctionsAndClasses.StatObjectConstructor import *
# from FunctionsAndClasses.utils import *
# from FunctionsAndClasses.CONSTANTS import *

# from FunctionsAndClasses.UNet_Attention_simple import *

from HEADER_SCRIPTS import *

###############

C = CONSTANTS()

current_model = DefineModelAttributes(is_train=True,
                                      is_attention_model=False,
                                      with_terrains=['diff'],
                                      predictor_vars=['t2m'],
                                      target_vars=['t2m'],
                                      BATCH_SIZE=24,
                                      NUM_EPOCHS=50)

TRAINING_LOG_FILEPATH = f"{C.DIR_UNET_MAIN}/Training_logs/training_log_20250828_UNIVERSAL_CONSTS_SRUNetsimple.txt"

TrainOneModel(current_model, is_attention_model=False, NUM_GPUS_TO_USE=2, TRAINING_LOG_FILEPATH = TRAINING_LOG_FILEPATH, TRAINED_MODEL_SAVEPATH = C.DIR_TRAINED_MODELS)

