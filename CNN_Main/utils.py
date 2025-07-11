import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import time
import datetime as dt
from netCDF4 import Dataset as nc_Dataset
import pandas as pd
import numpy as np
import xarray as xr

from HRRR_URMA_Datasets_AllVars import *


######################################################################################################################################################

def get_model_output_at_idx(model_attrs, model, pred_var="t2m", targ_var = "t2m", idx=0, is_unnormed=True, device="cuda:3"):
    """
    Inputs:
        - model_attrs: DefineModelAttributes object. MUST HAVE .create_dataset() ALREADY CALLED! 
        - model: Pytorch model to use, with weights loaded and device initialized
        - pred_var: string for the predictor variable to get the output of
        - targ_var: string for the target variable to get the output of
        - idx: index to get the output of
        - is_unnormed: if true, returns the unnormed (by mean and stddev) data for pred and targ
        - device: cuda device, default to last GPU, don't bother changing this probably
    """
    pred,targ = model_attrs.dataset[idx]
    pred = pred[np.newaxis,:] 
    pred_gpu = torch.from_numpy(pred).cuda(device)
    
    with torch.no_grad():
        model_output = model(pred_gpu.float())
        model_output = model_output.cpu().numpy()
    
    date = model_attrs.dataset.xr_datasets_pred[model_attrs.predictor_vars.index(pred_var)][idx].valid_time.data
    dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
    
    if len(model_attrs.dataset.datasets_pred_normed_means) > 1: #i.e. for normalization scheme == "per hour"
        hour_idx = np.where(np.array(model_attrs.dataset.hours) == dt_current.hour)[0][0] #overengineered but will work for non-"all" hours
    else: #i.e. for normalization scheme == "all times" so there's only one mean and stddev over the whole dataset (will also technically activate if "per hour" is done over just a single hour - but in that case this is still fine, and I'm not doing that as of 7/7
        hour_idx = 0
    
    if is_unnormed:
        pred = model_attrs.dataset.datasets_pred_normed_stddevs[model_attrs.predictor_vars.index(pred_var)][hour_idx]*pred[0,0,:] + model_attrs.dataset.datasets_pred_normed_means[model_attrs.predictor_vars.index(pred_var)][hour_idx]
        targ = model_attrs.dataset.datasets_targ_normed_stddevs[model_attrs.target_vars.index(targ_var)][hour_idx]*targ + model_attrs.dataset.datasets_targ_normed_means[model_attrs.target_vars.index(targ_var)][hour_idx]
        model_output = model_attrs.dataset.datasets_targ_normed_stddevs[model_attrs.target_vars.index(targ_var)][hour_idx]*model_output + model_attrs.dataset.datasets_targ_normed_means[model_attrs.target_vars.index(targ_var)][hour_idx]
    else:
        pred = pred[0,0,:]

    return pred, targ, model_output, dt_current

########################################################

