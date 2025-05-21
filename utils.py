import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchinfo import summary

import torch.nn.functional as F

import os
from netCDF4 import Dataset as nc_Dataset
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

####################################################################

def plot_prediction(X,y,pred):
    #X,y,pred = input data, truth, prediction respectively, as numpy arrays
    fig, axes = plt.subplots(1,4, figsize=(20,5))
    maxtemp = np.max([np.max(X.squeeze()), np.max(y.squeeze()), np.max(pred.squeeze())])
    mintemp = np.min([np.min(X.squeeze()), np.min(y.squeeze()), np.min(pred.squeeze())])
    #avg = (abs(maxtemp)+abs(mintemp))/2
    avg = (maxtemp-mintemp)/10
    
    axes[0].imshow(X.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[0].set_title(f"Predictor temp. (HRRR 2.5km)")
    axes[0].axis("off")
    axes[1].imshow(pred.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[1].set_title(f"Predicted temp.")
    axes[1].axis("off")
    axes[2].imshow(y.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[2].set_title(f"True temp. (URMA)")
    axes[2].axis("off")
    pos = axes[3].imshow((pred.squeeze() - y.squeeze()), cmap="coolwarm", vmin = -1*avg, vmax = avg) #Note we want the error centered around zero - may need to adjust the bounds.
    axes[3].set_title(f"Prediction - Truth")
    axes[3].axis("off")

    cbar = fig.colorbar(pos, ax=axes[3], fraction=0.03) #needs to be made more elegant
    cbar.set_label('Error')
    
    plt.suptitle(f"Maximum = {maxtemp:.4f} | Minimum = {mintemp:.4f}", va="bottom", fontsize=14)
    plt.tight_layout()
    plt.show()

####################################################################

def plot_epoch_losses(epoch_losses, loss_fcn_str):
    chop_off = 50
    window_len = 20
    x = np.arange(chop_off, len(epoch_losses),1)
    
    fig, axes = plt.subplots(1,1, figsize=(6,6))
    
    plt1 = plt.scatter(x,epoch_losses[chop_off:], marker='o', color='k', alpha=0.08, facecolors='none', label="Epoch loss")
    plt2 = plt.plot(x[window_len-1:], rolling_avg(epoch_losses[chop_off:], window_len), color='r', label=f"{window_len}-epoch average")

    plt.title('Average loss per epoch')
    plt.xlabel('Epoch number')
    plt.ylabel(f'{loss_fcn_str} loss')
    plt.legend()

    return

def rolling_avg(x, window_len):
    return np.convolve(x, np.ones(window_len), 'valid') / window_len