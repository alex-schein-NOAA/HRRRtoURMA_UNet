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

def plot_prediction(X, pred, y, date_str="DATE", savename="MODEL_NAME", to_save=False):
    #X,pred,y = input data, prediction, truth respectively, as numpy arrays
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
    axes[3].set_title(f"Prediction - Truth (RMSE = {np.sqrt(np.mean((pred.squeeze() - y.squeeze())**2)):.4f})")
    axes[3].axis("off")

    cbar = fig.colorbar(pos, ax=axes[3], fraction=0.03) #needs to be made more elegant
    cbar.set_label('Error')
    
    plt.suptitle(f"{savename} | Date = {date_str} | Maximum = {maxtemp:.1f} | Minimum = {mintemp:.1f}", va="bottom", fontsize=14)
    plt.tight_layout()

    if to_save:
        plt.savefig("temp.png",dpi=300, bbox_inches="tight")

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

####################################################################

def plot_prediction_2models(X, y, pred_1, pred_2, idx):
    fig, axes = plt.subplots(2,3, figsize=(15,10))
    maxtemp = np.max([np.max(X.squeeze()), np.max(y.squeeze()), np.max(pred_1.squeeze()), np.max(pred_2.squeeze())])
    mintemp = np.min([np.min(X.squeeze()), np.min(y.squeeze()), np.min(pred_1.squeeze()), np.min(pred_2.squeeze())])
    #avg = (abs(maxtemp)+abs(mintemp))/2
    avg = (maxtemp-mintemp)/10
    
    axes[0][0].imshow(X.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[0][0].set_title(f"Predictor temp. (HRRR 2.5km) (idx={idx})")
    axes[0][0].axis("off")
    axes[0][1].imshow(pred_1.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[0][1].set_title(f"Predicted, model 1")
    axes[0][1].axis("off")
    axes[0][2].imshow(pred_2.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[0][2].set_title(f"Predicted, model 2")
    axes[0][2].axis("off")
    
    axes[1][0].imshow(y.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[1][0].set_title(f"True temp. (URMA) (idx={idx})")
    axes[1][0].axis("off")
    axes[1][1].imshow((pred_1.squeeze()-y.squeeze()), cmap="coolwarm", vmin = -1*avg, vmax = avg)
    axes[1][1].set_title(f"Error, model 1 (mean = {np.mean((pred_1.squeeze()-y.squeeze())):.3f}) (RMSE={np.sqrt(np.mean(((pred_1.squeeze()-y.squeeze())**2))):.3f})")
    axes[1][1].axis("off")
    pos = axes[1][2].imshow((pred_2.squeeze() - y.squeeze()), cmap="coolwarm", vmin = -1*avg, vmax = avg) #Note we want the error centered around zero - may need to adjust the bounds.
    axes[1][2].set_title(f"Error, model 2 (mean = {np.mean((pred_2.squeeze()-y.squeeze())):.3f}) (RMSE={np.sqrt(np.mean(((pred_2.squeeze()-y.squeeze())**2))):.3f})")
    axes[1][2].axis("off")

    cbar = fig.colorbar(pos, ax=axes[1][2], fraction=0.03) #needs to be made more elegant
    cbar.set_label('Error')
    
    plt.suptitle(f"Model 1 = {pred_model_1_name} \n Model 2 = {pred_model_2_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

####################################################################

def plot_prediction_2models_unnormed(X, y, pred_1, pred_2, idx, train_ds_model_1, train_ds_model_2, pred_model_1_name, pred_model_2_name):
    fig, axes = plt.subplots(2,3, figsize=(15,10))

    X_unnormed = X.squeeze()*train_ds_model_1.hrrr_std + train_ds_model_1.hrrr_mean
    y_unnormed = y.squeeze()*train_ds_model_1.urma_std + train_ds_model_1.urma_mean
    pred_1_unnormed = pred_1.squeeze()*train_ds_model_1.urma_std + train_ds_model_1.urma_mean
    pred_2_unnormed = pred_2.squeeze()*train_ds_model_2.urma_std + train_ds_model_2.urma_mean
    
    maxtemp = np.max([np.max(X_unnormed), np.max(y_unnormed), np.max(pred_1_unnormed), np.max(pred_2_unnormed)])
    mintemp = np.min([np.min(X_unnormed), np.min(y_unnormed), np.min(pred_1_unnormed), np.min(pred_2_unnormed)])
    avg = (maxtemp-mintemp)/10 #Error range
    
    axes[0][0].imshow(X_unnormed, cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[0][0].set_title(f"Predictor temp. (HRRR 2.5km) (idx={idx})")
    axes[0][0].axis("off")
    axes[0][1].imshow(pred_1_unnormed, cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[0][1].set_title(f"Predicted, model 1")
    axes[0][1].axis("off")
    axes[0][2].imshow(pred_2_unnormed, cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[0][2].set_title(f"Predicted, model 2")
    axes[0][2].axis("off")
    
    axes[1][0].imshow(y_unnormed, cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[1][0].set_title(f"True temp. (URMA) (idx={idx})")
    axes[1][0].axis("off")
    axes[1][1].imshow(pred_1_unnormed-y_unnormed, cmap="coolwarm", vmin = -1*avg, vmax = avg)
    axes[1][1].set_title(f"Error, model 1 (mean = {np.mean(pred_1_unnormed-y_unnormed):.3f}) (RMSE={np.sqrt(np.mean(((pred_1_unnormed-y_unnormed)**2))):.3f})")
    axes[1][1].axis("off")
    pos = axes[1][2].imshow(pred_2_unnormed-y_unnormed, cmap="coolwarm", vmin = -1*avg, vmax = avg) #Note we want the error centered around zero - may need to adjust the bounds.
    axes[1][2].set_title(f"Error, model 2 (mean = {np.mean(pred_2_unnormed-y_unnormed):.3f}) (RMSE={np.sqrt(np.mean(((pred_2_unnormed-y_unnormed)**2))):.3f})")
    axes[1][2].axis("off")

    cbar = fig.colorbar(pos, ax=axes[1][2], fraction=0.025) #needs to be made more elegant
    cbar.set_label('Error (°C)')
    
    plt.suptitle(f"Model 1 = {pred_model_1_name} \n Model 2 = {pred_model_2_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

####################################################################

def plot_model1_model2_difference_unnormed(X, y, pred_1, pred_2, idx, train_ds_model_1, train_ds_model_2, pred_model_1_name, pred_model_2_name):
    fig, axes = plt.subplots(1,1, figsize=(5,5))

    X_unnormed = X.squeeze()*train_ds_model_1.hrrr_std + train_ds_model_1.hrrr_mean
    y_unnormed = y.squeeze()*train_ds_model_1.urma_std + train_ds_model_1.urma_mean
    pred_1_unnormed = pred_1.squeeze()*train_ds_model_1.urma_std + train_ds_model_1.urma_mean
    pred_2_unnormed = pred_2.squeeze()*train_ds_model_2.urma_std + train_ds_model_2.urma_mean    

    maxtemp = np.max([np.max(X_unnormed), np.max(y_unnormed), np.max(pred_1_unnormed), np.max(pred_2_unnormed)])
    mintemp = np.min([np.min(X_unnormed), np.min(y_unnormed), np.min(pred_1_unnormed), np.min(pred_2_unnormed)])
    avg = (maxtemp-mintemp)/50 #Error range

    pos = axes.imshow(pred_1_unnormed-pred_2_unnormed, cmap="coolwarm", vmin = -1*avg, vmax = avg) #Note we want the error centered around zero - may need to adjust the bounds.
    axes.set_title(f"Model 1 - Model 2 (idx={idx})")
    axes.axis("off")

    cbar = fig.colorbar(pos, ax=axes, fraction=0.025) #needs to be made more elegant
    cbar.set_label('Difference (°C)')
    
    plt.suptitle(f"Model 1 = {pred_model_1_name} \n Model 2 = {pred_model_2_name}", fontsize=14)
    plt.tight_layout()
    plt.show()