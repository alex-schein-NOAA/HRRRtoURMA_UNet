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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.markers import MarkerStyle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.SR_UNet_simple import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.RMSELoss import *

######################################################################################################################################################

def TrainOneModel(current_model, NUM_GPUS_TO_USE=3, loss_fcn="MAE"):
    """
    Fully trains one model, whose attributes have already been defined before being fed to this function
    Defaults to using the first 3 GPUs but this is tunable with input params
    
    Inputs:
        - current_model = DefineModelAttributes object whose parameters have already been defined. Don't need to create a dataset beforehand, this function does that
        - NUM_GPUS_TO_USE = int, 1 to 4, for # GPUs to use with DataParallel
        - loss_fcn = "MAE" or "RMSE" only currently
    """
    TRAINING_LOG_FILEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Training_logs/training_log.txt"
    TRAINED_MODEL_SAVEPATH = "/scratch/RTMA/alex.schein/CNN_Main/Trained_models"

    MULTIGPU_BATCH_SIZE = current_model.BATCH_SIZE*NUM_GPUS_TO_USE

    if not os.path.exists(f"/scratch/RTMA/alex.schein/CNN_Main/Trained_models/{current_model.savename}.pt"):
        appendation = ""
    else: #model under that name already exists
        print(f"A model under {current_model.savename} already exists in the save directory! Change that model's name, or otherwise ensure non overlapping names")
        appendation = "_nonoverlapping"
        print(f"Running with model name {current_model.savename}{appendation}")
    
    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Starting {current_model.savename} | Current time = {current_time} \n")
    
    current_model.create_dataset()
    current_model_dataloader = DataLoader(current_model.dataset, batch_size=MULTIGPU_BATCH_SIZE, shuffle=True, num_workers=4*NUM_GPUS_TO_USE, pin_memory=True)

    with open(TRAINING_LOG_FILEPATH, "a") as file:
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Data loaded | Current time = {current_time} \n")
    

    model = SR_UNet_simple(n_channels_in=current_model.num_channels_in, n_channels_out=current_model.num_channels_out)
    device = torch.device("cuda")
    model = nn.DataParallel(model, device_ids=[i for i in range(NUM_GPUS_TO_USE)])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=[0.5,0.999]) #torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    if loss_fcn == "MAE":    
        loss_function = torch.nn.L1Loss() #Might want to use L2 loss or RMSE... maybe revisit this
    elif loss_fcn == "RMSE":
        loss_function = RMSELoss()
    else:
        print(f"Must use ''MAE'' or ''RMSE'' for loss_fcn. Defaulting to MAE")
        loss_function = torch.nn.L1Loss()
    lowest_loss = 999

    model.train()
    
    for epoch in range(1,current_model.NUM_EPOCHS+1):
        epoch_loss = 0.0
        start = time.time()
        for i, (inputs,labels) in enumerate(current_model_dataloader):    
            
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = model(inputs.float()) #weird datatype mismatching... for some reason it's seeing HRRR data as double
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
            
        end2 = time.time() 
        
        if epoch_loss <= lowest_loss: #only save models that have lower loss than previous best
            lowest_loss = epoch_loss
            #Note need to use model.module when doing DataParallel, else saved model needs to be loaded on the same # of GPUs as training
            torch.save(model.module.state_dict(), f"{TRAINED_MODEL_SAVEPATH}/{current_model.savename}{appendation}_TEMP.pt")
        
        # write log at every interval so we can read back all the history if needed
        with open(TRAINING_LOG_FILEPATH, "a") as file: 
            file.write(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(current_model_dataloader):.5f} | Time for epoch = {end2-start:.1f} sec \n")
    
    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Training finished | Current time = {current_time} \n")
        
    os.rename(f"{TRAINED_MODEL_SAVEPATH}/{current_model.savename}{appendation}_TEMP.pt", 
              f"{TRAINED_MODEL_SAVEPATH}/{current_model.savename}{appendation}.pt") #so if training is interrupted, previously saved model under the same name isn't wiped out
    
    return

########################################################

def get_model_output_at_idx(model_attrs, model, pred_var="t2m", targ_var="t2m", idx=0, is_unnormed=True, device="cuda"):
    """
    Inputs:
        - model_attrs: DefineModelAttributes object. MUST HAVE .create_dataset() ALREADY CALLED! 
        - model: Pytorch model to use, with weights loaded and device initialized
        - pred_var: string for the predictor variable to get the output of
        - targ_var: string for the target variable to get the output of
        - idx: index to get the output of
        - is_unnormed: if true, returns the unnormed (by mean and stddev) data for pred and targ
        - device: cuda device, default to just "cuda". Might need to change this in calling function, be careful

    Outputs:
        - predictor @ index, UNNORMED if is_unnormed
        - target @ index, UNNORMED if is_unnormed
        - model output @ index, UNNORMED if is_unnormed
        - dt_current as string, for plot title purposes
    """
    pred,targ = model_attrs.dataset[idx]
    pred = pred[np.newaxis,:] 
    pred_gpu = torch.from_numpy(pred).cuda(device)
    
    with torch.no_grad():
        model_output = model(pred_gpu.float())
        model_output = model_output.cpu().numpy()
    
    date = model_attrs.dataset.xr_datasets_pred[model_attrs.predictor_vars.index(pred_var)][idx].valid_time.data
    dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S")
    
    if (model_attrs.normalization_scheme == "per hour"):
        hour_idx = np.where(np.array(model_attrs.dataset.hours) == dt_current.hour)[0][0] #overengineered but will work for non-"all" hours
    else: #i.e. for normalization scheme == "all times" so there's only one mean and stddev over the whole dataset (will also technically activate if "per hour" is done over just a single hour - but in that case this is still fine, and I'm not doing that as of 7/7
        hour_idx = 0
    
    if is_unnormed:
        pred = ( model_attrs.dataset.datasets_pred_normed_stddevs[model_attrs.predictor_vars.index(pred_var)][hour_idx]
                 *pred[0,model_attrs.predictor_vars.index(pred_var),:] 
                 + model_attrs.dataset.datasets_pred_normed_means[model_attrs.predictor_vars.index(pred_var)][hour_idx] )
        
        targ = ( model_attrs.dataset.datasets_targ_normed_stddevs[model_attrs.target_vars.index(targ_var)][hour_idx]
                 *targ[model_attrs.target_vars.index(targ_var),:] #unlike pred and model_output, this should only ever have a single channel, so no need to prepend with 0 - honestly, is prepending with a new axis even needed for "pred" in the code above? Should test...
                 + model_attrs.dataset.datasets_targ_normed_means[model_attrs.target_vars.index(targ_var)][hour_idx] )
        
        model_output = ( model_attrs.dataset.datasets_targ_normed_stddevs[model_attrs.target_vars.index(targ_var)][hour_idx]
                         *model_output[0,model_attrs.target_vars.index(targ_var),:] 
                         + model_attrs.dataset.datasets_targ_normed_means[model_attrs.target_vars.index(targ_var)][hour_idx] )
    else:
        pred = pred[0,model_attrs.predictor_vars.index(pred_var),:]

    return pred, targ, model_output, dt_current

########################################################

def plot_predictor_output_truth_error(X, pred, y, date_str="DATE", title="MODEL_NAME", to_save=False, fig_savename="temp"):
    """
    Plots the 1x4 figure for spatial display of [input | prediction | truth | output-truth], given a single hour's worth of data
    Inputs:
        - X = predictor input (i.e. 2.5 km HRRR for our purposes)
        - pred = model output, or whatever else (e.g. Smartinit field)
        - y = truth (i.e. URMA for our purposes)
        - date_str = string of format dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S") from get_model_output()
        - title = model name/params/whatever to identify that plot
        - to_save = bool for saving; if True, saves to directory this script is called from (currently this function is not intended for formalized plot saving)
        - fig_savename = string for file savename, if to_save = True
    """
    #X,pred,y = input data, prediction, truth respectively, as numpy arrays
    fig, axes = plt.subplots(1,4, figsize=(20,5))
    maxtemp = np.max([np.max(X.squeeze()), np.max(y.squeeze()), np.max(pred.squeeze())])
    mintemp = np.min([np.min(X.squeeze()), np.min(y.squeeze()), np.min(pred.squeeze())])

    avg = (maxtemp-mintemp)/10 #Denominator chosen arbitrarily; adjust if needed
    
    axes[0].imshow(X.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[0].set_title(f"Predictor (HRRR 2.5km)")
    axes[0].axis("off")
    axes[1].imshow(pred.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[1].set_title(f"Predicted")
    axes[1].axis("off")
    axes[2].imshow(y.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp)
    axes[2].set_title(f"Truth (URMA)")
    axes[2].axis("off")
    pos = axes[3].imshow((pred.squeeze() - y.squeeze()), cmap="coolwarm", vmin = -1*avg, vmax = avg) #Note we want the error centered around zero - may need to adjust the bounds.
    axes[3].set_title(f"Prediction - Truth (RMSE = {np.sqrt(np.mean((pred.squeeze() - y.squeeze())**2)):.4f})")
    axes[3].axis("off")

    cbar = fig.colorbar(pos, ax=axes[3], fraction=0.03) #needs to be made more elegant
    cbar.set_label('Error')
    
    plt.suptitle(f"{title} | Date = {date_str} | Maximum = {maxtemp:.1f} | Minimum = {mintemp:.1f}", va="bottom", fontsize=14)
    plt.tight_layout()

    if to_save:
        plt.savefig(f"{fig_savename}.png",dpi=300, bbox_inches="tight")

    plt.show()
    return

########################################################

def rolling_avg(x, window_len):
    return np.convolve(x, np.ones(window_len), 'valid') / window_len

########################################################

def plot_model_vs_smartinit_RMSE(model_attrs, statsobj_model, statsobj_smartinit, units_str="UNITS", to_save=False):
    """
    Inputs: 
        - model_attrs = instance of DefineModelAttributes class for the current model
        - statsobj_model = instance of StatObjectConstructor for the current model, with .CalcDomainAvgRMSEAllTimes() already done
        - statsobj_smartinit = same but for smartinit
        - units_str = string for the current variable's units.
            - t2m & d2m --> deg C, pressurf --> Pa, u10m & v10m --> m/s
        - to_save = bool to save fig or not
    """
    PLOT_SAVE_DIR = f"/scratch/RTMA/alex.schein/CNN_Main/Plots/f01/SmartinitComparisonPlots"
    
    rmse_diff = np.array(statsobj_smartinit.domain_avg_rmse_alltimes_list) - np.array(statsobj_model.domain_avg_rmse_alltimes_list)
    model_title = f"pred={model_attrs.pred_str}, targ={model_attrs.targ_str}"

    window_len = 24

    fig, axes = plt.subplots(figsize=(12,6))
    plt.plot(model_attrs.dataset_date_list, rmse_diff, 
             ".", linestyle='None', markersize=0.5, color='g', alpha=0.5, label="RMSE diff.")
    plt.plot(model_attrs.dataset_date_list[window_len-1:], rolling_avg(rmse_diff, window_len), 
             linewidth=1, color="g", label=f"RMSE diff., {window_len}-hr mean")
    plt.hlines(np.mean(rmse_diff), xmin=model_attrs.dataset_date_list[0], xmax=model_attrs.dataset_date_list[-1], 
               color="r", linewidth=2, label=f"RMSE diff. 2024 mean ({np.mean(rmse_diff):.3f})")
    plt.hlines(0, xmin=model_attrs.dataset_date_list[0], xmax=model_attrs.dataset_date_list[-1], linestyle='--', color="k", linewidth=2)
    plt.xlim([model_attrs.dataset_date_list[0], model_attrs.dataset_date_list[-1]])
    
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    for label in axes.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    plt.legend(loc="upper left")
    
    plt.ylabel(f"RMSE difference ({units_str})")
    plt.xlabel("Date")
    plt.title(f"{statsobj_smartinit.target_var} domain-average RMSE difference, Smartinit minus Model ({model_title}), 2024", fontsize=9)

    if to_save:
        fig_savename = f"RMSE_{statsobj_smartinit.target_var}_pred({model_attrs.pred_str})_targ({model_attrs.targ_str})"
        plt.savefig(f"{PLOT_SAVE_DIR}/{fig_savename}.png",dpi=300, bbox_inches="tight")
        print(f"{fig_savename} saved to {PLOT_SAVE_DIR}")
    else:
        plt.show()

    return