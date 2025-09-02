# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# import os
# import time
# import datetime as dt
# from netCDF4 import Dataset as nc_Dataset
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
# from FunctionsAndClasses.UNet_Attention_simple import *
# from FunctionsAndClasses.DefineModelAttributes import *
# from FunctionsAndClasses.RMSELoss import *
# from FunctionsAndClasses.CONSTANTS import *

from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_plotting import *
from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.SR_UNet_simple import *
from FunctionsAndClasses.UNet_Attention_simple import *
from FunctionsAndClasses.UNet_Residual import *
from FunctionsAndClasses.UNet_Residual_NoResidual import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.StatObjectConstructor import *
from FunctionsAndClasses.CONSTANTS import *


######################################################################################################################################################

def TrainOneModel(current_model, 
                  is_attention_model=False,
                  is_residual_model=False,
                  use_residual_block=False,
                  NUM_GPUS_TO_USE=2, 
                  TRAINING_LOG_FILEPATH = None,
                  TRAINED_MODEL_SAVEPATH = None
                 ):
    """
    Fully trains one model, whose attributes have already been defined before being fed to this function
    Defaults to using 2 GPUs (default per interactive node) but this is tunable with input params
    
    Inputs:
        - current_model = DefineModelAttributes object whose parameters have already been defined. Don't need to create a dataset beforehand, this function does that
        - (ADDED 2025-08-25) is_attention_model = bool to select if training is for an attention model from UNet_Attention_simple class, or not (SR_UNet_simple)
        - (ADDED 2025-09-01) is_residual_model = bool to select if training is for a residual model from UNet_Residual class
            - Now that there's 2 modifications, need to modify the savename deal - honestly, should really be doing this in DefineModelAttributes since these are model attributes!
        - (ADDED 2025-09-01) use_residual_block = bool to determine if the residual block is actually used in the residual model - this is a temporary feature, needed for training a model with the more complex ConvBlock in UNet_Residual, but without the actual residual feature, for comparison purposes
        - NUM_GPUS_TO_USE = int for # GPUs to use with DataParallel and num_workers
        - TRAINING_LOG_FILEPATH = filepath to save training log to, including file name - should generally not be changed unless training multiple models simultaneously
        - TRAINED_MODEL_SAVEPATH = filepath to save trained models to - might need to differ if doing different losses, num epochs, etc
    """
    
    MULTIGPU_BATCH_SIZE = current_model.BATCH_SIZE*NUM_GPUS_TO_USE

    if not os.path.exists(f"{TRAINED_MODEL_SAVEPATH}/{current_model.savename}.pt"):
        appendation = ""
    else: #model under that name already exists
        print(f"A model under {current_model.savename} already exists in the save directory! Change that model's name, or otherwise ensure non overlapping names")
        appendation = "_nonoverlapping"
        print(f"Running script with model name {current_model.savename}{appendation}")
    
    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Starting {current_model.savename} | Current time = {current_time} \n")
    
    current_model.create_dataset()
    current_model_dataloader = DataLoader(current_model.dataset, 
                                          batch_size=MULTIGPU_BATCH_SIZE, 
                                          shuffle=True, 
                                          num_workers=4*NUM_GPUS_TO_USE, 
                                          pin_memory=True, 
                                          persistent_workers=True) #persistent_workers added 2025-08-25

    with open(TRAINING_LOG_FILEPATH, "a") as file:
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Data loaded | Current time = {current_time} \n")
    
    
    if is_attention_model:
        current_model.savename = f"ATTENTION_SIMPLE_{current_model.savename}" #(2025-08-25) Need to differentiate these. Should change this to a more elegant solution later (involving DefineModelAttributes)
        model = UNet_Attention_simple(n_channels_in=current_model.num_channels_in, 
                                      n_channels_out=current_model.num_channels_out)
    elif is_residual_model: #(2025-09-01) This needs to be cleaned up, like the rest of this mess....
        if use_residual_block:
            current_model.savename = f"RESIDUAL_{current_model.savename}"
            model = UNet_Residual(n_channels_in=current_model.num_channels_in,
                                  n_channels_out=current_model.num_channels_out) #, is_residual=use_residual_block)
        else:
            current_model.savename = f"RESIDUAL_NORESIDUAL_{current_model.savename}"
            model = UNet_Residual_NoResidual(n_channels_in = current_model.num_channels_in,
                                             n_channels_out = current_model.num_channels_out)
    else:
        model = SR_UNet_simple(n_channels_in=current_model.num_channels_in, 
                               n_channels_out=current_model.num_channels_out)
    
    device = torch.device("cuda")
    model = nn.DataParallel(model, device_ids=[i for i in range(NUM_GPUS_TO_USE)])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=[0.5,0.999])

    loss_function = torch.nn.L1Loss() #Might want to use L2 loss or RMSE... maybe revisit this
    
    model.train()
    
    lowest_loss = 999
    
    for epoch in range(1,current_model.NUM_EPOCHS+1):
        epoch_loss = 0.0
        start = time.time()
        for i, (inputs,labels) in enumerate(current_model_dataloader):    
            start_batch = time.time()            
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = model(inputs.float()) #weird datatype mismatching... for some reason it's seeing HRRR data as double
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
            
            if (i+1)%int(len(current_model_dataloader)/100)==0:
                #with open(TRAINING_LOG_FILEPATH, "a") as file:
                    #file.write
                print(f"Done with batch {i+1}/{len(current_model_dataloader)}. Time taken = {time.time()-start_batch:.1f} sec. Running loss = {epoch_loss/(i+1):.5f}") # \n")
            
        end = time.time() 
        
        if epoch_loss <= lowest_loss: #only save models that have lower loss than previous best
            lowest_loss = epoch_loss
            #Note need to use model.module when doing DataParallel, else saved model needs to be loaded on the same # of GPUs as training
            torch.save(model.module.state_dict(), f"{TRAINED_MODEL_SAVEPATH}/{current_model.savename}{appendation}_TEMP.pt")
        
        # write log at every interval so we can read back all the history if needed
        with open(TRAINING_LOG_FILEPATH, "a") as file: 
            file.write(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(current_model_dataloader):.5f} | Time for epoch = {end-start:.1f} sec \n")
    
    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Training finished | Current time = {current_time} \n")
        
    os.rename(f"{TRAINED_MODEL_SAVEPATH}/{current_model.savename}{appendation}_TEMP.pt", 
              f"{TRAINED_MODEL_SAVEPATH}/{current_model.savename}{appendation}.pt") #so if training is interrupted, previously saved model under the same name isn't wiped out
    
    return

########################################################

def get_model_output_at_idx(model_attrs, 
                            model, 
                            pred_var="t2m", 
                            targ_var="t2m", 
                            idx=0, 
                            is_unnormed=True, 
                            device="cuda"
                           ):
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
        - dt_current as dt.datetime object, for plot title purposes
    """
    
    pred,targ = model_attrs.dataset[idx]
    pred = pred[np.newaxis,:] 
    pred_gpu = torch.from_numpy(pred).cuda(device)
    
    with torch.no_grad():
        model_output = model(pred_gpu.float())
        model_output = model_output.cpu().numpy()
    
    date = model_attrs.dataset.xr_datasets_pred[model_attrs.predictor_vars.index(pred_var)][idx].valid_time.data
    dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='m')), "%Y-%m-%dT%H:%M")
    
    # (2025-08-22) Removed "per hour" code as it was causing problems and is not needed
    
    if is_unnormed:
        
        #(2025-08-27) updating method to call directly from xr_datasets, which contain the unnormed data. But model_output is still normalized data, so that gets unnormalized

        pred = model_attrs.dataset.xr_datasets_pred[model_attrs.predictor_vars.index(pred_var)][idx].data
        targ = model_attrs.dataset.xr_datasets_targ[model_attrs.target_vars.index(targ_var)][idx].data

        model_output = ( model_attrs.dataset.datasets_targ_normed_stddevs[model_attrs.target_vars.index(targ_var)]
                         *model_output[0,model_attrs.target_vars.index(targ_var),:] 
                         + model_attrs.dataset.datasets_targ_normed_means[model_attrs.target_vars.index(targ_var)] )
        
        # pred = ( model_attrs.dataset.datasets_pred_normed_stddevs[model_attrs.predictor_vars.index(pred_var)]
        #          *pred[0,model_attrs.predictor_vars.index(pred_var),:] 
        #          + model_attrs.dataset.datasets_pred_normed_means[model_attrs.predictor_vars.index(pred_var)] )
        
        # targ = ( model_attrs.dataset.datasets_targ_normed_stddevs[model_attrs.target_vars.index(targ_var)]
        #          *targ[model_attrs.target_vars.index(targ_var),:] #unlike pred and model_output, this should only ever have a single channel, so no need to prepend with 0
        #          + model_attrs.dataset.datasets_targ_normed_means[model_attrs.target_vars.index(targ_var)] )
        
        ### (8/1) Normalization based on 7/31 tag-up meeting: try normalizing w.r.t mean/stddev of predictor, not target
        # Hurts model performance vs Smartinit!
        # Models are trained to emulate URMA-normalized data so that mean/stddev should apply to model output (i.e. models transform from HRRR-normalized space to URMA-normalized space
        # In practice, computing climatological mean and stddev of URMA is not hard and model output can be denormalized w.r.t these quantities without much trouble
        
    
    else:
        pred = pred[0,model_attrs.predictor_vars.index(pred_var),:]
        targ = targ[model_attrs.target_vars.index(targ_var),:]

    return pred, targ, model_output, dt_current

########################################################

def get_smartinit_output_at_idx(i, 
                                FORECAST_LEAD_HOURS, 
                                smartinit_directory,
                                smartinit_var_select_dict, 
                                varname_translation_dict, 
                                target_var, 
                                IDX_MIN_LON=250-201,
                                IDX_MIN_LAT=400-1,
                                IMG_SIZE_LON=800,
                                IMG_SIZE_LAT=800,
                                START_DATE=None
                               ):
    """
    Method to open one Smartinit file and return its output for one variable, restricted to whatever spatial domain we define.
    Designed for StatObjectConstructor but can be called from anywhere else that Smartinit output is needed.

    Inputs:
        - i = int of index to select. Should line up with sample_idx indexing from HRRR
        - FORECAST_LEAD_HOURS = int of forecast lead time. Should already have offset START_DATE as detailed previously
        - smartinit_directory = directory of smartinit data which is NOT subset in any way but is named according to the convention in the code below
        - smartinit_var_select_dict = as in StatObjectConstructor
        - varname_translation_dict = as in StatObjectConstructor
        - target_var = string of a valid target variable, e.g. "t2m"
        - IDXs and IMG_SIZEs = define region as usual. Should be the same as whatever model domain being compared against. Default = Western domain (as of 2025-08-26)
            - Problem introduced by use of URMA extended grid for regridded HRRR and URMA: the indices no longer align between that grid and Smartinit's!
            - Even worse, the GRIDS THEMSELVES don't exactly align! 
                - Experimentally determined offset indices for lon(=201) and lat (=1) are used to correct for this, but the grids are still off
            - SHOULD FIX
        - START_DATE = dt.datetime object. Currently (as of 7/29) only have Smartinit data for 2024, so this should be 2023/12/31 23z or later. 
            - CURRENTLY SHOULD NOT BE CHANGED FROM THE DEFAULT! 
            - !!! VERY IMPORTANT: calling function should have START_DATE = dt.datetime([20240101_00z or greater])-dt.timedelta(hours=FORECAST_LEAD_HOURS) !!!

    Outputs:
        - xarray object of the smartinit, sliced down to the variable and domain of interest.
            - Returns xarray object, not just data, so calling function should invoke .data if that's what's desired
    """

    if START_DATE is None: #This should be the default, as this method is written to count hours from the first available Smartinit date, which is (as of 2025-08-26) 2024-01-01 00z
        START_DATE = dt.datetime(2024,1,1,0)-dt.timedelta(hours=FORECAST_LEAD_HOURS)
    
    DATE_STR = dt.date.strftime(START_DATE + dt.timedelta(hours=i), "%Y%m%d")
    file_to_open = f"{smartinit_directory}/hrrr_smartinit_{DATE_STR}_t{str((START_DATE.hour+i)%24).zfill(2)}z_f{str(FORECAST_LEAD_HOURS).zfill(2)}.grib2"
    xr_smartinit = xr.open_dataset(file_to_open,
                                   engine="cfgrib", 
                                   backend_kwargs=smartinit_var_select_dict[varname_translation_dict[target_var]],
                                   decode_timedelta=True)
    xr_smartinit = xr_smartinit[varname_translation_dict[target_var]]
    xr_smartinit = xr_smartinit.isel(y=slice(IDX_MIN_LAT, IDX_MIN_LAT+IMG_SIZE_LAT),
                                     x=slice(IDX_MIN_LON, IDX_MIN_LON+IMG_SIZE_LON))
    
    return xr_smartinit

    


########################################################

def plot_predictor_output_truth_error(X, 
                                      pred, 
                                      y, 
                                      date_str="DATE", 
                                      title="MODEL_NAME", 
                                      to_save=False, 
                                      save_dir=f"/scratch3/BMC/wrfruc/aschein/UNet_main", 
                                      fig_savename="temp.png", 
                                      error_units="", 
                                      avg_denom=10
                                     ):
    """
    Plots the 1x4 figure for spatial display of [input | prediction | truth | output-truth], given a single hour's worth of data
    Inputs:
        - X = predictor input (i.e. 2.5 km HRRR for our purposes)
        - pred = model output, or whatever else (e.g. Smartinit field)
        - y = truth (i.e. URMA for our purposes)
        - date_str = string of format dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S") from get_model_output()
        - title = model name/params/whatever to identify that plot
        - to_save = bool for saving; if True, saves to directory this script is called from (currently this function is not intended for formalized plot saving)
        - save_dir = master save directory
        - fig_savename = string for file savename, if to_save = True. Should include .png
        - error_units = string (including parentheses) for error units, e.g. "(deg C)"
        - avg_denom = how much to scale error plot by. Should be ~10 normally, but for pressurf, should be ~150
    """

    #X,pred,y = input data, prediction, truth respectively, as numpy arrays
    fig, axes = plt.subplots(1,4, figsize=(20,5))
    maxtemp = np.max([np.max(X.squeeze()), np.max(y.squeeze()), np.max(pred.squeeze())])
    mintemp = np.min([np.min(X.squeeze()), np.min(y.squeeze()), np.min(pred.squeeze())])

    avg = (maxtemp-mintemp)/avg_denom #Denominator chosen arbitrarily; adjust if needed
    
    axes[0].imshow(X.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
    axes[0].set_title(f"Predictor (HRRR 2.5km)")
    axes[0].axis("off")
    axes[1].imshow(pred.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
    axes[1].set_title(f"Predicted")
    axes[1].axis("off")
    axes[2].imshow(y.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
    axes[2].set_title(f"Truth (URMA)")
    axes[2].axis("off")
    pos = axes[3].imshow((pred.squeeze() - y.squeeze()), cmap="coolwarm", vmin = -1*avg, vmax = avg, origin='lower') #Note we want the error centered around zero - may need to adjust the bounds.
    axes[3].set_title(f"Prediction - Truth (RMSE = {np.sqrt(np.mean((pred.squeeze() - y.squeeze())**2)):.3f})")
    axes[3].axis("off")

    cbar = fig.colorbar(pos, ax=axes[3], fraction=0.03) #needs to be made more elegant
    cbar.set_label(f'Error {error_units}')
    
    plt.suptitle(f"{title} | Date = {date_str} | Maximum = {maxtemp:.1f} | Minimum = {mintemp:.1f}", va="bottom", fontsize=14)
    plt.tight_layout()

    if to_save:
        plt.savefig(f"{save_dir}/{fig_savename}",dpi=300, bbox_inches="tight")

    plt.show()
    return

########################################################

def rolling_avg(x, window_len):
    return np.convolve(x, np.ones(window_len), 'valid') / window_len

########################################################

def plot_model_vs_smartinit_RMSE(model_attrs, 
                                 statsobj_model, 
                                 statsobj_smartinit, 
                                 units_str="UNITS", 
                                 to_save=False, 
                                 PLOT_SAVE_DIR = f"/scratch3/BMC/wrfruc/aschein/UNet_main/Plots"
                                ):
    """
    Inputs: 
        - model_attrs = instance of DefineModelAttributes class for the current model
        - statsobj_model = instance of StatObjectConstructor for the current model, with .calc_domain_avg_RMSE_alltimes() already done
        - statsobj_smartinit = same but for smartinit
        - units_str = string for the current variable's units.
            - pressurf --> Pa, t2m & d2m --> deg C, spfh2m --> kg/kg, u10m & v10m --> m/s
        - to_save = bool to save fig or not
        - PLOT_SAVE_DIR = full directory path of where to save plots if to_save=True. SHOULD BE CHANGED FROM DEFAULT BY CALLING FUNCTION!
    """
    
    rmse_diff = np.array(statsobj_smartinit.domain_avg_rmse_alltimes_list) - np.array(statsobj_model.domain_avg_rmse_alltimes_list)

    window_len = 24

    fig, axes = plt.subplots(figsize=(12,6))
    plt.plot(model_attrs.dataset_date_list, rmse_diff, 
             ".", linestyle='None', markersize=0.5, color='g', alpha=0.5, label="RMSE diff.")
    plt.plot(model_attrs.dataset_date_list[window_len-1:], rolling_avg(rmse_diff, window_len), 
             linewidth=1, color="g", label=f"RMSE diff., rolling {window_len}-hr mean")
    plt.hlines(np.mean(rmse_diff), xmin=model_attrs.dataset_date_list[0], xmax=model_attrs.dataset_date_list[-1], 
               color="r", linewidth=2, label=f"RMSE diff. 2024 mean ({np.mean(rmse_diff):.3f})")
    plt.hlines(0, xmin=model_attrs.dataset_date_list[0], xmax=model_attrs.dataset_date_list[-1], linestyle='--', color="k", linewidth=2)
    plt.xlim([model_attrs.dataset_date_list[0], model_attrs.dataset_date_list[-1]])
    
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    for label in axes.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    plt.legend(loc="upper left")
    
    plt.ylabel(f"RMSE improvement ({units_str})")
    plt.xlabel("Date")
    plt.title(f"{statsobj_smartinit.target_var} domain-average RMSE difference, Smartinit minus Model, 2024 \n \
                Model = {model_attrs.savename}", fontsize=9)

    if to_save:
        fig_savename = f"RMSE_{statsobj_smartinit.target_var}_model({model_attrs.savename})"
        plt.savefig(f"{PLOT_SAVE_DIR}/{fig_savename}.png",dpi=300, bbox_inches="tight")
        print(f"{fig_savename} saved to {PLOT_SAVE_DIR}")
    else:
        plt.show()

    return

########################################################

def plot_model_vs_model_RMSE(model_1_attrs, 
                             model_2_attrs, 
                             statsobj_model_1, 
                             statsobj_model_2, 
                             TARG_VAR, 
                             units_str="UNITS", 
                             to_save=False, 
                             PLOT_SAVE_DIR = f"/scratch3/BMC/wrfruc/aschein/UNet_main/Plots"
                            ):
    """
    Inputs: 
        - model_1_attrs = instance of DefineModelAttributes class for the first (baseline) model
        - model_2_attrs =  instance of DefineModelAttributes class for the second (comparison) model
        - statsobj_model_1 = instance of StatObjectConstructor for the first model, with .calc_domain_avg_RMSE_alltimes() already done
        - statsobj_model_2 = same but for second model
        - TARG_VAR = string of current target variable
        - units_str = string for the current variable's units.
            - t2m & d2m --> deg C, pressurf --> Pa, u10m & v10m --> m/s
        - to_save = bool to save fig or not
        - PLOT_SAVE_DIR = full directory path of where to save plots if to_save=True. SHOULD BE CHANGED FROM DEFAULT BY CALLING FUNCTION!
    """

    rmse_diff = np.array(statsobj_model_1.domain_avg_rmse_alltimes_list) - np.array(statsobj_model_2.domain_avg_rmse_alltimes_list)

    window_len = 24

    fig, axes = plt.subplots(figsize=(12,6))
    plt.plot(model_1_attrs.dataset_date_list, rmse_diff, 
             ".", linestyle='None', markersize=0.5, color='b', alpha=0.5, label="RMSE diff.")
    plt.plot(model_1_attrs.dataset_date_list[window_len-1:], rolling_avg(rmse_diff, window_len), 
             linewidth=1, color="b", label=f"RMSE diff., rolling {window_len}-hr mean")
    plt.hlines(np.mean(rmse_diff), xmin=model_1_attrs.dataset_date_list[0], xmax=model_1_attrs.dataset_date_list[-1], 
               color="r", linewidth=2, label=f"RMSE diff. 2024 mean ({np.mean(rmse_diff):.3f})")
    plt.hlines(0, xmin=model_1_attrs.dataset_date_list[0], xmax=model_1_attrs.dataset_date_list[-1], linestyle='--', color="k", linewidth=2)
    plt.xlim([model_1_attrs.dataset_date_list[0], model_1_attrs.dataset_date_list[-1]])
    
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    for label in axes.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    plt.legend(loc="upper left")
    
    plt.ylabel(f"RMSE improvement ({units_str})")
    plt.xlabel("Date")
    plt.title(f"{TARG_VAR} domain-average RMSE difference, Model 1 minus Model 2, 2024 \n \
                Model 1 = {model_1_attrs.savename} \n \
                Model 2 = {model_2_attrs.savename}", fontsize=9)

    if to_save:
        fig_savename = f"RMSE_{TARG_VAR}_model1({model_1_attrs.savename})_model2({model_2_attrs.savename})"
        plt.savefig(f"{PLOT_SAVE_DIR}/{fig_savename}.png",dpi=300, bbox_inches="tight")
        print(f"{fig_savename} saved to {PLOT_SAVE_DIR}")
    else:
        plt.show()

    return
