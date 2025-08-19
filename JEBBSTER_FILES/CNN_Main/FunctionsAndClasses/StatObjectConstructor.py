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

import csv

from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.SR_UNet_simple import *
from FunctionsAndClasses.utils import *

######################################################################################################################################################

class ConstructStatObject():
    """
    Class to aid in computing statistical quantities for trained models, and Smartinit.
    """
    def __init__(self,
                 is_smartinit=False,
                 current_model_attrs=None,
                 predictor_var=None,
                 target_var=None, 
                 model_loss="MAE"):
            
        """
        - Inputs:
            - is_smartinit = bool; if True, then calculate stats relative to Smartinit, otherwise use current_model_attrs
                - (AS OF 7/21): only have smartinit for 2024, so make sure current_model_attrs.is_train=False !!
            - current_model_attrs = already-initialized instance of DefineModelAttributes class
            - predictor_var = only used for get_model_output_at_idx call; needs to match at least one of the predictor variables in self.current_model_attrs.predictor_vars, but is otherwise not important
            - target_var = SINGLE STRING (not a list of multiple!) of the desired target variable whose quantities will be computed
                - Valid options (as of 7/21): "t2m" | "d2m" | "pressurf" | "u10m" | "v10m"
            - model_loss = string of "MAE" or "RMSE". Only used for model save directory select - calling function should make sure that RMSE is properly differentiated in plotting/filename saving!
        """
        
        #########################################

        self.is_smartinit = is_smartinit
        self.current_model_attrs = current_model_attrs
        self.predictor_var = predictor_var if current_model_attrs==None else current_model_attrs.predictor_vars[0] 
        self.target_var = target_var
        self.model_loss = model_loss

        self.smartinit_directory = f"/data1/projects/RTMA/alex.schein/HRRR_Smartinit_Data"

        if self.model_loss=="MAE":
            self.trained_models_directory = f"/scratch/RTMA/alex.schein/CNN_Main/Trained_models/MAE_Loss"
        elif self.model_loss=="RMSE":
            self.trained_models_directory = f"/scratch/RTMA/alex.schein/CNN_Main/Trained_models/MAE_Loss"
        else:
            print(f"''model_loss'' must be set to ''MAE'' or ''RMSE'' !")
            self.trained_models_directory=None
        
        self.domain_avg_rmse_alltimes_list=None
        
        self.varname_units_dict = {"t2m":"deg C",
                                    "d2m":"deg C",
                                    "pressurf":"Pa",
                                    "u10m":"m/s",
                                    "v10m":"m/s"}
        
        
        self.varname_translation_dict = {"t2m":"t2m",
                                        "d2m":"d2m",
                                        "pressurf":"sp",
                                        "u10m":"u10",
                                        "v10m":"v10"}
        
        self.smartinit_var_select_dict = {"t2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                                          "d2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                                          "sp":{'filter_by_keys':{'typeOfLevel': 'surface'}},
                                          "u10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}},
                                          "v10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}}}
        



    ######################################### FUNCTIONS #########################################

    def calc_domain_avg_RMSE_alltimes(self):
        """
        Ouput: list of domain-averaged RMSE for our current model, or smartinit
        """
        if self.domain_avg_rmse_alltimes_list is None: #Only compute if not already done
            self.domain_avg_rmse_alltimes_list = []


            
            # xr_urma = xr.open_dataarray(f"/data1/projects/RTMA/alex.schein/URMA_train_test/test_urma_alltimes_{self.target_var}.nc", decode_timedelta=True)
            xr_urma = xr.open_dataarray(f"/data1/projects/RTMA/alex.schein/URMA_train_test/OLD_DATASETS/master_netcdfs_co_domain/test_urma_alltimes_{self.target_var}.nc", decode_timedelta=True)
            
            if self.is_smartinit:
                if not os.path.exists(f"/scratch/RTMA/alex.schein/CNN_Main/Smartinit_stats/smartinit_RMSE_alltimes_{self.target_var}.csv"): 
                    # IDX_MIN_LON=596
                    # IDX_MIN_LAT=645
                    # IMG_SIZE_LON=180
                    # IMG_SIZE_LAT=180
                    FORECAST_LEAD_HOURS = 1
                    START_DATE = dt.datetime(2024,1,1,0)-dt.timedelta(hours=FORECAST_LEAD_HOURS) 
                    END_DATE = dt.datetime(2024,12,31,23)-dt.timedelta(hours=FORECAST_LEAD_HOURS) 
                    NUM_HOURS = int((END_DATE-START_DATE).total_seconds()/3600) #surprisingly no built in hours function...
                    for i in range(NUM_HOURS+1):
                        # DATE_STR = dt.date.strftime(START_DATE + dt.timedelta(hours=i), "%Y%m%d")
                        # file_to_open = f"{self.smartinit_directory}/hrrr_smartinit_{DATE_STR}_t{str((START_DATE.hour+i)%24).zfill(2)}z_f{str(FORECAST_LEAD_HOURS).zfill(2)}.grib2"
                        # xr_smartinit = xr.open_dataset(file_to_open,
                        #                                engine="cfgrib", 
                        #                                backend_kwargs=self.smartinit_var_select_dict[self.varname_translation_dict[self.target_var]],
                        #                                decode_timedelta=True)
                        # xr_smartinit = xr_smartinit[self.varname_translation_dict[self.target_var]]
                        # xr_smartinit = xr_smartinit.isel(y=slice(IDX_MIN_LAT, IDX_MIN_LAT+IMG_SIZE_LAT),
                        #                                  x=slice(IDX_MIN_LON, IDX_MIN_LON+IMG_SIZE_LON))
                        xr_smartinit = get_smartinit_output_at_idx(i, 
                                                                    START_DATE, 
                                                                    FORECAST_LEAD_HOURS, 
                                                                    self.smartinit_var_select_dict, 
                                                                    self.varname_translation_dict, 
                                                                    self.target_var)
                        
                        self.domain_avg_rmse_alltimes_list.append(self.calc_domain_avg_RMSE_onetime(xr_smartinit.data, xr_urma[i].data))
    
                        if i%int(NUM_HOURS/50)==0:
                            print(f"{(i/NUM_HOURS)*100:.0f}% done (smartinit)")

                else: #RMSE data already saved as csv - MUCH PREFERRED
                    print(f"{self.target_var} RMSE data for Smartinit exists on disk")
                    with open(f"/scratch/RTMA/alex.schein/CNN_Main/Smartinit_stats/smartinit_RMSE_alltimes_{self.target_var}.csv", 'r', newline='') as file:
                        reader = csv.reader(file)
                        self.domain_avg_rmse_alltimes_list = [np.float32(x) for x in (list(reader))[0]]
                    print(f"{self.target_var} RMSE data has been read in")
                        
    
            else: #doing model
                if self.current_model_attrs.dataset is None:
                    self.current_model_attrs.create_dataset()
                    
                model = SR_UNet_simple(n_channels_in=self.current_model_attrs.num_channels_in, n_channels_out=self.current_model_attrs.num_channels_out)
                device = torch.device("cuda")
                model.to(device)
                model.load_state_dict(torch.load(f"{self.trained_models_directory}/{self.current_model_attrs.savename}.pt", weights_only=True))

                print(f"Calculating RMSE for all times ({self.target_var}, {self.current_model_attrs.savename})")
                for i, urma_arr in enumerate(xr_urma.data):
                    _, _, model_output, _ = get_model_output_at_idx(self.current_model_attrs, model, pred_var=self.predictor_var, targ_var=self.target_var, idx=i)
                    self.domain_avg_rmse_alltimes_list.append(self.calc_domain_avg_RMSE_onetime(model_output, urma_arr))
                    
                    if i%int(np.shape(xr_urma.data)[0]/20)==0:
                        print(f"{(i/np.shape(xr_urma.data)[0])*100:.0f}% done")
            
        return

    #########################################

    def calc_domain_avg_RMSE_onetime(self, pred_arr, truth_arr):
        """
        Inputs: 
            - pred_arr = input array (must be numpy array or castable as such) of input data (i.e. model output [UNNORMALIZED] or smartinit) over the domain for one variable at one time
            - truth_arr = input array of truth (i.e. URMA) over the domain for the same variable at the same time
        """
        return np.sqrt(np.mean((pred_arr-truth_arr)**2))

    #########################################

    