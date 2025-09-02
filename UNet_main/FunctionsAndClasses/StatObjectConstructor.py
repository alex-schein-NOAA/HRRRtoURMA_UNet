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

# import csv



from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.SR_UNet_simple import *
from FunctionsAndClasses.UNet_Attention_simple import *
from FunctionsAndClasses.utils import *
from FunctionsAndClasses.CONSTANTS import *

import csv

######################################################################################################################################################

class ConstructStatObject():
    """
    Class to aid in computing statistical quantities for trained models and Smartinit.
    """
    def __init__(self,
                 is_smartinit=False,
                 current_model_attrs=None,
                 predictor_var=None,
                 target_var=None
                ):
            
        """
        - Inputs:
            - is_smartinit = bool; if True, then calculate stats relative to Smartinit, otherwise use current_model_attrs
                - (AS OF 7/21): only have smartinit for 2024, so make sure current_model_attrs.is_train=False for that!!
            - current_model_attrs = already-initialized instance of DefineModelAttributes class
            - predictor_var = only used for get_model_output_at_idx call; needs to match at least one of the predictor variables in self.current_model_attrs.predictor_vars, but is otherwise not important
            - target_var = SINGLE STRING (not a list of multiple!) of the desired target variable whose quantities will be computed
                - Valid options (as of 8/21): "pressurf" | "t2m" | "d2m" | "spfh2m" | "u10m" | "v10m"
        """
        
        #########################################

        self.C = CONSTANTS()
        
        self.is_smartinit = is_smartinit
        self.current_model_attrs = current_model_attrs
        self.predictor_var = predictor_var if current_model_attrs==None else current_model_attrs.predictor_vars[0] 
        self.target_var = target_var

        self.smartinit_directory = self.C.DIR_SMARTINIT_DATA
        self.trained_models_directory = self.C.DIR_TRAINED_MODELS
        
        self.domain_avg_rmse_alltimes_list=None
        
        self.varname_units_dict = self.C.varname_units_dict
        self.varname_translation_dict = self.C.varname_translation_dict
        self.smartinit_var_select_dict = self.C.urma_var_select_dict #they share the same keys
        

    ######################################### FUNCTIONS #########################################

    def calc_domain_avg_RMSE_alltimes(self):
        """
        Ouput: list of domain-averaged RMSE for our current model, or smartinit
        """
        if self.domain_avg_rmse_alltimes_list is None: #Only compute if not already done
            self.domain_avg_rmse_alltimes_list = []

            xr_urma = xr.open_dataarray(f"{self.C.DIR_TRAIN_TEST}/test_urma_alltimes_{self.target_var}.grib2", decode_timedelta=True, engine='cfgrib')
            
            if self.is_smartinit:
                if not os.path.exists(f"{self.C.DIR_UNET_MAIN}/Smartinit_stats/smartinit_RMSE_alltimes_{self.target_var}.csv"): #Smartinit data doesn't exist on disk; calculate it and write to disk for future use
                    
                    # Experimentally determined offsets for best alignment between smartinit (NDFD grid) and extended URMA grid subregion (i.e. western study region)
                    # DOES NOT EXACTLY ALIGN - but pretty close...
                    # SHOULD PROBABLY BE ADDRESSED - TBD as of 2025-08-21
                    SMARTINIT_LON_OFFSET = 201
                    SMARTINIT_LAT_OFFSET = 1
                    
                    FORECAST_LEAD_HOURS = 1
                    START_DATE = dt.datetime(2024,1,1,0)-dt.timedelta(hours=FORECAST_LEAD_HOURS) 
                    END_DATE = dt.datetime(2024,12,31,23)-dt.timedelta(hours=FORECAST_LEAD_HOURS) 
                    NUM_HOURS = int((END_DATE-START_DATE).total_seconds()/3600) #surprisingly no built in hours function...
                    for i in range(NUM_HOURS+1):
                        xr_smartinit = get_smartinit_output_at_idx(i, 
                                                                    START_DATE, 
                                                                    FORECAST_LEAD_HOURS,
                                                                    self.C.DIR_SMARTINIT_DATA,
                                                                    self.smartinit_var_select_dict, 
                                                                    self.varname_translation_dict, 
                                                                    self.target_var, 
                                                                    IDX_MIN_LON=self.C.IDX_MIN_LON-SMARTINIT_LON_OFFSET,
                                                                    IDX_MIN_LAT=self.C.IDX_MIN_LAT-SMARTINIT_LAT_OFFSET,
                                                                    IMG_SIZE_LON=self.C.IMG_SIZE_LON,
                                                                    IMG_SIZE_LAT=self.C.IMG_SIZE_LAT)
                        
                        self.domain_avg_rmse_alltimes_list.append(self.calc_domain_avg_RMSE_onetime(xr_smartinit.data, xr_urma[i].data))
                        
                        if i%int(NUM_HOURS/100)==0:
                            print(f"{(i/NUM_HOURS)*100:.0f}% done (smartinit)")

                    # Write completed data so this doesn't have to be done again
                    with open(f"{self.C.DIR_UNET_MAIN}/Smartinit_stats/smartinit_RMSE_alltimes_{self.target_var}.csv", "w", newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(self.domain_avg_rmse_alltimes_list)
                    print(f"{self.target_var} csv written")

                else: #RMSE data already saved as csv - MUCH PREFERRED
                    print(f"{self.target_var} RMSE data for Smartinit exists on disk")
                    with open(f"{self.C.DIR_UNET_MAIN}/Smartinit_stats/smartinit_RMSE_alltimes_{self.target_var}.csv", 'r', newline='') as file:
                        reader = csv.reader(file)
                        self.domain_avg_rmse_alltimes_list = [np.float32(x) for x in (list(reader))[0]]
                    print(f"{self.target_var} RMSE data has been read in")
                        
    
            else: #doing model
                if self.current_model_attrs.dataset is None:
                    self.current_model_attrs.create_dataset()

                if "ATTENTION" in self.current_model_attrs.savename: #(2025-08-26) hack to get correct model - should be made more elegant with updates to current_model_attributes and decision on savename conventions
                    model = UNet_Attention_simple(n_channels_in=self.current_model_attrs.num_channels_in, n_channels_out=self.current_model_attrs.num_channels_out)
                else:
                    model = SR_UNet_simple(n_channels_in=self.current_model_attrs.num_channels_in, n_channels_out=self.current_model_attrs.num_channels_out)
                device = torch.device("cuda")
                model.to(device)
                model.load_state_dict(torch.load(f"{self.trained_models_directory}/{self.current_model_attrs.savename}.pt", weights_only=True))

                print(f"Calculating RMSE for all times ({self.target_var}, {self.current_model_attrs.savename})")
                for i, urma_arr in enumerate(xr_urma):
                    _, _, model_output, _ = get_model_output_at_idx(self.current_model_attrs, model, pred_var=self.predictor_var, targ_var=self.target_var, idx=i)
                    self.domain_avg_rmse_alltimes_list.append(self.calc_domain_avg_RMSE_onetime(model_output, urma_arr.data))
                    
                    if i%int(len(xr_urma)/20)==0:
                        print(f"{(i/len(xr_urma))*100:.0f}% done")
            
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

    