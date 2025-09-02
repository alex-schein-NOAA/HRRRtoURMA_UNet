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

from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HRRR_URMA_Datasets_AllVars import *

######################################################################################################################################################

class DefineModelAttributes():
    """
    Class to aid in constructing models and their Pytorch datasets for use in training. Must set an object of this class whose attributes can then be used to initialized and control model training.
    """
    def __init__(self,
                 BATCH_SIZE=24,
                 NUM_EPOCHS=20,
                 is_train=True,
                 months=[1,12],  
                 hours="all", 
                 forecast_lead_time=1, 
                 normalization_scheme="all times",
                 with_terrains=["diff"], 
                 with_yearly_time_sig=False, 
                 with_hourly_time_sig=False,
                 predictor_vars=["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"],
                 target_vars=["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"]
                ):
            ####
            # See HRRR_URMA_Datasets_AllVars.py for variable definitions/restrictions
            ####
        
        #########################################
    
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.is_train = is_train
        self.months = months
        self.hours = hours
        self.forecast_lead_time = forecast_lead_time
        self.normalization_scheme = normalization_scheme
        self.with_terrains = with_terrains
        self.with_yearly_time_sig = with_yearly_time_sig
        self.with_hourly_time_sig = with_hourly_time_sig
        self.predictor_vars = predictor_vars
        self.target_vars = target_vars

        self.create_save_name()

        self.dataset = None #call create_dataset() in whatever calling function needs it
        self.num_channels_in = None
        self.num_channels_out = None
        
    ######################################### FUNCTIONS #########################################
    
    def create_save_name(self):
        """
        Savename is additive, i.e. only includes attributes present in the model
        Ordering: 
            - Batch size (BS#)
            - Num epochs (NE#)
            - Months IF NOT 1-12
            - Hours IF NOT all
            - Forecast lead time IF NOT 1
            - Normalization scheme IF NOT "all times" (TO DO - though "all times" norm is so much better it should always be used...)
            - Terrains (tH, tU, tD)
            - Yearly/hourly time sigs (sY, sH)
            - Predictor variables, in parentheses, separated by dashes
                Example: pred(t2m-d2m-u10m)
            - Target variables, in parentheses, separated by dashes
                Example: targ(t2m-pressurf)
        """
        ## Define all optional/constructive arguments
        month_str = ""
        hours_str = ""
        forecast_str = ""
        terrain_str = ""
        time_sig_list = []
        if self.months != [1,12]:
            month_str = f"months{self.months[0]}-{self.months[1]}"
        if self.hours != "all":
            hours_str = f"hours{'-'.join([str(hour) for hour in self.hours])}"
        if self.forecast_lead_time != 1:
            forecast_str = f"f{str(forecast_lead_time).zfill(2)}"
        terrain_str = "_".join([f"t{x[0].capitalize()}" for x in self.with_terrains]) if self.with_terrains is not None else ""
        if self.with_yearly_time_sig:
            time_sig_list.append("sY")
        if self.with_hourly_time_sig:
            time_sig_list.append("sH")
        time_sig_str = "_".join(time_sig_list)

        optional_str_list = [month_str, hours_str, forecast_str, terrain_str, time_sig_str]
        optional_str = "_".join([x for x in optional_str_list if x != ""])
        
        # Doesn't play nice if defined within the f-string
        # Making as variables so these can be called independently for other plotting purposes (don't forget to add "pred" and "targ" in calling code!)
        self.pred_str = "-".join(self.predictor_vars)
        self.targ_str = "-".join(self.target_vars)

        savename_attrs_list = [f"BS{self.BATCH_SIZE}", f"NE{self.NUM_EPOCHS}", f"{optional_str}", f"pred({self.pred_str})", f"targ({self.targ_str})"]
        
        self.savename = "_".join([x for x in savename_attrs_list if x !=""])
                        # f"BS{self.BATCH_SIZE}" \
                        # f"_NE{self.NUM_EPOCHS}" \
                        # f"_{optional_str}" \
                        # f"_pred({self.pred_str})" \
                        # f"_targ({self.targ_str})"
        
        return 

    #########################################

    def create_dataset(self):
        # Creates the requisite Dataset for use in Pytorch Dataloader
        # NOT called by default (due to calculation expense) - must be invoked by calling function 

        # Check if this was already done, to save duplicate computation
        if self.dataset is None:
            print(f"Making dataset for model {self.savename}")
            self.dataset = HRRR_URMA_Dataset_AllVars(is_train = self.is_train,
                                                     months = self.months,  
                                                     hours = self.hours, 
                                                     forecast_lead_time = self.forecast_lead_time, 
                                                     normalization_scheme = self.normalization_scheme,
                                                     with_terrains = self.with_terrains, 
                                                     with_yearly_time_sig = self.with_yearly_time_sig, 
                                                     with_hourly_time_sig = self.with_hourly_time_sig,
                                                     predictor_vars = self.predictor_vars,
                                                     target_vars = self.target_vars)
            
            # Returns a list of dt.datetime objects of all dates in the current dataset. Useful for plotting and time series stuff
            self.dataset_date_list = [dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S") for date in self.dataset.xr_datasets_pred[0].valid_time.data]
        else:
            print(f"Dataset for the model {self.savename} was already computed. If it needs to be recomputed, set [current model].dataset=None and rerun .create_dataset()")
        
        self.num_channels_in = np.shape(self.dataset[0][0])[0]
        self.num_channels_out = np.shape(self.dataset[0][1])[0] #Don't forget to put this in SR_UNet_Simple model definition when targeting >1 var!!!
        return

    #########################################
    
    def set_model_attrs_from_savename(self, savename):
        #Input: str of model savename, formatted as in self.create_save_name()
        #output: all relevant fields
        self.with_terrains = []
        strs = savename.split("_")
        for string in strs:
            #(7/10) as currently written, will fail to detect some attrs for models with "months", "hours", "sY", "sH" in the name, but I'm not planning on dealing with such models for now, so fix this if needed
            if "BS" in string:
                self.BATCH_SIZE = "".join([char for char in string if char.isdigit()])
            elif "NE" in string:
                self.NUM_EPOCHS = "".join([char for char in string if char.isdigit()])
            elif string=="tH":
                self.with_terrains.append("hrrr")
            elif string=="tU":
                self.with_terrains.append("urma")
            elif string=="tD":
                self.with_terrains.append("diff")
            elif "pred(" in string:
                self.predictor_vars = ((string.split("(")[1])[:-1]).split("-")
            elif "targ(" in string:
                if ".pt" in string: #bad hack but w/e... this whole function is a bad hack
                    self.target_vars = ((string.split("(")[1])[:-4]).split("-")
                else:
                    self.target_vars = ((string.split("(")[1])[:-1]).split("-")
        
        self.create_save_name() #needs to be re-called to properly set attributes
        return