# import os
# import time
# import datetime as dt
# from netCDF4 import Dataset as nc_Dataset
# import pandas as pd
# import numpy as np
# import xarray as xr


#################

class CONSTANTS():
    def __init__(self):

        """
        Stores constants/variables that are frequently used across different files and classes.
        CONTENTS:
        - varname_translation_dict: converts from my naming conventions for the variables, to how they're actually named in the .grib2 files. Should be used in xarray variable selection, e.g. "xr_dataset[varname_translation_dict["pressurf"]]"
        - urma_selection_dict: to be used in xarray call as backend_kwargs for loading an URMA file with multiple levels (e.g. the data in Regridded_URMA). Note the keys are the actual var names, so should go through varname_translation_dict, e.g. "backend_kwargs=urma_selection_dict[varname_translation_dict["pressurf"]]"
        - Means and stddevs for variables over the larger western domain, as they're non-trivial to compute. Should be used for normalization in the Dataset class, and for de-normalization in plotting funcs (though can just set the means and stddevs in the Dataset class and existing code will work). NOTE: mean and stddev taken over all hours in the dataset - any subset will require recomputation, but I'm not planning on that at the moment 
            - Structured as dictionaries, split on HRRR/URMA and again on means/stddevs. Each dict has 2 levels: first select 'train' or 'test' as needed, then '[var_name]' to get the quantity.
        
        !!!! (8/11) MEANS ARE BEING CALCULATED WITH LOOP OVER ALL TIMES, TAKING MEAN @ EACH TIME, APPENDING TO LIST, THEN TAKING MEAN OVER THAT LIST! About an order of magnitude faster than xarray's built in .mean() function but gives subtly different results, even though it's mathematically equivalent... anyway, stddev then taken with np.std([xr_dataset].[variable].data, mean=[mean from above calculation]), which also saves time over xarray's built in .std()
        
        - IDX_MIN_LON, IDX_MIN_LAT = lon/lat of southwesternmost grid point of domain. Currently set up for larger west coast domain.
        - IMG_SIZE_LON, IMG_SIZE_LAT = size of domain in pixels (i.e. grid cells). Currently set up for larger west coast domain.
            !!! LEGACY VALUES FOR CO DOMAIN, HRRR GRID !!!
                IDX_MIN_LON = 596
                IDX_MIN_LAT = 645
                IMG_SIZE_LON = 180
                IMG_SIZE_LAT = 180
            !!! LEGACY VALUES FOR CO DOMAIN, URMA GRID !!!
                IDX_MIN_LON = 796 
                IDX_MIN_LAT = 645 
                IMG_SIZE_LON = 180
                IMG_SIZE_LAT = 180
        """
        
        self.varname_translation_dict = {"pressurf":"sp",
                                         "t2m":"t2m",
                                         "d2m":"d2m",
                                         "spfh2m":"sh2",
                                         "u10m":"u10",
                                         "v10m":"v10"}
        
        self.urma_var_select_dict = {"sp":{'filter_by_keys':{'typeOfLevel': 'surface'}},
                                     "t2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                                     "d2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                                     "sh2":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}},
                                     "u10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}},
                                     "v10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}}}

        # For plot labeling purposes
        self.varname_units_dict = {"pressurf":"Pa",
                                   "t2m":"deg C",
                                   "d2m":"deg C",
                                   "spfh2m":"kg/kg",
                                   "u10m":"m/s",
                                   "v10m":"m/s"}
        
        
        self.hrrr_means_dict = {'train':{'pressurf':88264.4,
                                         't2m':284.451, 
                                         'd2m':273.703, 
                                         'spfh2m':0.00529336, 
                                         'u10m':1.17153, 
                                         'v10m':-0.313557}, 
                                'test':{'pressurf':88250.4,
                                         't2m':284.61, 
                                         'd2m':273.83, 
                                         'spfh2m':0.00530784, 
                                         'u10m':1.19730, 
                                         'v10m':-0.339266} }
        
        self.hrrr_stddevs_dict = {'train':{'pressurf':8493.89,
                                         't2m':11.0624, 
                                         'd2m':9.32916, 
                                         'spfh2m':0.00311317, 
                                         'u10m':3.14697, 
                                         'v10m':3.69828}, 
                                  'test':{'pressurf':8490.22,
                                         't2m':10.9822, 
                                         'd2m':9.14753, 
                                         'spfh2m':0.00306908, 
                                         'u10m':3.14104, 
                                         'v10m':3.71392} }
        
        self.urma_means_dict = {'train':{'pressurf':88264.4,
                                         't2m':284.451, 
                                         'd2m':273.703, 
                                         'spfh2m':0.00529336, 
                                         'u10m':1.17153, 
                                         'v10m':-0.313557}, 
                                'test':{'pressurf':88250.4,
                                         't2m':284.61, 
                                         'd2m':273.83, 
                                         'spfh2m':0.00530784, 
                                         'u10m':1.19730, 
                                         'v10m':-0.339266} }

        self.urma_stddevs_dict = {'train':{'pressurf':8493.89,
                                         't2m':11.0624, 
                                         'd2m':9.32916, 
                                         'spfh2m':0.00311317, 
                                         'u10m':3.14697, 
                                         'v10m':3.69828}, 
                                  'test':{'pressurf':8490.22,
                                         't2m':10.9822, 
                                         'd2m':9.14753, 
                                         'spfh2m':0.00306908, 
                                         'u10m':3.14104, 
                                         'v10m':3.71392} }
        
        
        
        # These are for the extended URMA grid (2345 lon x 1597 lat) which URMA and regridded HRRR should both live on
        # Problematic for Smartinit, which lives on the regular NDFD grid (2145 x 1377) !
        self.IDX_MIN_LON = 250
        self.IDX_MIN_LAT = 400
        self.IMG_SIZE_LON = 800
        self.IMG_SIZE_LAT = 800

        ## Common directories 
        self.DIR_TRAIN_TEST = f"/scratch3/BMC/wrfruc/aschein/Train_Test_Files"
        self.DIR_UNET_MAIN = f"/scratch3/BMC/wrfruc/aschein/UNet_main"
        self.DIR_TRAINED_MODELS = f"/scratch3/BMC/wrfruc/aschein/UNet_main/Trained_models"
        self.DIR_SMARTINIT_DATA = f"/scratch3/BMC/wrfruc/aschein/SMARTINIT_STUFF/smartinit_2024/output_files"

        #### (2025-08-27) Trying out universal means and sttdevs
        ### (2025-08-28) Updating to weighted mean: training = 3x the weight of testing b/c it has 3x the years
        # pressurf | mean = 88264.3583984375 | stddev = 8493.885375976562
        # t2m | mean = 284.45083236694336 | stddev = 11.062410235404968
        # d2m | mean = 273.70266342163086 | stddev = 9.329157590866089
        # spfh2m | mean = 0.005293358175549675 | stddev = 0.0031131717842072123
        # u10m | mean = 1.1715264171361923 | stddev = 3.1469656825065613
        # v10m | mean = -0.33926586061716074 | stddev = 3.7139175832271576

        ### Original means over western domain
        # self.hrrr_means_dict = {'train':{'pressurf':88272.8828125,
        #                                  't2m':284.317626953125, 
        #                                  'd2m':273.1583251953125, 
        #                                  'spfh2m':0.0051491828635335, 
        #                                  'u10m':1.1351977586746216, 
        #                                  'v10m':-0.4091095924377441}, 
        #                         'test':{'pressurf':88216.890625,
        #                                  't2m':284.95648193359375, 
        #                                  'd2m':273.657470703125, 
        #                                  'spfh2m':0.0051941452547908, 
        #                                  'u10m':1.2506071329116821, 
        #                                  'v10m':-0.3076884448528290} }
        
        # self.hrrr_stddevs_dict = {'train':{'pressurf':8476.998046875,
        #                                  't2m':11.1520891189575195, 
        #                                  'd2m':9.6434526443481445, 
        #                                  'spfh2m':0.0031340329442173, 
        #                                  'u10m':3.1567144393920898, 
        #                                  'v10m':3.7417495250701904}, 
        #                           'test':{'pressurf':8463.5615234375,
        #                                  't2m':10.8300733566284180, 
        #                                  'd2m':8.9211654663085938, 
        #                                  'spfh2m':0.0029620509594679, 
        #                                  'u10m':3.1530795097351074, 
        #                                  'v10m':3.7042634487152100} }
        
        # self.urma_means_dict = {'train':{'pressurf':88283.8046875,
        #                                  't2m':284.2685546875, 
        #                                  'd2m':273.9832763671875, 
        #                                  'spfh2m':0.0054085613228381, 
        #                                  'u10m':1.1562995910644531, 
        #                                  'v10m':-0.3208401799201965}, 
        #                         'test':{'pressurf':88227.9140625,
        #                                  't2m':284.8916320800781250, 
        #                                  'd2m':274.539031982421875, 
        #                                  'spfh2m':0.0054794875904918, 
        #                                  'u10m':1.2471121549606323, 
        #                                  'v10m':-0.2165891230106354} }

        # self.urma_stddevs_dict = {'train':{'pressurf':8518.107421875,
        #                                  't2m':11.1331777572631836, 
        #                                  'd2m':9.3781099319458008, 
        #                                  'spfh2m':0.0031805015169084, 
        #                                  'u10m':3.1490650177001953, 
        #                                  'v10m':3.7173516750335693},
        #                           'test':{'pressurf':8502.205078125,
        #                                  't2m':10.8134078979492188, 
        #                                  'd2m':8.6474075317382812, 
        #                                  'spfh2m':0.0029997199308127, 
        #                                  'u10m':3.1053075790405273, 
        #                                  'v10m':3.6297736167907715} }

        