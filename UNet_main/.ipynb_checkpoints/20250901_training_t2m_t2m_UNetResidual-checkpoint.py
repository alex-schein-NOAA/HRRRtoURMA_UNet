
from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_FunctionsAndClasses import *

###############

C = CONSTANTS()

current_model = DefineModelAttributes(is_train=True,
                                      with_terrains=['diff'],
                                      predictor_vars=['t2m'],
                                      target_vars=['t2m'],
                                      BATCH_SIZE=18,
                                      NUM_EPOCHS=30)

TRAINING_LOG_FILEPATH = f"{C.DIR_UNET_MAIN}/Training_logs/training_log_20250901_UNetResidual.txt"

TrainOneModel(current_model, 
              is_attention_model=False, 
              is_residual_model=True,
              use_residual_block=True,
              NUM_GPUS_TO_USE=2, 
              TRAINING_LOG_FILEPATH=TRAINING_LOG_FILEPATH, 
              TRAINED_MODEL_SAVEPATH=C.DIR_TRAINED_MODELS)

