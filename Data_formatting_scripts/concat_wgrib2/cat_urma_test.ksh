#!/bin/bash

declare -a VARS=("pressurf" "t2m" "d2m" "spfh2m" "u10m" "v10m") 

for VAR_OF_INTEREST in "${VARS[@]}"
do
    #VAR_OF_INTEREST=spfh2m
    MASTER_FILE_PATH=/data1/projects/RTMA/alex.schein/URMA_train_test
    SP_REST_PATH=/data1/projects/RTMA/alex.schein/URMA_train_test/LOOSE_FILES/test/${VAR_OF_INTEREST}
    
    ##### Concatenate files. MUST MAKE SURE THEY ARE ORDERED AND ALL CORRECT #####
    output_filename=test_urma_alltimes_${VAR_OF_INTEREST}.grib2
    if ! test -f ${MASTER_FILE_PATH}/${output_filename}; then #master grib2 doesn't exist; make it
        cd ${SP_REST_PATH}
        echo "Concatenating testing files for ${VAR_OF_INTEREST}"
        cat *.grib2 > ${MASTER_FILE_PATH}/${output_filename}
        echo "Done concatenating ${VAR_OF_INTEREST}"
    else
        echo "${output_filename} already exists in ${MASTER_FILE_PATH}. Skipping it"
    fi
    
done #end loop over all vars