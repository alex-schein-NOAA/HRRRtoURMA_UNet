#!/bin/bash

declare -a VARS=("pressurf" "t2m" "d2m" "spfh2m" "u10m"  "v10m") 

for VAR_OF_INTEREST in "${VARS[@]}"
do
    #VAR_OF_INTEREST=spfh2m
    REGRID_PATH=/data1/projects/RTMA/alex.schein/Regridded_HRRR/${VAR_OF_INTEREST}
    SP_REST_PATH=/data1/projects/RTMA/alex.schein/Regridded_HRRR_train_test/LOOSE_FILES/test_spatiallyrestricted_f01/${VAR_OF_INTEREST}
    
    ##### REGRID FILES, PUT THEM INTO REGRIDDED DIRECTORY #####
    
    cd ${REGRID_PATH}
    
    for yyyymmdd in * 
    do
        if [[ "${yyyymmdd}" == "2024"* ]]; then #"test --> only deal with 2024 files
        	cd ${yyyymmdd} #now working in REGRID_PATH/[yyyymmmdd]
        	cwd_hrrr_yyyymmmdd=$PWD
        	#echo ${yyyymmdd}
        
            for file in *; do
                if [[ ! "${file}" == *"idx" ]]; then #make sure it's not trying to do anything with the index files, if they exist
                    newfilename="${file::-6}_sp_rest.grib2" 
                    
                    # echo ${newfilename}
            
                    if ! test -f ${SP_REST_PATH}/${newfilename}; then #hasn't been restricted yet - do it
                        wgrib2 ${file} -ijsmall_grib 250:1049 400:1199 ${SP_REST_PATH}/${newfilename}
                        echo "${newfilename} has been created in ${SP_REST_PATH}"
                    fi
                fi  
            done #end working in Regridded_HRRR/${VAR_OF_INTEREST}/[yyyymmmdd]
            cd .. #return to Regridded_HRRR/${VAR_OF_INTEREST}
        fi #end 2024 check
    done #end main loop for current var

    ## PUT CODE TO DELETE 20241231 23z HERE
    
done #end loop over all vars