#!/bin/bash

declare -a VARS=("pressurf" "t2m" "d2m" "spfh2m" "u10m"  "v10m") 

for VAR_OF_INTEREST in "${VARS[@]}"
do
    #VAR_OF_INTEREST=spfh2m
    REGRID_PATH=/data1/projects/RTMA/alex.schein/Regridded_URMA/
    SP_REST_PATH=/data1/projects/RTMA/alex.schein/URMA_train_test/LOOSE_FILES/test/${VAR_OF_INTEREST}
    
    ##########
    if [[ "${VAR_OF_INTEREST}" == "pressurf" ]]; then
        SELECTION_STR="PRES:surface"
    elif [[ "${VAR_OF_INTEREST}" == "t2m" ]]; then
        SELECTION_STR="TMP:2 m"
    elif [[ "${VAR_OF_INTEREST}" == "d2m" ]]; then
        SELECTION_STR="DPT:2 m"
    elif [[ "${VAR_OF_INTEREST}" == "spfh2m" ]]; then
        SELECTION_STR="SPFH:2 m"
    elif [[ "${VAR_OF_INTEREST}" == "u10m" ]]; then
        SELECTION_STR="UGRD:10 m"
    elif [[ "${VAR_OF_INTEREST}" == "v10m" ]]; then
        SELECTION_STR="VGRD:10 m"
    fi
    
    cd ${REGRID_PATH}
    for yyyymmdd in * 
    do
        if [[ "${yyyymmdd}" == "2024"* ]]; then #"test --> only deal with 2024 files
        	cd ${yyyymmdd} #now working in REGRID_PATH/[yyyymmmdd]
        	cwd_hrrr_yyyymmmdd=$PWD
        	#echo ${yyyymmdd}
        
            for file in *; do
                if [[ ! "${file}" == *"idx" ]]; then #make sure it's not trying to do anything with the index files, if they exist
                    tmp="${file:8:4}"
                    newfilename="urma_regridded_${yyyymmdd}_${tmp}_sp_rest.grib2" 
                    
                    # echo ${newfilename}
            
                    if ! test -f ${SP_REST_PATH}/${newfilename}; then #hasn't been restricted yet - do it
                        wgrib2 ${file} -match "${SELECTION_STR}" -ijsmall_grib 250:1049 400:1199 ${SP_REST_PATH}/${newfilename}
                        echo "${newfilename} has been created in ${SP_REST_PATH}"
                    fi
                fi  
            done #end working in Regridded_HRRR/${VAR_OF_INTEREST}/[yyyymmmdd]
            cd .. #return to Regridded_HRRR/${VAR_OF_INTEREST}
        fi #end 2024 check
    done #end main loop for current var
    
done #end loop over all vars