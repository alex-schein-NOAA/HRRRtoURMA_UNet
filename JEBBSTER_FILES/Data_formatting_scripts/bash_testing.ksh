#!/bin/bash

declare -a VARS=("spfh2m" "fail") # "v10m" "d2m" "pressurf" "t2m") #(7/25) Redoing all vars with wexp grid (see regridding notes)

for VAR_OF_INTEREST in "${VARS[@]}"
do
    # VAR_OF_INTEREST=t2m
    HRRR_PATH=/data1/projects/RTMA/alex.schein/Herbie_downloads/${VAR_OF_INTEREST}/hrrr
    REGRID_PATH=/data1/projects/RTMA/alex.schein/Regridded_HRRR/${VAR_OF_INTEREST}
    
    ##### REGRID FILES, PUT THEM INTO REGRIDDED DIRECTORY #####
    
    cd ${REGRID_PATH}
    
    for yyyymmdd in * 
    do
    	cd ${yyyymmdd}
    	cwd_hrrr_yyyymmmdd=$PWD
    
        for file in *; do
            if [[ ! "${file}" == *"idx" ]]; then #make sure it's not trying to do anything with the index files, if they exist
                if [[ "$file" == *"_grib2_f01"* ]]; then
                
                rm ${file}
                
                fi
				
            fi  
        done #end working in in Herbie_downloads/${VAR_OF_INTEREST}/hrrr/[yyyymmmdd]
        
        cd .. #return to in Herbie_downloads/${VAR_OF_INTEREST}/hrrr
        
    done #end main loop for current var
done #end loop over all vars