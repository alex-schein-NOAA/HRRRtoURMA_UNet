#!/bin/bash

declare -a VARS=("u10m" "v10m" "d2m" "pressurf" "t2m" "spfh2m") #(7/25) Redoing all vars with wexp grid

for VAR_OF_INTEREST in "${VARS[@]}"
do
    HRRR_PATH=/data1/projects/RTMA/alex.schein/Herbie_downloads/${VAR_OF_INTEREST}/hrrr
    REGRID_PATH=/data1/projects/RTMA/alex.schein/Regridded_HRRR/${VAR_OF_INTEREST}
    
    ##### MAKE REGRID DIRECTORY #####
    cd ${HRRR_PATH}
    for filename in *  
    do
    	cd ${filename} #now working in Herbie_downloads/hrrr/[currentdate]
    	echo ${PWD}
    	yyyymmdd=${PWD##*/} #nab the yyyymmdd folder name
    	#cd ../../.. #go back up to alex.schein
    	mkdir -p ${REGRID_PATH}/${yyyymmdd} #make corresponding folder in regrid path, if doesn't exist
    	cd ${HRRR_PATH} #return to start
    done
done