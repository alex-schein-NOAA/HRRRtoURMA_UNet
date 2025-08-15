#!/bin/bash

# Paths relative to RTMA/alex.schein directory
#PARENT_PATH=$PWD
HRRR_PATH=/ai-datadepot/models/hrrr/conus/grib2
REGRID_PATH=/scratch/RTMA/alex.schein/Regridded_HRRR_aidatadepot

##### MAKE REGRID DIRECTORY #####
cd ${HRRR_PATH}
for filename in *  
do
	cd ${filename} #now working in/ai-datadepot/models/hrrr/conus/grib2/[yyyymmdd]
	echo ${PWD}
	yyyymmdd=${PWD##*/} #nab the yyyymmdd folder name
	cd ${REGRID_PATH} #go back up to alex.schein
	mkdir -p ${REGRID_PATH}/${yyyymmdd} #make corresponding folder in regrid path, if doesn't exist
	cd ${HRRR_PATH} #return to start
done