#!/bin/bash

URMA_AIDATADEPOT_PATH=/data1/ai-datadepot/models/urma/2p5km/grib2
REGRID_PATH=/data1/projects/RTMA/alex.schein/Regridded_URMA

##### MAKE REGRID DIRECTORY #####
### Run this if there's no YYYYMMDD folders in Regridded_URMA
# cd ${URMA_AIDATADEPOT_PATH}
# for filename in *  
# do
# 	cd ${URMA_AIDATADEPOT_PATH}/${filename} #now working in/ai-datadepot/models/hrrr/conus/grib2/[yyyymmdd]
# 	yyyymmdd=${PWD##*/} #nab the yyyymmdd folder name
# 	mkdir -p ${REGRID_PATH}/${yyyymmdd} #make corresponding folder in regrid path, if doesn't exist
# 	cd ${URMA_AIDATADEPOT_PATH} #return to start
#     echo "${yyyymmdd} created"
# done

######################################

cd ${URMA_AIDATADEPOT_PATH}

for yyyymmdd in * 
do
    cd ${yyyymmdd} #now working in /data1/ai-datadepot/models/urma/2p5km/grib2/[yyyymmmdd]
	cwd_hrrr_yyyymmmdd=$PWD
	echo ${yyyymmdd}

    for file in *; do
        if [[ ! "${file}" == *"idx" ]]; then #make sure it's not trying to do anything with the index files, if they exist
            if ! test -f ${REGRID_PATH}/${yyyymmdd}/${file}; then #hasn't been regridded yet - do it
                wgrib2 ${file} -set_radius 1:6370000 -grib_out ${REGRID_PATH}/${yyyymmdd}/${file}
                echo "${file} created"
            else
    			echo "${file} already exists in ${REGRID_PATH}/${yyyymmdd}"
            fi #end if ! test -f ${REGRID_PATH}/${yyyymmdd}/${file}
        fi #end if [[ ! "${file}" == *"idx" ]]
    done #done with current yyyymmdd
    cd ${URMA_AIDATADEPOT_PATH} #return to top when done with current yyyymmdd
done #end for yyyymm dd