#!/bin/bash

# Paths relative to RTMA/alex.schein directory
HRRR_PATH=/ai-datadepot/models/hrrr/conus/grib2
REGRID_PATH=scratch/Regridded_HRRR

##### REGRID FILES, PUT THEM INTO REGRIDDED DIRECTORY #####

#This assumes we launch the script from /scracth/RTMA/alex.schein/hrrr_CNNN_testing/Data formatting scripts
cd ../../../../..
cd ${HRRR_PATH}

for yyyymmdd in * 
do
	cd ${yyyymmdd} #now working in [yyyymmmdd]
	cwd_hrrr_yyyymmmdd=$PWD
	echo ${yyyymmdd}

    for file in *; do
        if [[ "${file}" == *"wrfnat"* ]]; then
            if [[ ! "${file}" == *"idx" ]]; then #make sure it's not trying to do anything with the index files
                # Extract just the tXXz part of file name, assuming it looks like [something].tXXz.something.grib2
                tXXz=${file#*.}
                tXXz=${tXXz%.*}
                tXXz=${tXXz%.*}
        
                newfilename="hrrr_regridded_${yyyymmdd}_${tXXz}.grib2"
                echo ${newfilename}
        
                #done #done with file
            fi 
        fi 
    done #end working in [yyyymmdd] directory in /ai-datadepot/models/hrrr/conus/grib2
    
    cd .. #return to main grib2 directory (/ai-datadepot/models/hrrr/conus/grib2)
    
done #end main loop