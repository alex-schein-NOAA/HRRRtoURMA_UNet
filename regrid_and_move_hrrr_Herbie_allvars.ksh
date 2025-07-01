#!/bin/bash

declare -a VARS=("u10m" "v10m") #"d2m" "pressurf" "t2m") #(6/25) Need to redo u and v

for VAR_OF_INTEREST in "${VARS[@]}"
do
    # VAR_OF_INTEREST=t2m
    HRRR_PATH=/data1/projects/RTMA/alex.schein/Herbie_downloads/${VAR_OF_INTEREST}/hrrr
    REGRID_PATH=/data1/projects/RTMA/alex.schein/Regridded_HRRR/${VAR_OF_INTEREST}
    
    ##### REGRID FILES, PUT THEM INTO REGRIDDED DIRECTORY #####
    
    cd ${HRRR_PATH}
    
    for yyyymmdd in * 
    do
    	cd ${yyyymmdd} #now working in Herbie_downloads/${VAR_OF_INTEREST}/hrrr/[yyyymmmdd]
    	cwd_hrrr_yyyymmmdd=$PWD
    	echo ${yyyymmdd}
    
        for file in *; do
            if [[ ! "${file}" == *"idx" ]]; then #make sure it's not trying to do anything with the index files, if they exist
                # Extract just the tXXz part of file name, assuming it looks like [something].tXXz.[something].grib2
                tXXz=${file#*.}
                tXXz=${tXXz%.*}
                tXXz=${tXXz%.*}
        
                newfilename="hrrr_regridded_${yyyymmdd}_${tXXz}_f01.grib2" # !!! CHANGE THE "f[YY]" PORTION WHEN DOING DIFFERENT TIMES - if doing multiple forecast times then figure out how to loop that in, but not a concern as of 6/10
                
                #echo ${newfilename}
        
                # This part assumes the filepath to [regrid directory]/[yyyymmdd] ALREADY EXISTS!
				if ! test -f ${REGRID_PATH}/${yyyymmdd}/${newfilename}; then #hasn't been regridded yet - do it
					wgrib2 ${file} -set_radius 1:6371950 -set_grib_type c3 -set_bitmap 0 -new_grid_winds grid -new_grid_interpolation bilinear -new_grid ncep grid 184 ${newfilename}
					echo "${newfilename} has been created"
		
					# cd ../../.. #go up to aschein
					mv ${HRRR_PATH}/${yyyymmdd}/${newfilename} ${REGRID_PATH}/${yyyymmdd}/${newfilename}
					echo "${newfilename} has been moved to ${REGRID_PATH}/${yyyymmdd}/"
					cd ${cwd_hrrr_yyyymmmdd} #return to Herbie_downloads/hrrr/yyyymmdd
				else
					echo "${newfilename} already exists in ${REGRID_PATH}/${yyyymmdd}"	
				fi
            fi  
        done #end working in in Herbie_downloads/${VAR_OF_INTEREST}/hrrr/[yyyymmmdd]
        
        cd .. #return to in Herbie_downloads/${VAR_OF_INTEREST}/hrrr
        
    done #end main loop for current var
done #end loop over all vars