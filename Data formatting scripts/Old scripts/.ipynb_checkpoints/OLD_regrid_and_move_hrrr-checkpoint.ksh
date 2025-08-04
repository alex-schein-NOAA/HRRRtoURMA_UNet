#!/bin/bash

# Load wgrib2
module load wgrib2

# Paths relative to RTMA/alex.schein directory
HRRR_PATH=Herbie_downloads/hrrr
REGRID_PATH=Regridded_HRRR

##### REGRID FILES, PUT THEM INTO REGRIDDED DIRECTORY #####
cd ${HRRR_PATH}
for yyyymmdd in * 
do
	cd ${yyyymmdd} #now working in Herbie_downloads/hrrr/[yyyymmmdd]
	cwd_hrrr_yyyymmmdd=$PWD
	echo ${yyyymmdd}

	for file in *
	do
		# Extract just the tXXz part of file name, assuming it looks like [something].tXXz.something.grib2
		tXXz=${file#*.}
		tXXz=${tXXz%.*}
		tXXz=${tXXz%.*}

		newfilename="hrrr_regridded_${yyyymmdd}_${tXXz}.grib2"
		echo ${newfilename}

		# # This part assumes the filepath to [regrid directory]/[yyyymmdd] ALREADY EXISTS!
		# if ! test -f ${newfilename}; then #hasn't been regridded yet - do it
		# 	wgrib2 ${file} -set_radius 1:6370000 -set_grib_type c3 -set_bitmap 0 -new_grid_winds grid -new_grid_interpolation bilinear -new_grid ncep grid 184 ${newfilename}
		# 	echo "${newfilename} has been created"

		# 	cd ../../.. #go up to aschein
		# 	mv ${HRRR_PATH}/${yyyymmdd}/${newfilename} ${REGRID_PATH}/${yyyymmdd}/${newfilename}
		# 	echo "${newfilename} has been moved to ${REGRID_PATH}/${yyyymmdd}/${newfilename}"
		# 	cd ${cwd_hrrr_yyyymmmdd} #return to Herbie_downloads/hrrr/yyyymmdd
		# else
		# 	echo "${newfilename} already exists in ${cwd_hrrr_yyyymmmdd}"	
		# 	cd ../../.. #go up to aschein
		# 	mv ${HRRR_PATH}/${yyyymmdd}/${newfilename} ${REGRID_PATH}/${yyyymmdd}/${newfilename}
		# 	echo "${newfilename} has been moved to ${REGRID_PATH}/${yyyymmdd}/${newfilename}"
		# 	cd ${cwd_hrrr_yyyymmmdd} #return to Herbie_downloads/hrrr/yyyymmdd
		# fi

	done #end looping over grib2 files in yyyymmdd directory

	cd .. #return to Herbie_downloads/hrrr

done #end looping over all yyyymmdd directories in Herbie_downloads/hrrr

