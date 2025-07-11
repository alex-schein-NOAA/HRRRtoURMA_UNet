import argparse
import cfgrib
import os
import numpy as np
import xarray as xr
from pyproj import Proj, Transformer
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime as dt
import pandas as pd
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def open_hrrr_from_grib(path):
    """
    Given a path to a hrrr grib file, use cfgrib to open it up and extract data
    as numpy arrays. These are then packaged into dictionaries and returned.
    """
    #cfgrib opens all the level types (surface, isobaric, etc) as xarrays in a list 
    datasets = cfgrib.open_datasets(path, backend_kwargs = {"indexpath" : ""})

    #specify the useful level indices
    ENTIRE_ATMOSPHERE_LEVEL_IDX = 0
    HEIGHT_ABOVE_GROUND_10m_IDX = 2
    HEIGHT_ABOVE_GROUND_2m_IDX = 3
    ISOBARIC_LEVELS_IDX = 4

    #get non-pressure-level (i.e. 2D) fields
    refc = datasets[ENTIRE_ATMOSPHERE_LEVEL_IDX]["refc"].values[np.newaxis,...] #adds a "layer" dimension
    u10 = datasets[HEIGHT_ABOVE_GROUND_10m_IDX]["u10"].values[np.newaxis,...]
    v10 = datasets[HEIGHT_ABOVE_GROUND_10m_IDX]["v10"].values[np.newaxis,...]
    t2m = datasets[HEIGHT_ABOVE_GROUND_2m_IDX]["t2m"].values[np.newaxis,...]
    sh2 = datasets[HEIGHT_ABOVE_GROUND_2m_IDX]["sh2"].values[np.newaxis,...]

    #get pressure level (i.e. 3D) fields
    plvls = np.linspace(50., 1000., 20) # 50., 100., 150., ..., 950., 1000.
    level_indices = np.where(np.isin(datasets[ISOBARIC_LEVELS_IDX]["isobaricInhPa"].values, plvls))[0]
    
    u = datasets[ISOBARIC_LEVELS_IDX]["u"].values[level_indices, ...]
    v = datasets[ISOBARIC_LEVELS_IDX]["v"].values[level_indices, ...]
    w = datasets[ISOBARIC_LEVELS_IDX]["w"].values[level_indices, ...]
    q = datasets[ISOBARIC_LEVELS_IDX]["q"].values[level_indices, ...]
    gh = datasets[ISOBARIC_LEVELS_IDX]["gh"].values[level_indices, ...]
    t = datasets[ISOBARIC_LEVELS_IDX]["t"].values[level_indices, ...]

    #get coordinate info
    p = datasets[ISOBARIC_LEVELS_IDX]["isobaricInhPa"].values[level_indices]
    time = datasets[ISOBARIC_LEVELS_IDX]["time"].values
    step = datasets[ISOBARIC_LEVELS_IDX]["step"].values
    x = datasets[ISOBARIC_LEVELS_IDX]["x"].values * datasets[ISOBARIC_LEVELS_IDX]["t"].attrs["GRIB_DxInMetres"]
    y = datasets[ISOBARIC_LEVELS_IDX]["y"].values * datasets[ISOBARIC_LEVELS_IDX]["t"].attrs["GRIB_DyInMetres"]
    lats = datasets[ISOBARIC_LEVELS_IDX]["latitude"].values
    lons = datasets[ISOBARIC_LEVELS_IDX]["longitude"].values

    #get map projection attributes
    hrrr_ds_variable = datasets[ISOBARIC_LEVELS_IDX]['t']
    lat_1=hrrr_ds_variable.attrs["GRIB_Latin1InDegrees"]
    lat_2=hrrr_ds_variable.attrs["GRIB_Latin2InDegrees"]
    lat_0=hrrr_ds_variable.attrs["GRIB_LaDInDegrees"]
    lon_0=hrrr_ds_variable.attrs["GRIB_LoVInDegrees"]
    lat_xy0=hrrr_ds_variable.attrs["GRIB_latitudeOfFirstGridPointInDegrees"]
    lon_xy0=hrrr_ds_variable.attrs["GRIB_longitudeOfFirstGridPointInDegrees"]
    
    #organize into dicts and return
    coordinate_dict = {
        "p" : p,
        "x" : x,
        "y" : y,
        "lats" : lats,
        "lons" : lons,
        "time" : time,
        "step" : step,
    }
    
    variable_dict = {
        "refc" : refc,
        "u10"  : u10,
        "v10"  : v10,
        "t2m"  : t2m,
        "sh2"  : sh2,
        "u"    : u,
        "v"    : v,
        "w"    : w,
        "q"    : q,
        "gh"   : gh,
        "t"    : t
    }

    proj_dict = {
        "lat_1"   : lat_1,
        "lat_2"   : lat_2,
        "lat_0"   : lat_0,
        "lon_0"   : lon_0,
        "lat_xy0" : lat_xy0,
        "lon_xy0" : lon_xy0
    }

    return coordinate_dict, variable_dict, proj_dict

def concatenate_across_timesteps(data_dict):
    """
    Given the lists of variable data, coordinate data, and projection data across
    all timesteps, construct dictionaries with data concatenated across the 
    timestep dimension. E.g. a 2D field will go from (1, y, x) to (tstep, y, x).
    A 3D field will go from (plvls, y, x) to (tstep, plvls, y, x).

    Returns tuple of dicts as in open_hrrr_from_grib(...)
    """

    NUM_PRESSURE_LEVELS = 20 #big ole hardcoded assumption... TODO: maybe kwarg?
    
    #handle variable concatenation first
    num_steps = len(data_dict["var"])
    var_names = list(data_dict["var"][0].keys())
    concat_variable_dict = {}
    
    for var_name in var_names:
    
        #make sure the field is either 2D or on pressure levels
        var_shape = data_dict["var"][0][var_name].shape
        assert var_shape[0] == 1 or var_shape[0] == NUM_PRESSURE_LEVELS #get angry if plvl assumption goes awry
    
        #make an empty array to assign data into across all timesteps
        is_2D = var_shape[0] == 1
        if is_2D:
            concatenation_shape = (num_steps, var_shape[1], var_shape[2])
        else:
            concatenation_shape = (num_steps, var_shape[0], var_shape[1], var_shape[2])
        concat_data = np.empty(concatenation_shape, dtype = np.float32)
    
        #assign the data to the array and add to variable dictionary
        for step_idx in range(num_steps):
            if is_2D:
                concat_data[step_idx,...] = data_dict["var"][step_idx][var_name][0,...]
            else:
                concat_data[step_idx,...] = data_dict["var"][step_idx][var_name]
        concat_variable_dict[var_name] = concat_data

    #with coordinates, only the step changes
    step_list = []
    for step_idx in range(num_steps):
        step_list.append(data_dict["coord"][step_idx]["step"])
    
    coordinate_dict = data_dict["coord"][0]
    coordinate_dict["step"] = np.array(step_list)
    
    #and projection parameters are even easier... they stay the same
    proj_dict = data_dict["proj"][0]

    return coordinate_dict, concat_variable_dict, proj_dict

def create_xr_dataset(coordinate_dict, variable_dict, proj_dict):
    """
    Given the concatenated output from concatenate_across_timesteps(...),
    construct an xarray dataset.
    """

    #add coordinate information to variables
    for v in variable_dict:
        data = variable_dict[v]
        if len(data.shape) == 3:
            variable_dict[v] = (["step","y", "x"], data.squeeze())
        else:
            variable_dict[v] = (["step","p", "y", "x"], data)

    #add coordinate information to lat-lon coords
    for coord in coordinate_dict:
        data = coordinate_dict[coord]
        if len(coordinate_dict[coord].shape) > 1:
            coordinate_dict[coord] = (["y", "x"], data)

    return xr.Dataset(
        data_vars = variable_dict,
        coords = coordinate_dict,
        attrs = proj_dict
    )

def get_nc_save_path(initialization_datetime):
    """
    Little helper function to consolidate figuring out where to save an
    initialization time's netCDF file containing all variables across
    all timesteps.
    """
    save_path_fmt = "/data1/projects/AIWP/Marshall.Baldwin/hrrr/raw/%Y%m%d/hrrr.%Y%m%d%H.nc"
    return dt.strftime(initialization_datetime, save_path_fmt)

def all_steps_are_present(sorted_paths, initialization_time):
    """
    Checks whether every step forecast step for a particular initialization time is
    present in a list of sorted paths to HRRR grib files. Returns a bool.

    :arg sorted_paths: list[str] - Sorted full paths to HRRR grib files. An example path
    would be /data1/ai-datadepot/models/hrrr/conus/grib2-subset/20241219/hrrr.t00z.wrfprsf00.grib2
    :arg initialization_time: datetime.datetime - The initialization time. Needed to check
    whether forecast goes out to 18 or 48 hours.
    """
    
    EXTENDED_FORECAST_STEPS = 49 #analysis + 48 1hr steps
    STANDARD_FORECAST_STEPS = 19 #analysis + 18 1hr steps
    
    fHr_steps = [int(f[-8:-6]) for f in sorted_paths] #should be [0, 1, ..., num_fHr_steps]
    if initialization_time.hour % 6 == 0:
        return fHr_steps == [i for i in range(EXTENDED_FORECAST_STEPS)]
    else:
        return fHr_steps == [i for i in range(STANDARD_FORECAST_STEPS)]

def save_all_steps_to_nc(paths, datetime, compression_level = 1, save_nc = True):
    """
    Given paths to all HRRR grib files for a particular initialization time,
    save a netcdf containing data across all available forecast hours.

    TODO: What about the case where a step is missing?? Does that mess with
    opening multiple files with Dask?
    """

    data_dict = {
        "coord" : [],
        "var"   : [],
        "proj"  : []
    }
    
    #open each grib file and append to the data dictionaries
    for path in paths:
        coordinate_dict, variable_dict, proj_dict = open_hrrr_from_grib(path)
        data_dict["coord"].append(coordinate_dict)
        data_dict["var"].append(variable_dict)
        data_dict["proj"].append(proj_dict)

    #concatenate across the timestep dimension and create xarray Dataset
    coordinate_dict, variable_dict, proj_dict = concatenate_across_timesteps(data_dict)
    full_ds = create_xr_dataset(coordinate_dict, variable_dict, proj_dict)
    
    #save concatenated dataset to netcdf
    save_path = get_nc_save_path(datetime)
    os.makedirs(os.path.dirname(save_path), exist_ok = True) #make sure the daily directory exists
    if save_nc:
        comp = dict(zlib=True, complevel=compression_level)
        encoding = {var: comp for var in full_ds.data_vars}
        full_ds.to_netcdf(save_path, encoding=encoding)
    else:
        return full_ds
        
def create_hrrr_projection(proj_dict):
    """
    Given a dict of hrrr map projection attributes from a grib file,
    use those grib attrs to construct a projection from lat-lon to
    the hrrr's LCC projection space.
    """ 
    
    #define the lambert conic conformal projection with no northing or easting
    EARTH_SPHERE_RADIUS = 6371229.
    proj = Proj(
        proj='lcc',
        lat_1=proj_dict["lat_1"],
        lat_2=proj_dict["lat_2"],
        lat_0=proj_dict["lat_0"],
        lon_0=proj_dict["lon_0"],
        R=EARTH_SPHERE_RADIUS,
        ellps='sphere'
    )
    
    #find the easting and northing by plugging in (x,y) = (0,0) in naive projection
    x_0, y_0 = proj(proj_dict["lon_xy0"], proj_dict["lat_xy0"])
    
    #remake projection with easting and northing
    return Proj(
            proj='lcc',
            lat_1=proj_dict["lat_1"],
            lat_2=proj_dict["lat_2"],
            lat_0=proj_dict["lat_0"],
            lon_0=proj_dict["lon_0"],
            x_0 = -x_0,
            y_0 = -y_0,
            R=EARTH_SPHERE_RADIUS,
            ellps='sphere'
        )

def create_regular_lat_lon_grid(grid_spacing = 3., bottom = 24., top = 50., left=233., right=294.):
    """
    Returns meshes of latitude and longitude with a specified -- but approximate! -- grid spacing.

    :kwarg grid_spacing: float - The desired grid spacing in kilometers
    :kwarg bottom: float - The southmost latitude in degrees north; grid starts from here
    :kwarg top: float - The northmost latitude in degrees north; grid goes at most to here
    :kwarg left: float - The westmost latitude in degrees east; grid stats from here
    :kwarg right: float - The eastmost latutude in degrees east; grid goes at most to here
    """
    #what 3km corresponds to in degrees of latitude and longitude
    KM_IN_1DEG_LATITUDE = 111.12
    dlat = grid_spacing / KM_IN_1DEG_LATITUDE
    dlon = dlat / np.cos((bottom + top) * (np.pi/ 360.)) #accounting for longitude compression as a function of latitude
    
    #create regular lat lon grid over conus
    synth_lons = np.arange(left, right, dlon)
    synth_lats = np.arange(bottom, top, dlat)
    lon_mesh, lat_mesh = np.meshgrid(synth_lons, synth_lats)
    return lon_mesh, lat_mesh

def regrid_data_to_regular_lat_lon(variable_array, proj, x, y, lon_mesh, lat_mesh):
    """
    Regrids variables from the regular LCC hrrr grid to a regular lat-lon grid

    :arg variable_array: np.array(..., dtype = float-like) - Array containing all of
                         the hrrr fields that need regridding. Shape = (num_fields, y, x)
    :arg proj: pyproj.Transformer: Projection from lat-lon to hrrr's LCC grid
    :arg x: np.array - 1D array of the hrrr grid's x coordinates
    :arg y: np.array - 1D array of the hrrr grid's y coordinates
    :arg lon_mesh: np.array - 2D array of the longitudes on the regular lat-lon grid
    :arg lat_mesh: np.array - 2D array of the latitudes on the regular lat-lon grid
    """

    #instantiate Transformer that goes from LCC projection (hrrr's) to lat-lon
    transformer_object = Transformer.from_proj(
        proj_from=proj,
        proj_to='epsg:4326',
        always_xy=True
    )

    #get the projected coordinates for the regular lat lon mesh
    desired_x_matrix_metres, desired_y_matrix_metres = (
        transformer_object.transform(
            lon_mesh, lat_mesh,
            direction='INVERSE'
        )
    )
    desired_point_matrix = np.stack([desired_y_matrix_metres, desired_x_matrix_metres], axis=-1)

    #make an interpolator for each channel via list comprehension
    interp_objects = [
        RegularGridInterpolator(
        points=(y, x),
        values=variable_array[i, ...],
        method='linear',
        bounds_error=False,
        fill_value=np.nan) for i in range(concatenated_vars.shape[0])
    ]
    
    #interpolate all of the channels using their respective interpolator object
    desired_data_matrix = np.stack(
        [interp_object(desired_point_matrix) for interp_object in interp_objects], axis = 0
    )
    return desired_data_matrix

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, required=True, help='Date string in YYYYMMDD format')
    args = parser.parse_args()
    day = args.day

    #make sure the daily HRRR grib directory actually exists
    hrrr_dir = "/data1/ai-datadepot/models/hrrr/conus/grib2-subset"
    day_dir = os.path.join(hrrr_dir, day)
    if not os.path.exists(day_dir):
        print(f"Directory for {day} not found.")
        return

    #get all paths from the daily directory and parse initialization times present
    paths = [os.path.join(day_dir, file) for file in os.listdir(day_dir)]
    dt_fmt = "/data1/ai-datadepot/models/hrrr/conus/grib2-subset/%Y%m%d/hrrr.t%Hz"
    path_slice = slice(-16)
    initialization_times = sorted(list(set([dt.strptime(p[path_slice], dt_fmt) for p in paths])))

    #create an aggregate nc file for each initialization time
    for date in initialization_times:

        #skip if you've already created the nc file
        if os.path.exists(get_nc_save_path(date)): 
            continue

        #otherwise process an initialization times if all fHr steps are present
        paths_subset = sorted([p for p in paths if dt.strptime(p[path_slice], dt_fmt) == date])
        if all_steps_are_present(paths_subset, date):
            save_all_steps_to_nc(paths_subset, date)

if __name__ == "__main__":
    main()
        