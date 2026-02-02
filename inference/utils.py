import os
from datetime import datetime, timedelta
from typing import List
import xarray as xr
import numpy as np
import pandas as pd
import re
from calendar import monthrange 
from math import cos, sqrt
#
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Iterable, Sequence
import json


def filename_to_year(path: str) -> int:
    # Extracts the year from the filename, assuming the filename starts with a 4-digit year.
    filename = os.path.basename(path)
    return int(filename[:4])

def datetime_range(
    year: int, time_step: timedelta, n: int
) -> List[datetime]:
    # Generates a list of datetime objects starting from January 1st of the given year,
    initial_time = datetime(year=year, month=1, day=1)
    return [initial_time + time_step * i for i in range(n)]

def open_hdf5(*, path, f=None, metadata):
    # For creating initialization data in the correct format. 
    #   Opens an HDF5 file and returns an xarray Dataset.
    h5_path = metadata["h5_path"]
    dims = metadata["dims"]
    time_step_hours = metadata.get("dhours", 6)
    time_step = timedelta(hours=time_step_hours)

    ds = xr.open_dataset(f or path, engine="h5netcdf", phony_dims="sort")
    array = ds[h5_path]
    ds = array.rename(dict(zip(array.dims, dims)))
    year = filename_to_year(path)
    n = array.shape[0]
    ds = ds.assign_coords(
        time=datetime_range(year, time_step=time_step, n=n), **metadata["coords"]
    )
    ds = ds.assign_attrs(metadata["attrs"], path=path)
    return ds

def create_initialization_file(start_timestep=None, valid_timestep=None, init_fp='',):
    # Timesteps are in iso string format yyyy-mm-ddTHH:MM:SS

    # Filepath for ERA5 json data
    SFNO_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/ERA5_SFNO/testing_data"
    data_fp = os.path.join(SFNO_dir, 'data.json')

    print(f"Selecting timestep {start_timestep} to {valid_timestep}")

    # Open and load the JSON file
    with open(data_fp, 'r') as f:
        labels = json.load(f)

    # open initial conditions from stored ERA data
    year_of_timestep = datetime.fromisoformat(start_timestep).year
    data_create = open_hdf5(path = os.path.join(SFNO_dir, str(year_of_timestep)+'.h5'), metadata = labels)
    data_create = data_create.sel(time = [start_timestep, valid_timestep]) # this just selects the first and last time in the time range
    data_create = data_create.rename({"channel": "variable"})

    # Make dir of init_fp
    os.makedirs(os.path.dirname(init_fp), exist_ok=True)
    data_create.to_netcdf(init_fp)

def get_ivt(ds: xr.Dataset, bounding_box: dict = None) -> xr.DataArray:
    """
    Computes Integrated Vapor Transport (IVT) over the range 1000 hPa to 300 hPa.
    
    Args:
        ds: Xarray dataset containing the  model output variables.
        bounding_box: Dictionary with keys 'latitude_min', 'latitude_max', 
                      'longitude_min', 'longitude_max'.
                      
    Returns:
        xr.DataArray: The computed IVT magnitude.
    """
    # 1. Define the levels we want to integrate over (1000 to 300 hPa)
    # Ordered for integration (usually surface to aloft or vice versa; order matters for trapz sign)
    levels = [1000, 925, 850, 700, 600, 500, 400, 300]
    
    # 2. Construct variable names
    u_vars = [f"u{lvl}" for lvl in levels]
    v_vars = [f"v{lvl}" for lvl in levels]
    q_vars = [f"q{lvl}" for lvl in levels]
    
    # 3. Check if variables exist in ds
    missing = [v for v in u_vars + v_vars + q_vars if v not in ds]
    if missing:
        raise ValueError(f"Missing variables for IVT calculation: {missing}")

    # 4. Subset the Dataset to the Bounding Box FIRST (to save memory)
    # We create a temporary dataset with only the needed variables
    needed_vars = u_vars + v_vars + q_vars
    ds_sub = ds[needed_vars]
    
    if bounding_box:
        ds_sub = ds_sub.where(
            (ds['lat'] >= bounding_box['latitude_min']) & (ds['lat'] <= bounding_box['latitude_max']) &
            (ds['lon'] >= bounding_box['longitude_min']) & (ds['lon'] <= bounding_box['longitude_max']),
            drop=True
        )

    # 5. Stack variables into a vertical coordinate
    # We use xarray to concat along a new 'level' dimension
    u_stack = xr.concat([ds_sub[v] for v in u_vars], dim="level")
    v_stack = xr.concat([ds_sub[v] for v in v_vars], dim="level")
    q_stack = xr.concat([ds_sub[v] for v in q_vars], dim="level")
    
    # Assign the pressure levels as coordinates
    pressure_pa = np.array(levels) * 100.0 # should i convert to Pa?
    u_stack = u_stack.assign_coords(level=pressure_pa)
    v_stack = v_stack.assign_coords(level=pressure_pa)
    q_stack = q_stack.assign_coords(level=pressure_pa)

    # 6. Compute Zonal and Meridional IVT
    g = 9.80665
    # Calculate Fluxes (q * wind)
    qu = q_stack * u_stack
    qv = q_stack * v_stack
    
    # Integrate over pressure (axis=level)
    # Formula: -1/g * integral(q * u * dp)
    ivt_u = np.trapz(qu, x=pressure_pa, axis=u_stack.get_axis_num('level')) / g
    ivt_v = np.trapz(qv, x=pressure_pa, axis=v_stack.get_axis_num('level')) / g
    
    # 7. Compute Magnitude
    ivt_mag = np.hypot(ivt_u, ivt_v)
    
    # 8. Wrap result in DataArray
    # Use the coordinates from one of the sliced 2D variables (e.g., u1000)
    # to ensure lat/lon metadata is preserved.
    template = ds_sub[u_vars[0]]
    ivt_da = xr.DataArray(
        ivt_mag,
        coords=template.coords,
        dims=template.dims,
        name="ivt"
    )
    
    return ivt_da