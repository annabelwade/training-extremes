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
