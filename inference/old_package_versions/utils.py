import os
from datetime import datetime, timedelta
from typing import List
import xarray as xr
import numpy as np
import pandas as pd
import re
from calendar import monthrange 
from math import cos, sqrt
from geopy import distance
import scipy.stats

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
    # Opens an HDF5 file and returns an xarray Dataset.
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





