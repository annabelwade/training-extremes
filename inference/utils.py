import os
from datetime import datetime, timedelta
from typing import List
import xarray as xr
import numpy as np
import pandas as pd
import re
from calendar import monthrange 
from math import cos, sqrt

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Iterable, Sequence
import json
import matplotlib.pyplot as plt

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

# def create_initialization_file(start_timestep=None, valid_timestep=None, init_fp='',):
#     # Timesteps are in iso string format yyyy-mm-ddTHH:MM:SS

#     # See if init_fp already exists, if so skip
#     if os.path.exists(init_fp):
#         # print(f"Initialization file {init_fp} already exists, skipping creation.")
#         return

#     # Filepath for ERA5 json data
#     SFNO_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/ERA5_SFNO/testing_data"
#     data_fp = os.path.join(SFNO_dir, 'data.json')

#     # print(f"Selecting timestep {start_timestep} to {valid_timestep}")

#     # Open and load the JSON file
#     with open(data_fp, 'r') as f:
#         labels = json.load(f)

#     # open initial conditions from stored ERA data
#     year_of_timestep = datetime.fromisoformat(start_timestep).year
#     data_create = open_hdf5(path = os.path.join(SFNO_dir, str(year_of_timestep)+'.h5'), metadata = labels)
#     data_create = data_create.sel(time = [start_timestep, valid_timestep]) # this just selects the first and last time in the time range
#     data_create = data_create.rename({"channel": "variable"})

#     # Make dir of init_fp
#     os.makedirs(os.path.dirname(init_fp), exist_ok=True)
#     data_create.to_netcdf(init_fp)

def create_initialization_file(start_timestep=None, valid_timestep=None, init_fp=''):
    # Timesteps are in iso string format yyyy-mm-ddTHH:MM:SS
    tmp_fp = init_fp + ".tmp"

    # Check if init_fp already exists and is healthy
    if os.path.exists(init_fp):
        size_bytes = os.path.getsize(init_fp)
        if size_bytes < 1024: 
            print(f" Warning: Found corrupted file {init_fp} ({size_bytes} bytes). Deleting...")
            os.remove(init_fp)
        else:
            try:
                with xr.open_dataset(init_fp) as temp_ds:
                    pass
                return # File is healthy, skip creation
            except Exception as e:
                print(f" Warning: {init_fp} exists but cannot be opened ({type(e).__name__}). Deleting...")
                os.remove(init_fp)

    # Check if another task is currently generating this file
    if os.path.exists(tmp_fp):
        # print(f"Task detected {tmp_fp}. Waiting for generation to finish...")
        while not os.path.exists(init_fp):
            time.sleep(5)
        return # Once the final file appears, we can safely return

    # Claim the file by creating an empty .tmp file immediately
    os.makedirs(os.path.dirname(init_fp), exist_ok=True)
    with open(tmp_fp, 'w') as f:
        pass 

    # Generate the data
    SFNO_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/ERA5_SFNO/testing_data"
    data_fp = os.path.join(SFNO_dir, 'data.json')

    with open(data_fp, 'r') as f:
        labels = json.load(f)

    year_of_timestep = datetime.fromisoformat(start_timestep).year
    data_create = open_hdf5(path = os.path.join(SFNO_dir, str(year_of_timestep)+'.h5'), metadata = labels)
    data_create = data_create.sel(time = [start_timestep, valid_timestep]) 
    data_create = data_create.rename({"channel": "variable"})
    
    # write to temp file first, then rename to final file
    data_create.to_netcdf(tmp_fp)
    os.rename(tmp_fp, init_fp)

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

### 2NN Intrinsic Dimension Estimation helpers ###
def get_sequential_initializations(years, n_samples, n_steps=1, init_dir='/projectnb/eb-general/wade/sfno/inference_runs/intrinsic_dim/init_files'):
    """
    Generates sequential initialization files for a given list of years, number of samples, and steps.
    Args:
    - years: List of years to generate initializations from.
    - n_samples: Total number of initialization samples to generate.
    - n_steps: Number of steps to forecast (default is 1).
    - init_dir: Directory to save the initialization files.
    Returns:
    - init_fps: List of file paths for the generated initialization files.
    - start_timesteps: List of starting timesteps corresponding to each initialization file.
    """
    start_timesteps = []
    
    for year in years:
        curr_date = datetime(year, 1, 1, 0, 0)
        end_date = datetime(year, 12, 31, 18, 0)
        
        while curr_date <= end_date:
            start_timesteps.append(curr_date.isoformat())
            curr_date += timedelta(hours=6 * n_steps)
            
            # If we hit our target number of samples, break the while loop
            if len(start_timesteps) >= n_samples:
                break
                
        # If we hit our target, also break the outer year loop
        if len(start_timesteps) >= n_samples:
            break

    init_fps =[]
    n_steps = int(n_steps) # ensure n_steps is an int for filename formatting
    # save every initialization to an .nc file using create_initialization_file
    for idx, start in enumerate(start_timesteps):
        valid_time = (datetime.fromisoformat(start) + timedelta(hours=6 * n_steps)).isoformat()
        init_fp = init_dir +'/Initialize_'+start[:13].replace('-', '_') + "_nsteps" + str(n_steps) + ".nc"
        create_initialization_file(start_timestep=start, valid_timestep=valid_time, init_fp=init_fp)
        init_fps.append(init_fp)

        if idx % 10 == 0:
            print(f"Checked for existence/created {idx+1}/{len(start_timesteps)} initialization files.")

    return init_fps, start_timesteps

def compute_2nn_id(representations, task_id, job_id, epoch, layer_name, figs_dir, 
data_range=(None,None), nsteps=1, init_timesteps=None, filter_outliers=True):
    """
    Computes 2NN Intrinsic Dimension.
    Args:
    - representations: N x D numpy array of representations where D is the flattened length of that layer's activations.
    - task_id: task id num (used in plot naming).
    - job_id: job id num (used in plot naming).
    - epoch: epoch num (used in plot naming).
    - layer_name: name of the layer (used in plot naming).
    - figs_dir: directory to save the plot.
    - data_range: tuple of (min, max) of the date range of the data, for diagnostic purposes
    - nsteps: number of steps in the forecast, for diagnostic purposes
    Returns: linregress outputs
    - slope: slope of the linear fit.
    - intercept: intercept of the linear fit.
    - r_value: correlation coefficient of the fit.
    - p_value: two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
    - std_err: standard error of the estimated slope.
    """
    from scipy.stats import linregress
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    import numpy as np
    
    N = representations.shape[0]
    
    # 1. Compute Nearest Neighbors 
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='euclidean').fit(representations)
    distances, indices = nbrs.kneighbors(representations)
    assert distances.shape == (N, 3), f"Expected distances shape to be (N, 3) but got {distances.shape}"

    r1 = np.maximum(distances[:, 1], 1e-10) 
    r2 = np.maximum(distances[:, 2], 1e-10) 
    
    # 2. Compute mu & sort
    mu = np.sort(r2 / r1)
    x = np.log(mu)

    # check if any values are above 0.2.. this is purely for sandbox diagnosing my exp 6 (N=25) results where i had a couple unusual points above .2
    above_thresh = np.where(x > 0.2)
    if above_thresh[0].size > 0:    
        # print the mu value and the corresponding index and timestep
        print(f"Found {above_thresh[0].size} samples with mu > 0.2. Sample indices and mu values:")
        for idx in above_thresh[0]:
            timestep_info = f" (timestep: {init_timesteps[idx]})" if init_timesteps is not None else ""
            print(f"Index: {idx}, log(mu): {x[idx]:.4f}{timestep_info}")
    
    if filter_outliers:
        # Filter out samples that are above 3 stdevs
        x_mean = np.mean(x)
        x_std = np.std(x)
        upper_bound = x_mean + 3 * x_std
        below_upper_bound = x <= upper_bound
        print(f"Filtering out {np.sum(~below_upper_bound)} outliers above {upper_bound:.4f} (mean + 3*std)."
              f" log(mu) values and indices of filtered samples:")
        for idx in np.where(~below_upper_bound)[0]:
            timestep_info = f" (timestep: {init_timesteps[idx]})" if init_timesteps is not None else ""
            print(f"Index: {idx}, log(mu): {x[idx]:.4f}{timestep_info}")
        
        x = x[below_upper_bound]
        distances = distances[below_upper_bound]
        indices = indices[below_upper_bound]
        N = x.shape[0] 
        init_timesteps = [init_timesteps[i] for i in range(len(init_timesteps)) if below_upper_bound[i]] if init_timesteps is not None else None

    
    # 3. Linear Fit
    k_indices = np.arange(1, N + 1)
    x = x[:-1]; k_indices = k_indices[:-1] # filter out the last point where k_indices = N which would cause log(0) in the y calculation
    y = -np.log(1 - (k_indices / N))
    assert x.shape == y.shape, f"Expected x and y to have the same shape but got x.shape={x.shape} and y.shape={y.shape}"
    assert x.shape == (N-1,), f"Expected x and y to have shape (N-1,) but got {x.shape}"

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # --- HANDLING COLOR NORMALIZATION ---
    if init_timesteps is not None:
        # Convert list of ISO strings to list of datetime objects
        init_datetimes = [datetime.fromisoformat(t) for t in init_timesteps]
        # Get timestamps (floats) for numerical operations
        timestamps = [dt.timestamp() for dt in init_datetimes]
        
        cmap = plt.get_cmap('viridis')
        # Use Python's built-in min/max on the float list to avoid NumPy string ufunc errors
        t_min, t_max = min(timestamps), max(timestamps)
        
        norm = plt.Normalize(t_min, t_max)
        # Match the shape of x and y (N-1)
        colors = [cmap(norm(t)) for t in timestamps[:-1]]
    else:
        colors = ['blue'] * len(x)
    
    # --- PLOTTING ---
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=colors, s=15, alpha=0.7, edgecolors='none')
    
    # Add a colorbar to make sense of the time dimension
    if init_timesteps is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t_min, t_max))
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('initialization timestep')

    plt.plot(x, intercept + slope * x, color='red', 
             label=f'ID Fit (slope={slope:.2f}, $R^2$={r_value**2:.3f})', linewidth=2)
    
    plt.xlabel('$\ln(\mu)$')
    plt.ylabel('$-\ln(1 - k/N)$')
    
    # Use the passed data_range strings for the title
    plt.title(f'2NN ID: Epoch {epoch}, {layer_name}\n'
              f'Range: {data_range[0]} to {data_range[1]} | N={N}')
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs(figs_dir, exist_ok=True)
    plot_name = f"id_fit_job{job_id}_task{task_id}_epoch{epoch}_{layer_name}_timeColor.png"
    plt.savefig(os.path.join(figs_dir, plot_name), bbox_inches='tight', dpi=150)
    plt.close() 

    # make another plot without the time coloring
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='blue', s=15, alpha=0.7, edgecolors='none')
    plt.plot(x, intercept + slope * x, color='red', 
             label=f'ID Fit (slope={slope:.2f}, $R^2$={r_value**2:.3f})', linewidth=2)
    plt.xlabel('$\ln(\mu)$')
    plt.ylabel('$-\ln(1 - k/N)$')
    plt.title(f'2NN ID: Epoch {epoch}, {layer_name}\n'
              f'Range: {data_range[0]} to {data_range[1]} | N={N}')
    plt.legend()
    plt.grid(alpha=0.3)
    plot_name_no_color = f"id_fit_job{job_id}_task{task_id}_epoch{epoch}_{layer_name}.png"
    plt.savefig(os.path.join(figs_dir, plot_name_no_color), bbox_inches='tight', dpi=150)
    plt.close()
    
    return slope, intercept, r_value, p_value, std_err