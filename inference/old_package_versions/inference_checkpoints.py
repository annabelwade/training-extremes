import os
import subprocess
from dotenv import load_dotenv

from earth2studio.io import ZarrBackend
from SFNO_update import SFNO
import earth2studio.data as data
from earth2studio.models.auto import Package
from utils import filename_to_year, datetime_range, open_hdf5
from deterministic_update import deterministic

from datetime import datetime, timedelta
import json
import xarray as xr
from typing import List
import shutil
import sys
import gc
import numpy as np

import torch

if torch.cuda.is_available():
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    torch.cuda.set_device(gpu_id)
import time
time_start = time.time()

############# Double check these before running the script #############
slurm_select = int(sys.argv[1]) # I run this with 3 slurm array tasks so checkpoint 1-35, 36-70, 71-90 run at the same time

#select start datetime and n_steps, each n_step = 6hrs
start_datetime = '2019-03-22T00:00:00' 
variables_to_select = ['msl', 'tcwv'] #Only save selected variables - it slows down inference SIGNIFICANTLY to save all 74 variables
experiment_number = 5
n_steps = 12

boring = False
ema = False

# Create the inference name based on the start datetime and number of steps
inference_name = datetime.fromisoformat(start_datetime).strftime("%Y_%m_%dT%H")+'_nsteps'+str(n_steps)
data_create_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Initialize_data/Initialize_"+inference_name+".nc"

############# Double check these before running the script #############



# Calculate the final datetime based from the start datetime and number of steps
final_datetime = (datetime.fromisoformat(start_datetime) + timedelta(hours = int(n_steps*6))).isoformat() 


if os.path.exists(data_create_fp):
    print(f"Data already preprocessed: {data_create_fp}")
else:
    sys.exit(f"Data not found use Create_Initial_Data.ipynb to create: {data_create_fp}")

#make this xarray into a dataarray file for earth2studio
initial_data = data.DataArrayFile(data_create_fp)

time_1 = time.time()
print(f"Data loaded in {time_1 - time_start:.2f} seconds")

if slurm_select == 0:

    for n_epoch in np.arange(1,36,1):
        time_2 = time.time()
        if boring:
            # Create the final datetime string in the desired format
            results_out_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Forecasts_Boring/"+final_datetime[:10].replace("-", "_")+"/Checkpoint"+str(n_epoch)+"_"+inference_name+'.nc'
        else:# Create the final datetime string in the desired format
            if ema:
                results_out_fp = f"/barnes-engr-scratch2/C837824079/Experiment{str(experiment_number)}/Forecast/EMA_9/Checkpoint{n_epoch}_{inference_name}.nc"
            else:
                results_out_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Forecast/Checkpoint"+str(n_epoch)+"_"+inference_name+'.nc'
        # Check if the results file already exists
        if os.path.exists(results_out_fp):
            print(f"Results file {results_out_fp} already exists. Skipping to next epoch.")
            continue  # Skip the rest of the loop and go to the next iteration
        else:
            os.makedirs(os.path.dirname(results_out_fp), exist_ok=True)

            load_dotenv()  

            # Make temporary folder with all the metadata in it.

            src_dir = "/barnes-engr-scratch2/C837824079/Checkpoints_SFNO/sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999/"

            # Load the model package from storage
            model_package = Package(src_dir, cache = False)
            model = SFNO.load_model(model_package, checkpoint_name = 'ckpt_mp0_epoch'+str(n_epoch)+'.tar', EMA = ema)

            # Create the IO handler, store in memory
            io = ZarrBackend()
            
            with torch.no_grad():
                # run inference
                io = deterministic([start_datetime], n_steps, model, initial_data, io, variables_list=variables_to_select)

            print(io.root.tree())


            # save results to netcdf
            # Open the Zarr group from the in-memory store using xarray
            ds = xr.open_zarr(io.root.store)

            # Convert the 'time' coordinate in ds to datetime64 format
            ds["time"] = ds["time"].astype("datetime64[ns]")

            # Convert lead_time from nanoseconds to timedelta64[ns]
            base_time = ds["time"].values  # shape (n_time,)
            lead_timedelta = ds["lead_time"].values.astype("timedelta64[ns]")  # shape (n_lead_time,)
            # Broadcast to 2D: (time, lead_time)
            valid_timesteps = (base_time[:, None] + lead_timedelta[None, :]).flatten() 
            # Drop the old lead_time coordinate
            ds = ds.drop_vars("lead_time")

            # Assume ds has dimensions (time, lead_time, lat, lon) and only one time
            initial_time = str(ds["time"].values[0])  # Save the initial time as a string
            # Remove the time dimension by selecting the first (and only) time
            ds = ds.isel(time=0).drop_vars("time")
            # Add the initial time as a global attribute
            ds.attrs["initial_time"] = initial_time

            # Create valid_time by adding lead_timedelta to base_time
            ds = ds.rename({"lead_time": "valid_time"})
            # Assign valid_time as a coordinate
            ds = ds.assign_coords(valid_time=(("valid_time",), valid_timesteps))

            # only save the final time step
            if np.datetime64(final_datetime) in ds["valid_time"].values:
                ds = ds.sel(valid_time=[final_datetime])
                ds = ds[variables_to_select]
                ds.to_netcdf(results_out_fp, mode="w", format="NETCDF4")
                print(f"Results saved to {results_out_fp}")
            else:
                print(f"ERROR: final_datetime {final_datetime} not found in ds['valid_time']. No file saved.")


            #some cleanup
            torch.cuda.empty_cache()
            del model_package
            del model
            del io
            del ds
            gc.collect()
            time_3 = time.time()
            print(f"Epoch {n_epoch} done: {time_3 - time_2:.2f} seconds")

elif slurm_select == 1:

    for n_epoch in np.arange(36,71,1):
        time_2 = time.time()
        if boring:
            # Create the final datetime string in the desired format
            results_out_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Forecasts_Boring/"+final_datetime[:10].replace("-", "_")+"/Checkpoint"+str(n_epoch)+"_"+inference_name+'.nc'
        else:# Create the final datetime string in the desired format
            if ema:
                results_out_fp = f"/barnes-engr-scratch2/C837824079/Experiment{str(experiment_number)}/Forecast/EMA_9/Checkpoint{n_epoch}_{inference_name}.nc"         
            else:
                results_out_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Forecast/Checkpoint"+str(n_epoch)+"_"+inference_name+'.nc'

        # Check if the results file already exists
        if os.path.exists(results_out_fp):
            print(f"Results file {results_out_fp} already exists. Skipping to next epoch.")
            continue  # Skip the rest of the loop and go to the next iteration
        else:
            os.makedirs(os.path.dirname(results_out_fp), exist_ok=True)

            load_dotenv()  

            # Make temporary folder with all the metadata in it.
            src_dir = "/barnes-engr-scratch2/C837824079/Checkpoints_SFNO/sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999/"

            # Load the model package from storage
            model_package = Package(src_dir, cache = False)
            model = SFNO.load_model(model_package, checkpoint_name = 'ckpt_mp0_epoch'+str(n_epoch)+'.tar', EMA = ema)

            # Create the IO handler, store in memory
            io = ZarrBackend()
            
            with torch.no_grad():
                # run inference
                io = deterministic([start_datetime], n_steps, model, initial_data, io, variables_list=variables_to_select)

            print(io.root.tree())


            # save results to netcdf
            # Open the Zarr group from the in-memory store using xarray
            ds = xr.open_zarr(io.root.store)

            # Convert the 'time' coordinate in ds to datetime64 format
            ds["time"] = ds["time"].astype("datetime64[ns]")

            # Convert lead_time from nanoseconds to timedelta64[ns]
            base_time = ds["time"].values  # shape (n_time,)
            lead_timedelta = ds["lead_time"].values.astype("timedelta64[ns]")  # shape (n_lead_time,)
            # Broadcast to 2D: (time, lead_time)
            valid_timesteps = (base_time[:, None] + lead_timedelta[None, :]).flatten() 
            # Drop the old lead_time coordinate
            ds = ds.drop_vars("lead_time")

            # Assume ds has dimensions (time, lead_time, lat, lon) and only one time
            initial_time = str(ds["time"].values[0])  # Save the initial time as a string
            # Remove the time dimension by selecting the first (and only) time
            ds = ds.isel(time=0).drop_vars("time")
            # Add the initial time as a global attribute
            ds.attrs["initial_time"] = initial_time

            # Create valid_time by adding lead_timedelta to base_time
            ds = ds.rename({"lead_time": "valid_time"})
            # Assign valid_time as a coordinate
            ds = ds.assign_coords(valid_time=(("valid_time",), valid_timesteps))

            # only save the final time step
            if np.datetime64(final_datetime) in ds["valid_time"].values:
                ds = ds.sel(valid_time=[final_datetime])
                ds = ds[variables_to_select]
                ds.to_netcdf(results_out_fp, mode="w", format="NETCDF4")
                print(f"Results saved to {results_out_fp}")
            else:
                print(f"ERROR: final_datetime {final_datetime} not found in ds['valid_time']. No file saved.")


            #some cleanup
            torch.cuda.empty_cache()
            del model_package
            del model
            del io
            del ds
            gc.collect()
            time_3 = time.time()
            print(f"Epoch {n_epoch} done: {time_3 - time_2:.2f} seconds")

elif slurm_select == 2:
    for n_epoch in np.arange(1,21,1):
        time_2 = time.time()
        # Create the final datetime string in the desired format
        if boring:
            # Create the final datetime string in the desired format
            results_out_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Forecasts_Boring/"+final_datetime[:10].replace("-", "_")+"/Checkpoint"+str(n_epoch+70)+"_"+inference_name+'.nc'
        else:# Create the final datetime string in the desired format
            if ema:
                results_out_fp = f"/barnes-engr-scratch2/C837824079/Experiment{str(experiment_number)}/Forecast/EMA_9/Checkpoint{n_epoch+70}_{inference_name}.nc"
            else:
                results_out_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Forecast/Checkpoint"+str(n_epoch+70)+"_"+inference_name+'.nc'

       
        # Check if the results file already exists
        if os.path.exists(results_out_fp):
            print(f"Results file {results_out_fp} already exists. Skipping to next epoch.")
            continue  # Skip the rest of the loop and go to the next iteration
        else:
            os.makedirs(os.path.dirname(results_out_fp), exist_ok=True)

            load_dotenv()  

            # Make temporary folder with all the metadata in it.
            src_dir = "/barnes-engr-scratch2/C837824079/Checkpoints_SFNO/multistep_sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999-multistep2/"

            # Load the model package from storage
            model_package = Package(src_dir, cache = False)
            model = SFNO.load_model(model_package, checkpoint_name = 'ckpt_mp0_epoch'+str(n_epoch)+'.tar', EMA = ema)

            # Create the IO handler, store in memory
            io = ZarrBackend()

            print(f"Running inference for {inference_name}")
            with torch.no_grad():
                # run inference
                io = deterministic([start_datetime], n_steps, model, initial_data, io, variables_list=variables_to_select)

            # print(io.root.tree())

            # save results to netcdf
            # Open the Zarr group from the in-memory store using xarray
            ds = xr.open_zarr(io.root.store)

            # Convert the 'time' coordinate in ds to datetime64 format
            ds["time"] = ds["time"].astype("datetime64[ns]")

            # Convert lead_time from nanoseconds to timedelta64[ns]
            base_time = ds["time"].values  # shape (n_time,)
            lead_timedelta = ds["lead_time"].values.astype("timedelta64[ns]")  # shape (n_lead_time,)
            # Broadcast to 2D: (time, lead_time)
            valid_timesteps = (base_time[:, None] + lead_timedelta[None, :]).flatten() 
            # Drop the old lead_time coordinate
            ds = ds.drop_vars("lead_time")

            # Assume ds has dimensions (time, lead_time, lat, lon) and only one time
            initial_time = str(ds["time"].values[0])  # Save the initial time as a string
            # Remove the time dimension by selecting the first (and only) time
            ds = ds.isel(time=0).drop_vars("time")
            # Add the initial time as a global attribute
            ds.attrs["initial_time"] = initial_time

            # Create valid_time by adding lead_timedelta to base_time
            ds = ds.rename({"lead_time": "valid_time"})
            # Assign valid_time as a coordinate
            ds = ds.assign_coords(valid_time=(("valid_time",), valid_timesteps))

            # only save the final time step
            if np.datetime64(final_datetime) in ds["valid_time"].values:
                ds = ds.sel(valid_time=[final_datetime])
                ds = ds[variables_to_select]
                ds.to_netcdf(results_out_fp, mode="w", format="NETCDF4")
                print(f"Results saved to {results_out_fp}")
            else:
                print(f"ERROR: final_datetime {final_datetime} not found in ds['valid_time']. No file saved.")


            #some cleanup
            torch.cuda.empty_cache()
            del model_package
            del model
            del io
            del ds
            gc.collect()
            time_3 = time.time()
            print(f"Epoch {n_epoch+70} done: {time_3 - time_2:.2f} seconds")

else:
    sys.exit("Invalid slurm_select value. Please use 0, 1, or 2.")