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
import time

import torch

# Check if CUDA (GPU support) is available
is_available = torch.cuda.is_available()
print(f"Is CUDA available? {is_available}")

if is_available:
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    # Get the ID of the current GPU
    current_gpu = torch.cuda.current_device()
    print(f"Current GPU ID: {current_gpu}")

    # Get the name of the current GPU
    gpu_name = torch.cuda.get_device_name(current_gpu)
    print(f"Current GPU Name: {gpu_name}")

    print(f"Memory (VRAM):      {torch.cuda.get_device_properties(current_gpu).total_memory / 1e9:.2f} GB")
else:
    print("CUDA is not available. Running on CPU.")

time_start = time.time()

############# Double check these before running the script #############
#select start datetime and n_steps, each n_step = 6hrs
start_datetime = "2021-09-20T00:00:00" # "2021_09_20T00:00:00"
variables_to_select = ['tcwv'] #Only save selected variables - it slows down inference SIGNIFICANTLY to save all 74 variables
experiment_number = 0
n_steps = 20  #number of 6hr steps to forecast

boring = False
ema = False

# Create the inference name based on the start datetime and number of steps
inference_name = datetime.fromisoformat(start_datetime).strftime("%Y_%m_%dT%H")+'_nsteps'+str(n_steps)
data_create_fp = "/projectnb/eb-general/wade/sfno/inference_runs/Ian/Initialize_data/Initialize_"+inference_name+".nc"

# Calculate the final datetime based from the start datetime and number of steps
final_datetime = (datetime.fromisoformat(start_datetime) + timedelta(hours = int(n_steps*6))).isoformat() 

# Directories
results_out_dir = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/Experiment{str(experiment_number)}/{final_datetime[:10].replace('-', '_')}/"

############# Double check these before running the script #############


if os.path.exists(data_create_fp):
    print(f"Data already preprocessed: {data_create_fp}")
else:
    sys.exit(f"Data not found use Create_Initial_Data.ipynb to create: {data_create_fp}")

#make this xarray into a dataarray file for earth2studio
initial_data = data.DataArrayFile(data_create_fp)

time_1 = time.time()
print(f"Data loaded in {time_1 - time_start:.2f} seconds")


for n_epoch in np.arange(1,3): #70,1):
    time_2 = time.time()
    # if boring:
    #     # Create the final datetime string in the desired format
    #     
    # else:# Create the final datetime string in the desired format

    if ema:
        results_out_fp = results_out_dir+f"EMA_Checkpoint{n_epoch}_{inference_name}.nc"
    else:
        results_out_fp = results_out_fp = results_out_dir+"/Checkpoint"+str(n_epoch)+"_"+inference_name+'.nc' 
    
    # Check if the results file already exists
    if os.path.exists(results_out_fp):
        print(f"Results file {results_out_fp} already exists. Skipping to next epoch.")
        continue  # Skip the rest of the loop and go to the next iteration
    else:
        os.makedirs(os.path.dirname(results_out_fp), exist_ok=True)

        load_dotenv()  

        # Make temporary folder with all the metadata in it.
        src_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/Checkpoints_SFNO/sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999/"
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



#     for n_epoch in np.arange(36,71,1):
#         time_2 = time.time()
#         if boring:
#             # Create the final datetime string in the desired format
#             results_out_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Forecasts_Boring/"+final_datetime[:10].replace("-", "_")+"/Checkpoint"+str(n_epoch)+"_"+inference_name+'.nc'
#         else:# Create the final datetime string in the desired format
#             if ema:
#                 results_out_fp = f"/barnes-engr-scratch2/C837824079/Experiment{str(experiment_number)}/Forecast/EMA_9/Checkpoint{n_epoch}_{inference_name}.nc"         
#             else:
#                 results_out_fp = "/projectnb/eb-general/rbaiman/SFNO/Example_Inference/Example_Forecast/Checkpoint"+str(n_epoch)+"_"+inference_name+'.nc'

#         # Check if the results file already exists
#         if os.path.exists(results_out_fp):
#             print(f"Results file {results_out_fp} already exists. Skipping to next epoch.")
#             continue  # Skip the rest of the loop and go to the next iteration
#         else:
#             os.makedirs(os.path.dirname(results_out_fp), exist_ok=True)

#             load_dotenv()  

#             # Make temporary folder with all the metadata in it.
#             src_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/Checkpoints_SFNO/sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999/"

#             # Load the model package from storage
#             model_package = Package(src_dir, cache = False)
#             model = SFNO.load_model(model_package, checkpoint_name = 'ckpt_mp0_epoch'+str(n_epoch)+'.tar', EMA = ema)

#             # Create the IO handler, store in memory
#             io = ZarrBackend()
            
#             with torch.no_grad():
#                 # run inference
#                 io = deterministic([start_datetime], n_steps, model, initial_data, io, variables_list=variables_to_select)

#             print(io.root.tree())


#             # save results to netcdf
#             # Open the Zarr group from the in-memory store using xarray
#             ds = xr.open_zarr(io.root.store)

#             # Convert the 'time' coordinate in ds to datetime64 format
#             ds["time"] = ds["time"].astype("datetime64[ns]")

#             # Convert lead_time from nanoseconds to timedelta64[ns]
#             base_time = ds["time"].values  # shape (n_time,)
#             lead_timedelta = ds["lead_time"].values.astype("timedelta64[ns]")  # shape (n_lead_time,)
#             # Broadcast to 2D: (time, lead_time)
#             valid_timesteps = (base_time[:, None] + lead_timedelta[None, :]).flatten() 
#             # Drop the old lead_time coordinate
#             ds = ds.drop_vars("lead_time")

#             # Assume ds has dimensions (time, lead_time, lat, lon) and only one time
#             initial_time = str(ds["time"].values[0])  # Save the initial time as a string
#             # Remove the time dimension by selecting the first (and only) time
#             ds = ds.isel(time=0).drop_vars("time")
#             # Add the initial time as a global attribute
#             ds.attrs["initial_time"] = initial_time

#             # Create valid_time by adding lead_timedelta to base_time
#             ds = ds.rename({"lead_time": "valid_time"})
#             # Assign valid_time as a coordinate
#             ds = ds.assign_coords(valid_time=(("valid_time",), valid_timesteps))

#             # only save the final time step
#             if np.datetime64(final_datetime) in ds["valid_time"].values:
#                 ds = ds.sel(valid_time=[final_datetime])
#                 ds = ds[variables_to_select]
#                 ds.to_netcdf(results_out_fp, mode="w", format="NETCDF4")
#                 print(f"Results saved to {results_out_fp}")
#             else:
#                 print(f"ERROR: final_datetime {final_datetime} not found in ds['valid_time']. No file saved.")


#             #some cleanup
#             torch.cuda.empty_cache()
#             del model_package
#             del model
#             del io
#             del ds
#             gc.collect()
#             time_3 = time.time()
#             print(f"Epoch {n_epoch} done: {time_3 - time_2:.2f} seconds")


# for n_epoch in np.arange(1,21,1):
#     time_2 = time.time()
#     # Create the final datetime string in the desired format
#     if boring:
#         # Create the final datetime string in the desired format
#         results_out_fp = "/barnes-engr-scratch2/C837824079/Experiment"+str(experiment_number)+"/Forecasts_Boring/"+final_datetime[:10].replace("-", "_")+"/Checkpoint"+str(n_epoch+70)+"_"+inference_name+'.nc'
#     else:# Create the final datetime string in the desired format
#         if ema:
#             results_out_fp = f"/barnes-engr-scratch2/C837824079/Experiment{str(experiment_number)}/Forecast/EMA_9/Checkpoint{n_epoch+70}_{inference_name}.nc"
#         else:
#             results_out_fp = "/projectnb/eb-general/rbaiman/SFNO/Example_Inference/Example_Forecast/Checkpoint"+str(n_epoch+70)+"_"+inference_name+'.nc'

    
#     # Check if the results file already exists
#     if os.path.exists(results_out_fp):
#         print(f"Results file {results_out_fp} already exists. Skipping to next epoch.")
#         continue  # Skip the rest of the loop and go to the next iteration
#     else:
#         os.makedirs(os.path.dirname(results_out_fp), exist_ok=True)

#         load_dotenv()  

#         # Make temporary folder with all the metadata in it.
#         src_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/Checkpoints_SFNO/multistep_sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999-multistep2/"

#         # Load the model package from storage
#         model_package = Package(src_dir, cache = False)
#         model = SFNO.load_model(model_package, checkpoint_name = 'ckpt_mp0_epoch'+str(n_epoch)+'.tar', EMA = ema)

#         # Create the IO handler, store in memory
#         io = ZarrBackend()

#         print(f"Running inference for {inference_name}")
#         with torch.no_grad():
#             # run inference
#             io = deterministic([start_datetime], n_steps, model, initial_data, io, variables_list=variables_to_select)

#         # print(io.root.tree())

#         # save results to netcdf
#         # Open the Zarr group from the in-memory store using xarray
#         ds = xr.open_zarr(io.root.store)

#         # Convert the 'time' coordinate in ds to datetime64 format
#         ds["time"] = ds["time"].astype("datetime64[ns]")

#         # Convert lead_time from nanoseconds to timedelta64[ns]
#         base_time = ds["time"].values  # shape (n_time,)
#         lead_timedelta = ds["lead_time"].values.astype("timedelta64[ns]")  # shape (n_lead_time,)
#         # Broadcast to 2D: (time, lead_time)
#         valid_timesteps = (base_time[:, None] + lead_timedelta[None, :]).flatten() 
#         # Drop the old lead_time coordinate
#         ds = ds.drop_vars("lead_time")

#         # Assume ds has dimensions (time, lead_time, lat, lon) and only one time
#         initial_time = str(ds["time"].values[0])  # Save the initial time as a string
#         # Remove the time dimension by selecting the first (and only) time
#         ds = ds.isel(time=0).drop_vars("time")
#         # Add the initial time as a global attribute
#         ds.attrs["initial_time"] = initial_time

#         # Create valid_time by adding lead_timedelta to base_time
#         ds = ds.rename({"lead_time": "valid_time"})
#         # Assign valid_time as a coordinate
#         ds = ds.assign_coords(valid_time=(("valid_time",), valid_timesteps))

#         # only save the final time step
#         if np.datetime64(final_datetime) in ds["valid_time"].values:
#             ds = ds.sel(valid_time=[final_datetime])
#             ds = ds[variables_to_select]
#             ds.to_netcdf(results_out_fp, mode="w", format="NETCDF4")
#             print(f"Results saved to {results_out_fp}")
#         else:
#             print(f"ERROR: final_datetime {final_datetime} not found in ds['valid_time']. No file saved.")


#         #some cleanup
#         torch.cuda.empty_cache()
#         del model_package
#         del model
#         del io
#         del ds
#         gc.collect()
#         time_3 = time.time()
#         print(f"Epoch {n_epoch+70} done: {time_3 - time_2:.2f} seconds")

