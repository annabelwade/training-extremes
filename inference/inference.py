import os
import subprocess
from dotenv import load_dotenv

from earth2studio.io import ZarrBackend
from deterministic_update import deterministic
from SFNO_update import SFNO

import earth2studio.data as data
from earth2studio.models.auto import Package
from utils import create_initialization_file #filename_to_year, datetime_range, open_hdf5 # these aren't used in this script currently

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

############# CONFIGURATIONS #############

if len(sys.argv) > 1:
    # If provided a number (e.g., "3"), use it as the experiment number
    experiment_number = int(sys.argv[1])
else:
    experiment_number = 2

# Load Configuration from JSON
config_path = f'./configs/exp{experiment_number}.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Parse Experiment Setup
exp_params = config['experiment_setup']
event_type = exp_params['event_type']
variables_to_save = exp_params['variables_to_save'] # Mapping JSON key to internal variable name
valid_timestep = exp_params['valid_timestep']
leadtimes = exp_params['leadtimes_days']
ema = exp_params['ema']

# Parse Model Parameters
model_params = config['model_parameters']
fine_tuning_start_epoch = model_params['fine_tuning_start_epoch']
epochs_setting = model_params['epochs_to_run']

# Parse Epoch Logic
if epochs_setting == "odds":
    epochs_to_run = np.arange(1, 90, 2)
elif epochs_setting == "evens":
    epochs_to_run = np.arange(2, 91, 2)
elif epochs_setting == "all":
    epochs_to_run = np.arange(1, 91, 1)
elif isinstance(epochs_setting, list):
    epochs_to_run = np.array(epochs_setting)
else:
    raise ValueError(f"Unknown epochs_to_run setting: {epochs_setting}")

# Compute initialization timesteps 
valid_datetime = datetime.fromisoformat(valid_timestep)
init_timesteps = [] 
for lt in leadtimes:
    init_timesteps.append( (valid_datetime - timedelta(days=lt)).isoformat() )
n_6h_steps = [lt * 4 for lt in leadtimes]

# Directories
path_params = config['paths']
base_output_dir = path_params['base_output_dir']
results_out_dir = f"{base_output_dir}/Experiment{str(experiment_number)[0]}/{valid_timestep[:10].replace('-', '_')}/"
# make experiment directory if it DNE
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

#################################################
# --- PARALLELIZATION SETUP ---
# SGE_TASK_ID is 1-indexed 
task_id_env = os.environ.get('SGE_TASK_ID')
if task_id_env is None:
    print("WARNING: SGE_TASK_ID not found. Defaulting to Task 1 of 1 (Running all epochs).")
    task_id = 0
    num_tasks = 1
else:
    task_id = int(task_id_env) - 1 # Convert to 0-indexed for numpy splitting
    num_tasks = 4 # Fixed to 4 for now

# Split epochs among tasks
epochs_subset = np.array_split(epochs_to_run, num_tasks)[task_id]
print(f"--- JOB ARRAY INFO ---")
print(f"Task ID (0-indexed): {task_id} / {num_tasks - 1}")
print(f"Total Epochs: {len(epochs_to_run)}")
print(f"Epochs Assigned to this Task: {epochs_subset}")
print(f"----------------------")

# --- LOGGING SETUP: Define log file path and write header if new ---
# Append task_id to log file to prevent write conflicts
log_fp = results_out_dir + f"{str(experiment_number)}_performance_log_{valid_timestep[:10]}_task{task_id+1}.csv"
os.makedirs(results_out_dir, exist_ok=True) # Ensure dir exists for the log
if not os.path.exists(log_fp):
    with open(log_fp, "w") as f:
        f.write("Epoch,Total_s,Load_s,Infer_s,Save_s,GPU_Util,Peak_VRAM_GB,Timestamp\n")
print(f"Logging performance stats to: {log_fp}")

time_start = time.time()

# Outer loop = Epochs (Load model once), Inner loop = Init times
for n_epoch in epochs_subset: 
    # --- Epoch Start ---
    t_epoch_start = time.time()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    load_dotenv()

    # --- LOADING MODEL (Once per Epoch) ---
    if n_epoch < fine_tuning_start_epoch: 
        src_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/Checkpoints_SFNO/sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999/"
        checkpoint_name = 'ckpt_mp0_epoch'+str(n_epoch)+'.tar'
    else:
        src_dir = "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/Checkpoints_SFNO/multistep_sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999-multistep2/"
        n_epoch_multistep2 = n_epoch - (fine_tuning_start_epoch - 1) 
        checkpoint_name = 'ckpt_mp0_epoch'+str(n_epoch_multistep2)+'.tar'

    t_load_start = time.time()
    print(f"Loading model: {checkpoint_name}...")
    model_package = Package(src_dir, cache = False)
    model = SFNO.load_model(model_package, checkpoint_name = checkpoint_name, EMA = ema)
    t_load_end = time.time()
    
    # Initialize accumulators for timing across all inits for this epoch
    total_infer_dur = 0
    total_save_dur = 0
    
    # --- Run Inference for all Initializations ---
    for init_ind, start_timestep in enumerate(init_timesteps):
        
        # Create the inference name based on the start datetime and number of steps
        time_0 = time.time()
        n_steps = n_6h_steps[init_ind]
        start_datetime = datetime.fromisoformat(start_timestep)
        inference_name = start_datetime.strftime("%Y_%m_%dT%H")+'_nsteps'+str(n_steps)
        
        if ema:
            results_out_fp = results_out_dir+f"EMA_Checkpoint{n_epoch}_{inference_name}.nc"
        else:
            results_out_fp =  results_out_dir+"Checkpoint"+str(n_epoch)+"_"+inference_name+'.nc' 
        
        # Check if the results file already exists
        if os.path.exists(results_out_fp):
            print(f"Results file {results_out_fp} already exists. Skipping.")
            continue 
        else:
            os.makedirs(os.path.dirname(results_out_fp), exist_ok=True)
            
            # Prepare Initialization Data
            data_create_fp = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/init_files/Initialize_"+inference_name+".nc" 
            if not os.path.exists(data_create_fp):
                create_initialization_file(start_timestep=start_timestep, valid_timestep=valid_timestep, init_fp=data_create_fp, )
            initial_data = data.DataArrayFile(data_create_fp) 
            
            time_1 = time.time(); dt_1 = time_1 - time_0
            print(f"Initialization data ready for start time {start_timestep}.")

            io = ZarrBackend() # Temporary in-memory Zarr backend
            
            # --- INFERENCE ---
            t_infer_local_start = time.time()
            with torch.no_grad():
                io = deterministic([start_datetime], n_steps, model, initial_data, io, variables_list=variables_to_save)
            t_infer_local_end = time.time()
            total_infer_dur += (t_infer_local_end - t_infer_local_start)

            # --- SAVING --- 
            t_save_local_start = time.time()
            ds = xr.open_zarr(io.root.store)
            
            ds["time"] = ds["time"].astype("datetime64[ns]")

            base_time = ds["time"].values  
            lead_timedelta = ds["lead_time"].values.astype("timedelta64[ns]")  
            valid_timesteps_arr = (base_time[:, None] + lead_timedelta[None, :]).flatten() 
            ds = ds.drop_vars("lead_time")

            # Assume ds has dimensions (time, lead_time, lat, lon) and only one time
            initial_time = str(ds["time"].values[0])  
            ds = ds.isel(time=0).drop_vars("time")
            ds.attrs["initial_time"] = initial_time

            ds = ds.rename({"lead_time": "valid_time"})
            ds = ds.assign_coords(valid_time=(("valid_time",), valid_timesteps_arr))

            # Select desired valid time and save!
            if np.datetime64(valid_datetime) in ds["valid_time"].values:
                ds = ds.sel(valid_time=[valid_datetime])
                ds = ds[variables_to_save]
                ds.to_netcdf(results_out_fp, mode="w", format="NETCDF4")
                print(f"Results saved to {results_out_fp}")
            else:
                print(f"ERROR: Desired valid time {valid_datetime} not found in results valid_time coordinate.")

            t_save_local_end = time.time()
            total_save_dur += (t_save_local_end - t_save_local_start)
            
            # Cleanup per initialization
            del io; del ds; gc.collect()

    # --- Monitoring: Capture GPU stats after all inits for this epoch ---
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 # Convert to GB
    try:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            phys_id = cvd.split(",")[current_gpu]
        else:
            phys_id = str(current_gpu)
        
        gpu_stats = subprocess.check_output(
            ["nvidia-smi", f"--id={phys_id}", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 
            encoding='utf-8'
        ).strip()
        gpu_util = f"{gpu_stats}%"
    except Exception as e:
        gpu_util = f"Err: {e}"
        
    # Cleanup Model (Once per epoch)
    torch.cuda.empty_cache()
    del model_package
    del model
    gc.collect()
    
    # --- Monitoring: Final timing prints ---
    load_dur = t_load_end - t_load_start
    total_dur = time.time() - t_epoch_start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print to Notebook
    print(f"\n📊 Epoch {n_epoch} Analysis (All Initializations):")
    print(f"   Total Time: {total_dur:.2f}s")
    print(f"   ├── 📂 Loading:   {load_dur:.2f}s ({load_dur/total_dur:.0%})")
    print(f"   ├── 🚀 Inference: {total_infer_dur:.2f}s ({total_infer_dur/total_dur:.0%})")
    print(f"   └── 💾 Saving:    {total_save_dur:.2f}s ({total_save_dur/total_dur:.0%})")
    print(f"   GPU Util: {gpu_util} | Peak Mem: {peak_mem:.2f} GB")
    print("-" * 60 + "\n")

    # Write to CSV Log
    with open(log_fp, "a") as f:
        f.write(f"{n_epoch},{total_dur:.2f},{load_dur:.2f},{total_infer_dur:.2f},{total_save_dur:.2f},{gpu_util},{peak_mem:.2f},{timestamp}\n")
    # -----------------------------------