#### Disable loguru and tqdm outputs globally
from loguru import logger; import sys
logger.remove()
logger.add(sys.stderr, level="ERROR")
from tqdm import tqdm; from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

#### Imports
import os
import subprocess
from dotenv import load_dotenv
from earth2studio.io import ZarrBackend
from deterministic_update import deterministic
from SFNO_update import SFNO
import earth2studio.data as data
from earth2studio.models.auto import Package
from utils import get_sequential_initializations, compute_2nn_id #filename_to_year, datetime_range, open_hdf5 # these aren't used in this script currently
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
import time

if os.environ.get('JOB_ID') is not None:
    JOB_ID = os.environ.get('JOB_ID')
else:
    JOB_ID = None
    print("WARNING: JOB_ID not found in environment variables. Defaulting to None.")

### CUDA Setup
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

### CONFIGURATIONS
if len(sys.argv) > 1:
    # If provided a number (e.g., "3"), use it as the experiment number
    experiment_number = int(sys.argv[1])
else:
    print("No experiment number provided as argument. Defaulting to Experiment 1.")
    experiment_number = 1 

config_path = f'./configs/exp{experiment_number}.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Parse Experiment Setup
exp_params = config['experiment_setup']
event_type = exp_params['event_type']
variables_to_save = exp_params['variables_to_save'] 
valid_timestep = exp_params['valid_timestep']
leadtimes = exp_params['leadtimes_days']
n_steps_list = [leadtime * 4 for leadtime in leadtimes] # Convert days to 6-hourly steps
ema = exp_params['ema']
# if compute_ivt is a parameter in experiment setup, set the variable here
compute_ivt = False 
if 'compute_ivt' in exp_params:
    compute_ivt = exp_params['compute_ivt']
bounding_box = {}
if 'bounding_box' in exp_params:
    bounding_box = exp_params['bounding_box']
    # convert the values to be float 
    for key in bounding_box:
        bounding_box[key] = float(bounding_box[key])

# Parse ID params
id_params = exp_params['intrinsic_dimension']
n_samples = id_params['n_samples']
data_years = id_params['data_years']
target_layers = id_params['layers_of_interest'] # "all" or list like ["encoder", "blocks.0"]
    
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

# Directories
path_params = config['paths']
base_output_dir = path_params['base_output_dir']
results_out_dir = f"{base_output_dir}/Experiment{str(experiment_number)[0]}/{valid_timestep[:10].replace('-', '_')}/"
# make experiment directory if it DNE
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

### PARALLELIZATION
# SGE_TASK_ID is 1-indexed 
# check if sge task id is available in environment variables, if not, default to 1 task (running all epochs in one go)
task_id_env = os.environ.get('SGE_TASK_ID')
print(f"SGE_TASK_ID from environment: {task_id_env}") # Diagnostic print to check the value of SGE_TASK_ID

if task_id_env is None or not task_id_env.isdigit():
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
logs_dir = os.path.join(os.getcwd(),'logs',f'Experiment{str(experiment_number)}')
os.makedirs(results_out_dir, exist_ok=True) # Ensure dir exists for the log
csv_out = os.path.join(logs_dir, f"id_results_job{JOB_ID}_task{task_id+1}.csv")
if not os.path.exists(csv_out):
    with open(csv_out, "w") as f:
        f.write("Epoch,Layer,N,IntrinsicDim,ID_Time,Epoch_Time\n")
print(f"Logging intrinsic dimension results to: {csv_out}")

figs_dir = os.path.join("/projectnb/eb-general/wade/sfno/inference/viz/figures/", f'Experiment{str(experiment_number)}')
os.makedirs(figs_dir, exist_ok=True)

# Get initialization file paths and start timesteps for the specified years and number of samples
if task_id == 0:
    # Task 0 does the heavy lifting and makes the data files if they don't exist
    init_fps, start_timesteps = get_sequential_initializations(data_years, n_samples)
else:
    # Other tasks wait for Task 0 to finish creating files, 3 min as a large buffer. to avoid permission errors!
    time.sleep(3*60) 
    # Now they just read the paths, assuming files exist
    init_fps, start_timesteps = get_sequential_initializations(data_years, n_samples)

print(f"Input Samples: {len(init_fps)}, first one: {init_fps[0]}")
print(f"Target Layers: {target_layers}")

t_script_start = time.time()

# Outer loop = Epochs, Inner loop = Init times
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
    model_package = Package(src_dir, cache=False)
    model_wrapper = SFNO.load_model(model_package, checkpoint_name=checkpoint_name, EMA=ema)
    t_load_end = time.time()

    # --- REGISTER HOOKS TO LAYER(S) ---
    activation_storage = {}
    hooks = []

    def get_activation(name):
        def hook(module, input, output):
            # Move to CPU immediately to save VRAM
            if isinstance(output, tuple):
                print(f"!!! Warning: Output of layer {name} is a tuple with shapes {[o.shape for o in output]}. Only the first tuple element will be accessed by the hook. !!!")
                data = output[0].detach().cpu()
            else:
                data = output.detach().cpu()
            
            if name not in activation_storage:
                activation_storage[name] = []
            activation_storage[name].append(data)
        return hook

    print("  Registering hooks...")
    hooked_layers = []

    # Dynamically search through every nested module in the wrapper
    for name, module in model_wrapper.named_modules():
        
        # Target "all" layers
        if target_layers == "all":
            # We want the encoder, decoder, and the individual 8 blocks (not the ModuleList itself)
            if name.endswith("encoder") or name.endswith("decoder") or (
                "blocks." in name and type(module).__name__ == 'FourierNeuralOperatorBlock'
            ):
                clean_name = name.split('.')[-2] + '.' + name.split('.')[-1] if 'blocks.' in name else name.split('.')[-1]
                hooks.append(module.register_forward_hook(get_activation(clean_name)))
                hooked_layers.append(clean_name)
                print(f"    Hooked to: {name} as {clean_name}")
                
        # Target specific layers (e.g., ["blocks.7"])
        else:
            for target in target_layers:
                if name.endswith(target):
                    clean_name = target
                    hooks.append(module.register_forward_hook(get_activation(clean_name)))
                    hooked_layers.append(clean_name)
                    print(f"    Hooked to: {name} as {clean_name}")

    # --- INFERENCE LOOP ---
    print(f"Running inference on {len(init_fps)} samples...")
    for idx, init_fp in enumerate(init_fps):
        
        # Load Data
        initial_data = data.DataArrayFile(init_fp)
        io = ZarrBackend()
        # filepath format is: /projectnb/eb-general/wade/sfno/inference_runs/intrinsic_dim/init_files/Initalialize_2022_01_01T06_nsteps1.nc
        filename = os.path.basename(init_fp)
        start_timestep = start_timesteps[idx]
        start_datetime = datetime.fromisoformat(start_timestep)
        # print('SANITY CHECK: Filename:', filename, '| Start datetime:', start_datetime, '| Start timestep:', start_timestep)

        # Run (Hooks capture data)
        n_steps = int(n_steps_list[0]) #TODO: ADD MULTIPLE LEADTIMES FUNCTIONALITY
        # print('SANITY CHECK: n_steps:', n_steps)
        with torch.no_grad():
            deterministic([start_datetime], n_steps, model_wrapper, initial_data, io, variables_list=None) #TODO: ADD MULTIPLE LEADTIMES FUNCTIONALITY, WITH CORRECT MONITORING OF THE HOOKS
        
        # Cleanup
        del io; del initial_data; gc.collect()
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(init_fps)}")

    # --- COMPUTE ID ---
    print("Computing Intrinsic Dimension...")
    t_id_start = time.time() # Start timing ID computation
    results_buffer = []
    
    for layer_name in sorted(activation_storage.keys()):
        tensor_list = activation_storage[layer_name]
        
        try:
            stacked = torch.cat(
                tensor_list,
                dim=0 # first dim is sample, stack along this
            ).numpy()
        except RuntimeError as e:
            print(f"  Error stacking {layer_name} (OOM?) error: {e}")
            del tensor_list; activation_storage[layer_name] = [] 
            continue
            
        # DIAGNOSTIC PRINT 1: Original Shape
        print(f"  [{layer_name}] Original stacked shape: {stacked.shape}")
            
        N = stacked.shape[0]
        flattened = stacked.reshape(N, -1)
        
        # DIAGNOSTIC PRINT 2: Flattened Shape
        print(f"  [{layer_name}] Flattened shape: {flattened.shape}")
        
        # Free up memory immediately after flattening
        del stacked; 
        activation_storage[layer_name] = []
        gc.collect()
        
        try:
            slope, intercept, r_value, p_value, std_err = compute_2nn_id(
                representations=flattened,
                task_id=task_id+1,
                job_id=JOB_ID,
                epoch=n_epoch, 
                layer_name=layer_name,
                figs_dir=figs_dir, 
                data_range=(str(start_timesteps[0]), str(start_timesteps[-1])), # strings for plot title
                nsteps=n_steps_list[0],
                init_timesteps=start_timesteps
            )
            print(f"\n~~~~~~ {layer_name}: ID = {slope:.4f} ~~~~~~\n")
            t_epoch = time.time() - t_epoch_start; t_id = time.time() - t_id_start;
            results_buffer.append(f"{n_epoch},{layer_name},{N},{slope:.4f}, {t_id:.2f}, {t_epoch:.2f}")
        except Exception as e:
            print(f"!!! Error calculating ID for {layer_name}: {e} !!!")
        
        del flattened; gc.collect()

    # --- SAVE TO CSV ---
    with open(csv_out, "a") as f:
        for line in results_buffer:
            f.write(line + "\n")

    # --- CLEANUP EPOCH ---
    for h in hooks: h.remove()
    del model_wrapper; gc.collect()
    activation_storage.clear() # Empty the dict
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    
