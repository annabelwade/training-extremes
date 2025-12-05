#!/bin/bash
#$ -S /bin/bash

# Run these commands to check how many nodes my job could run on given two versions of resource requests:
#    qselect -l h_rt=12:00:00 -l mem_per_core=18G -pe omp 16 -l gpu_c=7 -l gpus=1 -U wade | wc -l
#    qselect -l h_rt=12:00:00 -l mem_per_core=28G -pe omp 12 -l gpu_c=7 -l gpus=1 -U wade | wc -l

# Total Memory Request: 256GB.
#    Requesting 16 cores (slots) with 18G/core guarantees 288GB total memory on a single node.
#    Requesting 16 cores (slots) with 28G/core guarantees 448GB total memory on a single node.
#$ -pe omp 16 
#$ -l mem_per_core=18G

# GPU Request
#$ -l gpus=1
#$ -l gpu_c=7

#  Run Time Limit
#$ -l h_rt=12:00:00 

# Job Name and Output Files
#$ -N DL_GPU_Job
#$ -o output.$JOB_ID
#$ -e error.$JOB_ID
#$ -P eb-general             # Specify the SCC project name you want to use
#$ -m ea                    # Send email when job ends or aborts
#$ -N inference_run0             # Give job a name **edit this**

module load miniconda 
conda activate dl
module load pytorch/1.13.1

cd /projectnb/eb-general/wade

echo "=== Node & Resource Diagnostics ==="
echo "Node Hostname: $(hostname)"
echo "Start Time:    $(date)"
echo "Using $NSLOTS CPU cores."

# Verifies total memory available on the node and current usage buffers.
free -h

# Shows the specific GPU ID assigned to YOU, Driver version, and VRAM.
echo "--- GPU Allocation ---"
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,memory.total,memory.free,memory.used --format=csv

# Ensures PyTorch actually sees the device.
echo "--- PyTorch Visibility ---"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Current Device: {torch.cuda.get_device_name(0)}')"

echo "=================================="

# Define a log file for resource usage
LOG_FILE="resource_usage_${JOB_ID}.log"

echo "Timestamp, CPU_User%, MEM_Used_MB, GPU_Util%, GPU_Mem_MB" > $LOG_FILE

# Logs GPU stats every 120 seconds to a file
nvidia-smi dmon -s put -d 120 -o T > gpu_metrics_${JOB_ID}.txt & 
#     The ampersand runs this subshell in the background

### ADD .PY SCRIPT HERE ###

