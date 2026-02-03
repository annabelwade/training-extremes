#!/bin/bash -l

# Set the time limit 
#$ -l h_rt=03:00:00

# Name the job
#$ -N inference_parallel

# Request the buyin queue
#$ -l buyin

# Request 1 GPU per task
#$ -l gpus=1
#$ -l gpu_c=7.0  

# Request 4 cores, each with 8GB memory
#$ -pe omp 4
#$ -l mem_per_core=8G

# Request 4 tasks (Job Array 1-4)
#$ -t 1-4

# Merge output and error files
#$ -j y

#$ -P eb-general             # Specify the SCC project name you want to use
#$ -m ea                    # Send email when job ends or aborts

# Get the experiment number from the command line argument
EXP_NUM=$1

# Define the log directory and file name
LOG_DIR="/projectnb/eb-general/wade/sfno/inference/logs/Experiment${EXP_NUM}"
LOG_FILE="${LOG_DIR}/inference_parallel_${JOB_ID}_${SGE_TASK_ID}.log"

# Create the directory if it doesn't exist
mkdir -p $LOG_DIR

exec > $LOG_FILE 2>&1

# Environment Setup
module load miniconda 
conda activate earth2studio

cd /projectnb/eb-general/wade/sfno/inference/
python ./inference.py $EXP_NUM