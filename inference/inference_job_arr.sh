#!/bin/bash -l

# Set the time limit 
#$ -l h_rt=03:00:00

# Name the job
#$ -N inference_parallel

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

# Specify output file naming
#$ -o /projectnb/eb-general/wade/sfno/inference/logs/inference_parallel_$TASK_ID.log

#$ -P eb-general             # Specify the SCC project name you want to use
#$ -m ea                    # Send email when job ends or aborts

module load miniconda 
conda activate earth2studio

cd /projectnb/eb-general/wade/sfno/inference/

# Run the python script
python ./inference.py $1