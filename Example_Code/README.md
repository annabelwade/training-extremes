RESOURCES
-I took a lot of this code from https://nvidia.github.io/earth2studio/#, NVIDIA's open source software for running models
-Some of this was also adapted from https://nvidia.github.io/earth2mip/, which is the older version

FILE SYSTEM ORGANIZATION
(not necessarily recommended, just to help you navigate code)
    -Project File
        -ERA5_SFNO
        -Checkpoints_SFNO
        -Experiment n 
            -Initialize data
            -Forecast
            -Forecasts_Boring

FILES INCLUDED
environment.yml
    -yaml file for environment "earth2studio" 
Create Initial data
    -Create netcdf files with initialize date and forecast date data to use in training and analysis
Inference_Single.py
    -Runs a single SFNO inference for 90 checkpoints in 3 parallel chunks
    -Required: before running inference you need to create initial data
Inference_Single.script
    -Runs 3 parallel slurm tasks
    -Requires 1 gpu for each task 
    -If you are trying to run a bunch of forecasts, I would use parallelization to run one forecast date per gpu
utils.py
    -Some functions for opening data
deterministic_update.py
    -Copied functionality from earth2studio but modified to save only the variables you want- this change sped up runtime ALOT
SFNO_update.py
    -Copied functionality from earth2studio but modified to include input of which checkpoint to run (including EMA checkpoints)