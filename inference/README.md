# Running inference using Earth2Studio

This folder contains inference scripts for running SFNO forecasts with earth2studio 0.10.0.

## Environment setup *specific to BU SCC*
```
conda create -n earth2studio python=3.12 -y 
conda activate earth2studio
pip install uv
export UV_CACHE_DIR="/projectnb/eb-general/wade/uv_cache"
uv pip install "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.10.0"
uv pip install "earth2studio[fcn]"
uv pip install numpy matplotlib pandas xarray cartopy cmocean tqdm
uv pip install "makani @ git+https://github.com/NVIDIA/modulus-makani.git@28f38e3e929ed1303476518552c64673bbd6f722"
uv pip install earth2studio[sfno]
```
- Run this to check earth2studio wasn't installed in the home directory
```
python -c "import earth2studio; print('Found at:', earth2studio.__file__)"
```

- the UV cache directory resets to be the home directory on the BU SCC after each session ends, so you may want to add the export UV_CACHE_DIR line to your .bashrc file with:
```
echo 'export UV_CACHE_DIR="/projectnb/eb-general/wade/uv_cache"' >> ~/.bashrc
source ~/.bashrc
echo $UV_CACHE_DIR # to verify it worked!
```

## Directory contents:
##### `deterministic_update.py`

- rewrites the earth2studio deterministic function to handle saving only specific variables in the output.

##### `SFNO_update.py`
- rewrites the earth2studio SFNO code (and some relevant makani code) to handle loading specific SFNO checkpoints and running inference with them.

##### `utils.py`
- helper functions primarily used for creating Initialization files, and opening files.

##### `inference.py`
- main script to run SFNO inference with earth2studio.

#### `inference_job_arr.sh`
- uses a job array to parallelize inference runs on BU SCC.

#### `configs/`
- contains .json config files for different experiments and inference runs.

## Running inference
To run inference, submit the `inference_job_arr.sh` script with a specified experiment number:
```
qsub inference_job_arr.sh <experiment_number>
```
- Ensure that the experiment number you provide corresponds to a valid configuration file with the desired settings in at `configs/expN.json` where `N` is the experiment number.
- Not providing a number will default to experiment 2 (`configs/exp2.json`).

