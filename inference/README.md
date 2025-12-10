# Running inference using Earth2Studio

This folder contains inference scripts for running SFNO forecasts with earth2studio. The `new_package_versions/Inference_Example.py` script mirrors the legacy `old_package_versions` workflow while targeting earth2studio 0.10.x.

## Environment setup *specific to BU SCC*
```
conda create -n earth2studio python=3.12 -y 
conda activate earth2studio
pip install uv
export UV_CACHE_DIR="/projectnb/eb-general/wade/uv_cache"
uv pip install "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.10.0"
uv pip install "earth2studio[fcn]"
uv pip install numpy matplotlib pandas xarray cartopy cmocean tqdm
# Run this to check earth2studio wasn't installed in the home directory
python -c "import earth2studio; print('Found at:', earth2studio.__file__)"
uv pip install "makani @ git+https://github.com/NVIDIA/modulus-makani.git@28f38e3e929ed1303476518552c64673bbd6f722"
uv pip install earth2studio[sfno]
```
- additional dependencies for specific models: https://nvidia.github.io/earth2studio/v/0.10.0/userguide/about/install.html#prognostics 
- the UV cache directory resets to be the home directory on the BU SCC after each session ends, so you may want to add the export UV_CACHE_DIR line to your .bashrc file with:
```
echo 'export UV_CACHE_DIR="/projectnb/eb-general/wade/uv_cache"' >> ~/.bashrc
source ~/.bashrc
echo $UV_CACHE_DIR # to verify it worked!
```

## Running Inference_Example 

### without CLI flags
Edit the `DEFAULT_CONFIG` block near the top of `new_package_versions/Inference_Example.py` to set your start time, checkpoint location, output path, and variable list. Once those values are saved, launch the script without any flags (e.g., inside a batch script):

```bash
python Example_Code/inference/new_package_versions/Inference_Example.py
```

Argparse will see no command-line inputs and will run entirely from `DEFAULT_CONFIG`, so you do not need to type flags each time.

### option: with CLI overrides
If you prefer to pass arguments explicitly, any flag you provide will override the corresponding entry in `DEFAULT_CONFIG`:

```bash
python Example_Code/inference/new_package_versions/Inference_Example.py \
  --start 2019-03-22T00:00:00 \
  --steps 12 \
  --init-data /path/to/Initialize_2019_03_22T00_nsteps12.nc \
  --checkpoint-dir /path/to/checkpoints \
  --checkpoint-name ckpt_mp0_epoch1.tar \
  --output /path/to/output/forecast.nc \
  --variables msl tcwv \
  --ema
```

### Outputs
The script loads the initial NetCDF, runs inference, and writes the final forecast step to the requested NetCDF path (creating parent directories if needed).


### Notes
- still need to setup code for generating initialization files, but for now will work with the initializtion files Becca provided, located in `/projectnb/eb-general/wade/sfno/inference_runs/Ian/Initialize_data`
