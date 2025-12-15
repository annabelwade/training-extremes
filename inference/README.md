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

Notes:
- from the SFNO installation instructions: makani v0.2.0 is installed from the specific commit even though earth2studio v0.10.0 is compatible with makani 0.2.1. Keep this in mind, since we have to edit `makani/models/model_package.py` for checkpoint selection, and I noticed that this function changed between makani 0.2.0 and 0.2.1 (see comparison here: https://github.com/NVIDIA/makani/compare/v0.2.0...v0.2.1)

