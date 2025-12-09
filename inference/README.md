# Running inference using Earth2Studio

### environment setup *specific to BU SCC*
As recommended from https://nvidia.github.io/earth2studio/v/0.10.0/userguide/about/install.html#install-using-uv-recommended:
```
conda create -n earth2studio_new python=3.12
conda activate earth2studio_new
pip install uv
export UV_CACHE_DIR="/projectnb/eb-general/wade/uv_cache"
mkdir earth2studio-project && cd earth2studio-project
uv init --python=3.12
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.10.0"
uv add earth2studio --extra fcn
```
What I ran: (nearly identical, just different location of my uv project)
```
conda create -n earth2studio_new python=3.12
conda activate earth2studio_new
pip install uv
export UV_CACHE_DIR="/projectnb/eb-general/wade/uv_cache"
uv init --python=3.12
uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.10.0"
uv add earth2studio --extra fcn
```

- additional dependencies for specific models: https://nvidia.github.io/earth2studio/v/0.10.0/userguide/about/install.html#prognostics 
- the UV cache directory resets to be the home directory on the BU SCC after each session ends, so you may want to add the export UV_CACHE_DIR line to your .bashrc file

### Directory structure
```
inference/
  ├── old_package_versions/  
```

old_package_versions/
- contains code compatible with earth2studio 0.7.0 and makani 0.1.0

