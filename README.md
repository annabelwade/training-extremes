# training-extremes
Code for project on exploring how AI weather models learn to predict extreme weather events.

### Environment setup
```
conda create -n earth2studio python=3.12
conda activate earth2studio
pip install uv
pip cache dir # I got ERROR: pip cache commands can not function since cache is disabled.
uv cache dir # I got /usr3/graduate/wade/.cache/uv so will need to point the uv cache dir to project dir
UV_CACHE_DIR="/projectnb/eb-general/wade/uv_cache" uv cache dir # check where the uv cache dir is now
uv pip install -p /projectnb/eb-general/wade/.conda/envs/earth2studio/bin/python "earth2studio@git+https://github.com/NVIDIA/earth2studio.git@0.7.0"
git clone https://github.com/NVIDIA/makani.git
cd makani
pip install -e .
pip install matplotlib
pip install geopy
pip install scipy
pip install cartopy
pip install jsbeautifier
pip install moviepy
pip install ruamel.yaml
pip install --no-deps nvidia-modulus
conda install ipykernel
```
