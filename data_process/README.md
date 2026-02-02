## Generating .h5 files for inputting to SFNO model
This version of SFNO has 74 channels instead of 73, with the 74th being 2m dewpoint temperature (d2m). To generate .h5 files with this additional channel for testing the model.

### Environment Setup

##### This is the troubleshooting steps I took to get the environment working:
```
conda create -n makani_data python=3.12 -y
conda activate makani_data
pip install uv
export UV_CACHE_DIR="/projectnb/eb-general/wade/uv_cache"
uv pip install "makani @ git+https://github.com/NVIDIA/modulus-makani.git@28f38e3e929ed1303476518552c64673bbd6f722"
uv pip install numpy mpi4py xarray zarr gcsfs h5py tqdm
uv pip install progressbar2 scipy 
uv pip install "ruamel.yaml" "moviepy" 
uv pip install "zarr>=2.18,<3"
uv pip install dask distributed
uv pip uninstall h5py mpi4py
conda install -y -c conda-forge "h5py=*=mpi*" mpi4py
``` 
dependencies;
- mpi4py: for parallel processing in the script
- gcsfs, zarr, xarray: for reading the Google Cloud data
- h5py: for writing the output files
<!-- "einops" "nvidia-ml-py" "wandb" "tensorboard" "timm" "imageio" -->

### Files
- `wb2_helpers.py`: adapted from `nvidia/makani/makani-main/data_process/wb2_helpers.py` to include d2m channel. This script maps tthe metadata channel names to ERA5 var names.
- `convert_wb2_to_makani_input.py`:  from `nvidia/makani/makani-main/data_process/convert_wb2_to_makani_input.py`.

### Commands for processing data
```
module load openmpi
cd /projectnb/eb-general/wade/sfno/data_process/
mpirun -n 3 python convert_wb2_to_makani_input.py \
    --input_file "gs://weatherbench2/datasets/era5/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2" \
    --output_dir "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/ERA5_SFNO/testing_data" \
    --metadata_file "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/ERA5_SFNO/testing_data/data.json" \
    --years 2018 2020 2023 \
    --batch_size 32 \
    --impute_missing_timestamps
```
- Adjust -n N to the number of CPU cores you want to use
- batch_size can be adjusted based on available memory
- impute_missing_timestamps flag will fill in any missing timestamps with NaNs

```
module load openmpi
cd /projectnb/eb-general/wade/sfno/data_process/
mpirun -n 3 python convert_wb2_to_makani_input.py \
    --input_file "gs://weatherbench2/datasets/era5/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2" \
    --output_dir "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/ERA5_SFNO/testing_data/troubleshooting/" \
    --metadata_file "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/ERA5_SFNO/testing_data/data.json" \
    --years 2018 2020 \
    --batch_size 32 \
    --impute_missing_timestamps
```


#### To check what ARCO-ERA5 data is available on weatherbench2
```
python -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='anon')
files = fs.ls('weatherbench2/datasets/era5/')
print('--- Available ERA5 Datasets ---')
for f in files:
    print(f)
"
```
- as of 1/26/25, only upto 2023 is available