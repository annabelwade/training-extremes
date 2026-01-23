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

## Configuration options

## 1. Experiment Setup (`experiment_setup`)

| Key | Type | Description | Options / Examples |
| :--- | :--- | :--- | :--- |
| **`event_type`** | `string` | Defines the specific extreme event category. This determines default variable selections in the code logic. | `"atmospheric_river"`, `"tropical_cyclone"`, `"heat_wave"`, `"severe_convection"` |
| **`valid_timestep`** | `string` | The target datetime to forecast for (ISO 8601 format). | `"2022-12-27T00:00:00"` |
| **`leadtimes_days`** | `list[int]` | The number of days *prior* to the valid timestep to initialize the model. | `[3, 5, 7]` (Forecasts initialized 3, 5, and 7 days before the event) |
| **`variables_to_save`** | `list[str]` | Specific variables to save to the output NetCDF to save space/time. | `["tcwv", "u700", "v700", "z500"]`, `["msl", "u10m", "v10m"]` |
| **`ema`** | `bool` | Whether to load **Exponential Moving Average** weights or standard checkpoint weights. | `true` (use EMA), `false` (standard) |
| **`compute_ivt`** | `bool` | Flag to calculate Integrated Vapor Transport (IVT) during inference. | `true` or `false` |

---

## 2. Model Parameters (`model_parameters`)

| Key | Type | Description | Options / Examples |
| :--- | :--- | :--- | :--- |
| **`fine_tuning_start_epoch`** | `int` | The epoch number where the directory structure switches from pre-training to fine-tuning. | `71` |
| **`epochs_to_run`** | `str` or `list` | Determines which checkpoints to process. | `"all"` (1-90)<br>`"odds"` (1, 3, 5...)<br>`"evens"` (2, 4, 6...)<br>`[10, 20, 85]` (Specific list) |

---

## 3. Paths (`paths`)

| Key | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`base_output_dir`** | `string` | The root directory where log and forecast output will be saved. | `"/projectnb/eb-general/wade/sfno/inference_runs/sandbox"` |

---

#### 📝 Example Configuration
```json
{
    "experiment_setup": {
        "event_type": "atmospheric_river",
        "valid_timestep": "2022-12-27T00:00:00",
        "leadtimes_days": [3, 5, 7],
        "variables_to_save": ["tcwv", "u700", "v700", "z500"],
        "ema": false,
        "compute_ivt": true
    },
    "model_parameters": {
        "fine_tuning_start_epoch": 71,
        "epochs_to_run": "evens"
    },
    "paths": {
        "base_output_dir": "/projectnb/eb-general/wade/sfno/inference_runs/sandbox"
    }
}
```
### Variable options: 
[
    "u10m","v10m", "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
]

