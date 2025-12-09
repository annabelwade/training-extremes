# Inference examples

This folder contains inference scripts for running SFNO forecasts with earth2studio. The `new_package_versions/Inference_Example.py` script mirrors the legacy `old_package_versions` workflow while targeting earth2studio 0.10.x.

## Running without CLI flags (batch-friendly)
Edit the `DEFAULT_CONFIG` block near the top of `new_package_versions/Inference_Example.py` to set your start time, checkpoint location, output path, and variable list. Once those values are saved, launch the script without any flags (e.g., inside a SLURM `sbatch` script):

```bash
python Example_Code/inference/new_package_versions/Inference_Example.py
```

Argparse will see no command-line inputs and will run entirely from `DEFAULT_CONFIG`, so you do not need to type flags each time.

## Running with CLI overrides
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

## How argument parsing works
- `DEFAULT_CONFIG` holds the in-code defaults that mirror the original batch-oriented script.
- When the script starts, it checks for a passed `InferenceConfig`; otherwise, argparse consumes the CLI flags.
- Any missing flags fall back to the values in `DEFAULT_CONFIG`, so you can combine both approaches (for example, keep all defaults but change `--checkpoint-name` on the command line).

## Outputs
The script loads your initial NetCDF, runs deterministic inference, converts earth2studio `time` and `lead_time` into usable `valid_time` coordinates, and writes the final forecast step to the requested NetCDF path (creating parent directories if needed).
