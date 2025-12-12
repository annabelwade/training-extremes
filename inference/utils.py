import os
from datetime import datetime, timedelta
from typing import List
import xarray as xr
import numpy as np
import pandas as pd
import re
from calendar import monthrange 
from math import cos, sqrt
#
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Iterable, Sequence

def filename_to_year(path: str) -> int:
    # Extracts the year from the filename, assuming the filename starts with a 4-digit year.
    filename = os.path.basename(path)
    return int(filename[:4])

def datetime_range(
    year: int, time_step: timedelta, n: int
) -> List[datetime]:
    # Generates a list of datetime objects starting from January 1st of the given year,
    initial_time = datetime(year=year, month=1, day=1)
    return [initial_time + time_step * i for i in range(n)]

def open_hdf5(*, path, f=None, metadata):
    # For creating initialization data in the correct format. 
    #   Opens an HDF5 file and returns an xarray Dataset.
    h5_path = metadata["h5_path"]
    dims = metadata["dims"]
    time_step_hours = metadata.get("dhours", 6)
    time_step = timedelta(hours=time_step_hours)

    ds = xr.open_dataset(f or path, engine="h5netcdf", phony_dims="sort")
    array = ds[h5_path]
    ds = array.rename(dict(zip(array.dims, dims)))
    year = filename_to_year(path)
    n = array.shape[0]
    ds = ds.assign_coords(
        time=datetime_range(year, time_step=time_step, n=n), **metadata["coords"]
    )
    ds = ds.assign_attrs(metadata["attrs"], path=path)
    return ds


@dataclass
class InferenceConfig:
    """Configuration container for running the inference script."""

    start: str = "2019-09-03T00:00:00" # fix that this has to be a different format from the initialize time string
    steps: int = 4
    init_data: Path = Path(
        "/projectnb/eb-general/wade/sfno/inference_runs/Ian/Initialize_data/Initialize_2019_09_03T00_nsteps4.nc"
    )
    checkpoint_dir: Path = Path(
        "/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/Checkpoints_SFNO/"
        "sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999/"
    )  # TODO CHECK IF THIS PATH NEEDS TO HAVE ADDITIONAL DIRECTORY TO THE MULTISTEP OR NON MULTISTEP DIR + seed + training_checkpoints
    checkpoint_name: str = "ckpt_mp0_epoch1.tar"  # for now, the code will just grab best ckpt, not this one.
    output: Path = Path(
        "/projectnb/eb-general/wade/sfno/inference_runs/sandbox/best_ckpt_2019_09_04T00_nsteps4.nc"
    )  # todo make this automated to just take the directory, not the full path+name

    variables: list[str] | None = None
    ema: bool = False

def _parse_args(defaults: InferenceConfig, cli_args: Sequence[str] | None = None) -> InferenceConfig:
    """Parse CLI arguments while honoring editable in-file defaults.

    Parameters
    ----------
    defaults : InferenceConfig
        Baseline values to use when a flag is omitted. Edit ``DEFAULT_CONFIG``
        to change these defaults without typing flags.
    cli_args : Sequence[str] | None
        Argument vector to parse. When ``None`` (the common case), argparse
        inspects ``sys.argv``. Passing an explicit sequence is useful for
        programmatic invocation.
    """

    parser = argparse.ArgumentParser(description="Run SFNO inference with earth2studio 0.10.x")
    parser.add_argument(
        "--start",
        default=defaults.start,
        help="ISO8601 start datetime for the forecast (e.g. 2019-03-22T00:00:00)",
    )
    parser.add_argument("--steps", type=int, default=defaults.steps, help="Number of 6-hour steps to forecast")
    parser.add_argument(
        "--init-data", type=Path, default=defaults.init_data, help="Path to the preprocessed initial state NetCDF"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=defaults.checkpoint_dir,
        help="Directory containing the SFNO checkpoints",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=defaults.checkpoint_name,
        help="Checkpoint file name inside the checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.output,
        help="Output NetCDF path for the final forecast timestep",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=defaults.variables,
        help="Optional list of variables to keep (defaults to all variables)",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=defaults.ema,
        help="Load EMA weights instead of the standard checkpoint",
    )

    args = parser.parse_args(args=cli_args)
    args_dict = vars(args)
    if args_dict.get("variables") == []:
        args_dict["variables"] = None

    return InferenceConfig(**args_dict)
    


