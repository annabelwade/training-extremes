"""Example inference runner compatible with earth2studio 0.10.x.

This script mirrors the workflow from the legacy
``inference/old_package_verisons/Inference_Example.py`` but updates the
imports and I/O handling to align with the newer earth2studio API. It loads
an SFNO checkpoint, runs deterministic inference starting from a provided
initial state, and writes the requested variables for the final forecast time
step to NetCDF.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import torch
import xarray as xr
from earth2studio.data import DataArrayFile
from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package

# # start with basic inference -- no edits to backend code 
# # - using best ckpt and save all vars#

# # TODO ONCE RUNNING INFERENCE - CHECK HOW THESE FUNCTIONS HAVE CHANGED IN THE NEW API, AND WHETHER OUR EDITS NEED TO BE CHANGED 
# from deterministic_update import deterministic # vars -- less important
# from SFNO_update import SFNO  # checkpoint access -- most important
# take really diligent notes -- specific package versions
# - get a yaml of the env with versions!!!


@dataclass
class InferenceConfig:
    """Configuration container for running the inference script."""

    start: str
    steps: int
    init_data: Path
    checkpoint_dir: Path
    checkpoint_name: str
    output: Path
    variables: list[str] | None
    ema: bool


# Edit these defaults to set preferred settings when launching the
# script without CLI arguments (e.g., from a batch script).
DEFAULT_CONFIG = InferenceConfig(
    start="2019-03-22T00:00:00",
    steps=12,
    init_data=Path("/projectnb/eb-general/wade/sfno/inference_runs/Ian/Initialize_data/Initialize_2019_08_27T00_nsteps20.nc"),
    checkpoint_dir=Path("/projectnb/eb-general/shared_data/data/processed/FourCastNet_sfno/Checkpoints_SFNO/sfno_linear_74chq_sc3_layers8_edim384_dt6h_wstgl2/v0.1.0-seed999/training_checkpoints/"), # TODO CHECK IF THIS PATH NEEDS TO HAVE ADDITIONAL DIRECTORY TO THE MULTISTEP OR NON MULTISTEP DIR + seed + training_checkpoints
    checkpoint_name="ckpt_mp0_epoch1.tar",
    output=Path("/projectnb/eb-general/wade/sfno/inference_runs/sandbox/"), # todo  fill in with desired output path?? or make it just output to dir + filename
    variables=['msl'],
    ema=False,
)


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
    parser.add_argument("--init-data", default=str(defaults.init_data), help="Path to the preprocessed initial state NetCDF")
    parser.add_argument("--checkpoint-dir", default=str(defaults.checkpoint_dir), help="Directory containing the SFNO checkpoints")
    parser.add_argument(
        "--checkpoint-name",
        default=defaults.checkpoint_name,
        help="Checkpoint file name inside the checkpoint directory",
    )
    parser.add_argument(
        "--output",
        default=str(defaults.output),
        help="Output NetCDF path for the final forecast timestep",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=defaults.variables,
        help="Optional list of variables to keep (defaults to all variables)",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=defaults.ema,
        help="Load EMA weights instead of the standard checkpoint when available",
    )

    args = parser.parse_args(args=cli_args)
    return InferenceConfig(
        start=args.start,
        steps=args.steps,
        init_data=Path(args.init_data),
        checkpoint_dir=Path(args.checkpoint_dir),
        checkpoint_name=args.checkpoint_name,
        output=Path(args.output),
        variables=args.variables,
        ema=args.ema,
    )


def _normalize_valid_times(ds: xr.Dataset, final_timestamp: np.datetime64) -> xr.Dataset:
    """Align the earth2studio ``time`` and ``lead_time`` axes to usable timestamps.

    earth2studio returns forecasts with two axes: the initialization time (``time``)
    and the offsets from that initialization (``lead_time``). To mirror the legacy
    inference example, we compute the actual verification timestamps (``valid_time``),
    attach them as coordinates, and extract the requested final forecast hour.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset returned by earth2studio after running inference.
    final_timestamp : np.datetime64
        Target forecast verification time to extract.
    """

    ds = ds.copy()
    ds["time"] = ds["time"].astype("datetime64[ns]")
    base_time = ds["time"].values
    lead_timedelta = ds["lead_time"].values.astype("timedelta64[ns]")
    valid_timesteps = (base_time[:, None] + lead_timedelta[None, :]).flatten()

    # Preserve the initial time as metadata while collapsing the time dimension.
    initial_time = str(ds["time"].values[0])
    ds = ds.isel(time=0).drop_vars("time")
    ds.attrs["initial_time"] = initial_time

    ds = ds.rename({"lead_time": "valid_time"})
    ds = ds.assign_coords(valid_time=(("valid_time",), valid_timesteps))

    if final_timestamp not in ds["valid_time"].values:
        raise ValueError(
            f"Requested final datetime {final_timestamp} not found in forecast valid times: {ds['valid_time'].values!r}"
        )

    return ds.sel(valid_time=[final_timestamp])


def _load_model(package_dir: Path, checkpoint_name: str, ema: bool) -> torch.nn.Module:
    model_package = Package(str(package_dir), cache=False)
    return SFNO.load_model(model_package, checkpoint_name=checkpoint_name, EMA=ema)


def _run_inference(
    start_datetime: datetime,
    n_steps: int,
    checkpoint_dir: Path,
    checkpoint_name: str,
    init_data_path: Path,
    output_path: Path,
    variables: Iterable[str] | None,
    ema: bool,
) -> None:
    if not init_data_path.exists():
        raise FileNotFoundError(
            f"Initial data file {init_data_path} not found. Generate it before running inference."
        )

    initial_data = DataArrayFile(str(init_data_path))
    model = _load_model(checkpoint_dir, checkpoint_name, ema)

    io = ZarrBackend()
    with torch.no_grad():
        io = deterministic([start_datetime.isoformat()], n_steps, model, initial_data, io, variables_list=variables)

    ds = xr.open_zarr(io.root.store)
    final_datetime = start_datetime + timedelta(hours=6 * n_steps)
    final_timestamp = np.datetime64(final_datetime)
    final_ds = _normalize_valid_times(ds, final_timestamp)

    if variables is not None:
        final_ds = final_ds[variables]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_ds.to_netcdf(output_path, mode="w", format="NETCDF4")



def main(cli_args: Sequence[str] | None = None, config: InferenceConfig | None = None) -> None:
    """Run inference using CLI args or a provided configuration.

    When ``config`` is omitted, argparse consumes ``cli_args`` (or ``sys.argv``)
    and fills in any missing values from ``DEFAULT_CONFIG``. To hard-code
    settings for batch runs without passing flags, edit ``DEFAULT_CONFIG`` and
    invoke the script without arguments.
    """

    cfg = config or _parse_args(DEFAULT_CONFIG, cli_args)
    start_dt = datetime.fromisoformat(cfg.start)

    _run_inference(
        start_datetime=start_dt,
        n_steps=cfg.steps,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=cfg.checkpoint_name,
        init_data_path=cfg.init_data,
        output_path=cfg.output,
        variables=cfg.variables,
        ema=cfg.ema,
    )


if __name__ == "__main__":
    main()
