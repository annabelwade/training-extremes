from collections import OrderedDict
from datetime import datetime
from math import ceil

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array
from loguru import logger
from tqdm import tqdm

# earth2studio 0.7.0 deterministic func: https://nvidia.github.io/earth2studio/v/0.7.0/_modules/earth2studio/run.html#deterministic
# earth2studio 0.10.0 deterministic func: https://nvidia.github.io/earth2studio/_modules/earth2studio/run.html#deterministic
# The function did not change between these two versions. - Annabel 12/18/25

def deterministic(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    data: DataSource,
    io: IOBackend,
    variables_list: None, # added for variable subsetting
    output_coords: CoordSystem = OrderedDict({}),
    device: torch.device | None = None,
) -> IOBackend:
    """Built in deterministic workflow.
    This workflow creates a determinstic inference pipeline to produce a forecast
    prediction using a prognostic model.

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    prognostic : PrognosticModel
        Prognostic model
    data : DataSource
        Data source
    io : IOBackend
        IO object
    output_coords: CoordSystem, optional
        IO output coordinate system override, by default OrderedDict({})
    device : torch.device, optional
        Device to run inference on, by default None

    Returns
    -------
    IOBackend
        Output IO object
    """
    # sphinx - deterministic end
    # logger.info("Running simple workflow!")
    # Load model onto the device
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    # sphinx - fetch data start
    # Fetch data from data source and load onto device
    prognostic_ic = prognostic.input_coords()
    time = to_time_array(time)

    if hasattr(prognostic, "interp_method"):
        interp_to = prognostic_ic
        interp_method = prognostic.interp_method
    else:
        interp_to = None
        interp_method = "nearest"

    x, coords = fetch_data(
        source=data,
        time=time,
        variable=prognostic_ic["variable"],
        lead_time=prognostic_ic["lead_time"],
        device=device,
        interp_to=interp_to,
        interp_method=interp_method,
    )

    # logger.success(f"Fetched data from {data.__class__.__name__}")
    # sphinx - fetch data end

    # Set up IO backend
    total_coords = prognostic.output_coords(prognostic.input_coords()).copy()
    for key, value in prognostic.output_coords(
        prognostic.input_coords()
    ).items():  # Scrub batch dims
        if value.shape == (0,):
            del total_coords[key]
    total_coords["time"] = time
    total_coords["lead_time"] = np.asarray(
        [
            prognostic.output_coords(prognostic.input_coords())["lead_time"] * i
            for i in range(nsteps + 1)
        ]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)

    for key, value in total_coords.items():
        total_coords[key] = output_coords.get(key, value)
    var_names = total_coords.pop("variable")
    io.add_array(total_coords, var_names)

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic.input_coords())
    # Create prognostic iterator
    model = prognostic.create_iterator(x, coords)

    # logger.info("Inference starting!")
    with tqdm(total=nsteps + 1, desc="Running inference", position=1) as pbar:
        for step, (x, coords) in enumerate(model):
            # Subselect domain/variables as indicated in output_coords
            x, coords = map_coords(x, coords, output_coords)
            #### Becca Edit - subset data saved to io
            if variables_list is None:
                io.write(*split_coords(x, coords))
            elif variables_list is not None:
                coords_subset = coords.copy()
                indices = []
                variables_list_subset = []
                for var in variables_list:
                    indices.append(coords["variable"].tolist().index(var))
                    variables_list_subset.append(str(coords["variable"][coords["variable"].tolist().index(var)]))
                
                x_subset = x[:,:,indices,:,:]
                coords_subset["variable"] = np.asarray(variables_list_subset)
                io.write(*split_coords(x_subset, coords_subset))
            #### Becca Edit End
            pbar.update(1)
            if step == nsteps:
                break

    # logger.success("Inference complete")
    return io