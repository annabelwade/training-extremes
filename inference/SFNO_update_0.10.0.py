# --------------------------------------- #
# The following code is sourced from https://github.com/NVIDIA/makani/blob/v0.2.0/makani/models/model_package.py (makani v0.2.0)
# since SFNO installation instructions install makani v0.2.0 from the specific commit, I am going to make my edits to that commit's model_package.py, not the latest version (v0.2.1) which has some minor changes - Annabel 12/15/25
# --------------------------------------- #

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model package for easy inference/packaging. Model packages contain all the necessary data to
perform inference and its interface is compatible with earth2mip
"""
import os
import shutil
import json
import jsbeautifier
import numpy as np
import torch
from makani.utils.YParams import ParamsBase
from makani.utils.driver import Driver
from makani.third_party.climt.zenith_angle import cos_zenith_angle
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.models import model_registry
import datetime
import logging


logger = logging.getLogger(__name__)



class LocalPackage:
    """
    Implements the earth2mip/modulus Package interface.
    """

    # These define the model package in terms of where makani expects the files to be located
    THIS_MODULE = "makani.models.model_package"
    MODEL_PACKAGE_CHECKPOINT_PATH = "training_checkpoints/best_ckpt_mp0.tar"
    MINS_FILE = "mins.npy"
    MAXS_FILE = "maxs.npy"
    MEANS_FILE = "global_means.npy"
    STDS_FILE = "global_stds.npy"
    OROGRAPHY_FILE = "orography.nc"
    LANDMASK_FILE = "land_mask.nc"
    SOILTYPE_FILE = "soil_type.nc"

    def __init__(self, root):
        self.root = root

    def get(self, path):
        return os.path.join(self.root, path)

    @staticmethod
    def _load_static_data(root, params):
        if params.get("add_orography", False):
            params.orography_path = os.path.join(root, self.OROGRAPHY_FILE)
        if params.get("add_landmask", False):
            params.landmask_path = os.path.join(root, self.LANDMASK_FILE)
        if params.get("add_soiltype", False):
            params.soiltype_path = os.path.join(root, self.SOILTYPE_FILE)

        # alweays load all normalization files
        if params.get("global_means_path", None) is not None:
            params.global_means_path = os.path.join(root, self.MEANS_FILE)
        if params.get("global_stds_path", None) is not None:
            params.global_stds_path = os.path.join(root, self.STDS_FILE)
        if params.get("min_path", None) is not None:
            params.min_path = os.path.join(root, self.MINS_FILE)
        if params.get("max_path", None) is not None:
            params.max_path = os.path.join(root, self.MAXS_FILE)


class ModelWrapper(torch.nn.Module):
    """
    Model wrapper to make inference simple outside of makani.

    Attributes
    ----------
    model : torch.nn.Module
        ML model that is wrapped.
    params : ParamsBase
        parameter object containing information on how the model was initialized in makani

    Methods
    -------
    forward(x, time):
        performs a single prediction steps
    """

    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params
        nlat = params.img_shape_x
        nlon = params.img_shape_y

        # configure lats
        if "lat" in self.params:
            self.lats = np.asarray(self.params.lat)
        else:
            self.lats = np.linspace(90, -90, nlat, endpoint=True)

        # configure lons
        if "lon" in self.params:
            self.lons =	np.asarray(self.params.lon)
        else:
            self.lons = np.linspace(0, 360, nlon, endpoint=False)

        # zenith angle
        self.add_zenith = params.get("add_zenith", False)
        if self.add_zenith:
            self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)

        # load the normalization files
        bias, scale = get_data_normalization(self.params)

        # convert them to torch
        in_bias = torch.as_tensor(bias[:, self.params.in_channels]).to(torch.float32)
        in_scale = torch.as_tensor(scale[:, self.params.in_channels]).to(torch.float32)
        out_bias = torch.as_tensor(bias[:, self.params.out_channels]).to(torch.float32)
        out_scale = torch.as_tensor(scale[:, self.params.out_channels]).to(torch.float32)

        self.register_buffer("in_bias", in_bias)
        self.register_buffer("in_scale", in_scale)
        self.register_buffer("out_bias", out_bias)
        self.register_buffer("out_scale", out_scale)

    @property
    def in_channels(self):
        return self.params.get("channel_names", None)

    @property
    def out_channels(self):
        return self.params.get("channel_names", None)

    @property
    def timestep(self):
        return self.params.dt * self.params.dhours

    def forward(self, x, time, normalized_data=True, replace_state=None):
        if not normalized_data:
            x = (x - self.in_bias) / self.in_scale

        if self.add_zenith:
            cosz = cos_zenith_angle(time, self.lon_grid, self.lat_grid)
            cosz = cosz.astype(np.float32)
            z = torch.as_tensor(cosz).to(device=x.device)
            while z.ndim != x.ndim:
                z = z[None]
            self.model.preprocessor.cache_unpredicted_features(None, None, xz=z, yz=None)

        out = self.model(x, replace_state=replace_state)

        if not normalized_data:
            out = out * self.out_scale + self.out_bias

        return out


def save_model_package(params):
    """
    Saves out a self-contained model-package.
    The idea is to save anything necessary for inference beyond the checkpoints in one location.
    """
    # save out the current state of the parameters, make it human readable
    config_path = os.path.join(params.experiment_dir, "config.json")
    jsopts = jsbeautifier.default_options()
    jsopts.indent_size = 2

    with open(config_path, "w") as f:
        msg = jsbeautifier.beautify(json.dumps(params.to_dict()), jsopts)
        f.write(msg)

    if params.get("add_orography", False):
        shutil.copy(params.orography_path, os.path.join(params.experiment_dir, os.path.basename(params.orography_path)))

    if params.get("add_landmask", False):
        shutil.copy(params.landmask_path, os.path.join(params.experiment_dir, os.path.basename(params.landmask_path)))

    if params.get("add_soiltype", False):
        shutil.copy(params.soiltype_path, os.path.join(params.experiment_dir, os.path.basename(params.soiltype_path)))

    # always save out all normalization files
    if params.get("global_means_path", None) is not None:
        shutil.copy(params.global_means_path, os.path.join(params.experiment_dir, os.path.basename(params.global_means_path)))
    if params.get("global_stds_path", None) is not None:
        shutil.copy(params.global_stds_path, os.path.join(params.experiment_dir, os.path.basename(params.global_stds_path)))
    if params.get("min_path", None) is not None:
        shutil.copy(params.min_path, os.path.join(params.experiment_dir, os.path.basename(params.min_path)))
    if params.get("max_path", None) is not None:
        shutil.copy(params.max_path, os.path.join(params.experiment_dir, os.path.basename(params.max_path)))

    # write out earth2mip metadata.json
    fcn_mip_data = {
        "entrypoint": {"name": f"{LocalPackage.THIS_MODULE}:load_time_loop"},
    }
    with open(os.path.join(params.experiment_dir, "metadata.json"), "w") as f:
        msg = jsbeautifier.beautify(json.dumps(fcn_mip_data), jsopts)
        f.write(msg)


# TODO: this is not clean and should be reworked to allow restoring from params + checkpoint file
def load_model_package(package, pretrained=True, device="cpu", multistep=False):
    """
    Loads model package and return the wrapper which can be used for inference.
    """
    path = package.get("config.json")
    params = ParamsBase.from_json(path)
    LocalPackage._load_static_data(package.root, params)

    # assume we are not distributed
    # distributed checkpoints might be saved with different params values
    params.img_local_offset_x = 0
    params.img_local_offset_y = 0
    params.img_local_shape_x = params.img_shape_x
    params.img_local_shape_y = params.img_shape_y

    # get the model and
    model = model_registry.get_model(params, multistep=multistep).to(device)

    if pretrained:
        best_checkpoint_path = package.get(LocalPackage.MODEL_PACKAGE_CHECKPOINT_PATH)
        Driver.restore_from_checkpoint(best_checkpoint_path, model)

    model = ModelWrapper(model, params=params)

    # by default we want to do evaluation so setting it to eval here
    model.eval()

    return model


def load_time_loop(package, device=None, time_step_hours=None):
    """This function loads an earth2mip TimeLoop object that
    can be used for inference.

    A TimeLoop encapsulates normalization, regridding, and other logic, so is a
    very minimal interface to expose to a framework like earth2mip.

    See https://github.com/NVIDIA/earth2mip/blob/main/docs/concepts.rst
    for more info on this interface.
    """

    from earth2mip.networks import Inference
    from earth2mip.grid import equiangular_lat_lon_grid
    from physicsnemo.distributed.manager import DistributedManager

    config = package.get("config.json")
    params = ParamsBase.from_json(config)

    if params.in_channels != params.out_channels:
        raise NotImplementedError("Non-equal input and output channels are not implemented yet.")

    names = [params.data_channel_names[i] for i in params.in_channels]
    params.min_path = package.get(LocalPackage.MINS_FILE)
    params.max_path = package.get(LocalPackage.MAXS_FILE)
    params.global_means_path = package.get(LocalPackage.MEANS_FILE)
    params.global_stds_path = package.get(LocalPackage.STDS_FILE)

    center, scale = get_data_normalization(params)

    model = load_model_package(package, pretrained=True, device=device)
    shape = (params.img_crop_shape_x, params.img_crop_shape_y)

    # TODO: insert a check to see if the grid e2mip computes is the same that makani uses
    grid = equiangular_lat_lon_grid(nlat=params.img_crop_shape_x, nlon=params.img_crop_shape_y, includes_south_pole=True)

    if time_step_hours is None:
        hour = datetime.timedelta(hours=1)
        time_step = hour * params.get("dt", 6)
    else:
        time_step = datetime.timedelta(hours=time_step_hours)

    # Here we use the built-in class earth2mip.networks.Inference
    # will later be extended to use the makani inferencer
    inference = Inference(
        model=model,
        channel_names=names,
        center=center[:, params.in_channels],
        scale=scale[:, params.out_channels],
        grid=grid,
        n_history=params.n_history,
        time_step=time_step,
    )
    inference.to(device)
    return inference


# --------------------------------------- #
# The following code is sourced from https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/sfno.py (earth2studio v0.10.0)
# --------------------------------------- #

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import fnmatch
import os
from collections import OrderedDict
from collections.abc import Generator, Iterator
from datetime import datetime

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

try:
    from makani.models import model_registry
    # from makani.models.model_package import (
    #     LocalPackage,
    #     ModelWrapper,
    #     load_model_package,
    # ) ## WE EDIT THESE FUNCTIONS ABOVE, SO DON'T IMPORT THEM
    from makani.utils.driver import Driver
    from makani.utils.YParams import ParamsBase
except ImportError:
    OptionalDependencyFailure("sfno")
    load_model_package = None
    Driver = None
    ParamsBase = None
    LocalPackage = None
    model_registry = None
    ModelWrapper = None

VARIABLES = [
    "u10m",
    "v10m",
    "u100m",
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


@check_optional_dependencies()
class SFNO(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Spherical Fourier Operator Network global prognostic model.
    Consists of a single model with a time-step size of 6 hours.
    FourCastNet operates on 0.25 degree lat-lon grid (south-pole excluding)
    equirectangular grid with 73 variables.

    Note
    ----
    This model and checkpoint are trained using Modulus-Makani. For more information
    see the following references:

    - https://arxiv.org/abs/2306.03838
    - https://github.com/NVIDIA/modulus-makani
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/sfno_73ch_small

    Parameters
    ----------
    core_model : torch.nn.Module
        Core PyTorch model with loaded weights
    center : torch.Tensor
        Model center normalization tensor
    scale : torch.Tensor
        Model scale normalization tensor
    variables : np.array, optional
        Variables associated with model, by default 73 variable model.
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        variables: np.array = np.array(VARIABLES),
    ):
        super().__init__()
        self.model = core_model
        self.variables = variables
        if "2d" in self.variables:
            self.variables[self.variables == "2d"] = "d2m"

    def __str__(self) -> str:
        return "sfno_73ch_small"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model
        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(self.variables),
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        if input_coords is None:
            return output_coords
        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key not in ["batch", "time"]:
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)
        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0",
            cache_options={
                "cache_storage": Package.default_cache("sfno"),
                "same_names": True,
            },
        )
        package.root = os.path.join(package.root, "sfno_73ch_small")
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls, package: Package, variables: list = VARIABLES, device: str = "cpu"
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        variables : list, optional
            Model variable override, by default VARIABLES for SFNO 73 channel

        Returns
        -------
        PrognosticModel
            Prognostic model
        """

        # Makani load_model_package
        path = package.resolve("config.json")
        params = ParamsBase.from_json(path)

        # Set global_means_path and global_stds_path
        params.global_means_path = package.resolve("global_means.npy")
        params.global_stds_path = package.resolve("global_stds.npy")
        # Need to manually set min and max paths to none.
        params.min_path = None
        params.max_path = None

        # Need to manually set in and out channels to all variables.
        if params.channel_names is None:
            params.channel_names = variables
        else:
            variables = params.channel_names
        params.in_channels = np.arange(len(variables))
        params.out_channels = np.arange(len(variables))

        LocalPackage._load_static_data(package, params)

        # assume we are not distributed
        # distributed checkpoints might be saved with different params values
        params.img_local_offset_x = 0
        params.img_local_offset_y = 0
        params.img_local_shape_x = params.img_shape_x
        params.img_local_shape_y = params.img_shape_y

        # set grid type to sinusoidal without cosine features added in makani 0.2.0
        if params.get("add_cos_to_grid", None) is None:
            params.add_cos_to_grid = False

        # get the model
        model = model_registry.get_model(params, multistep=False).to(device)

        # Load checkpoint
        best_checkpoint_path = package.get(LocalPackage.MODEL_PACKAGE_CHECKPOINT_PATH)
        checkpoint = torch.load(
            best_checkpoint_path, weights_only=False, map_location=device
        )
        state_dict = checkpoint["model_state"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )

        # Resize model.blocks filters for some reason
        keys_to_resize = fnmatch.filter(
            state_dict.keys(), "model.blocks.*.filter.filter.weight"
        )
        for key in keys_to_resize:
            state_dict[key] = state_dict[key].unsqueeze(0)

        model.load_state_dict(state_dict)

        # Wrap model
        model = ModelWrapper(model, params=params)

        # Set model to eval mode
        model.eval()

        # Load variables
        variables = np.array(model.params.channel_names)

        return cls(model, variables=variables)

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)
        x = x.clone().squeeze(2)
        for j, _ in enumerate(coords["batch"]):
            for i, t in enumerate(coords["time"]):
                # https://github.com/NVIDIA/modulus-makani/blob/933b17d5a1ebfdb0e16e2ebbd7ee78cfccfda9e1/makani/third_party/climt/zenith_angle.py#L197
                # Requires time zone data
                t = [
                    datetime.fromisoformat(dt.isoformat() + "+00:00")
                    for dt in timearray_to_datetime(t + coords["lead_time"])
                ]
                x[j, i : i + 1] = self.model(x[j, i : i + 1], t, normalized_data=False)
        x = x.unsqueeze(2)
        return x, output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        ------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system
        """
        return self._forward(x, coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()
        self.output_coords(coords)
        yield x, coords
        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)
            # Forward is identity operator
            x, coords = self._forward(x, coords)
            # Rear hook
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)