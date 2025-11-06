from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import einops
from omegaconf import DictConfig
from pde_diff.utils import DatasetRegistry
import xarray as xr

import pde_diff.data.const as const

@DatasetRegistry.register("dataset1")
class Dataset1(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.transform = None
        self.dataset = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple:
        """Return a given sample from the dataset."""
        x, y = self.dataset[idx]
        return x, y

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        # Implement your preprocessing logic here
        pass

@DatasetRegistry.register("fluid_data")
class FluidData(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.data_paths = [Path(cfg.path) / 'K_data.csv', Path(cfg.path) / 'p_data.csv']
        channels = len(self.data_paths)

        for i in range(channels):
            if i == 0:
                self.data = pd.read_csv(self.data_paths[i], header=None)
            else:
                self.data = np.stack((self.data, pd.read_csv(self.data_paths[i], header=None)), axis=-1)

        dtype = torch.float64 if cfg.use_double else torch.float32
        self.data = torch.tensor(self.data, dtype=dtype)
        self.num_datapoints = len(self.data)

        assert len(self.data.shape) == 3
        self.data = generalized_b_xy_c_to_image(self.data)

    def normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val)

    def unnorm(self, arr, min_val, max_val):
        return arr * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if index >= self.num_datapoints:
            raise IndexError('index out of range')
        return self.data[index]


def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return einops.rearrange(tensor, pattern, x=pixels_x, y=pixels_y)

"""
The ERA5Dataset class is responsible for loading and preprocessing the ERA5 dataset. This is a
slightly modified version of the GenCastDataset class from
 https://github.com/openclimatefix/graph_weather/blob/main/graph_weather/data/gencast_dataloader.py

It has to:
- load, normalize and concatenate (across the channel dimension) the input timesteps 0 and 1.
- load and normalize the residual between timesteps 2 and 1.

(Noise sampling and corruption of targets has been removed from original code and will be handled elsewhere)
"""
@DatasetRegistry.register("era5")
class ERA5Dataset(Dataset):
    """
    Dataset class for ERA5 training data.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        """
        Initialize the GenCast dataset object.

        Args the cfg should have:
            path: dataset path.
            atmospheric_features: list of features depending on pressure levels.
            single_features: list of features not depending on pressure levels.
            static_features: list of features not depending on time.
            max_year: max year to include in training set. Defaults to 2018.
            time_step: time step between predictions.
                        E.g. 12h steps correspond to time_step = 2 in a 6h dataset. Defaults to 2.
        """
        super().__init__()
        self.data = xr.open_zarr(cfg.path, chunks={})

        # Downsample longitude and latitude
        downsample_factor = cfg.get("downsample_factor", 2)  # Keep every 2nd grid point
        if downsample_factor:
            self.data = self.data.isel(
                longitude=slice(0, None, downsample_factor),
                latitude=slice(0, None, downsample_factor)
            )
            print(f"Downsampled data by a factor of {downsample_factor}.")
        self.max_year = cfg.max_year

        # Subset longitude and latitude
        lon_range = cfg.get("lon_range", None)  # Example: [0, 50]
        lat_range = cfg.get("lat_range", None)  # Example: [-50, 0]

        if lon_range:
            self.data = self.data.sel(longitude=slice(*lon_range))
            print(f"Limiting longitude to range: {lon_range}")
        if lat_range:
            self.data = self.data.sel(latitude=slice(*lat_range))
            print(f"Limiting latitude to range: {lat_range}")

        self.grid_lon = self.data["longitude"].values
        self.grid_lat = self.data["latitude"].values
        self.num_lon = len(self.grid_lon)
        self.num_lat = len(self.grid_lat)
        self.num_vars = len(self.data.keys())
        self.pressure_levels = np.array(self.data["isobaricInhPa"].values).astype(
            np.float32
        )  # Need them for loss weighting
        self.output_features_dim = len(cfg.atmospheric_features) * len(self.pressure_levels) + len(
            cfg.single_features
        )
        self.input_features_dim = self.output_features_dim + len(cfg.static_features) + 4

        self.time_step = cfg.time_step  # e.g. 2h steps correspond to time_step = 2 in a 1h dataset

        self.atmospheric_features = list(cfg.atmospheric_features)
        self.single_features = list(cfg.single_features)
        self.static_features = list(cfg.static_features)

        if cfg.get("normalize", True):
            self.means, self.stds, self.diff_means, self.diff_stds = self._init_means_and_stds()


    def _init_means_and_stds(self):
        # TODO: Handle missing features more gracefully aka actually implement them
        means = []
        stds = []
        diff_means = []
        diff_stds = []

        for var in self.atmospheric_features:
            try:
                means.extend(const.ERA5_MEANS[var])
                stds.extend(const.ERA5_STD[var])
                diff_means.extend(const.ERA5_DIFF_MEAN[var])
                diff_stds.extend(const.ERA5_DIFF_STD[var])
            except:
                means.extend(np.array([0.0,0.0]))
                stds.extend(np.array([1.0,1.0]))
                diff_means.extend(np.array([0.0,1.0]))
                diff_stds.extend(np.array([1.0,1.0]))

        for var in self.single_features:
            try:
                means.append(const.ERA5_MEANS[var])
                stds.append(const.ERA5_STD[var])
                diff_means.append(const.ERA5_DIFF_MEAN[var])
                diff_stds.append(const.ERA5_DIFF_STD[var])
            except:
                means.append(np.array([0.0,0.0]))
                stds.append(np.array([1.0,1.0]))
                diff_means.append(np.array([0.0,0.0]))
                diff_stds.append(np.array([1.0,1.0]))

        for var in self.static_features:
            try:
                means.append(const.ERA5_MEANS[var])
                stds.append(const.ERA5_STD[var])
            except:
                means.append(np.array([0.0,0.0]))
                stds.append(np.array([1.0,1.0]))

        return (
            np.array(means).astype(np.float32),
            np.array(stds).astype(np.float32),
            np.array(diff_means).astype(np.float32),
            np.array(diff_stds).astype(np.float32),
        )

    def _normalize(self, data, means, stds):
        return (data - means[:, None, None]) / (stds[:, None, None] + 0.0001)

    def _sin_cos_emb(self, x):
        return np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)

    def _generate_clock_features(self, ds):
        # Compute sin/cos embedding for day of the year
        day_of_year = ds.time.dt.dayofyear.values
        day_of_year_grid = einops.repeat(
            day_of_year, "t -> t lon lat", lon=self.num_lon, lat=self.num_lat
        )
        sin_day_of_year, cos_day_of_year = self._sin_cos_emb(day_of_year_grid / 365.0)


        # Compute sin/cos embedding for local mean time
        hour_of_day = ds.time.dt.hour.values
        hour_of_day_grid = einops.repeat(
            hour_of_day, "t -> t lon lat", lon=self.num_lon, lat=self.num_lat
        )
        local_mean_time = hour_of_day_grid + ds["longitude"].values[None, :, None] * 4 / 60.0
        sin_local_mean_time, cos_local_mean_time = self._sin_cos_emb(local_mean_time / 24.0)

        # Stack clock features
        clock_input_data = np.stack(
            [sin_day_of_year, cos_day_of_year, sin_local_mean_time, cos_local_mean_time], axis=-1
        ).astype(np.float32)
        clock_input_data = einops.rearrange(clock_input_data, "t lon lat clock -> t clock lon lat")

        return clock_input_data

    def __len__(self):
        return sum(self.data["time.year"].values <= self.max_year) - 2 * self.time_step

    def __getitem__(self, item):
        ds_conditionals = self.data.isel(time=[item, item + self.time_step])
        ds_state = self.data.isel(time=item + 2 * self.time_step)

        # Load inputs data
        ds_conditionals_atm = (
            ds_conditionals[self.atmospheric_features]
            .to_array()
            .transpose("time", "longitude", "latitude", "isobaricInhPa", "variable")
            .values
        )
        ds_conditionals_atm = einops.rearrange(ds_conditionals_atm, "t lon lat lev var -> t  (var lev) lon lat")
        raw_inputs = ds_conditionals_atm

        if self.single_features:
            ds_conditionals_single = (
                ds_conditionals[self.single_features]
                .to_array()
                .transpose("time", "longitude", "latitude", "variable")
                .values
            )
            ds_conditionals_single = einops.rearrange(ds_conditionals_single, "t lon lat var -> t var lon lat")
            raw_inputs = np.concatenate([raw_inputs, ds_conditionals_single], axis=1)

        if self.static_features:
            ds_conditionals_static = (
                ds_conditionals[self.static_features]
                .to_array()
                .transpose("longitude", "latitude", "variable")
                .values
            )
            ds_conditionals_static = np.stack([ds_conditionals_static] * 2, axis=0)
            ds_conditionals_static = einops.rearrange(ds_conditionals_static, "t lon lat var -> t (var) lon lat")
            raw_inputs = np.concatenate([raw_inputs, ds_conditionals_static], axis=1)


        # Normalize inputs
        if cfg.get("normalize", True):
            inputs_norm = self._normalize(raw_inputs, self.means, self.stds)
        else:
            inputs_norm = raw_inputs

        # Add time features
        clock_features = self._generate_clock_features(ds_conditionals)
        inputs = np.concatenate([inputs_norm, clock_features], axis=1)

        # Concatenate timesteps
        inputs = np.concatenate([inputs[0, :, :, :], inputs[1, :, :, :]], axis=0)
        prev_inputs = np.nan_to_num(inputs).astype(np.float32)

        # Load target data
        ds_state_atm = (
            ds_state[self.atmospheric_features]
            .to_array()
            .transpose("longitude", "latitude", "isobaricInhPa", "variable")
            .values
        )
        ds_state_atm = einops.rearrange(ds_state_atm, "lon lat lev var -> (var lev) lon lat")
        raw_state = ds_state_atm
        if self.single_features:
            ds_state_single = (
                ds_state[self.single_features]
                .to_array()
                .transpose("longitude", "latitude", "variable")
                .values
            )
            ds_state_single = einops.rearrange(ds_state_single, "lon lat var -> (var) lon lat")
            raw_state = np.concatenate([raw_state, ds_state_single], axis=0)

        # Normalize target residuals
        raw_state_change = raw_state - raw_inputs[1, : raw_state.shape[1], :, :]
        state_change = self._normalize(raw_state_change, self.diff_means, self.diff_stds)
        state_change = np.nan_to_num(state_change).astype(np.float32)

        return (prev_inputs, state_change)

class BatchedERA5Dataset(ERA5Dataset):
    """
    Dataset class for ERA5 batched training data.

    This dataset object returns a full batch as a single sample, it may be faster.

    Args that should be in config:
        path: Dataset path.
        atmospheric_features: List of features dependent on pressure levels.
        single_features: List of features not dependent on pressure levels.
        static_features: List of features not dependent on time.
        max_year (optional): Max year to include in training set. Defaults to 2018.
        time_step (optional): Time step between predictions.
                    E.g. 12h steps correspond to time_step = 2 in a 6h dataset. Defaults to 2.
        batch_size (optional): Size of the batch. Defaults to 32.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        """
        Initialize the GenCast dataset object.
        """
        super().__init__(cfg=cfg)

        self.batch_size = cfg.batch_size

    def __len__(self):
        return (super().__len__()) // self.batch_size

    def _batchify_inputs(self, data):
        start_idx = []
        for i in range(self.batch_size):
            start_idx.append([i, i + self.time_step])
        return data[start_idx]

    def _batchify_diffs(self, data):
        prev_idx = []
        target_idx = []
        for i in range(self.batch_size):
            prev_idx.append(i + self.time_step)
            target_idx.append(i + 2 * self.time_step)
        return data[target_idx] - data[prev_idx]

    def _generate_clock_features(self, ds):
        day_of_year = ds.time.dt.dayofyear.values / 365.0
        sin_day_of_year = (
            np.sin(2 * np.pi * day_of_year)[:, None, None]
            * np.ones((self.num_lon, self.num_lat))[None, :, :]
        )
        cos_day_of_year = (
            np.cos(2 * np.pi * day_of_year)[:, None, None]
            * np.ones((self.num_lon, self.num_lat))[None, :, :]
        )

        local_mean_time = (
            np.ones((self.num_lon, self.num_lat))[None, :, :]
            * (ds.time.dt.hour.values[:, None, None])
            + ds["longitude"].values[None, :, None] * 4 / 60.0
        )
        sin_local_mean_time = np.sin(2 * np.pi * local_mean_time / 24.0)
        cos_local_mean_time = np.cos(2 * np.pi * local_mean_time / 24.0)

        clock_input_data = np.stack(
            [sin_day_of_year, cos_day_of_year, sin_local_mean_time, cos_local_mean_time], axis=-1
        ).astype(np.float32)
        clock_input_data = einops.rearrange(clock_input_data, "t lon lat clock -> t clock lon lat")
        return clock_input_data

    def __getitem__(self, item):
        # Compute the starting and ending point of the batch.
        starting_point = self.batch_size * item
        ending_point = starting_point + 3 * self.time_step + self.batch_size - 1

        # Load data
        ds = self.data.isel(time=np.arange(starting_point, ending_point))
        ds_atm = (
            ds[self.atmospheric_features]
            .to_array()
            .transpose("time", "longitude", "latitude", "isobaricInhPa", "variable")
            .values
        )
        ds_atm = einops.rearrange(ds_atm, "t lon lat lev var -> t (var lev) lon lat")
        raw_inputs = ds_atm
        if self.single_features:
            ds_single = (
                ds[self.single_features]
                .to_array()
                .transpose("time", "longitude", "latitude", "variable")
                .values
            )
            ds_single = einops.rearrange(ds_single, "t lon lat var -> t var lon lat")

            raw_inputs = np.concatenate([raw_inputs, ds_single], axis=1)
        if self.static_features:
            ds_static = (
                ds[self.static_features]
                .to_array()
                .transpose("longitude", "latitude", "variable")
                .values
            )
            ds_static = np.stack([ds_static] * (ending_point - starting_point), axis=0)
            ds_static = einops.rearrange(ds_static, "lon lat var -> (var) lon lat")

            raw_inputs = np.concatenate([raw_inputs, ds_static], axis=1)

        # Compute inputs
        batched_inputs = self._batchify_inputs(raw_inputs)
        batched_inputs_norm = self._normalize(batched_inputs, self.means, self.stds)

        # Add time features
        ds_clock = self._batchify_inputs(self._generate_clock_features(ds))
        inputs = np.concatenate([batched_inputs_norm, ds_clock], axis=2)
        # Concatenate timesteps
        inputs = np.concatenate([inputs[:, 0, :, :, :], inputs[:, 1, :, :, :]], axis=1)
        prev_inputs = np.nan_to_num(inputs).astype(np.float32)

        # Compute targets residuals
        raw_state_change = np.concatenate([ds_atm, ds_single], axis=2) if self.single_features else ds_atm
        batched_state_change = self._batchify_diffs(raw_state_change)
        state_change = self._normalize(batched_state_change, self.diff_means, self.diff_stds)
        state_change = np.nan_to_num(state_change).astype(np.float32)

        return (prev_inputs, state_change)

def increment_clock_features(clock_features: torch.Tensor, step_size: int) -> torch.Tensor:
    """
    Increment clock features by a given hourly step size using PyTorch.
    """
    # Extract the day and time and compute their angles
    sin_day_of_year, cos_day_of_year = clock_features[..., 0], clock_features[..., 1]
    sin_local_mean_time, cos_local_mean_time = clock_features[..., 2], clock_features[..., 3]
    day_of_year_angle = torch.atan2(sin_day_of_year, cos_day_of_year)
    local_mean_time_angle = torch.atan2(sin_local_mean_time, cos_local_mean_time)

    # Increment the angles
    day_of_year_angle += 2 * torch.pi * step_size / (365 * 24)  # Increment day of year angle
    local_mean_time_angle += 2 * torch.pi * step_size / 24      # Increment local mean time angle
    day_of_year_angle = day_of_year_angle % (2 * torch.pi)
    local_mean_time_angle = local_mean_time_angle % (2 * torch.pi)

    sin_day_of_year = torch.sin(day_of_year_angle)
    cos_day_of_year = torch.cos(day_of_year_angle)
    sin_local_mean_time = torch.sin(local_mean_time_angle)
    cos_local_mean_time = torch.cos(local_mean_time_angle)

    updated_clock_features = torch.stack(
        [sin_day_of_year, cos_day_of_year, sin_local_mean_time, cos_local_mean_time], dim=-1
    )
    return updated_clock_features

if __name__ == "__main__":
    # Test ERA5 data:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "path": "data/era5/zarr",  # Replace with the actual path to your Zarr dataset
        "atmospheric_features": ["u", "v", "vo"],
        "single_features": [],
        "static_features": [],
        "max_year": 2024,
        "time_step": 1,
        "lat_range": [70, 46],  # For subsetting
        "downsample_factor": 3  # For downsampling
    })
    ERA5_dataset = ERA5Dataset(cfg=cfg)
    ERA5_dataset_len = len(ERA5_dataset)
    print(f"Longitude grid size: {len(ERA5_dataset.grid_lon)}")
    print(f"Latitude grid size: {len(ERA5_dataset.grid_lat)}")
    print(f"ERA5 Dataset length: {ERA5_dataset_len}")

    sample_idx = 10
    prev_inputs, target_residuals = ERA5_dataset[sample_idx]
    print(f"Sample prev_inputs shape: {prev_inputs.shape}, Sample target_residuals shape: {target_residuals.shape}")

    # Test Batched ERA5 data:
    batch_size = 4
    cfg.batch_size = batch_size
    Batched_ERA5_dataset = BatchedERA5Dataset(cfg=cfg)
    Batched_ERA5_dataset_len = len(Batched_ERA5_dataset)
    print(f"Batched ERA5 Dataset length: {Batched_ERA5_dataset_len}")
    batched_sample_idx = 2
    prev_inputs_batched, target_residuals_batched = Batched_ERA5_dataset[batched_sample_idx]
    print(f"Batched Sample prev_inputs shape: {prev_inputs_batched.shape}, Batched Sample target_residuals shape: {target_residuals_batched.shape}")
