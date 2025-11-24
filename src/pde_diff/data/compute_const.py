import xarray as xr
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from datasets import ERA5Dataset  # Import the ERA5Dataset class

# Path to the ERA5 dataset (update this to the actual path)
ERA5_DATASET_PATH = "data/era5/zarr"

# List of features to compute means and stds for
FEATURES = [
    "v",
    "u",
    "pv",
    "t",
    "z"
]

# Pressure levels to include for 3D variables
PRESSURE_LEVELS = ["450", "500", "550"]  # Add more levels as needed

# Output dictionaries
ERA5_MEANS = {}
ERA5_STD = {}
ERA5_DIFF_MEAN = {}
ERA5_DIFF_STD = {}

def compute_means_and_stds(dataset, features, pressure_levels):
    """
    Compute means, standard deviations, and timestep differences for the specified features.

    Args:
        dataset (ERA5Dataset): Initialized ERA5Dataset object.
        features (list): List of features to compute statistics for.
        pressure_levels (list): List of pressure levels for 3D variables.

    Returns:
        dict, dict, dict, dict: Dictionaries containing means, stds, diff means, and diff stds.
    """
    means = {}
    stds = {}
    diff_means = {}
    diff_stds = {}

    for feature in features:
        if feature in dataset.data:
            if "isobaricInhPa" in dataset.data[feature].dims:  # 3D variable with pressure levels
                means[feature] = {}
                stds[feature] = {}
                diff_means[feature] = {}
                diff_stds[feature] = {}
                for level in pressure_levels:
                    data = dataset.data[feature].values
                    means[feature] = np.mean(data, axis=(0,2,3))
                    stds[feature] = np.std(data,axis=(0,2,3))

                    # Compute differences between timesteps
                    diff_data = np.diff(data, axis=0)
                    diff_means[feature] = np.mean(diff_data,axis=(0,2,3))
                    diff_stds[feature] = np.std(diff_data,axis=(0,2,3))
            else:  # 2D variable
                data = dataset.data[feature].values
                means[feature] = np.mean(data)
                stds[feature] = np.std(data)

                # Compute differences between timesteps
                diff_data = np.diff(data, axis=0)
                diff_means[feature] = np.mean(diff_data)
                diff_stds[feature] = np.std(diff_data)
        else:
            print(f"Feature '{feature}' not found in the dataset.")

    return means, stds, diff_means, diff_stds

# Initialize the ERA5Dataset
cfg = OmegaConf.create({
    "path": ERA5_DATASET_PATH,
    "atmospheric_features": FEATURES,
    "single_features": [],
    "static_features": [],
    "max_year": 2024,
    "time_step": 1,
    "normalize": False,
    "lat_range": [70, 46],
    "downsample_factor": 3

})
dataset = ERA5Dataset(cfg=cfg)

# Compute means, stds, diff means, and diff stds
ERA5_MEANS, ERA5_STD, ERA5_DIFF_MEAN, ERA5_DIFF_STD = compute_means_and_stds(
    dataset, FEATURES, PRESSURE_LEVELS
)

# Print the results
print("ERA5_MEANS =", ERA5_MEANS)
print("ERA5_STD =", ERA5_STD)
print("ERA5_DIFF_MEAN =", ERA5_DIFF_MEAN)
print("ERA5_DIFF_STD =", ERA5_DIFF_STD)

# Save the results to the constants file (optional)
CONSTANTS_FILE = Path("src/pde_diff/data/const_new.py")
if CONSTANTS_FILE.exists():
    with open(CONSTANTS_FILE, "r") as f:
        content = f.readlines()

    with open(CONSTANTS_FILE, "w") as f:
        for line in content:
            if "ERA5_MEANS =" in line:
                f.write(f"ERA5_MEANS = {ERA5_MEANS}\n")
            elif "ERA5_STD =" in line:
                f.write(f"ERA5_STD = {ERA5_STD}\n")
            elif "ERA5_DIFF_MEAN =" in line:
                f.write(f"ERA5_DIFF_MEAN = {ERA5_DIFF_MEAN}\n")
            elif "ERA5_DIFF_STD =" in line:
                f.write(f"ERA5_DIFF_STD = {ERA5_DIFF_STD}\n")
            else:
                f.write(line)
