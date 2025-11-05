
# def test_data_normalization():
#     from omegaconf import OmegaConf
#     from pde_diff.data.datasets import CIFAR10
#     from torchvision import transforms
#     import torch

#     cfg = OmegaConf.create({
#         "name": "cifar10",
#         "path": "./data/cifar10/",
#         "train": True
#     })

#     dataset = CIFAR10(cfg)
#     sample = dataset[0]  # Get the first sample

#     # Check if the sample is a tensor
#     assert isinstance(sample, torch.Tensor), "Sample is not a tensor"

#     # Check the shape of the sample (C, H, W)
#     assert sample.shape == (3, 32, 32), f"Unexpected sample shape: {sample.shape}"

#     # Check normalization: mean should be close to 0 and std close to 1
#     mean = sample.mean(dim=[1, 2])
#     std = sample.std(dim=[1, 2])

    # assert torch.allclose(mean, torch.zeros(3), atol=0.1), f"Mean not close to 0: {mean}"
    # assert torch.allclose(std, torch.ones(3), atol=0.1), f"Std not close to 1: {std}"

def test_increment_clock_features():
    import torch
    from pde_diff.data.datasets import increment_clock_features

    # Create dummy clock features
    num_lon, num_lat = 10, 10
    clock_features = torch.zeros((1, num_lon, num_lat, 4), dtype=torch.float32)

    # Initialize clock features with specific values
    clock_features[..., 0] = torch.sin(torch.tensor(0.0))  # sin_day_of_year
    clock_features[..., 1] = torch.cos(torch.tensor(0.0))  # cos_day_of_year
    clock_features[..., 2] = torch.sin(torch.tensor(0.0))  # sin_local_mean_time
    clock_features[..., 3] = torch.cos(torch.tensor(0.0))  # cos_local_mean_time

    # Increment clock features by 6 hours
    step_size = 6
    updated_clock_features = increment_clock_features(clock_features, step_size)

    # Compute expected values
    day_of_year_angle = torch.tensor(2 * torch.pi * step_size / (365 * 24))
    local_mean_time_angle = torch.tensor(2 * torch.pi * step_size / 24)

    expected_sin_day_of_year = torch.sin(day_of_year_angle)
    expected_cos_day_of_year = torch.cos(day_of_year_angle)
    expected_sin_local_mean_time = torch.sin(local_mean_time_angle)
    expected_cos_local_mean_time = torch.cos(local_mean_time_angle)

    # Assert that the updated clock features match the expected values
    assert torch.allclose(updated_clock_features[..., 0], expected_sin_day_of_year, atol=1e-5), \
        f"Sin of day_of_year does not match: {updated_clock_features[..., 0]} vs {expected_sin_day_of_year}"
    assert torch.allclose(updated_clock_features[..., 1], expected_cos_day_of_year, atol=1e-5), \
        f"Cos of day_of_year does not match: {updated_clock_features[..., 1]} vs {expected_cos_day_of_year}"
    assert torch.allclose(updated_clock_features[..., 2], expected_sin_local_mean_time, atol=1e-5), \
        f"Sin of local_mean_time does not match: {updated_clock_features[..., 2]} vs {expected_sin_local_mean_time}"
    assert torch.allclose(updated_clock_features[..., 3], expected_cos_local_mean_time, atol=1e-5), \
        f"Cos of local_mean_time does not match: {updated_clock_features[..., 3]} vs {expected_cos_local_mean_time}"

    print("test_increment_clock_features passed!")

def test_era5_dataset():
    import numpy as np
    from omegaconf import OmegaConf
    from pde_diff.data.datasets import ERA5Dataset

    # Mock configuration for the ERA5 dataset
    cfg = OmegaConf.create({
        "path": "data/era5/zarr",  # Replace with the actual path to your Zarr dataset
        "atmospheric_features": ["u", "v", "vo"],
        "single_features": [],
        "static_features": [],
        "max_year": 2024,
        "time_step": 1
    })

    # Initialize the dataset
    dataset = ERA5Dataset(cfg)

    # Check dataset length
    assert len(dataset) > 0, "Dataset length should be greater than 0"

    # Test a sample from the dataset
    sample_idx = 0
    prev_inputs, target_residuals = dataset[sample_idx]

    # Check the shapes of the outputs
    assert isinstance(prev_inputs, np.ndarray), "prev_inputs should be a numpy array"
    assert isinstance(target_residuals, np.ndarray), "target_residuals should be a numpy array"
    assert prev_inputs.ndim == 3, f"prev_inputs should have 3 dimensions, got {prev_inputs.ndim}"
    assert target_residuals.ndim == 3, f"target_residuals should have 3 dimensions, got {target_residuals.ndim}"

    # Print some information for debugging
    print(f"Sample prev_inputs shape: {prev_inputs.shape}")
    print(f"Sample target_residuals shape: {target_residuals.shape}")

    # Additional checks (optional)
    assert not np.isnan(prev_inputs).any(), "prev_inputs contains NaN values"
    assert not np.isnan(target_residuals).any(), "target_residuals contains NaN values"

    print("ERA5Dataset test passed!")
