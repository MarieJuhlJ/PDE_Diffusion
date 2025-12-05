from pathlib import Path
import yaml

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from pde_diff.data.datasets import ERA5Dataset
from pde_diff.model import DiffusionModel
from pde_diff.utils import dict_to_namespace

plt.style.use("pde_diff.custom_style")

# Used for plotting samples:
EXTENT_FULL = [0.0, 359.25, 90.0, -90.0]
EXTENT_SUBSET = [0.0, 359.25, 69.75, 46.5]
PLOT_TYPE = ".png"

VAR_NAMES = {
    "u": "u",
    "v": "v",
    "t": "T",
    "z": r"$\Phi$",
    "pv": "q",
}

colors = ["brown", "white", "teal"]  # Transition from blue to white to red
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
colors = ["teal", "white", "brown"]  # Transition from blue to white to red
custom_cmap2 = LinearSegmentedColormap.from_list("custom_cmap", colors)

COLOR_BARS = {
    "u": "RdBu",
    "v": "RdBu",
    "t": "coolwarm",
    "z": custom_cmap2,
    "pv": custom_cmap,
}

def plot_darcy_samples(model, model_id, n=4, out_dir=Path('./reports/figures')):
    save_dir = Path(out_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    samples = model.sample_loop(batch_size=n)
    samples = samples.cpu().numpy()
    fig, axs = plt.subplots(2, n, figsize=(n*3, 3))
    for i in range(n):
        axs[0, i].imshow(samples[i, 0], cmap='magma')
        axs[0, i].set_title(f'Sample {i+1} - K')
        axs[0, i].axis('off')
        axs[1, i].imshow(samples[i, 1], cmap='magma')
        axs[1, i].set_title(f'Sample {i+1} - P')
        axs[1, i].axis('off')
    plt.savefig(save_dir / f'samples{PLOT_TYPE}', bbox_inches='tight')
    print(f"Saved samples to {save_dir / f'samples{PLOT_TYPE}'}")

def plot_training_metrics(model_id, out_dir=Path("./reports/figures")):
    df = pd.read_csv(Path("./logs") / model_id / "version_0" / "metrics.csv").sort_values(["epoch", "step"])
    save_dir = Path(out_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("train_loss", True,  "Train Loss vs Epoch"),
        ("val_loss", True,  "Validation Loss vs Epoch"),
        ("val_mse", False, "Validation MSE vs Epoch"),
        ("val_darcy_residual", True, "Validation Darcy Residual vs Epoch"),
    ]

    for col, logy, title in metrics:
        if col not in df:
            continue
        sub = df[["epoch", col]].dropna()
        if sub.empty:
            continue

        ax = sub.plot(x="epoch", y=col, legend=False, figsize=(8, 4.5))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(save_dir / f"{col}{PLOT_TYPE}", dpi=150)
        plt.close(ax.figure)

    if {"val_loss", "val_mse"}.issubset(df.columns):
        sub = df[["epoch", "val_loss", "val_mse"]].dropna()
        if not sub.empty:
            ax = sub.plot(x="epoch", y=["val_loss", "val_mse"], figsize=(8, 4.5))
            ax.set_title("Validation Metrics vs Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.figure.tight_layout()
            ax.figure.savefig(save_dir / f"val_metrics_combined{PLOT_TYPE}", dpi=150)
            plt.close(ax.figure)


def get_data_sample(dataset, sample_idx, variable, level=500):
    """
    Extract data for visualization from the ERA5 dataset.

    """
    # Extract the sample
    sample = dataset[sample_idx]
    inputs, _ = sample

    # Find the index of the variable and level
    var_idx = dataset.atmospheric_features.index(variable)
    level_idx = np.where(dataset.pressure_levels == level)[0][0]

    # Extract the data for the variable and level
    data = inputs[var_idx * len(dataset.pressure_levels) + level_idx]
    return data

def get_time_series(dataset, variable, level=500, coords=(12.568, 55.676)):
    """
    Extract a time series of a variable at a specific pressure level from the ERA5 dataset.

    Args:
        dataset (ERA5Dataset): The ERA5 dataset.
        variable (str): Variable to extract (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        coords (tuple): Tuple of (longitude, latitude) to extract the time series from. Default is (12.568, 55.676) (Copenhagen).
    """
    level_idx = np.where(dataset.pressure_levels == level)[0][0]
    lon, lat = coords
    lon_idx = np.argmin(np.abs(dataset.grid_lon - lon))
    lat_idx = np.argmin(np.abs(dataset.grid_lat - lat))
    time_series = dataset.data[variable][:, level_idx, lon_idx, lat_idx]
    return time_series


def visualize_era5_sample(data_sample, variable, level=500, big_data_sample=None, sample_idx=None, dir=Path("./reports/figures/samples")):
    """
    Visualize a sample from the ERA5 dataset.

    Args:
        data_sample (np.ndarray): The data sample to visualize.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        big_data_sample (np.ndarray, optional): The entire dataset for context.
        sample_idx (int, optional): Index of the sample (for title purposes).
    """
    # Plot the data
    fig, ax = plt.subplots(figsize=(6,3))

    if big_data_sample is not None:
        vmin = min(big_data_sample.min(), data_sample.min())
        vmax = max(big_data_sample.max(), data_sample.max())

        # Draw entire dataset first
        ax.imshow(big_data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_FULL, origin='lower',vmin=vmin, vmax=vmax, alpha=0.5)
        ax.set_xlim(EXTENT_FULL[0], EXTENT_FULL[1])
        ax.set_ylim(EXTENT_FULL[3], EXTENT_FULL[2])

        # Draw subset on top in correct place
        ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower',vmin=vmin, vmax=vmax)
        rect = patches.Rectangle(
            (EXTENT_SUBSET[0], EXTENT_SUBSET[2]),           # bottom-left corner (lon_min, lat_min)
            EXTENT_SUBSET[1]-EXTENT_SUBSET[0],              # width (lon_max - lon_min)
            EXTENT_SUBSET[3]-EXTENT_SUBSET[2],              # height (lat_max - lat_min)
            linewidth=2,
            edgecolor='black',                         # border color
            facecolor='none'                           # transparent fill
        )
        ax.add_patch(rect)
        fig.colorbar(ax.images[-1], ax=ax, shrink=0.4)
    else:
        ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower')
        fig.colorbar(ax.images[-1], ax=ax, shrink=0.4, location="bottom")

    ax.set_title(f"{VAR_NAMES.get(variable, variable)} at {level} hPa")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa"
    plot_path += "_full" if big_data_sample is not None else ""
    plot_path += PLOT_TYPE
    plot_path = dir / plot_path
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved visualization to {plot_path}")

def visualize_era5_sample_full(big_data_sample, variable, level=500, sample_idx=None, dir=Path("./reports/figures/samples")):
    """
    Visualize a sample from the ERA5 dataset.

    Args:
        data_sample (np.ndarray): The data sample to visualize.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        big_data_sample (np.ndarray, optional): The entire dataset for context.
        sample_idx (int, optional): Index of the sample (for title purposes).
    """
    # Plot the data
    fig, ax = plt.subplots(figsize=(8,4))

    # Draw entire data
    ax.imshow(big_data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_FULL, origin='lower')
    #remove all ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa_full_only"
    plot_path += PLOT_TYPE
    plot_path = dir / plot_path
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {plot_path}")

def visualize_noise_schedule(scheduler, data_sample, variable, level=500, sample_idx=None, dir=Path("./reports/figures/samples"), steps= 10):
    """
    Visualize a sample from the ERA5 dataset.

    Args:
        data_sample (np.ndarray): The data sample to visualize.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        sample_idx (int, optional): Index of the sample (for title purposes).
    """
    # Plot the data
    fig, ax = plt.subplots(figsize=(data_sample.shape[1] / 5, data_sample.shape[0] / 5))
    ax.imshow(data_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower')
    ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])
    ax.axis('off')  # Remove axes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) #Remove margins

    # Save the plot
    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa_noise_{0}"
    plot_path += PLOT_TYPE
    plot_path = dir / plot_path
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved visualization to {plot_path}")

    for t in range(100,scheduler.config.num_train_timesteps+1, scheduler.config.num_train_timesteps//steps):
        noised_sample = scheduler.add_noise(torch.tensor(data_sample).unsqueeze(0).unsqueeze(0).float(), torch.randn_like(torch.tensor(data_sample).unsqueeze(0).unsqueeze(0).float()), torch.tensor([t-1]))
        noised_sample = noised_sample.squeeze().squeeze().numpy()
        fig, ax = plt.subplots(figsize=(noised_sample.shape[1] / 5, noised_sample.shape[0] / 5))
        ax.imshow(noised_sample.T, cmap=COLOR_BARS.get(variable, 'coolwarm'), extent=EXTENT_SUBSET, origin='lower')
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])
        ax.axis('off')  # Remove axes
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) #Remove margins

        # Save the plot
        plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{variable}_{level}hPa_noise_{t}"
        plot_path += PLOT_TYPE
        plot_path = dir / plot_path
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved visualization to {plot_path}")


def visualize_time_series(dataset, variable, level=500, dir=Path("./reports/figures/time_series"), coords=(12.568, 55.676)):
    """
    Visualize a time series of a variable at a specific pressure level from the ERA5 dataset.

    Args:
        dataset (ERA5Dataset): The ERA5 dataset.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        dir (Path): Directory to save the plots.
        coords (tuple): Tuple of (longitude, latitude) to extract the time series from. Default is (12.568, 55.676) (Copenhagen).
    """

    dir.mkdir(parents=True, exist_ok=True)
    lon, lat = coords

    time_series = np.array(get_time_series(dataset, variable, level, coords))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_series, label=f"{VAR_NAMES.get(variable, variable)} at {level} hPa")
    ax.set_title(f"Time Series of {VAR_NAMES.get(variable, variable)} at {level} hPa\nLocation: Lon {lon}, Lat {lat}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(VAR_NAMES.get(variable, variable))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = dir / f"time_series_{variable}_{level}hPa_lon{lon}_lat{lat}{PLOT_TYPE}"
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved time series plot to {plot_path}")

def plot_and_save_era5(csv_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).apply(pd.to_numeric, errors="ignore").dropna(subset=["step"])
    df = df.sort_values("step")

    # ---- Loss plot ----
    train_df = df.dropna(subset=["train_loss"])
    val_df = df.dropna(subset=["val_loss"])
    
    plt.plot(train_df.epoch, train_df.train_loss, label="train_loss")
    plt.plot(val_df.epoch, val_df.val_loss, label="val_loss")
    plt.legend(); plt.grid(); plt.title("Loss")
    plt.savefig(out_dir / "loss.png"); plt.clf()

    res_df = df.dropna(subset=["val_era5_vorticity_residual"])
    mse_df = df.dropna(subset=["val_mse_(weighted)"])

    # ---- Residual + MSE plot ----
    plt.plot(res_df.epoch, res_df.val_era5_vorticity_residual, label="era5_residual")
    plt.plot(mse_df.epoch, mse_df["val_mse_(weighted)"], label="mse_weighted")
    plt.legend(); plt.grid(); plt.title("Residuals & MSE")
    plt.savefig(out_dir / "residuals.png"); plt.clf()

    return


if __name__ == "__main__":
#     model_path = Path('./models')
#     model_id = 'exp1-xbvcn'

#     cfg = OmegaConf.load(model_path / model_id / "config.yaml")

#     dataset = DatasetRegistry.create(cfg.dataset)
#     loss_fn = LossRegistry.create(cfg.loss)
#     if cfg.dataset.name == 'era5' and cfg.loss.name == 'vorticity': #semi cursed (TODO clean up)
#         loss_fn.set_mean_and_std(dataset.means, dataset.stds,
#                               dataset.diff_means, dataset.diff_stds)
#     diffusion_model = DiffusionModel(cfg, loss_fn)
#     diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")
#     diffusion_model = diffusion_model.to('cuda' if torch.cuda.is_available() else 'cpu')
#     plot_and_save_era5(Path('logs') / model_id / 'version_0/metrics.csv', Path('./reports/figures') / model_id)

    # Load the dataset configuration
    config_path = Path("configs/dataset/era5.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.normalize = False

    # Initialize the ERA5 dataset
    era5_dataset = ERA5Dataset(cfg)
    cfg.lat_range = None

    era5_dataset_full = ERA5Dataset(cfg)

    # Visualize all variables of a sample from the dataset
    sample_idx = 0  # Index of the sample to visualize
    variable = "t"  # Variable to visualize
    level = 500  # Pressure level in hPa
    """for variable in cfg.atmospheric_features:
        data_sample = get_data_sample(era5_dataset, sample_idx, variable, level)
        data_sample_full = get_data_sample(era5_dataset_full, sample_idx, variable, level)
        visualize_era5_sample(data_sample, variable, level, big_data_sample=data_sample_full)

    # Visualize the noise schedule
    config_path = Path("configs/scheduler/ddpm.yaml")
    cfg_scheduler = OmegaConf.load(config_path)
    from pde_diff.utils import SchedulerRegistry
    scheduler = SchedulerRegistry.create(cfg_scheduler)

    variable = "t"
    data_sample = get_data_sample(era5_dataset, sample_idx, variable, level)

    #visualize_noise_schedule(scheduler, data_sample, variable, level, sample_idx)

    # Visualize a full sample from the dataset without the subset overlay
    visualize_era5_sample_full(data_sample_full, variable, level)"""

    # Visualize time series
    for variable in cfg.atmospheric_features:
        visualize_time_series(era5_dataset, variable=variable, level=500, coords=(12.568, 55.676))

