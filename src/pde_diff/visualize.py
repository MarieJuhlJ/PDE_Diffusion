from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from omegaconf import OmegaConf
import torch

from pde_diff.data.datasets import ERA5Dataset
from pde_diff.model import DiffusionModel
from pde_diff.utils import dict_to_namespace


# Used for plotting samples:
EXTENT_FULL = [0.0, 359.25, 90.0, -90.0]
EXTENT_SUBSET = [0.0, 359.25, 69.75, 46.5]

def plot_samples(model, n=4, out_dir=Path('./reports/figures')):
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
    plt.savefig(save_dir / 'samples.png', bbox_inches='tight')
    print(f"Saved samples to {save_dir / 'samples.png'}")

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
        ax.figure.savefig(save_dir / f"{col}.png", dpi=150)
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
            ax.figure.savefig(save_dir / "val_metrics_combined.png", dpi=150)
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


def visualize_era5_sample(data_sample, variable, level=500, big_data_sample=None,sample_idx=None):
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
    fig, ax = plt.subplots(figsize=(8,6))

    if big_data_sample is not None:
        vmin = min(big_data_sample.min(), data_sample.min())
        vmax = max(big_data_sample.max(), data_sample.max())

        # Draw entire dataset first
        ax.imshow(big_data_sample.T, cmap='coolwarm', extent=EXTENT_FULL, origin='lower',vmin=vmin, vmax=vmax, alpha=0.5)
        ax.set_xlim(EXTENT_FULL[0], EXTENT_FULL[1])
        ax.set_ylim(EXTENT_FULL[3], EXTENT_FULL[2])

        # Draw subset on top in correct place
        ax.imshow(data_sample.T, cmap='coolwarm', extent=EXTENT_SUBSET, origin='lower',vmin=vmin, vmax=vmax)
        rect = patches.Rectangle(
            (EXTENT_SUBSET[0], EXTENT_SUBSET[2]),           # bottom-left corner (lon_min, lat_min)
            EXTENT_SUBSET[1]-EXTENT_SUBSET[0],              # width (lon_max - lon_min)
            EXTENT_SUBSET[3]-EXTENT_SUBSET[2],              # height (lat_max - lat_min)
            linewidth=2,
            edgecolor='black',                         # border color
            facecolor='none'                           # transparent fill
        )
        ax.add_patch(rect)

    else:
        ax.imshow(data_sample.T, cmap='coolwarm', extent=EXTENT_SUBSET, origin='lower')

    fig.colorbar(ax.images[-1], ax=ax, shrink=0.4, location="bottom")
    ax.set_title(f"{variable.capitalize()} at {level} hPa")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plot_path = f"reports/figures/samples/era5_sample{sample_idx if sample_idx is not None else ""}_{variable}_{level}hPa"
    plot_path += "_full" if big_data_sample is not None else ""
    plt.savefig(plot_path + ".png")

if __name__ == "__main__":
    #model_path = Path('./models')
    #model_id = 'exp1-ihnrf'

    #plot_training_metrics(model_id)

    #with open(model_path / model_id / 'config.yaml', 'r') as f:
    #    cfg = yaml.safe_load(f)
    #cfg = dict_to_namespace(cfg)
    #diffusion_model = DiffusionModel(cfg)
    #diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")
    #diffusion_model = diffusion_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    #plot_samples(diffusion_model, n=4)

    # Load the dataset configuration
    config_path = Path("configs/dataset/era5.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.normalize = False

    # Initialize the ERA5 dataset
    era5_dataset = ERA5Dataset(cfg)
    cfg.lat_range = None

    era5_dataset_full = ERA5Dataset(cfg)

    # Visualize a sample from the dataset
    sample_idx = 0  # Index of the sample to visualize
    variable = "t"  # Variable to visualize
    level = 500  # Pressure level in hPa

    data_sample = get_data_sample(era5_dataset, sample_idx, variable, level)
    data_sample_full = get_data_sample(era5_dataset_full, sample_idx, variable, level)

    visualize_era5_sample(data_sample, variable, level, big_data_sample=data_sample_full)
