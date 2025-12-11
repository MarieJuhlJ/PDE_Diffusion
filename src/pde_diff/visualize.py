from pathlib import Path
import yaml

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from pde_diff.data.datasets import ERA5Dataset
from pde_diff.model import DiffusionModel
from pde_diff.loss import DarcyLoss
from pde_diff.utils import dict_to_namespace

# Path to your TTF file
font_path = "./times.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'

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

def plot_darcy_samples(model, model_id, out_dir=Path('./reports/figures')):
    from matplotlib.colors import LogNorm
    save_dir = Path(out_dir) / model_id
    cfg = OmegaConf.create({
        "c_residual": None
    })
    loss = DarcyLoss(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    samples = model.sample_loop(batch_size=1)
    loss_samples = loss.compute_residual_field_for_plot(
        samples.to('cuda' if torch.cuda.is_available() else 'cpu')
    )
    loss_samples = loss_samples.cpu().numpy()
    samples = samples.cpu().numpy()

    # Set up figure
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot #1
    im0 = axs[0].imshow(samples[0, 0], cmap='magma')
    axs[0].set_title(r'Permeability - $K$')
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # Plot #2
    im1 = axs[1].imshow(np.rot90(samples[0, 1], k=3), cmap='magma')
    axs[1].set_title(r'Pressure - $P$')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # Plot #3

    im2 = axs[2].imshow(
        loss_samples[0],
        cmap='magma',
        norm=LogNorm()   # ← Log scale here
    )

    axs[2].set_title(r'Residual - $\mathcal{R}_{\text{MAE}}$')
    axs[2].axis('off')

    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # Save
    plt.savefig(save_dir / f'samples{PLOT_TYPE}', bbox_inches='tight')
    plt.close(fig)

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

def plot_darcy_val_metrics(model_id_1, model_id_2, fold_num, log_path, out_dir, smooth_window=10):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # --- Global styling tweaks ---
    plt.rcParams.update({
        "figure.figsize": (13, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    # Custom colour palette (different from default Matplotlib cycle)
    diffusion_color = "#2A9D8F"   # teal
    diffusion_ci    = "#9AD1C5"   # light teal
    pidm_color      = "#E76F51"   # warm coral
    pidm_ci         = "#F4A261"   # light orange
    train_loss_pidm_color = "#EC46DB"
    train_loss_pidm_ci    = "#D378CA"
    val_loss_pidm_color   = "#DE2B37"
    val_loss_pidm_ci      = "#DD6069"
    train_loss_diff_color = "#2341EC"
    train_loss_diff_ci    = "#6955EE"
    val_loss_diff_color   = "#E9AF11"
    val_loss_diff_ci      = "#E0BC58"

    def moving_average_2d(arr, window):
        """Apply moving average along the time axis for a 2D array [fold, time]."""
        if window <= 1:
            return arr
        kernel = np.ones(window) / window
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=arr
        )

    def load_model_stats(model_id, smooth_window=1):
        residual_errors = []
        weighted_mse_errors = []
        train_loss = []
        val_loss = []
        steps = None

        for fold in range(1, fold_num + 1):
            current_model_id = f"{model_id}-{fold}"
            csv_path = Path(log_path) / current_model_id / "version_0" / "metrics.csv"

            df = (
                pd.read_csv(csv_path)
                .apply(pd.to_numeric, errors="ignore")
                .dropna(subset=["step"])
                .sort_values("step")
            )

            if steps is None:
                steps = df["step"].values

            residual_errors.append(df["val_darcy_residual"].dropna().values)
            weighted_mse_errors.append(df["val_mse_(weighted)"].dropna().values)
            train_loss.append(df["train_loss"].dropna().values)
            val_loss.append(df["val_loss"].dropna().values)

        residual_errors = np.array(residual_errors)      # shape: [fold, time]
        weighted_mse_errors = np.array(weighted_mse_errors)
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)

        # --- Smooth over time (epochs) ---
        residual_errors = moving_average_2d(residual_errors, smooth_window)
        weighted_mse_errors = moving_average_2d(weighted_mse_errors, smooth_window)
        train_loss = moving_average_2d(train_loss, smooth_window)
        val_loss = moving_average_2d(val_loss, smooth_window)

        n = residual_errors.shape[0]

        res_mean = residual_errors.mean(axis=0)
        res_std = residual_errors.std(axis=0)
        res_low = res_mean - 1.96 * res_std / np.sqrt(n)
        res_high = res_mean + 1.96 * res_std / np.sqrt(n)

        mse_mean = weighted_mse_errors.mean(axis=0)
        mse_std = weighted_mse_errors.std(axis=0)
        mse_low = mse_mean - 1.96 * mse_std / np.sqrt(n)
        mse_high = mse_mean + 1.96 * mse_std / np.sqrt(n)

        train_loss_mean = train_loss.mean(axis=0)
        train_loss_std = train_loss.std(axis=0)
        val_loss_mean = val_loss.mean(axis=0)
        val_loss_std = val_loss.std(axis=0)

        train_loss_low = train_loss_mean - 1.96 * train_loss_std / np.sqrt(n)
        train_loss_high = train_loss_mean + 1.96 * train_loss_std / np.sqrt(n)
        val_loss_low = val_loss_mean - 1.96 * val_loss_std / np.sqrt(n)
        val_loss_high = val_loss_mean + 1.96 * val_loss_std / np.sqrt(n)


        # Number of *smoothed* points
        epochs = res_mean.shape[0]

        return {
            "epochs": epochs,
            "res_mean": res_mean,
            "res_low": res_low,
            "res_high": res_high,
            "mse_mean": mse_mean,
            "mse_low": mse_low,
            "mse_high": mse_high,
            "train_loss_mean": train_loss_mean,
            "train_loss_low": train_loss_low,
            "train_loss_high": train_loss_high,
            "val_loss_mean": val_loss_mean,
            "val_loss_low": val_loss_low,
            "val_loss_high": val_loss_high,
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats1 = load_model_stats(model_id_1, smooth_window)
    stats2 = load_model_stats(model_id_2, smooth_window)

    epochs = min(stats1["epochs"], stats2["epochs"])
    x = np.arange(epochs)

    fig, axes = plt.subplots(1, 3, sharex=True)

    # ---------------- Residual ----------------
    ax = axes[0]

    # Diffusion
    ax.plot(x, stats1["res_mean"][:epochs],
            label="Diffusion mean",
            linewidth=2.2,
            color=diffusion_color)
    ax.fill_between(
        x,
        stats1["res_low"][:epochs],
        stats1["res_high"][:epochs],
        alpha=0.3,
        color=diffusion_ci,
        label="Diffusion 95% CI",
    )

    # PIDM
    ax.plot(x, stats2["res_mean"][:epochs],
            label="PIDM mean",
            linewidth=2.2,
            color=pidm_color)
    ax.fill_between(
        x,
        stats2["res_low"][:epochs],
        stats2["res_high"][:epochs],
        alpha=0.3,
        color=pidm_ci,
        label="PIDM 95% CI",
    )

    ax.set_xlabel(f"Epoch (smoothed, window={smooth_window})")
    ax.set_ylabel(r"$\mathcal{R}_{\text{MAE}}(\mathbf{x_0}) \sim p_\theta (\mathbf{x_0})$")
    ax.set_title("Validation Darcy Residual")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    # ---------------- MSE ----------------
    ax = axes[1]

    # Diffusion
    ax.plot(x, stats1["mse_mean"][:epochs],
            label="Diffusion mean",
            linewidth=2.2,
            color=diffusion_color)
    ax.fill_between(
        x,
        stats1["mse_low"][:epochs],
        stats1["mse_high"][:epochs],
        alpha=0.3,
        color=diffusion_ci,
        label="Diffusion 95% CI",
    )

    # PIDM
    ax.plot(x, stats2["mse_mean"][:epochs],
            label="PIDM mean",
            linewidth=2.2,
            color=pidm_color)
    ax.fill_between(
        x,
        stats2["mse_low"][:epochs],
        stats2["mse_high"][:epochs],
        alpha=0.3,
        color=pidm_ci,
        label="PIDM 95% CI",
    )

    ax.set_xlabel(f"Epoch (smoothed, window={smooth_window})")
    ax.set_ylabel(r"$\mathbb{E}_{t, \mathbf{x_0}}[\lambda_t \|\mathbf{x_0} - \hat{\mathbf{x}}_0\|^2]$")
    ax.set_title("Validation Weighted MSE")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    ax = axes[2]

    # Train loss
    ax.plot(x, stats1["train_loss_mean"][:epochs],
            label="Train loss mean",
            linewidth=2.2,
            color=train_loss_diff_color)
    ax.fill_between(
        x,
        stats1["train_loss_low"][:epochs],
        stats1["train_loss_high"][:epochs],
        alpha=0.3,
        color=train_loss_diff_ci,
        label="Train loss 95% CI (Diffusion)",
    )

    # Val loss
    ax.plot(x, stats1["val_loss_mean"][:epochs],
            label="Val loss mean",
            linewidth=2.2,
            color=val_loss_diff_color)
    ax.fill_between(
        x,
        stats1["val_loss_low"][:epochs],
        stats1["val_loss_high"][:epochs],
        alpha=0.3,
        color=val_loss_diff_ci,
        label="Val loss 95% CI (Diffusion)",
    )

    ax.plot(x, stats2["train_loss_mean"][:epochs],
            label="Train loss mean",
            linewidth=2.2,
            color=train_loss_pidm_color)
    ax.fill_between(
        x,
        stats2["train_loss_low"][:epochs],
        stats2["train_loss_high"][:epochs],
        alpha=0.3,
        color=train_loss_pidm_ci,
        label="Train loss 95% CI (PIDM)",
    )

    # Val loss
    ax.plot(x, stats2["val_loss_mean"][:epochs],
            label="Val loss mean",
            linewidth=2.2,
            color=val_loss_pidm_color)
    ax.fill_between(
        x,
        stats2["val_loss_low"][:epochs],
        stats2["val_loss_high"][:epochs],
        alpha=0.3,
        color=val_loss_pidm_ci,
        label="Val loss 95% CI (PIDM)",
    )

    ax.set_xlabel(f"Epoch (smoothed, window={smooth_window})")
    ax.set_ylabel(r"$\mathbb{E}_{t, \mathbf{x_0}}[\lambda_t \|\mathbf{x_0} - \hat{\mathbf{x}}_0\|^2] + \frac{1}{2 \tilde{\Sigma}} || \mathcal{R}(\mathbf{x}_0^*)(\mathbf{x}_t,t)||^2$")
    ax.set_title("Train vs Validation loss")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")


    # Improve spacing
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    save_path = out_dir / f"{model_id_1}_vs_{model_id_2}_darcy_val_metrics.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Darcy validation metrics plot to {save_path}")

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

    res_df_geo = df.dropna(subset=["val_era5_geo_wind_residual"])
    res_df_planetary = df.dropna(subset=["val_era5_planetary_residual"])
    res_df_qgpv = df.dropna(subset=["val_era5_qgpv_residual"])
    mse_df = df.dropna(subset=["val_mse_(weighted)"])
    data_geo_scaled = res_df_geo.val_era5_geo_wind_residual / res_df_geo.val_era5_geo_wind_residual[0]
    data_planetary_scaled = res_df_planetary.val_era5_planetary_residual / res_df_planetary.val_era5_planetary_residual[0]
    data_qgpv_scaled = res_df_qgpv.val_era5_qgpv_residual / res_df_qgpv.val_era5_qgpv_residual[0]

    # ---- Residual + MSE plot ----
    plt.plot(res_df_geo.epoch, data_geo_scaled, label="geo_wind_residual (scaled)")
    plt.plot(res_df_planetary.epoch, data_planetary_scaled, label="planetary_residual (scaled)")
    plt.plot(res_df_qgpv.epoch, data_qgpv_scaled, label="qgpv_residual (scaled)")
    plt.plot(mse_df.epoch, mse_df["val_mse_(weighted)"], label="mse_weighted")
    plt.legend(); plt.grid(); plt.title("Residuals & MSE")
    plt.savefig(out_dir / "residuals.png"); plt.clf()

    return


if __name__ == "__main__":
    from pde_diff.utils import DatasetRegistry, LossRegistry

    model_path = Path('./models')
    model_id = 'exp1-aaaad'
    model_id_2 = 'exp1-aaaaf'
    # cfg = OmegaConf.load(model_path / (model_id) / "config.yaml")
    # diffusion_model = DiffusionModel(cfg)
    # diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")
    # plot_darcy_samples(diffusion_model, model_id, Path('./reports/figures') / model_id)

    plot_darcy_val_metrics(
        model_id_1=model_id,
        model_id_2=model_id_2,
        fold_num=5,
        log_path="logs",
        out_dir=f"reports/figures/{model_id}",
        smooth_window=20,
    )
    breakpoint()

    cfg = OmegaConf.load(model_path / model_id / "config.yaml")
    dataset = DatasetRegistry.create(cfg.dataset)
    diffusion_model = DiffusionModel(cfg)
    diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")
    diffusion_model = diffusion_model.to('cuda' if torch.cuda.is_available() else 'cpu')

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

