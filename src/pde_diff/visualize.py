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
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

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

def plot_darcy_val_metrics(
    model_id_1,
    model_id_2,
    fold_num,
    log_path,
    out_dir,
    smooth_window=10,
    dpi=300,
):
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

    # --- Colors ---
    diffusion_color = "#8904DC"
    diffusion_ci    = "#A658D6"
    pidm_color      = "#DB0105"
    pidm_ci         = "#D86567"

    train_loss_pidm_color = "#E7002A"
    train_loss_pidm_ci    = "#FB3D60"
    val_loss_pidm_color   = "#E20C69"
    val_loss_pidm_ci      = "#E13958"

    train_loss_diff_color = "#3915EB"
    train_loss_diff_ci    = "#684EEC"
    val_loss_diff_color   = "#1599EB"
    val_loss_diff_ci      = "#53ADE5"

    # ---------------- Helpers ----------------
    def moving_average_2d(arr: np.ndarray, window: int) -> np.ndarray:
        """Moving average along time axis for 2D array [fold, time]. Returns shorter 'valid' length."""
        if window <= 1:
            return arr
        kernel = np.ones(window) / window
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=arr)

    def _read_metric_series(df: pd.DataFrame, col: str) -> np.ndarray:
        """Read a column as float array, drop NaNs."""
        if col not in df.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)

    def _stack_and_trim(series_list):
        """Stack 1D arrays into 2D [fold, time] by trimming all to the minimum length."""
        if len(series_list) == 0:
            return np.empty((0, 0), dtype=float)
        lengths = [len(s) for s in series_list if s is not None]
        if len(lengths) == 0 or min(lengths) == 0:
            return np.empty((0, 0), dtype=float)
        L = min(lengths)
        trimmed = [s[:L] for s in series_list]
        return np.vstack(trimmed)

    def _mean_ci(arr2d: np.ndarray):
        """
        Compute mean and 95% CI over folds at each time.
        Returns (mean, low, high). If folds==0 => empty arrays.
        """
        if arr2d.size == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty
        n = arr2d.shape[0]
        mean = arr2d.mean(axis=0)
        std = arr2d.std(axis=0)
        half = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
        return mean, mean - half, mean + half

    def load_model_stats(model_id: str, smooth_window: int):
        residuals, mses, train_losses, val_losses = [], [], [], []

        for fold in range(1, fold_num + 1):
            current_model_id = f"{model_id}-{fold}"
            csv_path = Path(log_path) / current_model_id / "version_0" / "metrics.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing metrics.csv: {csv_path}")

            df = (
                pd.read_csv(csv_path)
                .apply(pd.to_numeric, errors="ignore")
            )

            # Prefer sorting by step if present
            if "step" in df.columns:
                df = df.dropna(subset=["step"]).sort_values("step")

            residuals.append(_read_metric_series(df, "val_darcy_residual"))
            mses.append(_read_metric_series(df, "val_mse_(weighted)"))
            train_losses.append(_read_metric_series(df, "train_loss"))
            val_losses.append(_read_metric_series(df, "val_loss"))

        # Stack + trim so folds align in time
        residuals = _stack_and_trim(residuals)
        mses = _stack_and_trim(mses)
        train_losses = _stack_and_trim(train_losses)
        val_losses = _stack_and_trim(val_losses)

        # Smooth (valid mode => shorter length)
        residuals = moving_average_2d(residuals, smooth_window)
        mses = moving_average_2d(mses, smooth_window)
        train_losses = moving_average_2d(train_losses, smooth_window)
        val_losses = moving_average_2d(val_losses, smooth_window)

        res_mean, res_low, res_high = _mean_ci(residuals)
        mse_mean, mse_low, mse_high = _mean_ci(mses)
        tr_mean, tr_low, tr_high = _mean_ci(train_losses)
        va_mean, va_low, va_high = _mean_ci(val_losses)

        epochs = min(
            len(res_mean) if len(res_mean) else np.inf,
            len(mse_mean) if len(mse_mean) else np.inf,
            len(tr_mean) if len(tr_mean) else np.inf,
            len(va_mean) if len(va_mean) else np.inf,
        )
        epochs = int(epochs) if epochs != np.inf else 0

        return {
            "epochs": epochs,
            "res_mean": res_mean, "res_low": res_low, "res_high": res_high,
            "mse_mean": mse_mean, "mse_low": mse_low, "mse_high": mse_high,
            "train_loss_mean": tr_mean, "train_loss_low": tr_low, "train_loss_high": tr_high,
            "val_loss_mean": va_mean, "val_loss_low": va_low, "val_loss_high": va_high,
        }

    # ---------------- Plotting functions (reusable) ----------------
    def plot_residual(ax, x, s1, s2, epochs):
        ax.plot(x, s1["res_mean"][:epochs], label="Diffusion mean", linewidth=2.2, color=diffusion_color)
        ax.fill_between(x, s1["res_low"][:epochs], s1["res_high"][:epochs], alpha=0.3, color=diffusion_ci, label="Diffusion 95% CI")

        ax.plot(x, s2["res_mean"][:epochs], label="PIDM mean", linewidth=2.2, color=pidm_color)
        ax.fill_between(x, s2["res_low"][:epochs], s2["res_high"][:epochs], alpha=0.3, color=pidm_ci, label="PIDM 95% CI")

        ax.set_xlabel(f"Epoch (smoothed, window={smooth_window})")
        ax.set_ylabel(r"$\mathcal{R}_{\text{MAE}}(\mathbf{x_0}) \sim p_\theta (\mathbf{x_0})$")
        ax.set_title("Validation Darcy Residual")
        ax.set_yscale("log")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    def plot_mse(ax, x, s1, s2, epochs):
        ax.plot(x, s1["mse_mean"][:epochs], label="Diffusion mean", linewidth=2.2, color=diffusion_color)
        ax.fill_between(x, s1["mse_low"][:epochs], s1["mse_high"][:epochs], alpha=0.3, color=diffusion_ci, label="Diffusion 95% CI")

        ax.plot(x, s2["mse_mean"][:epochs], label="PIDM mean", linewidth=2.2, color=pidm_color)
        ax.fill_between(x, s2["mse_low"][:epochs], s2["mse_high"][:epochs], alpha=0.3, color=pidm_ci, label="PIDM 95% CI")

        ax.set_xlabel(f"Epoch (smoothed, window={smooth_window})")
        ax.set_ylabel(r"$\mathbb{E}_{t, \mathbf{x_0}}[\lambda_t \|\mathbf{x_0} - \hat{\mathbf{x}}_0\|^2]$")
        ax.set_title("Validation Weighted MSE")
        ax.set_yscale("log")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    def plot_train_val_loss(ax, x, s1, s2, epochs):
        # Diffusion train/val
        ax.plot(x, s1["train_loss_mean"][:epochs], label="Train loss mean (Diffusion)", linewidth=2.2, color=train_loss_diff_color)
        ax.fill_between(x, s1["train_loss_low"][:epochs], s1["train_loss_high"][:epochs], alpha=0.3, color=train_loss_diff_ci, label="Train loss 95% CI (Diffusion)")

        ax.plot(x, s1["val_loss_mean"][:epochs], label="Val loss mean (Diffusion)", linewidth=2.2, color=val_loss_diff_color)
        ax.fill_between(x, s1["val_loss_low"][:epochs], s1["val_loss_high"][:epochs], alpha=0.3, color=val_loss_diff_ci, label="Val loss 95% CI (Diffusion)")

        # PIDM train/val
        ax.plot(x, s2["train_loss_mean"][:epochs], label="Train loss mean (PIDM)", linewidth=2.2, color=train_loss_pidm_color)
        ax.fill_between(x, s2["train_loss_low"][:epochs], s2["train_loss_high"][:epochs], alpha=0.3, color=train_loss_pidm_ci, label="Train loss 95% CI (PIDM)")

        ax.plot(x, s2["val_loss_mean"][:epochs], label="Val loss mean (PIDM)", linewidth=2.2, color=val_loss_pidm_color)
        ax.fill_between(x, s2["val_loss_low"][:epochs], s2["val_loss_high"][:epochs], alpha=0.3, color=val_loss_pidm_ci, label="Val loss 95% CI (PIDM)")

        ax.set_xlabel(f"Epoch (smoothed, window={smooth_window})")
        ax.set_ylabel(r"$\mathbb{E}_{t, \mathbf{x_0}}[\lambda_t \|\mathbf{x_0} - \hat{\mathbf{x}}_0\|^2] + \frac{1}{2 \tilde{\Sigma}} || \mathcal{R}(\mathbf{x}_0^*)(\mathbf{x}_t,t)||^2$")
        ax.set_title("Train vs Validation loss")
        ax.set_yscale("log")
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right")

    def save_single(plot_fn, filename, figsize=(6.5, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        plot_fn(ax, x, stats1, stats2, epochs)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # ---------------- Main ----------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats1 = load_model_stats(model_id_1, smooth_window)
    stats2 = load_model_stats(model_id_2, smooth_window)

    epochs = min(stats1["epochs"], stats2["epochs"])
    if epochs <= 0:
        raise ValueError("No epochs to plot after trimming/smoothing. Check your CSVs and smooth_window.")

    x = np.arange(epochs)

    # Combined figure
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(13, 5))
    plot_residual(axes[0], x, stats1, stats2, epochs)
    plot_mse(axes[1], x, stats1, stats2, epochs)
    plot_train_val_loss(axes[2], x, stats1, stats2, epochs)

    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    combined_path = out_dir / f"{model_id_1}_vs_{model_id_2}_darcy_val_metrics.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Individual subplots
    save_single(plot_residual,     f"{model_id_1}_vs_{model_id_2}_darcy_residual.png")
    save_single(plot_mse,          f"{model_id_1}_vs_{model_id_2}_weighted_mse.png")
    save_single(plot_train_val_loss, f"{model_id_1}_vs_{model_id_2}_train_vs_val_loss.png")

    print(f"Saved combined plot to: {combined_path}")
    print(f"Saved individual plots to: {out_dir}")

def plot_and_save_era5(csv_path, out_dir, loss_title="Loss", residual_title="Residuals & MSE", log_scale=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = (
        pd.read_csv(csv_path)
        .apply(pd.to_numeric, errors="ignore")
        .dropna(subset=["step"])
        .sort_values("step")
    )

    # ---- Loss plot ----
    train_df = df.dropna(subset=["train_loss"])
    val_df = df.dropna(subset=["val_loss"])

    plt.plot(train_df.epoch, train_df.train_loss, label="train_loss")
    plt.plot(val_df.epoch, val_df.val_loss, label="val_loss")

    if log_scale:
        plt.yscale("log")

    plt.legend()
    plt.grid(True, which="both")
    plt.title(loss_title)
    plt.savefig(out_dir / "loss.png", bbox_inches="tight")
    plt.clf()
    print(f"saved fig to {out_dir}/loss.png")

    # ---- Residual + MSE plot ----
    res_df_geo = df.dropna(subset=["val_era5_geo_wind_residual(norm)"])
    res_df_planetary = df.dropna(subset=["val_era5_planetary_residual(norm)"])
    res_df_qgpv = df.dropna(subset=["val_era5_qgpv_residual(norm)"])
    mse_df = df.dropna(subset=["val_mse_(weighted)"])

    plt.plot(res_df_geo.epoch, res_df_geo["val_era5_geo_wind_residual(norm)"], label="geo_wind_residual (normalized)")
    plt.plot(res_df_planetary.epoch, res_df_planetary["val_era5_planetary_residual(norm)"], label="planetary_residual (normalized)")
    plt.plot(res_df_qgpv.epoch, res_df_qgpv["val_era5_qgpv_residual(norm)"], label="qgpv_residual (normalized)")
    plt.plot(mse_df.epoch, mse_df["val_mse_(weighted)"], label="mse_weighted")

    if log_scale:
        plt.yscale("log")

    plt.legend()
    plt.grid(True, which="both")
    plt.title(residual_title)
    plt.savefig(out_dir / "residuals.png", bbox_inches="tight")
    plt.clf()
    print(f"saved fig to {out_dir}/residuals.png")

def darcy_models_summary_latex_table(
    model_ids,
    fold_num,
    log_path,
    smooth_window=10,
    caption="Model comparison (best epoch; mean $\\pm$ 95\\% CI across folds)",
    label="tab:model_summary",
    sig_figs=3,
    wrap_math=True,
):
    """
    Returns a LaTeX table (string) summarizing models with:
      - Best (min) Darcy residual (mean ± 95% CI)
      - Best (min) weighted MSE (mean ± 95% CI)
      - Darcy residual evaluated at the epoch where weighted MSE is best (mean ± 95% CI)
      - Best (min) val loss (mean ± 95% CI)

    CI half-width = 1.96 * std / sqrt(n_folds), computed across folds at each epoch.
    Smoothing is moving average (valid mode) along the epoch axis, applied per-fold.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import math

    # --------- metrics to load ----------
    COL_RES = "val_darcy_residual"
    COL_WMSE = "val_mse_(weighted)"
    COL_VLOSS = "val_loss"
    required_cols = [COL_RES, COL_WMSE, COL_VLOSS]

    # --------- helpers ----------
    def moving_average_2d(arr: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return arr
        kernel = np.ones(window) / window
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=arr)

    def _read_metric_series(df: pd.DataFrame, col: str) -> np.ndarray:
        if col not in df.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)

    def _stack_and_trim(series_list):
        lengths = [len(s) for s in series_list if s is not None]
        if len(lengths) == 0 or min(lengths) == 0:
            return np.empty((0, 0), dtype=float)
        L = min(lengths)
        trimmed = [s[:L] for s in series_list]
        return np.vstack(trimmed)

    def _mean_ci_halfwidth(arr2d: np.ndarray):
        if arr2d.size == 0:
            empty = np.array([], dtype=float)
            return empty, empty
        n = arr2d.shape[0]
        mean = arr2d.mean(axis=0)
        std = arr2d.std(axis=0)
        half = 1.96 * std / np.sqrt(n) if n > 0 else np.zeros_like(mean)
        return mean, half

    def tex_escape(s: str) -> str:
        return s.replace("_", r"\_")

    def to_latex_sci(x: float, sig_figs: int = 3) -> str:
        """Format as LaTeX scientific notation like 1.23\\times 10^{-4} (no e-04)."""
        if x == 0 or (isinstance(x, float) and (math.isfinite(x) is False)):
            # handle 0 / inf / nan simply
            if x == 0:
                return "0"
            return r"\mathrm{nan}" if math.isnan(x) else (r"\infty" if x > 0 else r"-\infty")

        sign = "-" if x < 0 else ""
        ax = abs(x)
        exp = int(math.floor(math.log10(ax)))
        mant = ax / (10 ** exp)

        # round mantissa to sig_figs
        mant_rounded = round(mant, sig_figs - 1)
        if mant_rounded >= 10:
            mant_rounded /= 10
            exp += 1

        # render mantissa without trailing zeros
        mant_str = f"{mant_rounded:.{max(sig_figs-1,0)}f}".rstrip("0").rstrip(".")
        if exp == 0:
            return f"{sign}{mant_str}"
        if mant_str == "1":
            return f"{sign}10^{{{exp}}}"
        return f"{sign}{mant_str}\\times 10^{{{exp}}}"

    def fmt_pm(mean: float, half: float) -> str:
        s = to_latex_sci(mean, sig_figs) + r" \pm " + to_latex_sci(half, sig_figs)
        return f"${s}$" if wrap_math else s

    def load_model_all_stats(model_id: str):
        """
        Load required metrics for all folds, align lengths across folds AND across metrics,
        smooth, then return mean/half arrays per metric.
        """
        per_col_series = {c: [] for c in required_cols}

        for fold in range(1, fold_num + 1):
            run_id = f"{model_id}-{fold}"
            csv_path = Path(log_path) / run_id / "version_0" / "metrics.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing metrics.csv: {csv_path}")

            df = pd.read_csv(csv_path).apply(pd.to_numeric, errors="ignore")
            if "step" in df.columns:
                df = df.dropna(subset=["step"]).sort_values("step")

            for c in required_cols:
                per_col_series[c].append(_read_metric_series(df, c))

        # stack each metric across folds (trim across folds)
        per_col_arr2d = {c: _stack_and_trim(per_col_series[c]) for c in required_cols}

        # align across metrics too (use shared time length)
        lengths = [per_col_arr2d[c].shape[1] for c in required_cols if per_col_arr2d[c].size > 0]
        if len(lengths) == 0:
            raise ValueError(f"No usable data found for model '{model_id}' in columns {required_cols}.")
        L = min(lengths)

        for c in required_cols:
            arr = per_col_arr2d[c]
            if arr.size == 0:
                raise ValueError(f"Metric '{c}' missing/empty for model '{model_id}'.")
            per_col_arr2d[c] = arr[:, :L]

        # smooth (valid) -> shorter
        for c in required_cols:
            per_col_arr2d[c] = moving_average_2d(per_col_arr2d[c], smooth_window)

        # after smoothing, lengths still identical across metrics (same L and same window)
        out = {}
        for c in required_cols:
            mean, half = _mean_ci_halfwidth(per_col_arr2d[c])
            out[c] = {"mean": mean, "half": half}

        return out

    def best_idx(mean: np.ndarray, direction: str = "min") -> int:
        return int(np.argmax(mean)) if direction == "max" else int(np.argmin(mean))

    # --------- compute per-model stats ----------
    model_stats = {mid: load_model_all_stats(mid) for mid in model_ids}

    # --------- build rows ----------
    # For each model:
    # - residual_best: best over residual mean
    # - wmse_best: best over wmse mean
    # - residual_at_best_wmse: residual at wmse_best epoch
    # - vloss_best: best over vloss mean
    rows = [
        {"key": "res_best", "label": r"Darcy residual (best)", "direction": "min"},
        {"key": "wmse_best", "label": r"Weighted MSE (best)", "direction": "min"},
        {"key": "res_at_wmse", "label": r"Darcy residual @ best Weighted MSE", "direction": "min"},
        {"key": "vloss_best", "label": r"Val loss (best)", "direction": "min"},
    ]

    # values[row_key][model_id] = dict(mean, half, epoch)
    values = {r["key"]: {} for r in rows}

    for mid in model_ids:
        res = model_stats[mid][COL_RES]
        wmse = model_stats[mid][COL_WMSE]
        vloss = model_stats[mid][COL_VLOSS]

        i_res = best_idx(res["mean"], "min")
        i_wmse = best_idx(wmse["mean"], "min")
        i_vloss = best_idx(vloss["mean"], "min")

        # shared length sanity
        T = min(len(res["mean"]), len(wmse["mean"]), len(vloss["mean"]))
        i_res = min(i_res, T - 1)
        i_wmse = min(i_wmse, T - 1)
        i_vloss = min(i_vloss, T - 1)

        values["res_best"][mid] = {"epoch": i_res, "mean": float(res["mean"][i_res]), "half": float(res["half"][i_res])}
        values["wmse_best"][mid] = {"epoch": i_wmse, "mean": float(wmse["mean"][i_wmse]), "half": float(wmse["half"][i_wmse])}
        values["res_at_wmse"][mid] = {"epoch": i_wmse, "mean": float(res["mean"][i_wmse]), "half": float(res["half"][i_wmse])}
        values["vloss_best"][mid] = {"epoch": i_vloss, "mean": float(vloss["mean"][i_vloss]), "half": float(vloss["half"][i_vloss])}

    # best model per row (based on mean, respecting direction)
    best_model_per_row = {}
    for r in rows:
        key = r["key"]
        direction = r.get("direction", "min")
        items = list(values[key].items())
        if direction == "max":
            best_mid = max(items, key=lambda kv: kv[1]["mean"])[0]
        else:
            best_mid = min(items, key=lambda kv: kv[1]["mean"])[0]
        best_model_per_row[key] = best_mid

    # --------- assemble LaTeX ----------
    header_cols = "l" + "c" * len(model_ids)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{" + header_cols + r"}")
    lines.append(r"\toprule")
    lines.append("Metric & " + " & ".join([tex_escape(mid) for mid in model_ids]) + r" \\")
    lines.append(r"\midrule")

    for r in rows:
        key = r["key"]
        row_cells = [r["label"]]
        for mid in model_ids:
            v = values[key][mid]
            cell = fmt_pm(v["mean"], v["half"]) + rf" (ep={v['epoch']})"
            if mid == best_model_per_row[key]:
                cell = r"\textbf{" + cell + "}"
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def era5_residuals_plot(model, conditional, model_id, normalize=True):
    loss_fn = model.loss_fn
    loss_fn.set_mean_and_std(diffusion_model.means, diffusion_model.stds, diffusion_model.diff_means, diffusion_model.diff_stds)
    model_sample = model.sample_loop(conditionals = conditional)
    loss_geo_wind = loss_fn.compute_residual_geostrophic_wind(x0_previous=conditional[:, 19:34], x0_change_pred=model_sample, normalize=normalize).abs().mean(1)
    loss_planetary = loss_fn.compute_residual_planetary_vorticity(x0_previous=conditional[:, 19:34], x0_change_pred=model_sample, normalize=normalize).abs().mean(1)
    loss_qgpv = loss_fn.compute_residual_qgpv(x0_previous=conditional[:, 19:34], x0_change_pred=model_sample, normalize=normalize).abs()
    print(f"mean loss geo wind: {loss_geo_wind.mean().item()}")
    print(f"mean loss planetary vorticity: {loss_planetary.mean().item()}")
    print(f"mean loss qgpv: {loss_qgpv.mean().item()}")
    residual_variables = ['geo_wind', 'planetary_vorticity', 'qgpv']
    residual_losses = [loss_geo_wind, loss_planetary, loss_qgpv]
    for res_var, res_loss in zip(residual_variables, residual_losses):
        residual = res_loss[0].cpu().numpy()
        visualize_era5_sample(residual, res_var, "500", big_data_sample=None, dir=Path("./reports/figures") / model_id / f"residuals")    

if __name__ == "__main__":
    from pde_diff.utils import DatasetRegistry, LossRegistry

    model_path = Path('./models')
    # model_id = 'exp1-aaaaa'
    # model_id_2 = 'exp1-aaaab'
    # cfg = OmegaConf.load(model_path / (model_id) / "config.yaml")
    # diffusion_model = DiffusionModel(cfg)
    # diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")
    # plot_darcy_samples(diffusion_model, model_id, Path('./reports/figures') / model_id)

    # plot_darcy_val_metrics(
    #     model_id_1=model_id,
    #     model_id_2=model_id_2,
    #     fold_num=5,
    #     log_path="logs",
    #     out_dir=f"reports/figures/{model_id}",
    #     smooth_window=20,
    # )
    # latex = darcy_models_summary_latex_table(
    # model_ids=[model_id, model_id_2],
    # fold_num=5,
    # log_path="logs",
    # smooth_window=10,
    # )
    # print(latex)

    model_id = 'exp1-jnham'
    model_id_2 = 'exp1-xosel'
    cfg = OmegaConf.load(model_path / model_id / "config.yaml")
    dataset = DatasetRegistry.create(cfg.dataset)
    diffusion_model = DiffusionModel(cfg)
    diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")
    diffusion_model = diffusion_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = Path('./logs') / model_id / 'version_0/metrics.csv'
    out_dir = Path('./reports') / 'figures' / model_id
    plot_and_save_era5(csv_path, out_dir, log_scale=True)
    
    #2nd model
    cfg_2 = OmegaConf.load(model_path / model_id_2 / "config.yaml")
    diffusion_model_2 = DiffusionModel(cfg_2)
    diffusion_model_2.load_model(model_path / model_id_2 / f"best-val_loss-weights.pt")
    diffusion_model_2 = diffusion_model_2.to('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = Path('./logs') / model_id_2 / 'version_0/metrics.csv'
    out_dir = Path('./reports') / 'figures' / model_id_2
    plot_and_save_era5(csv_path, out_dir, log_scale=True)
    breakpoint()

    #load loss
    loss_fn = LossRegistry.create(cfg.loss)
    loss_fn.set_mean_and_std(diffusion_model.means, diffusion_model.stds, diffusion_model.diff_means, diffusion_model.diff_stds)

    # Load the dataset configuration
    config_path = Path("configs/dataset/era5.yaml")
    cfg = OmegaConf.load(config_path)
    # cfg.normalize = True
    # Initialize the ERA5 dataset
    era5_dataset = ERA5Dataset(cfg)

    cfg.lat_range = None
    era5_dataset_full = ERA5Dataset(cfg)
    era5_dataset_full.normalization_on = False

    # Visualize all variables of a sample from the dataset
    sample_idx = 2  # Index of the sample to visualize
    variable = "t"  # Variable to visualize
    level = 500  # Pressure level in hPa
    conditionals = torch.from_numpy(era5_dataset[10][sample_idx - 2][None, :]).to("cuda")
    model_sample = diffusion_model.sample_loop(conditionals = conditionals)
    prev, current = loss_fn.get_original_states(x0_previous=conditionals[:, 19:34], x0_change_pred=model_sample)
    
    print("Diffusion Model Residuals:")
    print(f"Model ID: {model_id}")
    era5_residuals_plot(diffusion_model, conditionals, model_id, normalize=True)
    print(f"Model ID: {model_id_2}")
    era5_residuals_plot(diffusion_model_2, conditionals, model_id_2, normalize=True)
    
    dir = Path("./reports/figures") / model_id
    for idx, variable in enumerate(cfg.atmospheric_features):
        data_sample = current[0,1,idx].cpu()
        data_sample_full = get_data_sample(era5_dataset_full, sample_idx, variable, level)
        visualize_era5_sample(data_sample, variable, level, big_data_sample=data_sample_full, dir=dir)

    breakpoint()
    """
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

