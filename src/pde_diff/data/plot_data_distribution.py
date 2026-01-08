import torch
from pde_diff.data.datasets import ERA5Dataset
from pde_diff.utils import LossRegistry
from pde_diff.loss import VorticityLoss
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt


def to_tensor(x, device):
    # Handles numpy arrays or already-a-tensor
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(x)
    # Ensure floating point
    if not torch.is_floating_point(t):
        t = t.float()
    return t.to(device, non_blocking=True)


def append_hist(x, buf, max_samples: int):
    """
    Append flattened residual values to a Python list up to max_samples.
    Prevents unbounded RAM usage on large datasets.
    """
    x = x.detach().flatten().cpu()
    if len(buf) >= max_samples:
        return
    remaining = max_samples - len(buf)
    buf.extend(x[:remaining].tolist())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cfg = OmegaConf.load("./configs/dataset/era5.yaml")
    cfg_loss = OmegaConf.create(
        {
            "name": "vorticity",
            "c_residual": 1.0,
        }
    )

    loss_fn = LossRegistry.create(cfg_loss)
    dataset = ERA5Dataset(cfg=cfg)
    loss_fn.set_mean_and_std(dataset.means, dataset.stds, dataset.diff_means, dataset.diff_stds)

    # Store raw (unnormalized) residual values for histograms
    hist_data = {
        "geostrophic_wind": [],
        "planetary_vorticity": [],
        "qgpv": [],
    }

    # Cap number of samples to avoid excessive memory usage
    MAX_SAMPLES = 2_000_000

    with torch.no_grad():
        for data in tqdm(dataset, desc="Processing dataset"):
            prev_np, curr_np = data[0][None, 19:34], data[1][None, :]

            prev = to_tensor(prev_np, device)
            curr = to_tensor(curr_np, device)

            # Compute residuals (NOT normalized)
            r_gw = loss_fn.compute_residual_geostrophic_wind(prev, curr, normalize=False)
            r_pv = loss_fn.compute_residual_planetary_vorticity(prev, curr, normalize=False)
            r_qg = loss_fn.compute_residual_qgpv(prev, curr, normalize=False)

            append_hist(r_gw, hist_data["geostrophic_wind"], MAX_SAMPLES)
            append_hist(r_pv, hist_data["planetary_vorticity"], MAX_SAMPLES)
            append_hist(r_qg, hist_data["qgpv"], MAX_SAMPLES)

            del prev, curr, r_gw, r_pv, r_qg

    # --- Plot 3 histograms side-by-side (counts, not normalized) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    plots = [
        ("qgpv", r"\mathcal{R}_1 QGPV Residual"),
        ("planetary_vorticity", r"\mathcal{R}_2 Planetary Vorticity Residual"),
        ("geostrophic_wind", r"\mathcal{R}_3 Geostrophic Wind Residual"),
    ]

import numpy as np

for ax, (key, title) in zip(axes, plots):
    data = np.asarray(hist_data[key])

    mean = data.mean()
    std = data.std()

    # Robust x-limits to avoid extreme tails dominating
    lo, hi = np.percentile(data, [0.5, 99.5])

    # Histogram
    ax.hist(
        data,
        bins=120,
        range=(lo, hi),
        histtype="stepfilled",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.8,
    )

    # Mean and std lines
    ax.axvline(mean, linestyle="-", linewidth=2, label="Mean")
    ax.axvline(mean - std, linestyle="--", linewidth=1.5, label="±1 Std")
    ax.axvline(mean + std, linestyle="--", linewidth=1.5)

    # Annotation box
    textstr = (
        f"Mean = {mean:.3e}\n"
        f"Std  = {std:.3e}"
    )
    ax.text(
        0.97,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Residual value")
    ax.set_xlim(lo, hi)

    ax.grid(True, linestyle=":", alpha=0.6)

axes[0].set_ylabel("Count")

plt.tight_layout()
plt.savefig("reports/figures/era5_residual_histograms.png", dpi=300)
print("Saved histogram figure to reports/figures/era5_residual_histograms.png")
