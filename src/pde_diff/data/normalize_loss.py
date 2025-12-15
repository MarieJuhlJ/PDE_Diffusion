import torch
from pde_diff.loss import VorticityLoss
from pde_diff.data.datasets import ERA5Dataset
from pde_diff.utils import init_means_and_stds_era5
from omegaconf import OmegaConf
import json
from pathlib import Path


def to_tensor(x, device):
    # Handles numpy arrays or already-a-tensor
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(x)
    # make sure we’re in floating point (often needed for PDE ops)
    if not torch.is_floating_point(t):
        t = t.float()
    return t.to(device, non_blocking=True)


def update_stats(x, stats):
    x = x.detach()
    stats["count"] += x.numel()
    stats["sum"] += x.sum().item()
    stats["sumsq"] += (x ** 2).sum().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cfg = OmegaConf.load("./configs/dataset/era5.yaml")
    loss_cfg = OmegaConf.create({
        "name": "VorticityLoss",
        "c_residual": [1.0, 1.0, 1.0],
        # if your loss reads a device entry, this helps; if not, it’s harmless
        "device": str(device),
    })

    loss_fn = VorticityLoss(cfg=loss_cfg)
    dataset = ERA5Dataset(cfg=cfg)
    means, stds, diff_means, diff_stds = init_means_and_stds_era5(cfg.atmospheric_features, cfg.single_features, cfg.static_features)
    loss_fn.set_mean_and_std(means, stds, diff_means, diff_stds)
    

    stats = {
        "geostrophic_wind": {"count": 0, "sum": 0.0, "sumsq": 0.0},
        "planetary_vorticity": {"count": 0, "sum": 0.0, "sumsq": 0.0},
        "qgpv": {"count": 0, "sum": 0.0, "sumsq": 0.0},
    }

    with torch.no_grad():
        for data in dataset:
            prev_np, curr_np = data[0][None, 19:34], data[1][None, :]

            prev = to_tensor(prev_np, device)
            curr = to_tensor(curr_np, device)

            r_gw = loss_fn.compute_residual_geostrophic_wind(prev, curr, normalize=False)
            r_pv = loss_fn.compute_residual_planetary_vorticity(prev, curr, normalize=False)
            r_qg = loss_fn.compute_residual_qgpv(prev, curr, normalize=False)

            update_stats(r_gw, stats["geostrophic_wind"])
            update_stats(r_pv, stats["planetary_vorticity"])
            update_stats(r_qg, stats["qgpv"])

            del prev, curr, r_gw, r_pv, r_qg

    results = {}
    for key, s in stats.items():
        mean = s["sum"] / s["count"]
        var = s["sumsq"] / s["count"] - mean ** 2
        std = var ** 0.5
        results[key] = {"mean": mean, "std": std}

    out_path = Path("src/pde_diff/data/residual_stats.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved residual statistics to {out_path}")
