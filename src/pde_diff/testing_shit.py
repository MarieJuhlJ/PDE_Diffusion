import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pde_diff.utils import LossRegistry, DatasetRegistry, GradientHelper
from pde_diff.data import datasets
from pde_diff.loss import *


cfg = OmegaConf.load("./configs/dataset/era5.yaml")
cfg_loss = OmegaConf.create(
    {
        "name": "vorticity",
        "c_residual": 1.0,
        "device": "cpu",
    }
)

dataset = DatasetRegistry.create(cfg)

loss = LossRegistry.create(cfg_loss)
loss.set_mean_and_std(dataset.means, dataset.stds, dataset.diff_means, dataset.diff_stds)

losses = []
for i in range(1000):
    sample = torch.randn((15,480,32))
    sample_change = torch.randn((15,480,32))
    #print(sample.abs().mean())

    residual_planetary = loss.compute_residual_planetary_vorticity(sample, sample_change).abs().mean()
    residual_geo_wind = loss.compute_residual_geostrophic_wind(sample, sample_change).abs().mean()
    residual_qgpv = loss.compute_residual_qgpv(sample, sample_change).abs().mean()
    #print(residual_planetary, residual_geo_wind, residual_qgpv)
    losses.append([residual_planetary,residual_geo_wind,residual_qgpv])

losses = np.array([losses])
print(losses.shape)
print(losses.mean(1))
print(losses.std(1))
losses = []

for data in dataset:
    state1_np, target_np = data[0][None, 19:34], data[1]
    state1 = torch.from_numpy(state1_np)
    target = torch.from_numpy(target_np)

    residual_planetary = loss.compute_residual_planetary_vorticity(state1, target).abs().mean()
    residual_geo_wind = loss.compute_residual_geostrophic_wind(state1, target).abs().mean()
    residual_qgpv = loss.compute_residual_qgpv(state1, target).abs().mean()
    #print(residual_planetary, residual_geo_wind, residual_qgpv)
    losses.append([residual_planetary,residual_geo_wind,residual_qgpv])

losses = np.array([losses])
print(losses.shape)
print(losses.mean(1))
print(losses.std(1))
