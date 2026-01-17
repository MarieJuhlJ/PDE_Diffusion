import random
import string
import numpy as np
import torch
import torch.nn.functional as F
import pde_diff.data.const as const
from types import SimpleNamespace

_ALPHABET = string.ascii_lowercase

class LayerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(layer_cls):
            cls._registry[name] = layer_cls
            return layer_cls
        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown layer type: {name}")
        return cls._registry[name](*args, **kwargs)

class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(layer_cls):
            cls._registry[name] = layer_cls
            return layer_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown dataset: {cfg.name}")
        return cls._registry[cfg.name](cfg)

class LossRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(loss_cls):
            cls._registry[name] = loss_cls
            return loss_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown loss function: {cfg.name}")
        return cls._registry[cfg.name](cfg)

class SchedulerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(scheduler_cls):
            cls._registry[name] = scheduler_cls
            return scheduler_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown scheduler: {cfg.name}")
        return cls._registry[cfg.name](cfg)

class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if type(cfg) == list:
            return cls._registry[cfg[0].name](cfg)
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown model: {cfg.name}")
        return cls._registry[cfg.name](cfg)
    
class CallbackRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(callback_cls):
            cls._registry[name] = callback_cls
            return callback_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name == None:
            return None
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown callback: {cfg.name}")
        return cls._registry[cfg.name](cfg)

def unique_id(existing: set[str] | None = None, length: int = 5) -> str:
    """
    Return a random a–z ID of `length` letters that isn't in `existing`.
    `existing` should be a set of already-issued IDs (optional).
    """
    existing = existing or set()
    while True:
        uid = ''.join(random.choices(_ALPHABET, k=length))
        if uid not in existing:
            return uid

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d

def init_means_and_stds_era5(atmospheric_features, single_features, static_features):
    # TODO: Handle missing features more gracefully aka actually implement them
    means = []
    stds = []
    diff_means = []
    diff_stds = []

    for var in atmospheric_features:
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

    for var in single_features:
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

    for var in static_features:
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

class GradientHelper:
    def __init__(self, grid_distances):
        self.grid_distances = grid_distances

    def d_in_dx(self, input):
        b, lev, lon, lat = input.shape
        
        inp = F.pad(input, (0, 0, 1, 1), mode='circular')

        kernel = torch.tensor([-1, 0, 1], dtype=input.dtype, device=input.device) / 2.0
        kernel = kernel.view(1, 1, 3, 1)

        dx = F.conv2d(
            inp.view(b * lev, 1, lon + 2, lat),
            kernel,
            padding=0,
        )
        dx = dx.view(b, lev, lon, lat)
        return dx / (self.grid_distances['dx'])

    def d_in_dy(self, input):
        b, lev, lon, lat = input.shape

        inp = F.pad(input, (1, 1, 0, 0), mode='reflect')

        kernel = torch.tensor([-1, 0, 1], dtype=input.dtype, device=input.device) / 2.0
        kernel = kernel.view(1, 1, 1, 3)

        dy = F.conv2d(
            inp.view(b * lev, 1, lon, lat + 2),
            kernel,
            padding=0,
        )
        dy = dy.view(b, lev, lon, lat)
        return dy / (self.grid_distances['dy'])

    def d2_in_dx2(self, input):
        b, lev, lon, lat = input.shape
        inp = F.pad(input, (0, 0, 1, 1), mode="circular")

        k = torch.tensor([1., -2., 1.], dtype=input.dtype, device=input.device).view(1, 1, 3, 1)
        out = F.conv2d(inp.view(b * lev, 1, lon + 2, lat), k, padding=0).view(b, lev, lon, lat)

        dx = self.grid_distances["dx"]
        return out / (dx * dx)

    def d2_in_dy2(self, input):
        b, lev, lon, lat = input.shape
        inp = F.pad(input, (1, 1, 0, 0), mode="reflect")

        k = torch.tensor([1., -2., 1.], dtype=input.dtype, device=input.device).view(1, 1, 1, 3)
        out = F.conv2d(inp.view(b * lev, 1, lon, lat + 2), k, padding=0).view(b, lev, lon, lat)

        dy = self.grid_distances["dy"]
        return out / (dy * dy)

    def laplacian_horizontal(self, input):
        return self.d2_in_dx2(input) + self.d2_in_dy2(input)

    def gradient_horizontal(self, input):
        dx = self.d_in_dx(input)
        dy = self.d_in_dy(input)
        return dx, dy

if __name__ == "__main__":
    random_point = torch.randn(1, 3, 480, 32)

    grid_distances = {
        'dx': torch.ones_like(random_point),
        'dy': torch.ones_like(random_point),
    }

    gh = GradientHelper(grid_distances)

    dx = gh.d_in_dx(random_point)
    dy = gh.d_in_dy(random_point)
    lap = gh.laplacian_horizontal(random_point)
    gradx, grady = gh.gradient_horizontal(random_point)

    print(dx.shape, dy.shape, lap.shape, gradx.shape, grady.shape)
