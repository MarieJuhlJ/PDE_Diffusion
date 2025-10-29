import random
import string
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
