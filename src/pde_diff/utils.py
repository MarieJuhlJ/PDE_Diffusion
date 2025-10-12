from types import SimpleNamespace

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
