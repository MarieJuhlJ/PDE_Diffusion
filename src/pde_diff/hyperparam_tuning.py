"""
Running this script will launch an Optuna hyperparameter optimization study.
"""

import gc
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import hydra
import torch

import lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from pde_diff.utils import DatasetRegistry, LossRegistry, unique_id
from pde_diff.model import DiffusionModel
from pde_diff.callbacks import SaveBestModel
from pde_diff.data.utils import split_dataset

class ObjectiveFunction(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.cfg.id = unique_id(length=5) if cfg.id == None else cfg.id
        self.cfg.model.dims = self.cfg.dataset.dims

        pl.seed_everything(self.cfg.seed)

    def __call__(self, trial: optuna.trial.Trial):
        # Suggest hyperparameters
        self.suggest_from_config(trial, self.cfg)

        dataset = DatasetRegistry.create(self.cfg.dataset)
        model = DiffusionModel(self.cfg)

        dataset_train, dataset_val = split_dataset(self.cfg, dataset)

        train_dataloader = DataLoader(dataset_train, batch_size=self.cfg.experiment.hyperparameters.batch_size, shuffle=True, num_workers=4,persistent_workers=True)
        val_dataloader = DataLoader(dataset_val, batch_size=self.cfg.experiment.hyperparameters.batch_size, shuffle=False, num_workers=4,persistent_workers=True)

        acc = "gpu" if torch.cuda.is_available() else "cpu"

        logger = pl.pytorch.loggers.CSVLogger("logs", name=f"optuna-{self.cfg.experiment.name}-{self.cfg.id}")

        early_stop_callback = pl.pytorch.callbacks.early_stopping.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=False,
            mode='min'
        )

        trainer = pl.Trainer(
            accelerator=acc,
            max_epochs=self.cfg.experiment.hyperparameters.max_epochs,
            enable_checkpointing=False,
            logger=logger,
            log_every_n_steps=self.cfg.experiment.hyperparameters.log_every_n_steps,
            callbacks=[early_stop_callback, PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        )

        trainer.fit(model, train_dataloader, val_dataloader)

        val_loss = trainer.callback_metrics["val_loss"].item()

        #try to clean up space
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return val_loss

    def suggest_from_config(self, trial: optuna.trial.Trial, config: DictConfig):
        for param_name, param_cfg in config.hp_study.params.items():
            print(f"Suggesting parameter {param_name} of type {param_cfg.type} with config {param_cfg}")
            if param_cfg.type == "categorical":
                suggestion = trial.suggest_categorical(param_name, list(param_cfg.choices))
            elif param_cfg.type == "float":
                suggestion = trial.suggest_float(param_name, param_cfg.low, param_cfg.high, log=param_cfg.get("log", False))
            elif param_cfg.type == "int":
                suggestion = trial.suggest_int(param_name, param_cfg.low, param_cfg.high, log=param_cfg.get("log", False))
            else:
                raise ValueError(f"Unknown parameter type: {param_cfg.type}")
            print(f"Trial {trial.number} suggests {param_name}: {suggestion}")
            self.cfg.experiment.hyperparameters[param_name] = suggestion

@hydra.main(version_base=None, config_name="config.yaml", config_path="../../configs")
def run_optimization(cfg: DictConfig):
    hp_cfg = cfg.hp_study

    storage_url = f"sqlite:///{hp_cfg.name}.db"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=hp_cfg.pruner.n_startup_trials,
                                         n_warmup_steps=hp_cfg.pruner.n_warmup_steps,
                                         n_min_trials=hp_cfg.pruner.n_min_trials)
    study = optuna.create_study(study_name=hp_cfg.name,
                                direction='minimize',
                                pruner=pruner,
                                storage=storage_url,
                                load_if_exists=True)
    study.optimize(ObjectiveFunction(cfg), n_trials=hp_cfg.n_trials, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study


if __name__ == "__main__":
    run_optimization()
