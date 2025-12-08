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
        batch_size = trial.suggest_categorical("batch_size", [4,8,16])
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        self.cfg.experiment.hyperparameters.batch_size = batch_size
        self.cfg.experiment.hyperparameters.lr = learning_rate
        self.cfg.experiment.hyperparameters.weight_decay = weight_decay

        dataset = DatasetRegistry.create(self.cfg.dataset)
        model = DiffusionModel(self.cfg)

        dataset_train, dataset_val = split_dataset(self.cfg, dataset)

        train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,persistent_workers=True)
        val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4,persistent_workers=True)

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

@hydra.main(version_base=None, config_name="config.yaml", config_path="../../configs")
def run_optimization(cfg: DictConfig):
    storage_url = "sqlite:///hp_param_study.db"

    cfg.n_trials = cfg.get("n_trials", 20)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, n_min_trials=5)
    study = optuna.create_study(study_name="hp_tuning", direction='minimize', pruner=pruner, storage=storage_url, load_if_exists=True)
    study.optimize(ObjectiveFunction(cfg), n_trials=cfg.n_trials, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study


if __name__ == "__main__":
    run_optimization()
