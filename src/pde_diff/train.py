import os
import hydra
import torch

import lightning as pl
import torch.cuda as cuda
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold

from pde_diff.utils import DatasetRegistry, unique_id
from pde_diff.model import DiffusionModel
from pde_diff.callbacks import DarcyLogger, SaveBestModel
import pde_diff.callbacks
import pde_diff.data

@hydra.main(version_base=None, config_name="config.yaml", config_path="../../configs")
def train(cfg: DictConfig):
    hp_config =cfg.experiment.hyperparameters
    cfg.model.id = unique_id(length=5) if cfg.model.id == None else cfg.model.id
    model = DiffusionModel(cfg)

    dataset = DatasetRegistry.create(cfg.dataset)

    if cfg.get("k_folds", None):
        # Create a split and select appropriate subset of data for this fold:
        kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
        train_idx, val_idx = list(kf.split(dataset_train))[cfg.idx_fold-1]
        print(f"Training fold {cfg.idx_fold + 1}/{cfg.k_folds}")
        dataset_val = Subset(dataset_train, val_idx)
        dataset_train = Subset(dataset_train, train_idx)
    else:
        # Split dataset into train and validation sets
        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        if cfg.dataset.time_series:
            dataset_train, dataset_val = dataset[:train_size], dataset[train_size:]
        else:
            dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset_train, batch_size=hp_config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset_val, batch_size=hp_config.batch_size, shuffle=False, num_workers=4)

    wandb_name = f"{cfg.experiment.name}-{cfg.model.id}"
    wandb_name += f"-{cfg.idx_fold}-of-{cfg.k_folds}-folds" if cfg.get("k_folds", None) else ""

    acc = "gpu" if cuda.is_available() else "cpu"

    if cfg.wandb:
        logger = pl.pytorch.loggers.WandbLogger(name=wandb_name, entity="franka-ppo", project="pde-diff", config=OmegaConf.to_container(cfg.experiment, resolve=True))
    else:
        os.makedirs("logs", exist_ok=True)
        logger = pl.pytorch.loggers.CSVLogger("logs", name=wandb_name)

    darcy_logger = DarcyLogger()
    save_best_model = SaveBestModel()

    trainer = pl.Trainer(
        accelerator=acc,
        max_epochs=hp_config.max_epochs,
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=hp_config.log_every_n_steps,
        callbacks=[save_best_model, darcy_logger],
    )

    print(f"Starting training of model {cfg.model.id} for {hp_config.max_epochs} epochs")
    trainer.fit(model, train_dataloader, val_dataloader)
    print(f"Training completed of model {cfg.model.id}")

if __name__ == "__main__":
    train()
