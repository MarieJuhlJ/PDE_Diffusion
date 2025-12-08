import os
import hydra
import torch

import lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from pde_diff.utils import DatasetRegistry, LossRegistry, unique_id
from pde_diff.model import DiffusionModel
from pde_diff.callbacks import SaveBestModel
import pde_diff.callbacks
from pde_diff.data.utils import split_dataset

@hydra.main(version_base=None, config_name="config.yaml", config_path="../../configs")
def train(cfg: DictConfig):
    hp_config = cfg.experiment.hyperparameters
    if cfg.get("k_folds") is not None:
        assert cfg.id is not None, "If k_folds is used, an id must be provided."

    cfg.id = unique_id(length=5) if cfg.id == None else cfg.id
    cfg.id += f"-{cfg.idx_fold}" if cfg.get("k_folds", None) else ""
    pl.seed_everything(cfg.seed)
    cfg.model.dims = cfg.dataset.dims

    dataset = DatasetRegistry.create(cfg.dataset)
    model = DiffusionModel(cfg)

    dataset_train, dataset_val = split_dataset(cfg, dataset)

    train_dataloader = DataLoader(dataset_train, batch_size=hp_config.batch_size, shuffle=True, num_workers=4,persistent_workers=True)
    val_dataloader = DataLoader(dataset_val, batch_size=hp_config.batch_size, shuffle=False, num_workers=4,persistent_workers=True)
    
    wandb_name = f"{cfg.experiment.name}-{cfg.id}"

    acc = "gpu" if torch.cuda.is_available() else "cpu"

    if cfg.wandb:
        logger = pl.pytorch.loggers.WandbLogger(name=wandb_name, entity="franka-ppo", project="pde-diff", config=OmegaConf.to_container(cfg.experiment, resolve=True))
    else:
        os.makedirs("logs", exist_ok=True)
        logger = pl.pytorch.loggers.CSVLogger("logs", name=wandb_name)

    trainer = pl.Trainer(
        accelerator=acc,
        max_epochs=hp_config.max_epochs,
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=hp_config.log_every_n_steps,
        callbacks=[SaveBestModel()]
    )

    print(f"Starting training of model {cfg.id}")
    trainer.fit(model, train_dataloader, val_dataloader)
    print(f"Training completed of model {cfg.id}")

if __name__ == "__main__":
    train()
