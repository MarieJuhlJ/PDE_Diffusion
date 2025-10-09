import os
import hydra
import torch

import lightning as pl
import torch.cuda as cuda
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


from pde_diff.utils import DatasetRegistry
from pde_diff.model import DiffusionModel
import pde_diff.data

@hydra.main(version_base=None, config_name="config.yaml", config_path="../../configs")
def train(cfg: DictConfig):
    hp_config =cfg.experiment.hyperparameters
    model = DiffusionModel(cfg)

    #Prepare datasets
    dataset_cfg = cfg.dataset.copy()  # Make a copy
    dataset_train = DatasetRegistry.create(cfg.dataset)
    dataset_cfg.update({"train": False})  # Modify the copy for validation
    dataset_val = DatasetRegistry.create(cfg.dataset)

    if cfg.get("k_folds", None):
        # Create a split and select appropriate subset of data for this fold:
        kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
        train_idx, val_idx = list(kf.split(dataset_train))[cfg.idx_fold-1]
        print(f"Training fold {cfg.idx_fold + 1}/{cfg.k_folds}")
        dataset_val = Subset(dataset_train, val_idx)
        dataset_train = Subset(dataset_train, train_idx)

    train_dataloader = DataLoader(dataset_train, batch_size=hp_config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset_val, batch_size=hp_config.batch_size, shuffle=False, num_workers=4)

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")

    wandb_name = f"some_name-{cfg.experiment.name}"
    wandb_name += f"-{cfg.idx_fold}-of-{cfg.k_folds}-folds" if cfg.get("k_folds", None) else ""

    acc = "gpu" if cuda.is_available() else "cpu"

    if cfg.wandb:
        logger = pl.pytorch.loggers.WandbLogger(name=wandb_name, entity="franka-ppo", project="pde-diff", config=OmegaConf.to_container(cfg.experiment, resolve=True))
    else:
        os.makedirs("logs", exist_ok=True)
        logger = pl.pytorch.loggers.CSVLogger("logs", name=wandb_name)

    trainer = pl.Trainer(
        accelerator=acc,
        max_epochs=hp_config.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=hp_config.log_every_n_steps)

    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the final model
    torch.save(model.state_dict(), f"./models/{wandb_name}-final.ckpt")
    print(f"Model saved to ./models/{wandb_name}-final.ckpt")

if __name__ == "__main__":
    train()
