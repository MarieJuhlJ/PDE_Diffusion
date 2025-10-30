from pde_diff.data import datasets
import torch
import lightning as pl
from diffusers import UNet2DModel
from omegaconf import OmegaConf
from pde_diff.utils import SchedulerRegistry, LossRegistry, ModelRegistry
from pathlib import Path
import pde_diff.scheduler
import pde_diff.loss

class DiffusionModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = ModelRegistry.create(cfg.model)
        self.scheduler = SchedulerRegistry.create(cfg.scheduler)
        self.loss_fn = LossRegistry.create(cfg.loss)
        self.loss_name = cfg.loss
        self.hp_config = cfg.experiment.hyperparameters
        self.data_dims = cfg.dataset.dims

        self.validation_metrics = cfg.dataset.validation_metrics
        self.cfg = cfg
        self.save_model = cfg.model.save_best_model
        self.monitor = cfg.model.monitor
        self.mode = cfg.model.mode
        self.best_score = None
        self.save_dir = Path(cfg.model.save_dir) / Path(cfg.experiment.name + "-" + cfg.model.id) if cfg.model.id else None

    def training_step(self, batch, batch_idx):
        sample = batch if isinstance(batch, torch.Tensor) else batch["data"]
        noise = torch.randn_like(sample)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (sample.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(sample, noise, steps)
        residual = self.model(noisy_images, steps)
        with torch.no_grad():
            x0_hat = self.scheduler.reconstruct_x0(noisy_images, residual, steps) if self.loss_name != "mse" else None
        loss = self.loss_fn(residual, noise, x0_hat, self.scheduler.Sigmas[steps])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sample = batch if isinstance(batch, torch.Tensor) else batch["data"]
        noise = torch.randn_like(sample)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (sample.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(sample, noise, steps)
        residual = self.model(noisy_images, steps)
        with torch.no_grad():
            x0_hat = self.scheduler.reconstruct_x0(noisy_images, residual, steps) if self.loss_name != "mse" else None
        mse = torch.nn.functional.mse_loss(residual, noise)
        loss = self.loss_fn(residual, noise, x0_hat, self.scheduler.Sigmas[steps])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", mse, prog_bar=True)
        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hp_config.lr, weight_decay=self.hp_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def forward(self, samples, t):
        self.model.eval()
        with torch.no_grad():
            z = torch.randn_like(samples) if t > 0 else 0
            t_batch =torch.tensor([t]*samples.size(0), device=self.device)
            samples = self.scheduler.sample(self.model(samples, t_batch), t_batch, samples) + self.scheduler.sigmas[t] * z
        return samples

    def sample_loop(self, batch_size=1):
        samples = torch.randn((batch_size,int(self.data_dims.z), int(self.data_dims.x), int(self.data_dims.y)), device=self.device)
        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            samples = self.forward(samples, t)
        return samples

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def _save_cfg(self, save_dir):
        """Save self.cfg next to checkpoints as YAML (if OmegaConf) or JSON."""
        import json
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            if hasattr(self, "cfg") and (
                getattr(self.cfg, "__class__", None).__name__ in {"DictConfig", "ListConfig"}
                or hasattr(self.cfg, "to_container")
            ):
                (save_dir / "config.yaml").write_text(OmegaConf.to_yaml(self.cfg))
                return
        except Exception:
            pass

@ModelRegistry.register("dummy")
class DummyModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.layer2 = torch.nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x, t):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

@ModelRegistry.register("unet2d")
class UNet2DWrapper(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = int(cfg.dims.z)
        out_channels = int(cfg.dims.z)
        self.unet = UNet2DModel(
            sample_size=int(cfg.dims.x),  # only needed for some schedulers
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t):
        return self.unet(sample=x, timestep=t).sample

if __name__ == "__main__":
    pass
