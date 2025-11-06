import torch
import numpy as np
import lightning as pl
from diffusers import UNet2DModel, UNet2DConditionModel
from omegaconf import OmegaConf
from pathlib import Path

from pde_diff.data.datasets import increment_clock_features

# Imports for registries
from pde_diff.utils import SchedulerRegistry, LossRegistry, ModelRegistry
import pde_diff.scheduler
import pde_diff.loss
from pde_diff.data import datasets


class DiffusionModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # combine model and dataset config
        model_cfg = OmegaConf.merge(cfg.model, OmegaConf.create({"dims": cfg.dataset.dims}))
        self.model = ModelRegistry.create(model_cfg)
        self.scheduler = SchedulerRegistry.create(cfg.scheduler)
        self.loss_fn = LossRegistry.create(cfg.loss)
        self.loss_name = cfg.loss
        self.hp_config = cfg.experiment.hyperparameters
        self.data_dims = cfg.dataset.dims

        self.conditional = cfg.dataset.time_series # Add conditional flag
        self.validation_metrics = cfg.dataset.validation_metrics
        self.cfg = cfg
        self.save_model = cfg.model.save_best_model
        self.monitor = cfg.model.monitor
        self.mode = cfg.model.mode
        self.best_score = None
        self.save_dir = Path(cfg.model.save_dir) / Path(cfg.experiment.name + "-" + cfg.model.id) if cfg.model.id else None

    def training_step(self, batch, batch_idx):

        if self.conditional:
            # Apply conditional logic here
            conditionals, state = batch
        else:
            state = batch

        noise = torch.randn_like(state)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (state.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(state, noise, steps)

        if self.conditional:
            noisy_images = torch.cat([conditionals, noisy_images], dim=1)

        residual = self.model(noisy_images, steps)
        with torch.no_grad():
            if self.conditional:
                x_t = conditionals[:,conditionals.shape[1]//2:,:,:][:,:(noisy_images.shape[1]-conditionals.shape[1]),:,:]
            else:
                x_t = noisy_images
            x0_hat = self.scheduler.reconstruct_x0(x_t, residual, steps) if self.loss_name != "mse" else None
        loss = self.loss_fn(residual, noise, x0_hat, self.scheduler.Sigmas[steps])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.conditional:
            conditionals, target = batch
        else:
            target = batch

        noise = torch.randn_like(target)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (target.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(target, noise, steps)

        if self.conditional:
            noisy_images = torch.cat([conditionals, noisy_images], dim=1)

        residual = self.model(noisy_images, steps)
        with torch.no_grad():
            if self.conditional:
                x_t = conditionals[:,conditionals.shape[1]//2:,:,:][:,:(noisy_images.shape[1]-conditionals.shape[1]),:,:]
            else:
                x_t = noisy_images
            x0_hat = self.scheduler.reconstruct_x0(x_t, residual, steps) if self.loss_name != "mse" else None
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
            t_batch =torch.tensor([t]*samples.size(0), device=self.device)
            samples = self.scheduler.sample(self.model(samples, t_batch), t_batch, samples)
            z = torch.randn_like(samples) if t > 0 else 0
            samples += self.scheduler.sigmas[t] * z
        return samples

    def sample_loop(self, batch_size=1, conditionals=None):
        samples = torch.randn((batch_size,int(self.data_dims.output_dims), int(self.data_dims.x), int(self.data_dims.y)), device=self.device)
        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            if self.conditional and conditionals is not None:
                samples = torch.cat([conditionals, samples], dim=1)
            samples = self.forward(samples, t)
        return samples

    def forecast(self, initial_condition, steps):
        self.model.eval()
        current_state = initial_condition.to(self.device)
        forecasted_states = [current_state.cpu()]

        for step in range(steps):
            with torch.no_grad():
                prediction = self.sample_loop(batch_size=current_state.size(0), conditionals=current_state)
                next_state = current_state[:,:,:,-(self.data_dims.input_dims-self.data_dims.output_dims)//2:]
                next_state[:, :, :, :self.data_dims.output_dims] += prediction
                # Update next_state with time information:
                next_state[:, :, :, -4:] = increment_clock_features(
                    next_state[:, :, :, -4:], step_size=self.cfg.dataset.time_step
                ).to(self.device)
                current_state = torch.cat([current_state[:,-(self.data_dims.input_dims-self.data_dims.output_dims)//2:,:,:], next_state], dim=1)
            forecasted_states.append(current_state.cpu())

        return torch.stack(forecasted_states, dim=1)

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
        in_channels = int(cfg.dims.input_dims)
        out_channels = int(cfg.dims.output_dims)
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

@ModelRegistry.register("unet2d_conditional")
class UNet2DConditionalWrapper(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = int(cfg.dims.input_dims)
        self.out_channels = int(cfg.dims.output_dims)
        self.unet = UNet2DConditionModel(
            sample_size=(int(cfg.dims.x), int(cfg.dims.y)),
            in_channels=self.out_channels,
            out_channels=self.out_channels,
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
            cross_attention_dim=self.in_channels - self.out_channels,  # Assuming half of input channels are for conditioning
        )

    def forward(self, x, t):
        # Split x into conditionals and actual input
        cond_channels = self.in_channels - self.out_channels
        conditionals = x[:, :cond_channels, :, :]
        batch_size, cond_channels, height, width = conditionals.shape
        conditionals = conditionals.view(batch_size, cond_channels, height * width).permute(0, 2, 1)
        inputs = x[:, cond_channels:, :, :]
        return self.unet(sample=inputs, timestep=t, encoder_hidden_states=conditionals).sample

if __name__ == "__main__":
    pass
