from pde_diff import data
import torch
import lightning as pl
from pde_diff.utils import SchedulerRegistry, LossRegistry, ModelRegistry
import pde_diff.scheduler
import pde_diff.loss
from omegaconf import OmegaConf

class DiffusionModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = ModelRegistry.create(cfg.model) # Replace with actual model
        self.scheduler = SchedulerRegistry.create(cfg.scheduler)
        self.loss_fn = LossRegistry.create(cfg.loss)
        self.hp_config = cfg.experiment.hyperparameters
        self.data_dims = cfg.dataset.dims

    def training_step(self, batch, batch_idx):
        sample = batch["data"]
        noise = torch.randn_like(sample)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (sample.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(sample, noise, steps)
        residual = self.model(noisy_images, steps)
        loss = self.loss_fn(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx): # this is crap
        sample = batch["data"]
        noise = torch.randn_like(sample)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (sample.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(sample, noise, steps)
        residual = self.model(noisy_images, steps)
        loss = self.loss_fn(residual, noise)
        self.log("val_loss", loss, prog_bar=True)
        mse = torch.nn.functional.mse_loss(residual, noise)
        self.log("val_mse", mse, prog_bar=True)
        return loss

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

if __name__ == "__main__":
    pass
