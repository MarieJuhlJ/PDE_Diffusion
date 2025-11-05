from pde_diff import data
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
        self.mse = torch.nn.MSELoss(reduction='none')
        self.best_score = None
        self.save_dir = Path(cfg.model.save_dir) / Path(cfg.experiment.name + "-" + cfg.model.id) if cfg.model.id else None

    def training_step(self, batch, batch_idx):
        sample = batch if isinstance(batch, torch.Tensor) else batch["data"]
        noise = torch.randn_like(sample)
        T = self.scheduler.config.num_train_timesteps
        steps = torch.randint(0, T, (sample.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(sample, noise, steps)
        model_out = self.model(noisy_images, steps)
        variance = self.scheduler.posterior_variance[steps]
        self.loss_fn.c_data = self.scheduler.p2_loss_weight[steps] #https://arxiv.org/pdf/2303.09556.pdf
        loss = self.loss_fn(model_out=model_out, target=sample, x0_hat=sample, var=variance)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sample = batch if isinstance(batch, torch.Tensor) else batch["data"]
        noise = torch.randn_like(sample)
        T = self.scheduler.config.num_train_timesteps
        steps = torch.randint(0, T, (sample.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(sample, noise, steps)
        model_out = self.model(noisy_images, steps)
        variance = self.scheduler.posterior_variance[steps]
        self.loss_fn.c_data = self.scheduler.p2_loss_weight[steps]
        loss = self.loss_fn(model_out=model_out, target=sample, x0_hat=sample, var=variance)
        mse = (self.mse(model_out, sample) * self.scheduler.p2_loss_weight[steps][:, None, None, None]).mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=sample.size(0))
        self.log("val_mse_(weighted)",  mse,  prog_bar=True, on_step=False, on_epoch=True, batch_size=sample.size(0))
        return mse

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hp_config.lr, weight_decay=self.hp_config.weight_decay)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hp_config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def forward(self, samples, t):
        """One reverse diffusion step when the model predicts x0 instead of ε."""

        self.model.eval()
        with torch.no_grad():
            t_batch = torch.full((samples.size(0),), t, device=self.device, dtype=torch.long)
            x0_pred = self.model(samples, t_batch)
            mean = self.scheduler.posterior_mean_coef1[t_batch][:, None, None, None] * x0_pred + self.scheduler.posterior_mean_coef2[t_batch][:, None, None, None] * samples
            z = torch.randn_like(samples) if t > 0 else 0
            samples = mean + self.scheduler.sigmas[t] * z
        return samples

    def sample_loop(self, batch_size=1):
        samples = torch.randn((batch_size,int(self.data_dims.z), int(self.data_dims.x), int(self.data_dims.y)), device=self.device)
        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            samples = self.forward(samples, t)
        return samples

    def ddim_forward(self, samples, t, t_prev): #this assumes model predicts epsilon
        z = torch.randn_like(samples) if t_prev > 0 else 0
        t_batch =torch.tensor([t]*samples.size(0), device=self.device)
        t_prev_batch =torch.tensor([t_prev]*samples.size(0), device=self.device)
        samples = self.scheduler.ddim_sample(self.model(samples, t_batch), t_batch, t_prev_batch, z, samples) + self.scheduler.sigmas[t] * z
        return samples

    def sample_loop_ddim(self, input=None, residual=None, current_time=None, tau_length=100, microbatch=16, use_amp=False):
        device = self.device
        if microbatch is None:
            microbatch = input.size(0)

        ct = current_time.to(torch.long)
        tau = int(tau_length)

        def make_sched(T: int, tau: int):
            s = list(range(0, max(T, 0), tau))
            if not s or s[-1] != T:
                s.append(T)
            return s

        schedules = [make_sched(int(T.item()), tau) for T in ct]
        groups = {}
        for i, s in enumerate(schedules):
            key = tuple(s)
            groups.setdefault(key, []).append(i)

        x = input
        amp_cm = torch.autocast(device_type=device.type if hasattr(device, "type") else "cuda", dtype=torch.float16) if use_amp else torch.cuda.amp.autocast(enabled=False)

        with amp_cm:
            for sched, idxs in groups.items():
                idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
                timesteps = list(sched)

                for t_prev, t in zip(reversed(timesteps[1:]), reversed(timesteps[:-1])):
                    for start in range(0, idx_tensor.numel(), microbatch):
                        sel = idx_tensor[start:start + microbatch]
                        x_sel = torch.index_select(x, 0, sel)
                        x_sel = self.ddim_forward(x_sel, int(t), int(t_prev))
                        x.index_copy_(0, sel, x_sel)
        return x



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
            sample_size=int(cfg.dims.x),
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
