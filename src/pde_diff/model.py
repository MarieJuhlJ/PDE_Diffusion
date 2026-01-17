import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
import lightning as pl
from diffusers import UNet2DModel, UNet2DConditionModel
from omegaconf import OmegaConf
from pathlib import Path

from pde_diff.data.datasets import increment_clock_features

from pde_diff.utils import SchedulerRegistry, LossRegistry, ModelRegistry, init_means_and_stds_era5
import pde_diff.scheduler
import pde_diff.loss
from pde_diff.data import datasets
from pde_diff.unet_model import Unet3D
import matplotlib.pyplot as plt
VAR_NAMES = {
    "u": "u",
    "v": "v",
    "t": "T",
    "z": r"$\Phi$",
    "pv": "q",
}

VAR_UNITS = {
    "u": r"$m \cdot s^{-1}$",
    "v": r"$m \cdot s^{-1}$",
    "t": r"$K$",
    "z": r"$m^2 \cdot s^{-2}$",
    "pv": r"$s^{-1}$",
}

class DiffusionModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.scheduler = SchedulerRegistry.create(cfg.scheduler)
        self.loss_fn = LossRegistry.create(cfg.loss)
        self.hp_config = cfg.experiment.hyperparameters
        self.model = ModelRegistry.create([cfg.model, self.hp_config])
        if cfg.dataset.name == 'era5' and cfg.loss.name == 'vorticity': #semi cursed (TODO clean up)
            self.atmospheric_features = cfg.dataset.atmospheric_features
            self.single_features = cfg.dataset.single_features
            self.static_features = cfg.dataset.static_features
            self.means, self.stds, self.diff_means, self.diff_stds = init_means_and_stds_era5(self.atmospheric_features,
                                                                                              self.single_features,
                                                                                              self.static_features)
            self.loss_fn.set_mean_and_std(self.means, self.stds, self.diff_means, self.diff_stds)

        self.data_dims = cfg.dataset.dims

        self.conditional = cfg.dataset.time_series # Add conditional flag
        self.cfg = cfg
        self.save_model = cfg.model.save_best_model
        self.mse = torch.nn.MSELoss(reduction='none')
        self.best_score = None
        self.save_dir = Path(cfg.save_dir) / Path(cfg.experiment.name + "-" + cfg.id) if cfg.get("id", None) else None

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
            model_in = torch.cat((conditionals, noisy_images), dim=1)
        else:
            model_in = noisy_images

        model_out = self.model(model_in, steps)

        if self.conditional:
            x0_hat = torch.cat([conditionals[:, 19:34], model_out], dim=1) #hardcoded for now (TODO)
        else:
            x0_hat = model_out

        variance = self.scheduler.posterior_variance[steps]
        self.loss_fn.c_data = self.scheduler.p2_loss_weight[steps] #https://arxiv.org/pdf/2303.09556.pdf
        loss = self.loss_fn(model_out=model_out, target=state, x0_hat=x0_hat, var=variance)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.additional_validation_metrics(model_out, state, x0_hat, steps, validation=False)
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

        model_out = self.model(noisy_images, steps)
        x0_hat = model_out
        if self.conditional:
            x0_hat = torch.cat([conditionals[:, 19:34], model_out], dim=1) #hardcoded for now (TODO)

        variance = self.scheduler.posterior_variance[steps]
        self.loss_fn.c_data = self.scheduler.p2_loss_weight[steps]
        loss = self.loss_fn(model_out=model_out, target=target, x0_hat=x0_hat, var=variance)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.additional_validation_metrics(model_out, target, x0_hat, steps)

    def test_step(self, batch, batch_idx):
        if self.conditional:
            conditionals, target = batch
        else:
            target = batch

        noise = torch.randn_like(target)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (target.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(target, noise, steps)

        if self.conditional:
            noisy_images = torch.cat([conditionals, noisy_images], dim=1)

        model_out = self.model(noisy_images, steps)
        x0_hat = model_out
        if self.conditional:
            x0_hat = torch.cat([conditionals[:, 19:34], model_out], dim=1) #hardcoded for now (TODO)

        variance = self.scheduler.posterior_variance[steps]
        self.loss_fn.c_data = self.scheduler.p2_loss_weight[steps]
        loss = self.loss_fn(model_out=model_out, target=target, x0_hat=x0_hat, var=variance)
        self.log("test_loss", loss, prog_bar=True, batch_size=model_out.size(0))
        self.additional_validation_metrics(model_out, target,x0_hat, steps)

    def additional_validation_metrics(self, model_out, target, x0_hat, steps, validation=True):
        if validation:
            metrics = self.cfg.dataset.validation_metrics
            for metric_name in metrics:
                if metric_name == "mse":
                    mse = (self.mse(model_out, target) * self.scheduler.p2_loss_weight[steps][:, None, None, None]).mean()
                    self.log("val_mse_(weighted)", mse, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))

                if metric_name == 'era5_vorticity':
                    assert self.cfg.loss.name == 'vorticity', "era5_vorticity metric can only be used with vorticity loss"
                    num_channels = x0_hat.shape[1]
                    x0_previous = x0_hat[:, :num_channels//2, :, :]
                    x0_change_pred = x0_hat[:, num_channels//2:, :, :]
                    residual_planetary = self.loss_fn.compute_residual_planetary_vorticity(x0_previous, x0_change_pred).abs().mean()
                    residual_geo_wind = self.loss_fn.compute_residual_geostrophic_wind(x0_previous, x0_change_pred).abs().mean()
                    residual_qgpv = self.loss_fn.compute_residual_qgpv(x0_previous, x0_change_pred).abs().mean()
                    self.log("val_era5_planetary_residual(norm)", residual_planetary, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))
                    self.log("val_era5_geo_wind_residual(norm)", residual_geo_wind, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))
                    self.log("val_era5_qgpv_residual(norm)", residual_qgpv, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))
        else:
            metrics = self.cfg.dataset.test_metrics
            for metric_name in metrics:
                if metric_name == "mse":
                    mse = (self.mse(model_out, target) * self.scheduler.p2_loss_weight[steps][:, None, None, None]).mean()
                    self.log("train_mse_(weighted)", mse, prog_bar=True, on_step=False, on_epoch=True, batch_size=model_out.size(0))

    def on_validation_epoch_end(self):
        """For on_epoch_end validation metrics"""
        metrics = self.cfg.dataset.validation_metrics
        for metric_name in metrics:
            if metric_name == "darcy":
                with torch.no_grad():
                    x0_preds = self.sample_loop(batch_size=16)
                    darcy_res = self.loss_fn.compute_residual(x0_preds).mean().abs()
                self.log("val_darcy_residual", darcy_res, prog_bar=True, on_epoch=True, sync_dist=True)
            if metric_name == 'era5_vorticity':
                with torch.no_grad():
                    # Get a uniform validation batch (conditionals and targets)
                    batch = self._uniform_val_batch(n=16)
                    if self.conditional:
                        val_conditionals = batch[0].to(self.device)
                        val_targets = batch[1].to(self.device)
                    else:
                        # If not conditional, skip these diagnostics
                        continue

                    # 1) single-step x0_hat similar to validation_step (use fixed seed for reproducibility)
                    torch.manual_seed(0)
                    noise = torch.randn_like(val_targets)
                    steps = torch.randint(self.scheduler.config.num_train_timesteps, (val_targets.size(0),), device=self.device)
                    noisy_images = self.scheduler.add_noise(val_targets, noise, steps)
                    noisy_inputs = torch.cat([val_conditionals, noisy_images], dim=1)
                    model_out = self.model(noisy_inputs, steps)
                    x0_hat = torch.cat([val_conditionals[:, 19:34], model_out], dim=1)

                    # 2) sample_loop predictions (stochastic)
                    x0_preds_stoch = self.sample_loop(batch_size=16, conditionals=val_conditionals, deterministic=False)

                    # 3) sample_loop deterministic predictions (no sampling noise)
                    x0_preds_det = self.sample_loop(batch_size=16, conditionals=val_conditionals, deterministic=True)

                    # 4) completely random samples (match per-channel mean/std of targets)
                    # create random samples with the same per-channel mean/std as the validation targets
                    chan_mean = val_targets.mean(dim=(0, 2, 3), keepdim=True)  # shape [1, C, 1, 1]
                    chan_std = val_targets.std(dim=(0, 2, 3), unbiased=False, keepdim=True)
                    x0_random = torch.randn_like(val_targets) * chan_std + chan_mean

                    # x0_prev (previous state slice) used for residual computations
                    x0_prev = val_conditionals[:, 19:34]

                    # compute residuals (normalized and raw) for targets, x0_hat,x0_preds_stoch, random
                    names = ["targets", "x0_hat", "x0_preds_stoch", "random"]
                    x0_changes = [val_targets, model_out,  x0_preds_stoch, x0_random]

                    res_types = [
                        ("planetary_vorticity", self.loss_fn.compute_residual_planetary_vorticity),
                        ("geostrophic_wind", self.loss_fn.compute_residual_geostrophic_wind),
                        ("qgpv", self.loss_fn.compute_residual_qgpv),
                    ]

                    results = {}
                    for n, xchg in zip(names, x0_changes):
                        results[n] = {}
                        for rname, fn in res_types:
                            # normalized (default) and raw (normalize=False)
                            try:
                                norm_val = fn(x0_prev, xchg).abs().mean().detach().cpu().item()
                            except Exception:
                                norm_val = float("nan")
                            try:
                                raw_val = fn(x0_prev, xchg, normalize=False).abs().mean().detach().cpu().item()
                            except Exception:
                                raw_val = float("nan")
                            results[n][rname] = {"norm": norm_val, "raw": raw_val}

                    # print per-channel means and biases (mean difference between preds and targets)
                    channel_means = {}
                    for n, xchg in zip(names, x0_changes):
                        # For per-channel means we need full x0 predicted states; if xchg is change, reconstruct predicted x0
                        # Here we compute means of the change tensor across batch and spatial dims
                        means = xchg.mean(dim=(0,2,3)).detach().cpu().numpy()
                        channel_means[n] = means

                    print("==== Residual diagnostics (validation epoch end) ====")
                    for n in names:
                        print(f"--- {n} per-channel means ---")
                        print(channel_means[n])
                        bias = channel_means[n] - channel_means["targets"]
                        print(f"bias vs targets: {bias}")

                    print("Residual summary (normalized | raw):")
                    for rname, _ in res_types:
                        line = [f"{rname}: "]
                        for n in names:
                            vals = results[n][rname]
                            line.append(f"{n} norm={vals['norm']:.4e}, raw={vals['raw']:.4e}")
                        print("  ".join(line))

                    # Produce histogram plots similar to plot_data_distribution.py
                    out_dir = Path(self.save_dir or "logs") / "debug_figures"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Colors for the five series: targets, x0_hat, stochastic, random
                    colors = ["#2A9D8F", "#E76F51", "#1A4DAC", "#970383"]

                    for rname, fn in res_types:
                        # collect flattened numpy arrays for normalized and raw residuals
                        arrays_norm = {}
                        arrays_raw = {}
                        for n, xchg in zip(names, x0_changes):
                            try:
                                t_norm = fn(x0_prev, xchg).detach().cpu().flatten().numpy()
                            except Exception:
                                t_norm = np.array([])
                            try:
                                t_raw = fn(x0_prev, xchg, normalize=False).detach().cpu().flatten().numpy()
                            except Exception:
                                t_raw = np.array([])
                            arrays_norm[n] = t_norm
                            arrays_raw[n] = t_raw

                        # Plot normalized residual histograms
                        all_norm = np.concatenate([v for v in arrays_norm.values() if v.size > 0]) if any(v.size > 0 for v in arrays_norm.values()) else np.array([])
                        if all_norm.size > 0:
                            lo, hi = np.percentile(all_norm, [0.5, 99.5])
                        else:
                            lo, hi = (0.0, 1.0)

                        fig, ax = plt.subplots(figsize=(6, 3))
                        for k, n in enumerate(names):
                            data = arrays_norm[n]
                            if data.size == 0:
                                continue
                            mean = data.mean()
                            std = data.std()
                            ax.hist(
                                data,
                                bins=120,
                                range=(lo, hi),
                                histtype="stepfilled",
                                alpha=0.6,
                                color=colors[k],
                                edgecolor="black",
                                linewidth=0.8,
                                label=f"{n}: mean={mean:.3e}, std={std:.3e}",
                            )
                            ax.axvline(mean, color=colors[k], linestyle="-", linewidth=2)
                            ax.axvline(mean - std, color=colors[k], linestyle="--", linewidth=1.5)
                            ax.axvline(mean + std, color=colors[k], linestyle="--", linewidth=1.5)

                        ax.set_title(f"{rname} residuals (normalized)")
                        ax.set_xlabel("Residual value (normalized)")
                        ax.set_xlim(lo, hi)
                        ax.grid(True, linestyle=":", alpha=0.6)
                        ax.legend()
                        fig.tight_layout()
                        fnorm = out_dir / f"hist_{rname}_normalized_epoch_{self.current_epoch if hasattr(self, 'current_epoch') else 'NA'}.png"
                        fig.savefig(fnorm, dpi=150)
                        plt.close(fig)
                        print(f"Saved normalized histogram for {rname} to {fnorm}")

                        # Plot raw residual histograms
                        all_raw = np.concatenate([v for v in arrays_raw.values() if v.size > 0]) if any(v.size > 0 for v in arrays_raw.values()) else np.array([])
                        if all_raw.size > 0:
                            lo, hi = np.percentile(all_raw, [0.5, 99.5])
                        else:
                            lo, hi = (0.0, 1.0)

                        fig, ax = plt.subplots(figsize=(6, 3))
                        text_lines = []
                        for k, n in enumerate(names):
                            data = arrays_raw[n]
                            if data.size == 0:
                                continue
                            mean = data.mean()
                            std = data.std()
                            text_lines.append(f"{n}: mean={mean:.2e}, std={std:.2e}")
                            ax.hist(
                                data,
                                bins=120,
                                range=(lo, hi),
                                histtype="stepfilled",
                                alpha=0.6,
                                color=colors[k],
                                edgecolor="black",
                                linewidth=0.8,
                                label=n,
                            )
                            ax.axvline(mean, color=colors[k], linestyle="-", linewidth=2)
                            ax.axvline(mean - std, color=colors[k], linestyle="--", linewidth=1.5)
                            ax.axvline(mean + std, color=colors[k], linestyle="--", linewidth=1.5)

                        ax.set_title(f"{rname} residuals (raw)")
                        ax.set_xlabel("Residual value (raw)")
                        ax.set_xlim(lo, hi)
                        ax.grid(True, linestyle=":", alpha=0.6)
                        ax.legend()
                        ax.text(
                            0.97,
                            0.97,
                            "\n".join(text_lines),
                            transform=ax.transAxes,
                            fontsize=9,
                            verticalalignment="top",
                            horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                        )
                        fig.tight_layout()
                        fraw = out_dir / f"hist_{rname}_raw_epoch_{self.current_epoch if hasattr(self, 'current_epoch') else 'NA'}.png"
                        fig.savefig(fraw, dpi=150)
                        plt.close(fig)
                        print(f"Saved raw histogram for {rname} to {fraw}")

                    # --- Plot variable distributions at 500hPa (level index 1 for each variable) ---
                    # channels: u,v,pv,t,z correspond to channel indices [1,4,7,10,13]
                    var_list = ["u", "v", "pv", "t", "z"]
                    channel_indices = [1, 4, 7, 10, 13]
                    for j, var in enumerate(var_list):
                        idx = channel_indices[j]
                        arrays_var = {}
                        for n, xchg in zip(names, x0_changes):
                            try:
                                arr = xchg.detach().cpu().numpy()
                                # if unnormalization info available, unnormalize using diff_means/diff_stds
                                if hasattr(self, "diff_means") and hasattr(self, "diff_stds"):
                                    means = np.asarray(self.diff_means)
                                    stds = np.asarray(self.diff_stds)
                                    # arr shape [B, C, X, Y]
                                    arr_unnorm = arr * stds[None, :, None, None] + means[None, :, None, None]
                                    vals = arr_unnorm[:, idx, :, :].flatten()
                                else:
                                    vals = arr[:, idx, :, :].flatten()
                            except Exception:
                                vals = np.array([])
                            arrays_var[n] = vals

                        # Compose global x-limits robustly
                        all_vals = np.concatenate([v for v in arrays_var.values() if v.size > 0]) if any(v.size > 0 for v in arrays_var.values()) else np.array([])
                        if all_vals.size > 0:
                            lo, hi = np.percentile(all_vals, [0.5, 99.5])
                        else:
                            lo, hi = (0.0, 1.0)

                        fig, ax = plt.subplots(figsize=(4.3, 4.3))
                        for k, n in enumerate(names):
                            data = arrays_var[n]
                            if data.size == 0:
                                continue
                            mean = data.mean()
                            std = data.std()
                            ax.hist(
                                data,
                                bins=120,
                                range=(lo, hi),
                                histtype="stepfilled",
                                alpha=0.6,
                                color=colors[k],
                                edgecolor="black",
                                linewidth=0.8,
                                label=f"{n}: mean={mean:.2e}, std={std:.2e}",
                            )
                            ax.axvline(mean, color=colors[k], linestyle="-", linewidth=2)
                            ax.axvline(mean - std, color=colors[k], linestyle="--", linewidth=1.5)
                            ax.axvline(mean + std, color=colors[k], linestyle="--", linewidth=1.5)

                        ax.set_title(f"Distribution of {VAR_NAMES.get(var, var)} at 500hPa")
                        ax.set_xlabel(VAR_UNITS.get(var, ""))
                        ax.set_xlim(lo, hi)
                        ax.grid(True, linestyle=":", alpha=0.6)
                        ax.legend()
                        fig.tight_layout()
                        pvfn = out_dir / f"var_{var}_500hpa_epoch_{self.current_epoch if hasattr(self, 'current_epoch') else 'NA'}.png"
                        fig.savefig(pvfn, dpi=150)
                        plt.close(fig)
                        print(f"Saved variable distribution for {var} to {pvfn}")
                self.log("val_era5_sampled_planetary_residual(norm)", results["x0_preds_stoch"]["planetary_vorticity"]["norm"], prog_bar=True, on_epoch=True, sync_dist=True)
                self.log("val_era5_sampled_geo_wind_residual(norm)", results["x0_preds_stoch"]["geostrophic_wind"]["norm"], prog_bar=True, on_epoch=True, sync_dist=True)
                self.log("val_era5_sampled_qgpv_residual(norm)", results["x0_preds_stoch"]["qgpv"]["norm"], prog_bar=True, on_epoch=True, sync_dist=True)

    def _uniform_val_batch(self, n=16):
        vdl = self.trainer.val_dataloaders
        vdl = vdl[0] if isinstance(vdl, (list, tuple)) else vdl

        ds = vdl.dataset
        N = len(ds)

        # 16 evenly spaced indices in [0, N-1]
        idx = torch.linspace(0, N - 1, steps=n).long().tolist()

        # fetch items and collate into a batch like the dataloader would
        items = [ds[i] for i in idx]
        batch = default_collate(items)

        return batch

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hp_config.lr, weight_decay=self.hp_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def forward(self, samples, t):
        """One reverse diffusion step when the model predicts x0 instead of ε."""
        self.model.eval()
        with torch.no_grad():
            t_batch = torch.full((samples.size(0),), t, device=self.device, dtype=torch.long)
            x0_pred = self.model(samples, t_batch)
            if self.conditional:
                samples = samples[:,-self.data_dims.output_dims:,:,:]
            mean = self.scheduler.posterior_mean_coef1[t_batch][:, None, None, None] * x0_pred
            mean.add_(self.scheduler.posterior_mean_coef2[t_batch][:, None, None, None] * samples)
            # deterministic option: if caller sets samples.dtype==torch.bool in a special field it is awkward;
            # instead callers will pass `deterministic` via keyword by calling this method explicitly.
            # To keep backward compatibility we accept an optional attribute on samples named `_deterministic`.
            deterministic = getattr(samples, "_deterministic", False)
            z = torch.zeros_like(samples) if (t > 0 and deterministic) else (torch.randn_like(samples) if t > 0 else 0)
            samples = mean + self.scheduler.sigmas[t] * z
        return samples

    def sample_loop(self, batch_size=1, conditionals=None, deterministic=False):
        """Run the full reverse diffusion sampling loop.

        If `deterministic=True` we zero out sampling noise (z=0) at each step to get a deterministic trajectory.
        """
        samples = torch.randn((batch_size,int(self.data_dims.output_dims), int(self.data_dims.x), int(self.data_dims.y)), device=self.device)
        # attach deterministic flag to samples so forward() can read it
        if deterministic:
            # monkey-patch attribute on tensor object (works in CPython)
            setattr(samples, "_deterministic", True)

        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            if self.conditional:
                model_in = torch.cat((conditionals, samples), dim=1)
            else:
                model_in = samples

            samples = self.forward(model_in, t)
            if deterministic:
                # ensure flag persists after forward (new tensors may have been created)
                setattr(samples, "_deterministic", True)

        return samples

    def forecast(self, initial_condition, steps):
        """Forecast multiple steps ahead given an initial condition tensor.
        Returns the predicted changes at each forecast step."""
        self.model.eval()
        current_state = initial_condition.to(self.device)
        forecasted_changes = []
        for step in range(steps):
            with torch.no_grad():
                prediction = self.sample_loop(batch_size=current_state.size(0), conditionals=current_state)
                next_state = current_state[:,-(self.data_dims.input_dims-self.data_dims.output_dims)//2:,:,:].clone().to(self.device)
                next_state[:,:self.data_dims.output_dims, :, :] = self.loss_fn.get_original_states(x0_previous=next_state[:,:self.data_dims.output_dims], x0_change_pred=prediction, rearrange=False)[1]
                next_state[:,:self.data_dims.output_dims] = self.loss_fn.get_normalized_states(x0=next_state[:,:self.data_dims.output_dims, :, :] )
                # Update next_state with time information:
                next_state[:, -4:, :, :] = increment_clock_features(
                    next_state[:, -4:, :, :], step_size=self.cfg.dataset.time_step
                ).to(self.device)
                current_state = torch.cat([current_state[:,-(self.data_dims.input_dims-self.data_dims.output_dims)//2:,:,:], next_state], dim=1)
            forecasted_changes.append(prediction)

        return torch.stack(forecasted_changes, dim=1)

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

@ModelRegistry.register("unet3d")
class UNet3DWrapper(torch.nn.Module):
    def __init__(self, cfg_list):
        super().__init__()
        model_hp, hp_params = cfg_list[0], cfg_list[1]
        self.unet = Unet3D(dim = 32, channels = model_hp.dims.output_dims)

    def forward(self, x, t):
        return self.unet(x, t)

@ModelRegistry.register("unet3d_conditional")
class UNet3DWrapperConditional(torch.nn.Module):
    def __init__(self, cfg_list):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_hp, hp_params = cfg_list[0], cfg_list[1]
        self.in_channels = int(model_hp.dims.input_dims)
        self.out_channels = int(model_hp.dims.output_dims)
        self.unet = Unet3D(dim = 32, channels=self.out_channels, out_dim=self.out_channels, cond_dim=self.in_channels - self.out_channels).to(device)

    def forward(self, x, t):
        cond_channels = self.in_channels - self.out_channels
        conditionals = x[:, :cond_channels, :, :]
        inputs = x[:, cond_channels:, :, :]
        return self.unet(x = inputs, time = t, cond = conditionals)


@ModelRegistry.register("unet2d")
class UNet2DWrapper(torch.nn.Module):
    def __init__(self, cfg_list):
        super().__init__()
        model_hp, hp_params = cfg_list[0], cfg_list[1]
        in_channels = int(model_hp.dims.input_dims)
        out_channels = int(model_hp.dims.output_dims)
        self.unet = UNet2DModel(
            sample_size=int(model_hp.dims.x),
            in_channels=in_channels,
            out_channels=out_channels,
            dropout = hp_params.dropout,
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
    def __init__(self, cfg_list):
        super().__init__()
        model_hp, hp_params = cfg_list[0], cfg_list[1]
        self.in_channels = int(model_hp.dims.input_dims)
        self.out_channels = int(model_hp.dims.output_dims)
        self.unet = UNet2DConditionModel(
            sample_size=(int(model_hp.dims.x), int(model_hp.dims.y)),
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            dropout = hp_params.dropout,
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
