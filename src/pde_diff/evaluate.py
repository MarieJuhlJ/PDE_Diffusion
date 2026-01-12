import hydra
import einops as ein
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pandas as pd

import lightning as pl
from sklearn.model_selection import KFold

from pde_diff.utils import DatasetRegistry, LossRegistry
from pde_diff.model import DiffusionModel
from pde_diff.data.utils import split_dataset
import pde_diff.loss
from pde_diff.visualize import visualize_era5_sample
from pde_diff.visualize_eval import plot_forecast_loss_vs_steps, plot_sample_target_absdiff_stacked, plot_forecasts_vs_targets,plot_residuals_with_truth



@hydra.main(version_base=None, config_name="evaluate.yaml", config_path="../../configs")
def evaluate(cfg: DictConfig):
    # load model config from cfg.model.path
    best_fold= find_best_fold(cfg.model.path)
    print(f"Best fold: {best_fold[0]} with val loss {best_fold[1]}")
    model_cfg = OmegaConf.load(cfg.model.path.split("/")[0] + "/" + best_fold[0] + "/config.yaml")
    cfg = OmegaConf.merge(model_cfg, cfg)

    pl.seed_everything(cfg.seed)
    model = DiffusionModel(cfg)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(cfg.model.path + "/best-val_loss-weights.pt"))

    cfg.dataset.path= "./data/era5/zarr_test/"
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.dataset.forecast_steps = cfg.steps #TODO: this config stuff could be nicer...
    dataset_test = DatasetRegistry.create(cfg.dataset)

    dataloader_test = DataLoader(dataset_test,
                                 batch_size=cfg.experiment.hyperparameters.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 persistent_workers=True)
    dir = "logs"
    name = cfg.experiment.name + "_" +cfg.id+"_eval"
    os.makedirs(dir, exist_ok=True)
    logger = pl.pytorch.loggers.CSVLogger(dir, name=name)

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=logger)
    #results = trainer.test(model, dataloader_test) #TODO: extend to multiple losses
    #print(f"Test results: {results}")

    if cfg.dataset.name in ["era5"]:
        print("Evaluating forecasting performance...")
        # Creating special test dataset that returns multiple forecast steps
        cfg.dataset.name = "era5_test"
        dataset_test = DatasetRegistry.create(cfg.dataset)

        forecasting_losses = evaluate_forecasting(model, dataset_test, steps=cfg.steps, losses=["mse"],dir=os.path.join(dir, name))
        # save losses to csv
        for loss_name, loss_dict in forecasting_losses.items():
            df = pd.DataFrame(loss_dict)
            df.index = [f"Forecast{i+1}" for i in range(len(df))]
            df.index.name = "forecast_step"
            plot_forecast_loss_vs_steps(df, dir=os.path.join(dir, name), loss_name=loss_name)
            path_csv = os.path.join(dir, name, f"forecasting_losses_{loss_name}.csv")
            df.to_csv(path_csv)
            print(f"Forecasting losses saved to CSV {path_csv}.")

        for loss_name, loss_values in forecasting_losses.items():
            for step, values in loss_values.items():
                mean_loss = sum(values) / len(values)
                print(f"Step {step}: {loss_name} = {mean_loss}")

def find_best_fold(model_path, fold_no=5, log_path="logs"):
    if os.path.exists(os.path.join(log_path, f"{model_path.split('/')[-1]}")):
        print("No folds found, using single model.")
        return (model_path.split('/')[-1], None)

    if os.path.exists(os.path.join(log_path, f"{model_path.split('/')[-1]}-1")):
        best_model = (None, float("inf"))
        for fold in range(1, fold_no+1):
            current_model_id = f"{model_path.split("/")[-1]}-{fold}"
            csv_path = Path(log_path) / current_model_id / "version_0" / "metrics.csv"

            df = (
                pd.read_csv(csv_path)
                .apply(pd.to_numeric)
                .dropna(subset=["step"])
                .sort_values("step")
            )

            min_val = min(df["val_mse_(weighted)"].dropna().values)

            if min_val < best_model[1]:
                best_model = (current_model_id, min_val)
    else:
        # in case there are no csvs (only happens if a model was trained elsewhere)
        best_model = (model_path.split('/')[-1], None)

    return best_model


def save_samples(forecasted_states, dir: Path, target_states=None):
    os.makedirs(dir, exist_ok=True)
    for i in range(forecasted_states.shape[1]):
        if i==0:
            for j,var in enumerate(["u","v","pv","t","z"]):
                if target_states is not None:
                    min_val = min(torch.min(target_states[i, j, :, :]),torch.min(forecasted_states[i, j, :, :]))
                    max_val = max(torch.max(target_states[i, j, :, :]),torch.max(forecasted_states[i, j, :, :]))
                visualize_era5_sample(target_states[i, j, :, :], variable=var, sample_idx=f"_target_sample_T{i}", dir=dir, color_bar_limit=(min_val, max_val))
                visualize_era5_sample(forecasted_states[i, j, :, :], variable=var, sample_idx=f"_forecast_sample_T{i}", dir=dir, color_bar_limit=(min_val, max_val) if target_states is not None else None)
                plot_sample_target_absdiff_stacked(forecasted_states[i, j, :, :], target=target_states[i, j, :, :], variable=var, sample_idx=f"_T{i}", dir=dir)
        else:
            if target_states is not None:
                    min_val = min(torch.min(target_states[i, 3, :, :]),torch.min(forecasted_states[i, 3, :, :]))
                    max_val = max(torch.max(target_states[i, 3, :, :]),torch.max(forecasted_states[i, 3, :, :]))
            visualize_era5_sample(forecasted_states[i, 3, :, :], variable="t", sample_idx=f"_forecast_sample_T{i}", dir=dir, color_bar_limit=(min_val, max_val))
            if target_states is not None:
                visualize_era5_sample(target_states[i, 3, :, :], variable="t", sample_idx=f"_target_sample_T{i}", dir=dir, color_bar_limit=(min_val, max_val))

    pickle_path = os.path.join(dir,f"forecast_sample.pt")
    torch.save(forecasted_states, pickle_path)
    print(f"Forecasted samples saved to {pickle_path}.")


def evaluate_forecasting(model: DiffusionModel, dataset_test: Dataset, steps: int=8, losses: list[str]=["mse", "fb"], dir=Path("reports/figures")) -> dict[str, list[float]]:

    dataloader_test = DataLoader(dataset_test,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            persistent_workers=True,
                            )
    vor_loss = model.loss_fn
    vor_loss.set_mean_and_std(dataset_test.means, dataset_test.stds, dataset_test.diff_means, dataset_test.diff_stds)
    loss_fns = {"mse": LossRegistry.create(OmegaConf.create({"name":"mse"})) ,
                "residual": model.loss_fn}

    losses = ["mse", "val_era5_sampled_planetary_residual(norm)", "val_era5_sampled_geo_wind_residual(norm)", "val_era5_sampled_qgpv_residual(norm)"]
    loss = {loss: {str(i+1): [] for i in range(steps)} for loss in losses}

    for i, state in enumerate(tqdm(dataloader_test)):
        conditionals, targets, raw_target_states = state
        conditionals = conditionals.to(model.device)
        targets = targets[0].to(model.device)
        raw_target_states = raw_target_states[0].to(model.device)

        forecasted_changes = model.forecast(conditionals, steps=steps)[0]
        forecasted_states = get_states(conditionals, forecasted_changes)
        target_states = get_states(conditionals,targets)
        prev_states_true = torch.cat([conditionals[:, 19:34],target_states[:-1]], dim=0)
        prev_states = torch.cat([conditionals[:, 19:34],forecasted_states[:-1]], dim=0)
        for loss_name, loss_fn in loss_fns.items():
            if loss_name=="mse":
                for k in range(steps):
                    l = loss_fn(forecasted_changes[k], targets[k])
                    loss[loss_name][str(k+1)].append(l.item())
            else:
                loss_geo_wind = loss_fn.compute_residual_geostrophic_wind(x0_previous=prev_states, x0_change_pred=forecasted_changes, normalize=True).abs()
                mean_loss_geo_wind = loss_geo_wind.mean(dim=(1, 2, 3))
                loss_planetary = loss_fn.compute_residual_planetary_vorticity(x0_previous=prev_states, x0_change_pred=forecasted_changes, normalize=True).abs()
                mean_loss_planetary = loss_planetary.mean(dim=(1, 2, 3))
                loss_qgpv = loss_fn.compute_residual_qgpv(x0_previous=prev_states, x0_change_pred=forecasted_changes, normalize=True).abs()
                mean_loss_qgpv = loss_qgpv.mean(dim=(1, 2))
                for k in range(steps):
                    loss["val_era5_sampled_planetary_residual(norm)"][str(k+1)].append(mean_loss_planetary[k].item())
                    loss["val_era5_sampled_geo_wind_residual(norm)"][str(k+1)].append(mean_loss_geo_wind[k].item())
                    loss["val_era5_sampled_qgpv_residual(norm)"][str(k+1)].append(mean_loss_qgpv[k].item())

        if i<3:
            dir_sample=os.path.join(dir, f"sample_{i}")
            os.makedirs(dir_sample, exist_ok=True)
            lvl = 1

            ### Plot residual errors:
            loss_geo_wind_target = loss_fn.compute_residual_geostrophic_wind(x0_previous=prev_states_true, x0_change_pred=targets, normalize=True).abs()
            loss_planetary_target = loss_fn.compute_residual_planetary_vorticity(x0_previous=prev_states_true, x0_change_pred=targets, normalize=True).abs()
            loss_qgpv_target = loss_fn.compute_residual_qgpv(x0_previous=prev_states_true, x0_change_pred=targets, normalize=True).abs()

            plot_residuals_with_truth(loss_geo_wind[0,lvl],loss_geo_wind_target[0,lvl],"Geostrophic Wind",sample_idx=0, dir=dir_sample)
            plot_residuals_with_truth(loss_planetary[0,lvl],loss_planetary_target[0,lvl],"Planetary Vorticity", sample_idx=0,dir=dir_sample)
            plot_residuals_with_truth(loss_qgpv[0],loss_qgpv_target[0],"QGPV", sample_idx=0, dir=dir_sample)

            """un_norm_cond = dataset_test._unnormalize(conditionals.detach().cpu(), dataset_test.means, dataset_test.stds)
            un_norm_forecasted_changes = dataset_test._unnormalize(forecasted_changes.detach().cpu(), dataset_test.diff_means, dataset_test.diff_stds)
            un_norm_forecasted_states = get_states(un_norm_cond, un_norm_forecasted_changes)
            un_norm_forecasted_states = ein.rearrange(un_norm_forecasted_states, "b (var lev) lon lat -> b lev var lon lat", lev = 3)

            un_norm_targets = dataset_test._unnormalize(targets.detach.cpu(), dataset_test.diff_means, dataset_test.diff_stds)
            un_norm_target_states = get_states(un_norm_cond, un_norm_targets)
            un_norm_target_states = ein.rearrange(un_norm_target_states, "b (var lev) lon lat -> b lev var lon lat", lev = 3)"""


            # un_norm_forecasted_states has shape [steps, levels,vars, lon, lat]
            un_norm_forecasted_states =loss_fns["residual"].get_original_states(x0_previous=prev_states, x0_change_pred=forecasted_changes)[1]
            un_norm_target_states =loss_fns["residual"].get_original_states(x0_previous=prev_states_true, x0_change_pred=targets[0])[1]

            raw_target_states = ein.rearrange(raw_target_states, "b (var lev) lon lat -> b lev var lon lat", lev = 3)

            # Plot all targets and forecasts:
            for j,var in enumerate(["u","v","pv","t","z"]):
                plot_forecasts_vs_targets(
                    forecasts=[un_norm_forecasted_states.detach().cpu()[k,lvl, j,:,:].detach().cpu() for k in range(un_norm_forecasted_states.shape[0])],
                    targets=[raw_target_states.detach().cpu()[k,lvl, j,:,:] for k in range(un_norm_target_states.shape[0])],
                    variable=var,
                    sample_idx=i,
                    dir=dir_sample)

            save_samples(un_norm_forecasted_states[:,lvl].detach().cpu(), dir=dir_sample, target_states=un_norm_target_states[:,lvl].detach().cpu())
    return loss

def get_states(conds, state_changes):
    states = []
    next_state = conds[0,19:34,:,:] + state_changes[0,:,:,:]
    states.append(next_state)
    for step in range(1, state_changes.shape[0]):
        next_state = next_state + state_changes[step,:,:,:]
        states.append(next_state)
    return torch.stack(states, dim=0)

if __name__ == "__main__":
    evaluate()
