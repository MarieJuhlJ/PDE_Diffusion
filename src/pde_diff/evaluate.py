import hydra
import einops as ein
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

import lightning as pl
from sklearn.model_selection import KFold

from pde_diff.utils import DatasetRegistry, LossRegistry
from pde_diff.model import DiffusionModel
from pde_diff.data.utils import split_dataset
import pde_diff.loss
from pde_diff.visualize import visualize_era5_sample
from pde_diff.visualize_eval import *

@hydra.main(version_base=None, config_name="evaluate.yaml", config_path="../../configs")
def evaluate(cfg: DictConfig):
    # load model config from cfg.model.path
    best_fold= find_best_fold(cfg.model.path)
    print(f"Best fold: {best_fold[0]} with val loss {best_fold[1]}")
    model_path = cfg.model.path.split("/")[0] + "/" + best_fold[0]
    model_cfg = OmegaConf.load(model_path + "/config.yaml")
    cfg = OmegaConf.merge(model_cfg, cfg)

    pl.seed_everything(cfg.seed)
    model = DiffusionModel(cfg)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path + "/best-val_loss-weights.pt"))

    cfg.dataset.path= "./data/era5/zarr_test_ood/"
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.dataset.forecast_steps = cfg.steps #TODO: this config stuff could be nicer...
    dataset_test = DatasetRegistry.create(cfg.dataset)

    dataloader_test = DataLoader(dataset_test,
                                 batch_size=cfg.experiment.hyperparameters.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 persistent_workers=True)
    dir = "reports/figures/evaluation"
    name = cfg.experiment.name + "-" +cfg.id+"_eval_ood"
    os.makedirs(dir, exist_ok=True)
    logger = pl.pytorch.loggers.CSVLogger(dir, name=name)
    if True:
        trainer = pl.Trainer(accelerator="cuda" if torch.cuda.is_available() else "cpu", logger=logger)
        results = trainer.test(model, dataloader_test) #TODO: extend to multiple losses
        print(f"Test results: {results}")
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Save samples
    if False:
        dir_sample=os.path.join(dir,name,f"sample_{0}_forward")
        os.makedirs(dir_sample, exist_ok=True)
        conditionals, target_changes = next(iter(dataloader_test))
        conditionals = conditionals.to(model.device)
        target_changes = target_changes.to(model.device)
        pred_changes = model.sample_loop(batch_size=1, conditionals=conditionals[None,0])
        targets_changes_rearranged = ein.rearrange(target_changes, "b (lev var) lon lat -> b lev var lon lat", lev = 3)
        pred_changes_rearranged = ein.rearrange(pred_changes, "b (lev var) lon lat -> b lev var lon lat", lev = 3)

        for j,var in enumerate(["u","v","pv","t","z"]):
            plot_sample_target_absdiff_stacked(pred_changes_rearranged[0,1, j, :, :], target=targets_changes_rearranged[0,1, j, :, :], variable=var, sample_idx=f"_T{0}", dir=dir_sample)
        
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

        # Generate single-step forecast error distributions (one-step ahead) and save per-variable PNGs
        if True:
            print("Generating single-step forecast error distributions (step 1)...")
            out_dir_dist = os.path.join(dir, name, "forecast_error_distributions")
            generate_prediction_error_distributions(
                model,
                dataset_test,
                max_hist_samples_per_step=10,
                out_dir=out_dir_dist,
            )
            print(f"Saved forecast error distributions to {out_dir_dist}")
        

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
        print("No logs found, using single model.")
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

    losses = ["mse", "mse_change", "val_era5_sampled_planetary_residual(norm)", "val_era5_sampled_geo_wind_residual(norm)"]
    loss = {loss: {str(i+1): [] for i in range(steps)} for loss in losses}

    var_names = ["u", "v", "pv", "t", "z"]
    mse_accum = None     # shape: (n_vars, lon, lat)
    total_counts = 0

    for i, state in enumerate(tqdm(dataloader_test)):
        conditionals, targets, raw_target_states = state
        conditionals = conditionals.to(model.device)
        targets = targets[0].to(model.device)
        raw_target_states = raw_target_states[0].to(model.device)

        forecasted_changes = model.forecast(conditionals, steps=steps)[0]
        forecasted_states = get_states(conditionals, forecasted_changes)
        target_states = get_states(conditionals,targets)
        num_vars = (conditionals.shape[1] - 8) // 6
        prev_states_true = torch.cat([conditionals[:, num_vars*3+4:-4],target_states[:-1]], dim=0)
        prev_states = torch.cat([conditionals[:, num_vars*3+4:-4],forecasted_states[:-1]], dim=0)
        for loss_name, loss_fn in loss_fns.items():
            if loss_name=="mse":
                for k in range(steps):
                    l = loss_fn(forecasted_states[k], target_states[k])
                    loss[loss_name][str(k+1)].append(l.item())
            elif loss_name=="mse_change":
                for k in range(steps):
                    l = loss_fn(forecasted_changes[k], targets[k])
                    loss[loss_name][str(k+1)].append(l.item())
            else:
                loss_geo_wind = loss_fn.compute_residual_geostrophic_wind(x0_previous=prev_states, x0_change_pred=forecasted_changes, normalize=True).abs()
                mean_loss_geo_wind = loss_geo_wind.mean(dim=(1, 2, 3))
                loss_planetary = loss_fn.compute_residual_planetary_vorticity(x0_previous=prev_states, x0_change_pred=forecasted_changes, normalize=True).abs()
                mean_loss_planetary = loss_planetary.mean(dim=(1, 2))
                for k in range(steps):
                    loss["val_era5_sampled_planetary_residual(norm)"][str(k+1)].append(mean_loss_planetary[k].item())
                    loss["val_era5_sampled_geo_wind_residual(norm)"][str(k+1)].append(mean_loss_geo_wind[k].item())

        targets_rearranged = ein.rearrange(targets, "b (lev var) lon lat -> b lev var lon lat", lev = 3)
        forecasted_changes_rearranged = ein.rearrange(forecasted_changes, "b (lev var) lon lat -> b lev var lon lat", lev = 3)
        sq_err = (targets_rearranged[0,1] - forecasted_changes_rearranged[0,1])**2

        if mse_accum is None:
            mse_accum = sq_err.detach().cpu().numpy()
        else:
            mse_accum += sq_err.detach().cpu().numpy()
        total_counts += 1

        if i<1:
            dir_sample=os.path.join(dir, f"sample_{i}")
            os.makedirs(dir_sample, exist_ok=True)
            lvl = 1

            ### Plot residual errors:
            loss_geo_wind_target = loss_fn.compute_residual_geostrophic_wind(x0_previous=prev_states_true, x0_change_pred=targets, normalize=True).abs()
            loss_planetary_target = loss_fn.compute_residual_planetary_vorticity(x0_previous=prev_states_true, x0_change_pred=targets, normalize=True).abs()

            limits = {
                "plan": (2.966096644740901e-06, 13.07596206665039,2.3655593395233154e-06, 10.719855308532715),
                "gw": (7.870156878198031e-06, 5.983729839324951,1.1920928955078125e-06, 0.7330731153488159),
            }

            plot_residuals_with_truth(loss_geo_wind[0,lvl],loss_geo_wind_target[0,lvl],"gw",sample_idx=0, dir=dir_sample, limits=limits["gw"])
            plot_residuals_with_truth(loss_planetary[0],loss_planetary_target[0],"plan", sample_idx=0,dir=dir_sample, limits=limits["plan"])

            conditionals_rearranged = ein.rearrange(conditionals, "b (state var) lon lat -> b state var lon lat", state = 2)
            un_norm_cond = dataset_test._unnormalize(conditionals_rearranged[:,1,:15].detach().cpu(), dataset_test.means, dataset_test.stds)
            un_norm_forecasted_changes = dataset_test._unnormalize(forecasted_changes.detach().cpu(), dataset_test.diff_means, dataset_test.diff_stds)
            un_norm_forecasted_states = get_states(un_norm_cond, un_norm_forecasted_changes)
            un_norm_forecasted_states = ein.rearrange(un_norm_forecasted_states, "b (var lev) lon lat -> b lev var lon lat", lev = 3)

            un_norm_targets = dataset_test._unnormalize(targets.detach().cpu(), dataset_test.diff_means, dataset_test.diff_stds)
            un_norm_target_states = get_states(un_norm_cond, un_norm_targets)
            un_norm_target_states = ein.rearrange(un_norm_target_states, "b (var lev) lon lat -> b lev var lon lat", lev = 3)

            un_norm_targets = ein.rearrange(un_norm_targets, "b (var lev) lon lat -> b lev var lon lat", lev = 3)
            un_norm_forecasted_changes = ein.rearrange(un_norm_forecasted_changes, "b (var lev) lon lat -> b lev var lon lat", lev = 3)

            # un_norm_forecasted_states has shape [steps, levels,vars, lon, lat]
            #un_norm_forecasted_states =loss_fns["residual"].get_original_states(x0_previous=prev_states, x0_change_pred=forecasted_changes)[1]
            #un_norm_target_states =loss_fns["residual"].get_original_states(x0_previous=prev_states_true, x0_change_pred=targets[0])[1]

            raw_target_states = ein.rearrange(raw_target_states, "b (var lev) lon lat -> b lev var lon lat", lev = 3)
            assert torch.allclose(raw_target_states.detach().cpu(),un_norm_target_states,atol=1e-5,rtol=1e-6), "Problem, norm back and forth changing values too much"

            # Plot limits var : [(states), (changes)]
            limits_combined = {
                "u": [
                    (-25.34462, 57.650707, 6.1392784e-06, 20.092274),
                    (-10.266342, 11.275477, 2.041459e-06, 13.979344),
                ],
                "v": [
                    (-38.033234, 51.942154, 5.9604645e-06, 18.638378),
                    (-20.54956, 12.931, 2.3841858e-06, 14.197182),
                ],
                "pv": [
                    (-1.3699745e-06, 5.4534376e-06, 3.410605e-13, 3.9785564e-06),
                    (-2.7604012e-06, 3.20898e-06, 2.8386182e-12, 2.8250042e-06),
                ],
                "t": [
                    (234.98526, 266.46527, 0.0, 11.396439),
                    (-4.2630463, 3.8316803, 1.5199184e-06, 3.2969234),
                ],
                "z": [
                    (49978.4, 57700.15, 0.0, 725.2578),
                    (-384.3789, 317.4961, 7.772446e-05, 271.37427),
                ],
            }

            # Plot all targets and forecasts:
            for j,var in enumerate(["u","v","pv","t","z"]):
                plot_forecasts_vs_targets(
                    forecasts=[un_norm_forecasted_states.detach().cpu()[k,lvl, j,:,:].detach().cpu() for k in range(un_norm_forecasted_states.shape[0])],
                    targets=[raw_target_states.detach().cpu()[k,lvl, j,:,:] for k in range(un_norm_target_states.shape[0])],
                    variable=var,
                    sample_idx=i,
                    dir=dir_sample,
                    limits=limits_combined[var][0])
                plot_forecasts_vs_targets(
                    forecasts=[un_norm_forecasted_changes.detach().cpu()[k,lvl, j,:,:].detach().cpu() for k in range(un_norm_forecasted_states.shape[0])],
                    targets=[un_norm_targets.detach().cpu()[k,lvl, j,:,:] for k in range(un_norm_target_states.shape[0])],
                    variable=var,
                    sample_idx=i,
                    dir=dir_sample,
                    states=False,
                    limits=limits_combined[var][1])

            #save_samples(un_norm_forecasted_states[:,lvl].detach().cpu(), dir=dir_sample, target_states=un_norm_target_states[:,lvl].detach().cpu())
        #if i==1:
        #    break
        
    if mse_accum is not None and total_counts > 0:
        mse_maps = mse_accum / float(total_counts)  # shape [var, lon, lat]
        out_dir = Path(dir) / "mse_maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        for j, var in enumerate(var_names):
            plot_mse_of_vars(mse_maps[j], out_dir=out_dir, var=var)
        plot_mse_of_all_vars(mse_maps, out_dir=out_dir, vars=var_names)
    
    return loss


def generate_prediction_error_distributions(
    model: DiffusionModel,
    dataset_test: Dataset,
    var_names: list[str] | None = None,
    max_hist_samples_per_step: int = 8,
    out_dir: Path | str = Path("reports/figures/forecast_error_distributions"),
    unnormalize: bool = False,
    dataset_obj=None,
):
    """
    Collect bounded samples of forecast-change and target-change across the test dataset
    and save one PNG per variable showing histograms of (forecast - target) for each step.

    - model: the trained DiffusionModel (must implement .forecast)
    - dataset_test: Dataset instance used for evaluation (will be wrapped in DataLoader)
    - unnormalize: if True, `dataset_obj` must be provided and used to un-normalize values
    """

    if var_names is None:
        var_names = ["u", "v", "pv", "t", "z"]

    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)

    # Initialize collectors
    collectors_f = {v: [[] for _ in range(3)] for v in var_names}
    collectors_t = {v: [[] for _ in range(3)] for v in var_names}

    for i, state in enumerate(tqdm(dataloader_test, desc="Collecting forecast error samples")):
        conditionals, targets, raw_target_states = state
        conditionals = conditionals.to(model.device)
        targets = targets[0].to(model.device)

        pred_changes = model.sample_loop(batch_size=1, conditionals=conditionals[None,0])

        targets_rearranged = ein.rearrange(targets, "b (lev var) lon lat -> b lev var lon lat", lev = 3)
        pred_changes_rearranged = ein.rearrange(pred_changes, "b (lev var) lon lat -> b lev var lon lat", lev = 3)

        # Collect only the single requested forecast_step
        for j, var in enumerate(var_names):
            if all(len(collectors_f[var][lvl]) >= max_hist_samples_per_step for lvl in range(3)):
                continue
            
            f_arr = pred_changes_rearranged[0, :, j, :, :].detach().cpu()
            t_arr = targets_rearranged[0, :, j, :, :].detach().cpu()
            # optional un-normalize (left as noop unless dataset_obj provided and user requests)
            if unnormalize and dataset_obj is not None:
                pass
            
            for lvl in range(3):
                mask = np.isfinite(f_arr[lvl].numpy().ravel()) & np.isfinite(t_arr[lvl].numpy().ravel())
                if mask.sum() == 0:
                    continue
                collectors_f[var][lvl].append(f_arr[lvl].numpy().ravel()[mask])
                collectors_t[var][lvl].append(t_arr[lvl].numpy().ravel()[mask])

        # break early if all collectors full
        # break early if collectors for the requested step are full for all vars
        all_full = all(len(collectors_f[v][lvl]) >= max_hist_samples_per_step for v in var_names for lvl in range(3))
        if all_full:
            break

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for var in var_names:
        # build single-step lists (length 1) for the plotting helper
        if len(collectors_f[var]) == 0:
            preds = [np.array([])]
            targs = [np.array([])]
        else:
            preds = [np.concatenate(collectors_f[var][lvl]) for lvl in range(3)]
            targs = [np.concatenate(collectors_t[var][lvl]) for lvl in range(3)]

        # Save one PNG per variable (include step in subdirectory)
        try:
            plot_forecast_error_distributions(preds, targs, variable=var, dir=out_dir)
        except Exception as e:
            print(f"Failed to plot forecast error distributions for var {var}: {e}")

def get_states(conds, state_changes):
    states = []
    num_vars = (conds.shape[1] - 8) // 6
    next_state = conds[0,num_vars*3+4:-4,:,:] + state_changes[0,:,:,:] if conds.shape[1]>15 else conds[0,:,:,:] + state_changes[0,:,:,:]
    states.append(next_state.clone())
    for step in range(1, state_changes.shape[0]):
        next_state = next_state + state_changes[step,:,:,:]
        states.append(next_state.clone())
    return torch.stack(states, dim=0)

if __name__ == "__main__":
    evaluate()
