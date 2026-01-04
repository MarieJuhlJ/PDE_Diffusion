import hydra
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(version_base=None, config_name="evaluate.yaml", config_path="../../configs")
def evaluate(cfg: DictConfig):
    # load model config from cfg.model.path
    best_fold = find_best_fold(cfg.model.path)
    print(f"Best fold: {best_fold} with val")
    model_path = cfg.model.path + f"-{best_fold[0]}"
    model_cfg = OmegaConf.load(model_path + "/config.yaml")
    cfg = OmegaConf.merge(model_cfg, cfg)

    pl.seed_everything(cfg.seed)
    print(cfg)
    model = DiffusionModel(cfg)
    model.load_state_dict(torch.load(cfg.model.path + "/best-val_loss-weights.pt"))
    cfg.dataset.path= "./data/era5/zarr_test/"  #TODO: not hardcoded
    dataset_test = DatasetRegistry.create(cfg.dataset)

    dataloader_test = DataLoader(dataset_test,
                                 batch_size=cfg.experiment.hyperparameters.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 persistent_workers=True)
    dir = "logs"
    name = cfg.id+"_eval"
    os.makedirs(dir, exist_ok=True)
    logger = pl.pytorch.loggers.CSVLogger(dir, name=name)

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=logger)
    results = trainer.test(model, dataloader_test) #TODO: extend to multiple losses
    print(f"Test results: {results}")

    if cfg.dataset.name in ["era5"]:
        print("Evaluating forecasting performance...")
        forecasting_losses = evaluate_forecasting(model, dataset_test, steps=cfg.steps, losses=["mse"],dir=os.path.join(dir, name))
        # save losses to csv
        for loss_name in forecasting_losses.keys():
            path_csv = os.path.join(dir, name, f"forecasting_losses_{loss_name}.csv")
            with open(path_csv, "w") as f:
                f.write("forecast_step," + ",".join(forecasting_losses[loss_name].keys()) + "\n")
                for forecast_no in range(len(forecasting_losses[loss_name][str(1)])):
                    f.write(f"Forecast{forecast_no+1}")
                    for step in forecasting_losses[loss_name].keys():
                        f.write(f",{forecasting_losses[loss_name][step][forecast_no]}")
                    f.write("\n")
            print(f"Forecasting losses saved to CSV {path_csv}.")

        #save mean losses
        path_csv = os.path.join(dir, name, f"forecasting_losses_mean.csv")
        with open(path_csv, "w") as f:
            f.write("forecast_step," + ",".join(forecasting_losses[loss_name].keys()) + "\n")
            for step in range(1, cfg.steps+1):
                f.write(f"{step}")
                for loss_name in forecasting_losses.keys():
                    mean_loss = sum(forecasting_losses[loss_name][str(step)]) / len(forecasting_losses[loss_name][str(step)])
                    f.write(f",{mean_loss}")
                f.write("\n")
        print(f"Mean forecasting losses saved to CSV {path_csv}.")

        for loss_name, loss_values in forecasting_losses.items():
            for step, values in loss_values.items():
                mean_loss = sum(values) / len(values)
                print(f"Step {step}: {loss_name} = {mean_loss}")

def find_best_fold(model_path, fold_no=5, log_path="logs"):
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

    return best_model


def save_samples(forecasted_states, dir: Path, target_state=None):
    os.makedirs(dir, exist_ok=True)
    for i in range(forecasted_states.shape[1]):
        if i==0:
            for j,var in enumerate(["u","v","pv","t","z"]):
                if target_state is not None:
                    min_val = min(torch.min(target_state[:, j*3+1, :, :]),torch.min(forecasted_states[:,i, j*3+1, :, :]))
                    max_val = max(torch.max(target_state[:, j*3+1, :, :]),torch.max(forecasted_states[:,i, j*3+1, :, :]))
                visualize_era5_sample(target_state[:, j*3+1, :, :], variable=var, sample_idx=f"_target_var_{j*3+1}_sample_T{i}", dir=dir, color_bar_limit=(min_val, max_val))
                visualize_era5_sample(forecasted_states.detach().cpu()[0, i, j*3+1, :, :], variable=var, sample_idx=f"_forecast_var_{j*3+1}_sample_T{i}", dir=dir, color_bar_limit=(min_val, max_val) if target_state is not None else None)
        else:
            visualize_era5_sample(forecasted_states.detach().cpu()[0, i, 10, :, :], variable="t", sample_idx=f"_forecast_var_{10}_sample_T{i}", dir=dir)
    pickle_path = os.path.join(dir,f"forecast_sample.pt")
    torch.save(forecasted_states.detach().cpu(), pickle_path)
    print(f"Forecasted samples saved to {pickle_path}.")


def evaluate_forecasting(model: DiffusionModel, dataset_test: Dataset, steps: int=8, losses: list[str]=["mse", "fb"], dir=Path("reports/figures")) -> dict[str, list[float]]:

    dataloader_test = DataLoader(dataset_test,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4,
                                 persistent_workers=True)
    max_steps = len(dataloader_test)

    loss_fns = {loss: LossRegistry.create(OmegaConf.create({"name":loss})) for loss in losses}
    loss = {loss: {str(i+1): [] for i in range(steps)} for loss in losses}

    forecasts = []

    #use tqdm for progress bar

    for i, state in enumerate(tqdm(dataloader_test)):
        conditionals, target = state

        if not steps>max_steps:
            forecasted_states = model.forecast(conditionals, steps=steps)

            max_steps -= 1

            forecasts.append(forecasted_states.detach().cpu())

            if i<3:
                dir_sample=os.path.join(dir, f"sample_{i}")
                os.makedirs(dir_sample, exist_ok=True)

                target_state = dataset_test._unnormalize(conditionals[:,19:34,:,:], dataset_test.means, dataset_test.stds)
                + dataset_test._unnormalize(target.detach().cpu(), dataset_test.diff_means, dataset_test.diff_stds) 
                
                un_norm_forecasted_states = dataset_test._unnormalize(forecasted_states.detach().cpu(), dataset_test.means, dataset_test.stds)

                save_samples(un_norm_forecasted_states, dir=dir_sample,target_state=target_state) 

        for loss_name, loss_fn in loss_fns.items():
            for n in range(len(forecasts)):
                remaining_forecast = forecasts[n].shape[1]

                if remaining_forecast==0:
                    #Need to remove from list
                    continue

                l = loss_fn(forecasts[n][:,0], target.detach().cpu())

                #remove forecasted step
                forecasts[n] = forecasts[n][:,1:]

                #Append loss to correct number forecast step
                forecast_step = steps-remaining_forecast+1
                loss[loss_name][str(forecast_step)].append(l.item())

            if forecasts[0].shape[1]==0:
                forecasts.pop(0)
    return loss






if __name__ == "__main__":
    evaluate()
