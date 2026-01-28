"""Plot forecast losses for one or more models.

Usage:
    python -m pde_diff.plot_forecast_losses modelA modelB ...

The script will look for evaluation CSVs under `logs/<model_id>*`.
If forecasting CSVs are missing it will run the evaluate script for that model:
    python src/pde_diff/evaluate.py model.path=models/<model_id>

When CSVs exist the script will load `forecasting_losses_*.csv` and plot mean
loss vs forecast step (with 95% CI) and save PNGs next to the CSV files.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pde_diff.evaluate import find_best_fold

def get_csvs_for_model(model_id: str, models_dir: str = "models", logs_dir: str = "logs", evaluate_script: str = "src/pde_diff/evaluate.py", no_run: bool = False, timeout: int = 300) -> Optional[str]:
    """Ensure forecasting CSVs exist for `model_id`.

    Returns the log directory path if CSVs are (or become) available, otherwise None.
    """
    model_eval_dir = os.path.join(logs_dir,model_id)
    print(model_eval_dir)
    if os.path.exists(model_eval_dir):
        csvs = glob.glob(os.path.join(model_eval_dir, "forecasting_losses_*.csv"))
        if len(csvs) >= 1:
            return csvs

    if no_run:
        print("No running evaluate script allowed, returning None")
        return None

    # Run evaluate script for the model and wait for CSVs to appear
    cmd = [sys.executable, evaluate_script, f"model.path={models_dir}/{model_id}"]
    print(f"Running evaluate for model {model_id}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Evaluate script failed for model {model_id} (exit code != 0). Continuing.")

    if os.path.exists(model_eval_dir):
        csvs = glob.glob(os.path.join(model_eval_dir, "forecasting_losses_*.csv"))
        if len(csvs) >= 1:
            return csvs
    print(f"The path {model_eval_dir} does not exist")
    return None

def load_loss_from_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)

    cols = list(df.columns)
    try:
        x = sorted([int(c) for c in cols])
        cols_sorted = [str(c) for c in x]
    except Exception:
        cols_sorted = cols

    mean = df[cols_sorted].astype(float).mean(axis=0)
    std = df[cols_sorted].astype(float).std(axis=0)
    confidence = 1.96 * std / np.sqrt(len(df))
    return cols_sorted, mean, confidence

def plot_csv_loss(csv_paths,model_ids,loss_name, out_path: Optional[str] = None) -> str:

    plt.figure(figsize=(3, 2.5))
    colors = ["#8800FF", "#BF0251"]

    for k, (csv_path, model_id) in enumerate(zip(csv_paths,model_ids)):
        cols_sorted, mean, confidence = load_loss_from_csv(csv_path)
        print(f"Mean: {mean.values[0]}, CI: {confidence.values[0]} for model {model_id}")

        plt.plot(range(1, len(cols_sorted) + 1), mean.values,'o-', label=f"{model_id} - mean", color=colors[k])
        plt.fill_between(range(1, len(cols_sorted) + 1), (mean - confidence).values, (mean + confidence).values, alpha=0.3,color=colors[k])
    error_names = {
        "geo_wind": ("Geo. Wind", r"$\mathcal{R}_{2,MAE}(\tilde{x}_0)$"),
        "planetary": ("Plan. Vort.", r"$\mathcal{R}_{1,MAE}(\tilde{x}_0)$"),
        "mse": ("MSE of states", r"$||x_0-\tilde{x}_0||^2$"),
        "mse_change": ("MSE of change", r"$||\Delta x_0-\Delta \tilde{x}_0||^2$"),  }

    error = error_names.get(loss_name, loss_name)
    plt.xlabel("Forecast horizon (in hours)")
    plt.ylabel(f"{error[1]}")
    plt.title(f"{error[0]} (OOD Dec 2024)")
    plt.xticks(range(1, len(cols_sorted) + 1))
    plt.legend()
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(csv_path), f"comb_forecast_loss_vs_steps_{loss_name}.png")

    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Plot forecast losses for given model ids")
    #parser.add_argument("models", nargs="+", help="Model ids (folders inside `models/`)")
    parser.add_argument("--models-dir", default="models", help="Directory where model folders live")
    parser.add_argument("--logs-dir", default="logs", help="Directory where logs are stored")
    parser.add_argument("--evaluate-script", default="src/pde_diff/evaluate.py", help="Path to evaluate script")
    parser.add_argument("--no-run", action="store_true", help="Do not run evaluate if CSVs are missing; just skip")
    args = parser.parse_args()

    models=["DDPM", r"PIDM-$\mathcal{R}_1$: c=1e-2"]

    #for model_id in models:
    #    print(f"Processing model: {model_id}")
    #    best_fold= find_best_fold("models/"+model_id)
    #    print(f"Best fold: {best_fold[0]} with val loss {best_fold[1]}")
    #    model_eval_dir = best_fold[0] +"_eval"
        #csv_files = get_csvs_for_model(model_eval_dir, models_dir=args.models_dir, logs_dir=args.logs_dir, evaluate_script=args.evaluate_script, no_run=args.no_run)

    csv_paths_loss_sorted = {
        "mse": ["reports/figures/evaluation/era5_clean_hp3-baseline-full-retrain-retrain_eval_ood_12/forecasting_losses_mse.csv","reports/figures/evaluation/era5_clean_hp3-c1e2_pv-full-retrain-retrain_eval_ood_12/forecasting_losses_mse.csv"],
        "mse_change": ["reports/figures/evaluation/era5_clean_hp3-baseline-full-retrain-retrain_eval_ood_12/forecasting_losses_mse_change.csv","reports/figures/evaluation/era5_clean_hp3-c1e2_pv-full-retrain-retrain_eval_ood_12/forecasting_losses_mse_change.csv"],
        "geo_wind": ["reports/figures/evaluation/era5_clean_hp3-baseline-full-retrain-retrain_eval_ood_12/forecasting_losses_val_era5_sampled_geo_wind_residual(norm).csv","reports/figures/evaluation/era5_clean_hp3-c1e2_pv-full-retrain-retrain_eval_ood_12/forecasting_losses_val_era5_sampled_geo_wind_residual(norm).csv"],
        "planetary": ["reports/figures/evaluation/era5_clean_hp3-baseline-full-retrain-retrain_eval_ood_12/forecasting_losses_val_era5_sampled_planetary_residual(norm).csv","reports/figures/evaluation/era5_clean_hp3-c1e2_pv-full-retrain-retrain_eval_ood_12/forecasting_losses_val_era5_sampled_planetary_residual(norm).csv"],
    }

    for loss, csv_paths in csv_paths_loss_sorted.items():
        plot_csv_loss(csv_paths,models, loss)

if __name__ == "__main__":
    main()
