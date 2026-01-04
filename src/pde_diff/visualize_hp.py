"""
Functions for visualizing the results of hyperparameter optimization with optuna.
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import datetime

import optuna

# Path to your TTF file
font_path = "./times.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'

def print_trial(trial, title=None):
    if title:
        print(title)
        print("--" * 20)
    print(f"Trial number: {trial.number}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    if trial.values:
        print("Objective value:", trial.values[-1])

    if trial.datetime_complete and trial.datetime_start:
        print(
            "Time to train:",
            (trial.datetime_complete - trial.datetime_start).total_seconds()
        )

    print("--" * 20)

def filter_trials(
    trials,
    filter_large_objectives=0,
    std_filter=False,
    std_k=2.0,
):
    """Filter trials based on objective value outliers."""
    values = np.array([t.values[-1] for t in trials if t.values])

    if len(values) == 0:
        return trials

    mask = np.ones(len(trials), dtype=bool)

    if filter_large_objectives:
        # Drop the number of largest objective value given in filter_large_objectives
        for _ in range(filter_large_objectives):
            max_val = np.max(values[mask])
            mask &= values < max_val

    if std_filter:
        mean = np.mean(values[mask])
        std = np.std(values[mask])
        mask &= np.abs(values - mean) <= std_k * std
    removed_trials = [t for t, m in zip(trials, mask) if not m]
    for t in removed_trials:
        print(f"Removing trial {t.number} with objective value {t.values[-1]} as an outlier.")
        print(f"status: {t.state}")
        print(f"Params: {t.params}")    
    return [t for t, m in zip(trials, mask) if m]

def load_study_and_filter(study_name, filter_large_objectives=3, std_filter=False, std_k=2.0):
    storage_url = f"sqlite:///{study_name}.db"
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )

    complete_trials = [
        t for t in study.get_trials()
        if t.state in [optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.PRUNED]
        ]
    print(f"Total complete trials: {len(complete_trials)}")
    
    valid_trials = filter_trials(
        complete_trials,
        filter_large_objectives=filter_large_objectives,
        std_filter=std_filter,
        std_k=std_k,
    )
    print(f"Valid trials after filtering: {len(valid_trials)}")

    return study, valid_trials

if __name__ == "__main__":
    dir = "reports/figures/hp_visualizations/"
    os.makedirs(dir, exist_ok=True)

    study_name = sys.argv[1]
    
    study, valid_trials = load_study_and_filter(study_name)

    if sys.argv.__len__() > 2:
        study_name2 = sys.argv[2]
        study2, valid_trials2 = load_study_and_filter(study_name2)
        # Merge valid trials
        study_name += "_" + study_name2
        valid_trials.extend(valid_trials2)

    # Create a temporary in-memory study
    filtered_study = optuna.create_study(
        study_name=study_name + "_filtered",
        direction=study.direction
    )

    # Add the filtered trials
    filtered_study.add_trials(valid_trials)

    print_trial(filtered_study.best_trial, title="Best filtered trial")

    # Contour plot
    ax = optuna.visualization.matplotlib.plot_contour(
        filtered_study, params=["batch_size","lr"], target_name="MSE Loss"
    )
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(f"{dir}{study_name}_contour_lr_batch_size.png", bbox_inches='tight')

    # Contour plot
    ax = optuna.visualization.matplotlib.plot_contour(
        filtered_study, params=["lr", "weight_decay"], target_name="MSE Loss"
    )
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(f"{dir}{study_name}_contour_lr_weight_decay.png", bbox_inches='tight')

    # Parallel coordinate plot
    ax = optuna.visualization.matplotlib.plot_parallel_coordinate(filtered_study)
    ax.figure.set_size_inches(10, 4)
    ax.figure.savefig(f"{dir}{study_name}_parallel_coordinate.png", bbox_inches='tight')

    # Timeline plot
    ax = optuna.visualization.matplotlib.plot_timeline(filtered_study)
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(f"{dir}{study_name}_timeline.png", bbox_inches='tight')

    # Parameter importances plot
    ax = optuna.visualization.matplotlib.plot_param_importances(filtered_study)
    ax.figure.set_size_inches(4, 4)
    ax.figure.savefig(f"{dir}{study_name}_param_importances.png", bbox_inches='tight')

    # Intermediate values plot
    ax = optuna.visualization.matplotlib.plot_intermediate_values(filtered_study)
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(f"{dir}{study_name}_intermediate_values.png", bbox_inches='tight')

    ax = optuna.visualization.matplotlib.plot_optimization_history(filtered_study)
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(f"{dir}{study_name}_history.png", bbox_inches='tight')
