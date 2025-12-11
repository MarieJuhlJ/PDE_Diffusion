"""
Functions for visualizing the results of hyperparameter optimization with optuna.
"""
import os
import sys

import optuna

if __name__ == "__main__":
    dir = "reports/figures/hp_visualizations/"
    os.makedirs(dir, exist_ok=True)

    study_name = sys.argv[1]
    study_path = f"sqlite:///{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=study_path)

    fig = optuna.visualization.plot_contour(study, params=["learning_rate", "weight_decay"])
    fig.savefig(f"{dir}{study_name}_contour.png")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.savefig(f"{dir}{study_name}_parallel_coordinate.png")

    fig = optuna.visualization.plot_timeline(study)
    fig.savefig(f"{dir}{study_name}_timeline.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.savefig(f"{dir}{study_name}_param_importances.png")

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.savefig(f"{dir}{study_name}_intermediate_values.png")
