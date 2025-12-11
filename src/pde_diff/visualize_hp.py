"""
Functions for visualizing the results of hyperparameter optimization with optuna.
"""
import os
import sys

import matplotlib.pyplot as plt
import optuna

if __name__ == "__main__":
    dir = "reports/figures/hp_visualizations/"
    os.makedirs(dir, exist_ok=True)

    study_name = sys.argv[1]
    study_path =  "sqlite:///"+sys.argv[2]
    print(study_name, study_path)
    study = optuna.load_study(study_name=study_name, storage=study_path)

    # Contour plot
    ax = optuna.visualization.matplotlib.plot_contour(
        study, params=["lr", "weight_decay"]
    )
    ax.figure.set_size_inches(6, 3)
    ax.figure.savefig(f"{dir}{study_name}_contour.png", bbox_inches="tight")

    # Parallel coordinate plot
    ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    ax.figure.set_size_inches(6, 3)
    ax.figure.savefig(f"{dir}{study_name}_parallel_coordinate.png", bbox_inches="tight")

    # Timeline plot
    ax = optuna.visualization.matplotlib.plot_timeline(study)
    ax.figure.set_size_inches(6, 3)
    ax.figure.savefig(f"{dir}{study_name}_timeline.png", bbox_inches="tight")

    # Parameter importances plot
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    ax.figure.set_size_inches(6, 3)
    ax.figure.savefig(f"{dir}{study_name}_param_importances.png", bbox_inches="tight")

    # Intermediate values plot
    ax = optuna.visualization.matplotlib.plot_intermediate_values(study)
    ax.figure.set_size_inches(6, 3)
    ax.figure.savefig(f"{dir}{study_name}_intermediate_values.png", bbox_inches="tight")
