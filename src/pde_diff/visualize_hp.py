"""
Functions for visualizing the results of hyperparameter optimization with optuna.
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import optuna

# Path to your TTF file
font_path = "./times.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'


if __name__ == "__main__":
    dir = "reports/figures/hp_visualizations/"
    os.makedirs(dir, exist_ok=True)

    study_name = sys.argv[1]
    study_path =  "sqlite:///"+study_name+".db"
    print(study_name, study_path)
    study = optuna.load_study(study_name=study_name, storage=study_path)

    
    valid_trials = [
        t for t in study.get_trials()
        if t.state in (optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED)
    ]

    # Create a temporary in-memory study
    filtered_study = optuna.create_study(
        study_name=study_name + "_filtered",
        direction=study.direction
    )

    # Add the filtered trials
    filtered_study.add_trials(valid_trials)

    # Contour plot
    ax = optuna.visualization.matplotlib.plot_contour(
        filtered_study, params=["lr", "weight_decay"]
    )
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(f"{dir}{study_name}_contour.png", bbox_inches='tight')

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
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(f"{dir}{study_name}_param_importances.png", bbox_inches='tight')

    # Intermediate values plot
    ax = optuna.visualization.matplotlib.plot_intermediate_values(filtered_study)
    ax.figure.set_size_inches(8, 4)
    ax.figure.savefig(f"{dir}{study_name}_intermediate_values.png", bbox_inches='tight')
