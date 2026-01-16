#!/bin/sh 
#BSUB -q gpua100
#BSUB -J era5_c1e3[1-5]
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
#BSUB -o ../hpc_files/%J.out
#BSUB -e ../hpc_files/%J.err

# 1) Activate the virtual environment
. /work3/s194572/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

# python src/pde_diff/train.py experiment.hyperparameters.max_epochs=200 dataset=darcy loss.name=darcy model.name=unet3d
# python src/pde_diff/train.py experiment.hyperparameters.max_epochs=200 experiment.hyperparameters.lr=1e-3 dataset=darcy loss.name=darcy loss.c_residual=1e-3 scheduler.num_train_timesteps=100 model.name=unet3d dataset.path=./data/darcy/big id=dp
# python src/pde_diff/data/darcy_data_generation.py

# python src/pde_diff/train.py experiment.hyperparameters.max_epochs=200 dataset=era5 loss.name=vorticity model.name=unet3d_conditional

#lr: 0.0012420100674942623
#weight: 3.978931581387639e-06
#batch_size: 16

# python src/pde_diff/train.py dataset=era5 loss.name=vorticity model.name=unet3d_conditional experiment.hyperparameters.lr=0.0012420100674942623 experiment.hyperparameters.batch_size=16 experiment.hyperparameters.weight_decay=3.978931581387639e-06 loss.c_residual=[0.0,0.0,1e-2]
python src/pde_diff/train.py model.name=unet3d_conditional experiment=era5_clean_hp2_50e id=c1e3 k_folds=5 idx_fold=$LSB_JOBINDEX loss.name=vorticity loss.c_residual=1e-3
# python src/pde_diff/train.py model.name=unet3d_conditional experiment=era5_clean_hp2_50e id=base_no_eps_realsample loss.name=vorticity loss.c_residual=0.0
# python src/pde_diff/data/plot_data_distribution.py