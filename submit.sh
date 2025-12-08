#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J darcy_pde[1-5]
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

# python src/pde_diff/train.py experiment.hyperparameters.max_epochs=200 dataset=darcy loss.name=darcy
python src/pde_diff/train.py experiment.hyperparameters.max_epochs=400 experiment.hyperparameters.dropout=0.25 experiment.hyperparameters.lr=1e-3 dataset=darcy loss.name=darcy loss.c_residual=1e-3 scheduler.num_train_timesteps=100 idx_fold=$LSB_JOBINDEX k_folds=5 id=aaaad 
