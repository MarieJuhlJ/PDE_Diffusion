import torch
import random
from omegaconf import DictConfig

from torch.utils.data import Subset, Dataset
from sklearn.model_selection import KFold

DEFAULT_ERROR_EST_METHOD = "cv-bl"

def split_dataset(cfg: DictConfig, dataset: Dataset) -> tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and validation sets based on the config.
    Args:
        cfg.dataset.time_series (bool): Whether the dataset is a time series dataset.
        cfg.k_folds (int, optional): Number of folds for cross-validation. If not
        provided, a simple train/val split is used.
        cfg.idx_fold (int, optional): Index of the current fold for cross-validation.

    Returns:
        dataset_train (Dataset): Training subset of the dataset.
        dataset_val (Dataset): Validation subset of the dataset.
    """
    print(f"Splitting dataset of size {len(dataset)} into training and validation sets...")
    if cfg.dataset.time_series:
        if cfg.get("k_folds", None):
            # Create a split and select appropriate subset of data for this fold:
            assert cfg.get("idx_fold", None) is not None, "idx_fold must be specified for k-fold cross-validation."

            error_est_method = cfg.get("error_est_method", DEFAULT_ERROR_EST_METHOD)

            if error_est_method == "cv-bl":
                len_dataset = len(dataset)
                val_size = len_dataset // cfg.k_folds
                val_start = (cfg.idx_fold - 1) * val_size
                val_start = val_start+1 if val_start!=0 else val_start
                val_range = range(val_start, val_start+val_size-1)
                train_range = list(set(range(len_dataset)) - set(list(val_range)+[min(val_range)-1,min(val_range)-2, max(val_range)+1, max(val_range)+2]))
                dataset_train = Subset(dataset, train_range)
                dataset_val = Subset(dataset, val_range)
                print(f"Training fold {cfg.idx_fold + 1}/{cfg.k_folds} using CV-BL with validation size {len(val_range)} and train size {len(train_range)}")

            elif error_est_method == "rep-hold-out":
                # Use repeated Hold-out
                random.seed(cfg.seed+cfg.idx_fold)
                a = random.randint(int(len(dataset)*0.8), len(dataset)-1)
                print(a)
                dataset_train = Subset(dataset, range(0, a))
                dataset_val = Subset(dataset, range(a, len(dataset)))
                random.seed(cfg.seed)  # Reset seed
            else:
                raise NotImplementedError(f"Error estimation method {error_est_method} not implemented.")
        else:
            # Use all data
            dataset_train = dataset
            dataset_val = None
            print("Using all training data to train model")
    else:
        if cfg.get("k_folds", None):
            assert cfg.get("idx_fold", None) is not None, "idx_fold must be specified for k-fold cross-validation."
            # Create a split and select appropriate subset of data for this fold:
            kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
            train_idx, val_idx = list(kf.split(dataset))[cfg.idx_fold-1]
            print(f"Training fold {cfg.idx_fold + 1}/{cfg.k_folds}")
            dataset_val = Subset(dataset, val_idx)
            dataset_train = Subset(dataset, train_idx)
        else:
            # Split dataset into train and validation sets
            val_size = int(len(dataset) * 0.1)
            train_size = len(dataset) - val_size
            dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    return dataset_train, dataset_val
