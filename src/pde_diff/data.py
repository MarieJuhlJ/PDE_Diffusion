from pathlib import Path
from torch.utils.data import Dataset
from omegaconf import DictConfig
from pde_diff.utils import DatasetRegistry

@DatasetRegistry.register("dataset1")
class Dataset1(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.transform = None
        self.dataset = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple:
        """Return a given sample from the dataset."""
        x, y = self.dataset[idx]
        return x, y

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        # Implement your preprocessing logic here
        pass

def preprocess(name: str, data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = DatasetRegistry.create({"name": name, "data_path": data_path})
    dataset.preprocess(output_folder)

def get_train_val_dataset(cfg):
    train = DatasetRegistry.create(cfg)
    cfg.update({"train": False})
    val = DatasetRegistry.create(cfg)
    return train, val
