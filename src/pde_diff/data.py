from pathlib import Path
from torch.utils.data import Dataset
from einops import rearrange
import pandas as pd
import numpy as np
import torch
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

@DatasetRegistry.register("fluid_data")
class FluidData(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.data_paths = [Path(cfg.path) / 'K_data.csv', Path(cfg.path) / 'p_data.csv']
        channels = len(self.data_paths)

        for i in range(channels):
            if i == 0:
                self.data = pd.read_csv(self.data_paths[i], header=None)
            else:
                self.data = np.stack((self.data, pd.read_csv(self.data_paths[i], header=None)), axis=-1)

        dtype = torch.float64 if cfg.use_double else torch.float32
        self.data = torch.tensor(self.data, dtype=dtype)
        self.num_datapoints = len(self.data)

        assert len(self.data.shape) == 3
        self.data = generalized_b_xy_c_to_image(self.data)

    def normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val)

    def unnorm(self, arr, min_val, max_val):
        return arr * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if index >= self.num_datapoints:
            raise IndexError('index out of range')
        return self.data[index]


def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)


if __name__ == "__main__":
    data_path = './data/darcy/'
    dataset = Dataset(data_path, use_double=False, return_img=True, gaussian_prior=False)
    print(f"Dataset length: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample K shape: {x.shape}, Sample p shape: {y.shape}")
