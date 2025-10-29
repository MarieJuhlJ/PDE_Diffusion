from pathlib import Path
from model import DiffusionModel
from utils import dict_to_namespace
import torch
import matplotlib.pyplot as plt
import yaml

def plot_samples(model, n=4, path=Path('./reports/figures')):
    samples = model.sample_loop(batch_size=n)
    samples = samples.cpu().numpy()
    fig, axs = plt.subplots(2, n, figsize=(n*3, 3))
    for i in range(n):
        axs[0, i].imshow(samples[i, 0], cmap='magma')
        axs[0, i].set_title(f'Sample {i+1} - K')
        axs[0, i].axis('off')
        axs[1, i].imshow(samples[i, 1], cmap='magma')
        axs[1, i].set_title(f'Sample {i+1} - P')
        axs[1, i].axis('off')
    plt.savefig(path / 'samples.png', bbox_inches='tight')
    print(f"Saved samples to {path / 'samples.png'}")

if __name__ == "__main__":
    model_path = Path('./models')
    model_id = 'exp1-ozjsn'

    with open(model_path / model_id / 'config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = dict_to_namespace(cfg)
    diffusion_model = DiffusionModel(cfg)
    diffusion_model.load_model(model_path / model_id / f"best-{cfg.model.monitor}-weights.pt")
    diffusion_model = diffusion_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    plot_samples(diffusion_model, n=4)
