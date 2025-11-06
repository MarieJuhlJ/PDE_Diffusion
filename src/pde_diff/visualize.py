from pathlib import Path
from model import DiffusionModel
from utils import dict_to_namespace
import torch
import matplotlib.pyplot as plt
import pandas as pd
import yaml

def plot_samples(model, n=4, out_dir=Path('./reports/figures')):
    save_dir = Path(out_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(save_dir / 'samples.png', bbox_inches='tight')
    print(f"Saved samples to {save_dir / 'samples.png'}")

def plot_training_metrics(model_id, out_dir=Path("./reports/figures")):
    df = pd.read_csv(Path("./logs") / model_id / "version_0" / "metrics.csv").sort_values(["epoch", "step"])
    save_dir = Path(out_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("train_loss", True,  "Train Loss vs Epoch"),
        ("val_loss", True,  "Validation Loss vs Epoch"),
        ("val_mse", False, "Validation MSE vs Epoch"),
        ("val_darcy_residual", True, "Validation Darcy Residual vs Epoch"),
    ]

    for col, logy, title in metrics:
        if col not in df:
            continue
        sub = df[["epoch", col]].dropna()
        if sub.empty:
            continue

        ax = sub.plot(x="epoch", y=col, legend=False, figsize=(8, 4.5))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(save_dir / f"{col}.png", dpi=150)
        plt.close(ax.figure)

    if {"val_loss", "val_mse"}.issubset(df.columns):
        sub = df[["epoch", "val_loss", "val_mse"]].dropna()
        if not sub.empty:
            ax = sub.plot(x="epoch", y=["val_loss", "val_mse"], figsize=(8, 4.5))
            ax.set_title("Validation Metrics vs Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.figure.tight_layout()
            ax.figure.savefig(save_dir / "val_metrics_combined.png", dpi=150)
            plt.close(ax.figure)

if __name__ == "__main__":
    model_path = Path('./models')
    model_id = 'exp1-ihnrf'

    plot_training_metrics(model_id)

    with open(model_path / model_id / 'config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = dict_to_namespace(cfg)
    diffusion_model = DiffusionModel(cfg)
    diffusion_model.load_model(model_path / model_id / f"best-val_loss-weights.pt")
    diffusion_model = diffusion_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    plot_samples(diffusion_model, n=4)
