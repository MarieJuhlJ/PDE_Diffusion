
from omegaconf import OmegaConf
from src.pde_diff.scheduler import SchedulerRegistry
import torch

def test_scheduler_dimensions():

    # Quick test to verify the registry works
    config = OmegaConf.create({
        "name": "ddpm",
        "num_train_timesteps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02
    })
    scheduler = SchedulerRegistry.create(config)
    print(f"Created scheduler: {scheduler}")

    # Test add_noise method
    samples = torch.randn((4, 3, 64, 64))  # Example batch of images
    noise = torch.randn_like(samples)
    steps = torch.randint(0, config["num_train_timesteps"], (4,))
    noised_samples = scheduler.add_noise(samples, noise, steps)
    print(f"Noised samples shape: {noised_samples.shape}")
    assert noised_samples.shape == samples.shape, "Noised samples shape mismatch"
    print("add_noise method gives correct dimensions.")

    # Test sample method
    model_output = torch.randn_like(samples)
    timesteps = torch.randint(0, config["num_train_timesteps"], (4,))
    sampled = scheduler.sample(model_output, timesteps, samples)
    print(f"Sampled shape: {sampled.shape}")
    assert sampled.shape == samples.shape, "Sampled shape mismatch"
    print("sample method gives correct dimensions.")
