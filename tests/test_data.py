
def test_data_normalization():
    from omegaconf import OmegaConf
    from src.pde_diff.data import CIFAR10
    from torchvision import transforms
    import torch

    cfg = OmegaConf.create({
        "name": "cifar10",
        "path": "./data/cifar10/",
        "train": True
    })

    dataset = CIFAR10(cfg)
    sample = dataset[0]  # Get the first sample

    # Check if the sample is a tensor
    assert isinstance(sample, torch.Tensor), "Sample is not a tensor"

    # Check the shape of the sample (C, H, W)
    assert sample.shape == (3, 32, 32), f"Unexpected sample shape: {sample.shape}"

    # Check normalization: mean should be close to 0 and std close to 1
    mean = sample.mean(dim=[1, 2])
    std = sample.std(dim=[1, 2])

    assert torch.allclose(mean, torch.zeros(3), atol=0.1), f"Mean not close to 0: {mean}"
    assert torch.allclose(std, torch.ones(3), atol=0.1), f"Std not close to 1: {std}"
