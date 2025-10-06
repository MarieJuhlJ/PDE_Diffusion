from omegaconf import OmegaConf
from src.pde_diff.model import DiffusionModel

def test_model_dimensions():
    # Quick test to verify the registrys etc. work, no test of correctness of the model

    model = DiffusionModel(OmegaConf.create({
        "experiment": {
        "hyperparameters": {
            "lr": 1e-3, 
            "weight_decay": 1e-4, 
            "batch_size": 32, 
            "max_epochs": 10, 
            "log_every_n_steps": 10}
        }, 
        "dataset": {"name": "mse"}, 
        "scheduler": {
            "name": "ddpm",
            "num_train_timesteps": 10,
            "beta_start": 0.0001,
            "beta_end": 0.02
        },
        "loss": {"name": "mse"},
        "model": {"name": "dummy"},
        "dataset":
        {
            "size": {
                "x": 64,
                "y": 64,
                "z": 2
            }
        }
    }))

    assert model.sample_loop(batch_size=2).shape == (2, int(model.data_size.z), int(model.data_size.x), int(model.data_size.y))