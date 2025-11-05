
def test_model_dimensions():
    from omegaconf import OmegaConf
    from src.pde_diff.model import DiffusionModel
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
        "scheduler": {
            "name": "ddpm",
            "num_train_timesteps": 10,
            "beta_start": 0.0001,
            "beta_end": 0.02
        },
        "loss": {"name": "mse"},
        "model": {"name": "dummy",
                  "save_best_model": True,
                  "monitor": "val_loss",
                  "mode": "min",
                  "save_dir": './models',
                  "id": None,
                  "x0_pred": 'mean'
        },
        "dataset":
        {
            "dims": {
                "x": 64,
                "y": 64,
                "z": 3,
                "input_dims": 26,
                "output_dims": 6
            },
            "validation_metrics": ['mse']
        }
    }))

    assert model.sample_loop(batch_size=2).shape == (2, int(model.dataset.dims.output_dims), int(model.dataset.dims.x), int(model.dataset.dims.y))
