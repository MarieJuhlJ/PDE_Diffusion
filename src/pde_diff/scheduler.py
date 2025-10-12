import torch
from pde_diff.utils import SchedulerRegistry

class Scheduler:
    def __init__(self, config):
        self.config = config

    def add_noise(self, images, noise, steps):
        # Implement noise addition logic here
        return images + noise  # Placeholder implementation

@SchedulerRegistry.register("ddpm")
class DDPM_Scheduler(Scheduler):
    def __init__(self, config):
        super().__init__(config)
        # Initialize DDPM-specific parameters here
        self.num_train_timesteps = config.num_train_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])
        self.sigmas = torch.sqrt(self.betas)

    def add_noise(self, samples, noise, steps):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[steps])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[steps])[:, None, None, None]
        return sqrt_alpha_cumprod * samples + sqrt_one_minus_alpha_cumprod * noise

    def sample(self, model_output, timesteps, sample):
        alpha_sqrt = torch.sqrt(self.alphas[timesteps])[:, None, None, None]
        one_minus_alpha = 1 - self.alphas[timesteps]
        sqrt_one_minus_alpha = torch.sqrt(one_minus_alpha)[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[timesteps])[:, None, None, None]
        return 1/alpha_sqrt *(sample - sqrt_one_minus_alpha / sqrt_one_minus_alpha_cumprod * model_output)

if __name__ == "__main__":
    pass
