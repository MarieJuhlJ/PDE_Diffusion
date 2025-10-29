import torch
from pde_diff.utils import SchedulerRegistry

class Scheduler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def add_noise(self, images, noise, steps):
        # Implement noise addition logic here
        return images + noise  # Placeholder implementation

@SchedulerRegistry.register("ddpm")
class DDPM_Scheduler(Scheduler):
    def __init__(self, config):
        super().__init__(config)
        self.num_train_timesteps = config.num_train_timesteps
        beta_start = float(config.beta_start)
        beta_end = float(config.beta_end)

        betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0)
        sigmas = torch.sqrt((1-alpha_cumprod_prev)/(1-alpha_cumprod) * betas)
        Sigmas = sigmas**2

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("Sigmas", Sigmas)

    def add_noise(self, samples, noise, steps):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[steps])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[steps])[:, None, None, None]
        return sqrt_alpha_cumprod * samples + sqrt_one_minus_alpha_cumprod * noise

    def reconstruct_x0(self, x_t, model_output, timesteps):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[timesteps])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[timesteps])[:, None, None, None]
        x0_hat = (x_t - sqrt_one_minus_alpha_cumprod * model_output) / sqrt_alpha_cumprod
        return x0_hat 

    def sample(self, model_output, timesteps, sample):
        alpha_sqrt = torch.sqrt(self.alphas[timesteps])[:, None, None, None]
        one_minus_alpha = 1 - self.alphas[timesteps]
        sqrt_one_minus_alpha = torch.sqrt(one_minus_alpha)[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[timesteps])[:, None, None, None]
        return 1/alpha_sqrt *(sample - sqrt_one_minus_alpha / sqrt_one_minus_alpha_cumprod * model_output)

if __name__ == "__main__":
    pass
