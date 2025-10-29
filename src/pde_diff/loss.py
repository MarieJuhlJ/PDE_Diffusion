import torch.nn as nn
import torch
from pde_diff.utils import LossRegistry, DatasetRegistry
from pde_diff import data

class PDE_loss(nn.Module):
    def __init__(self, residual_fns, weights=None):
        super().__init__()
        self.mse = MSE(None)
        self.residual_fns = residual_fns
        self.weights = [1.0] * len(residual_fns) if weights is None else weights

    def forward(self, model_output, target, x0_hat, sigma_t):
        total = self.mse(model_output, target, x0_hat)
        for fn, w in zip(self.residual_fns, self.weights):
            r = fn(x0_hat, sigma_t)
            total = total + w * r
        return total

@LossRegistry.register("mse")
class MSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, model_output, target, x0_hat):
        return nn.functional.mse_loss(model_output, target)

@LossRegistry.register("darcy")
class DarcyLoss(PDE_loss):
    def __init__(self, cfg):
        residual_fns = [self.darcy_residual_loss]
        weights = [1.0]
        self.c = 10e-3
        super().__init__(residual_fns, weights)

    def darcy_source(self, H, W, r=10.0, w_frac=0.125, device=None, dtype=None):
        w = int(round(w_frac * H))
        f = torch.zeros((H, W), device=device, dtype=dtype)
        f[:w, :w] = r
        f[-w:, -w:] = -r
        return f

    def darcy_residual_F(self, x):
        B, C, H, W = x.shape
        assert C == 2 and H == W
        # dx = 1.0 / (H - 1)
        dx = 1.0
        device, dtype = x.device, x.dtype

        K, p = x[:, 0], x[:, 1]

        dp_dy, dp_dx = torch.gradient(p, dim=(1, 2), spacing=(dx, dx), edge_order=2)

        ux = -K * dp_dx
        uy = -K * dp_dy

        dux_dx = torch.gradient(ux, dim=(2,), spacing=(dx,), edge_order=2)[0]
        duy_dy = torch.gradient(uy, dim=(1,), spacing=(dx,), edge_order=2)[0]

        div_u = dux_dx + duy_dy
        div_K_grad_p = -div_u

        f = self.darcy_source(H, W, r=10.0, w_frac=0.125, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        return div_K_grad_p + f

    def darcy_residual_loss(self, x, sigma_t):
        sigma_bar = sigma_t / self.c
        F = self.darcy_residual_F(x)
        mse = (F**2).mean(dim=(1,2)) * (1 / 2 * sigma_bar)
        return mse.mean()

if __name__ == "__main__":
    from types import SimpleNamespace

    data_dirs = ('./data/darcy/K_data.csv', './data/darcy/p_data.csv')
    cfg = SimpleNamespace(
        name="fluid_data",
        data_directories=data_dirs,
        use_double=False,
        return_img=True,
        gaussian_prior=False
    )

    dataset = DatasetRegistry.create(cfg)

    print(f"Dataset length: {len(dataset)}")
    K, p = dataset[0]
    randomK = torch.randn_like(K)
    randomp = torch.randn_like(p)
    dl = DarcyLoss()
    x_true   = torch.stack((K, p), dim=0).unsqueeze(0)
    x_rand   = torch.stack((randomK, randomp), dim=0).unsqueeze(0)
    phys_true = dl.darcy_physics_residual(x_true).item()
    phys_rand = dl.darcy_physics_residual(x_rand).item()
    print("physics(true) :", phys_true)
    print("physics(random):", phys_rand)

    print("full loss (random->true):", dl(x_rand, x_true).item())
    print("full loss (true->true):  ", dl(x_true, x_true).item())
