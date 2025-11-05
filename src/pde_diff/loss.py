import torch.nn as nn
import torch
import torch.nn.functional as F
import einops as ein
from pde_diff.utils import LossRegistry, DatasetRegistry
from pde_diff.grad_utils import *
from pde_diff import data

class PDE_loss(nn.Module):
    def __init__(self, residual_fns):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.residual_fns = residual_fns
        self.c_data = None
        self.c_residuals = [1e-3 for _ in residual_fns]

    def forward(self, model_output, target, **kwargs):
        x0_hat = kwargs.get('x0_hat', None)
        var = kwargs.get('var', None)

        if self.c_data is None:
            total = self.mse(model_output, target).mean()
        else:
            total = (self.mse(model_output, target) * self.c_data[:, None, None, None]).mean()
        for fn, w in zip(self.residual_fns, self.c_residuals):
            r = fn(x0_hat, var)
            total += + w * r
        return total

@LossRegistry.register("mse")
class MSE(nn.Module):
    def __init__(self, cfg):
        self.c_data = None
        self.mse = nn.MSELoss(reduction='none')
        super().__init__()

    def forward(self, model_output, target, **kwargs):
        if self.c_data is None:
            return self.mse(model_output, target).mean()
        else:
            return (self.mse(model_output, target) * self.c_data[:, None, None, None]).mean()

@LossRegistry.register("darcy")
class DarcyLoss(PDE_loss):
    def __init__(self, cfg):
        residual_fns = [self.darcy_residual_loss]
        self.eps = 1e-8
        self.eps_K = 1e-6
        self.c = 1e-6
        self.input_dim = 2
        self.pixels_per_dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        domain_length = 1.0
        d0 = domain_length / (self.pixels_per_dim - 1)
        d1 = domain_length / (self.pixels_per_dim - 1)

        self.grads = GradientsHelper(d0=d0, d1=d1, fd_acc = 2, periodic=False, device=self.device)
        self.trapezoidal_weights = self.create_trapezoidal_weights()
        f_s = self.darcy_source(self.pixels_per_dim, self.pixels_per_dim)
        self.f_s = generalized_image_to_b_xy_c(f_s.unsqueeze(0)).to(self.device)
        super().__init__(residual_fns)

    def darcy_source(self, H, W, r=10.0, w_frac=0.125, device=None, dtype=None):
        w = int(round(w_frac * H))
        f = torch.zeros((H, W), device=device, dtype=dtype)
        f[:w, :w] = r
        f[-w:, -w:] = -r 
        return f

    def darcy_residual_F(self, x):
        B, C, H, W = x.shape
        assert C == 2 and H == W
        dx = 1.0 / (H - 1)

        K = x[:, 0]
        p = x[:, 1]

        dp_dy, dp_dx = torch.gradient(p, dim=(1, 2), spacing=(dx, dx), edge_order=2)
        ux = -K * dp_dx
        uy = -K * dp_dy
        dux_dx = torch.gradient(ux, dim=(2,), spacing=(dx,), edge_order=2)[0]
        duy_dy = torch.gradient(uy, dim=(1,), spacing=(dx,), edge_order=2)[0]
        div_u = dux_dx + duy_dy
        div_K_grad_p = -div_u

        f = self.darcy_source(H, W, r=10.0, w_frac=0.125, device=x.device, dtype=x.dtype)
        f = f.unsqueeze(0).expand(B, -1, -1)
        return div_K_grad_p + f

    def darcy_residual_loss(self, x0_hat, var):
        # sigma_t = torch.as_tensor(sigma_t, device=x0_hat.device, dtype=x0_hat.dtype)

        # x = x0_hat.float()
        # F = self.darcy_residual_F(x)

        # weight = 0.5 * self.c / sigma_t
        # per_sample = (F**2).mean(dim=(1, 2)) * weight
        # return per_sample.mean()
        residual = self.compute_residual(x0_hat)
        residual_log_likelihood = gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var)
        residual_loss = -1. * residual_log_likelihood
        return residual_loss.mean()

    def create_trapezoidal_weights(self):        
        # identify corner nodes
        trapezoidal_weights = torch.zeros((1, self.pixels_per_dim, self.pixels_per_dim))
        trapezoidal_weights = trapezoidal_weights.to(self.device)
        trapezoidal_weights[..., 0,0] = 1.
        trapezoidal_weights[..., 0,-1] = 1.
        trapezoidal_weights[..., -1,0] = 1.
        trapezoidal_weights[..., -1,-1] = 1.
        # identify boundary nodes
        trapezoidal_weights[..., 1:-1,0] = 2.
        trapezoidal_weights[..., 1:-1,-1] = 2.
        trapezoidal_weights[..., 0,1:-1] = 2.
        trapezoidal_weights[..., -1,1:-1] = 2.
        # identify interior nodes
        trapezoidal_weights[..., 1:-1,1:-1] = 4.
        # assert that no node is 0
        assert torch.all(trapezoidal_weights != 0)
        trapezoidal_weights *= (1./self.pixels_per_dim)**2 / 4.
        trapezoidal_weights = generalized_image_to_b_xy_c(trapezoidal_weights)
        return trapezoidal_weights

    def compute_residual(self, x0_pred):
        assert len(x0_pred.shape) == 4, 'Model output must be a tensor shaped as an image (with explicit axes for the spatial dimensions).'
        batch_size, output_dim, pixels_per_dim, pixels_per_dim = x0_pred.shape
        
        p = x0_pred[:, 0]
        permeability_field = x0_pred[:, 1]
        p_d0 = self.grads.stencil_gradients(p, mode='d_d0')
        p_d1 = self.grads.stencil_gradients(p, mode='d_d1')
        grad_p = torch.stack([p_d0, p_d1], dim=-3)
        p_d00 = self.grads.stencil_gradients(p, mode='d_d00')
        p_d11 = self.grads.stencil_gradients(p, mode='d_d11')
        perm_d0 = self.grads.stencil_gradients(permeability_field, mode='d_d0')
        perm_d1 = self.grads.stencil_gradients(permeability_field, mode='d_d1')
        velocity_jacobian = torch.zeros(batch_size, output_dim, self.input_dim, pixels_per_dim, pixels_per_dim, device=x0_pred.device, dtype=x0_pred.dtype)
        velocity_jacobian[:, 0, 0] = -permeability_field * p_d00 - perm_d0 * p_d0
        velocity_jacobian[:, 1, 1] = -permeability_field * p_d11 - perm_d1 * p_d1
        x0_pred = generalized_image_to_b_xy_c(x0_pred)
        grad_p = generalized_image_to_b_xy_c(grad_p)
        velocity_jacobian = generalized_image_to_b_xy_c(velocity_jacobian)
                
        # obtain equilibrium equations for residual
        eq_0 = velocity_jacobian[:,:, 0, 0] + velocity_jacobian[:, :, 1, 1] - self.f_s
        residual = eq_0

        p_int = self.trapezoidal_weights * x0_pred[..., 0].detach()
        correction = ein.reduce(p_int, 'b ... -> b 1', 'sum')

        x0_pred_zero_p = x0_pred[:,:,0] - correction
        x0_pred_zero_p = torch.stack([x0_pred_zero_p, x0_pred[:,:,1]], dim=-1)
        x0_pred = x0_pred_zero_p
        
        # manually add BCs
        # reshape output to match image shape
        grad_p_img = generalized_b_xy_c_to_image(grad_p)
        residual_bc = torch.zeros_like(grad_p_img)
        residual_bc[:,0,0,:] = -grad_p_img[:,0,0,:] # xmin / top (acc. to matplotlib visualization)
        residual_bc[:,0,-1,:] = grad_p_img[:,0,-1,:] # xmax / bot
        residual_bc[:,1,:,0] = grad_p_img[:,1,:,0] # ymin / left
        residual_bc[:,1,:,-1] = -grad_p_img[:,1,:,-1] # ymax / right

        residual_bc = generalized_image_to_b_xy_c(residual_bc)
        residual = torch.cat([eq_0.unsqueeze(-1), residual_bc], dim=-1)

        return residual.mean(dim=tuple(range(1, residual.ndim))) if residual.ndim > 1 else residual

def gaussian_log_likelihood(x, means, variance, return_full = False):
    centered_x = x - means    
    squared_diffs = (centered_x ** 2) / variance
    if return_full:
        log_likelihood = -0.5 * (squared_diffs + torch.log(variance) + torch.log(2 * torch.pi)) # full log likelihood with constant terms
    else:
        log_likelihood = -0.5 * squared_diffs

    # avoid log(0)
    log_likelihood = torch.clamp(log_likelihood, min=-27.6310211159)

    return log_likelihood

if __name__ == "__main__":
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        name="fluid_data",
        path="./data/darcy/",
        use_double=False,
        time_series=False,
        validation_metrics=["mse", "darcy"],
        dims=SimpleNamespace(
            x=64,
            y=64,
            z=2
        ),
        train=True,
        device="cpu"
    )

    dataset = DatasetRegistry.create(cfg)

    print(f"Dataset length: {len(dataset)}")
    K, p = dataset[0]
    randomK = torch.randn_like(K)
    randomp = torch.randn_like(p)
    dl = DarcyLoss(cfg)
    x_true   = torch.stack((K, p), dim=0).unsqueeze(0)
    x_rand   = torch.stack((randomK, randomp), dim=0).unsqueeze(0)
    phys_true = dl.compute_residual(x_true).mean().item()
    phys_rand = dl.compute_residual(x_rand).mean().item()
    phys_true_own = dl.darcy_residual_F(x_true).mean().item()
    phys_rand_own = dl.darcy_residual_F(x_rand).mean().item()
    phys_true_loss = dl.darcy_residual_loss(x_true, var=1.0).item()
    phys_rand_loss = dl.darcy_residual_loss(x_rand, var=1.0).item()
    print("physics(true) :", phys_true)
    print("physics(random):", phys_rand)
    print("physics true (own)", phys_true_own)
    print("physics rand (own)", phys_rand_own)
    print("physics true (loss)", phys_true_loss)
    print("physics rand (loss)", phys_rand_loss)
