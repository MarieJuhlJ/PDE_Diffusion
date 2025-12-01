import torch.nn as nn
import torch
import torch.nn.functional as F
import einops as ein
import math
from pde_diff.utils import LossRegistry, DatasetRegistry, GradientHelper
from pde_diff.grad_utils import *
from pde_diff import data
from pde_diff.data import datasets

class PDE_loss(nn.Module):
    def __init__(self, residual_fns):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.residual_fns = residual_fns
        self.c_data = None
        self.c_residuals = [1e-3 for _ in residual_fns]

    def forward(self, model_out, target, **kwargs):
        x0_hat = kwargs.get('x0_hat', None)
        var = kwargs.get('var', None)
        conditionals = kwargs.get('conditionals', None)

        if self.c_data is None:
            total = self.mse(model_out, target).mean()
        else:
            total = (self.mse(model_out, target) * self.c_data[:, None, None, None]).mean()
        for fn, w in zip(self.residual_fns, self.c_residuals):
            r = fn(x0_hat, var)
            total += + w * r
        return total

@LossRegistry.register("mse")
class MSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_data = None
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, model_out, target, **kwargs):
        if self.c_data is None:
            return self.mse(model_out, target).mean()
        else:
            return (self.mse(model_out, target) * self.c_data[:, None, None, None]).mean()

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

@LossRegistry.register("vorticity")
class VorticityLoss(PDE_loss):
    def __init__(self, cfg):
        residual_fns = [self.vorticity_residual_loss]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Omega = 7.292e-5 # rad s^-1
        self.T_0 = 288.15 #K
        self.R = 287 #J K^-1 kg^-1
        self.c_p = 1004 #J K^-1 kg^-1
        self.p_0 = 101325 #Pa
        self.a = 6.37e6 # m
        self.lat_range = [70, 46] # hardcoded for now
        self.lon_range = [0, 359] # hardcoded for now
        self.p = [45000, 50000, 55000] # (Pa) hardcoded for now
        self.dp = 5000 # (Pa) hardcoded for now

        lon_grid, lat_grid, dx_grid, dy_grid = self.make_distance_grids(
            nlon=480, # hardcoded for now
            lon_range=self.lon_range,
            nlat=32, # hardcoded for now
            lat_range=self.lat_range,
            device=self.device,
            dtype=torch.float32,
        )

        self.gradient_helper = GradientHelper(grid_distances={
            'dx': dx_grid[None, None, :, :],
            'dy': dy_grid[None, None, :, :],
        })
        super().__init__(residual_fns)

    def make_distance_grids(self, nlon, lon_range, nlat, lat_range, device, dtype):
        lon = torch.linspace(lon_range[0], lon_range[1], nlon, device=device, dtype=dtype)
        lat = torch.linspace(lat_range[0], lat_range[1], nlat, device=device, dtype=dtype)
        lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='ij')  # shape: (nlon, nlat)

        lon_n = lon_grid.roll(shifts=-1, dims=0)
        lat_n = lat_grid.roll(shifts=-1, dims=0)
        dx_grid = self.haversine(lon_grid, lat_grid, lon_n, lat_n)

        lon_e = lon_grid.roll(shifts=-1, dims=1)
        lat_e = lat_grid.roll(shifts=-1, dims=1)
        dy_grid = self.haversine(lon_grid, lat_grid, lon_e, lat_e)

        return lon_grid, lat_grid, dx_grid, dy_grid
    
    def haversine(self, lon1, lat1, lon2, lat2):
            lon1 = torch.deg2rad(lon1)
            lat1 = torch.deg2rad(lat1)
            lon2 = torch.deg2rad(lon2)
            lat2 = torch.deg2rad(lat2)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            pi = math.pi
            dlon = (dlon + pi) % (2 * pi) - pi

            a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
            c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a).clamp(min=1e-12))

            return self.a * c


    def set_mean_and_std(self, mean, std, diff_mean, diff_std):
        device = self.device

        self.mean = torch.as_tensor(mean).to(device)
        self.std = torch.as_tensor(std).to(device)
        self.diff_mean = torch.as_tensor(diff_mean).to(device)
        self.diff_std = torch.as_tensor(diff_std).to(device)

    def theta_0(self, p):
        return self.T_0 * (self.p_0 / p)**(self.R / self.c_p)
    
    def dx_dp(self, x):
        return ((x[:, -1] - x [:, 0]) / (self.dp * 2))
    
    def dxx_dpp(self, x):
        diff_1 = (x[:, 1] - x[:, 0]) / self.dp
        diff_2 = (x [:, 2] - x[:, 1]) / self.dp
        return ((diff_2 - diff_1)/self.dp)
    
    def sigma(self, p):
        kappa = self.R / self.c_p
        return (self.R * self.T_0 * kappa) / (p**2)


    def get_original_states(self, x0_previous, x0_pred):
        previous_states_unnormalized = (x0_previous * self.std[None, :, None, None]) + self.mean[None, :, None, None]
        current_state_unnormalized = (x0_pred * self.diff_std[None, :, None, None]) + self.mean[None, :, None, None]
        previous_state = ein.rearrange(previous_states_unnormalized, "b (var lev) lon lat -> b lev var lon lat", lev = 3)
        current_state = ein.rearrange(current_state_unnormalized, "b (var lev) lon lat -> b lev var lon lat", lev = 3)
        return previous_state, current_state

    def compute_residual(self, x0_previous, x0_change_pred):
        device = x0_change_pred.device
        dtype = x0_change_pred.dtype
        previous, current = self.get_original_states(x0_previous, x0_change_pred + x0_previous)

        wind_u_p, wind_v_p, pv_p, temp_p, geo_p = [previous[:, :, i] for i in range(5)]
        wind_u_c, wind_v_c, pv_c, temp_c, geo_c = [current[:, :, i] for i in range(5)]
        
        b, lev, nlon, nlat = geo_c.shape

        lats_deg = torch.linspace(self.lat_range[0],
                                  self.lat_range[1],
                                  nlat, device=device, dtype=dtype)
        phi = lats_deg * math.pi / 180.0
        f = 2.0 * self.Omega * torch.sin(phi)[None, None, None, :]
        phi0 = phi.mean()
        f0 = 2.0 * self.Omega * torch.sin(phi0)

        p = torch.tensor(self.p, device=device, dtype=dtype)[None, :, None, None]
        sigma = self.sigma(p)
        lap_geo = self.gradient_helper.laplacian_horizontal(geo_c)
        A = f0 / sigma
        term_vert = self.dx_dp(A) * self.dx_dp(geo_c) + A[:,1] * self.dxx_dpp(geo_c)
        residual = (1/f0 * lap_geo + f)[:,1] + term_vert - pv_c[:, 1] # Holton 6.66
        return residual
    
    def vorticity_residual_loss(self, x0_hat, var):
        num_channels = x0_hat.shape[1]
        num_batch = x0_hat.shape[0]
        x0_previous = x0_hat[:, :num_channels//2, :, :]
        x0_change_pred = x0_hat[:, num_channels//2:, :, :]

        residual = self.compute_residual(x0_previous, x0_change_pred)
        residual_log_likelihood = gaussian_log_likelihood(torch.zeros_like(residual), means=residual, variance=var[:num_batch])
        residual_loss = -1. * residual_log_likelihood
        return residual_loss.mean()
        

def gaussian_log_likelihood(x, means, variance, return_full = False):
    centered_x = x - means
    expand_dims = centered_x.dim() - variance.dim()
    variance_broadcast = variance.view(*variance.shape, *([1] * expand_dims))
    squared_diffs = (centered_x ** 2) / variance_broadcast
    if return_full:
        log_likelihood = -0.5 * (squared_diffs + torch.log(variance) + torch.log(2 * torch.pi)) # full log likelihood with constant terms
    else:
        log_likelihood = -0.5 * squared_diffs

    # avoid log(0)
    log_likelihood = torch.clamp(log_likelihood, min=-27.6310211159)

    return log_likelihood

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "name": "era5",
            "path": "/work3/s194572/data/era5/zarr/",  # jacobs path
            "use_double": False,
            "time_series": True,
            "validation_metrics": ["mse", "era5_vorticity"],
            "dims": {
                "x": 480,
                "y": 32,
                "input_dims": 53,
                "output_dims": 15,
            },
            "train": True,
            "atmospheric_features": ["u", "v", "pv", "t", "z"],
            "single_features": [],
            "static_features": [],
            "max_year": 2024,
            "time_step": 2,
            "lon_range": None,         # [0, 50]
            "lat_range": [70, 46],
            "downsample_factor": 3,
        }
    )

    dataset = DatasetRegistry.create(cfg)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    loss = VorticityLoss(None)
    loss.set_mean_and_std(dataset.means, dataset.stds,
                          dataset.diff_means, dataset.diff_stds)
    loss.device = torch.device("cpu")

    batch = next(iter(loader))   # <--- THIS is what you want
    x = batch[0]
    y = batch[1]
    previous = x[:, 19:34]       # keep batch dim, slice second dim
    current = y

    r_era5 = loss.compute_residual(previous, current).pow(2).mean()
    mean = previous.mean(dim=(0,2,3), keepdim=True)
    std  = previous.std(dim=(0,2,3), keepdim=True)

    random_prev = torch.randn_like(previous)
    random_curr = torch.randn_like(current)
    r_rand = loss.compute_residual(random_prev, random_curr).pow(2).mean()

    print("ERA5 residual:", r_era5.item())
    print("Random residual:", r_rand.item())

