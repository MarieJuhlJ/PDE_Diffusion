import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from pde_diff.data import datasets
from pde_diff.utils import DatasetRegistry

def _accumulate_sum_sumsq(t, sum_, sumsq_):
    # t: [B, C, H, W]
    sum_ += t.sum(dim=(0, 2, 3))
    sumsq_ += (t * t).sum(dim=(0, 2, 3))
    return sum_, sumsq_

@torch.no_grad()
def compute_state_change_mean_std(
    loader,
    state_slice=slice(19, 34),
    device="cpu",
):
    state_sum = None
    state_sumsq = None
    change_sum = None
    change_sumsq = None

    n_state = 0
    n_change = 0

    for x, y in tqdm(loader, desc="Computing mean/std", unit="batch"):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        state = x[:, state_slice]  # [B, C_state, H, W]

        if y.dim() == 3:
            y = y.unsqueeze(1)     # ensure [B, C, H, W]

        B, C_s, H, W = state.shape
        _, C_c, Hc, Wc = y.shape
        assert (Hc, Wc) == (H, W)

        if state_sum is None:
            state_sum = torch.zeros(C_s, device=device)
            state_sumsq = torch.zeros(C_s, device=device)
            change_sum = torch.zeros(C_c, device=device)
            change_sumsq = torch.zeros(C_c, device=device)

        state_sum, state_sumsq = _accumulate_sum_sumsq(
            state, state_sum, state_sumsq
        )
        change_sum, change_sumsq = _accumulate_sum_sumsq(
            y, change_sum, change_sumsq
        )

        n_state += B * H * W
        n_change += B * H * W

    state_mean = state_sum / n_state
    change_mean = change_sum / n_change

    state_var = (state_sumsq / n_state) - state_mean**2
    change_var = (change_sumsq / n_change) - change_mean**2

    state_std = torch.sqrt(torch.clamp(state_var, min=0.0))
    change_std = torch.sqrt(torch.clamp(change_var, min=0.0))

    return (
        state_mean.cpu().tolist(),
        state_std.cpu().tolist(),
        change_mean.cpu().tolist(),
        change_std.cpu().tolist(),
    )

# ---- usage ----
cfg = OmegaConf.load("./configs/dataset/era5.yaml")
cfg.normalize = False
dataset = DatasetRegistry.create(cfg)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

state_mean, state_std, change_mean, change_std = compute_state_change_mean_std(
    loader,
    state_slice=slice(19, 34),
    device="cuda",  # or "cuda"
)

print("STATE_MEAN =", state_mean)
print("STATE_STD  =", state_std)
print("CHANGE_MEAN =", change_mean)
print("CHANGE_STD  =", change_std)