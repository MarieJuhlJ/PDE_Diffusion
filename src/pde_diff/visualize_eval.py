import matplotlib.ticker as ticker

from pde_diff.visualize import *

RES_NAMES = {
    "plan": ("Planetary Vorticity",1, r"$s^{-2}$"),
    "gw": ("Geostrophic Wind", 2,r"$m \cdot s^{-1}$")
}

VAR_NAMES = {
    "u": "u",
    "v": "v",
    "t": "T",
    "z": "\Phi",
    "pv": "q_{E,}",
}

VAR_FULL_NAMES = {
    "u": "Eastward Wind",
    "v": "Northward Wind",
    "t": "Temperature",
    "z": "Geopotential",
    "pv": "Ertel Pot. Vort.",
}

# Define plotting helpers and load data. Keep functions concise and parameterized.
def plot_forecast_loss_vs_steps(df, figsize=(8,5), dir=None, loss_name=None):
    """Reads CSV saved by evaluate.py and plots loss vs forecast step for each Forecast# row."""
    cols = list(df.columns)
    try:
        x = sorted([int(c) for c in cols])
        cols_sorted = [str(c) for c in x]
    except Exception:
        cols_sorted = cols
    plt.figure(figsize=figsize)
    mean = df[cols_sorted].astype(float).mean(axis=0)
    std = df[cols_sorted].astype(float).std(axis=0)
    confidence = 1.96 * std / np.sqrt(len(df))
    plt.plot(range(1, len(cols_sorted)+1), mean.values, marker='o', label='Mean')
    plt.fill_between(range(1, len(cols_sorted)+1), (mean - confidence).values, (mean + confidence).values, alpha=0.3, label='Mean 95% CI')
    plt.xlabel('Forecast step')
    plt.ylabel(f'{loss_name} Loss')
    plt.title('Forecast loss vs forecast steps')
    plt.xticks(range(1, len(cols_sorted)+1))
    plt.legend()
    plt.tight_layout()
    if dir is not None:
        save_path = os.path.join(dir, f'forecast_loss_vs_steps_{loss_name}.png')
    else:
        save_path = f'forecast_loss_vs_steps_{loss_name}.png'
    plt.savefig(save_path)
    print(f'Forecast loss vs steps plot saved to {save_path}.')

def plot_sample_target_absdiff_stacked(
    sample: torch.Tensor,
    target: torch.Tensor,
    variable: str = "t",
    sample_idx: int = 0,
    dir=Path("./reports/figures/samples"),
):

    sample = sample.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    vmin = min(sample.min(), target.min())
    vmax = max(sample.max(), target.max())

    fig = plt.figure(figsize=(8, 2.4), constrained_layout=True)

    fig.set_constrained_layout_pads(
        w_pad=0.00,
        h_pad=0.00,
        wspace=0.01,
        hspace=0.00,
    )

    # 3 rows, 2 columns (second column = colorbars)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[40, 1],   # thin colorbars
        height_ratios=[1, 1, 1],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    cax_top = fig.add_subplot(gs[0:2, 1])  # shared colorbar
    cax_bot = fig.add_subplot(gs[2, 1])    # bottom colorbar
    var = VAR_NAMES.get(variable, variable)
    var_name = VAR_FULL_NAMES.get(variable, variable)

    im1 = ax1.imshow(
        sample.T,
        cmap=COLOR_BARS.get(variable, "coolwarm"),
        extent=EXTENT_SUBSET,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax1.set_title(rf"Prediction of {var_name}, $\hat{{{var}}}_0$")
    ax1.set_xticklabels([])
    ax1.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax1.set_ylim(EXTENT_SUBSET[2],EXTENT_SUBSET[3])

    im2 = ax2.imshow(
        target.T,
        cmap=COLOR_BARS.get(variable, "coolwarm"),
        extent=EXTENT_SUBSET,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax2.set_title(rf"True state of {var_name}, ${{{var}}}_0$")
    ax2.set_xticklabels([])
    ax2.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax2.set_ylim(EXTENT_SUBSET[2],EXTENT_SUBSET[3])

    plt.colorbar(im1, cax=cax_top,label=f"{var}")

    diff = np.abs(target - sample)
    im3 = ax3.imshow(
        diff.T,
        cmap="Oranges",
        extent=EXTENT_SUBSET,
        aspect="auto",
    )
    ax3.set_title(rf"MAE of {var_name}, $|{{{var}}}_0-\hat{{{var}}}_0|$")
    ax3.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax3.set_ylim(EXTENT_SUBSET[2],EXTENT_SUBSET[3])
    print(f"Max absolute difference for variable {var_name}, sample {sample_idx}: {diff.max():.4f}")

    plt.colorbar(im3, cax=cax_bot,label=f"{VAR_UNITS.get(variable, variable)}")

    for ax in (ax1, ax2, ax3):
        ax.set_yticks([])

    out_path = os.path.join(dir, f"sample_w_target_diff_{sample_idx}_{variable}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {out_path}")

def plot_forecasts_vs_targets(
    forecasts: list,   # list of tensors or numpy arrays, length = 5
    targets: list,     # list of tensors or numpy arrays, length = 5
    variable: str = "t",
    sample_idx: int = 0,
    dir=Path("./reports/figures/samples"),
    states=True,
    limits=None
):
    """
    Plot recursive forecasts vs targets.
    Left column: forecasts (numbered)
    Right column: targets
    One shared colorbar on the right.
    """

    assert len(forecasts) == len(targets) == 5, "Expect exactly 5 steps"

    # Convert to numpy
    forecasts = [
        f.detach().cpu().numpy() if hasattr(f, "detach") else f
        for f in forecasts
    ]
    targets = [
        t.detach().cpu().numpy() if hasattr(t, "detach") else t
        for t in targets
    ]
    differences = [np.abs(t-f) for f, t in zip(forecasts, targets)]

    # Shared color scale
    vmin = min(f.min() for f in forecasts + targets) if not limits else limits[0]
    vmax = max(f.max() for f in forecasts + targets) if not limits else limits[1]
    vmin_d = min(f.min() for f in differences) if not limits else limits[2]
    vmax_d = max(f.max() for f in differences) if not limits else limits[3]
    print(f"For var {variable}, vmin, vmax, vmin_d, vmax_d")
    print(vmin, vmax, vmin_d, vmax_d)

    fig = plt.figure(figsize=(10, 2.4),constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.02,
        h_pad=0.01,
        wspace=0.02,
        hspace=0.02,
    )

    # 5 rows, 3 columns (forecast | target | colorbar | differences |colorbar)
    gs = fig.add_gridspec(
        nrows=5,
        ncols=5,
        width_ratios=[1, 1, 0.05,1,0.05],
        hspace=0.00,
        wspace=0.05,
    )

    cmap = COLOR_BARS.get(variable, "coolwarm")

    axes_forecast = []
    axes_target = []
    axes_diff = []

    var = VAR_NAMES.get(variable, variable)
    var_name = VAR_FULL_NAMES.get(variable, variable)

    for i in range(5):
        ax_f = fig.add_subplot(gs[i, 0])
        ax_t = fig.add_subplot(gs[i, 1])
        ax_d = fig.add_subplot(gs[i, 3])

        im = ax_f.imshow(forecasts[i].T,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto",extent=EXTENT_SUBSET,)
        ax_t.imshow(targets[i].T, cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto",extent=EXTENT_SUBSET,)
        im_diff = ax_d.imshow(differences[i].T,vmin=vmin_d,vmax=vmax_d, cmap="Oranges",aspect="auto",extent=EXTENT_SUBSET,)

        for axs in [ax_f, ax_t, ax_d]:
            axs.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
            axs.set_ylim(EXTENT_SUBSET[2],EXTENT_SUBSET[3])
            axs.set_yticks([])
            if i < 4:
                axs.set_xticklabels([])
        # Step numbering on the left
        ax_f.set_ylabel(f"Step {i+1}", rotation=0, labelpad=15, va="center")

        axes_forecast.append(ax_f)
        axes_target.append(ax_t)
        axes_diff.append(ax_d)

    
    if states:
        axes_forecast[0].set_title(rf"Forecasts of {var_name}, $\hat{{{var}}}_0$")
        axes_target[0].set_title(rf"True states of {var_name}, ${{{var}}}_0$")
        axes_diff[0].set_title(rf"MAE, $|{{{var}}}_0-\hat{{{var}}}_0|$")
    else:
        axes_forecast[0].set_title(rf"Forecasts of {var_name}, $\Delta \hat{{{var}}}_0$")
        axes_target[0].set_title(rf"True changes of {var_name}, $\Delta {{{var}}}_0$")
        axes_diff[0].set_title(rf"MAE, $|\Delta {{{var}}}_0- \Delta \hat{{{var}}}_0|$")

    cax = fig.add_subplot(gs[:, 2])
    cbar = plt.colorbar(im, cax=cax)
    #cbar.ax.tick_params(pad=6)
    cbar.set_label(f"{VAR_UNITS.get(variable, variable)}", labelpad=0)
    if variable == "z" or variable == "pv":
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))  # always use scientific notation

        cbar.formatter = formatter
        cbar.update_ticks()

    cax2 = fig.add_subplot(gs[:, 4])
    cbar2 = plt.colorbar(im_diff, cax=cax2)
    if variable == "pv":
        formatter2 = ticker.ScalarFormatter(useMathText=True)
        formatter2.set_scientific(True)
        formatter2.set_powerlimits((0, 0))
        cbar2.formatter = formatter2
        cbar2.update_ticks()

    out_path = os.path.join(dir, f"forecasts_vs_targets_w_diff_sample_{sample_idx}_{variable}{"changes" if not states else ""}.png")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved forecast vs target plot to {out_path}")


def plot_forecast_error_distributions(
    forecasts,
    targets,
    variable: str = "t",
    dir=Path("./reports/figures/"),
    bins: int = 120,
    percentiles=(1.0, 99.0),
):
    """
    Plot histograms of forecast-change minus target-change for each step in the forecast horizon.

    Parameters
    ----------
    forecasts : list[tensor|ndarray]
        Sequence (length = horizon) of forecast arrays for the step. Each element may be a torch Tensor
        or numpy array of spatial values.
    targets : list[tensor|ndarray]
        Sequence (same length as `forecasts`) of true target arrays for the same steps.
    variable : str
        Variable name used for file naming and titles.
    dir : Path or str
        Output directory to save the resulting figure.
    bins : int
        Number of histogram bins.
    figsize_per_step : (width, height)
        Width and height per subplot; total figure width is width * n_steps.
    percentiles : (low_pct, high_pct)
        Percentiles used to trim extreme tails when choosing x-limits for the histograms.
    """

    assert len(forecasts) == len(targets), "forecasts and targets must have same length"
    n_steps = len(forecasts)

    var = VAR_NAMES.get(variable, variable)

    # Convert to numpy arrays and collect per-step errors
    errors_per_lvl = [[] for _ in range(3)]
    for lvl in range(3):
        for f, t in zip(forecasts[lvl], targets[lvl]):
            if hasattr(f, "detach"):
                f = f.detach().cpu().numpy()
            if hasattr(t, "detach"):
                t = t.detach().cpu().numpy()
            err = (np.asarray(f) - np.asarray(t)).flatten()
            err = err[np.isfinite(err)] # filter non-finite values
            errors_per_lvl[lvl].append(err)

    # Determine common x-limits robustly from concatenated finite errors
    all_errors = np.concatenate([e for e in errors_per_lvl])
    
    low_p, high_p = percentiles
    # if too few samples, fallback to min/max
    if all_errors.size > 2:
        lo, hi = np.percentile(all_errors, [low_p, high_p])
    else:
        lo, hi = float(all_errors.min()), float(all_errors.max())
    # protect against degenerate range
    if lo == hi:
        lo -= 1e-6
        hi += 1e-6

    # Build figure with one subplot per forecast step
    fig, ax = plt.subplots(figsize=(3.8,3.8))
    colors = ["#2A9D8F", "#E76F51", "#1A4DAC"]
    levels = ["450hPa", "500hPa", "550hPa"]
    for lvl in range(3):
        data = np.array(errors_per_lvl[lvl]).flatten()

        mean = float(data.mean())
        std = float(data.std())

        ax.hist(data, bins=bins, range=(lo, hi), histtype="stepfilled", alpha=0.6, edgecolor="black", color=colors[lvl], label=f"{levels[lvl]}: Mean={mean:.2e},Std={std:.2e}")
        ax.axvline(mean, linestyle="-", linewidth=2, color=colors[lvl])
        ax.axvline(mean - std, linestyle="--", linewidth=1.2, color=colors[lvl])
        ax.axvline(mean + std, linestyle="--", linewidth=1.2, color=colors[lvl])

    ax.set_title(rf"Prediction errors of ${var}$" if variable!="pv" else r"Forecast errors of $q_{E}$")
    ax.set_xlabel(rf"$\Delta \hat{{{var}}}_0 - \Delta {{{var}}}_0$")
    ax.set_xlim(lo, hi)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, f"forecast_error_distribution_{variable}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved forecast error distribution plot to {out_path}")

def plot_residuals_with_truth(
    residual_pred: torch.Tensor,
    residual_true: torch.Tensor,
    name: str,
    sample_idx: int = 0,
    show_difference: bool = True,
    dir=Path("./reports/figures/samples"),
    limits=None,
):
    """
    Plot residual errors from forecast and true state, optionally with their difference.

    Parameters
    ----------
    residual_pred : torch.Tensor
        Residual error computed from forecasted changes
    residual_true : torch.Tensor
        Residual error computed from true changes
    name : str
        Short name of the residual (e.g. 'gw', 'qgpv' or 'plan')
    sample_idx : int
        Sample index for file naming
    show_difference : bool
        Whether to plot residual difference (pred - true)
    dir : Path
        Output directory
    """
    res_name, res_number, res_unit = RES_NAMES[name]

    residual_pred = residual_pred.detach().cpu().numpy()
    residual_true = residual_true.detach().cpu().numpy()

    vmin = min(residual_pred.min(), residual_true.min()) if not limits else limits[0]
    vmax = max(residual_pred.max(), residual_true.max()) if not limits else limits[1]
    print(f"{name} Residual colormap limits: {vmin}, {vmax}")

    nrows = 3 if show_difference else 2

    fig = plt.figure(figsize=(5, 2.4 if show_difference else 1.6), constrained_layout=True)

    fig.set_constrained_layout_pads(
        w_pad=0.00,
        h_pad=0.00,
        wspace=0.01,
        hspace=0.00,
    )

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=2,
        width_ratios=[40, 1],
        height_ratios=[1] * nrows,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    if show_difference:
        ax3 = fig.add_subplot(gs[2, 0])
        cax_top = fig.add_subplot(gs[0:2, 1])
        cax_bot = fig.add_subplot(gs[2, 1])
    else:
        cax_top = fig.add_subplot(gs[:, 1])

    # --- Forecast residual ---
    im1 = ax1.imshow(
        residual_pred.T,
        cmap="magma",
        extent=EXTENT_SUBSET,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax1.set_title( rf"{res_name} residual on forecast $\mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\hat{{\mathbf{{x}}}}_0)$")

    # --- True-state residual ---
    im2 = ax2.imshow(
        residual_true.T,
        cmap="magma",
        extent=EXTENT_SUBSET,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax2.set_title(rf"{res_name} residual on true state $\mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\mathbf{{x}}_0)$")

    plt.colorbar(im1, cax=cax_top, label=res_unit)

    # --- Difference (optional) ---
    if show_difference:
        diff = np.abs(residual_pred - residual_true)
        vmin_diff = diff.min() if not limits else limits[2]
        vmax_diff = diff.max() if not limits else limits[3]
        print(f"{name} Residual difference colormap limits: {vmin_diff}, {vmax_diff}")
        im3 = ax3.imshow(
            diff.T,
            cmap="Oranges",
            extent=EXTENT_SUBSET,
            aspect="auto",
            vmin=vmin_diff,
            vmax=vmax_diff
        )
        ax3.set_title(rf"Residual difference $|\mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\hat{{\mathbf{{x}}}}_0) - \mathcal{{R}}_{{\mathrm{{{res_number}}}}}(\mathbf{{x}}_0)|$")
        plt.colorbar(im3, cax=cax_bot, label=res_unit)

        print(
            f"Max abs residual difference ({res_name}, sample {sample_idx}): "
            f"{np.abs(diff).max():.4f}"
        )

    for ax in (ax1, ax2) if not show_difference else (ax1, ax2, ax3):
        ax.set_yticks([])
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[2], EXTENT_SUBSET[3])

    for ax in (ax1) if not show_difference else (ax1, ax2):
        ax.set_xticklabels([])

    suffix = "with_diff" if show_difference else "no_diff"
    out_path = os.path.join(
        dir, f"residual_{res_name.replace(" ", "_")}_{suffix}_{sample_idx}.png"
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved residual plot to {out_path}")

def plot_mse_of_vars(mse_map, out_dir,var):
    plt.figure(figsize=(6,3))
    im = plt.imshow(mse_map.T, extent=EXTENT_SUBSET, origin="lower", cmap="viridis")  # transpose if needed to orient correctly
    plt.title(f"MSE map — {var} (500 hPa)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    out_path = out_dir / f"mse_map_{var}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved MSE map for {var} to {out_path}")

def plot_mse_of_all_vars(mse_maps, out_dir,vars):
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(4.3, 2.7))
    
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        width_ratios=[40, 1],
        height_ratios=[1] * 5,
    )
    for i, var in enumerate(vars):
        var_name = VAR_NAMES.get(var, var)
        var_full = VAR_FULL_NAMES.get(var, var)
        ax = fig.add_subplot(gs[i, 0])

        im = ax.imshow(mse_maps[i].T, extent=EXTENT_SUBSET, origin="lower", cmap="viridis")  # transpose if needed to orient correctly
        ax.set_title(rf"Test MSE of $\small{{{{{var_name}}} , ||\Delta \hat{{{var_name}}}_0 - \Delta {{{var_name}}}_0||^2}}$")
        ax.set_yticklabels([])
        ax.set_yticks([])
        if i<4:
            ax.set_xticklabels([])
            ax.set_xticks([])
        ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
        ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])

    cax = fig.add_subplot(gs[:, 1])
    plt.colorbar(im, cax=cax, label="MSE")
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.94,
        bottom=0.06,
        hspace=0.00,
        wspace=0.02,
    )
        
    out_path = out_dir / f"mse_map_all_variables.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved MSE map for all variables to {out_path}")