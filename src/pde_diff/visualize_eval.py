import matplotlib.ticker as ticker

from pde_diff.visualize import *

RES_NAMES = {
    "qgpv": ("Quasi-geostrophic Potential Vorticity", 1, r"$s^{-1}$"),
    "plan": ("Planetary Vorticity",2, r"$s^{-2}$"),
    "gw": ("Geostrophic Wind", 3,r"$m \cdot s^{-1}$")
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

    im1 = ax1.imshow(
        sample.T,
        cmap=COLOR_BARS.get(variable, "coolwarm"),
        extent=EXTENT_SUBSET,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax1.set_title(f"Prediction of {VAR_NAMES.get(variable, variable)}")
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
    ax2.set_title(f"True state of {VAR_NAMES.get(variable, variable)}")
    ax2.set_xticklabels([])
    ax2.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax2.set_ylim(EXTENT_SUBSET[2],EXTENT_SUBSET[3])

    plt.colorbar(im1, cax=cax_top,label=f"{VAR_UNITS.get(variable, variable)}")

    diff = target - sample
    im3 = ax3.imshow(
        diff.T,
        cmap="PuOr",
        extent=EXTENT_SUBSET,
        aspect="auto",
    )
    ax3.set_title(f"Difference of {VAR_NAMES.get(variable, variable)} " +r"$(x_0-\hat{x}_0)$")
    ax3.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    ax3.set_ylim(EXTENT_SUBSET[2],EXTENT_SUBSET[3])
    print(f"Max absolute difference for variable {variable}, sample {sample_idx}: {diff.max():.4f}")

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
    states=True
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
    differences = [t-f for f, t in zip(forecasts, targets)]

    # Shared color scale
    vmin = min(f.min() for f in forecasts + targets)
    vmax = max(f.max() for f in forecasts + targets)
    vmin_d = min(f.min() for f in differences)
    vmax_d = max(f.max() for f in differences)

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


    for i in range(5):
        ax_f = fig.add_subplot(gs[i, 0])
        ax_t = fig.add_subplot(gs[i, 1])
        ax_d = fig.add_subplot(gs[i, 3])

        im = ax_f.imshow(forecasts[i].T,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto",extent=EXTENT_SUBSET,)
        ax_t.imshow(targets[i].T, cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto",extent=EXTENT_SUBSET,)
        im_diff = ax_d.imshow(differences[i].T,vmin=vmin_d,vmax=vmax_d, cmap="PuOr",aspect="auto",extent=EXTENT_SUBSET,)

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

    axes_forecast[0].set_title(f"Forecasts of {VAR_NAMES.get(variable, variable)}")
    if states:
        axes_target[0].set_title(f"True states of {VAR_NAMES.get(variable, variable)}")
    else:
        axes_target[0].set_title(f"True changes of {VAR_NAMES.get(variable, variable)}")

    axes_diff[0].set_title(r"Difference $(x_0-\hat{x}_0)$")

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

def plot_residuals_with_truth(
    residual_pred: torch.Tensor,
    residual_true: torch.Tensor,
    name: str,
    sample_idx: int = 0,
    show_difference: bool = True,
    dir=Path("./reports/figures/samples"),
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

    vmin = min(residual_pred.min(), residual_true.min())
    vmax = max(residual_pred.max(), residual_true.max())

    nrows = 3 if show_difference else 2

    fig = plt.figure(figsize=(8, 2.4 if show_difference else 1.6), constrained_layout=True)

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
        im3 = ax3.imshow(
            diff.T,
            cmap="Oranges",
            extent=EXTENT_SUBSET,
            aspect="auto",
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
    im = plt.imshow(mse_map.T, origin="lower", cmap="viridis")  # transpose if needed to orient correctly
    plt.title(f"MSE map — {var} (500 hPa)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    out_path = out_dir / f"mse_map_{var}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved MSE map for {var} to {out_path}")
