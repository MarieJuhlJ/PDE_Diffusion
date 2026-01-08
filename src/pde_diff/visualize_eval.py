import matplotlib.ticker as ticker

from visualize import *

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

    fig = plt.figure(figsize=(10, 3))

    # 3 rows, 2 columns (second column = colorbars)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[40, 1],   # thin colorbars
        height_ratios=[1, 1, 1],
        hspace=0.2,            # vertical spacing
        wspace=0.05,            # gap to colorbar
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
    ax1.set_title("Forecast sample")

    im2 = ax2.imshow(
        target.T,
        cmap=COLOR_BARS.get(variable, "coolwarm"),
        extent=EXTENT_SUBSET,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax2.set_title("Target")

    plt.colorbar(im1, cax=cax_top)

    diff = sample - target
    im3 = ax3.imshow(
        diff.T,
        cmap="PuOr",
        extent=EXTENT_SUBSET,
        aspect="auto",
    )
    ax3.set_title("Difference")
    print(f"Max absolute difference for variable {variable}, sample {sample_idx}: {diff.max():.4f}")

    plt.colorbar(im3, cax=cax_bot)

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

    fig = plt.figure(figsize=(10, 3),constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.02,
        h_pad=0.01,   # ← this is the important one
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

        # Step numbering on the left
        ax_f.set_ylabel(f"Step {i+1}", rotation=0, labelpad=15, va="center")

        # Clean look
        ax_f.set_yticks([])
        ax_t.set_yticks([])
        ax_d.set_yticks([])

        if i < 4:
            ax_f.set_xticklabels([])
            ax_t.set_xticklabels([])
            ax_d.set_xticklabels([])

        axes_forecast.append(ax_f)
        axes_target.append(ax_t)
        axes_diff.append(ax_d)

    axes_forecast[0].set_title(f"Forecasts of {VAR_NAMES.get(variable, variable)}")
    axes_target[0].set_title(f"True states of {VAR_NAMES.get(variable, variable)}")
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

    out_path = os.path.join(dir, f"forecasts_vs_targets_w_diff_sample_{sample_idx}_{variable}.png")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved forecast vs target plot to {out_path}")
