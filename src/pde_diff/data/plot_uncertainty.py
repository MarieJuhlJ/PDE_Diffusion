import os
import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# Path to your TTF file
font_path = "./times.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'

plt.style.use("pde_diff.custom_style")

def download_era5_data(months: list[str]=["09"], days: list[str]=[str(i).zfill(2) for i in range(1,32)], test_mode: bool=False, full: bool=False):
    dataset = "reanalysis-era5-pressure-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_ensemble_members"],
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "potential_vorticity",
            "temperature",
            "geopotential"
        ],
        "year": ["2024"],
        "month": months,
        "time": [
            "00:00"
        ],
        "pressure_level": ["450", "500","550"],
        "data_format": "grib",
        #"download_format": "unarchived"
    }
    if not full:
        request["area"]=[70, -180, 46, 180]  # North, West, South, East

    grib_file = "./data/era5/era5.grib"
    os.makedirs(os.path.dirname(grib_file), exist_ok=True)

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(grib_file)
    print(f"Downloaded ERA5 data to {grib_file}")

    ds = xr.open_dataset(grib_file, engine='cfgrib')
    save_path = './data/era5/zarr_ensemble'
    if test_mode:
        save_path += "_test"
    if full:
        save_path += "_full"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ds.to_zarr(save_path,mode="w")
    print(f"Converted GRIB file to Zarr format at {save_path}")

    #delete the grib file to save space
    os.remove(grib_file)
    print(f"Deleted temporary GRIB file {grib_file}")

EXTENT_SUBSET = [0.0, 359.25, 69.75, 46.5]
VAR_NAMES = {
    "u": "u",
    "v": "v",
    "t": "T",
    "z": r"$\Phi$",
    "pv": r"$q_E$",
}

def visualize_era5_sample(data_sample, variable, level=500, big_data_sample=None, sample_idx=None, dir=Path("./reports/figures/samples"), color_bar_limit=None):
    """
    Visualize a sample from the ERA5 dataset.

    Args:
        data_sample (np.ndarray): The data sample to visualize.
        variable (str): Variable to visualize (e.g., 'temperature').
        level (int): Pressure level to focus on (default: 500hPa).
        sample_idx (int, optional): Index of the sample (for title purposes).
        color_bar_limit (tuple, optional): Tuple of (vmin, vmax) for color bar limits.
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    # Plot the data
    fig, ax = plt.subplots(figsize=(5,3))
    ax.imshow(data_sample,extent=[0, 360, 46, 70],origin='lower')
    fig.colorbar(ax.images[-1], ax=ax, shrink=0.4, location="bottom") #, label=f"{VAR_UNITS.get(variable, variable)}")
    #ax.set_xlim(EXTENT_SUBSET[0], EXTENT_SUBSET[1])
    #ax.set_ylim(EXTENT_SUBSET[3], EXTENT_SUBSET[2])
    if variable in ["u", "v"]:
        ax.set_title(f"Standard Deviation for {VAR_NAMES.get(variable, variable)} at {level} hPa")
    else:
        ax.set_title(f"Monthly Coefficient of Variation for {VAR_NAMES.get(variable, variable)} at {level} hPa")
    plt.tight_layout()
    plot_path = f"era5_sample{sample_idx if sample_idx is not None else ''}_{'_'.join(variable.split(' '))}_{level}hPa"
    plot_path += "_full" if big_data_sample is not None else ""
    plot_path += ".png"
    plot_path = os.path.join(dir, plot_path)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")

def plot_uncertainty_on_all_vars(uncertain, out_dir,vars):
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(4, 2.4),constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.02,
        h_pad=0.01,
        wspace=0.02,
        hspace=0.02,
    )
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        width_ratios=[40, 1],
        height_ratios=[1] * 5,
    )
    for i, var in enumerate(vars):
        var_name = VAR_NAMES.get(var, var)
        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(uncertain[i], extent=[0, 360, 46, 70], origin="lower", cmap="viridis")  # transpose if needed to orient correctly
        if var in ["u", "v"]:
            ax.set_title(f"Monthly Standard Deviation for {var_name}")
        else:
            ax.set_title(f"Monthly Coefficient of Variation for {var_name}")
        ax.set_yticklabels([])
        ax.set_yticks([])
        if i<4:
            ax.set_xticklabels([])
            ax.set_xticks([])

        cax = fig.add_subplot(gs[i, 1])
        if var in ["u", "v"]:
            plt.colorbar(im, cax=cax, label="std.", orientation="vertical")
        else:
            plt.colorbar(im, cax=cax, label="\%", orientation="vertical")

    out_path = out_dir / f"uncertainty_all_variables.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved uncertainty map for all variables to {out_path}")

if __name__ == "__main__":
    #download_era5_data(test_mode=False)
    path="./data/era5/zarr_ensemble"
    data = xr.open_zarr(path, chunks={})
    sample = data.isel(isobaricInhPa=1,number=slice(0,10))[["u", "v", "pv", "t", "z"]]  # Select 500 hPa level, first time, first 10 ensemble members
    sample = sample.to_array().values  # Convert to numpy array
    print(sample.shape)  # Should be (5, 10, lon, lat)

    dir="./reports/figures/era5/uncertainty_sample_plots"
    uncertains = []
    vars = ["u", "v", "pv", "t", "z"]
    for i, var in enumerate(vars):
        mean_var = sample[i,:, :, :].mean(axis=0)
        std_var =  sample[i,:, :, :].std(axis=0)
        coef_of_variation = (std_var/mean_var)*100
        uncertains.append(std_var if var in ["u", "v"] else coef_of_variation)
        visualize_era5_sample(std_var if var in ["u", "v"] else coef_of_variation, variable=var, level=500, sample_idx=0, dir=dir)
    u = sample[0]
    v = sample[1]

    speed = np.sqrt(u**2 + v**2)
    speed_mean = speed.mean(axis=0)
    speed_std = speed.std(axis=0)

    cv_speed = speed_std / speed_mean * 100
    visualize_era5_sample(cv_speed, variable="wind speed", level=500, sample_idx=0, dir=dir)
    plot_uncertainty_on_all_vars(uncertain=uncertains,out_dir=Path(dir), vars=vars )