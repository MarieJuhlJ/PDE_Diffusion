"""To use the api one has to
1. Have an account and log in
2. Go to the data page and accept Terms of Use: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
3. Copy the api url and key from this page: https://cds.climate.copernicus.eu/how-to-api
4. Create a file called ".cdsapirc" containing the url and key in ones home/user folder. The file should look like this:
"""
import os
import cdsapi
import xarray as xr

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "u_component_of_wind",
        "v_component_of_wind",
        "vorticity"
    ],
    "year": ["2024"],
    "month": ["09"],
    "day": ["29", "30"],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "pressure_level": ["500","550"],
    "data_format": "grib",
    #"download_format": "unarchived"
}

grib_file = "./data/era5/era5.grib"
os.makedirs(os.path.dirname(grib_file), exist_ok=True)

client = cdsapi.Client()
client.retrieve(dataset, request).download(grib_file)
print(f"Downloaded ERA5 data to {grib_file}")

ds = xr.open_dataset(grib_file, engine='cfgrib')
ds.to_zarr('./data/era5/zarr',mode="w")
print("Converted GRIB file to Zarr format at data/era5/zarr")
