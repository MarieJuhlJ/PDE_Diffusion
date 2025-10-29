import xarray as xr

d = xr.open_dataset('231d71c5ceef7895e6415951cd4328b6.grib', engine = 'cfgrib')
print(d)
