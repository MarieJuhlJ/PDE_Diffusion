"""
Constants for use in normalizing data, etc.

where the variance is the variance in the 1 hour change for a variable averaged across all lat/lon
 and pressure levels
and time for (~100 random temporal frames, more the better)

Min/Max/Mean/Stddev for all those plus each type of observation in observation files

"""

import numpy as np

# Means and stds computed from gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr.
# The atmospheric variables are numpy arrays at 13 vertical pressure levels
# corresponding to the levels of the WeatherBench (Rasp et al., 2020) benchmark:
# 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, and 1000 hPa.
# ERA5_DIFF_MEAN and ERA5_DIFF_STD are the means and stds of 12h timesteps differences.

ERA5_MEANS = {
    "v":  np.array([-0.05439724, -0.08053342, -0.11585448], dtype=np.float32),
    "u":  np.array([ 7.827743,    8.673528,    9.614547  ], dtype=np.float32),
    "pv": np.array([ 6.6577229e-07, 6.5286338e-07, 6.3751554e-07], dtype=np.float32),
    "t":  np.array([258.6171,    254.0188,    248.74092 ], dtype=np.float32),
    "z":  np.array([47717.152,   54734.59,    62341.297 ], dtype=np.float32),
}

ERA5_STD = {
    "v":  np.array([ 9.286872,  10.145261,  11.228227], dtype=np.float32),
    "u":  np.array([ 9.454823,  10.330215,  11.460713], dtype=np.float32),
    "pv": np.array([ 3.5212761e-07, 3.6951292e-07, 4.2559611e-07], dtype=np.float32),
    "t":  np.array([ 5.674391,   5.7301693,  5.7954497], dtype=np.float32),
    "z":  np.array([1414.9316,  1542.6438,  1688.6382], dtype=np.float32),
}

ERA5_DIFF_MEAN = {
    "v":  np.array([ 8.0633363e-05,  4.3448286e-05, -2.4536726e-04], dtype=np.float32),
    "u":  np.array([ 2.95522e-03,     3.39574e-03,     4.08541e-03   ], dtype=np.float32),
    "pv": np.array([-5.3399271e-11,  -4.5271079e-12,   9.5704816e-12], dtype=np.float32),
    "t":  np.array([-7.26565e-03,    -7.09030e-03,    -6.85968e-03  ], dtype=np.float32),
    "z":  np.array([-1.1826048,      -1.381127,       -1.5934479    ], dtype=np.float32),
}

ERA5_DIFF_STD = {
    "v":  np.array([1.192583,   1.3096325, 1.4798834], dtype=np.float32),
    "u":  np.array([1.0820649,  1.1679299, 1.2982965], dtype=np.float32),
    "pv": np.array([1.7148871e-07, 1.8222345e-07, 2.0708451e-07], dtype=np.float32),
    "t":  np.array([0.34212112, 0.35110953, 0.36627045], dtype=np.float32),
    "z":  np.array([42.957813,  46.766777,  51.760887], dtype=np.float32),
}