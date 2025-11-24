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
    'v':  np.array([-0.0801555 , -0.10157621, -0.13813713], dtype=np.float32),
    'u':  np.array([10.284327, 11.377211, 12.609484], dtype=np.float32),
    'pv': np.array([6.3948687e-07, 6.3594905e-07, 6.3737781e-07], dtype=np.float32),
    't':  np.array([255.84659, 251.32251, 246.14778], dtype=np.float32),
    'z':  np.array([47262.523, 54204.25 , 61730.36 ], dtype=np.float32)
}

ERA5_STD = {
    'v':  np.array([ 9.1059265, 10.151677 , 11.447809 ], dtype=np.float32),
    'u':  np.array([ 9.811623, 10.794332, 12.126493], dtype=np.float32),
    'pv': np.array([3.8189577e-07, 4.0063446e-07, 4.8568654e-07], dtype=np.float32),
    't':  np.array([6.594297 , 6.6641073, 6.6774826], dtype=np.float32),
    'z':  np.array([1397.7808, 1548.3698, 1721.3713], dtype=np.float32)
}

ERA5_DIFF_MEAN = {
    'v':  np.array([-0.0007102 ,  0.00128436, -0.00047651], dtype=np.float32),
    'u':  np.array([-0.03137042, -0.02695089, -0.02360447], dtype=np.float32),
    'pv': np.array([1.3307146e-09, 1.7303672e-09, 1.5463171e-09], dtype=np.float32),
    't':  np.array([0.01117611, 0.01584899, 0.01999644], dtype=np.float32),
    'z':  np.array([-1.2831955 , -0.90843827, -0.355297  ], dtype=np.float32)
}

ERA5_DIFF_STD = {
    'v':  np.array([1.3959143, 1.53918  , 1.7233391], dtype=np.float32),
    'u':  np.array([1.1522577, 1.2496729, 1.412487 ], dtype=np.float32),
    'pv': np.array([1.7968893e-07, 1.9385455e-07, 2.2434311e-07], dtype=np.float32),
    't':  np.array([0.4210719 , 0.43234292, 0.4478308 ], dtype=np.float32),
    'z':  np.array([49.822533, 54.978558, 61.497757], dtype=np.float32)
}
