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

ERA5_MEANS = {'v':  np.array([-0.0801555 , -0.10157621], dtype=np.float32),
              'u': np.array([10.284327, 11.377211], dtype=np.float32),
              'vo': np.array([3.7109719e-06, 4.5307206e-06], dtype=np.float32)}


ERA5_STD = {'v': np.array([ 9.1059265, 10.151677 ], dtype=np.float32),
            'u': np.array([ 9.811623, 10.794332], dtype=np.float32),
            'vo': np.array([5.4005417e-05, 5.8901383e-05], dtype=np.float32)}


ERA5_DIFF_MEAN = {'v': np.array([-0.0007102 ,  0.00128436], dtype=np.float32),
                  'u': np.array([-0.03137042, -0.02695089], dtype=np.float32),
                  'vo': np.array([1.7552045e-08, 9.6490140e-09], dtype=np.float32)}


ERA5_DIFF_STD = {'v': np.array([1.3959143, 1.53918  ], dtype=np.float32),
                 'u': np.array([1.1522577, 1.2496729], dtype=np.float32),
                 'vo': np.array([2.6169731e-05, 2.8045424e-05], dtype=np.float32)}
