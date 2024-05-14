import utils.NSS.feature_extract as fe
import utils.NSS.feature_functions as ff
 
# import feature_extract as fe
# import feature_functions as ff

import time
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import glob
import joblib

dataset = "WPC2"
point_cloud_folder = dataset+"/point_clouds"


ply_pattern = f"{point_cloud_folder}/**/*.ply"

point_clouds = glob.glob(ply_pattern, recursive=True)

point_clouds = point_clouds[:100]


df_timings = pd.DataFrame(columns=['time'])

for pc in point_clouds:
    start_time = time.time()
    features = fe.get_feature_vector(pc)
    end_time = time.time()
    duration = end_time-start_time
    df_timings.loc[len(df_timings)] = duration

mean  = df_timings['time'].mean()
stdev = df_timings['time'].std()
min_time = df_timings['time'].min()
max_time = df_timings['time'].max()

print(f"Average time: {mean}")
print(f"Standard deviation: {stdev}")
print(f"Minimum time: {min_time}")
print(f"Maximum time: {max_time}")