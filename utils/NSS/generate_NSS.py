import feature_extract as fe
import feature_functions as ff
 
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

# get all point clouds in the folder
dataset = "WPC2"
point_cloud_folder = dataset+"/point_clouds"


ply_pattern = f"{point_cloud_folder}/**/*.ply"

point_clouds = glob.glob(ply_pattern, recursive=True)

feature_names = ["name","l_mean","l_std","l_entropy","a_mean","a_std","a_entropy","b_mean","b_std","b_entropy","curvature_mean","curvature_std","curvature_entropy","curvature_ggd1","curvature_ggd2","curvature_aggd1","curvature_aggd2","curvature_aggd3","curvature_aggd4","curvature_gamma1","curvature_gamma2","anisotropy_mean","anisotropy_std","anisotropy_entropy","anisotropy_ggd1","anisotropy_ggd2","anisotropy_aggd1","anisotropy_aggd2","anisotropy_aggd3","anisotropy_aggd4","anisotropy_gamma1","anisotropy_gamma2","linearity_mean","linearity_std","linearity_entropy","linearity_ggd1","linearity_ggd2","linearity_aggd1","linearity_aggd2","linearity_aggd3","linearity_aggd4","linearity_gamma1","linearity_gamma2","planarity_mean","planarity_std","planarity_entropy","planarity_ggd1","planarity_ggd2","planarity_aggd1","planarity_aggd2","planarity_aggd3","planarity_aggd4","planarity_gamma1","planarity_gamma2","sphericity_mean","sphericity_std","sphericity_entropy","sphericity_ggd1","sphericity_ggd2","sphericity_aggd1","sphericity_aggd2","sphericity_aggd3","sphericity_aggd4","sphericity_gamma1","sphericity_gamma2"]
df_features = pd.DataFrame(columns=feature_names)

start = time.time()
i = 0
start_time = time.time()
average_time = 0
for pc in point_clouds:
    
    features = fe.get_feature_vector(pc)
    ff.print_info(features)
    features.insert(0, pc)

    print(features)

    features_df = pd.DataFrame([features], columns=df_features.columns)
    df_features = pd.concat([df_features, features_df], ignore_index=True)


end_time = time.time()-start_time
print("final time is"+ str(end_time))


#alfabetize
sorted_df = df_features.iloc[1:].sort_values(by="name", key=lambda col: col.str.lower(), ascending = True)

#min-max_scaling

scaler = MinMaxScaler()
df_to_scale = sorted_df.copy()

feature_columns = sorted_df.columns[1:]  # All columns except the first one ("name")
df_to_scale[feature_columns] = scaler.fit_transform(df_to_scale[feature_columns])

min_values = scaler.min_
scale = scaler.scale_

scaler_params = np.array([min_values, scale])

np.save('WPC2/scaler_params.npy', scaler_params)

normalized_dataframe = df_to_scale

#rename
normalized_dataframe['name'] = normalized_dataframe['name'].str.rsplit('/', n=1).str[-1]
sorted_df['name'] = sorted_df['name'].str.rsplit('/',n=1).str[-1]

normalized_dataframe.to_csv(dataset + "/"  + dataset + "_NSS.csv", index=False)
sorted_df.to_csv(dataset + "/" + dataset + "_NSS_non_scaled.csv", index=False)
# print(all_names)