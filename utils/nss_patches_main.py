import get_patches as gp
import nss_patches.get_nss_features as nf
import nss_patches.nss_misc as misc
import numpy as np
import pandas as pd
import time
import glob
import joblib
import csv
from sklearn.preprocessing import MinMaxScaler


num_patches = 6
michiels_constant = 2048

# Get all the point clouds in the folder
dataset = "WPC"
point_cloud_folder = dataset+"/point_clouds"

ply_pattern = f"{point_cloud_folder}/**/*.ply"
point_clouds = glob.glob(ply_pattern, recursive=True)

feature_names = ["name","l_mean","l_std","l_entropy","a_mean","a_std","a_entropy","b_mean","b_std","b_entropy","curvature_mean","curvature_std","curvature_entropy","curvature_ggd1","curvature_ggd2","curvature_aggd1","curvature_aggd2","curvature_aggd3","curvature_aggd4","curvature_gamma1","curvature_gamma2","anisotropy_mean","anisotropy_std","anisotropy_entropy","anisotropy_ggd1","anisotropy_ggd2","anisotropy_aggd1","anisotropy_aggd2","anisotropy_aggd3","anisotropy_aggd4","anisotropy_gamma1","anisotropy_gamma2","linearity_mean","linearity_std","linearity_entropy","linearity_ggd1","linearity_ggd2","linearity_aggd1","linearity_aggd2","linearity_aggd3","linearity_aggd4","linearity_gamma1","linearity_gamma2","planarity_mean","planarity_std","planarity_entropy","planarity_ggd1","planarity_ggd2","planarity_aggd1","planarity_aggd2","planarity_aggd3","planarity_aggd4","planarity_gamma1","planarity_gamma2","sphericity_mean","sphericity_std","sphericity_entropy","sphericity_ggd1","sphericity_ggd2","sphericity_aggd1","sphericity_aggd2","sphericity_aggd3","sphericity_aggd4","sphericity_gamma1","sphericity_gamma2"]
df_features = misc.create_patched_features_df(feature_names, 6)

start = time.time()
i = 0
start_time = time.time()
average_time = 0
for pc in point_clouds:
    i+=1
    print(i)
    print(pc)
    # Patch up point cloud
    patches, indices = gp.knn_patch(pc, michiels_constant, num_patches)
    # print("Shape of indices: ", indices.shape)
    
    # Calculate features for N patches --> features[N, 64]
    features = nf.process_point_cloud_with_patches(pc, indices)
    features = features.flatten()
    features = features.tolist()
    features.insert(0, pc)
    # print(features)
    # print("shape of feautures: ", features.shape)
    #gp.visualize_patches_with_base(points, patches, indices)
    
    features_df = pd.DataFrame([features], columns=df_features.columns)
    df_features = pd.concat([df_features, features_df], ignore_index=True)
    
end_time = time.time() - start_time
print("final time is " + str(end_time))
    
# Alfabetize
print(df_features)
# print("df_features shape: ", df_features.shape)
# df_features.to_csv("test.csv", index= False)
sorted_df = df_features.sort_values(by="name", ascending=True)
# sorted_df.to_csv("test2.csv", index= False)
# print("sorted_shape: ", sorted_df.shape)

# Min-Max scaling
scaler = MinMaxScaler()
df_to_scale = sorted_df.copy()
feature_columns = sorted_df.columns[1:] # not for name column
# print(feature_columns)
df_to_scale[feature_columns] = scaler.fit_transform(df_to_scale[feature_columns])
joblib.dump(scaler, 'sc.joblib') 

normalized_dataframe = df_to_scale

# Rename
normalized_dataframe['name'] = normalized_dataframe['name'].str.rsplit('/', n=1).str[-1]
normalized_dataframe.to_csv(dataset + "/"  + dataset + "_NSS_patches.csv", index=False)    
# normalized_dataframe.to_csv("koekjes.csv", index=False)  