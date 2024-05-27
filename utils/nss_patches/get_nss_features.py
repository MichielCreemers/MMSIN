from pyntcloud import PyntCloud
import numpy as np
from skimage import color
import pandas as pd
from nss_patches.nss_functions import *

def normalize_pointcloud(points_df):
    """_summary_

    Args:
        points_df (_type_): _description_
    """
    centroid = points_df[['x', 'y', 'z']].mean()
    scale = (points_df[['x', 'y', 'z']] - centroid).abs().max()
    points_df[['x', 'y', 'z']] = (points_df[['x', 'y', 'z']] - centroid) / scale
    return points_df

def get_feature_vector_for_patch(patch_df):
    """Extract features from a single patch of a point cloud.

    Args:
        patch_df (_type_): _description_
    """
    
    cloud = PyntCloud(patch_df)
    k_neighbors = cloud.get_neighbors(k=10)
    ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    cloud.add_scalar_field("curvature", ev=ev)
    cloud.add_scalar_field("anisotropy", ev=ev)
    cloud.add_scalar_field("linearity", ev=ev)
    cloud.add_scalar_field("planarity", ev=ev)
    cloud.add_scalar_field("sphericity", ev=ev)

    curvature = cloud.points['curvature(11)'].to_numpy()
    anisotropy = cloud.points['anisotropy(11)'].to_numpy()
    linearity = cloud.points['linearity(11)'].to_numpy()
    planarity = cloud.points['planarity(11)'].to_numpy()
    sphericity = cloud.points['sphericity(11)'].to_numpy()
    
    rgb_color = cloud.points[['red', 'green', 'blue']].to_numpy() / 255
    lab_color = color.rgb2lab(rgb_color)
    l, a, b = lab_color[:, 0], lab_color[:, 1], lab_color[:, 2]
    
    nss_params = []
    # for tmp in [l, a, b, curvature, anisotropy, linearity, planarity, sphericity]:
    #     tmp_np = np.array(tmp)
    #     params = get_geometry_nss_param(tmp_np) if tmp is curvature else get_color_nss_param(tmp_np)
    #     nss_params.extend([item for sublist in params for item in sublist])
    
    for tmp in [l,a,b]:
      tmp_cupy = np.array(tmp)
      params = get_color_nss_param(tmp_cupy)
      #flatten the feature vector
      nss_params = nss_params + [i for item in params for i in item]
    # compute geomerty nss features
    for tmp in [curvature,anisotropy,linearity,planarity,sphericity]:
        tmp_cupy = np.array(tmp)
        params = get_geometry_nss_param(tmp_cupy)
        #flatten the feature vector
        nss_params = nss_params + [i for item in params for i in item]
        
    return nss_params

    
def process_point_cloud_with_patches(pc_path, patch_indices):
    """_summary_

    Args:
        pc_path (_type_): _description_
        patch_indices (_type_): _description_
    """
    original = PyntCloud.from_file(pc_path)
    original_points_df = original.points
    original_points_df = normalize_pointcloud(original_points_df)  # Normalize the entire point cloud
    
    # Process each patch in parallel
    patches = [original_points_df.iloc[idx].copy() for idx in patch_indices]
    features = process_patches_in_parallel(patches)
    return np.asarray(features)
    
def process_patches_in_parallel(patches):
    """Process patches in parallel to extract features faster"""
    results = [get_feature_vector_for_patch(patch) for patch in patches]
    return results


        
        
    
    
    
    
