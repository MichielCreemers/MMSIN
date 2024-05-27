import pandas as pd
import csv

def print_info(features):
    """Print feature information for multiple patches.

    Args:
        features (np.ndarray): An array of shape (N, 64) representing features for N patches.

    Returns:
        int: Always returns 1 (indicating success).
    """
    # Define the structure of the feature names based on the previous code
    feature_names = []
    for feature_domain in ["l", "a", "b"]:
        for param in ["mean", "std", "entropy"]:
            feature_names.append(f"{feature_domain}_{param}")

    for feature_domain in ['curvature', 'anisotropy', 'linearity', 'planarity', 'sphericity']:
        for param in ["mean", "std", "entropy", "ggd1", "ggd2", "aggd1", "aggd2", "aggd3", "aggd4", "gamma1", "gamma2"]:
            feature_names.append(f"{feature_domain}_{param}")

    # Loop over each patch
    for i, patch_features in enumerate(features):
        print(f"Patch {i+1} features:")
        for cnt, name in enumerate(feature_names):
            print(f"{name}: {patch_features[cnt]}")
        # print()  

    return 1

def create_patched_features_df(original_features, num_patches):
    patched_feature_names = ['name']
    
    for i in range(1, num_patches + 1):
        for feature in original_features[1:]: # skip 'name' feature
            patched_feature_names.append(f"{feature}_patch_{i}")
    
    df_patched_features = pd.DataFrame(columns=patched_feature_names)
    return df_patched_features

