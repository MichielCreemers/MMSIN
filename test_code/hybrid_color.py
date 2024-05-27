import open3d as o3d
import numpy as np
from scipy.stats import skew
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_point_cloud(file_path):
    """Just a simple function to load a point cloud from a file.

    Args:
        file_path (String): full path to the file
    """
    return o3d.io.read_point_cloud(file_path)

def compute_color_histograms(colors, bins=256):
    """Compute the color histograms for the RGB channels of a point cloud.

    Args:
        colors (_type_): _description_
        bins (int, optional): _description_. Defaults to 256.
    """
    hist = [np.histogram(colors[:, i], bins=bins, range=(0, 1))[0] for i in range(3)]
    return hist

def compute_color_moments(colors):
    """Computes the color moments: mean, standard deviation and skewness for the RGB channels
       The function returns them as a list of 3 tuples, one for each channel containing the mean, std and skewness.
    Args:
        colors (_type_): _description_
    """
    moments = [(colors[:, i].mean(), colors[:, i].std(), skew(colors[:, i])) for i in range(3)]
    return moments

def check_color_range(file_path):
    # Load the point cloud
    pc = o3d.io.read_point_cloud(file_path)
    
    # Check if point cloud has colors
    if not pc.has_colors():
        raise ValueError("Point cloud has no color information.")
    
    # Get colors and check their range
    colors = np.asarray(pc.colors)
    
    # If colors are floats, they are likely in the range [0, 1]
    if np.issubdtype(colors.dtype, np.floating):
        color_range = (np.min(colors), np.max(colors))
        if color_range[0] < 0.0 or color_range[1] > 1.0:
            raise ValueError("Color values are outside the expected [0, 1] range for floats.")
        return "floats in range [0, 1]"
    
    # If colors are integers, they are likely in the range [0, 255]
    elif np.issubdtype(colors.dtype, np.integer):
        color_range = (np.min(colors), np.max(colors))
        if color_range[0] < 0 or color_range[1] > 255:
            raise ValueError("Color values are outside the expected [0, 255] range for integers.")
        return "integers in range [0, 255]"


# Example usage
file_path = 'D:\point clouds\Rafa2_ply_TMC2-randomAc_r1\Rafa2_ply_400K_ctc_r1_006.ply'
print(check_color_range(file_path))

# pc = load_point_cloud(file_path)
# colors = np.asarray(pc.colors) # Assuming colors are normalized [0, 1]

# histograms = compute_color_histograms(colors)
# moments = compute_color_moments(colors)

# print('Histograms:', histograms)
# print('Moments:', moments)
