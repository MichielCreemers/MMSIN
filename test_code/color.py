import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def load_point_cloud(file_path):
    """Just a simple function to load a point cloud from a file.

    Args:
        file_path (String): full path to the file
    """
    return o3d.io.read_point_cloud(file_path)

def compute_color_histogram(point_cloud, bins=256, show_histogram=False):
    """Computes the color histogram of a point cloud.

    Args:
        point_cloud (_type_): _description_
        bins (int, optional): _description_. Defaults to 256.
        show_histogram (bool, optional): _description_. Defaults to False.
    """
    colors = np.asarray(point_cloud.colors)
    
    # Compute the histogram for each color channel
    red_histogram, _ = np.histogram(colors[:, 0], bins=bins, range=(0, 1))
    green_histogram, _ = np.histogram(colors[:, 1], bins=bins, range=(0, 1))
    blue_histogram, _ = np.histogram(colors[:, 2], bins=bins, range=(0, 1))
    
    # Show the histograms
    if show_histogram:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.bar(np.arange(bins), red_histogram, color='r')
        plt.title('Red channel')
        plt.subplot(1, 3, 2)
        plt.bar(np.arange(bins), green_histogram, color='g')
        plt.title('Green channel')
        plt.subplot(1, 3, 3)
        plt.bar(np.arange(bins), blue_histogram, color='b')
        plt.title('Blue channel')
        plt.show()
    print('Red:', red_histogram)  
    return red_histogram, green_histogram, blue_histogram

# Example usage
file_path = 'D:\point clouds\Rafa2_ply_TMC2-randomAc_r1\Rafa2_ply_400K_ctc_r1_006.ply'
point_cloud = load_point_cloud(file_path)
red_histogram, green_histogram, blue_histogram = compute_color_histogram(point_cloud, show_histogram=True)

file_path = 'D:\point clouds\Rafa2_ply_TMC2-randomAc_r5\Rafa2_ply_400K_ctc_r5_006.ply'
point_cloud = load_point_cloud(file_path)
red_histogram, green_histogram, blue_histogram = compute_color_histogram(point_cloud, show_histogram=True)