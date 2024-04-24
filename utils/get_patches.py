import open3d as o3d
import numpy as np
import os, time, random
import heapq



def normalize_pointcloud(pc):
    """Normalize the pointcloud to the origin and scale to fit in a unit sphere.

    Args:
        pc(np.ndarray): The N x 3 numpy array of the point cloud
    
    Returns:
        np.ndarray: Normalized point cloud
    """
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    max_distance = np.max(np.linalg.norm(pc, axis=1))
    pc /= max_distance
    return pc


def farthest_point_sample(pc_data, num_samples):
    """Sample points that are farthest from each other in the point cloud.

    Args:
        pc_data (np.ndarray): The N x 3 numpy array of point cloud data
        num_samples (int): Number of samples to select
    
    Returns:
        np.ndarray: Array of sampled point indices
        np.ndarray: Indices of the sampled points in the original point cloud
    """
    N, D = pc_data.shape
    centroids = np.zeros(num_samples, dtype=int)
    sampled_indices = np.zeros(num_samples, dtype=int)
    distance = np.ones(N) * np.inf
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        sampled_indices[i] = farthest
        centroid_point = pc_data[farthest, :3]
        dist = np.linalg.norm(pc_data[:, :3] - centroid_point, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return pc_data[centroids], sampled_indices


def knn_patch(pc_path, patch_size=2048, num_patches=6):
    """Generate patches from a point cloud using KNN algorithm

    Args:
        pc_path (str): Path to the point cloud
        patch_size (int, optional): The number of points each patch should include. Defaults to 2048.
        num_patches (int, optional): The number of patches the point cloud should be divides into. Defaults to 6.
    """
    # Load and normalize the point cloud
    pcd = o3d.io.read_point_cloud(pc_path)
    points = normalize_pointcloud(np.array(pcd.points))
    
    # Prepare KD-Tree 
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Sample center points using farthest point sampling
    _, center_indices = farthest_point_sample(points, num_patches)
    
    N, _ = points.shape
    dynamic_patch_size = (max(patch_size, N // num_patches)) // 2
    if dynamic_patch_size > 50000:
        dynamic_patch_size = 50000
    # print("Total number of points: ", N)
    # print("Number of patches: ", num_patches)
    # print("Dynamic patch size for each patch: ", dynamic_patch_size)
    
    # Collect patches centered around each sampled point
    patch_list = []
    indices_list = []
    
    for center_idx in center_indices:
        _, idx, _ = kdtree.search_knn_vector_3d(points[center_idx], dynamic_patch_size)
        patch_list.append(np.asarray(pcd.points)[idx, :])
        indices_list.append(idx)
        
    return np.asarray(patch_list), np.asarray(indices_list)



def visualize_with_highlights(points, highlighted_indices):
    """Visualize point cloud with specified points highlighted in red.

    Args:
        points (np.ndarray): Nx3 array containing the point cloud coordinates.
        highlighted_indices (list or np.ndarray): Indices of points to highlight in red.
    """
    # Validate highlighted indices
    if not isinstance(highlighted_indices, (list, np.ndarray)):
        raise TypeError("highlighted_indices must be a list or numpy array of integers.")
    
    if isinstance(highlighted_indices, np.ndarray) and highlighted_indices.dtype.kind not in {'i', 'u'}:
        raise TypeError("highlighted_indices array must contain integers.")

    # Check contents of highlighted_indices
    print("Highlighted Indices:", highlighted_indices)

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color all points grey
    num_points = len(points)
    colors = np.tile(np.array([0.5, 0.5, 0.5]), (num_points, 1))  # Grey color

    # Highlight selected points in red
    colors[highlighted_indices] = [1, 0, 0]  # Red color

    # Assign colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    
def visualize_patches_with_base(pcd_points, patches, patch_indices):
    """Optimized visualization of patches in different colors on the original point cloud in grey.

    Args:
        pcd_points (np.ndarray): The original point cloud points.
        patches (list of np.ndarray): List of arrays, each containing points of a patch.
        patch_indices (list of np.ndarray): List of arrays, each containing indices of points in a patch.
    """
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)

    # Color all points grey
    colors = np.tile(np.array([0.5, 0.5, 0.5]), (len(pcd_points), 1))  # Grey color for the base point cloud

    # Define a list of predefined basic colors
    predefined_colors = [
        [1, 0, 0],    # Red
        [0, 1, 0],    # Green
        [0, 0, 1],    # Blue
        [1, 1, 0],    # Yellow
        [1, 0, 1],    # Magenta
        [0, 1, 1],    # Cyan
        [1, 0.5, 0],  # Orange
        [0.5, 0, 0.5] # Purple
    ]

    # Assign colors to patches - each patch gets a color from the predefined set
    for i, indices in enumerate(patch_indices):
        color = predefined_colors[i % len(predefined_colors)]  # Cycle through predefined colors
        colors[indices] = color  # Apply color to the whole patch using indices

    # Set the colors of the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    
def farthest_point_sample2(pc_data, num_samples, use_normals=False):
    """Sample points that are farthest from each other in the point cloud, considering normals optionally.

    Args:
        pc_data (np.ndarray): The N x 6 numpy array of point cloud data (x, y, z, nx, ny, nz)
        num_samples (int): Number of samples to select
        use_normals (bool): Whether to use normals in sampling process
    
    Returns:
        np.ndarray: Indices of the sampled points in the original point cloud
    """
    N, D = pc_data.shape
    centroids = np.zeros(num_samples, dtype=int)
    distance = np.ones(N) * np.inf
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid_point = pc_data[farthest]
        if use_normals and D == 6:
            dist = np.linalg.norm(pc_data[:, :3] - centroid_point[:3], axis=1) + np.linalg.norm(pc_data[:, 3:6] - centroid_point[3:6], axis=1)
        else:
            dist = np.linalg.norm(pc_data[:, :3] - centroid_point[:3], axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return centroids

def knn_patch2(pc_path, patch_size=2048, num_patches=6, use_normals=False):
    """Generate patches from a point cloud using KNN algorithm, considering normals.

    Args:
        pc_path (str): Path to the point cloud
        patch_size (int, optional): The number of points each patch should include. Defaults to 2048.
        num_patches (int, optional): The number of patches the point cloud should be divided into. Defaults to 6.
        use_normals (bool): Whether to include normals in the patch creation.
    """
    # Load and process the point cloud
    pcd = o3d.io.read_point_cloud(pc_path)
    points = np.array(pcd.points)
    normals = np.array(pcd.normals) if use_normals else None
    pc_data = np.hstack((points, normals)) if use_normals else points
    pc_data = normalize_pointcloud(pc_data)

    # Prepare KD-Tree 
    pcd.points = o3d.utility.Vector3dVector(points)
    if use_normals:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Sample center points using farthest point sampling
    center_indices = farthest_point_sample2(pc_data, num_patches, use_normals=use_normals)
    
    # Collect patches centered around each sampled point
    N, _ = pc_data.shape
    patch_list = []
    indices_list = []
    
    for center_idx in center_indices:
        if use_normals:
            _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[center_idx], patch_size)
        else:
            _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[center_idx], patch_size)
        patch_list.append(np.asarray(pcd.points)[idx, :])
        indices_list.append(idx)
        
    return patch_list, indices_list