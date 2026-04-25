import numpy as np
from scipy.spatial import KDTree

def extract_traditional_features(coords):
    """
    Extracts 4 traditional geometric features from a 3D point cloud.
    Input: coords - numpy array of shape (N, 3)
    Output: list [num_points, bounding_radius, mean_nn_distance, mean_curvature]
    """
    # 1. Number of points
    num_points = coords.shape[0]
    
    # 2. Bounding radius (computed from the centroid)
    centroid = np.mean(coords, axis=0)
    distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    bounding_radius = np.max(distances_to_centroid)
    
    # Initialize KDTree for fast neighbor lookup
    tree = KDTree(coords)
    
    # 3. Mean nearest neighbor distance
    # k=2 because the closest point to itself is distance 0
    distances, _ = tree.query(coords, k=2)
    mean_nn_distance = np.mean(distances[:, 1]) 
    
    # 4. Mean local curvature (Local PCA on 5 nearest neighbors)
    k_neighbors = 5
    _, indices = tree.query(coords, k=k_neighbors)
    curvatures = []
    
    for neighbors_idx in indices:
        neighbors = coords[neighbors_idx]
        
        # Center the local patch and compute the covariance matrix
        local_centroid = np.mean(neighbors, axis=0)
        centered = neighbors - local_centroid
        cov_matrix = np.dot(centered.T, centered) / k_neighbors
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix) 
        
        # Curvature = smallest eigenvalue / sum of all eigenvalues
        # Add 1e-8 to prevent division by zero
        curvature = eigenvalues[0] / (np.sum(eigenvalues) + 1e-8)
        curvatures.append(curvature)
        
    mean_curvature = np.mean(curvatures)
    
    return [num_points, bounding_radius, mean_nn_distance, mean_curvature]