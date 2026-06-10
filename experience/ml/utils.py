import numpy as np
import os
import laspy


def preprocess_point_cloud(laz_file_path, bins = 32):
    las = laspy.read(laz_file_path)

    points = np.vstack((las.x, las.y, las.z)).transpose()

    centroid = np.mean(points, axis = 0)
    centered_points = points - centroid

    cov_matrix = np.cov(centered_points, rowvar=False)

    eigen_val, eigen_vect = np.linalg.eigh(cov_matrix)

    sort_indices = np.argsort(eigen_val)[::-1]
    sorted_eigenvect = eigen_vect[:, sort_indices]

    aligned_points = np.dot(centered_points, sorted_eigenvect)

    max_coord = np.max(np.abs(aligned_points))
    
    if max_coord > 0 :
        normalized_points = aligned_points/max_coord
    else :
        normalized_points = aligned_points

    grid, _ = np.histogramdd(
        normalized_points,
        bins = bins,
        range = [[-1, 1], [-1,1], [-1, 1]]
        )
    
    return grid


def calculate_vi(grid_a, grid_b):
    flat_a = grid_a.flatten()
    flat_b = grid_b.flatten()

    max_points = int(max(np.max(flat_a), np.max(flat_b)) + 1)

    joint_hist, _, _ = np.histogram2d(
        flat_a, flat_b,
        bins = max_points,
        range = [[0, max_points], [0, max_points]]
    )

    n_voxels = len(flat_a)
    p_joint = joint_hist / n_voxels

    p_a = np.sum(p_joint, axis = 1)
    p_b = np.sum(p_joint, axis = 0)

    p_joint_safe = p_joint[p_joint > 0]
    h_joint = -np.sum(p_joint_safe * np.log2(p_joint_safe))

    p_a_safe = p_a[p_a > 0]
    h_a = -np.sum(p_a_safe*np.log2(p_a_safe))

    p_b_safe = p_b[p_b > 0]
    h_b = -np.sum(p_b_safe * np.log2(p_b_safe))

    vi = 2*h_joint - h_a - h_b

    return vi


def build_distance_matrix(grids):
    n = len(grids)
    dist_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            dist = calculate_vi(grids[i], grids[j])
            dist_matrix[i, j] = dist
            dist_matrix[j,i] = dist
    
    return dist_matrix

