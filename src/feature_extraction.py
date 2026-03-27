from scipy.spatial import ConvexHull
from skimage.measure import CircleModel, ransac
import os
import pandas as pd
import laspy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

def get_robust_dbh(points, z_min, z_max):
    """
    Extracts the trunk's dbh using ransac algorithm
    """
    tree_height = z_max - z_min

    if tree_height > 3.0:
        target_z_start = z_min + 1.2
        target_z_end = z_min + 1.4
    else :
        target_z_start = z_min + (0.2*tree_height)
        target_z_end = z_min + 0.4*tree_height
    
    mask = (points[:,2] >= target_z_start) & (points[:,2] <= target_z_end)
    slice_points = points[mask][:,:2]

    n_points = len(slice_points)
    print(n_points)

    if n_points < 5:
        return 0, 0

    threshold = 0.01 if tree_height < 2.0 else 0.03

    if n_points >= 5:
        model, inliers = ransac(slice_points, CircleModel, min_samples = 3, residual_threshold=threshold, max_trials = 1000)
        diameter = model.radius
        quality = np.sum(inliers) / n_points

    return diameter, quality

def get_trunk_height(stem_diameter, points, z_max, z_min):

    """
    get the trunk height if a the point spread is over a certain threshold
    
    :param stem_diameter: Description
    :param points: the points of the trees
    :param z_max: highest point of the tree
    :param z_min: lowest point of the tree
    """
    
    tree_height = z_max - z_min

    if stem_diameter <= 0 or stem_diameter > tree_height/2:
        return 0
    
    multiplier = 3.0 if stem_diameter > 0.1 else 2.5
    crown_thresh = stem_diameter * multiplier

    if tree_height < 2.0:
        current_z = z_min + 0.2
        step_size = 0.1
    else:
        current_z = z_min + 1.0
        step_size = 0.2
    
    while current_z < z_max:
        mask = (points[:,2] >= current_z) & (points[:,2] < current_z + step_size)
        slice_points = points[mask][:,:2]

        if len(slice_points) < 5:
            current_z += step_size
            continue

        width_x = np.ptp(slice_points[:,0])
        width_y = np.ptp(slice_points[:,1])

        current_width = max(width_x, width_y)

        if current_width > crown_thresh:
            trunk_h = current_z - z_min
            return trunk_h
        
        current_z += step_size

    return tree_height

def extract_tree_features(laz_file_path):
    """
    Extracts geometric features from a .laz file representing a single tree.
    """
    try:
        las = laspy.read(laz_file_path)

        points = np.vstack((las.x, las.y, las.z)).transpose()

        if points.shape[0] < 10:
            return None

        z_max = np.max(points[:, 2])
        z_min = np.min(points[:, 2])

        tree_height = z_max - z_min

        z_coords = points[:, 2]
        z_percentiles = np.percentile(z_coords, [10, 50, 90])
        
        z_percentiles_rel = (z_percentiles - z_min) / tree_height
        
        stem_diameter, stem_quality = get_robust_dbh(points, z_min, z_max)

        trunk_height = get_trunk_height(stem_diameter, points, z_max, z_min)

        crown_mask = points[:, 2] > (z_min + trunk_height)
        crown_points = points[crown_mask]

        foliage_barycenter = np.mean(crown_points, axis = 0)

        try :
            hull = ConvexHull(points)
            tree_volume=hull.volume
            tree_area = hull.area
        except :
            tree_volume = 0
            tree_area = 0

        if len(crown_points) > 4:
            try:
                hull = ConvexHull(crown_points)
                crown_volume = hull.volume
                crown_area = hull.area
            except:
                crown_volume = 0
                crown_area = 0
            
            x_spread = np.ptp(crown_points[:, 0])
            y_spread = np.ptp(crown_points[:, 1])
            crown_diameter = max(x_spread, y_spread)
            
        else:
            crown_volume = 0
            crown_area = 0
            crown_diameter = 0  
        
        crown_ratio = crown_volume / tree_volume if tree_area > 0 else 0
        num_points = len(points)
        point_density = num_points/tree_height

        return {
            'filename': os.path.basename(laz_file_path),
            'height': tree_height,
            'crown_volume' : crown_volume/tree_height,
            'tree_volume' : tree_volume,
            'tree_area' : tree_area,
            'crown_shape' : crown_diameter/(tree_height-trunk_height),
            'slenderness_ratio' : tree_height/trunk_height,
            'crown_area' : crown_area,
            'point_density' : point_density,
            'crown_ratio' : crown_ratio,
            'stem_diameter' : stem_diameter,
            'stem_quality' : stem_quality,
            'trunk_height' : trunk_height,
            'is_sapling' : 1 if tree_height < 2.0 else 0,
            'p10_height_rel': z_percentiles_rel[0],
            'p90_height_rel' : z_percentiles_rel[2],
            'p50_height_rel' : z_percentiles_rel[1],
        }

    except Exception as e:
        print(f"Error processing {laz_file_path}: {e}")
        return None
    
if __name__ == '__main__':
    folder_path = '../data/train_data'

    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.laz')]

    with ProcessPoolExecutor() as executor:
        features = list(tqdm(executor.map(extract_tree_features, all_files), total = len(all_files)))

    results = [f for f in features if f is not None]

    df = pd.DataFrame(results)

    df.to_csv('../data/tableau_features.csv', index = False)
