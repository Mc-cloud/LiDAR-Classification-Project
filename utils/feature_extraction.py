from scipy.spatial import ConvexHull
import os
import pandas as pd
import laspy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

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

        target_points = 5000

        if len(points) > target_points : 
            idx = np.random.choice(len(points), target_points, replace = False)
            geom_points = points[idx]
        else :
            geom_points = points

        try :
            hull_3d = ConvexHull(geom_points)
            tree_volume = hull_3d.volume
        except:
            tree_volume = 0
        
        try :
            hull_2d = ConvexHull(geom_points[:,:2])
            crown_area = hull_2d.volume
        except :
            crown_area = 0
        
    
        points_centered = points - np.mean(points, axis=0)

        x_spread = np.max(points[:, 0]) - np.min(points[:, 0])
        y_spread = np.max(points[:, 1]) - np.min(points[:, 1])
        crown_diameter = max(x_spread, y_spread)

        point_density = len(points) / tree_volume if tree_volume > 0 else 0

        z_coords = points[:, 2]
        z_percentiles = np.percentile(z_coords, [10, 25, 50, 75, 90])
        
        z_percentiles_rel = (z_percentiles - z_min) / tree_height

        breast_height_start = z_min + 1.3
        breast_height_end = z_min + 1.5
        
        slice_mask = (points[:, 2] >= breast_height_start) & (points[:, 2] <= breast_height_end)
        stem_slice = points[slice_mask]
        
        dbh = 0
        if len(stem_slice) > 0:
            slice_x_spread = np.max(stem_slice[:, 0]) - np.min(stem_slice[:, 0])
            slice_y_spread = np.max(stem_slice[:, 1]) - np.min(stem_slice[:, 1])
            dbh = (slice_x_spread + slice_y_spread) / 2

        return {
            'filename': os.path.basename(laz_file_path),
            'num_points': len(points),
            'height': tree_height,
            'crown_diameter': crown_diameter,
            'crown_area': crown_area,
            'volume': tree_volume,
            'point_density': point_density,
            'dbh_approx': dbh,
            'p10_height_rel': z_percentiles_rel[0],
            'p50_height_rel': z_percentiles_rel[2], # Median height
            'p90_height_rel': z_percentiles_rel[4],
        }

    except Exception as e:
        print(f"Error processing {laz_file_path}: {e}")
        return None