from scipy.spatial import ConvexHull
from skimage.measure import CircleModel, ransac
import os
import pandas as pd
import laspy
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from skimage.measure import EllipseModel
from scipy.optimize import minimize
import numpy as np

def fit_crown_profiles(crown_points, n_angles=8):
    """
    Projette les points couronne sur N plans verticaux et fitte
    triangle, ellipse et rectangle. Retourne des features de forme
    décorrélées de la taille (normalisées par h_crown et w_crown).
    """
    if len(crown_points) < 10:
        return _empty_profile_features()

    z = crown_points[:, 2]
    h_crown = np.ptp(z)
    if h_crown < 0.1:
        return _empty_profile_features()

    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    all_features = []
    for angle in angles:
        # Projection sur l'axe perpendiculaire à `angle`
        proj = (crown_points[:, 0] * np.cos(angle)
              + crown_points[:, 1] * np.sin(angle))
        profile = np.column_stack([proj, z])

        feats = _fit_single_profile(profile)
        if feats is not None:
            all_features.append(feats)

    if not all_features:
        return _empty_profile_features()

    # Moyenne sur tous les profils → invariance à la rotation
    keys = all_features[0].keys()
    return {k: float(np.mean([f[k] for f in all_features])) for k in keys}


def _fit_single_profile(profile):
    """Fitte triangle, ellipse et rectangle sur un profil 2D (N,2)."""
    x, z = profile[:, 0], profile[:, 1]
    z_min, z_max = z.min(), z.max()
    h = z_max - z_min
    w = np.ptp(x)
    if h < 0.01 or w < 0.01:
        return None

    # ── 1. Rectangle ──────────────────────────────────────────────
    bbox_area = h * w
    # remplissage : fraction des cellules d'une grille 20x20 occupées
    grid_res = 20
    xi = np.floor((x - x.min()) / w * (grid_res - 1)).astype(int)
    zi = np.floor((z - z_min) / h * (grid_res - 1)).astype(int)
    occupied = len(set(zip(xi, zi)))
    rect_fill = occupied / (grid_res ** 2)
    wh_ratio = w / h  # normalisé → indépendant de la taille

    # ── 2. Ellipse (RANSAC) ───────────────────────────────────────
    try:
        model = EllipseModel()
        profile_norm = np.column_stack([(x - x.mean()) / w,
                                        (z - z.mean()) / h])
        model.estimate(profile_norm)
        xc, yc, a, b, theta = model.params
        ellipse_ab_ratio = min(a, b) / max(a, b) if max(a, b) > 0 else 0
        ellipse_ecc = np.sqrt(1 - (min(a,b)/max(a,b))**2) if max(a,b)>0 else 1
        residuals = model.residuals(profile_norm)
        ellipse_rmse = np.sqrt(np.mean(residuals**2))
    except Exception:
        ellipse_ab_ratio = 0
        ellipse_ecc = 1
        ellipse_rmse = 1.0

    # ── 3. Triangle isocèle ───────────────────────────────────────
    def triangle_residuals(params):
        apex_x, half_w = params
        # Bords gauche et droit du triangle
        # z normalisé entre 0 (base) et 1 (apex)
        z_norm = (z - z_min) / h
        x_norm = (x - x.min()) / w
        apex_xn = (apex_x - x.min()) / w
        hw_n = half_w / w
        # Distance de chaque point au bord le plus proche
        left_x  = apex_xn - hw_n * (1 - z_norm)
        right_x = apex_xn + hw_n * (1 - z_norm)
        dist_left  = np.abs(x_norm - left_x)
        dist_right = np.abs(x_norm - right_x)
        dist_inside = np.minimum(dist_left, dist_right)
        return np.sum(dist_inside**2)

    try:
        x0 = [x.mean(), w / 2]
        bounds = [(x.min(), x.max()), (0.01, w)]
        res = minimize(triangle_residuals, x0, bounds=bounds, method='L-BFGS-B')
        _, half_w_fit = res.x
        # Angle d'apex en radians (normalisé → invariant à la taille)
        apex_angle = 2 * np.arctan(half_w_fit / h)
        triangle_rmse = np.sqrt(res.fun / len(x))
    except Exception:
        apex_angle = np.pi / 2
        triangle_rmse = 1.0

    # ── Asymétrie du profil ───────────────────────────────────────
    x_center = x.mean()
    left_pts  = x[x < x_center]
    right_pts = x[x >= x_center]
    asymmetry = (abs(len(left_pts) - len(right_pts))
                 / len(x)) if len(x) > 0 else 0

    return {
        'wh_ratio':          wh_ratio,
        'rect_fill_ratio':   rect_fill,
        'ellipse_ab_ratio':  ellipse_ab_ratio,
        'ellipse_ecc':       ellipse_ecc,
        'ellipse_rmse':      ellipse_rmse,
        'apex_angle_rad':    apex_angle,      # petit = conifère pointu
        'triangle_rmse':     triangle_rmse,
        'profile_asymmetry': asymmetry,
    }


def _empty_profile_features():
    return {k: 0.0 for k in [
        'wh_ratio', 'rect_fill_ratio', 'ellipse_ab_ratio',
        'ellipse_ecc', 'ellipse_rmse', 'apex_angle_rad',
        'triangle_rmse', 'profile_asymmetry'
    ]}

def get_robust_dbh(points, z_min, z_max):
    """
    Extracts the trunk's height using ransac algorithm
    """

    tree_height = z_max - z_min

    if tree_height > 2.0:
        target_z_start = z_min + 1.3
        target_z_end = z_min + 1.5
    
    else :
        target_z_start = z_min + (0.2*tree_height)
        target_z_end = z_min + 0.4*tree_height
    
    mask = (points[:,2] >= target_z_start) & (points[:,2] <= target_z_end)
    slice_points = points[mask][:,:2]

    n_points = len(slice_points)

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

        try:
            hull = ConvexHull(points)
            tree_volume = hull.volume
            tree_area = hull.area
        except:
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

        crown_ratio = crown_volume / tree_volume if tree_volume > 0 else 0
        num_points = len(points)
        point_density = num_points / tree_height

        # ── Nouvelles features de forme de profil ──────────────────
        profile_feats = fit_crown_profiles(crown_points)  # 👈 appel ici
        # ───────────────────────────────────────────────────────────

        return {
            'filename': os.path.basename(laz_file_path),
            'height': tree_height,
            'crown_volume': crown_volume,
            'tree_volume': tree_volume,
            'tree_area': tree_area,
            'crown_diameter': crown_diameter,
            'crown_area': crown_area,
            'point_density': point_density,
            'crown_ratio': crown_ratio,
            'stem_diameter': stem_diameter,
            'stem_quality': stem_quality,
            'trunk_height': trunk_height,
            'is_sapling': 1 if tree_height < 2.0 else 0,
            'p10_height_rel': z_percentiles_rel[0],
            'p50_height_rel': z_percentiles_rel[1],
            'p90_height_rel': z_percentiles_rel[2],
            # 👇 les features de profil s'ajoutent ici automatiquement
            # elles seront nommées profile_wh_ratio, profile_apex_angle_rad, etc.
            **{f'profile_{k}': v for k, v in profile_feats.items()},
        }

    except Exception as e:
        print(f"Error processing {laz_file_path}: {e}")
        return None

    except Exception as e:
        print(f"Error processing {laz_file_path}: {e}")
        return None

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
