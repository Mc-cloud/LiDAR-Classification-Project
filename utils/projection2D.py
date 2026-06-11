import os
import math
import numpy as np
import laspy
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
LAZ_DIR = "dataset/test"
OUTPUT_DIR = "dataset/test_proj"
IMAGE_SIZE = 224
# On force les 4 angles pour obtenir exactement 5 vues (1 Top + 4 Sides)
SIDE_ANGLES = [0, 90, 180, 270] 

# ============================================================
# MOTEUR DE PROJECTION (Code strict du collègue)
# ============================================================
def read_laz_xyz(path):
    las = laspy.read(path)

    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)

    xyz = np.stack([x, y, z], axis=1)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]

    if len(xyz) == 0:
        raise ValueError(f"Nuage vide ou invalide : {path}")

    return xyz.astype(np.float32)


def centered_scaled_xyz(xyz):
    """
    Centre l'arbre horizontalement et normalise par une échelle commune.
    z est remis à partir du sol local de l'arbre.
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    sx = xmax - xmin
    sy = ymax - ymin
    sz = zmax - zmin

    scale = max(sx, sy, sz, 1e-6)

    xc = (x - cx) / scale
    yc = (y - cy) / scale
    zc = (z - zmin) / scale

    return xc.astype(np.float32), yc.astype(np.float32), zc.astype(np.float32)


def rotate_xy(x, y, angle_deg):
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)

    xr = c * x - s * y
    yr = s * x + c * y

    return xr.astype(np.float32), yr.astype(np.float32)


def render_projection(u, v, attr1, attr2, image_size):
    """
    Produit une image 3 canaux :
    - canal 0 : densité de points en log ;
    - canal 1 : attribut max par pixel ;
    - canal 2 : attribut moyen par pixel.
    """
    H = int(image_size)
    W = int(image_size)

    uu = ((u + 0.5) * (W - 1)).astype(np.int64)
    vv = ((v + 0.5) * (H - 1)).astype(np.int64)

    mask = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)

    uu = uu[mask]
    vv = vv[mask]
    a1 = np.clip(attr1[mask], 0.0, 1.0).astype(np.float32)
    a2 = np.clip(attr2[mask], 0.0, 1.0).astype(np.float32)

    count = np.zeros((H, W), dtype=np.float32)
    max_a1 = np.zeros((H, W), dtype=np.float32)
    sum_a2 = np.zeros((H, W), dtype=np.float32)

    np.add.at(count, (vv, uu), 1.0)
    np.maximum.at(max_a1, (vv, uu), a1)
    np.add.at(sum_a2, (vv, uu), a2)

    density = np.log1p(count)
    max_density = float(np.max(density))
    if max_density > 0.0:
        density = density / max_density

    mean_a2 = np.zeros_like(sum_a2)
    nonzero = count > 0
    mean_a2[nonzero] = sum_a2[nonzero] / count[nonzero]

    img = np.stack([density, max_a1, mean_a2], axis=0)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    return img


def make_views_from_laz(path, image_size=224, side_angles=None):
    """
    Retourne un tenseur numpy (V, 3, H, W).
    """
    if side_angles is None:
        side_angles = [0, 45, 90, 135, 180, 225, 270, 315]

    xyz = read_laz_xyz(path)
    x, y, z = centered_scaled_xyz(xyz)

    views = []

    # Vue du dessus : u=x, v=y, informations de hauteur.
    top = render_projection(
        u=x,
        v=y,
        attr1=z,
        attr2=z,
        image_size=image_size,
    )
    views.append(top)

    # Vues de côté.
    for angle in side_angles:
        xr, yr = rotate_xy(x, y, angle)

        side = render_projection(
            u=xr,
            v=z - 0.5,
            attr1=yr + 0.5,
            attr2=yr + 0.5,
            image_size=image_size,
        )
        views.append(side)

    arr = np.stack(views, axis=0).astype(np.float32)
    return arr

# ============================================================
# SCRIPT DE SAUVEGARDE NPY
# ============================================================
def generate_projections():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    laz_files = list(Path(LAZ_DIR).rglob("*.laz")) + list(Path(LAZ_DIR).rglob("*.las"))
    print(f"🌲 {len(laz_files)} fichiers LiDAR trouvés à projeter.")
    
    for filepath in tqdm(laz_files, desc="Génération des matrices .npy"):
        tree_id = filepath.stem
        out_path = os.path.join(OUTPUT_DIR, f"{tree_id}.npy")
        
        try:
            # Génération des 5 vues en utilisant le code strict du collègue
            views_np = make_views_from_laz(str(filepath), image_size=IMAGE_SIZE, side_angles=SIDE_ANGLES)
            
            # Sauvegarde de l'array de forme (5, 3, 224, 224)
            np.save(out_path, views_np)
            
        except Exception as e:
            print(f"❌ Erreur sur l'arbre {tree_id} : {e}")

    print(f"\n✅ Projections terminées. Fichiers sauvegardés dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_projections()