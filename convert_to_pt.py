import os
import numpy as np
import pandas as pd
import torch
import laspy
from concurrent.futures import ProcessPoolExecutor
import functools
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_single_file(row_data, input_dir, output_dir):
    """Fonction qui sera exécutée par les cœurs CPU en parallèle"""
    laz_path = os.path.join(input_dir, os.path.basename(row_data['filename']))
    output_path = os.path.join(output_dir, os.path.basename(laz_path).replace('.laz', '.pt').replace('.las', '.pt'))
    
    if os.path.exists(output_path): return None # On saute si déjà fait

    try:
        with laspy.open(laz_path) as f:
            las = f.read()
            return np.vstack((las.x, las.y, las.z)).T, output_path
    except:
        return None

def fps_pytorch(xyz, npoint):
    """
    xyz: (N, 3) tensor
    npoint: nombre de points à échantillonner
    """
    device = xyz.device
    N, C = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    
    # On commence par un point aléatoire
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        
        # Calcul de distance Euclidienne au carré (parallélisé sur GPU)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        
        # Mise à jour des distances minimales
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Le prochain point est celui qui est le plus loin de tous les points choisis
        farthest = torch.argmax(distance, -1)
        
    return centroids

def normalize_points(points):
    # Centrage sur (0,0,0)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    # Mise à l'échelle (sphère unité)
    dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / dist
    return points

# Configuration
input_dir = "train_data"
output_dir = "FPS_32k"
csv_path = "train_data/labels.csv"
num_points = 32768 # On peut déjà échantillonner ici pour gagner du temps

os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(csv_path)

num_workers = 16
batch_size = 32

rows = [row for _, row in df.iterrows()]

print(f"Début de la conversion de {len(df)} fichiers...")
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        
        # 1. Lecture parallèle des fichiers (CPU)
        func = functools.partial(process_single_file, input_dir=input_dir, output_dir=output_dir)
        results = list(executor.map(func, batch_rows))
        
        # 2. Calcul FPS (GPU) pour le batch
        for result in results:
            if result is None: continue
            points, out_path = result

            points_tensor = torch.from_numpy(points).float().to(device)
            
            # 2. Échantillonnage (Sampling) fixe pour gagner du temps au train
            if len(points_tensor) >= num_points:
                idx = fps_pytorch(points_tensor, num_points)
                points_sampled = points_tensor[idx]
            else:
                idx = torch.randint(0,len(points_tensor), (num_points,), device = device)
                points_sampled = points_tensor[idx]
            
            # 3. Normalisation
            centroid = torch.mean(points_sampled, dim = 0)
            points_sampled = points_sampled - centroid
            dist = torch.max(torch.sqrt(torch.sum(points_sampled**2, dim = 1)))
            points_sampled = points_sampled / dist
            
            # 4. Sauvegarde en format PyTorch
            torch.save(points_sampled, out_path)
            
            del points_tensor, points_sampled

print(f"Terminé ! Tes fichiers sont dans {output_dir}")