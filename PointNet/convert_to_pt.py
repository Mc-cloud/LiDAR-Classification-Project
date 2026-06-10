import os
import numpy as np
import torch
import laspy
from concurrent.futures import ProcessPoolExecutor
import functools
import glob

# Ensure we use GPU if available (Essential for FPS at 32k)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_single_file_simple(file_path, output_dir):
    """CPU side: Just read the file and prepare the output path"""
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, base_name.replace('.laz', '.pt').replace('.las', '.pt'))
    
    if os.path.exists(output_path): 
        return None 

    try:
        with laspy.open(file_path) as f:
            las = f.read()
            # Extracting XYZ
            return np.vstack((las.x, las.y, las.z)).T, output_path
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def fps_pytorch(xyz, npoint):
    N, C = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=xyz.device)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)
    return centroids

# --- Configuration ---
input_dir = "test_data"       # Folder containing your .laz files
output_dir = "FPS_32k_test"    # Folder for processed .pt files
num_points = 32768
num_workers = 16
batch_size = 32

os.makedirs(output_dir, exist_ok=True)

# Get all .laz and .las files in the directory
files_to_process = glob.glob(os.path.join(input_dir, "*.laz")) + glob.glob(os.path.join(input_dir, "*.las"))

print(f"🚀 Starting conversion of {len(files_to_process)} test files...")

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for i in range(0, len(files_to_process), batch_size):
        batch_files = files_to_process[i : i + batch_size]
        
        # 1. Parallel CPU Read
        func = functools.partial(process_single_file_simple, output_dir=output_dir)
        results = list(executor.map(func, batch_files))
        
        # 2. Batch GPU FPS & Normalization
        for result in results:
            if result is None: continue
            points_np, out_path = result

            # Move to GPU
            points_tensor = torch.from_numpy(points_np).float().to(device)
            
            # Sampling logic
            if len(points_tensor) >= num_points:
                idx = fps_pytorch(points_tensor, num_points)
                points_sampled = points_tensor[idx]
            else:
                # Upsampling if tree has too few points
                idx = torch.randint(0, len(points_tensor), (num_points,), device=device)
                points_sampled = points_tensor[idx]
            
            # Normalization (Center to origin and scale to unit sphere)
            centroid = torch.mean(points_sampled, dim=0)
            points_sampled = points_sampled - centroid
            dist = torch.max(torch.sqrt(torch.sum(points_sampled**2, dim=1)))
            points_sampled = points_sampled / dist
            
            # 4. Save
            torch.save(points_sampled, out_path)
            
            # Explicit memory cleanup (Critical for 32k points on DGX)
            del points_tensor, points_sampled

        if (i // batch_size) % 10 == 0:
            print(f"Processed batch {i // batch_size}...")

print(f"✅ Finished! Files are ready in {output_dir}")