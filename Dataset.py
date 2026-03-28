import torch
from torch.utils.data import Dataset
import numpy as np
import laspy 

class TreeLiDARDataset(Dataset):
    def __init__(self, file_paths, labels, transform = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self,idx):
        points = torch.load(self.file_paths[idx])
        label = self.labels[idx]

        if self.transform :
            points = self.transform(points)
        
        return points, label


class PointCloudTransforms :
    def __init__(self, rotation = True, jitter = True, scale = True):
        self.rotation = rotation
        self.jitter = jitter
        self.scale = scale
    
    def __call__(self, points):

        if self.rotation :
            theta = np.random.uniform(0, 2*np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            R = torch.tensor([[cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0,0,1]], dtype = torch.float32)

            points = torch.matmul(points, R)

        if self.scale :
            scale = np.random.uniform(0.9, 1.1)
            points = points*scale

        if self.jitter:
            noise = torch.randn_like(points)*0.01
            points = points + noise
        
        return points
