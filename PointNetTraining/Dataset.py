import torch
from torch.utils.data import Dataset
import numpy as np
import laspy 

class TreeLiDARDataset(Dataset):
    def __init__(self, file_paths, labels, num_points = 4096, transform = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.num_points = num_points
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self,idx):
        points = torch.load(self.file_paths[idx], map_location='cpu', weights_only = True)
        label = self.labels[idx]

        if points.shape[0] > self.num_points :
            indices = torch.randperm(points.shape[0])[:self.num_points]
            points = points[indices]

        if self.transform :
            points = self.transform(points)

        points = points.transpose(0, 1)
        
        return points, label


class PointCloudTransforms :
    def __init__(self, rotation = True, jitter = True, scale = False):
        self.rotation = rotation
        self.jitter = jitter
        self.scale = scale
    
    def __call__(self, points):

        if self.rotation :

            fold = np.random.randint(0, 6)
            theta = fold * (np.pi/3.0)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            R = torch.tensor([[cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0,0,1]], dtype = torch.float32)

            points = torch.matmul(points, R)

        if self.scale :
            scale = np.random.uniform(0.9, 1.1)
            points = points*scale

        if self.jitter:
            noise = torch.randn_like(points)*0.001
            points = points + noise
        
        return points


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for points, target in loader:
            points = points.to(device).transpose(2,1)
            feat, _ = model.get_global_features(points)

            features.append(feat.cpu().numpy())

            labels.append(target.numpy())

            return np.vstack(features), np.concatenate(labels)