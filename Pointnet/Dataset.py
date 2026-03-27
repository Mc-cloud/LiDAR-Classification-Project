import torch
from torch.utils.data import Dataset
import numpy as np

class TreeLiDARDataset(Dataset):
    def __init__(self, tree_arrays, labels, num_points = 4096):
        self.tree_arrays = tree_arrays
        self.labels = labels
        self.num_points = num_points
    
    def __len__(self):
        return len(self.tree_arrays)
    
    def __getitem__(self,idx):
        tree_data = self.tree_arrays[idx]
        label = self.labels[idx]

        num_current_points = tree_data.shape[0]

        if num_current_points >= self.num_points:
            choice = np.random.choice(num_current_points, self.num_points, replace = False)
        else : 
            choice = np.random.choice(num_current_points, self.num_points, replace = False)

        tree_sampled = tree_data[choice, :]

        centroid_xy = np.mean(tree_sampled[:,2], axis = 0)
        tree_sampled[:,:2] -= centroid_xy

        tree_sampled = tree_sampled.T

        tree_tensor = torch.from_numpy(tree_sampled).float()
        label_tensor = torch.tensor(label, dtype = torch.long)

        return tree_tensor, label_tensor