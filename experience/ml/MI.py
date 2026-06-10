import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils import * 
import pandas as pd

labels = pd.read_csv('../../data/train_data/labels.csv')

list_files = [os.path]

grids = [preprocess_point_cloud(laz_filename) for laz_filename in list_files]

dist_matrix_train = build_distance_matrix