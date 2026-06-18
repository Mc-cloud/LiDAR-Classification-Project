import os
import torch
import numpy as np
import pandas as pd
import re
from pointnet2_cls_msg import get_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- CONFIGURATION ---
MODE = "TRAIN" # Choisir "TRAIN" (pour LightGBM) ou "TEST" (pour la soumission)

MODEL_PN_PATH = 'best_model.pth'
TRAIN_LABEL_CSV = "../../data/labels_split_complex.csv"
TRAIN_PT_DIR = '../../data/FPS_32k_train'
TEST_PT_DIR = '../../data/FPS_32k_test'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. CHARGEMENT DU MODÈLE POINTNET
print(f"Chargement de PointNet++ sur {DEVICE}...")
pn_model = get_model(33, normal_channel=False).to(DEVICE)
checkpoint = torch.load(MODEL_PN_PATH, map_location=DEVICE)
pn_model.load_state_dict(checkpoint['model_state_dict'])
pn_model.eval()

# 2. PRÉPARATION DES FICHIERS SELON LE MODE
if MODE == "TRAIN":
    df_labels = pd.read_csv(TRAIN_LABEL_CSV)
    # On reconstruit les noms de fichiers .pt à partir du CSV
    file_list = [os.path.basename(f).replace('.laz', '.pt').replace('.las', '.pt') for f in df_labels['filename']]
    pt_dir = TRAIN_PT_DIR
    output_csv = "../../data/pointnet_features_train.csv"
else:
    file_list = [f for f in os.listdir(TEST_PT_DIR) if f.endswith('.pt')]
    pt_dir = TEST_PT_DIR
    output_csv = "../../data/pointnet_features_test.csv"

# 3. DATASET D'EXTRACTION
class ExtractionDataset(Dataset):
    def __init__(self, file_list, pt_dir):
        self.file_list = file_list
        self.pt_dir = pt_dir
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        fname = self.file_list[idx]
        return torch.load(os.path.join(self.pt_dir, fname)), fname

loader = DataLoader(ExtractionDataset(file_list, pt_dir), batch_size=16, shuffle=False, num_workers=8)

# 4. EXTRACTION
all_features = []
all_fnames = []

print(f"🚀 Extraction des features ({MODE} mode) sur {len(loader.dataset)} arbres...")

with torch.no_grad():
    for points, fnames in tqdm(loader):
        points = points.to(DEVICE).transpose(2, 1)
        
        # On ne garde que 'feat' (le vecteur 1024D avant la classification finale)
        _, _, feat = pn_model(points)
        
        all_features.append(feat.cpu().numpy())
        all_fnames.extend(fnames)

# Concaténer tous les batchs
features_matrix = np.vstack(all_features)

# 5. SAUVEGARDE EN CSV
# Créer des noms de colonnes dynamiques (pn_feat_0, pn_feat_1, ..., pn_feat_1023)
feat_cols = [f"pn_feat_{i}" for i in range(features_matrix.shape[1])]
df_features = pd.DataFrame(features_matrix, columns=feat_cols)
df_features['filename'] = all_fnames

# Réorganiser pour mettre 'filename' en première colonne
cols = ['filename'] + feat_cols
df_features = df_features[cols]

df_features.to_csv(output_csv, index=False)
print(f"✨ Terminé ! Features sauvegardées dans '{output_csv}'")