import os
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
import re
from PNTrain.pointnet2_cls_msg import get_model
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---
MODEL_PN_PATH = 'best_model.pth'
TRAIN_LABEL_CSV = "../../data/labels.csv" # Indispensable pour l'ordre des classes
TEST_PT_DIR = '../../data/FPS_32k_test'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. RECONSTRUCTION DU MAPPING (Ordre d'entraînement)
df_train = pd.read_csv(TRAIN_LABEL_CSV)
# .unique() sans tri = même ordre que celui vu par le LightGBM à l'entraînement
noms_especes = df_train['species'].unique()
id_to_species = {i: name for i, name in enumerate(noms_especes)}

print(f"✅ Mapping reconstruit : {len(id_to_species)} classes détectées.")
print(f"Exemple : Index 0 = {id_to_species[0]}")

# 2. CHARGEMENT DES MODÈLES
pn_model = get_model(33, normal_channel=False).to(DEVICE)
checkpoint = torch.load(MODEL_PN_PATH, map_location=DEVICE)
pn_model.load_state_dict(checkpoint['model_state_dict'])
pn_model.eval()

# 3. DATASET POUR LES FICHIERS TEST
class TestDataset(Dataset):
    def __init__(self, pt_dir):
        self.file_list = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
        self.pt_dir = pt_dir
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        fname = self.file_list[idx]
        return torch.load(os.path.join(self.pt_dir, fname)), fname

loader = DataLoader(TestDataset(TEST_PT_DIR), batch_size=8, shuffle=False)

# 4. INFERENCE
results = []
print(f"🚀 Inférence sur {len(loader.dataset)} arbres...")

with torch.no_grad():
    for points, fnames in loader:
        # A. Extraction des 1024 features (PointNet++)
        points = points.to(DEVICE).transpose(2, 1)
        pred, trans_feat, feat = pn_model(points)

        preds_idx = pred.argmax(1)

        # C. Traduction et Stockage
        for j in range(len(fnames)):
            # Extraction de l'ID numérique (ex: 'tree_523.pt' -> 523)
            tree_id = int(re.search(r'\d+', fnames[j]).group())
            species_name = id_to_species[preds_idx[j].item()]
            
            results.append({'treeID': tree_id, 'predicted_species': species_name})

# 5. SAUVEGARDE FORMAT BENCHMARK
# Le benchmark fait un pd.merge(..., on='treeID'), il faut donc ces colonnes exactes.
final_df = pd.DataFrame(results)

# On enregistre en format CSV standard (séparateur virgule) avec le header
final_df.to_csv("final_predictions.csv", sep=',', index=False, header=True)

print(f"✨ Terminé ! Fichier 'final_predictions.txt' prêt pour soumission.")