import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Le dossier contenant tes fichiers .npy générés pour le test
TEST_NPY_DIR = "dataset/test_proj"

# Ton modèle PyTorch sauvegardé
CHECKPOINT_FILE = "meilleur_modele_mlp.pt" 

# Le fichier CSV de sortie avec tes prédictions
OUTPUT_CSV = "predictions_soumission.csv"
# Fichier contenant les noms des classes (généré lors de l'entraînement)
CLASSES_FILE = "classes_arbres.txt"

# Modèle DINOv3
MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
DINO_DIM = 4096
NUM_CLASSES = 33 

# ==========================================
# 🧠 ARCHITECTURE DU MODÈLE
# ==========================================
class TreeStudent(nn.Module):
    def __init__(self, input_dim, num_classes, use_mlp=True):
        super().__init__()
        if use_mlp:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            self.network = nn.Linear(input_dim, num_classes)
            
    def forward(self, x):
        return self.network(x)

# ==========================================
# 📦 DATASET DE TEST (Épuré)
# ==========================================
class TestTreeDataset(Dataset):
    def __init__(self, npy_dir):
        self.filepaths = list(Path(npy_dir).rglob("*.npy"))
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        file_id = path.stem 
        
        # Chargement des 5 vues (5, 3, 224, 224)
        data = np.load(path)
        tensor_data = torch.from_numpy(data).to(torch.bfloat16)
        
        return tensor_data, file_id

# ==========================================
# 🚀 FONCTION DE PRÉDICTION
# ==========================================
def run_prediction():
    print("--- 1. CHARGEMENT DES MODÈLES ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation de l'accélérateur : {device}")

    # Chargement de la liste des classes (si elle existe)
    class_names = None
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"✅ Fichier de classes chargé ({len(class_names)} espèces).")
    else:
        print("⚠️ Fichier de classes introuvable. Les prédictions seront des numéros (0 à 32).")

    # 1A. Chargement de DINOv3
    print("Chargement de DINOv3...")
    dino_model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
    dino_model.eval()

    # 1B. Chargement de ton MLP
    print(f"Chargement de ton modèle PyTorch : {CHECKPOINT_FILE}")
    student_model = TreeStudent(DINO_DIM, NUM_CLASSES, use_mlp=True).to(device)
    student_model.load_state_dict(torch.load(CHECKPOINT_FILE))
    student_model.eval()

    print("\n--- 2. PRÉPARATION DES DONNÉES ---")
    dataset = TestTreeDataset(TEST_NPY_DIR)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)
    print(f"🌲 {len(dataset)} arbres à prédire.")

    all_ids = []
    all_preds = []

    print("\n--- 3. INFÉRENCE EN COURS ---")
    with torch.no_grad():
        for views, file_id in tqdm(loader, desc="Prédiction"):
            # A. DINO extrait les caractéristiques
            views = views.squeeze(0).to(device) 
            outputs = dino_model(pixel_values=views) 
            features = outputs.last_hidden_state[:, 0, :] 
            tree_embedding = features.mean(dim=0).float().unsqueeze(0) # Shape: (1, 4096)

            # B. Prédiction avec le MLP
            pred_logits = student_model(tree_embedding)
            pred_class_idx = pred_logits.argmax(dim=1).item()
            
            # C. Traduction du numéro en nom d'espèce si possible
            if class_names:
                final_prediction = class_names[pred_class_idx]
            else:
                final_prediction = pred_class_idx
                
            all_ids.append(file_id[0])
            all_preds.append(final_prediction)

    # 4. SAUVEGARDE DU CSV
    print("\n--- 4. GÉNÉRATION DU RÉSULTAT ---")
    out_df = pd.DataFrame({
        "treeID": all_ids,
        "predicted_species": all_preds 
    })
    
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV sauvegardé avec succès : {OUTPUT_CSV}")
    print(out_df.head())

if __name__ == "__main__":
    run_prediction()