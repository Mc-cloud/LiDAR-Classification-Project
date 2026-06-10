import os
import torch
import pandas as pd      # 🛠️ Import manquant
import numpy as np       # 🛠️ Import manquant
from pathlib import Path # 🛠️ Import manquant
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm    # Pour avoir une belle barre de progression

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
DATA_DIR = "./dataset/train"
CSV_PATH = "./labels.csv"  # 🛠️ Ajoute le chemin vers ton CSV
EMBEDDINGS_FILE = "dinov3_tree_embeddings.pt"

BATCH_SIZE = 1     # Reste à 1 pour l'extraction car un arbre = 5 images
NUM_WORKERS = 8    # Utilise les multiples cœurs CPU du DGX

print("1. Chargement du modèle...")
# processor = AutoImageProcessor.from_pretrained(MODEL_ID) # (Non utilisé vu que tes données sont déjà en tenseurs)
model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
model.eval()

# ==========================================
# DATASET
# ==========================================
class NpyTreeDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        print(f"Lecture du CSV : {csv_file}")
        self.df = pd.read_csv(csv_file)
        
        self.classes = sorted(self.df['species'].unique().tolist())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.filepaths = list(Path(root_dir).rglob("*.npy"))

        self.df['treeID'] = self.df['treeID'].astype(str).str.zfill(5) 
        
        self.file_to_species = dict(zip(self.df['treeID'], self.df['species']))
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        file_id = path.stem 
        
        # On cherche sa classe dans notre dictionnaire issu du CSV
        species_name = self.file_to_species[file_id]
        label = self.class_to_idx[species_name]
        
        # Chargement de la matrice
        data = np.load(path)
        
        # 🛠️ CORRECTION : Hugging Face s'attend à du BFloat16 puisque le modèle est chargé en BFloat16
        tensor_data = torch.from_numpy(data).to(torch.bfloat16)
        
        return tensor_data, label

# ==========================================
# EXTRACTION
# ==========================================
def extract_embeddings():
    print("\n--- DÉBUT DE L'EXTRACTION DES VECTEURS ---")
    
    # 🛠️ CORRECTION : On passe bien DATA_DIR et CSV_PATH
    dataset = NpyTreeDataset(DATA_DIR, CSV_PATH)
    print(f"🌲 {len(dataset)} arbres trouvés, répartis en {len(dataset.classes)} espèces.")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for views, label in tqdm(loader, desc="Extraction DINOv3"):
            # views a la forme (1, 5, 3, 224, 224) -> devient (5, 3, 224, 224)
            views = views.squeeze(0).to("cuda") 
            
            # 🛠️ CORRECTION : L'inférence avec Hugging Face 
            outputs = model(pixel_values=views) 
            
            # On récupère le CLS token (qui représente l'image entière). Shape: (5, 4096)
            features = outputs.last_hidden_state[:, 0, :] 
            
            # LE MEAN POOLING : On fait la moyenne des 5 vues
            tree_embedding = features.mean(dim=0) # Shape: (4096)
            
            all_embeddings.append(tree_embedding.cpu())
            all_labels.append(label.item())
            
    # Sauvegarde sur le disque
    torch.save({
        "embeddings": torch.stack(all_embeddings),
        "labels": all_labels,
        "class_names": dataset.classes
    }, EMBEDDINGS_FILE)
    print(f"✅ Fichier sauvegardé : {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    extract_embeddings()