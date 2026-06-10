import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
EMBEDDINGS_FILE = "dinov3_tree_embeddings.pt"
OUTPUT_MODEL_FILE = "models/meilleur_modele_mlp.pt"

# ==========================================
# 🧠 ARCHITECTURE DE L'ÉLÈVE
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
# 🚀 ENTRAÎNEMENT ET SAUVEGARDE
# ==========================================
def train_and_save_champion():
    print(f"\n--- 1. CHARGEMENT DES DONNÉES DINOv3 ---")
    data = torch.load(EMBEDDINGS_FILE)
    X_np = data["embeddings"].float().numpy()  # 4096 dimensions
    y_np = np.array(data["labels"])
    class_names = data["class_names"]
    num_classes = len(class_names)
    
    print(f"🌲 Nombre total d'arbres : {len(X_np)}")
    print(f"🏷️ Nombre d'espèces : {num_classes}")

    # 2. SPLIT STRATIFIÉ
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_np, y_np, test_size=0.20, random_state=42, stratify=y_np
    )
    
    # 3. LE SAMPLER (L'arme secrète)
    class_counts = np.bincount(y_train_np)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[t] for t in y_train_np])
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # 4. PRÉPARATION PYTORCH
    X_train = torch.from_numpy(X_train_np).float()
    X_val = torch.from_numpy(X_val_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    y_val = torch.from_numpy(y_val_np).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

    input_dim = X_train_np.shape[1] # 4096
    model = TreeStudent(input_dim, num_classes, use_mlp=True).to("cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 30
    best_macro_f1 = 0
    best_model_weights = None
    best_epoch = 0
    
    print("\n--- 2. DÉBUT DE L'ENTRAÎNEMENT ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda")
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Évaluation Validation
        model.eval()
        y_pred_epoch = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to("cuda")
                preds = model(batch_X).argmax(dim=1).cpu().numpy()
                y_pred_epoch.extend(preds)
                
        # Calcul du Macro F1
        _, _, macro_f1_epoch, _ = precision_recall_fscore_support(
            y_val_np, y_pred_epoch, average='macro', zero_division=0
        )
        macro_f1_epoch *= 100
        
        print(f"Époque {epoch+1:02d}/{epochs} | Perte: {total_loss/len(train_loader):.4f} | Macro F1: {macro_f1_epoch:.2f}%")
        
        # Sauvegarde en RAM de la meilleure version
        if macro_f1_epoch > best_macro_f1:
            best_macro_f1 = macro_f1_epoch
            best_epoch = epoch + 1
            best_model_weights = copy.deepcopy(model.state_dict())

    print("\n--- 3. SAUVEGARDE DU MODÈLE CHAMPION ---")
    print(f"🏆 Meilleur Macro F1 atteint : {best_macro_f1:.2f}% (à l'époque {best_epoch})")
    
    # 💾 SAUVEGARDE SUR LE DISQUE
    torch.save(best_model_weights, OUTPUT_MODEL_FILE)
    print(f"✅ Poids du modèle sauvegardés sous : {OUTPUT_MODEL_FILE}")
    
    # Astuce : On sauvegarde aussi la liste des classes dans un petit fichier texte
    # Ça sera ultra utile pour le script d'inférence !
    with open("classes_arbres.txt", "w") as f:
        for c in class_names:
            f.write(f"{c}\n")
    print("✅ Noms des espèces sauvegardés sous : classes_arbres.txt")

if __name__ == "__main__":
    train_and_save_champion()