import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import wandb
import scipy.stats as st

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, f1_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ============================================================
# 🧠 ARCHITECTURE MULTIMODALE (DINO + GÉOMÉTRIE)
# ============================================================
class TreeStudent(nn.Module):
    def __init__(self, dino_dim, geo_dim, num_classes, use_mlp=True):
        super().__init__()
        
        # 1️⃣ Sous-réseau dédié à la géométrie (Inspiré de ton collègue)
        geo_hidden_dim = 128
        self.geo_net = nn.Sequential(
            nn.LayerNorm(geo_dim),
            nn.Linear(geo_dim, geo_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),  # Dropout plus léger pour la géométrie
            nn.Linear(geo_hidden_dim, geo_hidden_dim),
            nn.GELU()
        )
        
        # 2️⃣ Réseau principal de fusion (DINO + Géo projetée)
        fused_dim = dino_dim + geo_hidden_dim
        
        if use_mlp:
            self.network = nn.Sequential(
                nn.Linear(fused_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            self.network = nn.Linear(fused_dim, num_classes)
            
    def forward(self, x_dino, x_geo):
        # On calcule les features géométriques
        geo_feats = self.geo_net(x_geo)
        # On les colle à côté des features DINO
        fused = torch.cat([x_dino, geo_feats], dim=1)
        # On passe le tout dans le classifieur final
        return self.network(fused)

# ============================================================
# ⚙️ BOUCLE D'ENTRAÎNEMENT PYTORCH
# ============================================================
def train_mlp(model, train_loader, val_loader, optimizer, epochs, device):
    best_macro_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        # NOUVEAU : On extrait x_dino ET x_geo
        for x_dino, x_geo, y in train_loader:
            x_dino = x_dino.to(device, dtype=torch.float32)
            x_geo = x_geo.to(device, dtype=torch.float32)
            y = y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_dino, x_geo)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            
        scheduler.step()

        # Évaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_dino, x_geo, y in val_loader:
                x_dino = x_dino.to(device, dtype=torch.float32)
                x_geo = x_geo.to(device, dtype=torch.float32)
                logits = model(x_dino, x_geo)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(y.numpy())

        val_macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model

# ============================================================
# 📊 FONCTIONS D'ÉVALUATION & WANDB
# ============================================================
def evaluate_predictions(all_targets, all_preds, num_classes):
    _, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(num_classes), zero_division=0
    )
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

    return f1_per_class, recall_per_class, macro_f1, weighted_f1, macro_recall

def compute_confidence_intervals(metrics_dict, noms_uniques, n_splits):
    print("\n📊 Calcul des intervalles de confiance finaux...")
    columns = ["Métrique / Espèce", "F1_Mean", "CI_Lower_95%", "CI_Upper_95%", "Écart-Type"]
    
    def get_ci_stats(values):
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        sem = st.sem(values)
        t_crit = st.t.ppf((1 + 0.95) / 2, n_splits - 1)
        margin = t_crit * sem if sem > 0 else 0
        return mean_val, max(0.0, mean_val - margin), min(1.0, mean_val + margin), std_val

    for model_name, state in metrics_dict.items():
        print(f"\n--- Logging WandB pour {model_name} ---")
        
        m_mean, m_low, m_high, m_std = get_ci_stats(state['macro_f1_array'])
        w_mean, w_low, w_high, w_std = get_ci_stats(state['weighted_f1_array'])
        w_rec_mean, m_rec_low, m_rec_high, m_rec_std = get_ci_stats(state['macro_recall_array'])

        data_global = [
            ["Macro F1 (Global)", m_mean, m_low, m_high, m_std],
            ["Weighted F1 (Global)", w_mean, w_low, w_high, w_std],
            ["Macro Recall (Global)", w_rec_mean, m_rec_low, m_rec_high, m_rec_std]
        ]
        wandb.log({f"{model_name}_Globaux": wandb.Table(data=data_global, columns=columns)})

        data_species = []
        for i in range(len(noms_uniques)):
            s_mean, s_low, s_high, s_std = get_ci_stats(state['f1_matrix'][i, :])
            data_species.append([noms_uniques[i], s_mean, s_low, s_high, s_std])
        wandb.log({f"{model_name}_F1_Especes": wandb.Table(data=data_species, columns=columns)})