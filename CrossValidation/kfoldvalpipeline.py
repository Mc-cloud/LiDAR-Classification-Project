import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb
import lightgbm as lgb
import scipy.stats as st
from utils.kfoldutils import *

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import precision_recall_fscore_support

from Dataset import TreeLiDARDataset, PointCloudTransforms
from models.pointnet2_cls_msg import get_model, get_loss

def main():
    CSV_PATH = 'data/train_data/labels.csv'
    BATCH_SIZE = 64
    EPOCHS_PN = 50
    NUM_POINTS = 16384

    NUM_CLASSES = 33
    N_SPLITS = 25
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    CHECKPOINT_FILE = "cv_checkpoint.pt"

    wandb.init(project="PointNet-LiDAR-CV", name="End-to-End KFold")

    tree_paths, labels, spatial_groups, noms_uniques = prepare_spatial_data(CSV_PATH)

    if os.path.exists(CHECKPOINT_FILE):
        print("🔄 Sauvegarde détectée ! Restauration de l'état précédent...")
        state = torch.load(CHECKPOINT_FILE, weights_only=False)
        
        f1_matrix = state['f1_matrix']
        recall_matrix = state['recall_matrix'] # NOUVEAU
        macro_f1_array = state['macro_f1_array']
        weighted_f1_array = state['weighted_f1_array']
        macro_recall_array = state['macro_recall_array'] # NOUVEAU
        start_fold = state['last_completed_fold'] + 1
        
        print(f"⏩ Reprise directe au Fold {start_fold + 1}")
    else:
        print("🆕 Nouvelle exécution, initialisation des matrices...")
        f1_matrix = np.zeros((NUM_CLASSES, N_SPLITS))
        recall_matrix = np.zeros((NUM_CLASSES, N_SPLITS)) # NOUVEAU
        macro_f1_array = np.zeros(N_SPLITS)
        weighted_f1_array = np.zeros(N_SPLITS)
        macro_recall_array = np.zeros(N_SPLITS) # NOUVEAU
        start_fold = 0


    #sgkf = StratifiedKFold(n_splits = N_SPLITS, shuffle = True, random_state = 42)
    sgkf = StratifiedKFold(n_splits = N_SPLITS, shuffle = True, random_state = 42)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(tree_paths, labels)):
        
        if fold < start_fold:
            print(f"⏭️ Fold {fold + 1}/{N_SPLITS} ignoré (déjà calculé).")
            continue    

        print(f"Démarrage du fold {fold + 1}/{N_SPLITS}")

        train_loader, ext_train_loader, ext_val_loader = build_dataloader(tree_paths[train_idx],
                                                                tree_paths[val_idx],
                                                                labels[train_idx],
                                                                labels[val_idx],
                                                                NUM_POINTS, 
                                                                BATCH_SIZE
                                                                        )
        
        model = get_model(NUM_CLASSES, normal_channel = False).to(device)
        optimizer = optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9, 0.999), weight_decay=1e-4)
        criterion = get_loss(class_weights = None).to(device)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor = 0.7, min_lr = 1e-6)
        
        model = train_pointnet(model, train_loader, ext_val_loader, optimizer, criterion, EPOCHS_PN, device)

        f1_per_class, recall_per_class, macro_f1, weighted_f1, macro_recall = evaluate_pointnet(model, ext_val_loader, NUM_CLASSES, device)

        f1_matrix[:, fold] = f1_per_class
        macro_f1_array[fold] = macro_f1
        weighted_f1_array[fold] = weighted_f1
        macro_recall_array[fold] = macro_recall
        recall_matrix[:, fold] = recall_per_class

        print(f"✅ Fold {fold + 1} terminé | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

        torch.save({
            'last_completed_fold': fold,
            'f1_matrix': f1_matrix,
            'recall_matrix': recall_matrix,
            'macro_f1_array': macro_f1_array,
            'weighted_f1_array': weighted_f1_array,
            'macro_recall_array': macro_recall_array
        }, CHECKPOINT_FILE)
        
        print("💾 État sauvegardé sur le disque.")

        del model, optimizer, train_loader, ext_train_loader, ext_val_loader
        torch.cuda.empty_cache()

    
    compute_confidence_intervals(f1_matrix, recall_matrix, macro_f1_array, weighted_f1_array, macro_recall_array, noms_uniques, N_SPLITS)
    wandb.finish()

if __name__ == "__main__":
    main()