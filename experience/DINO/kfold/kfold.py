from experience.DINO.kfold.kfoldutils import *
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset

def main():
    # --- CONFIGURATION ---
    FEATURES_PT = '../dinov3_tree_embeddings.pt' 
    GEO_FEATURES_NPZ = '../dev_geometry_features.npz' # 👈 VÉRIFIE QUE LE FICHIER S'APPELLE BIEN COMME ÇA
    CHECKPOINT_FILE = "../dino_cv_checkpoint_GEO.pt"
    PREDICTIONS_CSV = "cv_predictions_detaillees_GEO.csv"
    N_SPLITS = 25
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    MODELS_TO_RUN = ["MLP"]

    # --- 1. CHARGEMENT DES DONNÉES DINO ---
    print(f"⏳ Chargement des embeddings DINO depuis {FEATURES_PT}...")
    data = torch.load(FEATURES_PT, map_location='cpu', weights_only=False)
    
    features_raw = data["embeddings"] 
    if torch.is_tensor(features_raw):
        features = features_raw.float().cpu().numpy()
    else:
        features = np.array(features_raw).astype(np.float32)

    labels = np.array(data["labels"])
    noms_uniques = np.array(data["class_names"])
    N_ARBRES = len(labels)
    tree_ids = np.arange(N_ARBRES).astype(str)
    
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    noms_uniques = encoder.classes_
    NUM_CLASSES = len(noms_uniques)

    # --- 2. CHARGEMENT DES DONNÉES GÉOMÉTRIQUES ---
    print(f"⏳ Chargement de la géométrie depuis {GEO_FEATURES_NPZ}...")
    geo_data = np.load(GEO_FEATURES_NPZ, allow_pickle=True)
    geo_features = geo_data["geo_features"].astype(np.float32)
    
    if len(geo_features) != N_ARBRES:
        raise ValueError(f"Désynchronisation : {N_ARBRES} arbres DINO mais {len(geo_features)} arbres géométriques !")

    GEO_DIM = geo_features.shape[1]
    print(f"📐 Géométrie chargée : {GEO_DIM} features par arbre.")

    # --- INITIALISATION OU RESTAURATION ---
    if os.path.exists(CHECKPOINT_FILE):
        print("🔄 Restauration de l'état précédent...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu', weights_only=False)
        metrics_dict = checkpoint['metrics_dict']
        start_fold = checkpoint['last_completed_fold'] + 1
        predictions_list = checkpoint.get('predictions_list', []) 
    else:
        print("🆕 Initialisation des matrices...")
        metrics_dict = {}
        for m in MODELS_TO_RUN:
            metrics_dict[m] = {
                'f1_matrix': np.zeros((NUM_CLASSES, N_SPLITS)),
                'recall_matrix': np.zeros((NUM_CLASSES, N_SPLITS)),
                'macro_f1_array': np.zeros(N_SPLITS),
                'weighted_f1_array': np.zeros(N_SPLITS),
                'macro_recall_array': np.zeros(N_SPLITS)
            }
        start_fold = 0
        predictions_list = []

    sgkf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # ============================================================
    # 🔄 DÉMARRAGE DU K-FOLD
    # ============================================================
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(features, labels)):
        
        if fold < start_fold:
            print(f"⏭️ Fold {fold + 1}/{N_SPLITS} ignoré.")
            continue    

        print(f"\n{'-'*40}\n🚀 Démarrage du fold {fold + 1}/{N_SPLITS}\n{'-'*40}")

        y_train, y_val = labels[train_idx], labels[val_idx]
        svm_preds, lr_preds, mlp_preds = None, None, None

        # ----------------------------------------------------
        # 1. MODÈLE PYTORCH MULTIMODAL (TreeStudent DINO + GEO)
        # ----------------------------------------------------
        if "MLP" in MODELS_TO_RUN:
            print("👉 Entraînement du MLP (PyTorch)...")
            
            # Extraction DINO
            X_train_dino = features[train_idx]
            X_val_dino = features[val_idx]
            
            # Extraction GEO + Standardisation indispensable pour la géométrie
            geo_scaler = StandardScaler()
            X_train_geo = geo_scaler.fit_transform(geo_features[train_idx])
            X_val_geo = geo_scaler.transform(geo_features[val_idx])
            
            # PRÉPARATION PYTORCH
            X_train_dino_pt = torch.from_numpy(X_train_dino).float()
            X_val_dino_pt = torch.from_numpy(X_val_dino).float()
            X_train_geo_pt = torch.from_numpy(X_train_geo).float()
            X_val_geo_pt = torch.from_numpy(X_val_geo).float()
            y_train_pt = torch.from_numpy(y_train).long() 
            y_val_pt = torch.from_numpy(y_val).long()

            # LE SAMPLER
            class_counts = np.bincount(y_train)
            class_counts[class_counts == 0] = 1 
            class_weights = 1.0 / class_counts
            sample_weights = np.array([class_weights[t] for t in y_train])
            sample_weights_pt = torch.from_numpy(sample_weights).double()
            
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights_pt, 
                num_samples=len(sample_weights_pt), 
                replacement=True
            )

            # CRÉATION DES DATASETS AVEC 3 ÉLÉMENTS (DINO, GEO, LABEL)
            train_ds = TensorDataset(X_train_dino_pt, X_train_geo_pt, y_train_pt)
            val_ds = TensorDataset(X_val_dino_pt, X_val_geo_pt, y_val_pt)

            train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

            # ENTRAÎNEMENT
            model = TreeStudent(dino_dim=4096, geo_dim=GEO_DIM, num_classes=NUM_CLASSES).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            model = train_mlp(model, train_loader, val_loader, optimizer, epochs=40, device=device)

            # ÉVALUATION
            model.eval()
            mlp_preds_list = []
            with torch.no_grad():
                for x_d, x_g, _ in val_loader:
                    x_d, x_g = x_d.to(device), x_g.to(device) 
                    mlp_preds_list.extend(model(x_d, x_g).argmax(dim=1).cpu().numpy())
            
            mlp_preds = np.array(mlp_preds_list)

            # SAUVEGARDE DES MÉTRIQUES
            f1_pc, rec_pc, mac_f1, w_f1, mac_rec = evaluate_predictions(y_val, mlp_preds, NUM_CLASSES)
            metrics_dict["MLP"]['f1_matrix'][:, fold] = f1_pc
            metrics_dict["MLP"]['recall_matrix'][:, fold] = rec_pc
            metrics_dict["MLP"]['macro_f1_array'][fold] = mac_f1
            metrics_dict["MLP"]['weighted_f1_array'][fold] = w_f1
            metrics_dict["MLP"]['macro_recall_array'][fold] = mac_rec
            print(f"✅ MLP terminé | Macro F1: {mac_f1:.4f}")

        # ----------------------------------------------------
        # 2. MODÈLES CLASSIQUES (SVM & LogReg) - DINO UNIQUEMENT
        # ----------------------------------------------------
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(features[train_idx])
        X_val_flat = scaler.transform(features[val_idx])

        if "SVM" in MODELS_TO_RUN:
            print("👉 Entraînement du SVM (RBF)...")
            svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
            svm.fit(X_train_flat, y_train)
            svm_preds = svm.predict(X_val_flat)
            
            f1_pc, rec_pc, mac_f1, w_f1, mac_rec = evaluate_predictions(y_val, svm_preds, NUM_CLASSES)
            metrics_dict["SVM"]['f1_matrix'][:, fold] = f1_pc
            metrics_dict["SVM"]['recall_matrix'][:, fold] = rec_pc
            metrics_dict["SVM"]['macro_f1_array'][fold] = mac_f1
            metrics_dict["SVM"]['weighted_f1_array'][fold] = w_f1
            metrics_dict["SVM"]['macro_recall_array'][fold] = mac_rec
            print(f"✅ SVM terminé | Macro F1: {mac_f1:.4f}")

        if "LogReg" in MODELS_TO_RUN:
            print("👉 Entraînement de la Régression Logistique...")
            lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
            lr.fit(X_train_flat, y_train)
            lr_preds = lr.predict(X_val_flat)
            
            f1_pc, rec_pc, mac_f1, w_f1, mac_rec = evaluate_predictions(y_val, lr_preds, NUM_CLASSES)
            metrics_dict["LogReg"]['f1_matrix'][:, fold] = f1_pc
            metrics_dict["LogReg"]['recall_matrix'][:, fold] = rec_pc
            metrics_dict["LogReg"]['macro_f1_array'][fold] = mac_f1
            metrics_dict["LogReg"]['weighted_f1_array'][fold] = w_f1
            metrics_dict["LogReg"]['macro_recall_array'][fold] = mac_rec
            print(f"✅ LogReg terminée | Macro F1: {mac_f1:.4f}")

        # ============================================================
        # ENREGISTREMENT DES PRÉDICTIONS DE CE FOLD
        # ============================================================
        fold_results = {
            'tree_id': tree_ids[val_idx],
            'fold': fold + 1,
            'true_species': noms_uniques[y_val]
        }
        
        if "SVM" in MODELS_TO_RUN:
            fold_results['pred_SVM'] = noms_uniques[svm_preds]
        if "LogReg" in MODELS_TO_RUN:
            fold_results['pred_LogReg'] = noms_uniques[lr_preds]
        if "MLP" in MODELS_TO_RUN:
            fold_results['pred_MLP'] = noms_uniques[mlp_preds]

        predictions_list.append(pd.DataFrame(fold_results))

        # --- SAUVEGARDE DU CHECKPOINT ---
        torch.save({
            'last_completed_fold': fold,
            'metrics_dict': metrics_dict,
            'predictions_list': predictions_list
        }, CHECKPOINT_FILE)
        
        print("💾 État sauvegardé sur le disque.")

    # ============================================================
    # 🏁 FIN DU K-FOLD : EXPORTATION DES PRÉDICTIONS
    # ============================================================
    print("\n💾 Génération du fichier CSV global des prédictions...")
    final_predictions_df = pd.concat(predictions_list, ignore_index=True)
    final_predictions_df = final_predictions_df.sort_values(by=['fold', 'tree_id'])
    final_predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"✅ Fichier sauvegardé avec succès : {PREDICTIONS_CSV}")

    compute_confidence_intervals(metrics_dict, noms_uniques, N_SPLITS)

if __name__ == "__main__":
    main()