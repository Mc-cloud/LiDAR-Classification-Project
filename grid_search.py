import itertools
import json
import os
import numpy as np
import pandas as pd  # 👈 Indispensable pour créer le tableau de résultats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def run_grid_search(
    features,
    labels_encoded,
    classes,
    out_dir,
    seed=42,
    val_ratio=0.20,
    epochs=40,
    batch_size=256,
    weight_decay=1e-4,
    view_dropout=0.15,
    feature_noise=0.01,
    device=None,
):
    if device is None:
        device = get_device()

    ensure_dir(out_dir)
    num_classes = len(classes)
    dino_dim = int(features.shape[-1])

    # ==========================================
    # ⚙️ 1. DÉFINITION DE LA GRILLE
    # ==========================================
    param_grid = {
        'lr': [1e-3, 3e-3],
        'hidden_dim': [512, 1024],
        'dropout': [0.35, 0.50],
        'sampler_power': [0.5, 1.0],
        'sampler_multiplier': [1.0, 2.0]
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n🔍 DÉMARRAGE DU GRID SEARCH")
    print(f"📊 Nombre de combinaisons à tester : {len(combinations)}")
    print(f"⚡ Device : {device}\n")

    # ==========================================
    # 📦 2. PRÉPARATION DES DONNÉES
    # ==========================================
    idx = np.arange(len(features))
    train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=labels_encoded)

    X_train, y_train = features[train_idx], labels_encoded[train_idx]
    X_val, y_val = features[val_idx], labels_encoded[val_idx]

    train_ds = DinoFeatureDataset(X_train, y_train, train=True, seed=seed, view_dropout=view_dropout, feature_noise=feature_noise)
    val_ds = DinoFeatureDataset(X_val, y_val, train=False, seed=seed, view_dropout=0.0, feature_noise=0.0)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_global_f1 = -1.0
    best_global_params = None

    # 📝 Liste pour stocker l'historique complet de TOUTES les combinaisons
    all_results_history = []

    # ==========================================
    # 🚀 3. BOUCLE DE RECHERCHE
    # ==========================================
    for i, params in enumerate(combinations, 1):
        print(f"{'-'*50}\n🔄 Test {i}/{len(combinations)} | Paramètres : {params}")
        
        # --- A. CRÉATION DU SAMPLER DYNAMIQUE ---
        counts = np.bincount(y_train, minlength=num_classes)
        counts[counts == 0] = 1 
        freq = counts / counts.sum()
        
        class_weights_sampler = (1.0 / freq) ** params['sampler_power']
        sample_weights = np.array([class_weights_sampler[y] for y in y_train])
        num_samples = int(len(y_train) * params['sampler_multiplier'])

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(), 
            num_samples=num_samples, 
            replacement=True
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

        # --- B. INITIALISATION DU MODÈLE ---
        model = DinoViewAttentionHead(
            dino_dim=dino_dim, 
            num_classes=num_classes, 
            hidden_dim=params['hidden_dim'], 
            dropout=params['dropout']
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

        best_local_f1 = 0.0
        corresponding_acc = 0.0

        # --- C. ENTRAÎNEMENT ---
        for epoch in range(1, epochs + 1):
            model.train()
            for x, y in train_loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device)
                
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                
            scheduler.step()

            # --- D. ÉVALUATION ---
            model.eval()
            all_true, all_pred = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32)
                    logits = model(x)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    all_pred.extend(pred.tolist())
                    all_true.extend(y.numpy().tolist())

            acc = accuracy_score(all_true, all_pred)
            macro_f1 = f1_score(all_true, all_pred, average="macro")
            
            if macro_f1 > best_local_f1:
                best_local_f1 = macro_f1
                corresponding_acc = acc

        print(f"🎯 Meilleur Macro-F1 pour cette configuration : {best_local_f1:.4f} (Accuracy: {corresponding_acc:.4f})")

        # 💾 AJOUT DANS L'HISTORIQUE (On fusionne les paramètres et les métriques obtenues)
        run_record = copy.deepcopy(params)
        run_record['val_macro_f1'] = best_local_f1
        run_record['val_accuracy'] = corresponding_acc
        run_record['grid_index'] = i
        all_results_history.append(run_record)

        # Sauvegarde intermédiaire à chaque étape pour ne rien perdre en cas de crash du serveur
        df_temp = pd.DataFrame(all_results_history)
        df_temp.to_csv(os.path.join(out_dir, "grid_search_results_partial.csv"), index=False)

        # Suivi du recordman global
        if best_local_f1 > best_global_f1:
            best_global_f1 = best_local_f1
            best_global_params = params
            print("🏆 NOUVEAU RECORD !")

    # ==========================================
    # 💾 4. EXPORTATION DES FICHIERS FINAUX
    # ==========================================
    print(f"\n{'='*50}")
    print(f"🏁 FIN DU GRID SEARCH")
    print(f"🥇 Meilleur Macro-F1 Global : {best_global_f1:.4f}")
    print(f"⚙️ Meilleurs paramètres : {best_global_params}")
    print(f"{'='*50}")

    # 1. Sauvegarde du tableau complet de tous les runs (Trié du meilleur au moins bon !)
    df_final = pd.DataFrame(all_results_history)
    df_final = df_final.sort_values(by="val_macro_f1", ascending=False)
    
    csv_file = os.path.join(out_dir, "grid_search_results.csv")
    df_final.to_csv(csv_file, index=False)
    print(f"✅ Tableau CSV complet sauvegardé sous : {csv_file}")

    # 2. Sauvegarde classique du fichier JSON du gagnant
    json_file = os.path.join(out_dir, "best_grid_params.json")
    with open(json_file, "w") as f:
        json.dump(best_global_params, f, indent=4)
    print(f"✅ Fichier JSON du gagnant sauvegardé sous : {json_file}")

    # Nettoyage du fichier partiel devenu inutile
    if os.path.exists(os.path.join(out_dir, "grid_search_results_partial.csv")):
        os.remove(os.path.join(out_dir, "grid_search_results_partial.csv"))

    return best_global_params