import optuna
import torch
import torch.nn as nn
from Dataset import TreeLiDARDataset
from torch.utils.data import DataLoader
from pointnet2_cls_msg import get_model
from sklearn.model_selection import train_test_split
import os
import pandas as pd

NUM_CLASSES = 33
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("../../data/labels.csv")

noms_uniques = sorted(df['species'].unique()) 
mapping_species = {nom: i for i, nom in enumerate(noms_uniques)}
df['label_entier'] = df['species'].map(mapping_species)

tree_arrays = []
labels = []

for index, row in df.iterrows():
    laz_path = row['filename']
    label = row['label_entier']
    base_name = os.path.basename(laz_path).replace('.laz', '.pt').replace('.las', '.pt')
    full_path = os.path.join("FPS_32k", base_name)

    tree_arrays.append(full_path)
    labels.append(label)

print("Création du dataset", flush = True)

train_paths, val_paths, train_labels, val_labels = train_test_split(tree_arrays, labels, test_size = 0.1, random_state = 42, stratify = labels )


# --- ATTENTION ---
# Assure-toi d'avoir défini train_paths, train_labels, val_paths, val_labels 
# EN DEHORS de cette fonction pour ne pas recharger le CSV à chaque essai !

# Reprends la fonction make_collate_fn qu'on a vue plus tôt
def make_collate_fn(num_points):
    def uniform_size_collate(batch):
        batched_points, batched_labels = [], []
        for points, label in batch:
            n = points.shape[0]
            if n > num_points:
                idx = torch.randperm(n)[:num_points]
                points = points[idx]
            else:
                idx = torch.randint(0, n, (num_points,))
                points = points[idx]
            batched_points.append(points)
            batched_labels.append(label)
        return torch.stack(batched_points, 0), torch.tensor(batched_labels)
    return uniform_size_collate


def objective(trial):
    # 1. Définition des hyperparamètres
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    num_points = trial.suggest_categorical("num_points", [4096, 8192, 16384])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    
    # 2. Préparation des données (On injecte le num_points dynamiquement)
    train_loader = DataLoader(
        TreeLiDARDataset(train_paths, train_labels), 
        batch_size=32, # Tu peux monter à 64 sur la A100
        shuffle=True, 
        collate_fn=make_collate_fn(num_points),
        num_workers=16, pin_memory=True
    )
    val_loader = DataLoader(
        TreeLiDARDataset(val_paths, val_labels), 
        batch_size=32, 
        shuffle=False, 
        collate_fn=make_collate_fn(num_points),
        num_workers=4, pin_memory=True
    )

    # 3. Initialisation du Modèle et de l'Optimiseur
    model = get_model(NUM_CLASSES, normal_channel=False).to(device)
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    criterion = nn.CrossEntropyLoss()

    # 4. Boucle d'entraînement courte (15 à 20 époques suffisent pour comparer des hyperparamètres)
    EPOCHS = 15
    
    for epoch in range(EPOCHS):
        model.train()
        for points, target in train_loader:
            points, target = points.to(device), target.to(device)
            points = points.transpose(2, 1) # Format [Batch, 3, N] pour PointNet
            
            optimizer.zero_grad()
            # Selon ta version de PointNet, il renvoie souvent la prédiction et une matrice de transformation
            pred, trans_feat, _ = model(points) 
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
        # 5. Évaluation sur le set de validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for points, target in val_loader:
                points, target = points.to(device), target.to(device)
                points = points.transpose(2, 1)
                
                pred, _, _= model(points)
                pred_choice = pred.argmax(dim=1)
                correct += (pred_choice == target).sum().item()
                total += target.size(0)
                
        val_acc = correct / total
        
        # 6. La magie d'Optuna : Le Pruning (élagage)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            # Si le modèle est trop mauvais par rapport aux autres, on l'arrête tout de suite !
            raise optuna.exceptions.TrialPruned()
            
    # L'objectif est de maximiser cette valeur
    return val_acc

# ==========================================
# --- LANCEMENT DE L'ÉTUDE OPTUNA ---
# ==========================================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(), study_name = "First Try")
    print("Début de l'optimisation des hyperparamètres...")
    # n_trials = 30 signifie qu'on teste 30 combinaisons différentes
    study.optimize(objective, n_trials=30)
    
    print("\n--- MEILLEURE COMBINAISON TROUVÉE ---")
    print(f"Meilleure Accuracy : {study.best_value:.4f}")
    print("Paramètres :")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")