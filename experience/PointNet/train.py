import os
import json
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from Dataset import TreeLiDARDataset, PointCloudTransforms
from pointnet2_cls_msg import get_model, get_loss

# --- HYPERPARAMÈTRES ---
BATCH_SIZE = 16
EPOCHS = 60
NUM_POINTS = 16384
NUM_CLASSES = 33
p = 0.5
LEARNING_RATE = 0.0001

# --- SAUVEGARDE DE LA CONFIGURATION ---
config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_points": NUM_POINTS,
    "architecture": "PointNet++ MSG",
    "name": "Same amount of layers, more points at the beginning"
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
scaler = GradScaler('cuda')

print('Chargement des données', flush=True)

df = pd.read_csv("../../data/labels_split_complex.csv")

colonne_label = 'species'

noms_uniques = sorted(df[colonne_label].unique())
mapping_species = {nom : index for index, nom in enumerate(noms_uniques)}

df['label_entier'] = df[colonne_label].map(mapping_species)

train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'val'].copy()

class_counts = df.groupby('label_entier').size().sort_index().values
num_classes = len(class_counts)
total_samples = len(df)

weights = total_samples / (num_classes * class_counts)
raw_class_weights = torch.log1p((torch.FloatTensor(weights)).to(device))
class_weights = raw_class_weights / raw_class_weights.mean()

def get_path_labels(sub_df):
    paths = []
    labels = []

    for index, row in sub_df.iterrows():
        laz_path = row['filename']
        label = row['label_entier']

        base_name = os.path.basename(laz_path).replace('.laz', '.pt').replace('.las', '.pt')
        full_path = os.path.join("../../data/FPS_32k", base_name)

        paths.append(full_path)
        labels.append(label)
    
    return paths, labels

print("Création du dataset", flush = True)

train_paths, train_labels = get_path_labels(train_df)
val_paths, val_labels = get_path_labels(val_df)

train_class_counts = np.bincount(train_labels, minlength = NUM_CLASSES)
train_class_counts[train_class_counts == 0] = 1

sampler_weights = 1. / (train_class_counts**p)

samples_weights = np.array([sampler_weights[t] for t in train_labels])
samples_weights = torch.from_numpy(samples_weights).double()

sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

train_transforms = PointCloudTransforms(rotation=True, jitter=True, scale=False)

train_dataset = TreeLiDARDataset(train_paths, train_labels, num_points=NUM_POINTS, transform=train_transforms)
val_dataset = TreeLiDARDataset(val_paths, val_labels, transform=None, num_points=NUM_POINTS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True, drop_last=False, num_workers=16, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)

classifier = get_model(NUM_CLASSES, normal_channel=False).to(device)
criterion = get_loss(class_weights=class_weights).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("L'entrainement commence...")

best_val_macro_f1 = 0.0
training_history = [] # Liste pour stocker les métriques globales

for epoch in range(EPOCHS):
    print(f'epoch {epoch}')
    classifier.train()
    train_loss = 0.0
    train_correct = 0

    for batch_id, (points, target) in enumerate(train_loader):
        points, target = points.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            predictions, trans_feat, _ = classifier(points)
            loss = criterion(predictions, target, trans_feat)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_correct += predictions.argmax(1).eq(target).sum().item()

        if batch_id % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_id}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)
    
    classifier.eval()
    val_loss = 0.0
    val_correct = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (points, target) in enumerate(val_loader):
            points, target = points.to(device, non_blocking=True), target.to(device, non_blocking=True)

            with autocast('cuda'):
                pred, trans_feat, _ = classifier(points)
                loss = criterion(pred, target, trans_feat)

            val_loss += loss.item()

            pred_indices = pred.argmax(1)
            val_correct += pred_indices.eq(target).sum().item()

            all_preds.extend(pred_indices.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % 20 == 0:
                print(f"   Batch {i}/{len(val_loader)} | Acc actuelle: {val_correct/((i+1)*BATCH_SIZE):.3f}", flush=True)
        
    precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, 
            all_preds, 
            labels=range(NUM_CLASSES), 
            zero_division=0
    )
    current_macro_f1 = np.mean(f1)
        
    metrics = {
        'epoch': epoch,
        'lr': optimizer.param_groups[0]['lr'],
        'Loss/train': train_loss / len(train_loader),
        'Acc/train': train_correct / len(train_paths),
        'Loss/val': val_loss / len(val_loader),
        'Acc/val': val_correct / len(val_paths),
        'Macro_F1/val': current_macro_f1
    }

    # Sauvegarde dans l'historique et export CSV direct
    training_history.append(metrics)
    pd.DataFrame(training_history).to_csv("training_log.csv", index=False)

    scheduler.step()

    if current_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = current_macro_f1
        
        # 1. Sauvegarde des poids du modèle
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': best_val_macro_f1,
        }, "best_model.pth")
        print(f"⭐ Nouveau record ! Modèle sauvegardé (Macro F1: {best_val_macro_f1:.3f})", flush=True)

        # 2. Calcul et sauvegarde des métriques détaillées par classe
        cm = confusion_matrix(y_true=all_targets, y_pred=all_preds, labels=range(NUM_CLASSES))
        acc_per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-6)

        df_class_metrics = pd.DataFrame({
            "Espèce": noms_uniques[:NUM_CLASSES],
            "Précision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Accuracy": acc_per_class
        })
        df_class_metrics.to_csv("best_metrics_per_class.csv", index=False)
        print("📊 Tableau des métriques par classe mis à jour ('best_metrics_per_class.csv').")

    print(f"Epoch {epoch} | Macro F1 {metrics['Macro_F1/val']:.4f} | Train Acc: {metrics['Acc/train']:.3f} | Val_acc: {metrics['Acc/val']:.3f} | Loss_val: {metrics['Loss/val']:.4f}\n", flush=True)