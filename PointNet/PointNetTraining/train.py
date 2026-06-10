import os
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


from Dataset import TreeLiDARDataset, PointCloudTransforms

from PointNetArch.pointnet2_cls_msg import get_model, get_loss

BATCH_SIZE = 16
EPOCHS = 60

scaler = GradScaler('cuda')

NUM_POINTS = 16384
NUM_CLASSES = 33
p = 0.5

LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

wandb.init(
    project = "PointNet-LiDAR",
    config = {"learning_rate": LEARNING_RATE,
    "epochs" : EPOCHS,
    "batch_size" : BATCH_SIZE,
    "num_points" : NUM_POINTS,
    "architecture" : "PointNet++ MSG"},
    name = "Same amount of layers, more points at the beginning")

print('Chargement des données', flush=True)

df = pd.read_csv("train_data/labels.csv")

df['H_bin'] = pd.qcut(df['tree_H'], q = 20, labels = False, duplicates = 'drop')
df['strat_key'] = df['species'].astype(str) + "_" + df['data_type'].astype(str) + "_H" + df["H_bin"].astype(str)

val_counts = df['strat_key'].value_counts()
rare_keys = val_counts[val_counts < 2].index

df_rare = df[df['strat_key'].isin(rare_keys)]
df_common = df[~df['strat_key'].isin(rare_keys)]

colonne_label = 'species'


noms_uniques = df[colonne_label].unique()
mapping_species = {nom : index for index, nom in enumerate(noms_uniques)}

df['label_entier'] = df[colonne_label].map(mapping_species)

class_counts = df.groupby('label_entier').size().sort_index().values
num_classes = len(class_counts)
total_samples = len(df)

weights = total_samples / (num_classes * class_counts)
raw_class_weights = torch.log1p((torch.FloatTensor(weights)).to(device))
class_weights = raw_class_weights/ raw_class_weights.mean()


tree_arrays = []
labels = []
groups = []

for index, row in df.iterrows():
    laz_path = row['filename']
    label = row['label_entier']
    base_name = os.path.basename(laz_path).replace('.laz', '.pt').replace('.las', '.pt')
    full_path = os.path.join("data/FPS_32k", base_name)

    tree_arrays.append(full_path)
    labels.append(label)

print(tree_arrays)

print("Création du dataset", flush = True)

tree_arrays_np = np.array(tree_arrays)
labels_np = np.array(labels)

train_idx, val_idx = train_test_split(
    df_common.index, 
    test_size=0.20, 
    stratify=df_common['strat_key'], 
    random_state=42
)

train_paths = tree_arrays_np[train_idx].tolist()
val_paths = tree_arrays_np[val_idx].tolist()
train_labels = labels_np[train_idx].tolist()
val_labels = labels_np[val_idx].tolist()

train_class_counts = np.bincount(train_labels, minlength = NUM_CLASSES)
train_class_counts[train_class_counts == 0] = 1

sampler_weights = 1./(train_class_counts**p)

samples_weights = np.array([sampler_weights[t] for t in train_labels])
samples_weights = torch.from_numpy(samples_weights).double()

sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(samples_weights), replacement = True)

train_transforms = PointCloudTransforms(rotation = True, jitter = True, scale = False)

train_dataset = TreeLiDARDataset(train_paths, train_labels, num_points = NUM_POINTS, transform = train_transforms)
val_dataset = TreeLiDARDataset(val_paths, val_labels, transform = None, num_points= NUM_POINTS)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, sampler = sampler, pin_memory = True, drop_last = False, num_workers = 16, persistent_workers = True, prefetch_factor=4)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, pin_memory= True, num_workers = 16)


classifier = get_model(NUM_CLASSES, normal_channel= False).to(device)
criterion = get_loss(class_weights = class_weights).to(device)
optimizer = optim.Adam(classifier.parameters(), lr = LEARNING_RATE, betas = (0.9, 0.999), weight_decay = 1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

print("L'entrainement commence...")


best_val_macro_f1 = 0.0
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
            wandb.log({"train/loss_step" : loss.item(),
                       "epoch" : epoch})
            print(f"Epoch {epoch} | Batch {batch_id}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)
    
    classifier.eval()
    val_loss = 0.0
    val_correct = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i,  (points, target) in enumerate(val_loader):
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
        'Loss/train' : train_loss / len(train_loader),
        'Acc/train' : train_correct /len(train_paths),
        'Loss/val' : val_loss /len(val_loader),
        'Acc/val' : val_correct/len(val_paths),
        'Macro_F1/val' : current_macro_f1,
        'epoch' : epoch,
        'lr' : optimizer.param_groups[0]['lr']
    }

    scheduler.step()

    wandb.log(metrics)

    if current_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = current_macro_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': best_val_macro_f1,
        }, "best_model.pth")
        
        # On l'envoie aussi sur W&B pour être sûr
        wandb.save("best_model.pth")
        print(f"⭐ Nouveau record ! Modèle sauvegardé (Macro F1: {best_val_macro_f1:.3f})", flush=True)

            # 1. Créer une table W&B pour un affichage détaillé
        columns = ["Espèce", "Précision", "Recall", "F1-Score"]
        data = []
        for i in range(NUM_CLASSES):
            data.append([noms_uniques[i], precision[i], recall[i], f1[i]])
        
        table_metrics = wandb.Table(data=data, columns=columns)
        
        cm = confusion_matrix(y_true = all_targets, y_pred = all_preds, labels = range(NUM_CLASSES))
        acc_per_class = cm.diagonal() / (cm.sum(axis = 1) + 1e-6)

        data_table = [[noms_uniques[i], acc_per_class[i]] for i in range(NUM_CLASSES)]
        table = wandb.Table(data = data_table, columns = ["Espèces", "Accuracy"])
        bar_chart = wandb.plot.bar(table, 'Espèces', 'Accuracy', title = "Accuracy par espèce")
        
        bar_f1 = wandb.plot.bar(table_metrics, "Espèce", "F1-Score", title="F1-Score par Espèce")


        wandb.log({
            "Charts/ Accuracy_par_espece" : bar_chart,
            "Table/ Métrique Globale" : table_metrics,
            "Score F-1 par espèce" : bar_f1,
        })

    print(f"Epoch {epoch} | Macro F1 {metrics["Macro_F1/val"]}| Train Acc: {metrics['Acc/train']:.3f} | Val_acc {metrics['Acc/val']} | Loss_val {metrics['Loss/val']}", flush = True)