import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import wandb



from Dataset import TreeLiDARDataset, PointCloudTransforms

from models.pointnet2_cls_msg import get_model, get_loss

BATCH_SIZE = 16
EPOCHS = 100

NUM_POINTS = 16384
NUM_CLASSES = 33

LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

wandb.init(
    project = "PointNet-LiDAR",
    config = {"learning_rate": LEARNING_RATE,
    "epochs" : EPOCHS,
    "batch_size" : BATCH_SIZE,
    "num_points" : NUM_POINTS,
    "architecture" : "PointNet++ MSG"})

print('Chargement des données', flush=True)

df = pd.read_csv("train_data/labels.csv")

colonne_label = 'species'

noms_uniques = df[colonne_label].unique()
mapping_species = {nom : index for index, nom in enumerate(noms_uniques)}

df['label_entier'] = df[colonne_label].map(mapping_species)

class_counts = df.groupby('label_entier').size().sort_index().values
num_classes = len(class_counts)
total_samples = len(df)

weights = total_samples / (num_classes * class_counts)
class_weights = torch.sqrt(torch.FloatTensor(weights).to(device))

tree_arrays = []
labels = []

for index, row in df.iterrows():
    laz_path = row['filename']
    label = row['label_entier']
    base_name = os.path.basename(laz_path).replace('.laz', '.pt').replace('.las', '.pt')
    full_path = os.path.join("processed_data_FPS", base_name)

    tree_arrays.append(full_path)
    labels.append(label)

print("Création du dataset", flush = True)

train_paths, val_paths, train_labels, val_labels = train_test_split(tree_arrays, labels, test_size = 0.1, random_state = 42, stratify = labels )

train_transforms = PointCloudTransforms(rotation = True, jitter = True, scale = True)

train_dataset = TreeLiDARDataset(train_paths, train_labels, transform = train_transforms)
val_dataset = TreeLiDARDataset(val_paths, val_labels, transform = None)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, drop_last = True, num_workers = 8)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle = False, pin_memory= True, num_workers = 8)


classifier = get_model(NUM_CLASSES, normal_channel= False).to(device)
criterion = get_loss(class_weights = class_weights, smoothing = 0.1).to(device)
optimizer = optim.Adam(classifier.parameters(), lr = LEARNING_RATE, betas = (0.9, 0.999), weight_decay = 1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor = 0.7, min_lr = 1e-6)

print("L'entrainement commence...")


best_val_acc = 0.0
for epoch in range(EPOCHS):
    print(f'epoch {epoch}')
    classifier.train()
    train_loss = 0.0
    train_correct = 0

    for batch_id, (points, target) in enumerate(train_loader):

        
        points, target = points.to(device), target.to(device)

        optimizer.zero_grad()

        points = points.transpose(2, 1)
        predictions, trans_feat = classifier(points)

        loss, loss_cls, loss_reg = criterion(predictions, target, trans_feat)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += predictions.argmax(1).eq(target).sum().item()

        if batch_id % 10 == 0:
            wandb.log({"train/loss_step" : loss.item(),
                       "train/loss_classification" : loss_cls.item(),
                       "train/loss_regularization" : loss_reg.item(),
                       "train/reg_ratio" : (loss_reg.item()*0.001/loss.item())*100,
                       "epoch" : epoch})
            print(f"Epoch {epoch} | Batch {batch_id}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)
    
    classifier.eval()
    val_loss = 0.0
    val_correct = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i,  (points, target) in enumerate(val_loader):
            points, target = points.to(device), target.to(device)
            points = points.transpose(2,1)
            pred, trans_feat = classifier(points)
            loss, loss_cls, loss_reg = criterion(pred, target, trans_feat)

            val_loss += loss.item()

            pred_indices = pred.argmax(1)
            val_correct += pred_indices.eq(target).sum().item()

            all_preds.extend(pred_indices.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % 5 == 0:
                print(f"   Batch {i}/{len(val_loader)} | Acc actuelle: {val_correct/((i+1)*BATCH_SIZE):.3f}", flush=True)
        
    metrics = {
        'Loss/train' : train_loss / len(train_loader),
        'Acc/train' : train_correct /len(train_paths),
        'Loss/val' : val_loss /len(val_loader),
        'Acc/val' : val_correct/len(val_paths),
        'epoch' : epoch,
        'lr' : optimizer.param_groups[0]['lr']
    }

    scheduler.step(metrics['Loss/val'])

    if metrics['Acc/val'] > best_val_acc:
        best_val_acc = metrics['Acc/val']
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': best_val_acc,
        }, "best_model.pth")
        
        # On l'envoie aussi sur W&B pour être sûr
        wandb.save("best_model.pth")
        print(f"⭐ Nouveau record ! Modèle sauvegardé (Acc: {best_val_acc:.3f})", flush=True)

    
    cm = confusion_matrix(y_true = all_targets, y_pred = all_preds, labels = range(NUM_CLASSES))
    acc_per_class = cm.diagonal() / (cm.sum(axis = 1) + 1e-6)

    data_table = [[noms_uniques[i], acc_per_class[i]] for i in range(NUM_CLASSES)]
    table = wandb.Table(data = data_table, columns = ["Espèces", "Accuracy"])
    bar_chart = wandb.plot.bar(table, 'Espèces', 'Accuracy', title = "Accuracy par espèce")
    

    wandb.log({
        **metrics,
        "Charts/ Accuracy_par_espece" : bar_chart
    })
    print(f"Epoch {epoch} | Train Acc: {metrics['Acc/train']:.3f} | Val_acc {metrics['Acc/val']} | Loss_val {metrics['Loss/val']}", flush = True)