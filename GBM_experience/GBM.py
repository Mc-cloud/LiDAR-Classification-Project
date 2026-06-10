import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import lightgbm as lgb
import numpy as np
import wandb
from wandb.integration.lightgbm import wandb_callback

NUM_CLASSES = 33

params = {
    'objective': 'multiclass',
    'num_class': NUM_CLASSES,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'feature_fraction': 0.57,
    'bagging_fraction': 0.59,
    'bagging_freq': 5,
    'class_weight': 'balanced',
    'verbosity': -1
}

wandb.init(
    project = "PointNet-LiDAR",
    config = params,
    name = "LightGBM on pointnet features"
)

# 1. Chargement
tab_feat = pd.read_csv("tree_features.csv")
labels_df = pd.read_csv('train_data/labels.csv')

tab_feat['filename'] = tab_feat['filename'].str.lstrip(' /')
labels_df['filename'] = labels_df['filename'].str.lstrip(' /')

full_data = pd.merge(tab_feat, labels_df[['filename', 'species']], on='filename')
# 2. ENCODAGE DES LABELS (Crucial pour LightGBM multiclass)
noms_uniques = sorted(full_data['species'].unique()) # Sorted pour garder le même ordre
mapping_species = {nom: i for i, nom in enumerate(noms_uniques)}
y = full_data['species'].map(mapping_species)

# 3. NETTOYAGE DES FEATURES
# On supprime les colonnes de texte (comme filename) pour ne garder que les stats
# Adapte le nom de la colonne si elle s'appelle différemment
X = full_data.drop(columns=['filename', 'species', 'label_id'], errors='ignore')
X = X.select_dtypes(include=[np.number, bool])

print(X.columns.tolist())

print("--- DEBUGGING SHAPES ---")
print(f"Features type: {type(X)}")
print(f"Features count/shape: {getattr(X, 'shape', len(X))}")
print(f"Labels count/shape: {getattr(y, 'shape', len(y))}")
print("------------------------")

# 4. SPLIT
train_df, val_df, train_labels, val_labels = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 5. DATASETS
train_data = lgb.Dataset(train_df, label=train_labels)
val_data = lgb.Dataset(val_df, label=val_labels, reference=train_data)



# 6. ENTRAÎNEMENT
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50), 
        lgb.log_evaluation(period=20),
        wandb_callback()
    ]
)

# 7. PRÉDICTION
y_pred = gbm.predict(val_df)
y_pred_max = np.argmax(y_pred, axis=1)

# 8. RÉSULTATS (On utilise val_labels ici !)
print("\n Résultats LightGBM")
# Correction de val_df -> val_labels
print(f"Accuracy : {accuracy_score(val_labels, y_pred_max):.4f}")
print(classification_report(val_labels, y_pred_max, target_names=noms_uniques, zero_division=0))


acc = accuracy_score(val_labels, y_pred_max)
report_dict = classification_report(val_labels, y_pred_max, target_names=noms_uniques, zero_division=0, output_dict=True)
precision, recall, f1, _ = precision_recall_fscore_support(
                val_df, 
                y_pred, 
                labels=range(NUM_CLASSES), 
                zero_division=0
                )


columns = ["Espèce", "Précision", "Recall", "F1-Score"]
data = []

for i in range(NUM_CLASSES):
    data.append([noms_uniques[i], precision[i], recall[i], f1[i]])

table_metrics = wandb.Table(data = data, columns = columns)

bar_f1 = wandb.plot.bar(table_metrics)

wandb.log({
    "Métrique Globale" : bar_f1
})

wandb.finish()

gbm.save_model('best_tree_model.txt')