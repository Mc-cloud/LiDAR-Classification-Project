import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import lightgbm as lgb

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

# 1. Chargement
print("Chargement des données...")
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
X = full_data.drop(columns=['filename', 'species', 'label_id'], errors='ignore')
X = X.select_dtypes(include=[np.number, bool])

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
print("L'entraînement commence...")
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50), 
        lgb.log_evaluation(period=20)
    ]
)

# 7. PRÉDICTION
y_pred = gbm.predict(val_df)
y_pred_max = np.argmax(y_pred, axis=1)

# 8. RÉSULTATS 
acc = accuracy_score(val_labels, y_pred_max)
print(f"\nRésultats LightGBM")
print(f"Accuracy : {acc:.4f}")
print(classification_report(val_labels, y_pred_max, target_names=noms_uniques, zero_division=0))

report_dict = classification_report(val_labels, y_pred_max, target_names=noms_uniques, zero_division=0, output_dict=True)

# CORRECTION DU BUG ICI : val_labels au lieu de val_df, et y_pred_max au lieu de y_pred
precision, recall, f1, _ = precision_recall_fscore_support(
    val_labels, 
    y_pred_max, 
    labels=range(NUM_CLASSES), 
    zero_division=0
)

# --- 9. SAUVEGARDE EN LOCAL ---

# A. Sauvegarde du modèle
gbm.save_model('best_tree_model.txt')
print("✅ Modèle sauvegardé dans 'best_tree_model.txt'")

# B. Sauvegarde des paramètres et métriques globales en JSON
global_results = {
    "parameters": params,
    "metrics": {
        "accuracy": acc,
        "macro_f1": report_dict['macro avg']['f1-score'],
        "weighted_f1": report_dict['weighted avg']['f1-score']
    }
}
with open("lightgbm_training_summary.json", "w") as f:
    json.dump(global_results, f, indent=4)
print("✅ Résumé et paramètres sauvegardés dans 'lightgbm_training_summary.json'")

# C. Sauvegarde des métriques détaillées par classe en CSV
df_class_metrics = pd.DataFrame({
    "Espèce": noms_uniques[:NUM_CLASSES],
    "Précision": precision,
    "Recall": recall,
    "F1-Score": f1
})
df_class_metrics.to_csv("lightgbm_metrics_per_class.csv", index=False)
print("✅ Tableau des métriques par classe exporté dans 'lightgbm_metrics_per_class.csv'")