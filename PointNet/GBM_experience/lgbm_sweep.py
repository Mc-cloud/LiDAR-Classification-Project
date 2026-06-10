import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import lightgbm as lgb
import numpy as np

import wandb
from wandb.integration.lightgbm import wandb_callback


NUM_CLASSES = 33

sweep_config = {
    'method': 'bayes', # 'grid' teste tout, 'bayes' est plus intelligent et rapide
    'name': 'LGBM-Hyperparameter-Tuning',
    'metric': {
        'goal': 'maximize', 
        'name': 'final_val_accuracy'
    },
    'parameters': {
        'learning_rate': {'values': [0.005, 0.01, 0.05]},
        'num_leaves': {'values': [31, 50, 63, 127]},
        'feature_fraction': {'min': 0.5, 'max': 0.9},
        'bagging_fraction': {'min': 0.5, 'max': 0.9},
        'lambda_l1': {'values': [0.0, 0.1, 0.5]}, # Régularisation Lasso
        'lambda_l2': {'values': [0.0, 0.1, 0.5]}  # Régularisation Ridge
    }
}

print("Chargement des données...")

tab_feat = pd.read_csv("tree_features.csv")
labels_df = pd.read_csv("train_data/labels.csv")

full_data = pd.merge(tab_feat, labels_df[['filename', 'species']], on = 'filename')

noms_uniques = sorted(full_data['species'].unique())

mapping_species = {nom : i for i, nom in enumerate(noms_uniques)}

y = full_data['species'].map(mapping_species)

X = full_data.drop(columns = ["filename", 'species', 'label_id'], errors = "ignore")
X = X.select_dtypes(include = [np.number, bool])

train_df, val_df, train_labels, val_labels = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = lgb.Dataset(train_df, label = train_labels)
val_data = lgb.Dataset(val_df, label = val_labels, reference = train_data)

best_val_accuracy = 0.0

def train():
    global best_val_accuracy

    with wandb.init() as run:
        config = wandb.config

        params = {
            'objective' : 'multiclass',
            'num_class' : NUM_CLASSES,
            'metric' : 'multi_logloss',
            'boosting_type' : 'gbdt',
            'learning_rate' : config.learning_rate,
            'num_leaves' : config.num_leaves,
            'feature_fraction' : config.feature_fraction,
            'bagging_fraction' : config.bagging_fraction,
            'bagging_freq' : 5,
            'lambda_l1' : config.lambda_l1,
            'class_weight': 'balanced',
            'verbosity': -1,
            'n_jobs': 8
        }

        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                wandb_callback() # Enregistre l'évolution de la loss en direct
            ]
        )

        y_pred = gbm.predict(val_df)
        y_pred_max = np.argmax(y_pred, axis=1)
        
        acc = accuracy_score(val_labels, y_pred_max)
        report_dict = classification_report(val_labels, y_pred_max, zero_division=0, output_dict=True)

        if acc > best_val_accuracy:
            best_val_accuracy = acc
            print(f"\n🌟 Nouveau record d'Accuracy: {acc:.4f}! Sauvegarde du modèle...")
            
            # Sauvegarde locale
            model_path = "best_lgbm_model.txt"
            gbm.save_model(model_path)
            
            # Envoie le fichier physique sur le cloud WandB, lié à CET essai spécifique
            wandb.save(model_path)

        # Envoi du score final à wandb pour qu'il classe cet essai
        precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, 
                y_pred_max, 
                labels=range(NUM_CLASSES), 
                zero_division=0
                )
        
        columns = ["Espèce", "Précision", "Recall", "F1-Score"]
        data = []

        for i in range(NUM_CLASSES):
            data.append([noms_uniques[i], precision[i], recall[i], f1[i]])

        table_metrics = wandb.Table(data = data, columns = columns)
        
        wandb.log({
            "final_val_accuracy": acc,
            "final_val_macro_f1": report_dict['macro avg']['f1-score'],
            "final_val_weighted_f1": report_dict['weighted avg']['f1-score'],
            "Métrique Globale" : table_metrics
        })


if __name__ == "__main__":
    # Étape A : Créer le Sweep sur les serveurs wandb
    sweep_id = wandb.sweep(sweep_config, project="PointNet-LiDAR")
    
    print(f"Sweep ID créé : {sweep_id}")
    print("Démarrage de l'agent wandb...")
    
    # Étape B : Lancer un agent local qui va exécuter la fonction train() 50 fois
    wandb.agent(sweep_id, function=train, count=50)