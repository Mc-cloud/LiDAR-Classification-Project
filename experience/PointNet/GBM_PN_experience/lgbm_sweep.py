import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import lightgbm as lgb
import numpy as np
import optuna

NUM_CLASSES = 33

print("Chargement des données...")

tab_feat = pd.read_csv("../../data/tree_features.csv")
labels_df = pd.read_csv("../../data/labels.csv")

full_data = pd.merge(tab_feat, labels_df[['filename', 'species']], on='filename')

noms_uniques = sorted(full_data['species'].unique())
mapping_species = {nom: i for i, nom in enumerate(noms_uniques)}

y = full_data['species'].map(mapping_species)

X = full_data.drop(columns=["filename", 'species', 'label_id'], errors="ignore")
X = X.select_dtypes(include=[np.number, bool])

train_df, val_df, train_labels, val_labels = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = lgb.Dataset(train_df, label=train_labels)
val_data = lgb.Dataset(val_df, label=val_labels, reference=train_data)

best_val_accuracy = 0.0

def objective(trial):
    """
    Fonction d'objectif pour Optuna. Remplace la fonction train() du sweep de WandB.
    """
    global best_val_accuracy

    # Définition de l'espace de recherche (remplace sweep_config)
    params = {
        'objective': 'multiclass',
        'num_class': NUM_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.05]),
        'num_leaves': trial.suggest_categorical('num_leaves', [31, 50, 63, 127]),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': 5,
        'lambda_l1': trial.suggest_categorical('lambda_l1', [0.0, 0.1, 0.5]),
        'lambda_l2': trial.suggest_categorical('lambda_l2', [0.0, 0.1, 0.5]),
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
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0) # Désactive les logs d'itération polluants de lgb
        ]
    )

    y_pred = gbm.predict(val_df)
    y_pred_max = np.argmax(y_pred, axis=1)
    
    acc = accuracy_score(val_labels, y_pred_max)

    # Si c'est le meilleur modèle jusqu'à présent, on exporte tout
    if acc > best_val_accuracy:
        best_val_accuracy = acc
        print(f"\n🌟 Nouveau record d'Accuracy: {acc:.4f}! Sauvegarde du modèle...")
        
        # 1. Sauvegarde du modèle
        gbm.save_model("best_lgbm_model.txt")
        
        # 2. Sauvegarde des meilleurs paramètres en JSON
        with open("best_lgbm_params.json", "w") as f:
            json.dump(params, f, indent=4)
        
        # 3. Calcul et sauvegarde des métriques détaillées
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, 
            y_pred_max, 
            labels=range(NUM_CLASSES), 
            zero_division=0
        )
        
        df_class_metrics = pd.DataFrame({
            "Espèce": noms_uniques[:NUM_CLASSES],
            "Précision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
        df_class_metrics.to_csv("best_lgbm_metrics_per_class.csv", index=False)

    return acc

if __name__ == "__main__":
    print("Démarrage de l'optimisation des hyperparamètres avec Optuna...")
    
    # Création de l'étude (direction='maximize' car on veut maximiser l'Accuracy)
    study = optuna.create_study(direction='maximize', study_name="LGBM-Hyperparameter-Tuning")
    
    # Lancement des 50 essais
    study.optimize(objective, n_trials=50)

    print("\n✅ Optimisation terminée !")
    print(f"Meilleure Accuracy trouvée : {study.best_value:.4f}")
    
    # Sauvegarde de l'historique complet des 50 essais dans un fichier CSV
    df_trials = study.trials_dataframe()
    df_trials.to_csv("optuna_study_history.csv", index=False)
    print("📊 Historique des essais sauvegardé dans 'optuna_study_history.csv'")