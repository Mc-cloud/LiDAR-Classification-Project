import os
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_complex_split(df, test_size=0.20, random_state=42):
    """
    Split stratifié complexe basé sur : Espèce + Type de donnée + Hauteur.
    Les combinaisons rares sont automatiquement forcées dans le set d'entraînement.
    """
    print("Application du split stratifié COMPLEXE (Espèce + Origine + Hauteur)...")
    df = df.copy()
    
    # Création de la clé complexe
    df['H_bin'] = pd.qcut(df['tree_H'], q=20, labels=False, duplicates='drop')
    df['strat_key'] = df['species'].astype(str) + "_" + df['data_type'].astype(str) + "_H" + df["H_bin"].astype(str)

    # Isolation des cas rares pour éviter le crash du train_test_split
    val_counts = df['strat_key'].value_counts()
    rare_keys = val_counts[val_counts < 2].index

    df_common = df[~df['strat_key'].isin(rare_keys)]

    # Split uniquement sur les données communes
    train_idx, val_idx = train_test_split(
        df_common.index, 
        test_size=test_size, 
        stratify=df_common['strat_key'], 
        random_state=random_state
    )

    # Assignation
    df['split'] = 'train' # Par défaut (inclut les cas rares)
    df.loc[val_idx, 'split'] = 'val'

    return df.drop(columns=['H_bin', 'strat_key'])


def generate_simple_split(df, test_size=0.20, random_state=42):
    """
    Split stratifié simple basé UNIQUEMENT sur l'espèce.
    """
    print("Application du split stratifié SIMPLE (Espèce uniquement)...")
    df = df.copy()

    # Sécurité : vérifier s'il y a des espèces avec un seul arbre au total
    val_counts = df['species'].value_counts()
    rare_species = val_counts[val_counts < 2].index

    df_common = df[~df['species'].isin(rare_species)]

    # Split classique
    train_idx, val_idx = train_test_split(
        df_common.index,
        test_size=test_size,
        stratify=df_common['species'],
        random_state=random_state
    )

    # Assignation
    df['split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'

    return df


if __name__ == "__main__":
    # 1. Gestion des chemins (fonctionne même si lancé depuis la racine)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_CSV = os.path.join(BASE_DIR, "data", "labels.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "data", "labels_split.csv")

    # 2. Chargement des données
    df_labels = pd.read_csv(INPUT_CSV)

    # ---------------------------------------------------------
    # 3. CHOIX DE LA MÉTHODE DE SPLIT (Décommente celle que tu veux)
    # ---------------------------------------------------------
    
    #df_splitted = generate_complex_split(df_labels)
    df_splitted = generate_simple_split(df_labels)

    # ---------------------------------------------------------

    # 4. Sauvegarde
    df_splitted.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Fichier sauvegardé : {OUTPUT_CSV}")
    print(f"📈 Répartition -> Train: {len(df_splitted[df_splitted['split']=='train'])} | Val: {len(df_splitted[df_splitted['split']=='val'])}")