import torch
import numpy as np
import scipy.stats as st
import pandas as pd
import os

# --- 1. Configuration & Mapping ---
csv_path = 'train_data/labels.csv'
checkpoint_path = "cv_checkpoint.pt"
n_folds = 20

# Reproducing your exact species mapping logic
df_labels = pd.read_csv(csv_path)
noms_uniques = sorted(df_labels['species'].unique())

# --- 2. Load Checkpoint ---
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

def get_ci_stats(values, n_samples):
    """Calculates Mean, 95% CI, and Standard Deviation"""
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    sem = st.sem(values)
    t_crit = st.t.ppf((1 + 0.95) / 2, n_samples - 1)
    margin = t_crit * sem
    return {
        "mean": mean_val,
        "low": max(0.0, mean_val - margin),
        "high": min(1.0, mean_val + margin),
        "std": std_val
    }

# --- 3. Global Metrics Analysis ---
print(f"\n🌍 GLOBAL PERFORMANCE SUMMARY (First {n_folds} Folds)")
print("=" * 70)

global_metrics = {
    "Macro F1": state['macro_f1_array'][:n_folds],
    "Weighted F1": state['weighted_f1_array'][:n_folds],
    "Macro Recall": state['macro_recall_array'][:n_folds]
}

global_rows = []
for name, data in global_metrics.items():
    s = get_ci_stats(data, n_folds)
    global_rows.append([name, s['mean'], f"[{s['low']:.4f}, {s['high']:.4f}]", s['std']])

df_global = pd.DataFrame(global_rows, columns=["Metric", "Mean", "95% CI", "Std Dev"])
print(df_global.to_string(index=False))

# --- 4. Species-Level Analysis ---
print(f"\n📊 PER-SPECIES PERFORMANCE (n={n_folds} Folds)")
print("=" * 70)

f1_matrix = state['f1_matrix'][:, :n_folds]
recall_matrix = state['recall_matrix'][:, :n_folds]

species_rows = []
for i, name in enumerate(noms_uniques):
    f1 = get_ci_stats(f1_matrix[i, :], n_folds)
    rec = get_ci_stats(recall_matrix[i, :], n_folds)
    
    species_rows.append({
        "Species": name,
        "F1_Mean": f1['mean'],
        "F1_CI": f"[{f1['low']:.3f}, {f1['high']:.3f}]",
        "Rec_Mean": rec['mean'],
        "F1_Stability (Std)": f1['std']
    })

df_species = pd.DataFrame(species_rows).sort_values(by="F1_Mean", ascending=True)
print(df_species.to_string(index=False))

# --- 5. Diagnostic Warnings ---
print("\n" + "!" * 20 + " DIAGNOSTIC CHECK " + "!" * 20)
macro = df_global.loc[df_global['Metric'] == 'Macro F1', 'Mean'].values[0]
weighted = df_global.loc[df_global['Metric'] == 'Weighted F1', 'Mean'].values[0]

if (weighted - macro) > 0.05:
    print(f"⚠️ WARNING: Large Gap ({weighted-macro:.4f}) between Weighted and Macro F1.")
    print("   Your model is biased towards common species and failing on rare ones.")
    print("   This is the most common reason for a low benchmark score.")