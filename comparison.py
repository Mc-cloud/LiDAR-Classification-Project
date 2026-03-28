import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
folder_random = "processed_data"
folder_fps = "processed_data_fps"

# On prend le premier fichier disponible dans le dossier random
files = [f for f in os.listdir(folder_random) if f.endswith('.pt')]
if not files:
    print("Erreur : Aucun fichier .pt trouvé dans 'processed_data'")
    exit()

sample_file = "00071.pt" # Tu peux changer l'index pour voir un autre arbre

# --- Chargement des tenseurs ---
points_random = torch.load("processed_data/00070.pt").numpy()
points_fps = torch.load("processed_data_FPS/00070.pt").numpy()

print(f"Comparaison de l'arbre : {sample_file}")
print(f"Points par nuage : {points_random.shape[0]}")

# --- Visualisation ---
fig = plt.figure(figsize=(16, 8))

def plot_tree(ax, points, title, color_map='viridis'):
    # On utilise la coordonnée Z pour la couleur (plus facile à lire)
    z = points[:, 2]
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                    c=z, cmap=color_map, s=5, alpha=0.7)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Pour garder les proportions réelles
    ax.set_aspect('equal')
    # On enlève les axes pour un look plus propre
    ax.axis('off')

# Affichage Random
ax1 = fig.add_subplot(121, projection='3d')
plot_tree(ax1, points_random, f"RANDOM SAMPLING\n({sample_file})", 'Reds')

# Affichage FPS
ax2 = fig.add_subplot(122, projection='3d')
plot_tree(ax2, points_fps, f"FARTHEST POINT SAMPLING\n({sample_file})", 'Greens')

plt.tight_layout()
output_name = f"comparison_{sample_file.replace('.pt', '.png')}"
plt.savefig(output_name, dpi=300)
print(f"✅ Comparaison sauvegardée : {output_name}")