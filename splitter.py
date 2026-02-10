import pandas as pd
import os
from sklearn.model_selection import train_test_split


features_df = pd.read_csv("data/train_data/training_data_20k.csv") 

labels_df = pd.read_csv("data/train_data/labels.csv")                

print(f"Features (from laz): {len(features_df)}")
print(f"Labels (from csv):   {len(labels_df)}")

features_df['join_id'] = features_df['filename'].apply(
    lambda x: os.path.splitext(os.path.basename(x))[0]
)

labels_df['join_id'] = labels_df['filename'].apply(
    lambda x: os.path.splitext(os.path.basename(x))[0]
)

full_data = pd.merge(features_df, labels_df, on='join_id', how='inner')

print(f"Matched Trees: {len(full_data)}")

full_data['height_diff'] = abs(full_data['height'] - full_data['tree_H'])
print(f"Avg Height Diff: {full_data['height_diff'].mean():.2f}m")

clean_data = full_data[full_data['height_diff'] < 5.0].copy()
print(f"Trees after cleaning bad matches: {len(clean_data)}")

train_df, val_df = train_test_split(
    clean_data, 
    test_size=0.2, 
    random_state=42, 
    stratify=clean_data['species']
)

cols_to_keep = [
    'height', 'volume', 'crown_area', 'num_points', 'crown_diameter', 'point_density', 'dbh_approx', 'p10_height_rel', 'p50_height_rel', 'p90_height_rel',
    'species', 'genus'
]

train_df[cols_to_keep].to_csv("train_dataset_final.csv", index=False)
val_df[cols_to_keep].to_csv("val_dataset_final.csv", index=False)

print("Success! Created 'train_dataset_final.csv' and 'val_dataset_final.csv'")