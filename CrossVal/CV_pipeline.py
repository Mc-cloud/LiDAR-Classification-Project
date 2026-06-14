"""
Unified Cross-Validation Pipeline
===================================
Supports all classification methods in one single configurable run:
  - PointNet++        (end-to-end deep learning on raw point clouds)
  - DINO MLP          (multimodal: DINO embeddings + geometric features)
  - DINO SVM          (DINO embeddings → sklearn SVM)
  - DINO LogReg       (DINO embeddings → sklearn Logistic Regression)
  - LightGBM          (gradient boosting on tabular geometric features)

Usage
-----
Edit the CVConfig at the bottom, then run:

    python cv_pipeline.py

Results are written to:
  - cv_checkpoint.pt          resumable checkpoint (after every fold)
  - cv_predictions.csv        per-tree predictions for every fold and method
  - cv_metrics_summary.csv    mean / 95% CI / std per method, global + per species
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from CVutils import *

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cfg = CVConfig(
        methods=["DINO_MLP", "DINO_SVM", "DINO_LogReg"],  # add "PointNet", "LightGBM" as needed
        n_splits=25,
        labels_csv="data/train_data/labels.csv",
        dino_embeddings_pt="experience/DINO/dinov3_tree_embeddings.pt",
        dino_geo_npz="experience/DINO/dev_geometry_features.npz",
        lgbm_features_csv="data/tableau_features.csv",
        pointnet_pt_dir="PointNet/data/FPS_32k",
    )
    run_cv(cfg)