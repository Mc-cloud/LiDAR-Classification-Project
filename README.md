# LiDAR Tree Species Classification

Classifying individual trees by species from airborne/terrestrial LiDAR point clouds (`.laz`/`.las`), using a mix of geometric feature engineering, a Pair Correlation Function (PCF) descriptor, gradient boosting (LightGBM), PointNet++ on raw point clouds, and a DINOv3-embedding-based multimodal MLP. A unified cross-validation pipeline compares all methods on the same folds.

## Project structure

```
.
├── data/                     # Datasets (raw point clouds, labels, extracted features)
│   ├── labels.csv
│   ├── pcf_dataset.csv
│   ├── tableau_features.csv
│   ├── train_data/           # Training .laz files + labels/feature CSVs
│   └── test_data/            # Test .laz files (empty placeholder, see below)
├── utils/
│   ├── feature_extraction.py # Extract geometric tree features from .laz -> tableau_features.csv
│   ├── projection2D.py       # Project point clouds to 2D multi-view images (for DINO)
│   └── convert_to_pt.py       # Farthest-point-sample + normalize point clouds -> .pt tensors
├── experience/
│   ├── pcf/                  # Pair Correlation Function pipeline + KNN classifier
│   ├── ml/                   # Mutual information / KNN / LightGBM experiments on geometric features
│   ├── PointNet/             # PointNet++ training, grid search, GBM-on-PointNet experiments
│   ├── DINO/                 # DINOv3 embedding extraction, MLP training, k-fold CV, inference
│   └── InformationTheory/    # Information-theoretic feature analysis notebook
├── CrossVal/
│   ├── CV_pipeline.py         # Unified cross-validation entry point (all methods, same folds)
│   ├── CVutils.py              # Models, training loops, config dataclass
│   └── results/                # Saved checkpoints and per-fold predictions
├── plots/                     # Confusion matrix, correlation matrix figures
├── first_visu.ipynb            # 3D point cloud visualization (PyVista) + DBH/trunk debugging
├── pyproject.toml / uv.lock    # Project dependencies (managed with uv)
└── test_predictions.csv        # Example output predictions
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management (Python 3.12 or 3.13).

```bash
uv sync
```

This installs the dependencies listed in `pyproject.toml`/`uv.lock`, including PyTorch, scikit-learn, LightGBM, transformers, laspy, etc.

### GPU

PointNet++ training and DINOv3 embedding extraction are designed to run on a CUDA GPU (`torch.cuda.is_available()` is checked, falling back to CPU). DINOv3-ViT7B is large — extracting embeddings on CPU is impractical.

## Data

Place raw `.laz`/`.las` tree point clouds and the corresponding `labels.csv` (columns: `treeID`, `species`, `genus`, `dataset`, `data_type`, `tree_H`, `filename`) under `data/train_data/`. `data/test_data/` is currently an empty placeholder — add your test `.laz` files there.

Pre-computed feature tables are included:
- `data/tableau_features.csv` — geometric features (height, crown volume/area, stem diameter, etc.) per tree, produced by `utils/feature_extraction.py`.
- `data/pcf_dataset.csv` — Pair Correlation Function curves per tree, produced by `experience/pcf/compute_pcf_dataset.py`.

## Pipeline overview

1. **Feature extraction** (`utils/feature_extraction.py`) — reads `.laz` files, computes geometric features (alpha-shape volume/area, robust DBH via RANSAC, trunk/crown split, crown ratio, etc.), writes `data/tableau_features.csv`.
2. **PCF computation** (`experience/pcf/compute_pcf_dataset.py`) — computes 3D pair correlation function curves per tree, writes `data/pcf_dataset.csv`.
3. **2D projection** (`utils/projection2D.py`) — projects each tree's point cloud into 5 multi-view 2D images (1 top-down + 4 side views) for the DINOv3 pipeline.
4. **Point cloud preprocessing** (`utils/convert_to_pt.py`) — farthest-point sampling to a fixed number of points, normalization, saved as `.pt` tensors for PointNet++.
5. **Model training**:
   - `experience/PointNet/train.py` — trains PointNet++ (MSG) on the `.pt` point clouds.
   - `experience/DINO/training/teacher_ext.py` — extracts DINOv3 embeddings from the projected images.
   - `experience/DINO/training/student.py` — trains an MLP classifier on DINOv3 embeddings.
   - `experience/PointNet/GBM_PN_experience/GBM.py` / `experience/ml/lgbm_PN.py` — LightGBM on tabular geometric features.
   - `experience/pcf/knn_pcf.py` — KNN classifier on PCF curves.
6. **Cross-validation** (`CrossVal/CV_pipeline.py`) — runs a stratified k-fold CV comparing PointNet++, DINO+MLP, DINO+SVM, DINO+LogReg, and LightGBM on the same folds, producing `cv_checkpoint.pt`, `cv_predictions.csv`, and `cv_metrics_summary.csv`.
7. **Inference** (`experience/DINO/predictions/infer.py`) — runs the trained DINO MLP model on test data to produce `predictions_soumission.csv`.

## Visualization

`first_visu.ipynb` uses PyVista to render 3D point clouds, visualize the trunk/crown split and estimated DBH (debugging aid for `feature_extraction.py`), and inspect the PCF dataset.

## Results

See `plots/Confusion Matrix.png` and `plots/correlationmat.png` for example evaluation outputs, and `test_predictions.csv` / `CrossVal/results/` for sample predictions and checkpoints from prior runs.

## Known issues / TODO

- Several scripts hardcode file paths (e.g. relative paths like `../../data/...`, `dataset/test`, `PointNet/data/FPS_32k`) that assume a specific working directory or directory layout — these need to be standardized or made configurable (see checklist).
