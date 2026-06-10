"""
train_knn.py
============
KNN classifier on PCF curves.

Distance between two trees = sum_i ( g_new(r_i) - g_neighbour(r_i) )^2
i.e. standard L2 distance on the PCF feature vector.

Input
-----
    pcf_dataset.csv   – output of compute_pcf_dataset.py

Output
------
    knn_model.joblib  – fitted pipeline (StandardScaler + KNN)
    knn_report.txt    – classification report

Usage
-----
    python train_knn.py --pcf pcf_dataset.csv
    python train_knn.py --pcf pcf_dataset.csv --k 5 --test-size 0.2
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib


def train(
    pcf_path:     str   = "pcf_dataset.csv",
    k:            int   = 5,
    test_size:    float = 0.2,
    cv_folds:     int   = 5,
    out_model:    str   = "knn_model.joblib",
    out_report:   str   = "knn_report.txt",
    random_state: int   = 42,
):
    # ---- load --------------------------------------------------------------
    print(f"[1/4] Loading: {pcf_path}")
    df = pd.read_csv(pcf_path)

    pcf_cols = [c for c in df.columns if c.startswith("pcf_t_")][:8]
    if not pcf_cols:
        raise ValueError("No pcf_t_* columns found. Run compute_pcf_dataset.py first.")
    if "species" not in df.columns:
        raise ValueError("No 'species' column found.")

    n_before = len(df)
    df = df.dropna(subset=pcf_cols)
    if len(df) < n_before:
        warnings.warn(f"Dropped {n_before - len(df)} rows with NaN PCF values.")

    X = df[pcf_cols].values.astype(float)
    y = df["species"].astype(str).values

    print(f"    {len(df)} trees  |  {len(pcf_cols)} radii  |  {len(np.unique(y))} species")

    print("\nClass distribution:")
    for cls, cnt in pd.Series(y).value_counts().items():
        print(f"  {cls:<45} {cnt:>5}")

    # ---- train / test split ------------------------------------------------
    print(f"\n[2/4] Train/test split ({int((1-test_size)*100)}% / {int(test_size*100)}%) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y,
    )
    print(f"    Train: {len(X_train)}   Test: {len(X_test)}")

    # ---- pipeline: StandardScaler → KNN ------------------------------------
    # StandardScaler is important: KNN is distance-based and PCF magnitudes
    # vary across radius bins.
    print(f"\n[3/4] Training KNN (k={k}) …")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(
            n_neighbors = k,
            metric      = "euclidean",
            n_jobs      = -1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    print(f"    {cv_folds}-fold cross-validation …")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds, scoring="accuracy", n_jobs=-1,
    )
    print(f"    CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ---- evaluate ----------------------------------------------------------
    print(f"\n[4/4] Test set evaluation …")
    y_pred   = pipeline.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"    Test accuracy: {accuracy:.3f}")

    classes = np.unique(y)
    report  = classification_report(y_test, y_pred, zero_division=0)
    conf    = confusion_matrix(y_test, y_pred, labels=classes)
    print("\nClassification report:\n", report)

    # ---- save model --------------------------------------------------------
    joblib.dump(pipeline, out_model)
    print(f"Model  → {out_model}")

    # ---- save report -------------------------------------------------------
    lines = [
        "KNN Species Classifier",
        "======================",
        f"PCF dataset  : {pcf_path}",
        f"Features     : {len(pcf_cols)} PCF radii (pcf_t_*)",
        f"Classes      : {len(classes)} species",
        f"Train / Test : {len(X_train)} / {len(X_test)}",
        f"k            : {k}",
        f"Distance     : L2 on PCF vector = sum_i(g_new(r_i) - g_k(r_i))^2",
        f"",
        f"CV accuracy ({cv_folds}-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        f"Test accuracy        : {accuracy:.4f}",
        f"",
        "Classification report:",
        report,
        "",
        "Confusion matrix (rows=true, cols=predicted):",
        f"Classes: {list(classes)}",
        str(conf),
    ]
    Path(out_report).write_text("\n".join(lines))
    print(f"Report → {out_report}")

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KNN classifier on PCF features to predict tree species."
    )
    parser.add_argument("--pcf",        default="data/pcf_dataset.csv",
                        help="PCF dataset CSV (default: pcf_dataset.csv)")
    parser.add_argument("--k",          default=5, type=int,
                        help="Number of KNN neighbours (default: 5)")
    parser.add_argument("--test-size",  default=0.2, type=float,
                        help="Test fraction (default: 0.2)")
    parser.add_argument("--cv-folds",   default=5, type=int,
                        help="Cross-validation folds (default: 5)")
    parser.add_argument("--out-model",  default="knn_model.joblib",
                        help="Output model path (default: knn_model.joblib)")
    parser.add_argument("--out-report", default="knn_report.txt",
                        help="Output report path (default: knn_report.txt)")
    args = parser.parse_args()

    train(
        pcf_path   = args.pcf,
        k          = args.k,
        test_size  = args.test_size,
        cv_folds   = args.cv_folds,
        out_model  = args.out_model,
        out_report = args.out_report,
    )