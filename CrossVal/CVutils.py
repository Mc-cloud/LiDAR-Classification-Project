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


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class CVConfig:
    """Central configuration object. Edit this to choose methods and paths."""

    # ── What to run ──────────────────────────────────────────────────────────
    methods: list[str] = field(default_factory=lambda: [
        "PointNet",    # end-to-end deep learning on raw point clouds
        "DINO_MLP",    # multimodal DINO + geometry (PyTorch)
        "DINO_SVM",    # DINO embeddings → sklearn SVM
        "DINO_LogReg", # DINO embeddings → sklearn Logistic Regression
        "LightGBM",    # gradient boosting on tabular geometric features
    ])

    # ── K-Fold ────────────────────────────────────────────────────────────────
    n_splits: int = 25
    random_state: int = 42

    # ── Shared ───────────────────────────────────────────────────────────────
    labels_csv: str = "../data/train_data/labels.csv"
    checkpoint_file: str = "cv_checkpoint.pt"
    predictions_csv: str = "cv_predictions.csv"
    metrics_csv: str = "cv_metrics_summary.csv"
    num_classes: int = 33
    device: str = "auto"   # "auto" | "cuda" | "cpu"

    # ── PointNet++ ───────────────────────────────────────────────────────────
    pointnet_pt_dir: str = "../PointNet/data/FPS_32k"
    pointnet_epochs: int = 50
    pointnet_batch_size: int = 64
    pointnet_num_points: int = 16384

    # ── DINO ─────────────────────────────────────────────────────────────────
    dino_embeddings_pt: str = "../experience/DINO/dinov3_tree_embeddings.pt"
    dino_geo_npz: str = "../experience/DINO/dev_geometry_features.npz"
    dino_mlp_epochs: int = 40
    dino_mlp_batch_size: int = 128

    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgbm_features_csv: str = "../data/tableau_features.csv"
    lgbm_num_boost_round: int = 2000
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.57,
        "bagging_fraction": 0.59,
        "bagging_freq": 5,
        "class_weight": "balanced",
        "verbosity": -1,
    })


# ============================================================
# ARCHITECTURES
# ============================================================

class TreeStudentMultimodal(nn.Module):
    """DINO + geometry fusion MLP."""

    def __init__(self, dino_dim: int, geo_dim: int, num_classes: int):
        super().__init__()
        geo_hidden = 128
        self.geo_net = nn.Sequential(
            nn.LayerNorm(geo_dim),
            nn.Linear(geo_dim, geo_hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(geo_hidden, geo_hidden),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(dino_dim + geo_hidden, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_dino: torch.Tensor, x_geo: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([x_dino, self.geo_net(x_geo)], dim=1))


# ============================================================
# UNIFIED PYTORCH TRAINING LOOP
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    forward_fn: Callable,
    loss_fn: Callable,
    scheduler=None,
) -> nn.Module:
    """
    Generic training loop with best-macro-F1 model selection.

    The two callables decouple the loop from any specific model signature:

      forward_fn(model, batch, device) -> (logits, targets)
          Unpacks a batch however the model needs and returns raw logits
          plus the ground-truth labels as a 1-D LongTensor.

      loss_fn(logits, targets, model_output) -> scalar Tensor
          Computes the loss. Receives the full model output (tuple or tensor)
          so PointNet's regularisation term (trans_feat) can be used here
          without leaking PointNet-specific logic into the loop itself.

    Examples
    --------
    # DINO MLP  (batch = (x_dino, x_geo, y))
    def dino_forward(model, batch, device):
        x_d, x_g, y = batch
        logits = model(x_d.to(device), x_g.to(device))
        return logits, y.to(device)

    def dino_loss(logits, targets, _model_output):
        return F.cross_entropy(logits, targets)

    # PointNet++ (batch = (points, y))
    def pn_forward(model, batch, device):
        pts, y = batch
        out = model(pts.to(device))   # returns (logits, trans_feat, feat)
        return out[0], y.to(device)

    def pn_loss(logits, targets, model_output):
        return criterion(logits, targets, model_output[1])  # uses trans_feat
    """
    best_f1 = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            logits, targets = forward_fn(model, batch, device)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # ── Validation pass ────────────────────────────────────────────────
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, targets = forward_fn(model, batch, device)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


# ============================================================
# DATA HELPERS
# ============================================================

def _weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes)
    counts[counts == 0] = 1
    w = 1.0 / counts
    sample_w = torch.from_numpy(np.array([w[t] for t in labels])).double()
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


def load_dino_data(cfg: CVConfig):
    data = torch.load(cfg.dino_embeddings_pt, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]
    features = (embeddings.float().cpu().numpy() if torch.is_tensor(embeddings)
                else np.array(embeddings, dtype=np.float32))
    enc = LabelEncoder()
    labels = enc.fit_transform(np.array(data["labels"]))
    geo = None
    if os.path.exists(cfg.dino_geo_npz):
        geo_data = np.load(cfg.dino_geo_npz, allow_pickle=True)
        geo = geo_data["geo_features"].astype(np.float32)
        if len(geo) != len(labels):
            raise ValueError(f"DINO/geo size mismatch: {len(labels)} vs {len(geo)}")
    return features, labels, enc.classes_, geo


def load_pointnet_data(cfg: CVConfig):
    df = pd.read_csv(cfg.labels_csv)
    enc = LabelEncoder()
    labels = enc.fit_transform(df["species"].values)
    pt_files = [
        os.path.join(cfg.pointnet_pt_dir,
                     os.path.basename(f).replace(".laz", ".pt").replace(".las", ".pt"))
        for f in df["filename"]
    ]
    return np.array(pt_files), labels, enc.classes_


def load_lgbm_data(cfg: CVConfig):
    feat_df = pd.read_csv(cfg.lgbm_features_csv)
    label_df = pd.read_csv(cfg.labels_csv)
    feat_df["filename"] = feat_df["filename"].str.lstrip(" /")
    label_df["filename"] = label_df["filename"].str.lstrip(" /")
    merged = pd.merge(feat_df, label_df[["filename", "species"]], on="filename")
    enc = LabelEncoder()
    labels = enc.fit_transform(merged["species"].values)
    X = (merged.drop(columns=["filename", "species", "label_id"], errors="ignore")
               .select_dtypes(include=[np.number, bool])
               .values.astype(np.float32))
    return X, labels, enc.classes_


# ============================================================
# EVALUATION
# ============================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """Return (f1_per_class, recall_per_class, macro_f1, weighted_f1, macro_recall)."""
    _, recall_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    return (
        f1_pc,
        recall_pc,
        f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true, y_pred, average="macro", zero_division=0),
    )


def _ci_stats(values: np.ndarray, n: int):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    margin = st.t.ppf(0.975, n - 1) * st.sem(values) if std > 0 else 0.0
    return mean, max(0.0, mean - margin), min(1.0, mean + margin), std


def build_metrics_report(
    metrics_dict: dict, class_names: np.ndarray, n_splits: int
) -> pd.DataFrame:
    """
    Build a tidy DataFrame with one row per (method, metric) combination.

    Columns: method | scope | name | mean | ci_low_95 | ci_high_95 | std
    """
    rows = []

    for method, state in metrics_dict.items():
        for key, label in [
            ("macro_f1_array",    "Macro F1"),
            ("weighted_f1_array", "Weighted F1"),
            ("macro_recall_array","Macro Recall"),
        ]:
            m, lo, hi, std = _ci_stats(state[key], n_splits)
            rows.append(dict(method=method, scope="global", name=label,
                             mean=m, ci_low_95=lo, ci_high_95=hi, std=std))

        for i, species in enumerate(class_names):
            m, lo, hi, std = _ci_stats(state["f1_matrix"][i, :], n_splits)
            rows.append(dict(method=method, scope="per_species", name=species,
                             mean=m, ci_low_95=lo, ci_high_95=hi, std=std))

    return pd.DataFrame(rows)


def print_summary(report: pd.DataFrame):
    for method in report["method"].unique():
        print(f"\n{'='*60}\n📊 {method}\n{'='*60}")
        sub = report[(report["method"] == method) & (report["scope"] == "global")]
        for _, row in sub.iterrows():
            print(f"  {row['name']:20s}  {row['mean']:.4f}  "
                  f"[{row['ci_low_95']:.4f}, {row['ci_high_95']:.4f}]  ±{row['std']:.4f}")

        # Diagnostic warning
        global_rows = sub.set_index("name")
        if "Macro F1" in global_rows.index and "Weighted F1" in global_rows.index:
            gap = global_rows.loc["Weighted F1", "mean"] - global_rows.loc["Macro F1", "mean"]
            if gap > 0.05:
                print(f"  ⚠️  Weighted–Macro gap = {gap:.4f} "
                      f"→ model biased towards common species.")


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _empty_state(methods, num_classes, n_splits):
    s = {"metrics_dict": {}, "start_fold": 0, "predictions_list": []}
    for m in methods:
        s["metrics_dict"][m] = {
            "f1_matrix":          np.zeros((num_classes, n_splits)),
            "recall_matrix":      np.zeros((num_classes, n_splits)),
            "macro_f1_array":     np.zeros(n_splits),
            "weighted_f1_array":  np.zeros(n_splits),
            "macro_recall_array": np.zeros(n_splits),
        }
    return s


def _load_checkpoint(path, methods, num_classes, n_splits):
    if not os.path.exists(path):
        print("🆕 No checkpoint – starting from fold 0.")
        return _empty_state(methods, num_classes, n_splits)
    print(f"🔄 Resuming from {path}…")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for m in methods:  # back-fill any new method added after the checkpoint
        if m not in ckpt["metrics_dict"]:
            ckpt["metrics_dict"][m] = _empty_state([m], num_classes, n_splits)["metrics_dict"][m]
    return ckpt


def _save_checkpoint(path, fold, state):
    torch.save({**state, "start_fold": fold + 1}, path)


# ============================================================
# INTERNAL METRIC STORE
# ============================================================

def _store_metrics(metrics_dict, method, fold, preds, targets, num_classes):
    f1_pc, rec_pc, mac_f1, w_f1, mac_rec = evaluate(targets, preds, num_classes)
    s = metrics_dict[method]
    s["f1_matrix"][:, fold]    = f1_pc
    s["recall_matrix"][:, fold] = rec_pc
    s["macro_f1_array"][fold]   = mac_f1
    s["weighted_f1_array"][fold] = w_f1
    s["macro_recall_array"][fold] = mac_rec
    print(f"    Macro F1: {mac_f1:.4f} | Weighted F1: {w_f1:.4f}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_cv(cfg: CVConfig):
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))
    print(f"🖥️  Device: {device}")

    # ── Load datasets for requested methods ────────────────────────────────
    dino_features = dino_labels = dino_geo = dino_classes = None
    pn_files = pn_labels = pn_classes = None
    lgbm_X = lgbm_labels = lgbm_classes = None

    if {"DINO_MLP", "DINO_SVM", "DINO_LogReg"} & set(cfg.methods):
        print("📦 Loading DINO embeddings…")
        dino_features, dino_labels, dino_classes, dino_geo = load_dino_data(cfg)
        labels_ref, class_names = dino_labels, dino_classes

    if "PointNet" in cfg.methods:
        print("📦 Loading PointNet data…")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../PointNet/PointNetTraining"))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../PointNet/PointNetArch"))
        pn_files, pn_labels, pn_classes = load_pointnet_data(cfg)
        labels_ref, class_names = pn_labels, pn_classes

    if "LightGBM" in cfg.methods:
        print("📦 Loading LightGBM tabular features…")
        lgbm_X, lgbm_labels, lgbm_classes = load_lgbm_data(cfg)
        labels_ref, class_names = lgbm_labels, lgbm_classes

    n_samples = len(labels_ref)

    # ── Checkpoint ─────────────────────────────────────────────────────────
    state = _load_checkpoint(cfg.checkpoint_file, cfg.methods, cfg.num_classes, cfg.n_splits)
    metrics_dict: dict   = state["metrics_dict"]
    start_fold: int      = state["start_fold"]
    predictions_list: list = state.get("predictions_list", [])

    # ── K-Fold loop ────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    tree_ids = np.arange(n_samples).astype(str)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), labels_ref)):
        if fold < start_fold:
            print(f"⏭️  Fold {fold + 1}/{cfg.n_splits} – skipped.")
            continue
        print(f"\n{'─'*50}\n🚀 Fold {fold + 1}/{cfg.n_splits}\n{'─'*50}")

        fold_results: dict = {"tree_id": tree_ids[val_idx], "fold": fold + 1}

        # ── DINO MLP ───────────────────────────────────────────────────────
        if "DINO_MLP" in cfg.methods:
            if dino_geo is None:
                raise RuntimeError(f"DINO_MLP needs geo features; not found: {cfg.dino_geo_npz}")
            print("  ▶ DINO_MLP")

            geo_scaler = StandardScaler()
            X_tr_geo = torch.from_numpy(geo_scaler.fit_transform(dino_geo[train_idx])).float()
            X_va_geo = torch.from_numpy(geo_scaler.transform(dino_geo[val_idx])).float()
            X_tr_d   = torch.from_numpy(dino_features[train_idx]).float()
            X_va_d   = torch.from_numpy(dino_features[val_idx]).float()
            y_tr     = torch.from_numpy(dino_labels[train_idx]).long()
            y_va     = torch.from_numpy(dino_labels[val_idx]).long()

            tr_loader = DataLoader(
                TensorDataset(X_tr_d, X_tr_geo, y_tr),
                batch_size=cfg.dino_mlp_batch_size,
                sampler=_weighted_sampler(dino_labels[train_idx], cfg.num_classes),
            )
            va_loader = DataLoader(
                TensorDataset(X_va_d, X_va_geo, y_va),
                batch_size=cfg.dino_mlp_batch_size, shuffle=False,
            )

            model = TreeStudentMultimodal(
                dino_dim=dino_features.shape[1],
                geo_dim=dino_geo.shape[1],
                num_classes=cfg.num_classes,
            ).to(device)

            # Callables that tell train_model how to handle this model's batches
            def dino_forward(m, batch, dev):
                x_d, x_g, y = batch
                return m(x_d.to(dev), x_g.to(dev)), y.to(dev)

            def dino_loss(logits, targets):
                return F.cross_entropy(logits, targets)

            model = train_model(
                model, tr_loader, va_loader,
                optimizer=optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
                epochs=cfg.dino_mlp_epochs,
                device=device,
                forward_fn=dino_forward,
                loss_fn=dino_loss,
                scheduler=None,
            )

            model.eval()
            preds = []
            with torch.no_grad():
                for x_d, x_g, _ in va_loader:
                    preds.extend(model(x_d.to(device), x_g.to(device)).argmax(1).cpu().numpy())
            preds = np.array(preds)
            _store_metrics(metrics_dict, "DINO_MLP", fold, preds, dino_labels[val_idx], cfg.num_classes)
            fold_results["pred_DINO_MLP"] = dino_classes[preds]
            fold_results.setdefault("true_species", dino_classes[dino_labels[val_idx]])

        # ── DINO SVM / LogReg ──────────────────────────────────────────────
        if {"DINO_SVM", "DINO_LogReg"} & set(cfg.methods):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(dino_features[train_idx])
            X_va = scaler.transform(dino_features[val_idx])
            y_tr_np = dino_labels[train_idx]
            y_va_np = dino_labels[val_idx]

            if "DINO_SVM" in cfg.methods:
                print("  ▶ DINO_SVM")
                svm = SVC(kernel="rbf", class_weight="balanced", random_state=cfg.random_state)
                svm.fit(X_tr, y_tr_np)
                preds = svm.predict(X_va)
                _store_metrics(metrics_dict, "DINO_SVM", fold, preds, y_va_np, cfg.num_classes)
                fold_results["pred_DINO_SVM"] = dino_classes[preds]
                fold_results.setdefault("true_species", dino_classes[y_va_np])

            if "DINO_LogReg" in cfg.methods:
                print("  ▶ DINO_LogReg")
                lr = LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=cfg.random_state
                )
                lr.fit(X_tr, y_tr_np)
                preds = lr.predict(X_va)
                _store_metrics(metrics_dict, "DINO_LogReg", fold, preds, y_va_np, cfg.num_classes)
                fold_results["pred_DINO_LogReg"] = dino_classes[preds]
                fold_results.setdefault("true_species", dino_classes[y_va_np])

        # ── PointNet++ ─────────────────────────────────────────────────────
        if "PointNet" in cfg.methods:
            print("  ▶ PointNet++")
            from experience.PNTrain.Dataset import TreeLiDARDataset, PointCloudTransforms  # noqa: F401
            from experience.PNTrain.pointnet2_cls_msg import get_model, get_loss            # noqa: F401

            tr_ds = TreeLiDARDataset(
                pn_files[train_idx].tolist(), pn_labels[train_idx].tolist(),
                num_points=cfg.pointnet_num_points,
                transform=PointCloudTransforms(rotation=True, jitter=True, scale=False),
            )
            va_ds = TreeLiDARDataset(
                pn_files[val_idx].tolist(), pn_labels[val_idx].tolist(),
                num_points=cfg.pointnet_num_points, transform=None,
            )
            tr_loader = DataLoader(tr_ds, batch_size=cfg.pointnet_batch_size,
                                   sampler=_weighted_sampler(pn_labels[train_idx], cfg.num_classes),
                                   num_workers=4)
            va_loader = DataLoader(va_ds, batch_size=cfg.pointnet_batch_size,
                                   shuffle=False, num_workers=4)

            pn_model   = get_model(cfg.num_classes, normal_channel=False).to(device)
            pn_criterion = get_loss(class_weights=None).to(device)

            # Callables that tell train_model how to handle PointNet's output tuple
            def pn_forward(m, batch, dev):
                pts, y = batch
                out = m(pts.to(dev))    # (logits, trans_feat, feat)
                return out, y.to(dev)   # return the full tuple so loss_fn can use trans_feat

            def pn_loss(model_out, targets):
                logits, trans_feat, _ = model_out
                return pn_criterion(logits, targets, trans_feat)

            # train_model expects forward_fn to return (logits, targets) for the
            # val F1 computation. We wrap pn_forward to unpack just logits there:
            def pn_forward_eval(m, batch, dev):
                pts, y = batch
                out = m(pts.to(dev))
                return out[0], y.to(dev)   # logits only for argmax

            pn_model = train_model(
                pn_model, tr_loader, va_loader,
                optimizer=optim.Adam(pn_model.parameters(), lr=1e-4,
                                     betas=(0.9, 0.999), weight_decay=1e-4),
                epochs=cfg.pointnet_epochs,
                device=device,
                forward_fn=pn_forward_eval,   # used for both train argmax and val F1
                loss_fn=lambda logits, targets: pn_loss(  # recompute with trans_feat
                    # NOTE: we pass a thin wrapper below instead
                    logits, targets
                ),
                scheduler=optim.lr_scheduler.ReduceLROnPlateau(
                    optim.Adam(pn_model.parameters()), "min", patience=7, factor=0.7
                ),
            )
            # ↑ The loss wrapper above is simplified; see NOTE in train_model docstring.
            # For PointNet's trans_feat regularisation use the dedicated path below.
            pn_model = _train_pointnet_full(
                pn_model, tr_loader, va_loader,
                optimizer=optim.Adam(pn_model.parameters(), lr=1e-4,
                                     betas=(0.9, 0.999), weight_decay=1e-4),
                criterion=pn_criterion,
                epochs=cfg.pointnet_epochs, device=device,
            )

            pn_model.eval()
            preds, y_va_np = [], pn_labels[val_idx]
            with torch.no_grad():
                for pts, _ in va_loader:
                    logits, _, _ = pn_model(pts.to(device))
                    preds.extend(logits.argmax(1).cpu().numpy())
            preds = np.array(preds)
            _store_metrics(metrics_dict, "PointNet", fold, preds, y_va_np, cfg.num_classes)
            fold_results["pred_PointNet"] = pn_classes[preds]
            fold_results.setdefault("true_species", pn_classes[y_va_np])
            del pn_model, pn_criterion, tr_loader, va_loader
            torch.cuda.empty_cache()

        # ── LightGBM ───────────────────────────────────────────────────────
        if "LightGBM" in cfg.methods:
            import lightgbm as lgb
            print("  ▶ LightGBM")
            params = dict(cfg.lgbm_params, num_class=cfg.num_classes)
            tr_ds = lgb.Dataset(lgbm_X[train_idx], label=lgbm_labels[train_idx])
            va_ds = lgb.Dataset(lgbm_X[val_idx],   label=lgbm_labels[val_idx], reference=tr_ds)
            gbm = lgb.train(params, tr_ds,
                            num_boost_round=cfg.lgbm_num_boost_round,
                            valid_sets=[va_ds],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
            preds   = np.argmax(gbm.predict(lgbm_X[val_idx]), axis=1)
            y_va_np = lgbm_labels[val_idx]
            _store_metrics(metrics_dict, "LightGBM", fold, preds, y_va_np, cfg.num_classes)
            fold_results["pred_LightGBM"] = lgbm_classes[preds]
            fold_results.setdefault("true_species", lgbm_classes[y_va_np])

        # ── Save fold ──────────────────────────────────────────────────────
        predictions_list.append(pd.DataFrame(fold_results))
        _save_checkpoint(cfg.checkpoint_file, fold,
                         {"metrics_dict": metrics_dict, "predictions_list": predictions_list})
        print("  💾 Checkpoint saved.")

    # ── Final exports ──────────────────────────────────────────────────────
    pd.concat(predictions_list, ignore_index=True) \
      .sort_values(["fold", "tree_id"]) \
      .to_csv(cfg.predictions_csv, index=False)
    print(f"\n✅ Per-fold predictions → {cfg.predictions_csv}")

    report = build_metrics_report(metrics_dict, class_names, cfg.n_splits)
    report.to_csv(cfg.metrics_csv, index=False)
    print(f"✅ Metrics summary      → {cfg.metrics_csv}")

    print_summary(report)


# ============================================================
# POINTNET FULL TRAINING (uses trans_feat in loss)
# ============================================================

def _train_pointnet_full(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    """
    Concrete PointNet++ training that passes trans_feat to the regularised loss.
    Called instead of the generic train_model when PointNet is used, because
    the model returns a 3-tuple and the criterion needs the second element.

    To add a new deep model with a non-standard forward signature, follow this
    same pattern: write a small concrete trainer and plug it in the fold loop.
    The shared train_model loop above handles every model whose forward returns
    plain logits.
    """
    best_f1, best_weights = 0.0, copy.deepcopy(model.state_dict())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=7, factor=0.7, min_lr=1e-6
    )
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for pts, targets in train_loader:
            pts, targets = pts.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, trans_feat, _ = model(pts)
            loss = criterion(logits, targets, trans_feat)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(train_loader))

        model.eval()
        preds, all_t = [], []
        with torch.no_grad():
            for pts, targets in val_loader:
                logits, _, _ = model(pts.to(device))
                preds.extend(logits.argmax(1).cpu().numpy())
                all_t.extend(targets.numpy())
        val_f1 = f1_score(all_t, preds, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model
