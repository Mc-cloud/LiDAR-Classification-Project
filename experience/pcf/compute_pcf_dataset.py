import argparse
import os
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
from numba import njit


# ---------------------------------------------------------------------------
# PP3 — 3-D point pattern class
# ---------------------------------------------------------------------------

class PP3:
    """
    A 3-D point pattern inside a rectangular observation window.
    """
    def __init__(self, points, window=None):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be an (n, 3) array.")

        if window is None:
            mins = self.points.min(axis=0)
            maxs = self.points.max(axis=0)
            eps = 1e-6
            self.window = tuple(
                (float(mins[i]), float(maxs[i]) + eps) for i in range(3)
            )
        else:
            self.window = tuple((float(a), float(b)) for a, b in window)

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def sides(self) -> np.ndarray:
        return np.array([b - a for a, b in self.window])

    @property
    def volume(self) -> float:
        return float(np.prod(self.sides))

    @property
    def intensity(self) -> float:
        return self.n / self.volume if self.volume > 0 else 0.0

    def __len__(self):
        return self.n


# ---------------------------------------------------------------------------
# PCF Math & Accelerations
# ---------------------------------------------------------------------------

def _epanechnikov_cdf(r, h):
    """Bias-correction factor c(r) near r=0 matching spatstat's biascorrect=TRUE."""
    t = np.clip(r / h, -1.0, 1.0)
    return 0.5 + (3.0 / 4.0) * t - (1.0 / 4.0) * t ** 3

@njit(cache=True)
def _accumulate_hybrid(dists, weights, r_arr, bandwidth, lo_idx, hi_idx, numerators):
    """Numba hot-loop iterating tightly over narrow r-windows per pair."""
    h = bandwidth
    n_pairs = len(dists)
    for k in range(n_pairs):
        i0 = lo_idx[k]
        i1 = hi_idx[k]
        if i0 >= i1:
            continue
        d = dists[k]
        w = weights[k]
        for i in range(i0, i1):
            t = (d - r_arr[i]) / h
            numerators[i] += (0.75 / h) * (1.0 - t * t) / w


def accumulate_pcf(dists: np.ndarray,
                   weights: np.ndarray,
                   r_arr: np.ndarray,
                   bandwidth: float,
                   numerators: np.ndarray) -> None:
    """Uses numpy searchsorted to binary-search the boundary indices."""
    lo_idx = np.searchsorted(r_arr, dists - bandwidth, side='left').astype(np.int64)
    hi_idx = np.searchsorted(r_arr, dists + bandwidth, side='right').astype(np.int64)
    _accumulate_hybrid(
        dists.astype(np.float64),
        weights.astype(np.float64),
        r_arr.astype(np.float64),
        float(bandwidth),
        lo_idx,
        hi_idx,
        numerators,
    )


def pcf_pp3(pattern, r_values: np.ndarray, bandwidth: float | None = None, chunk_pairs: int = 50_000) -> np.ndarray:
    """Computes the 3-D pair correlation function for a point cloud pattern."""
    if not isinstance(pattern, PP3):
        pattern = PP3(pattern)

    points = pattern.points
    n = pattern.n
    sides = pattern.sides
    volume = pattern.volume

    if n < 2:
        return np.full(len(r_values), np.nan)

    lam = pattern.intensity
    tree = cKDTree(points)

    if bandwidth is None:
        nn_dist, _ = tree.query(points, k=2, workers=1)
        sigma = float(np.std(nn_dist[:, 1]))
        bandwidth = max(1.06 * sigma * n ** (-1.0 / 7.0), 1e-4)

    r_max = float(r_values[-1])
    radius = r_max + bandwidth
    n_r = len(r_values)
    r_arr = np.asarray(r_values, dtype=float)
    numerators = np.zeros(n_r)
    surface = 4.0 * np.pi * r_arr ** 2
    denominator = lam ** 2 * volume * surface

    pairs = tree.query_pairs(radius, output_type='ndarray')
    if len(pairs) == 0:
        return np.full(n_r, np.nan)

    for start in range(0, len(pairs), chunk_pairs):
        chunk = pairs[start:start + chunk_pairs]
        diff_ij = np.abs(points[chunk[:, 0]] - points[chunk[:, 1]])
        dists_ij = np.linalg.norm(diff_ij, axis=1)
        w_ij = np.prod(np.maximum(sides - diff_ij, 0.0), axis=1)

        valid = (dists_ij > 0) & (dists_ij <= radius) & (w_ij > 1e-12)
        dists_ij = dists_ij[valid]
        w_ij = w_ij[valid]

        if len(dists_ij) == 0:
            continue

        accumulate_pcf(dists_ij.astype(np.float64), w_ij.astype(np.float64), r_arr.astype(np.float64), bandwidth, numerators)

    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(denominator > 0, 2.0 * numerators / denominator, np.nan)
    g[r_arr <= 0] = np.nan

    c = _epanechnikov_cdf(r_arr, bandwidth)
    c = np.where(c < 1e-6, np.nan, c)

    with np.errstate(invalid="ignore", divide="ignore"):
        g = g / c

    return g


# ---------------------------------------------------------------------------
# Per-Tree Processing & Worker Pipeline
# ---------------------------------------------------------------------------

def _process_tree(args):
    """Worker core executing standard tasks on a single row/tree."""
    row, train_dir, t_values, r_fraction, r_cap, bandwidth, max_points, meta_cols, r_cols = args

    try:
        import laspy
    except ImportError:
        return None, "laspy not installed"

    tree_h = float(row.get("tree_H", 0)) if "tree_H" in row.index else 0.0
    r_max = min(r_fraction * tree_h, r_cap) if tree_h > 0 else r_cap
    if r_max <= 0:
        r_max = r_cap
    r_values = t_values * r_max

    bare_name = os.path.basename(str(row["filename"]))
    las_path = os.path.join(train_dir, bare_name)
    try:
        las = laspy.read(las_path)
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
    except FileNotFoundError:
        return None, f"not found: {las_path}"
    except Exception as e:
        return None, f"read error ({las_path}): {e}"

    n_pts = len(pts)
    if max_points > 0 and n_pts > max_points:
        rng = np.random.default_rng(seed=0)
        pts = pts[rng.choice(n_pts, size=max_points, replace=False)]

    pattern = PP3(pts)
    g = pcf_pp3(pattern, r_values, bandwidth=bandwidth)

    out_row = {c: row[c] for c in meta_cols if c in row.index}
    out_row["r_max"] = round(r_max, 4)
    out_row.update(dict(zip(r_cols, g)))
    return out_row, None


def _worker(args):
    """Multiprocessing wrapper providing safety boundaries."""
    try:
        return _process_tree(args)
    except Exception as e:
        return None, str(e)


def _init_worker():
    """Warms up numba compilation on startup before massive arrays arrive."""
    _d = np.array([0.1], dtype=np.float64)
    _accumulate_hybrid(_d, _d, _d, 0.1, np.zeros(1, np.int64), np.ones(1, np.int64), np.zeros(1, np.float64))


def build_pcf_dataset(train_dir="train_data", r_fraction=0.5, r_cap=5.0, r_steps=20,
                      bandwidth=None, max_points=10_000, out_path="pcf_dataset.csv", n_workers=0) -> pd.DataFrame:

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    labels_path = os.path.join(script_dir, train_dir, "labels.csv")
    
    print(f"[1/3] Loading labels: {labels_path}", flush=True)
    labels = pd.read_csv(labels_path)
    labels.columns = [c.strip() for c in labels.columns]

    id_col = "treeID" if "treeID" in labels.columns else "tree_id"
    meta_cols = [c for c in [id_col, "species", "genus", "dataset", "data_type", "tree_H"] if c in labels.columns]
    print(f"    {len(labels)} trees identified in metadata.", flush=True)

    t_values = np.linspace(1.0 / r_steps, 1.0, r_steps)
    r_cols = [f"pcf_t_{t:.4f}" for t in t_values]
    n_cpu = n_workers if n_workers > 0 else mp.cpu_count()

    print(f"[2/3] Processing parallel PCF over {n_cpu} cores...", flush=True)
    task_args = [
        (row, train_dir, t_values, r_fraction, r_cap, bandwidth, max_points, meta_cols, r_cols)
        for _, row in labels.iterrows()
    ]

    rows, skipped = [], 0
    with mp.Pool(processes=n_cpu, initializer=_init_worker) as pool:
        for result, err in tqdm(pool.imap_unordered(_worker, task_args, chunksize=4), total=len(task_args), desc="PCF Calculation"):
            if err:
                warnings.warn(f"Skipped tree due to error: {err}")
                skipped += 1
            else:
                rows.append(result)

    print(f"[3/3] Committing matrix updates down to → {out_path}", flush=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    print(f"\nDone. Saved {len(rows)} successfully ({skipped} tracks skipped). Matrix shape: {result_df.shape}")
    return result_df


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Compute tree-wise PCF features from distinct LAS inputs.")
    parser.add_argument("--train-dir", default="train_data", help="Root input folder.")
    parser.add_argument("--r-fraction", default=0.5, type=float, help="Scaling window ratio.")
    parser.add_argument("--r-cap", default=5.0, type=float, help="Hard ceiling cut-off distance.")
    parser.add_argument("--r-steps", default=20, type=int, help="Output columns resolution granularity.")
    parser.add_argument("--bandwidth", default=None, type=float, help="Custom kernel step size.")
    parser.add_argument("--max-points", default=10_000, type=int, help="Downsampling density throttle framework.")
    parser.add_argument("--out", default="pcf_dataset.csv", help="Destination table path.")
    parser.add_argument("--workers", default=0, type=int, help="Enforced thread count limitation boundaries.")
    args = parser.parse_args()

    build_pcf_dataset(
        train_dir=args.train_dir, r_fraction=args.r_fraction, r_cap=args.r_cap, r_steps=args.r_steps,
        bandwidth=args.bandwidth, max_points=args.max_points, out_path=args.out, n_workers=args.workers
    )