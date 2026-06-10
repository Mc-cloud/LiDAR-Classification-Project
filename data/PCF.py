"""
compute_pcf_dataset.py
======================
Computes the 3D Pair Correlation Function (PCF) for each tree in a LiDAR
dataset, mimicking R's spatstat::pcf.pp3.

Each tree already has its own individual .las file (as listed in labels.csv).

Input folder structure expected
--------------------------------
    train_data/
        labels.csv          # treeID, species, genus, dataset, data_type, tree_H, filename
        train/
            00070.las
            00071.las
            ...

labels.csv columns used
-----------------------
    treeID    – unique tree identifier
    species   – species label
    genus     – genus label
    dataset   – source dataset name
    data_type – sensor type (TLS / ULS …)
    tree_H    – tree height
    filename  – relative path to the .las file, e.g. /train/00070.las

Output
------
    pcf_dataset.csv  –  one row per tree:
        treeID | species | genus | dataset | data_type | tree_H |
        pcf_t_<t0> | pcf_t_<t1> | … | pcf_t_<tN>

Algorithm  (matches spatstat::pcf.pp3 defaults)
-----------------------------------------------
    g(r) ≈  1/(λ² |W|) · Σ_{i<j}  k_h(‖xᵢ−xⱼ‖ − r) / (4πr²·w_ij)  × 2

    where
      λ        = n / |W|                    (point intensity)
      |W|      = bounding-box volume of the crown
      k_h      = Epanechnikov kernel, bandwidth h
      w_ij     = translation-edge-correction weight

    Key optimisation: the Epanechnikov kernel k_h(d - r) is non-zero only for
    r in [d-h, d+h].  We use searchsorted to find those indices directly,
    so we never build the full (n_r × n_pairs) matrix that caused the hang.

Dependencies
------------
    pip install laspy[lazrs] numpy pandas scipy tqdm

Usage
-----
    python PCF.py --train-dir train_data

    # With custom options:
    python PCF.py \
        --train-dir  train_data \
        --r-fraction 0.5        \
        --r-cap      5.0        \
        --r-steps    50         \
        --bandwidth  0.5        \
        --max-points 10000      \
        --out        pcf_dataset.csv
"""

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

    Parameters
    ----------
    points  : (n, 3) array-like  –  x, y, z coordinates
    window  : ((x0,x1), (y0,y1), (z0,z1)) or None.
              If None the bounding box of the points is used,
              with a small epsilon to avoid zero-volume boxes.
    """

    def __init__(self, points, window=None):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be an (n, 3) array.")

        if window is None:
            mins = self.points.min(axis=0)
            maxs = self.points.max(axis=0)
            eps  = 1e-6
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

    def __repr__(self):
        w = self.window
        return (
            f"PP3: {self.n} points | "
            f"window x=[{w[0][0]:.2f},{w[0][1]:.2f}] "
            f"y=[{w[1][0]:.2f},{w[1][1]:.2f}] "
            f"z=[{w[2][0]:.2f},{w[2][1]:.2f}] | "
            f"intensity {self.intensity:.4f}"
        )


# ---------------------------------------------------------------------------
# PCF  (3-D, Epanechnikov kernel, translation-edge correction)
# ---------------------------------------------------------------------------

def _epanechnikov_cdf(r, h):
    """
    Bias-correction factor c(r) — the fraction of the Epanechnikov kernel
    mass that lies at non-negative distances (i.e. at distances <= r from
    the evaluation point r).

    For a kernel k_h(u) = (3/4h)(1 - (u/h)^2) supported on [-h, h]:

        c(r) = integral_{-h}^{r} k_h(u) du
             = 0.5 + (3/4)(r/h) - (1/4)(r/h)^3    for -h <= r <= h
             = 1                                     for r > h
             = 0                                     for r < -h

    When r >= h the full kernel support is positive, so c(r) = 1 and
    the correction has no effect. The correction only matters near r = 0.
    This matches spatstat's biascorrect=TRUE behaviour.
    """
    t = np.clip(r / h, -1.0, 1.0)
    return 0.5 + (3.0 / 4.0) * t - (1.0 / 4.0) * t ** 3

@njit(cache = True)
def _accumulate_hybrid(dists, weights, r_arr, bandwidth, lo_idx, hi_idx, numerators):
    """
    Numba inner loop, but only over the narrow r-window [lo_idx, hi_idx]
    found by searchsorted for each pair.  Each pair touches at most
    ceil(2h / Δr) r-bins instead of all R bins.
    """
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
    """
    NumPy searchsorted narrows the window; Numba does the hot arithmetic.
    """
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

    
def pcf_pp3(
    pattern,
    r_values: np.ndarray,
    bandwidth: float | None = None,
    chunk_pairs: int = 50_000,
) -> np.ndarray:
    """
    3-D pair correlation function at each radius in r_values.

    Parameters
    ----------
    pattern     : PP3 instance or (n, 3) float array.
    r_values    : 1-D array of evaluation radii (must be sorted ascending).
    bandwidth   : Epanechnikov kernel bandwidth h.
                  None -> Silverman's 3-D rule-of-thumb h = 1.06*sigma*n^(-1/7).
    chunk_pairs : pairs processed per iteration (tune for RAM; 50k ~ 20 MB).

    Returns
    -------
    g : 1-D array, same length as r_values.  NaN where undefined.
    """
    if not isinstance(pattern, PP3):
        pattern = PP3(pattern)

    points = pattern.points
    n      = pattern.n
    sides  = pattern.sides
    volume = pattern.volume

    if n < 2:
        return np.full(len(r_values), np.nan)

    lam  = pattern.intensity
    tree = cKDTree(points)

    # --- auto bandwidth -----------------------------------------------------
    if bandwidth is None:
        nn_dist, _ = tree.query(points, k=2, workers=1)
        sigma      = float(np.std(nn_dist[:, 1]))
        bandwidth  = max(1.06 * sigma * n ** (-1.0 / 7.0), 1e-4)

    r_max  = float(r_values[-1])
    radius = r_max + bandwidth
    n_r    = len(r_values)
    r_arr  = np.asarray(r_values, dtype=float)
    numerators = np.zeros(n_r)
    surface     = 4.0 * np.pi * r_arr ** 2
    denominator = lam ** 2 * volume * surface
    # --- get all pairs (i < j) in one C-level call -------------------------
    # query_pairs is O(n log n + P) and returns an (P, 2) ndarray directly.
    pairs = tree.query_pairs(radius, output_type='ndarray')  # (P, 2)
    if len(pairs) == 0:
        return np.full(n_r, np.nan)

    
    for start in range(0, len(pairs), chunk_pairs):
        chunk = pairs[start:start + chunk_pairs]
        diff_ij = np.abs(points[chunk[:,0]] - points[chunk[:,1]])

        dists_ij = np.linalg.norm(diff_ij, axis = 1)
        w_ij = np.prod(np.maximum(sides - diff_ij, 0.0), axis =1)

        valid = (dists_ij > 0) & (dists_ij <= radius) & (w_ij > 1e-12)

        diff_ij = diff_ij[valid]
        dists_ij = dists_ij[valid]
        w_ij = w_ij[valid]

        if len(dists_ij)== 0:
            continue

        accumulate_pcf(dists_ij.astype(np.float64),
                       w_ij.astype(np.float64),
                       r_arr.astype(np.float64),
                       bandwidth,
                       numerators
                       )


    # --- normalise (×2: upper triangle only) --------------------------------
    
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(denominator > 0, 2.0 * numerators / denominator, np.nan)
    g[r_arr <= 0] = np.nan

    c = _epanechnikov_cdf(r_arr, bandwidth)
    c = np.where(c < 1e-6, np.nan, c)

    with np.errstate(invalid = "ignore", divide = "ignore"):
        g = g/c

    return g


# ---------------------------------------------------------------------------
# Per-tree worker
# ---------------------------------------------------------------------------

def _process_tree(args):
    """Process a single tree: load LAS, subsample, compute PCF, return row."""
    row, train_dir, t_values, r_fraction, r_cap, bandwidth, max_points, meta_cols, r_cols = args

    try:
        import laspy
    except ImportError:
        return None, "laspy not installed"

    # --- per-tree r_max -----------------------------------------------------
    tree_h = float(row.get("tree_H", 0)) if "tree_H" in row.index else 0.0
    r_max  = min(r_fraction * tree_h, r_cap) if tree_h > 0 else r_cap
    if r_max <= 0:
        r_max = r_cap
    r_values = t_values * r_max

    # --- load LAS -----------------------------------------------------------
    bare_name = os.path.basename(str(row["filename"]))
    las_path  = os.path.join(train_dir, bare_name)
    try:
        las = laspy.read(las_path)
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
    except FileNotFoundError:
        return None, f"not found: {las_path}"
    except Exception as e:
        return None, f"read error ({las_path}): {e}"

    # --- random subsample ---------------------------------------------------
    n_pts = len(pts)
    if max_points > 0 and n_pts > max_points:
        rng = np.random.default_rng(seed=0)
        pts = pts[rng.choice(n_pts, size=max_points, replace=False)]

    # --- build PP3 then compute PCF ----------------------------------------
    pattern = PP3(pts)
    g = pcf_pp3(pattern, r_values, bandwidth=bandwidth)

    # --- assemble output row ------------------------------------------------
    out_row = {c: row[c] for c in meta_cols if c in row.index}
    out_row["r_max"] = round(r_max, 4)
    out_row.update(dict(zip(r_cols, g)))
    return out_row, None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _worker(args):
    """Top-level wrapper so multiprocessing can pickle it on Windows."""
    try:
        return _process_tree(args)
    except Exception as e:
        return None, str(e)
    
def _init_worker():
    try: import laspy
    except ImportError: pass
    _d = np.array([0.1], dtype=np.float64)
    _accumulate_hybrid(_d, _d, _d, 0.1,
                       np.zeros(1, np.int64), np.ones(1, np.int64),
                       np.zeros(1, np.float64))

def build_pcf_dataset(
    train_dir:   str        = "train_data",
    r_fraction:  float      = 0.5,
    r_cap:       float      = 5.0,
    r_steps:     int        = 20,
    bandwidth:   float|None = None,
    max_points:  int        = 10_000,
    out_path:    str        = "pcf_dataset.csv",
    n_workers:   int        = 0,          # 0 = use all CPU cores
) -> pd.DataFrame:

    # --- load labels --------------------------------------------------------
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(script_dir, train_dir, "labels.csv")
    print(f"[1/3] Loading labels: {labels_path}", flush=True)
    labels = pd.read_csv(labels_path)
    labels.columns = [c.strip() for c in labels.columns]

    id_col    = "treeID" if "treeID" in labels.columns else "tree_id"
    meta_cols = [c for c in [id_col, "species", "genus", "dataset", "data_type", "tree_H"]
                 if c in labels.columns]
    print(f"    {len(labels)} trees found.", flush=True)

    # --- shared config ------------------------------------------------------
    t_values = np.linspace(1.0 / r_steps, 1.0, r_steps)
    r_cols   = [f"pcf_t_{t:.4f}" for t in t_values]

    n_cpu = n_workers if n_workers > 0 else mp.cpu_count()
    print(
        f"[2/3] Computing PCF ({r_steps} steps, "
        f"r_max=min({r_fraction}*tree_H,{r_cap}m), "
        f"max_points={max_points}, workers={n_cpu}) …",
        flush=True,
    )

    task_args = [
        (row, train_dir, t_values, r_fraction, r_cap, bandwidth, max_points, meta_cols, r_cols)
        for _, row in labels.iterrows()
    ]

    rows    = []
    skipped = 0

    # imap_unordered streams results as workers finish — memory-efficient
    # and gives accurate tqdm progress across all cores.
    with mp.Pool(processes=n_cpu, initializer=_init_worker) as pool:
        for result, err in tqdm(
            pool.imap_unordered(_worker, task_args, chunksize=4),
            total=len(task_args),
            desc="PCF",
            unit="tree",
            dynamic_ncols=True,
        ):
            if err:
                warnings.warn(f"Skipped: {err}")
                skipped += 1
            else:
                rows.append(result)

    # --- save ---------------------------------------------------------------
    print(f"[3/3] Saving → {out_path}", flush=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    print(f"\nDone.  {len(rows)} trees saved, {skipped} skipped.", flush=True)
    print(f"Output shape: {result_df.shape}", flush=True)
    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mp.freeze_support()   # needed for Windows multiprocessing / PyInstaller
    parser = argparse.ArgumentParser(
        description="Build PCF dataset (one row per tree) from per-tree LAS files."
    )
    parser.add_argument(
        "--train-dir", default="train_data",
        help="Root folder containing labels.csv and .las files (default: train_data)",
    )
    parser.add_argument(
        "--r-fraction", default=0.5, type=float,
        help="r_max = r_fraction * tree_H  (default: 0.5)",
    )
    parser.add_argument(
        "--r-cap", default=5.0, type=float,
        help="Hard cap on r_max in metres: r_max = min(fraction*H, cap)  (default: 5.0)",
    )
    parser.add_argument(
        "--r-steps", default=20, type=int,
        help="Number of normalised radius steps in (0, 1]  (default: 50)",
    )
    parser.add_argument(
        "--bandwidth", default=None, type=float,
        help="Epanechnikov kernel bandwidth h. "
             "Omit to use Silverman's 3-D rule-of-thumb per tree.",
    )
    parser.add_argument(
        "--max-points", default=10_000, type=int,
        help="Randomly subsample each crown to at most this many points "
             "(default: 10000). Set 0 to disable subsampling.",
    )
    parser.add_argument(
        "--out", default="pcf_dataset.csv",
        help="Output CSV path (default: pcf_dataset.csv)",
    )
    parser.add_argument(
        "--workers", default=0, type=int,
        help="Number of parallel worker processes (default: 0 = all CPU cores).",
    )
    args = parser.parse_args()

    build_pcf_dataset(
        train_dir  = args.train_dir,
        r_fraction = args.r_fraction,
        r_cap      = args.r_cap,
        r_steps    = args.r_steps,
        bandwidth  = args.bandwidth,
        max_points = args.max_points,
        out_path   = args.out,
        n_workers  = args.workers,
    )