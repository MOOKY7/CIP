
#!/usr/bin/env python3
"""
Real-data evaluation suite for Conformal Information Projection (CIP)
====================================================================

This script mirrors the structure of simulation suite, but replaces the
synthetic DGP with open-source real regression datasets that have *natural
group attributes*.

Datasets implemented (default):
  (1) Diamonds price (group = cut)  [cross-sectional, 5 groups]
  (2) Ames Housing / "House Prices" (group = Neighborhood, top-K + Other)

For each dataset and each replicate split, we:
  - Fit a base Gibbs posterior q0(β) for linear regression (Gaussian prior).
  - Define K tail constraints using split-conformal quantiles from a reference predictor.
  - Compute CIP-Global and CIP-Group I-projections via the convex dual on a q0 sample pool,
    stabilized by discrete-pool SMC.
  - Compare against: TempTune (η tuning), ridge + split conformal, Huber + split conformal,
    Mondrian conformal ridge.

Outputs (per dataset):
  outdir/<dataset>/metrics.csv         (per replicate)
  outdir/<dataset>/summary.csv         (mean/std)
  outdir/<dataset>/figures/*.pdf, *.png

Dependencies:
  numpy, scipy, pandas, matplotlib, scikit-learn

Run:
  python real_data_cip_suite.py --outdir real_data_results --datasets diamonds ames --R 20

Notes for paper-quality runs:
  - Increase R (e.g., 30–50).
  - Increase S_pool (e.g., 40000–100000) and n_particles (e.g., 5000–10000) if needed.
  - Consider reporting sensitivity to K, tighten_factor, and group-collapse choices.
"""

import os
import time
import warnings
import argparse
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.optimize import minimize

from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# Utilities
# ============================================================

def safe_dirname(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", ".", "="):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_url(url: str, dst_path: str, timeout: int = 60) -> None:
    """
    Download a URL to dst_path (atomic write).
    """
    ensure_dir(os.path.dirname(dst_path) or ".")
    tmp = dst_path + ".tmp"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile:
      k = ceil((n+1)(1-alpha)), return k-th order statistic (1-indexed).
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.shape[0]
    k = int(np.ceil((n + 1) * (1 - float(alpha))))
    k = min(max(k, 1), n)
    return float(np.partition(residuals, k - 1)[k - 1])


def alpha_levels_for_K(K: int) -> np.ndarray:
    """
    Default tail-probability grids (ordered from larger to smaller alpha).
    """
    K = int(K)
    presets = {
        2: np.array([0.10, 0.05], dtype=float),
        3: np.array([0.20, 0.10, 0.05], dtype=float),
        5: np.array([0.30, 0.20, 0.10, 0.05, 0.02], dtype=float),
        8: np.array([0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01], dtype=float),
        10: np.array([0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01], dtype=float),
    }
    if K in presets:
        return presets[K].copy()
    al = np.geomspace(0.30, 0.02, K)
    al = np.sort(al)[::-1]
    return al.astype(float)


def tighten_levels(alpha_levels: np.ndarray, m: int, L: int, delta: float, factor: float) -> Tuple[np.ndarray, float]:
    """
    Mild tightening based on Hoeffding + union bound:
      eps = factor * sqrt( log( (2L)/delta ) / (2m) )
    """
    alpha_levels = np.asarray(alpha_levels, dtype=float)
    m = max(int(m), 1)
    L = max(int(L), 1)
    delta = max(float(delta), 1e-12)
    factor = float(factor)

    eps = factor * np.sqrt(np.log((2.0 * L) / delta) / (2.0 * m))
    return np.clip(alpha_levels - eps, 0.0, 1.0), float(eps)


def set_mpl_style() -> None:
    """NeurIPS-friendly Matplotlib defaults; avoids Type 3 fonts."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": False,
    })


# ============================================================
# Dataset loaders (open-source)
# ============================================================

@dataclass
class DatasetConfig:
    name: str
    loader: Callable[[str], pd.DataFrame]
    target_col: str
    group_col: str
    # collapse group levels to top_k + "Other" (None = no collapse)
    group_top_k: Optional[int] = None
    # "none" or "log1p"
    y_transform: str = "none"
    # split fractions: fit, thr, proj, cal, test (must sum to 1)
    frac_fit: float = 0.40
    frac_thr: float = 0.15
    frac_proj: float = 0.15
    frac_cal: float = 0.15
    frac_test: float = 0.15


def load_diamonds(cache_dir: str) -> pd.DataFrame:
    """
    Diamonds dataset (price regression) with categorical quality attributes.
    Source: seaborn-data repository (CSV).
    """
    ensure_dir(cache_dir)
    path = os.path.join(cache_dir, "diamonds.csv")
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
        print(f"[download] diamonds -> {path}")
        download_url(url, path)
    df = pd.read_csv(path)
    return df


def load_ames_house_prices(cache_dir: str) -> pd.DataFrame:
    """
    Ames Housing / House Prices dataset used widely in regression benchmarking.
    Preferred source: OpenML via sklearn.fetch_openml.
    Fallback: public GitHub mirror of Kaggle 'train.csv' (if OpenML is unavailable).
    """
    ensure_dir(cache_dir)

    # Try OpenML first (caches under sklearn by default)
    try:
        # OpenML data_id=42165 is commonly used for this dataset; name "house_prices" also works in many environments.
        ds = fetch_openml(data_id=42165, as_frame=True)  # type: ignore
        df = ds.frame.copy()
        return df
    except Exception as e:
        warnings.warn(f"fetch_openml failed ({e}). Falling back to GitHub mirror of train.csv.")

    path = os.path.join(cache_dir, "ames_train.csv")
    if not os.path.exists(path):
        # A widely mirrored copy of the Kaggle training file.
        url = "https://raw.githubusercontent.com/hemanth-HN/OR568/master/train.csv"
        print(f"[download] ames train.csv -> {path}")
        download_url(url, path)
    df = pd.read_csv(path)
    return df


def load_bike_sharing_hourly(cache_dir: str) -> pd.DataFrame:
    """
    UCI Bike Sharing hourly demand dataset.

    Target: cnt (hourly rental count).  Group: season_name.  We remove casual
    and registered because cnt = casual + registered.  We also drop month and
    the raw season code so the group-blind design does not recover the group
    label through a trivial calendar surrogate.
    """
    ensure_dir(cache_dir)
    zip_path = os.path.join(cache_dir, "bike_sharing_dataset.zip")
    if not os.path.exists(zip_path):
        url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
        print(f"[download] UCI Bike Sharing -> {zip_path}")
        download_url(url, zip_path, timeout=120)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        hour_name = "hour.csv" if "hour.csv" in names else next(n for n in names if n.endswith("/hour.csv"))
        with zf.open(hour_name) as f:
            df = pd.read_csv(f)

    season_map = {1: "1 Spring", 2: "2 Summer", 3: "3 Fall", 4: "4 Winter"}
    weather_map = {1: "1 Clear", 2: "2 Mist", 3: "3 LightPrecip", 4: "4 HeavyPrecip"}
    df["season_name"] = df["season"].map(season_map).astype(object)
    df["weather_name"] = df["weathersit"].map(weather_map).astype(object)

    # Calendar indicators are categorical factors.
    for c in ["yr", "hr", "holiday", "weekday", "workingday"]:
        df[c] = df[c].astype(int).astype(str)

    drop_cols = ["instant", "dteday", "season", "weathersit", "mnth", "casual", "registered"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


# ============================================================
# Preprocessing: group collapse, one-hot, scaling, intercept, y transform
# ============================================================

def collapse_groups_to_top_k(g: pd.Series, top_k: int, other_label: str = "Other") -> pd.Series:
    """
    Keep the most frequent top_k groups, collapse the rest into other_label.
    """
    vc = g.value_counts(dropna=False)
    keep = set(vc.index[:int(top_k)].tolist())
    out = g.astype(object).copy()
    out[~out.isin(keep)] = other_label
    out = out.fillna(other_label)
    return out


def make_onehot_encoder(drop_first: bool = False) -> OneHotEncoder:
    # sklearn <1.2 uses sparse=..., >=1.2 uses sparse_output=...
    kwargs = {"handle_unknown": "ignore", "drop": "first" if bool(drop_first) else None}
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def build_preprocessor(
    df_X: pd.DataFrame,
    drop_first_categorical: bool = False,
) -> ColumnTransformer:
    """
    Numeric: median impute + standardize.
    Categorical: most_frequent impute + one-hot.
    """
    num_cols = df_X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder(drop_first=drop_first_categorical)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


def y_forward(y: np.ndarray, transform: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if transform == "none":
        return y
    if transform == "log1p":
        if np.any(y < 0):
            raise ValueError("log1p transform requested but y has negative values.")
        return np.log1p(y)
    raise ValueError(f"Unknown y_transform='{transform}'")


def y_inverse(y_trans: np.ndarray, transform: str) -> np.ndarray:
    y_trans = np.asarray(y_trans, dtype=float)
    if transform == "none":
        return y_trans
    if transform == "log1p":
        return np.expm1(y_trans)
    raise ValueError(f"Unknown y_transform='{transform}'")


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    return np.column_stack([np.ones(n, dtype=float), X])


def standardize_y(y_fit: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(y_fit))
    sd = float(np.std(y_fit) + 1e-12)
    return mu, sd


def split_five_way_stratified(
    rng: np.random.Generator,
    group_codes: np.ndarray,
    frac_fit: float,
    frac_thr: float,
    frac_proj: float,
    frac_cal: float,
    frac_test: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified splits by group codes into fit / thr / proj / cal / test indices.

    The split is done sequentially with StratifiedShuffleSplit to preserve group proportions.
    """
    n = group_codes.shape[0]
    idx_all = np.arange(n)

    fracs = np.array([frac_fit, frac_thr, frac_proj, frac_cal, frac_test], dtype=float)
    fracs = fracs / fracs.sum()
    frac_fit, frac_thr, frac_proj, frac_cal, frac_test = fracs.tolist()

    # 1) split off test
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=frac_test, random_state=int(rng.integers(0, 2**32 - 1))
    )
    idx_rem, idx_test = next(sss1.split(idx_all, group_codes))

    # 2) split off cal from remaining
    rem_codes = group_codes[idx_rem]
    frac_cal_rel = frac_cal / (1.0 - frac_test)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=frac_cal_rel, random_state=int(rng.integers(0, 2**32 - 1))
    )
    idx_rem2_rel, idx_cal_rel = next(sss2.split(idx_rem, rem_codes))
    idx_rem2 = idx_rem[idx_rem2_rel]
    idx_cal = idx_rem[idx_cal_rel]

    # 3) split off thr from remaining
    rem2_codes = group_codes[idx_rem2]
    frac_thr_rel = frac_thr / (1.0 - frac_test - frac_cal)
    sss3 = StratifiedShuffleSplit(
        n_splits=1, test_size=frac_thr_rel, random_state=int(rng.integers(0, 2**32 - 1))
    )
    idx_rem3_rel, idx_thr_rel = next(sss3.split(idx_rem2, rem2_codes))
    idx_rem3 = idx_rem2[idx_rem3_rel]
    idx_thr = idx_rem2[idx_thr_rel]

    # 4) split fit/proj from remaining
    rem3_codes = group_codes[idx_rem3]
    frac_proj_rel = frac_proj / (1.0 - frac_test - frac_cal - frac_thr)
    sss4 = StratifiedShuffleSplit(
        n_splits=1, test_size=frac_proj_rel, random_state=int(rng.integers(0, 2**32 - 1))
    )
    idx_fit_rel, idx_proj_rel = next(sss4.split(idx_rem3, rem3_codes))
    idx_fit = idx_rem3[idx_fit_rel]
    idx_proj = idx_rem3[idx_proj_rel]

    return idx_fit, idx_thr, idx_proj, idx_cal, idx_test


def preprocess_splits(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    idx_fit: np.ndarray,
    idx_thr: np.ndarray,
    idx_proj: np.ndarray,
    idx_cal: np.ndarray,
    idx_test: np.ndarray,
    group_top_k: Optional[int],
    include_group_in_X: bool,
    y_transform_name: str,
    drop_first_categorical: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Preprocess into dense design matrices + standardized (transformed) y arrays.
    Returns numpy arrays:
      X_fit, X_thr, X_proj, X_cal, X_test
      y_fit, y_thr, y_proj, y_cal, y_test  (standardized, on transformed scale)
      y_*_orig  (original y, untransformed)
      group_* codes in {0,...,G-1}
      plus meta: group_levels, y_mu, y_sd, y_transform
    """
    df = df.copy()


    y_orig = df[target_col].to_numpy(dtype=float)
    g_raw = df[group_col].astype(object)

    if group_top_k is not None:
        g_raw = collapse_groups_to_top_k(g_raw, top_k=int(group_top_k), other_label="Other")

    # Map groups to integer codes
    g_levels = sorted(pd.unique(g_raw))
    g_to_code = {g: i for i, g in enumerate(g_levels)}
    g_code = g_raw.map(g_to_code).to_numpy(dtype=int)

    # Features
    drop_cols = [target_col]
    if not include_group_in_X:
        drop_cols.append(group_col)
    df_X = df.drop(columns=drop_cols)

    # y transform + standardize (on transformed scale)
    y_t = y_forward(y_orig, y_transform_name)
    y_mu, y_sd = standardize_y(y_t[idx_fit])
    y_std = (y_t - y_mu) / y_sd

    pre = build_preprocessor(
        df_X.iloc[idx_fit],
        drop_first_categorical=drop_first_categorical,
    )
    pre.fit(df_X.iloc[idx_fit])

    try:
        feature_names = ["intercept"] + [str(x) for x in pre.get_feature_names_out()]
    except Exception:
        n_features = np.asarray(pre.transform(df_X.iloc[idx_fit[:1]]), dtype=float).shape[1]
        feature_names = ["intercept"] + [f"x{j}" for j in range(n_features)]

    def transform_idx(idxs: np.ndarray) -> np.ndarray:
        X = pre.transform(df_X.iloc[idxs])
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        X = add_intercept(X)
        return X

    out = {
        "X_fit": transform_idx(idx_fit),
        "X_thr": transform_idx(idx_thr),
        "X_proj": transform_idx(idx_proj),
        "X_cal": transform_idx(idx_cal),
        "X_test": transform_idx(idx_test),
        "y_fit": y_std[idx_fit],
        "y_thr": y_std[idx_thr],
        "y_proj": y_std[idx_proj],
        "y_cal": y_std[idx_cal],
        "y_test": y_std[idx_test],
        "y_fit_orig": y_orig[idx_fit],
        "y_thr_orig": y_orig[idx_thr],
        "y_proj_orig": y_orig[idx_proj],
        "y_cal_orig": y_orig[idx_cal],
        "y_test_orig": y_orig[idx_test],
        "g_fit": g_code[idx_fit],
        "g_thr": g_code[idx_thr],
        "g_proj": g_code[idx_proj],
        "g_cal": g_code[idx_cal],
        "g_test": g_code[idx_test],
        "group_levels": np.array(g_levels, dtype=object),
        "feature_names": np.array(feature_names, dtype=object),
        "y_mu": np.array([y_mu], dtype=float),
        "y_sd": np.array([y_sd], dtype=float),
        "y_transform": np.array([y_transform_name], dtype=object),
        "preprocessor": pre,  # keep for debugging if needed
    }
    return out


# ============================================================
# Optional nonlinear random-feature posterior
# ============================================================

def _stable_standardize_features(X_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_base = np.asarray(X_base, dtype=float)
    mu = X_base.mean(axis=0)
    sd = X_base.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (X_base - mu) / sd
    return Z, mu, sd


def make_random_feature_state(
    rng: np.random.Generator,
    X_fit: np.ndarray,
    posterior_model: str = "linear",
    rf_dim: int = 256,
    rf_scale: float = 1.0,
    rf_include_linear: bool = True,
) -> Dict[str, object]:
    """
    Build a split-safe random feature map using only X_fit.

    The input matrices produced by preprocess_splits contain an intercept in the
    first column.  For nonlinear variants we keep an intercept, standardize the
    non-intercept columns using D_fit, and append random Fourier or random ReLU
    features.  The posterior is still Gaussian in the last-layer weights, so it
    is an approximate posterior over a nonlinear predictor.
    """
    model = str(posterior_model).lower()
    if model in ("linear", "none"):
        return {"model": "linear"}

    X_fit = np.asarray(X_fit, dtype=float)
    X_base = X_fit[:, 1:] if X_fit.shape[1] > 1 else X_fit
    _, mu, sd = _stable_standardize_features(X_base)
    d_in = int(X_base.shape[1])
    m = int(rf_dim)
    if m <= 0:
        raise ValueError("rf_dim must be positive for nonlinear random-feature models.")

    lengthscale = max(float(rf_scale), 1e-8)
    W = rng.normal(size=(m, d_in)) / lengthscale
    if model in ("rff", "fourier", "random_fourier"):
        b = rng.uniform(0.0, 2.0 * np.pi, size=m)
    elif model in ("relu", "relu_rf", "random_relu"):
        b = rng.normal(scale=0.5, size=m)
    elif model in ("tanh", "tanh_rf"):
        b = rng.normal(scale=0.5, size=m)
    else:
        raise ValueError(f"Unknown posterior_model='{posterior_model}'. Use linear, rff, relu, or tanh.")

    return {
        "model": model,
        "mu": mu,
        "sd": sd,
        "W": W,
        "b": b,
        "rf_dim": m,
        "rf_scale": lengthscale,
        "rf_include_linear": bool(rf_include_linear),
        "input_dim": d_in,
    }


def apply_random_feature_state(X: np.ndarray, state: Dict[str, object]) -> np.ndarray:
    model = str(state.get("model", "linear")).lower()
    X = np.asarray(X, dtype=float)
    if model == "linear":
        return X

    X_base = X[:, 1:] if X.shape[1] > 1 else X
    mu = np.asarray(state["mu"], dtype=float)
    sd = np.asarray(state["sd"], dtype=float)
    Z = (X_base - mu) / sd
    W = np.asarray(state["W"], dtype=float)
    b = np.asarray(state["b"], dtype=float)
    A = Z @ W.T + b
    m = max(int(W.shape[0]), 1)

    if model in ("rff", "fourier", "random_fourier"):
        R = np.sqrt(2.0 / m) * np.cos(A)
    elif model in ("relu", "relu_rf", "random_relu"):
        R = np.sqrt(2.0 / m) * np.maximum(A, 0.0)
    elif model in ("tanh", "tanh_rf"):
        R = np.tanh(A) / np.sqrt(float(m))
    else:
        raise ValueError(f"Unknown random-feature model='{model}'.")

    parts = [np.ones((X.shape[0], 1), dtype=float)]
    if bool(state.get("rf_include_linear", True)):
        parts.append(Z)
    parts.append(R)
    return np.concatenate(parts, axis=1)


def apply_random_features_to_splits(
    rng: np.random.Generator,
    X_fit: np.ndarray,
    X_thr: np.ndarray,
    X_proj: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    posterior_model: str = "linear",
    rf_dim: int = 256,
    rf_scale: float = 1.0,
    rf_include_linear: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    state = make_random_feature_state(
        rng,
        X_fit,
        posterior_model=posterior_model,
        rf_dim=rf_dim,
        rf_scale=rf_scale,
        rf_include_linear=rf_include_linear,
    )
    return (
        apply_random_feature_state(X_fit, state),
        apply_random_feature_state(X_thr, state),
        apply_random_feature_state(X_proj, state),
        apply_random_feature_state(X_cal, state),
        apply_random_feature_state(X_test, state),
        state,
    )


# ============================================================
# Gibbs posterior: squared loss + Gaussian prior => Gaussian posterior
# ============================================================

def fit_gibbs_gaussian_posterior(X: np.ndarray, y: np.ndarray, eta: float, tau2: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, bool]]:
    """
    q(β) ∝ exp(-η ||y - Xβ||^2) * N(0, diag(tau2))

    Precision: P = 2η X^T X + diag(1/tau2)
    Mean:      μ = P^{-1} (2η X^T y)

    tau2 can be scalar or length-d vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    d = X.shape[1]
    tau2 = np.asarray(tau2, dtype=float)
    if tau2.ndim == 0:
        tau2 = float(tau2) * np.ones(d, dtype=float)
    if tau2.shape != (d,):
        raise ValueError(f"tau2 must be scalar or shape ({d},), got {tau2.shape}")

    XtX = X.T @ X
    P = 2.0 * float(eta) * XtX + np.diag(1.0 / tau2)
    h = 2.0 * float(eta) * (X.T @ y)
    c, lower = cho_factor(P, lower=True, check_finite=False)
    mu = cho_solve((c, lower), h, check_finite=False)
    return mu, (c, lower)


def sample_from_precision_cholesky(rng: np.random.Generator, mu: np.ndarray, cho_prec: Tuple[np.ndarray, bool], S: int) -> np.ndarray:
    """
    If precision P = L L^T, then Cov = P^{-1} = L^{-T} L^{-1}.
    Sample: β = μ + L^{-1} z, z~N(0,I).
    """
    L, lower = cho_prec
    d = mu.shape[0]
    z = rng.standard_normal(size=(int(S), d))
    u = solve_triangular(L, z.T, lower=lower, check_finite=False).T
    return mu[None, :] + u


# ============================================================
# Constraint features g(beta) computed on the projection set
# ============================================================

def compute_g_global_and_group_blockwise(
    betas: np.ndarray,
    X_proj: np.ndarray,
    y_proj: np.ndarray,
    group_proj: np.ndarray,
    thresholds: np.ndarray,
    n_groups: int,
    block_size: int = 1500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each beta sample, compute:
      g_global_k(beta) = mean_j 1{|y_j - x_j^T beta| > t_k}
      g_group_{g,k}(beta) = mean_{j: G_j=g} 1{|y_j - x_j^T beta| > t_k}

    Returns:
      g_global: (S, K)
      g_group:  (S, G*K), ordering [g0k0,g0k1,...,g1k0,...]
    """
    thresholds = np.asarray(thresholds, dtype=float)
    betas = np.asarray(betas, dtype=float)

    S = betas.shape[0]
    K = thresholds.size

    g_global = np.empty((S, K), dtype=np.float32)
    g_group = np.empty((S, n_groups * K), dtype=np.float32)

    group_proj = np.asarray(group_proj)
    idx_by_group = [np.where(group_proj == g)[0] for g in range(n_groups)]
    sizes = np.array([idx.size for idx in idx_by_group], dtype=int)

    # (m, B) matrix multiplies dominate; keep blocks moderate to avoid RAM spikes
    for start in range(0, S, int(block_size)):
        end = min(S, start + int(block_size))
        betab = betas[start:end]
        preds = X_proj @ betab.T                      # (m, B)
        resid = np.abs(y_proj[:, None] - preds)       # (m, B)

        for k, t in enumerate(thresholds):
            exc = resid > t                           # bool (m, B)
            g_global[start:end, k] = exc.mean(axis=0)
            for g in range(n_groups):
                if sizes[g] == 0:
                    g_group[start:end, g * K + k] = 0.0
                else:
                    g_group[start:end, g * K + k] = exc[idx_by_group[g], :].mean(axis=0)

    return g_global.astype(float), g_group.astype(float)


# ============================================================
# Dual solve (sample-based convex program)
# ============================================================

def solve_dual_lbfgsb(
    g: np.ndarray,
    alpha: np.ndarray,
    maxiter: int = 3000,
    maxfun: int = 200000,
    restarts: int = 2,
    verbose: bool = False,
    *,
    # robust fallback (helps on real data where the optimum can be extremely sharp)
    newton_fallback: bool = True,
    newton_maxiter: int = 80,
    newton_tol: float = 1e-8,
    newton_ridge: float = 1e-8,
    lbfgsb_maxls: int = 80,
    lbfgsb_gtol: float = 1e-10,
):
    """
    Solve the convex dual (sample-based I-projection):

        min_{λ >= 0}  log( mean_s exp( - g_s^T λ ) ) + α^T λ

    Parameters
    ----------
    g : array, shape (S, K)
        Constraint features computed on the projection set for each q0 sample.
        Here g_{s,k} ≈ P_proj(|Y - Xβ_s| > t_k) (global) or the groupwise analogue.
    alpha : array, shape (K,)
        Target tail levels (possibly tightened). Constraints are E_q[g_k] <= alpha_k.
    maxiter, maxfun, restarts : L-BFGS-B controls
    verbose : bool
        Print diagnostics.
    newton_fallback : bool
        If True, fall back to a damped projected-Newton method when L-BFGS-B returns
        an 'ABNORMAL' termination or any non-success status. This is usually more
        robust on real data where the optimum can be highly ill-conditioned.
    newton_maxiter, newton_tol, newton_ridge : projected-Newton controls

    Returns
    -------
    lmbda : array, shape (K,)
    w     : array, shape (S,)
        Normalized pool weights w_s ∝ exp(-g_s^T λ).
    ess   : float
        Importance-sampling effective sample size = 1 / sum(w^2).
    logZ  : float
        log Z where Z = mean_s exp(-g_s^T λ).
    Eg    : array, shape (K,)
        Eg = sum_s w_s g_s.
    """
    g = np.asarray(g, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    S, K = g.shape
    assert alpha.shape == (K,)

    # ---- shared objective/gradient helpers ----
    def _obj_grad(lmbda: np.ndarray, need_hess: bool = False):
        a = -g @ lmbda  # (S,)
        lse = logsumexp(a)
        logZ = lse - np.log(S)
        f = logZ + alpha.dot(lmbda)

        logw = a - lse
        w = np.exp(logw)  # sums to 1
        Eg = w @ g
        grad = alpha - Eg  # gradient of f

        if not need_hess:
            return f, grad, w, logZ, Eg

        # Hessian = Cov_w(g)
        gc = g - Eg  # (S,K)
        H = (gc.T * w) @ gc  # (K,K)
        return f, grad, w, logZ, Eg, H

    # ---- (1) Try L-BFGS-B first (fast when it works) ----
    bounds = [(0.0, None)] * K

    def fun_with_grad(x):
        f, grad, *_ = _obj_grad(x, need_hess=False)
        return f, grad

    x0 = np.zeros(K, dtype=float)
    last_res = None
    for attempt in range(int(restarts) + 1):
        maxfun_attempt = int(maxfun * (5 ** attempt))
        last_res = minimize(
            fun_with_grad,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(maxiter),
                "maxfun": int(maxfun_attempt),
                "ftol": 1e-12,
                "gtol": float(lbfgsb_gtol),
                "maxls": int(lbfgsb_maxls),
            },
        )
        x0 = np.maximum(0.0, last_res.x)
        if last_res.success:
            break

    res = last_res
    use_newton = False
    if (res is None) or (not np.all(np.isfinite(res.x))) or (not np.isfinite(res.fun)):
        use_newton = True
    if (res is not None) and (not res.success) and newton_fallback:
        # SciPy often reports 'ABNORMAL' for sharp optima; Newton is more stable.
        use_newton = True

    if (res is not None) and (not res.success) and (not use_newton):
        warnings.warn(f"Dual solver warning (after restarts): {res.message}")

    # ---- (2) Robust fallback: damped projected-Newton ----
    if use_newton and newton_fallback:
        if verbose:
            msg = res.message if res is not None else "(no LBFGS result)"
            print(f"[dual] L-BFGS-B did not converge ({msg}); falling back to projected Newton")

        lmbda = np.maximum(0.0, x0)

        for it in range(int(newton_maxiter)):
            f, grad, w, logZ, Eg, H = _obj_grad(lmbda, need_hess=True)

            # projected gradient for the bound λ>=0
            pg = grad.copy()
            on_boundary = lmbda <= 1e-12
            pg[on_boundary] = np.minimum(pg[on_boundary], 0.0)

            pg_inf = float(np.max(np.abs(pg)))
            if pg_inf < float(newton_tol):
                break

            # stabilize Hessian
            H = np.asarray(H, dtype=float)
            H.flat[:: K + 1] += float(newton_ridge)

            try:
                step_dir = np.linalg.solve(H, pg)  # Newton step on projected gradient
            except np.linalg.LinAlgError:
                # fallback to (projected) gradient if Hessian is singular
                step_dir = pg

            dec = float(pg.dot(step_dir))  # should be >=0
            if not np.isfinite(dec) or dec <= 0.0:
                step_dir = pg
                dec = float(pg.dot(step_dir))

            # backtracking line search with projection
            step = 1.0
            accepted = False
            for _ in range(60):
                cand = np.maximum(0.0, lmbda - step * step_dir)
                f_cand, *_ = _obj_grad(cand, need_hess=False)
                if f_cand <= f - 1e-4 * step * dec:
                    lmbda = cand
                    accepted = True
                    break
                step *= 0.5
                if step < 1e-12:
                    break
            if not accepted:
                lmbda = np.maximum(0.0, lmbda - 0.1 * step_dir)

        # compute final quantities
        f, grad, w, logZ, Eg = _obj_grad(lmbda, need_hess=False)
        ess = 1.0 / np.sum(w ** 2)

        if verbose:
            max_viol = float(np.max(Eg - alpha))
            pg = grad.copy()
            pg[lmbda <= 1e-12] = np.minimum(pg[lmbda <= 1e-12], 0.0)
            pg_inf = float(np.max(np.abs(pg)))
            print(f"[dual-newton] it={it+1} | max_viol={max_viol:.3e} | pg_inf={pg_inf:.3e} | ESS={ess:.1f}")

        return lmbda, w, float(ess), float(logZ), Eg

    # ---- (3) Use the L-BFGS-B result ----
    lmbda = np.maximum(0.0, res.x)

    f, grad, w, logZ, Eg = _obj_grad(lmbda, need_hess=False)
    ess = 1.0 / np.sum(w ** 2)

    if verbose:
        max_viol = float(np.max(Eg - alpha))
        pg = grad.copy()
        pg[lmbda <= 1e-12] = np.minimum(pg[lmbda <= 1e-12], 0.0)
        pg_inf = float(np.max(np.abs(pg)))
        print(f"[dual-lbfgsb] success={res.success} | max_viol={max_viol:.3e} | pg_inf={pg_inf:.3e} | ESS={ess:.1f}")

    return lmbda, w, float(ess), float(logZ), Eg



# ============================================================
# SMC/AIS stabilization on a precomputed q0 pool (cheap)
# ============================================================

def systematic_resample(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    N = weights.size
    weights = weights / weights.sum()
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    idx = np.zeros(N, dtype=int)
    i = j = 0
    while i < N:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


def smc_sample_from_pool(
    rng: np.random.Generator,
    betas_pool: np.ndarray,
    g_pool: np.ndarray,
    lmbda: np.ndarray,
    n_particles: int = 3000,
    n_steps: int = 60,
    ess_threshold: float = 0.5,
    rejuvenate_steps: int = 4,
    *,
    final_resample: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Discrete-pool SMC approximation to:

        p(i) ∝ exp( - λ^T g_i )

    We temper from s=0 to s=1:

        p_s(i) ∝ exp( - s λ^T g_i )

    Parameters
    ----------
    final_resample : bool
        If True (default), returns an *unweighted* sample (uniform weights) via a final
        systematic resample. If False, returns the weighted particle approximation at s=1.
        Setting final_resample=False is often preferable on real datasets where the target
        can be very concentrated (otherwise the returned sample may contain many duplicates).

    Returns
    -------
    betas : (n_particles, d)
    weights : (n_particles,)
        Either uniform (if final_resample=True) or the final SMC weights (if False).
    info : dict
        Diagnostics: ess_min, ess_final, resamples, unique_frac, mh_accept.
    """
    betas_pool = np.asarray(betas_pool, dtype=float)
    g_pool = np.asarray(g_pool, dtype=float)
    lmbda = np.asarray(lmbda, dtype=float)

    S_pool = betas_pool.shape[0]
    n_particles = int(n_particles)

    phi_pool = g_pool @ lmbda  # (S_pool,)

    idx = rng.integers(0, S_pool, size=n_particles)
    weights = np.ones(n_particles, dtype=float) / n_particles

    schedule = np.linspace(0.0, 1.0, int(n_steps) + 1)[1:]
    s_prev = 0.0

    ess_min = float(n_particles)
    resamples = 0
    accept_total = 0
    accept_trials = 0

    for s in schedule:
        ds = s - s_prev
        if ds > 0:
            # incremental reweight
            weights *= np.exp(-ds * phi_pool[idx])
            sw = weights.sum()
            if (sw == 0.0) or (not np.isfinite(sw)):
                # numerical rescue
                logw = -ds * phi_pool[idx]
                logw -= logw.max()
                weights = np.exp(logw)
                sw = weights.sum()
            weights /= sw

        ess = 1.0 / np.sum(weights ** 2)
        ess_min = min(ess_min, float(ess))

        # resample if ESS is low
        if ess < float(ess_threshold) * n_particles:
            rs = systematic_resample(rng, weights)
            idx = idx[rs]
            weights.fill(1.0 / n_particles)
            resamples += 1

        # cheap rejuvenation: independence MH on indices
        # (still useful, but acceptance can be tiny when the tilt is very sharp)
        if rejuvenate_steps > 0:
            for _ in range(int(rejuvenate_steps)):
                prop = rng.integers(0, S_pool, size=n_particles)
                loga = -s * (phi_pool[prop] - phi_pool[idx])
                u = np.log(rng.random(n_particles))
                accept = u < np.minimum(0.0, loga)
                idx[accept] = prop[accept]
                accept_total += int(accept.sum())
                accept_trials += n_particles

        s_prev = s

    # normalize one last time at s=1
    weights /= weights.sum()
    ess_final = float(1.0 / np.sum(weights ** 2))
    unique_frac = float(np.unique(idx).size / n_particles)
    accept_rate = float(accept_total / max(1, accept_trials))

    info = {
        "ess_min": float(ess_min),
        "ess_final": float(ess_final),
        "resamples": int(resamples),
        "unique_frac": float(unique_frac),
        "mh_accept": float(accept_rate),
    }

    if final_resample:
        rs = systematic_resample(rng, weights)
        idx = idx[rs]
        weights = np.ones(n_particles, dtype=float) / n_particles
        info["unique_frac"] = float(np.unique(idx).size / n_particles)
        info["ess_final"] = float(n_particles)

    return betas_pool[idx], weights, info



def weighted_mean(betas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute a weighted mean of rows of `betas` (shape: S x d)."""
    betas = np.asarray(betas, dtype=float)
    w = np.asarray(weights, dtype=float)
    sw = float(w.sum())
    if (sw <= 0.0) or (not np.isfinite(sw)):
        w = np.ones_like(w) / max(1, w.size)
    else:
        w = w / sw
    return w @ betas

# ============================================================
# Evaluation metrics
# ============================================================

def miscoverage_blockwise(
    betas: np.ndarray,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: np.ndarray,
    block_size: int = 500,
) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=float)
    betas = np.asarray(betas, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    K = thresholds.size
    S = betas.shape[0]
    out = np.zeros(K, dtype=float)

    for start in range(0, S, int(block_size)):
        end = min(S, start + int(block_size))
        betab = betas[start:end]
        wb = w[start:end]
        preds = X @ betab.T
        resid = np.abs(y[:, None] - preds)
        for k, t in enumerate(thresholds):
            mis_s = (resid > t).mean(axis=0)
            out[k] += wb.dot(mis_s)

    return out


def group_miscoverage_blockwise(
    betas: np.ndarray,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    thresholds: np.ndarray,
    n_groups: int,
    block_size: int = 500,
) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=float)
    betas = np.asarray(betas, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    group = np.asarray(group)
    idx_by_group = [np.where(group == g)[0] for g in range(n_groups)]
    sizes = np.array([idx.size for idx in idx_by_group], dtype=int)

    K = thresholds.size
    S = betas.shape[0]
    out = np.zeros((n_groups, K), dtype=float)

    for start in range(0, S, int(block_size)):
        end = min(S, start + int(block_size))
        betab = betas[start:end]
        wb = w[start:end]
        preds = X @ betab.T
        resid = np.abs(y[:, None] - preds)

        for g in range(n_groups):
            if sizes[g] == 0:
                continue
            resid_g = resid[idx_by_group[g], :]
            for k, t in enumerate(thresholds):
                mis_s = (resid_g > t).mean(axis=0)
                out[g, k] += wb.dot(mis_s)

    return out


def tail_curve_blockwise(
    betas: np.ndarray,
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    t_grid: np.ndarray,
    block_size: int = 500,
) -> np.ndarray:
    t_grid = np.asarray(t_grid, dtype=float)
    betas = np.asarray(betas, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    T = t_grid.size
    S = betas.shape[0]
    out = np.zeros(T, dtype=float)

    for start in range(0, S, int(block_size)):
        end = min(S, start + int(block_size))
        betab = betas[start:end]
        wb = w[start:end]
        preds = X @ betab.T
        resid = np.abs(y[:, None] - preds)
        for i, t in enumerate(t_grid):
            mis_s = (resid > t).mean(axis=0)
            out[i] += wb.dot(mis_s)

    return out


def worst_violation_group(gm: np.ndarray, alpha_levels: np.ndarray) -> float:
    alpha_levels = np.asarray(alpha_levels, dtype=float)
    return float(np.max(np.maximum(gm - alpha_levels[None, :], 0.0)))


# ============================================================
# Baselines: ridge / Huber + split/Mondrian conformal
# ============================================================

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """
    Ridge with *unpenalized intercept* (assumes X includes intercept column at index 0).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    P = np.eye(d)
    P[0, 0] = 0.0  # no penalty on intercept
    A = X.T @ X + float(lam) * P
    b = X.T @ y
    return np.linalg.solve(A, b)


def fit_huber_irls(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1.0,
    delta: float = 1.345,
    maxiter: int = 60,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    IRLS for Huber regression with ridge penalty (intercept unpenalized).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    beta = fit_ridge(X, y, lam)

    P = np.eye(d)
    P[0, 0] = 0.0  # intercept not penalized

    for _ in range(int(maxiter)):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r)) + 1e-8
        u = r / s

        w = np.ones(n)
        mask = np.abs(u) > float(delta)
        w[mask] = float(delta) / np.abs(u[mask])

        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw

        A = Xw.T @ Xw + float(lam) * P
        b = Xw.T @ yw
        beta_new = np.linalg.solve(A, b)

        if np.linalg.norm(beta_new - beta) <= float(tol) * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new

    return beta


def split_conformal_halfwidth(y_cal: np.ndarray, yhat_cal: np.ndarray, alpha: float) -> float:
    resid = np.abs(y_cal - yhat_cal)
    return conformal_quantile(resid, alpha)


def mondrian_halfwidth_by_group(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    g_cal: np.ndarray,
    alpha: float,
    n_groups: int,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for g in range(int(n_groups)):
        idx = np.where(g_cal == g)[0]
        if idx.size == 0:
            out[g] = 0.0
        else:
            resid = np.abs(y_cal[idx] - yhat_cal[idx])
            out[g] = conformal_quantile(resid, alpha)
    return out


# ============================================================
# TempTune baseline: tune eta to minimize WORST-group violations on projection set
# ============================================================

def tune_eta_grid(
    rng: np.random.Generator,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_proj: np.ndarray,
    y_proj: np.ndarray,
    g_proj: np.ndarray,
    thresholds: np.ndarray,
    alpha_group_tight: np.ndarray,   # (G,K)
    eta_grid: np.ndarray,
    tau2: np.ndarray,
    n_groups: int,
    S_tune: int = 2000,
    block_size_g: int = 1500,
) -> Tuple[float, np.ndarray, Tuple[np.ndarray, bool]]:
    """
    Grid-search eta for Gibbs posterior to minimize worst-group violation:
      loss(eta) = max_{g,k} max( E[g_{g,k}(beta)] - alpha_{g,k}, 0 )
    on the projection set.

    Returns best_eta, posterior mean, posterior precision cholesky.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    alpha_group_tight = np.asarray(alpha_group_tight, dtype=float)

    best_eta = None
    best_loss = np.inf
    best_mu, best_cho = None, None

    for eta in np.asarray(eta_grid, dtype=float):
        mu, cho = fit_gibbs_gaussian_posterior(X_fit, y_fit, float(eta), tau2)
        betas = sample_from_precision_cholesky(rng, mu, cho, int(S_tune))

        _, g_grp = compute_g_global_and_group_blockwise(
            betas, X_proj, y_proj, g_proj, thresholds, n_groups=int(n_groups), block_size=int(block_size_g)
        )
        Eg_grp = g_grp.reshape(int(S_tune), int(n_groups), -1).mean(axis=0)  # (G,K)
        viol_grp = np.maximum(Eg_grp - alpha_group_tight, 0.0)
        loss = float(np.max(viol_grp))

        if loss < best_loss:
            best_loss = loss
            best_eta = float(eta)
            best_mu, best_cho = mu, cho

    if best_eta is None or best_mu is None or best_cho is None:
        raise RuntimeError("TempTune failed to select an eta (unexpected).")

    return best_eta, best_mu, best_cho


def fit_gibbs_gaussian_posterior_group_eta(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    eta_by_group: np.ndarray,
    tau2: np.ndarray,
) -> Tuple[np.ndarray, Tuple[np.ndarray, bool]]:
    """
    Gibbs posterior with one squared-loss learning rate per training group.

    q(beta) proportional to exp{-sum_i eta_{g_i} (y_i-x_i^T beta)^2} times
    N(0, diag(tau2)).  This is a single posterior and is therefore a stronger
    scalar-temperature baseline than TempTune.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    group = np.asarray(group, dtype=int)
    eta_by_group = np.asarray(eta_by_group, dtype=float)
    w = eta_by_group[group]

    d = X.shape[1]
    tau2 = np.asarray(tau2, dtype=float)
    if tau2.ndim == 0:
        tau2 = float(tau2) * np.ones(d, dtype=float)
    if tau2.shape != (d,):
        raise ValueError(f"tau2 must be scalar or shape ({d},), got {tau2.shape}")

    Xw = X * np.sqrt(2.0 * w)[:, None]
    P = Xw.T @ Xw + np.diag(1.0 / tau2)
    h = 2.0 * (X.T @ (w * y))
    c, lower = cho_factor(P, lower=True, check_finite=False)
    mu = cho_solve((c, lower), h, check_finite=False)
    return mu, (c, lower)


def tune_group_eta_coordinate_grid(
    rng: np.random.Generator,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    g_fit: np.ndarray,
    X_proj: np.ndarray,
    y_proj: np.ndarray,
    g_proj: np.ndarray,
    thresholds: np.ndarray,
    alpha_group_tight: np.ndarray,
    eta_grid: np.ndarray,
    tau2: np.ndarray,
    n_groups: int,
    eta_init: float,
    S_tune: int = 1000,
    passes: int = 2,
    block_size_g: int = 500,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, bool], float]:
    """
    Coordinate grid search for the GroupTemp baseline.

    GroupTemp keeps a single posterior but gives the training loss one learning
    rate per training group.  The eta vector is tuned on the projection split
    to minimize the same worst-group multi-threshold violation as TempTune.
    """
    G = int(n_groups)
    eta_grid = np.asarray(eta_grid, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    alpha_group_tight = np.asarray(alpha_group_tight, dtype=float)
    eta_vec = np.full(G, float(eta_init), dtype=float)

    def objective(evec: np.ndarray) -> Tuple[float, np.ndarray, Tuple[np.ndarray, bool]]:
        mu, cho = fit_gibbs_gaussian_posterior_group_eta(X_fit, y_fit, g_fit, evec, tau2)
        betas = sample_from_precision_cholesky(rng, mu, cho, int(S_tune))
        _, g_grp = compute_g_global_and_group_blockwise(
            betas, X_proj, y_proj, g_proj, thresholds, n_groups=G, block_size=int(block_size_g)
        )
        Eg = g_grp.reshape(int(S_tune), G, -1).mean(axis=0)
        loss = float(np.max(np.maximum(Eg - alpha_group_tight, 0.0)))
        return loss, mu, cho

    best_loss, best_mu, best_cho = objective(eta_vec)
    for _ in range(int(passes)):
        improved = False
        for g in range(G):
            local_eta = eta_vec[g]
            local_loss = best_loss
            local_mu, local_cho = best_mu, best_cho
            for eta in eta_grid:
                cand = eta_vec.copy()
                cand[g] = float(eta)
                loss, mu, cho = objective(cand)
                if loss < local_loss - 1e-12:
                    local_eta = float(eta)
                    local_loss = loss
                    local_mu, local_cho = mu, cho
            if local_loss < best_loss - 1e-12:
                eta_vec[g] = local_eta
                best_loss = local_loss
                best_mu, best_cho = local_mu, local_cho
                improved = True
        if not improved:
            break

    # Refit exactly at the selected vector so returned objects match eta_vec.
    best_mu, best_cho = fit_gibbs_gaussian_posterior_group_eta(X_fit, y_fit, g_fit, eta_vec, tau2)
    return eta_vec, best_mu, best_cho, float(best_loss)


# ============================================================
# Main real-data runner (per dataset)
# ============================================================

def run_real_data_dataset(
    cfg: DatasetConfig,
    outdir: str,
    seed: int = 0,
    R: int = 20,
    # posterior hyperparams (on standardized y-scale)
    eta0: float = 0.20,
    eta_ref: float = 2.00,
    tau2: float = 25.0,
    tau2_intercept: float = 1e6,
    # constraints
    K: int = 5,
    tighten_delta: float = 0.20,
    tighten_factor: float = 0.10,
    # Monte Carlo
    S_pool: int = 40000,
    n_particles: int = 4000,
    # SMC
    smc_steps_global: int = 35,
    smc_steps_group: int = 70,
    ess_threshold: float = 0.40,
    rejuvenate_global: int = 2,
    rejuvenate_group: int = 6,
    # TempTune
    eta_grid: Optional[np.ndarray] = None,
    S_tune: int = 2000,
    S_group_tune: Optional[int] = None,
    group_eta_passes: int = 2,
    # evaluation
    eval_block: int = 400,
    make_figures: bool = True,
    compute_conformal: bool = True,
    include_group_in_X: bool = False,
    drop_first_categorical: bool = False,
    tau2_group_feature: Optional[float] = None,
    posterior_model: str = "linear",
    rf_dim: int = 256,
    rf_scale: float = 1.0,
    rf_include_linear: bool = True,
    run_label: Optional[str] = None,
    verbose: bool = True,
    save_tail_curves: bool = False,
    cache_dir_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run CIP + baselines on one real dataset with R repeated random splits.
    """
    set_mpl_style()

    ensure_dir(outdir)
    figdir = os.path.join(outdir, "figures")
    if make_figures:
        ensure_dir(figdir)

    cache_dir = cache_dir_override if cache_dir_override is not None else os.path.join(outdir, "_cache")
    ensure_dir(cache_dir)

    # Load data
    df = cfg.loader(cache_dir)

    # Basic sanitation: drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Drop rows with missing target/group *before splitting* so indices stay consistent
    df = df.loc[~df[cfg.target_col].isna()].copy()
    df = df.loc[~df[cfg.group_col].isna()].copy()
    df = df.reset_index(drop=True)

    # Keep only columns that exist
    if cfg.target_col not in df.columns:
        raise ValueError(f"[{cfg.name}] target_col='{cfg.target_col}' not found in columns.")
    if cfg.group_col not in df.columns:
        raise ValueError(f"[{cfg.name}] group_col='{cfg.group_col}' not found in columns.")

    # Optionally subsample for speed (comment out if you want full data)
    # df = df.sample(n=min(len(df), 60000), random_state=seed).reset_index(drop=True)

    alpha_levels = alpha_levels_for_K(K)
    K = int(alpha_levels.size)

    # key alpha for "t_key" plots (nearest to 0.10)
    key_alpha = 0.10
    key_k = int(np.argmin(np.abs(alpha_levels - key_alpha)))

    if eta_grid is None:
        eta_grid = np.logspace(-2, 2, 21)
    if S_group_tune is None:
        S_group_tune = max(800, int(S_tune) // 2)

    rng_master = np.random.default_rng(int(seed))

    rows: List[Dict[str, float]] = []
    tail_curves = {"q0": [], "temp": [], "group_temp": [], "cip_global": [], "cip_group": []}
    t_list: List[np.ndarray] = []
    group_labels_for_plots: Optional[List[str]] = None

    # Fixed tail grid for averaging tail curves across splits (standardized y-scale)
    tail_t_grid = np.linspace(0.0, 12.0, 121)

    for rep in range(int(R)):
        t0_total = time.perf_counter()
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))

        # Build (possibly collapsed) group codes BEFORE splitting, so stratification is consistent
        g_raw = df[cfg.group_col].astype(object)
        if cfg.group_top_k is not None:
            g_raw = collapse_groups_to_top_k(g_raw, top_k=int(cfg.group_top_k), other_label="Other")
        g_levels = sorted(pd.unique(g_raw.fillna("Other")))
        g_to_code = {g: i for i, g in enumerate(g_levels)}
        g_code_all = g_raw.fillna("Other").map(g_to_code).to_numpy(dtype=int)
        n_groups = int(len(g_levels))
        if group_labels_for_plots is None:
            group_labels_for_plots = [str(x) for x in g_levels]

        # Five-way stratified split
        idx_fit, idx_thr, idx_proj, idx_cal, idx_test = split_five_way_stratified(
            rng,
            g_code_all,
            frac_fit=cfg.frac_fit,
            frac_thr=cfg.frac_thr,
            frac_proj=cfg.frac_proj,
            frac_cal=cfg.frac_cal,
            frac_test=cfg.frac_test,
        )

        # Preprocess into matrices
        data = preprocess_splits(
            df=df,
            target_col=cfg.target_col,
            group_col=cfg.group_col,
            idx_fit=idx_fit,
            idx_thr=idx_thr,
            idx_proj=idx_proj,
            idx_cal=idx_cal,
            idx_test=idx_test,
            group_top_k=cfg.group_top_k,
            include_group_in_X=include_group_in_X,
            y_transform_name=cfg.y_transform,
            drop_first_categorical=drop_first_categorical,
        )

        X_fit = data["X_fit"]; y_fit = data["y_fit"]
        X_thr = data["X_thr"]; y_thr = data["y_thr"]; g_thr = data["g_thr"]
        X_proj = data["X_proj"]; y_proj = data["y_proj"]; g_proj = data["g_proj"]
        X_cal = data["X_cal"];  y_cal  = data["y_cal"];  g_cal  = data["g_cal"]
        X_test = data["X_test"]; y_test = data["y_test"]; g_test = data["g_test"]

        y_mu = float(data["y_mu"][0]); y_sd = float(data["y_sd"][0])
        y_transform_name = str(data["y_transform"][0])

        # Optional nonlinear random-feature posterior.  The random feature map is
        # built using D_fit only, then applied to all splits.
        feature_dim_input = int(X_fit.shape[1])
        t_feature = time.perf_counter()
        X_fit, X_thr, X_proj, X_cal, X_test, rf_state = apply_random_features_to_splits(
            rng, X_fit, X_thr, X_proj, X_cal, X_test,
            posterior_model=str(posterior_model),
            rf_dim=int(rf_dim),
            rf_scale=float(rf_scale),
            rf_include_linear=bool(rf_include_linear),
        )
        time_feature_map = time.perf_counter() - t_feature
        feature_dim_posterior = int(X_fit.shape[1])

        # Prior variances: intercept unregularized (approx)
        d = X_fit.shape[1]
        tau2_vec = float(tau2) * np.ones(d, dtype=float)
        tau2_vec[0] = float(tau2_intercept)

        if tau2_group_feature is not None and str(posterior_model).lower() in ("linear", "none"):
            feature_names = [str(x) for x in data.get("feature_names", [])]
            prefix = f"cat__{cfg.group_col}_"
            group_mask = np.array([name.startswith(prefix) for name in feature_names], dtype=bool)
            if group_mask.size == d and np.any(group_mask):
                tau2_vec[group_mask] = float(tau2_group_feature)
            else:
                warnings.warn(
                    f"tau2_group_feature={tau2_group_feature} requested, but no feature names "
                    f"with prefix '{prefix}' were found."
                )
        elif tau2_group_feature is not None:
            warnings.warn(
                "tau2_group_feature is ignored for nonlinear random-feature posteriors; "
                "the prior is applied uniformly to non-intercept last-layer weights."
            )

        # ------------------------------------------------------------
        # q0 and reference predictor
        # ------------------------------------------------------------
        mu0, cho0 = fit_gibbs_gaussian_posterior(X_fit, y_fit, float(eta0), tau2_vec)
        mu_ref, _ = fit_gibbs_gaussian_posterior(X_fit, y_fit, float(eta_ref), tau2_vec)

        # ------------------------------------------------------------
        # Tail thresholds (t_k) computed from reference residuals on thr
        # ------------------------------------------------------------
        resid_ref = np.abs(y_thr - X_thr @ mu_ref)
        t_vec = np.array([conformal_quantile(resid_ref, a) for a in alpha_levels], dtype=float)
        t_list.append(t_vec)


        # ------------------------------------------------------------
        # Sample pool from q0
        # ------------------------------------------------------------
        t_pool = time.perf_counter()
        betas_pool = sample_from_precision_cholesky(rng, mu0, cho0, int(S_pool))
        time_pool_sample = time.perf_counter() - t_pool

        # ------------------------------------------------------------
        # Compute constraint features on pool
        # ------------------------------------------------------------
        t_g = time.perf_counter()
        g_global_pool, g_group_pool = compute_g_global_and_group_blockwise(
            betas_pool, X_proj, y_proj, g_proj, t_vec, n_groups=int(n_groups), block_size=400
        )
        time_g_compute = time.perf_counter() - t_g

        # ------------------------------------------------------------
        # Constraint tightening (global and groupwise)
        # ------------------------------------------------------------
        alpha_global_tight, eps_global = tighten_levels(
            alpha_levels, m=int(len(idx_proj)), L=int(K), delta=float(tighten_delta), factor=float(tighten_factor)
        )

        sizes = np.array([(g_proj == g).sum() for g in range(int(n_groups))], dtype=int)
        alpha_group_tight = np.zeros((int(n_groups), int(K)), dtype=float)
        eps_group = np.zeros(int(n_groups), dtype=float)
        L_group = int(n_groups) * int(K)
        for g in range(int(n_groups)):
            alpha_group_tight[g], eps_group[g] = tighten_levels(
                alpha_levels, m=int(sizes[g]), L=L_group, delta=float(tighten_delta), factor=float(tighten_factor)
            )

        # ------------------------------------------------------------
        # Solve duals on pool
        # ------------------------------------------------------------
        t_dual = time.perf_counter()
        lam_global, w_is_global, ess_is_global, logZ_global, Eg_global = solve_dual_lbfgsb(
            g_global_pool, alpha_global_tight, maxiter=2500
        )
        time_dual_global = time.perf_counter() - t_dual

        t_dual = time.perf_counter()
        lam_group, w_is_group, ess_is_group, logZ_group, Eg_group = solve_dual_lbfgsb(
            g_group_pool, alpha_group_tight.reshape(-1), maxiter=3000
        )
        time_dual_group = time.perf_counter() - t_dual

        # approximate KLs on the pool: KL(q*||q0) = -λ^T E_q*[g] - logZ
        kl_global = float(-lam_global.dot(Eg_global) - logZ_global)
        kl_group = float(-lam_group.dot(Eg_group) - logZ_group)

        # ------------------------------------------------------------
        # Stabilized CIP sampling (SMC on pool)
        # ------------------------------------------------------------
        t_smc = time.perf_counter()
        betas_cip_global, w_cip_global, info_cip_global = smc_sample_from_pool(
            rng, betas_pool, g_global_pool, lam_global,
            n_particles=int(n_particles), n_steps=int(smc_steps_global),
            ess_threshold=float(ess_threshold), rejuvenate_steps=int(rejuvenate_global),
            final_resample=False,
        )
        time_smc_global = time.perf_counter() - t_smc

        t_smc = time.perf_counter()
        betas_cip_group, w_cip_group, info_cip_group = smc_sample_from_pool(
            rng, betas_pool, g_group_pool, lam_group,
            n_particles=int(n_particles), n_steps=int(smc_steps_group),
            ess_threshold=float(ess_threshold), rejuvenate_steps=int(rejuvenate_group),
            final_resample=False,
        )
        time_smc_group = time.perf_counter() - t_smc

        # q0 sample for evaluation: subset from pool
        if int(S_pool) >= int(n_particles):
            idx0 = rng.choice(int(S_pool), size=int(n_particles), replace=False)
        else:
            idx0 = rng.integers(0, int(S_pool), size=int(n_particles))
        betas_q0 = betas_pool[idx0]
        w_q0 = np.ones(betas_q0.shape[0], dtype=float) / betas_q0.shape[0]

        # ------------------------------------------------------------
        # TempTune baseline (tune eta for worst-group violation on proj)
        # ------------------------------------------------------------
        t_tmp = time.perf_counter()
        eta_tuned, mu_tuned, cho_tuned = tune_eta_grid(
            rng,
            X_fit, y_fit,
            X_proj, y_proj, g_proj,
            thresholds=t_vec,
            alpha_group_tight=alpha_group_tight,
            eta_grid=np.asarray(eta_grid, dtype=float),
            tau2=tau2_vec,
            n_groups=int(n_groups),
            S_tune=int(S_tune),
            block_size_g=400,
        )
        time_temp_tune = time.perf_counter() - t_tmp

        betas_temp = sample_from_precision_cholesky(rng, mu_tuned, cho_tuned, int(n_particles))
        w_temp = np.ones(betas_temp.shape[0], dtype=float) / betas_temp.shape[0]

        # ------------------------------------------------------------
        # GroupTempTune: one learning rate per training group
        # ------------------------------------------------------------
        t_gtmp = time.perf_counter()
        eta_group_tuned, mu_group_temp, cho_group_temp, group_temp_proj_loss = tune_group_eta_coordinate_grid(
            rng,
            X_fit, y_fit, data["g_fit"],
            X_proj, y_proj, g_proj,
            thresholds=t_vec,
            alpha_group_tight=alpha_group_tight,
            eta_grid=np.asarray(eta_grid, dtype=float),
            tau2=tau2_vec,
            n_groups=int(n_groups),
            eta_init=float(eta_tuned),
            S_tune=int(S_group_tune),
            passes=int(group_eta_passes),
            block_size_g=400,
        )
        time_group_temp_tune = time.perf_counter() - t_gtmp
        betas_group_temp = sample_from_precision_cholesky(rng, mu_group_temp, cho_group_temp, int(n_particles))
        w_group_temp = np.ones(betas_group_temp.shape[0], dtype=float) / betas_group_temp.shape[0]

        # ------------------------------------------------------------
        # Evaluate miscoverage at ALL constrained thresholds on TEST
        # ------------------------------------------------------------
        t_eval = time.perf_counter()
        mcov_q0 = miscoverage_blockwise(betas_q0, w_q0, X_test, y_test, t_vec, block_size=int(eval_block))
        mcov_temp = miscoverage_blockwise(betas_temp, w_temp, X_test, y_test, t_vec, block_size=int(eval_block))
        mcov_group_temp = miscoverage_blockwise(betas_group_temp, w_group_temp, X_test, y_test, t_vec, block_size=int(eval_block))
        mcov_cipg = miscoverage_blockwise(betas_cip_global, w_cip_global, X_test, y_test, t_vec, block_size=int(eval_block))
        mcov_cipgrp = miscoverage_blockwise(betas_cip_group, w_cip_group, X_test, y_test, t_vec, block_size=int(eval_block))

        gm_q0 = group_miscoverage_blockwise(betas_q0, w_q0, X_test, y_test, g_test, t_vec, n_groups=int(n_groups), block_size=int(eval_block))
        gm_temp = group_miscoverage_blockwise(betas_temp, w_temp, X_test, y_test, g_test, t_vec, n_groups=int(n_groups), block_size=int(eval_block))
        gm_group_temp = group_miscoverage_blockwise(betas_group_temp, w_group_temp, X_test, y_test, g_test, t_vec, n_groups=int(n_groups), block_size=int(eval_block))
        gm_cipg = group_miscoverage_blockwise(betas_cip_global, w_cip_global, X_test, y_test, g_test, t_vec, n_groups=int(n_groups), block_size=int(eval_block))
        gm_cipgrp = group_miscoverage_blockwise(betas_cip_group, w_cip_group, X_test, y_test, g_test, t_vec, n_groups=int(n_groups), block_size=int(eval_block))

        wv_q0_group = worst_violation_group(gm_q0, alpha_levels)
        wv_temp_group = worst_violation_group(gm_temp, alpha_levels)
        wv_group_temp_group = worst_violation_group(gm_group_temp, alpha_levels)
        wv_cipg_group = worst_violation_group(gm_cipg, alpha_levels)
        wv_cipgrp_group = worst_violation_group(gm_cipgrp, alpha_levels)

        if make_figures or save_tail_curves:
            tail_curves["q0"].append(tail_curve_blockwise(betas_q0, w_q0, X_test, y_test, tail_t_grid, block_size=int(eval_block)))
            tail_curves["temp"].append(tail_curve_blockwise(betas_temp, w_temp, X_test, y_test, tail_t_grid, block_size=int(eval_block)))
            tail_curves["group_temp"].append(tail_curve_blockwise(betas_group_temp, w_group_temp, X_test, y_test, tail_t_grid, block_size=int(eval_block)))
            tail_curves["cip_global"].append(tail_curve_blockwise(betas_cip_global, w_cip_global, X_test, y_test, tail_t_grid, block_size=int(eval_block)))
            tail_curves["cip_group"].append(tail_curve_blockwise(betas_cip_group, w_cip_group, X_test, y_test, tail_t_grid, block_size=int(eval_block)))

        time_eval = time.perf_counter() - t_eval

        # ------------------------------------------------------------
        # Accuracy: posterior mean predictors (MSE on standardized transformed scale)
        # ------------------------------------------------------------
        beta_mean_q0 = betas_q0.mean(axis=0)
        beta_mean_temp = betas_temp.mean(axis=0)
        beta_mean_group_temp = betas_group_temp.mean(axis=0)
        beta_mean_cipg = weighted_mean(betas_cip_global, w_cip_global)
        beta_mean_cipgrp = weighted_mean(betas_cip_group, w_cip_group)

        yhat_q0 = X_test @ beta_mean_q0
        yhat_temp = X_test @ beta_mean_temp
        yhat_group_temp = X_test @ beta_mean_group_temp
        yhat_cipg = X_test @ beta_mean_cipg
        yhat_cipgrp = X_test @ beta_mean_cipgrp

        mse_q0 = float(np.mean((y_test - yhat_q0) ** 2))
        mse_temp = float(np.mean((y_test - yhat_temp) ** 2))
        mse_group_temp = float(np.mean((y_test - yhat_group_temp) ** 2))
        mse_cipg = float(np.mean((y_test - yhat_cipg) ** 2))
        mse_cipgrp = float(np.mean((y_test - yhat_cipgrp) ** 2))

        # Optional: MSE on original y-scale
        y_test_unstd = y_inverse(y_test * y_sd + y_mu, y_transform_name)
        yhat_q0_unstd = y_inverse(yhat_q0 * y_sd + y_mu, y_transform_name)
        yhat_temp_unstd = y_inverse(yhat_temp * y_sd + y_mu, y_transform_name)
        yhat_group_temp_unstd = y_inverse(yhat_group_temp * y_sd + y_mu, y_transform_name)
        yhat_cipg_unstd = y_inverse(yhat_cipg * y_sd + y_mu, y_transform_name)
        yhat_cipgrp_unstd = y_inverse(yhat_cipgrp * y_sd + y_mu, y_transform_name)

        mse_q0_orig = float(np.mean((y_test_unstd - yhat_q0_unstd) ** 2))
        mse_temp_orig = float(np.mean((y_test_unstd - yhat_temp_unstd) ** 2))
        mse_group_temp_orig = float(np.mean((y_test_unstd - yhat_group_temp_unstd) ** 2))
        mse_cipg_orig = float(np.mean((y_test_unstd - yhat_cipg_unstd) ** 2))
        mse_cipgrp_orig = float(np.mean((y_test_unstd - yhat_cipgrp_unstd) ** 2))

        # ------------------------------------------------------------
        # Conformal baselines (optional)
        # ------------------------------------------------------------
        alpha_cp = 0.10
        if compute_conformal:
            beta_ridge = fit_ridge(X_fit, y_fit, lam=1.0)
            beta_huber = fit_huber_irls(X_fit, y_fit, lam=1.0)

            # halfwidths on standardized transformed scale
            t_ridge = split_conformal_halfwidth(y_cal, X_cal @ beta_ridge, alpha_cp)
            t_huber = split_conformal_halfwidth(y_cal, X_cal @ beta_huber, alpha_cp)
            t_q0mean = split_conformal_halfwidth(y_cal, X_cal @ beta_mean_q0, alpha_cp)
            t_gtempmean = split_conformal_halfwidth(y_cal, X_cal @ beta_mean_group_temp, alpha_cp)
            t_cipgmean = split_conformal_halfwidth(y_cal, X_cal @ beta_mean_cipg, alpha_cp)
            t_cipgrpmean = split_conformal_halfwidth(y_cal, X_cal @ beta_mean_cipgrp, alpha_cp)

            # convert to original scale intervals for coverage/length
            def interval_stats(yhat_std: np.ndarray, halfwidth_std: float) -> Tuple[float, float]:
                lo = y_inverse((yhat_std - halfwidth_std) * y_sd + y_mu, y_transform_name)
                hi = y_inverse((yhat_std + halfwidth_std) * y_sd + y_mu, y_transform_name)
                cover = float(np.mean((y_test_unstd >= lo) & (y_test_unstd <= hi)))
                avg_len = float(np.mean(hi - lo))
                return cover, avg_len

            cov_ridge, len_ridge = interval_stats(X_test @ beta_ridge, t_ridge)
            cov_huber, len_huber = interval_stats(X_test @ beta_huber, t_huber)
            cov_q0mean, len_q0mean = interval_stats(yhat_q0, t_q0mean)
            cov_gtempmean, len_gtempmean = interval_stats(yhat_group_temp, t_gtempmean)
            cov_cipgmean, len_cipgmean = interval_stats(yhat_cipg, t_cipgmean)
            cov_cipgrpmean, len_cipgrpmean = interval_stats(yhat_cipgrp, t_cipgrpmean)

            # Mondrian (groupwise) conformal for ridge
            t_mond = mondrian_halfwidth_by_group(y_cal, X_cal @ beta_ridge, g_cal, alpha_cp, n_groups=int(n_groups))
            lo_m = np.empty_like(y_test_unstd, dtype=float)
            hi_m = np.empty_like(y_test_unstd, dtype=float)
            yhat_ridge = X_test @ beta_ridge
            for i in range(yhat_ridge.size):
                hw = float(t_mond[int(g_test[i])])
                lo_m[i] = y_inverse((yhat_ridge[i] - hw) * y_sd + y_mu, y_transform_name)
                hi_m[i] = y_inverse((yhat_ridge[i] + hw) * y_sd + y_mu, y_transform_name)
            cov_mond = float(np.mean((y_test_unstd >= lo_m) & (y_test_unstd <= hi_m)))
            len_mond = float(np.mean(hi_m - lo_m))

        else:
            cov_ridge = len_ridge = np.nan
            cov_huber = len_huber = np.nan
            cov_q0mean = len_q0mean = np.nan
            cov_gtempmean = len_gtempmean = np.nan
            cov_cipgmean = len_cipgmean = np.nan
            cov_cipgrpmean = len_cipgrpmean = np.nan
            cov_mond = len_mond = np.nan

        time_total = time.perf_counter() - t0_total

        # ------------------------------------------------------------
        # Record metrics
        # ------------------------------------------------------------
        row: Dict[str, float] = {
            "rep": float(rep),
            "seed": float(seed),
            "K": float(K),
            "n_groups": float(n_groups),
            "dataset_n": float(len(df)),
            "frac_fit": float(cfg.frac_fit),
            "frac_thr": float(cfg.frac_thr),
            "frac_proj": float(cfg.frac_proj),
            "frac_cal": float(cfg.frac_cal),
            "frac_test": float(cfg.frac_test),
            "tighten_factor": float(tighten_factor),
            "S_pool": float(S_pool),
            "n_particles": float(n_particles),
            "include_group_in_X": float(include_group_in_X),
            "drop_first_categorical": float(drop_first_categorical),
            "tau2_group_feature": float(tau2_group_feature) if tau2_group_feature is not None else np.nan,
            "posterior_model": str(posterior_model),
            "posterior_model_code": float({"linear": 0, "none": 0, "rff": 1, "fourier": 1, "random_fourier": 1, "relu": 2, "relu_rf": 2, "random_relu": 2, "tanh": 3, "tanh_rf": 3}.get(str(posterior_model).lower(), -1)),
            "rf_dim": float(rf_dim),
            "rf_scale": float(rf_scale),
            "rf_include_linear": float(rf_include_linear),
            "feature_dim_input": float(feature_dim_input),
            "feature_dim_posterior": float(feature_dim_posterior),
            "time_feature_map": float(time_feature_map),
            "eta0": float(eta0),
            "eta_ref": float(eta_ref),
            "eta_tuned": float(eta_tuned),
            "group_temp_proj_loss": float(group_temp_proj_loss),
            "eta_group_temp_min": float(np.min(eta_group_tuned)),
            "eta_group_temp_mean": float(np.mean(eta_group_tuned)),
            "eta_group_temp_max": float(np.max(eta_group_tuned)),
            "eta_group_temp_hit_max": float(np.any(eta_group_tuned >= np.max(eta_grid) - 1e-12)),
            "eta_grid_max": float(np.max(eta_grid)),
            "eta_tuned_hit_max": float(eta_tuned >= np.max(eta_grid) - 1e-12),
            "key_k": float(key_k),
            "key_alpha": float(alpha_levels[key_k]),
            "key_t": float(t_vec[key_k]),
            "eps_global": float(eps_global),
            "eps_group_min": float(np.min(eps_group)),
            "eps_group_max": float(np.max(eps_group)),
            "ess_is_global": float(ess_is_global),
            "ess_is_group": float(ess_is_group),
            "kl_cip_global": float(kl_global),
            "kl_cip_group": float(kl_group),
            "smc_global_essmin": float(info_cip_global["ess_min"]),
            "smc_global_unique": float(info_cip_global["unique_frac"]),
            "smc_global_accept": float(info_cip_global["mh_accept"]),
            "smc_global_resamples": float(info_cip_global["resamples"]),
            "smc_group_essmin": float(info_cip_group["ess_min"]),
            "smc_group_unique": float(info_cip_group["unique_frac"]),
            "smc_group_accept": float(info_cip_group["mh_accept"]),
            "smc_group_resamples": float(info_cip_group["resamples"]),
            "mcov_q0_key": float(mcov_q0[key_k]),
            "mcov_temp_key": float(mcov_temp[key_k]),
            "mcov_group_temp_key": float(mcov_group_temp[key_k]),
            "mcov_cip_global_key": float(mcov_cipg[key_k]),
            "mcov_cip_group_key": float(mcov_cipgrp[key_k]),
            "wv_group_q0": float(wv_q0_group),
            "wv_group_temp": float(wv_temp_group),
            "wv_group_group_temp": float(wv_group_temp_group),
            "wv_group_cip_global": float(wv_cipg_group),
            "wv_group_cip_group": float(wv_cipgrp_group),
            "mse_q0": float(mse_q0),
            "mse_temp": float(mse_temp),
            "mse_group_temp": float(mse_group_temp),
            "mse_cip_global": float(mse_cipg),
            "mse_cip_group": float(mse_cipgrp),
            "mse_q0_orig": float(mse_q0_orig),
            "mse_temp_orig": float(mse_temp_orig),
            "mse_group_temp_orig": float(mse_group_temp_orig),
            "mse_cip_global_orig": float(mse_cipg_orig),
            "mse_cip_group_orig": float(mse_cipgrp_orig),
            "cp_cov_ridge": float(cov_ridge),
            "cp_len_ridge": float(len_ridge),
            "cp_cov_huber": float(cov_huber),
            "cp_len_huber": float(len_huber),
            "cp_cov_q0mean": float(cov_q0mean),
            "cp_len_q0mean": float(len_q0mean),
            "cp_cov_group_temp_mean": float(cov_gtempmean),
            "cp_len_group_temp_mean": float(len_gtempmean),
            "cp_cov_cip_global_mean": float(cov_cipgmean),
            "cp_len_cip_global_mean": float(len_cipgmean),
            "cp_cov_cip_group_mean": float(cov_cipgrpmean),
            "cp_len_cip_group_mean": float(len_cipgrpmean),
            "mondrian_cov_ridge": float(cov_mond),
            "mondrian_len_ridge": float(len_mond),
            "time_pool_sample": float(time_pool_sample),
            "time_g_compute": float(time_g_compute),
            "time_dual_global": float(time_dual_global),
            "time_dual_group": float(time_dual_group),
            "time_smc_global": float(time_smc_global),
            "time_smc_group": float(time_smc_group),
            "time_temp_tune": float(time_temp_tune),
            "time_group_temp_tune": float(time_group_temp_tune),
            "time_eval": float(time_eval),
            "time_total": float(time_total),
        }

        for k in range(int(K)):
            row[f"alpha_{k}"] = float(alpha_levels[k])
            row[f"t_{k}"] = float(t_vec[k])
            row[f"mcov_q0_{k}"] = float(mcov_q0[k])
            row[f"mcov_temp_{k}"] = float(mcov_temp[k])
            row[f"mcov_group_temp_{k}"] = float(mcov_group_temp[k])
            row[f"mcov_cip_global_{k}"] = float(mcov_cipg[k])
            row[f"mcov_cip_group_{k}"] = float(mcov_cipgrp[k])

        for g in range(int(n_groups)):
            row[f"gm_q0_g{g}_key"] = float(gm_q0[g, key_k])
            row[f"gm_temp_g{g}_key"] = float(gm_temp[g, key_k])
            row[f"gm_group_temp_g{g}_key"] = float(gm_group_temp[g, key_k])
            row[f"gm_cip_global_g{g}_key"] = float(gm_cipg[g, key_k])
            row[f"gm_cip_group_g{g}_key"] = float(gm_cipgrp[g, key_k])
            row[f"n_thr_g{g}"] = float((g_thr == g).sum())
            row[f"n_proj_g{g}"] = float(sizes[g])
            row[f"eta_group_temp_g{g}"] = float(eta_group_tuned[g])

        if save_tail_curves and len(tail_curves["q0"]) > 0:
            curve_index = len(tail_curves["q0"]) - 1
            for jj, tt in enumerate(tail_t_grid):
                row[f"tail_t_{jj}"] = float(tt)
                row[f"tail_q0_{jj}"] = float(tail_curves["q0"][curve_index][jj])
                row[f"tail_temp_{jj}"] = float(tail_curves["temp"][curve_index][jj])
                row[f"tail_group_temp_{jj}"] = float(tail_curves["group_temp"][curve_index][jj])
                row[f"tail_cip_global_{jj}"] = float(tail_curves["cip_global"][curve_index][jj])
                row[f"tail_cip_group_{jj}"] = float(tail_curves["cip_group"][curve_index][jj])

        rows.append(row)

        if verbose:
            print(
                f"[{cfg.name}] rep {rep+1:02d}/{int(R)} | "
                f"G={n_groups} K={K} | "
                f"eta_tuned={eta_tuned:.3g} (hitmax={row['eta_tuned_hit_max']:.0f}) | "
                f"worstV: q0={wv_q0_group:.3g}, temp={wv_temp_group:.3g}, "
                f"gtemp={wv_group_temp_group:.3g}, cipGrp={wv_cipgrp_group:.3g} | "
                f"SMC uniq={info_cip_group['unique_frac']:.3f} (ESS_final={info_cip_group.get('ess_final', np.nan):.1f}) | "
                f"time={time_total:.1f}s"
            )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    # Save per-replicate dense tail curves for job-array aggregation.
    try:
        np.savez_compressed(
            os.path.join(outdir, "tail_curves_by_rep.npz"),
            tail_t_grid=np.asarray(tail_t_grid, dtype=float),
            t_matrix=np.vstack(t_list) if len(t_list) else np.empty((0, int(K))),
            alpha_levels=np.asarray(alpha_levels, dtype=float),
            q0=np.vstack(tail_curves["q0"]) if len(tail_curves["q0"]) else np.empty((0, len(tail_t_grid))),
            temp=np.vstack(tail_curves["temp"]) if len(tail_curves["temp"]) else np.empty((0, len(tail_t_grid))),
            group_temp=np.vstack(tail_curves["group_temp"]) if len(tail_curves["group_temp"]) else np.empty((0, len(tail_t_grid))),
            cip_global=np.vstack(tail_curves["cip_global"]) if len(tail_curves["cip_global"]) else np.empty((0, len(tail_t_grid))),
            cip_group=np.vstack(tail_curves["cip_group"]) if len(tail_curves["cip_group"]) else np.empty((0, len(tail_t_grid))),
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        warnings.warn(f"Could not save tail_curves_by_rep.npz: {exc}")

    # ------------------------------------------------------------
    # Save summary (mean/std)
    # ------------------------------------------------------------
    def mean_std(col: str) -> Tuple[float, float]:
        return float(df_out[col].mean()), float(df_out[col].std())

    summary_cols = [
        "mcov_q0_key", "mcov_temp_key", "mcov_group_temp_key", "mcov_cip_global_key", "mcov_cip_group_key",
        "wv_group_q0", "wv_group_temp", "wv_group_group_temp", "wv_group_cip_global", "wv_group_cip_group",
        "mse_q0", "mse_temp", "mse_group_temp", "mse_cip_global", "mse_cip_group",
        "mse_q0_orig", "mse_temp_orig", "mse_group_temp_orig", "mse_cip_global_orig", "mse_cip_group_orig",
        "kl_cip_global", "kl_cip_group",
        "ess_is_global", "ess_is_group",
        "cp_len_ridge", "cp_len_huber", "cp_len_q0mean", "cp_len_group_temp_mean", "cp_len_cip_group_mean", "mondrian_len_ridge",
        "time_total", "time_group_temp_tune", "time_feature_map", "feature_dim_input", "feature_dim_posterior",
        "eta_tuned_hit_max", "group_temp_proj_loss", "eta_group_temp_min", "eta_group_temp_mean", "eta_group_temp_max", "eta_group_temp_hit_max",
    ]
    if run_label is not None:
        run_name = str(run_label)
    else:
        run_name = cfg.name + ("_with_group_in_X" if include_group_in_X else "")
        if drop_first_categorical:
            run_name += "_drop1"
        if tau2_group_feature is not None:
            run_name += f"_groupPrior{float(tau2_group_feature):g}"
        if str(posterior_model).lower() not in ("linear", "none"):
            run_name += f"_{str(posterior_model).lower()}D{int(rf_dim)}"
    summ = {
        "dataset": cfg.name,
        "run_name": run_name,
        "include_group_in_X": float(include_group_in_X),
        "drop_first_categorical": float(drop_first_categorical),
        "tau2_group_feature": float(tau2_group_feature) if tau2_group_feature is not None else np.nan,
        "posterior_model": str(posterior_model),
        "posterior_model_code": float({"linear": 0, "none": 0, "rff": 1, "fourier": 1, "random_fourier": 1, "relu": 2, "relu_rf": 2, "random_relu": 2, "tanh": 3, "tanh_rf": 3}.get(str(posterior_model).lower(), -1)),
        "rf_dim": float(rf_dim),
        "rf_scale": float(rf_scale),
        "rf_include_linear": float(rf_include_linear),
        "R": float(R),
        "K": float(K),
        "n_groups": float(df_out["n_groups"].iloc[0]),
    }
    for c in summary_cols:
        if c in df_out.columns:
            m, s = mean_std(c)
            summ[c + "_mean"] = m
            summ[c + "_std"] = s
    pd.DataFrame([summ]).to_csv(os.path.join(outdir, "summary.csv"), index=False)
    if group_labels_for_plots is not None:
        pd.DataFrame({
            "group_code": np.arange(len(group_labels_for_plots)),
            "group_label": group_labels_for_plots,
        }).to_csv(os.path.join(outdir, "group_levels.csv"), index=False)

    # ------------------------------------------------------------
    # Figures (mean over replicates)
    # ------------------------------------------------------------
    if make_figures:
        # thresholds averaged over reps
        t_mean = np.mean(np.vstack(t_list), axis=0)

        def savefig(base: str) -> None:
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, base + ".pdf"))
            plt.savefig(os.path.join(figdir, base + ".png"))
            plt.close()

        # (1) Tail curve (global)
        plt.figure()
        for key, label in [
            ("q0", "Gibbs q0"),
            ("temp", "TempTune"),
            ("group_temp", "GroupTemp"),
            ("cip_global", "CIP-Global"),
            ("cip_group", "CIP-Group"),
        ]:
            mean_curve = np.mean(np.vstack(tail_curves[key]), axis=0)
            plt.plot(tail_t_grid, mean_curve, label=label)

        plt.scatter(t_mean, alpha_levels, marker="x", label="Targets (t_k, alpha_k)")
        plt.xlabel("Threshold t")
        plt.ylabel(r"Test tail miscoverage $\Pr(|Y-\hat Y|>t)$")
        plt.title(f"{cfg.name}: tail curve (global), mean over splits")
        plt.legend()
        savefig("tail_curve_global")

        # (2) Global miscoverage at key threshold (alpha approx 0.10)
        plt.figure()
        cols = [f"mcov_q0_{key_k}", f"mcov_temp_{key_k}", f"mcov_group_temp_{key_k}", f"mcov_cip_global_{key_k}", f"mcov_cip_group_{key_k}"]
        labels = ["q0", "TempTune", "GroupTemp", "CIP-Global", "CIP-Group"]
        means = [df_out[c].mean() for c in cols]
        stds = [df_out[c].std() for c in cols]
        x = np.arange(len(cols))
        plt.bar(x, means, yerr=stds)
        plt.axhline(alpha_levels[key_k], linestyle="--")
        plt.xticks(x, labels, rotation=20)
        plt.ylabel("Miscoverage at key threshold")
        plt.title(f"{cfg.name}: global miscoverage at t_key (alpha={alpha_levels[key_k]:.2f})")
        savefig("miscoverage_key_global")

        # (3) Groupwise miscoverage at key threshold
        plt.figure()
        methods = ["q0", "TempTune", "GroupTemp", "CIP-Global", "CIP-Group"]
        G = int(df_out["n_groups"].iloc[0])
        x = np.arange(len(methods))
        width = 0.80 / max(G, 1)

        for g in range(G):
            vals = [
                df_out[f"gm_q0_g{g}_key"].mean(),
                df_out[f"gm_temp_g{g}_key"].mean(),
                df_out[f"gm_group_temp_g{g}_key"].mean(),
                df_out[f"gm_cip_global_g{g}_key"].mean(),
                df_out[f"gm_cip_group_g{g}_key"].mean(),
            ]
            glab = group_labels_for_plots[g] if group_labels_for_plots and g < len(group_labels_for_plots) else f"Group {g}"
            plt.bar(x + (g - (G - 1) / 2) * width, vals, width, label=glab)

        plt.axhline(alpha_levels[key_k], linestyle="--")
        plt.xticks(x, methods, rotation=20)
        plt.ylabel("Miscoverage at key threshold")
        plt.title(f"{cfg.name}: groupwise miscoverage at t_key (mean)")
        plt.legend(ncol=2)
        savefig("miscoverage_key_by_group")

        # (4) Miscoverage at ALL constrained points (global)
        plt.figure()

        def mean_mcov(prefix: str) -> np.ndarray:
            return np.array([df_out[f"{prefix}_{k}"].mean() for k in range(K)], dtype=float)

        plt.plot(t_mean, mean_mcov("mcov_q0"), marker="o", label="q0")
        plt.plot(t_mean, mean_mcov("mcov_temp"), marker="o", label="TempTune")
        plt.plot(t_mean, mean_mcov("mcov_group_temp"), marker="o", label="GroupTemp")
        plt.plot(t_mean, mean_mcov("mcov_cip_global"), marker="o", label="CIP-Global")
        plt.plot(t_mean, mean_mcov("mcov_cip_group"), marker="o", label="CIP-Group")
        plt.plot(t_mean, alpha_levels, linestyle="--", marker="x", label="Target alpha")

        plt.xlabel("Threshold t_k (mean over splits)")
        plt.ylabel("Test miscoverage at t_k")
        plt.title(f"{cfg.name}: constraint satisfaction across tail points")
        plt.legend()
        savefig("miscoverage_all_constraints")

        # (5) Worst-group violation
        plt.figure()
        colsV = ["wv_group_q0", "wv_group_temp", "wv_group_group_temp", "wv_group_cip_global", "wv_group_cip_group"]
        labelsV = ["q0", "TempTune", "GroupTemp", "CIP-Global", "CIP-Group"]
        meansV = [df_out[c].mean() for c in colsV]
        stdsV = [df_out[c].std() for c in colsV]
        x = np.arange(len(colsV))
        plt.bar(x, meansV, yerr=stdsV)
        plt.xticks(x, labelsV, rotation=20)
        plt.ylabel(r"Worst-group violation $\max_{g,k}(\mathrm{MisCov}-\alpha)_+$")
        plt.title(f"{cfg.name}: worst-group multi-threshold violation")
        savefig("worst_group_violation")

        # (6) Calibration–accuracy summary: (worst-group violation, MSE)
        plt.figure()
        pts = [
            ("q0", "wv_group_q0", "mse_q0"),
            ("TempTune", "wv_group_temp", "mse_temp"),
            ("GroupTemp", "wv_group_group_temp", "mse_group_temp"),
            ("CIP-Global", "wv_group_cip_global", "mse_cip_global"),
            ("CIP-Group", "wv_group_cip_group", "mse_cip_group"),
        ]
        for lab, vx, my in pts:
            plt.scatter(df_out[vx].mean(), df_out[my].mean(), label=lab)
        plt.xlabel(r"Worst-group violation $\widehat V(q)$")
        plt.ylabel("Test MSE (standardized scale)")
        plt.title(f"{cfg.name}: calibration–accuracy summary")
        plt.legend()
        savefig("tradeoff_V_vs_MSE")

        # (7) Compact 2x2 panel for the main paper.  The caption in LaTeX can
        # describe the dataset; keeping plot titles short avoids duplication.
        fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.2))
        ax = axs[0, 0]
        for key, label in [
            ("q0", "q0"),
            ("temp", "TempTune"),
            ("group_temp", "GroupTemp"),
            ("cip_global", "CIP-Global"),
            ("cip_group", "CIP-Group"),
        ]:
            mean_curve = np.mean(np.vstack(tail_curves[key]), axis=0)
            ax.plot(tail_t_grid, mean_curve, label=label)
        ax.scatter(t_mean, alpha_levels, marker="x", label="Targets")
        ax.set_xlabel(r"Threshold $t$")
        ax.set_ylabel("Global tail rate")
        ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top")

        ax = axs[0, 1]
        G = int(df_out["n_groups"].iloc[0])
        x_methods = np.arange(len(methods))
        width = 0.80 / max(G, 1)
        for g in range(G):
            vals = [
                df_out[f"gm_q0_g{g}_key"].mean(),
                df_out[f"gm_temp_g{g}_key"].mean(),
                df_out[f"gm_group_temp_g{g}_key"].mean(),
                df_out[f"gm_cip_global_g{g}_key"].mean(),
                df_out[f"gm_cip_group_g{g}_key"].mean(),
            ]
            glab = group_labels_for_plots[g] if group_labels_for_plots and g < len(group_labels_for_plots) else f"Group {g}"
            ax.bar(x_methods + (g - (G - 1) / 2) * width, vals, width, label=glab)
        ax.axhline(alpha_levels[key_k], linestyle="--")
        ax.set_xticks(x_methods)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Group tail rate")
        ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, va="top")

        ax = axs[1, 0]
        ax.bar(np.arange(len(colsV)), meansV, yerr=stdsV)
        ax.set_xticks(np.arange(len(colsV)))
        ax.set_xticklabels(labelsV, rotation=20, ha="right")
        ax.set_ylabel(r"Worst-group violation $\widehat V(q)$")
        ax.text(0.02, 0.96, "(c)", transform=ax.transAxes, va="top")

        ax = axs[1, 1]
        for lab, vx, my in pts:
            ax.scatter(df_out[vx].mean(), df_out[my].mean(), label=lab)
        ax.set_xlabel(r"Worst-group violation $\widehat V(q)$")
        ax.set_ylabel("Predictive MSE")
        ax.text(0.02, 0.96, "(d)", transform=ax.transAxes, va="top")

        handles, labels_ = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=True)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(figdir, "paper_summary_2x2.pdf"))
        fig.savefig(os.path.join(figdir, "paper_summary_2x2.png"))
        plt.close(fig)

        # (8) Conformal interval lengths (original y-scale)
        if compute_conformal:
            plt.figure()
            cols_len = [
                "cp_len_ridge", "cp_len_huber", "cp_len_q0mean",
                "cp_len_group_temp_mean", "cp_len_cip_global_mean", "cp_len_cip_group_mean",
                "mondrian_len_ridge",
            ]
            labels_len = [
                "SC-Ridge", "SC-Huber", "SC-q0Mean",
                "SC-GroupTemp", "SC-CIPGlobMean", "SC-CIPGrpMean",
                "Mondrian-Ridge",
            ]
            means_len = [df_out[c].mean() for c in cols_len]
            stds_len = [df_out[c].std() for c in cols_len]
            x = np.arange(len(cols_len))
            plt.bar(x, means_len, yerr=stds_len)
            plt.xticks(x, labels_len, rotation=25, ha="right")
            plt.ylabel("Average interval length (original scale)")
            plt.title(f"{cfg.name}: conformal interval lengths (mean ± std)")
            savefig("conformal_lengths")

        # LaTeX helper for the main paper. Include this file from the dataset
        # section, or copy the body into the paper. It keeps the real-data
        # figure from floating back into the simulation section.
        snippet = rf"""% Auto-generated by real_data_cip_suite_v3_neurips.py
\begin{{figure}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/paper_summary_2x2.pdf}}
  \caption{{{cfg.name}: posterior-level calibration diagnostics. Group labels are the original labels used for stratified splitting and evaluation. GroupTemp denotes the group-specific temperature baseline.}}
  \label{{fig:{safe_dirname(cfg.name)}_realdata}}
\end{{figure}}
"""
        with open(os.path.join(figdir, "figure_snippet.tex"), "w", encoding="utf-8") as f:
            f.write(snippet)

    if verbose:
        print(f"\n[{cfg.name}] DONE. Saved metrics: {os.path.join(outdir, 'metrics.csv')}")
        if make_figures:
            print(f"[{cfg.name}] Saved figures to: {figdir}")

    return df_out


# ============================================================
# Suite runner (multiple datasets)
# ============================================================

def run_real_data_suite(
    outdir: str = "real_data_results",
    datasets: Optional[List[str]] = None,
    seed: int = 0,
    R: int = 20,
    K: int = 5,
) -> None:
    ensure_dir(outdir)

    configs: Dict[str, DatasetConfig] = {
        "diamonds": DatasetConfig(
            name="diamonds",
            loader=load_diamonds,
            target_col="price",
            group_col="cut",
            group_top_k=None,      # 5 groups already
            y_transform="log1p",   # price is heavy-tailed; log scale is standard
            frac_fit=0.40, frac_thr=0.15, frac_proj=0.15, frac_cal=0.15, frac_test=0.15,
        ),
        "bike": DatasetConfig(
            name="bike",
            loader=load_bike_sharing_hourly,
            target_col="cnt",
            group_col="season_name",
            group_top_k=None,      # four seasons
            y_transform="log1p",
            frac_fit=0.40, frac_thr=0.15, frac_proj=0.15, frac_cal=0.15, frac_test=0.15,
        ),
        "ames": DatasetConfig(
            name="ames",
            loader=load_ames_house_prices,
            target_col="SalePrice",
            group_col="Neighborhood",
            group_top_k=5,         # top 5 neighborhoods + Other
            y_transform="log1p",
            frac_fit=0.45, frac_thr=0.175, frac_proj=0.175, frac_cal=0.10, frac_test=0.10,
        ),
    }

    if datasets is None or len(datasets) == 0:
        datasets = ["diamonds", "bike"]

    all_summaries = []

    for ds_name in datasets:
        if ds_name not in configs:
            raise ValueError(f"Unknown dataset '{ds_name}'. Available: {list(configs.keys())}")

        cfg = configs[ds_name]

        # Main run: group-blind design matrix (group used only for thresholding / projection / evaluation)
        ds_out = os.path.join(outdir, safe_dirname(cfg.name))
        ensure_dir(ds_out)
        _ = run_real_data_dataset(
            cfg=cfg,
            outdir=ds_out,
            seed=seed,
            R=R,
            K=K,
            S_pool=40000 if ds_name == "diamonds" else 40000,
            n_particles=4000,
            make_figures=True,
            compute_conformal=True,
            include_group_in_X=False,
            run_label=cfg.name + "_blind",
            verbose=True,
        )
        summary_path = os.path.join(ds_out, "summary.csv")
        if os.path.exists(summary_path):
            all_summaries.append(pd.read_csv(summary_path))

        # Group-aware ablation: include the evaluation group in X with a stable
        # drop-one encoding and a stronger prior on group indicators.
        if ds_name in {"diamonds", "bike"}:
            ds_out_with_group = os.path.join(outdir, safe_dirname(cfg.name + "_with_group_in_X"))
            ensure_dir(ds_out_with_group)
            _ = run_real_data_dataset(
                cfg=cfg,
                outdir=ds_out_with_group,
                seed=seed,
                R=R,
                K=K,
                S_pool=40000,
                n_particles=4000,
                make_figures=True,
                compute_conformal=True,
                include_group_in_X=True,
                drop_first_categorical=True,
                tau2_group_feature=1.0,
                run_label=cfg.name + "_with_group_in_X_drop1_groupPrior1",
                verbose=True,
            )
            summary_path = os.path.join(ds_out_with_group, "summary.csv")
            if os.path.exists(summary_path):
                all_summaries.append(pd.read_csv(summary_path))

    if len(all_summaries) > 0:
        all_df = pd.concat(all_summaries, ignore_index=True)
        all_df.to_csv(os.path.join(outdir, "summary_all_datasets.csv"), index=False)
        print(f"\n[SUITE DONE] Combined summary saved to: {os.path.join(outdir, 'summary_all_datasets.csv')}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-data CIP evaluation suite")
    p.add_argument("--outdir", type=str, default="real_data_results")
    p.add_argument("--datasets", type=str, nargs="*", default=["diamonds", "bike"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--R", type=int, default=20)
    p.add_argument("--K", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_real_data_suite(outdir=args.outdir, datasets=args.datasets, seed=args.seed, R=args.R, K=args.K)