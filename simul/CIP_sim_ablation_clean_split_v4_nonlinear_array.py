"""
Simulation suite for Conformal Information Projection (CIP)
==========================================================

This file contains:

(1) A single-setting simulation runner `run_simulation(...)` that reproduces the strong baselines,
    all metrics, and saved figures from prior code, PLUS extra diagnostics:
      - runtime breakdown (pool sampling, g-compute, dual solve, SMC, TempTune, eval)
      - degeneracy metrics (IS ESS, SMC ESS-min, unique fraction, MH accept)
      - approximate KL(q* || q0) (pool-based estimate)

(2) A paper-style suite runner `run_paper_suite(...)` that automates the next-step experiments:
      (B) Ablation grid over (K, G) to show TempTune fails and CIP-Group advantage grows.
      (C) Severity ablations: group_scale_max, df (nu), hetero_strength.
      (D) Calibration–accuracy frontier: sweep tighten_factor and plot (V, MSE) + (V, KL).
      (E) Driving worst-group violations toward zero: sweep m_proj per group and tighten_factor
          + plot MSE vs m_proj and (V, MSE) as m_proj grows.
      (F) Algorithmic scaling diagnostics: runtime/ESS vs S_pool, K, G.

Dependencies: numpy, scipy, pandas, matplotlib
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif"})

from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.optimize import minimize


# ============================================================
# Utilities
# ============================================================

def toeplitz_cov(d: int, rho: float = 0.3) -> np.ndarray:
    idx = np.arange(d)
    return rho ** np.abs(np.subtract.outer(idx, idx))


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile:
      k = ceil((n+1)(1-alpha)), return k-th order statistic (1-indexed).
    """
    residuals = np.asarray(residuals)
    n = residuals.shape[0]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(residuals, k - 1)[k - 1])


def tighten_levels(alpha_levels: np.ndarray, m: int, L: int, delta: float, factor: float) -> tuple[np.ndarray, float]:
    """
    Mild tightening based on Hoeffding + union bound:

      eps = factor * sqrt( log( (2L)/delta ) / (2m) )

    We enforce alpha_tight = max(alpha - eps, 0).

    Practical interpretation:
      - factor=0 means no tightening.
      - larger factor makes constraints stricter on the projection set,
        improving out-of-sample satisfaction but increasing KL tilt / conservativeness.
    """
    alpha_levels = np.asarray(alpha_levels, dtype=float)
    m = max(int(m), 1)
    L = max(int(L), 1)
    delta = max(float(delta), 1e-12)
    factor = float(factor)

    eps = factor * np.sqrt(np.log((2.0 * L) / delta) / (2.0 * m))
    return np.clip(alpha_levels - eps, 0.0, 1.0), float(eps)


def alpha_levels_for_K(K: int) -> np.ndarray:
    """
    Convenience: sensible tail-probability grids.
    Always ordered from larger alpha (smaller threshold) to smaller alpha (larger threshold).
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
    # generic geometric grid from 0.30 down to 0.02
    al = np.geomspace(0.30, 0.02, K)
    al = np.sort(al)[::-1]
    return al.astype(float)


def safe_dirname(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", ".", "="):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


# ============================================================
# Optional nonlinear signal and random-feature posterior
# ============================================================

def nonlinear_signal_function(X: np.ndarray) -> np.ndarray:
    """Smooth nonlinear component used only when nonlinear_signal > 0."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] < 3:
        return np.sin(X[:, 0])
    z = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + 0.25 * X[:, 0] * X[:, 2] / np.sqrt(1.0 + X[:, 2] ** 2)
    z = z - z.mean()
    sd = z.std()
    if sd > 1e-8:
        z = z / sd
    return z


def add_nonlinear_signal(y: np.ndarray, X: np.ndarray, strength: float) -> np.ndarray:
    if float(strength) == 0.0:
        return y
    return np.asarray(y, dtype=float) + float(strength) * nonlinear_signal_function(X)


def _standardize_from_fit(X_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_fit = np.asarray(X_fit, dtype=float)
    mu = X_fit.mean(axis=0)
    sd = X_fit.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X_fit - mu) / sd, mu, sd


def make_random_feature_state(
    rng: np.random.Generator,
    X_fit: np.ndarray,
    posterior_model: str = "linear",
    rf_dim: int = 128,
    rf_scale: float = 1.0,
    rf_include_linear: bool = True,
) -> dict:
    model = str(posterior_model).lower()
    if model in ("linear", "none"):
        return {"model": "linear"}
    _, mu, sd = _standardize_from_fit(X_fit)
    d_in = int(X_fit.shape[1])
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


def apply_random_feature_state(X: np.ndarray, state: dict) -> np.ndarray:
    model = str(state.get("model", "linear")).lower()
    X = np.asarray(X, dtype=float)
    if model == "linear":
        return X
    mu = np.asarray(state["mu"], dtype=float)
    sd = np.asarray(state["sd"], dtype=float)
    Z = (X - mu) / sd
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
    parts = []
    if bool(state.get("rf_include_linear", True)):
        parts.append(Z)
    parts.append(R)
    return np.concatenate(parts, axis=1)


def apply_random_features_to_sim_splits(
    rng: np.random.Generator,
    X_fit: np.ndarray,
    X_thr: np.ndarray,
    X_proj: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    posterior_model: str = "linear",
    rf_dim: int = 128,
    rf_scale: float = 1.0,
    rf_include_linear: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    state = make_random_feature_state(
        rng, X_fit,
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
# Data generation: multi-group covariate scale shift + heavy tails
# ============================================================

def generate_group_data(
    rng: np.random.Generator,
    n: int,
    d: int,
    beta_true: np.ndarray,
    Sigma: np.ndarray,
    n_groups: int = 4,
    group_probs: np.ndarray | None = None,
    group_scales: np.ndarray | None = None,
    sigma: float = 1.0,
    df: float = 3.0,
    hetero_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DGP:
      G ~ Categorical(group_probs) on {0,...,G-1}
      Z ~ N(0,Sigma)
      X = s_G * Z
      Y = X^T beta_true + eps
      eps ~ Student-t(df) * sigma * sqrt(1 + hetero_strength * X_1^2)

    The group-dependent feature scale induces subgroup tail differences via parameter uncertainty:
      larger |X| => larger |X^T(β-β*)|, hence heavier predictive tails in high-scale groups.
    """
    if group_probs is None:
        group_probs = np.ones(n_groups) / n_groups
    group_probs = np.asarray(group_probs, dtype=float)
    group_probs = group_probs / group_probs.sum()
    group = rng.choice(np.arange(n_groups), size=n, p=group_probs)

    if group_scales is None:
        group_scales = np.linspace(1.0, 3.0, n_groups)
    group_scales = np.asarray(group_scales, dtype=float)

    Z = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    X = Z * group_scales[group][:, None]

    noise_scale = sigma
    if hetero_strength != 0.0:
        noise_scale = noise_scale * np.sqrt(1.0 + hetero_strength * (X[:, 1] ** 2))
    eps = noise_scale * rng.standard_t(df, size=n)

    y = X @ beta_true + eps
    return X, y, group


# ============================================================
# Gibbs posterior: squared loss + Gaussian prior => Gaussian posterior
# ============================================================

def fit_gibbs_gaussian_posterior(X: np.ndarray, y: np.ndarray, eta: float, tau2: float):
    """
    q(β) ∝ exp(-η ||y - Xβ||^2) * N(0, τ^2 I)

    Precision: P = 2η X^T X + (1/τ^2) I
    Mean:      μ = P^{-1} (2η X^T y)
    """
    XtX = X.T @ X
    P = 2.0 * float(eta) * XtX + (1.0 / float(tau2)) * np.eye(X.shape[1])
    h = 2.0 * float(eta) * (X.T @ y)
    c, lower = cho_factor(P, lower=True, check_finite=False)
    mu = cho_solve((c, lower), h, check_finite=False)
    return mu, (c, lower)


def sample_from_precision_cholesky(rng: np.random.Generator, mu: np.ndarray, cho_prec, S: int) -> np.ndarray:
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
    block_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
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
    maxiter: int = 2500,
    maxfun: int = 200000,
    restarts: int = 2,
    verbose: bool = False,
):
    """
    Solve:
      min_{λ>=0}  log( mean_s exp( - g_s^T λ ) ) + α^T λ

    Returns:
      λ,
      weights w_s ∝ exp(-g_s^T λ) (normalized),
      ESS,
      logZ where Z = mean_s exp(-g_s^T λ),
      Eg = sum_s w_s g_s
    """
    g = np.asarray(g, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    S, K = g.shape
    assert alpha.shape == (K,)

    def obj_and_grad(lmbda: np.ndarray):
        a = -g @ lmbda
        logZ = logsumexp(a) - np.log(S)
        f = logZ + alpha.dot(lmbda)
        logw = a - logsumexp(a)
        w = np.exp(logw)
        Eg = w @ g
        grad = alpha - Eg
        return f, grad

    bounds = [(0.0, None)] * K

    def fun_with_grad(x):
        return obj_and_grad(x)

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
            options={"maxiter": int(maxiter), "maxfun": maxfun_attempt, "ftol": 1e-12},
        )
        x0 = last_res.x
        if last_res.success:
            break

    res = last_res
    if not res.success:
        warnings.warn(f"Dual solver warning (after restarts): {res.message}")

    lmbda = res.x
    a = -g @ lmbda
    logZ = logsumexp(a) - np.log(S)
    logw = a - logsumexp(a)
    w = np.exp(logw)
    ess = 1.0 / np.sum(w ** 2)
    Eg = w @ g

    if verbose:
        max_viol = float(np.max(Eg - alpha))
        grad_inf = float(np.max(np.abs(alpha - Eg)))
        print(f"[dual] success={res.success} | max_viol={max_viol:.3e} | grad_inf={grad_inf:.3e} | ESS={ess:.1f}")

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
    n_particles: int = 4000,
    n_steps: int = 40,
    ess_threshold: float = 0.6,
    rejuvenate_steps: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Discrete-pool SMC approximation to:
      p(i) ∝ exp( - λ^T g_i )

    Tempering s from 0 to 1:
      p_s(i) ∝ exp( - s λ^T g_i )

    Returns:
      betas: (n_particles, d), weights: uniform, info: diagnostics
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
            weights *= np.exp(-ds * phi_pool[idx])
            sw = weights.sum()
            if (sw == 0.0) or (not np.isfinite(sw)):
                logw = -ds * phi_pool[idx]
                logw -= logw.max()
                weights = np.exp(logw)
                sw = weights.sum()
            weights /= sw

        ess = 1.0 / np.sum(weights ** 2)
        ess_min = min(ess_min, float(ess))

        if ess < float(ess_threshold) * n_particles:
            rs = systematic_resample(rng, weights)
            idx = idx[rs]
            weights.fill(1.0 / n_particles)
            resamples += 1

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

    # final resample to return unweighted sample
    weights /= weights.sum()
    rs = systematic_resample(rng, weights)
    idx = idx[rs]
    weights = np.ones(n_particles, dtype=float) / n_particles

    unique_frac = float(np.unique(idx).size / n_particles)
    accept_rate = float(accept_total / max(1, accept_trials))

    info = {
        "ess_min": float(ess_min),
        "resamples": int(resamples),
        "unique_frac": float(unique_frac),
        "mh_accept": float(accept_rate),
    }
    return betas_pool[idx], weights, info


# ============================================================
# Evaluation
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


# ============================================================
# Baselines: ridge / Huber + conformal
# ============================================================

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + float(lam) * np.eye(d)
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
    Simple IRLS for Huber regression with ridge regularization.
    """
    n, d = X.shape
    beta = fit_ridge(X, y, lam)

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

        A = Xw.T @ Xw + float(lam) * np.eye(d)
        b = Xw.T @ yw
        beta_new = np.linalg.solve(A, b)

        if np.linalg.norm(beta_new - beta) <= float(tol) * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new

    return beta


def split_conformal_interval_halfwidth(y_cal: np.ndarray, yhat_cal: np.ndarray, alpha: float) -> float:
    resid = np.abs(y_cal - yhat_cal)
    return conformal_quantile(resid, alpha)


def mondrian_halfwidth_by_group(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    g_cal: np.ndarray,
    alpha: float,
    n_groups: int,
) -> dict[int, float]:
    out = {}
    for g in range(int(n_groups)):
        idx = np.where(g_cal == g)[0]
        if idx.size == 0:
            out[g] = 0.0
        else:
            resid = np.abs(y_cal[idx] - yhat_cal[idx])
            out[g] = conformal_quantile(resid, alpha)
    return out


# ============================================================
# TempTune baseline: tune eta to minimize WORST-group violations over all constraints
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
    tau2: float,
    n_groups: int,
    S_tune: int = 2000,
) -> tuple[float, np.ndarray, tuple]:
    """
    Grid-search eta for Gibbs posterior to minimize worst-group violation:
      loss(eta) = max_{g,k} max( E[g_{g,k}(beta)] - alpha_{g,k}, 0 )
    on the projection set.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    alpha_group_tight = np.asarray(alpha_group_tight, dtype=float)

    best_eta = None
    best_loss = np.inf
    best_mu, best_cho = None, None

    for eta in np.asarray(eta_grid, dtype=float):
        mu, cho = fit_gibbs_gaussian_posterior(X_fit, y_fit, float(eta), float(tau2))
        betas = sample_from_precision_cholesky(rng, mu, cho, int(S_tune))

        _, g_grp = compute_g_global_and_group_blockwise(
            betas, X_proj, y_proj, g_proj, thresholds, n_groups=int(n_groups), block_size=2000
        )
        Eg_grp = g_grp.reshape(int(S_tune), int(n_groups), -1).mean(axis=0)  # (G,K)
        viol_grp = np.maximum(Eg_grp - alpha_group_tight, 0.0)
        loss = float(np.max(viol_grp))

        if loss < best_loss:
            best_loss = loss
            best_eta = float(eta)
            best_mu, best_cho = mu, cho

    return best_eta, best_mu, best_cho


def fit_gibbs_gaussian_posterior_group_eta(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    eta_by_group: np.ndarray,
    tau2: float,
):
    """
    Gibbs posterior with one learning rate per training group.

    This defines a single posterior and is used as the GroupTemp baseline.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    group = np.asarray(group, dtype=int)
    eta_by_group = np.asarray(eta_by_group, dtype=float)
    w = eta_by_group[group]
    d = X.shape[1]
    Xw = X * np.sqrt(2.0 * w)[:, None]
    P = Xw.T @ Xw + (1.0 / float(tau2)) * np.eye(d)
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
    tau2: float,
    n_groups: int,
    eta_init: float,
    S_tune: int = 1000,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, tuple, float]:
    """
    Coordinate grid search for a group-specific temperature baseline.

    The objective is the same projection-split worst-group multi-threshold
    violation used by TempTune, but the training loss has one eta per group.
    """
    G = int(n_groups)
    eta_grid = np.asarray(eta_grid, dtype=float)
    eta_vec = np.full(G, float(eta_init), dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    alpha_group_tight = np.asarray(alpha_group_tight, dtype=float)

    def objective(evec: np.ndarray):
        mu, cho = fit_gibbs_gaussian_posterior_group_eta(X_fit, y_fit, g_fit, evec, float(tau2))
        betas = sample_from_precision_cholesky(rng, mu, cho, int(S_tune))
        _, g_grp = compute_g_global_and_group_blockwise(
            betas, X_proj, y_proj, g_proj, thresholds, n_groups=G, block_size=2000
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

    best_mu, best_cho = fit_gibbs_gaussian_posterior_group_eta(X_fit, y_fit, g_fit, eta_vec, float(tau2))
    return eta_vec, best_mu, best_cho, float(best_loss)


# ============================================================
# Main simulation for a single setting (with defaults)
# ============================================================

def run_simulation(
    outdir: str = "sim_results",
    seed: int = 0,
    R: int = 20,
    # DGP
    n_groups: int = 4,
    d: int = 25,
    rho: float = 0.3,
    group_scale_max: float = 3.0,
    sigma: float = 1.0,
    df: float = 3.0,
    hetero_strength: float = 0.15,
    nonlinear_signal: float = 0.0,
    posterior_model: str = "linear",
    rf_dim: int = 128,
    rf_scale: float = 1.0,
    rf_include_linear: bool = True,
    # splits
    n_fit: int = 400,
    m_thr: int | None = None,
    m_proj: int = 800,
    r_cal: int = 400,
    n_test: int = 4000,
    # posterior
    tau2: float = 25.0,
    eta0: float = 0.2,
    eta_ref: float = 2.0,
    # constraints
    alpha_levels = (0.30, 0.20, 0.10, 0.05, 0.02),
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
    eta_grid = None,
    S_tune: int = 2000,
    S_group_tune: int | None = None,
    group_eta_passes: int = 2,
    # evaluation
    t_grid = None,
    eval_block: int = 500,
    # switches
    make_figures: bool = True,
    compute_conformal: bool = True,
    verbose: bool = True,
    save_tail_curves: bool = False,
) -> pd.DataFrame:
    """
    Runs R replicates for the specified configuration.

    Saves:
      - outdir/metrics.csv
      - outdir/figures/*.png   (if make_figures=True)

    Returns:
      - pandas DataFrame with per-replicate metrics.
    """
    os.makedirs(outdir, exist_ok=True)
    figdir = os.path.join(outdir, "figures")
    if make_figures:
        os.makedirs(figdir, exist_ok=True)

    if eta_grid is None:
        # fair wide grid (avoid reviewer complaint about truncation)
        eta_grid = np.logspace(-2, 2, 21)
    if S_group_tune is None:
        S_group_tune = max(800, int(S_tune) // 2)

    if m_thr is None:
        # Backward-compatible default: use a separate threshold split
        # with the same size as the projection split.
        m_thr = int(m_proj)

    alpha_levels = np.asarray(alpha_levels, dtype=float)
    K = alpha_levels.size

    # key alpha for "t1-style" plots (nearest to 0.10)
    key_alpha = 0.10
    key_k = int(np.argmin(np.abs(alpha_levels - key_alpha)))

    if t_grid is None:
        t_grid = np.linspace(0.0, 12.0, 121)

    rng_master = np.random.default_rng(int(seed))

    rows = []
    tail_curves = {"q0": [], "temp": [], "group_temp": [], "cip_global": [], "cip_group": []}
    t_list = []

    for rep in range(int(R)):
        t0_total = time.perf_counter()
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))

        Sigma = toeplitz_cov(int(d), float(rho))
        beta_true = np.zeros(int(d))
        beta_true[:min(8, int(d))] = rng.normal(size=min(8, int(d)))

        group_scales = np.linspace(1.0, float(group_scale_max), int(n_groups))

        X_fit, y_fit, g_fit = generate_group_data(
            rng, int(n_fit), int(d), beta_true, Sigma,
            n_groups=int(n_groups), group_scales=group_scales,
            sigma=float(sigma), df=float(df), hetero_strength=float(hetero_strength),
        )
        X_thr, y_thr, g_thr = generate_group_data(
            rng, int(m_thr), int(d), beta_true, Sigma,
            n_groups=int(n_groups), group_scales=group_scales,
            sigma=float(sigma), df=float(df), hetero_strength=float(hetero_strength),
        )
        X_proj, y_proj, g_proj = generate_group_data(
            rng, int(m_proj), int(d), beta_true, Sigma,
            n_groups=int(n_groups), group_scales=group_scales,
            sigma=float(sigma), df=float(df), hetero_strength=float(hetero_strength),
        )
        X_cal, y_cal, g_cal = generate_group_data(
            rng, int(r_cal), int(d), beta_true, Sigma,
            n_groups=int(n_groups), group_scales=group_scales,
            sigma=float(sigma), df=float(df), hetero_strength=float(hetero_strength),
        )
        X_test, y_test, g_test = generate_group_data(
            rng, int(n_test), int(d), beta_true, Sigma,
            n_groups=int(n_groups), group_scales=group_scales,
            sigma=float(sigma), df=float(df), hetero_strength=float(hetero_strength),
        )

        # Optional nonlinear response and random-feature posterior.
        y_fit = add_nonlinear_signal(y_fit, X_fit, float(nonlinear_signal))
        y_thr = add_nonlinear_signal(y_thr, X_thr, float(nonlinear_signal))
        y_proj = add_nonlinear_signal(y_proj, X_proj, float(nonlinear_signal))
        y_cal = add_nonlinear_signal(y_cal, X_cal, float(nonlinear_signal))
        y_test = add_nonlinear_signal(y_test, X_test, float(nonlinear_signal))
        feature_dim_input = int(X_fit.shape[1])
        t_feature = time.perf_counter()
        X_fit, X_thr, X_proj, X_cal, X_test, rf_state = apply_random_features_to_sim_splits(
            rng, X_fit, X_thr, X_proj, X_cal, X_test,
            posterior_model=str(posterior_model),
            rf_dim=int(rf_dim),
            rf_scale=float(rf_scale),
            rf_include_linear=bool(rf_include_linear),
        )
        time_feature_map = time.perf_counter() - t_feature
        feature_dim_posterior = int(X_fit.shape[1])

        # q0 and reference predictor
        mu0, cho0 = fit_gibbs_gaussian_posterior(X_fit, y_fit, float(eta0), float(tau2))
        mu_ref, _ = fit_gibbs_gaussian_posterior(X_fit, y_fit, float(eta_ref), float(tau2))

        # thresholds at tail points (from reference predictor residuals on the threshold split)
        resid_ref = np.abs(y_thr - X_thr @ mu_ref)
        t_vec = np.array([conformal_quantile(resid_ref, a) for a in alpha_levels], dtype=float)
        t_list.append(t_vec)

        # sample pool from q0
        t_pool = time.perf_counter()
        betas_pool = sample_from_precision_cholesky(rng, mu0, cho0, int(S_pool))
        time_pool_sample = time.perf_counter() - t_pool

        # compute constraint features on pool
        t_g = time.perf_counter()
        g_global_pool, g_group_pool = compute_g_global_and_group_blockwise(
            betas_pool, X_proj, y_proj, g_proj, t_vec, n_groups=int(n_groups), block_size=2000
        )
        time_g_compute = time.perf_counter() - t_g

        # mild constraint tightening
        alpha_global_tight, eps_global = tighten_levels(
            alpha_levels, m=int(m_proj), L=int(K), delta=float(tighten_delta), factor=float(tighten_factor)
        )

        sizes = np.array([(g_proj == g).sum() for g in range(int(n_groups))], dtype=int)
        alpha_group_tight = np.zeros((int(n_groups), int(K)), dtype=float)
        eps_group = np.zeros(int(n_groups), dtype=float)
        L_group = int(n_groups) * int(K)
        for g in range(int(n_groups)):
            alpha_group_tight[g], eps_group[g] = tighten_levels(
                alpha_levels, m=int(sizes[g]), L=L_group, delta=float(tighten_delta), factor=float(tighten_factor)
            )

        # Solve duals on pool
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

        # approximate KLs on the pool
        kl_global = float(-lam_global.dot(Eg_global) - logZ_global)
        kl_group = float(-lam_group.dot(Eg_group) - logZ_group)

        # Stabilized CIP sampling (SMC on pool)
        t_smc = time.perf_counter()
        betas_cip_global, w_cip_global, info_cip_global = smc_sample_from_pool(
            rng, betas_pool, g_global_pool, lam_global,
            n_particles=int(n_particles), n_steps=int(smc_steps_global),
            ess_threshold=float(ess_threshold), rejuvenate_steps=int(rejuvenate_global),
        )
        time_smc_global = time.perf_counter() - t_smc

        t_smc = time.perf_counter()
        betas_cip_group, w_cip_group, info_cip_group = smc_sample_from_pool(
            rng, betas_pool, g_group_pool, lam_group,
            n_particles=int(n_particles), n_steps=int(smc_steps_group),
            ess_threshold=float(ess_threshold), rejuvenate_steps=int(rejuvenate_group),
        )
        time_smc_group = time.perf_counter() - t_smc

        # q0 sample for evaluation: subset from pool
        if int(S_pool) >= int(n_particles):
            idx0 = rng.choice(int(S_pool), size=int(n_particles), replace=False)
        else:
            idx0 = rng.integers(0, int(S_pool), size=int(n_particles))
        betas_q0 = betas_pool[idx0]
        w_q0 = np.ones(betas_q0.shape[0], dtype=float) / betas_q0.shape[0]

        # TempTune baseline (worst-group over all constraints)
        t_tmp = time.perf_counter()
        eta_tuned, mu_tuned, cho_tuned = tune_eta_grid(
            rng,
            X_fit, y_fit,
            X_proj, y_proj, g_proj,
            thresholds=t_vec,
            alpha_group_tight=alpha_group_tight,
            eta_grid=np.asarray(eta_grid, dtype=float),
            tau2=float(tau2),
            n_groups=int(n_groups),
            S_tune=int(S_tune),
        )
        time_temp_tune = time.perf_counter() - t_tmp

        betas_temp = sample_from_precision_cholesky(rng, mu_tuned, cho_tuned, int(n_particles))
        w_temp = np.ones(betas_temp.shape[0], dtype=float) / betas_temp.shape[0]

        # GroupTemp baseline: one learning rate per training group, but one posterior.
        t_gtmp = time.perf_counter()
        eta_group_tuned, mu_group_temp, cho_group_temp, group_temp_proj_loss = tune_group_eta_coordinate_grid(
            rng,
            X_fit, y_fit, g_fit,
            X_proj, y_proj, g_proj,
            thresholds=t_vec,
            alpha_group_tight=alpha_group_tight,
            eta_grid=np.asarray(eta_grid, dtype=float),
            tau2=float(tau2),
            n_groups=int(n_groups),
            eta_init=float(eta_tuned),
            S_tune=int(S_group_tune),
            passes=int(group_eta_passes),
        )
        time_group_temp_tune = time.perf_counter() - t_gtmp
        betas_group_temp = sample_from_precision_cholesky(rng, mu_group_temp, cho_group_temp, int(n_particles))
        w_group_temp = np.ones(betas_group_temp.shape[0], dtype=float) / betas_group_temp.shape[0]

        # ---- Evaluate miscoverage at ALL constrained thresholds on TEST ----
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

        def worst_violation_group(gm_mat: np.ndarray) -> float:
            return float(np.max(np.maximum(gm_mat - alpha_levels[None, :], 0.0)))

        wv_q0_group = worst_violation_group(gm_q0)
        wv_temp_group = worst_violation_group(gm_temp)
        wv_group_temp_group = worst_violation_group(gm_group_temp)
        wv_cipg_group = worst_violation_group(gm_cipg)
        wv_cipgrp_group = worst_violation_group(gm_cipgrp)

        # Tail curves are needed for aggregate paper figures in job-array mode.
        if make_figures or save_tail_curves:
            tail_curve_q0 = tail_curve_blockwise(betas_q0, w_q0, X_test, y_test, t_grid, block_size=int(eval_block))
            tail_curve_temp = tail_curve_blockwise(betas_temp, w_temp, X_test, y_test, t_grid, block_size=int(eval_block))
            tail_curve_group_temp = tail_curve_blockwise(betas_group_temp, w_group_temp, X_test, y_test, t_grid, block_size=int(eval_block))
            tail_curve_cip_global = tail_curve_blockwise(betas_cip_global, w_cip_global, X_test, y_test, t_grid, block_size=int(eval_block))
            tail_curve_cip_group = tail_curve_blockwise(betas_cip_group, w_cip_group, X_test, y_test, t_grid, block_size=int(eval_block))
            if make_figures:
                tail_curves["q0"].append(tail_curve_q0)
                tail_curves["temp"].append(tail_curve_temp)
                tail_curves["group_temp"].append(tail_curve_group_temp)
                tail_curves["cip_global"].append(tail_curve_cip_global)
                tail_curves["cip_group"].append(tail_curve_cip_group)
        else:
            tail_curve_q0 = tail_curve_temp = tail_curve_group_temp = None
            tail_curve_cip_global = tail_curve_cip_group = None

        time_eval = time.perf_counter() - t_eval

        # posterior mean predictors + MSE
        beta_mean_q0 = betas_q0.mean(axis=0)
        beta_mean_temp = betas_temp.mean(axis=0)
        beta_mean_group_temp = betas_group_temp.mean(axis=0)
        beta_mean_cipg = betas_cip_global.mean(axis=0)
        beta_mean_cipgrp = betas_cip_group.mean(axis=0)

        mse_q0 = float(np.mean((y_test - X_test @ beta_mean_q0) ** 2))
        mse_temp = float(np.mean((y_test - X_test @ beta_mean_temp) ** 2))
        mse_group_temp = float(np.mean((y_test - X_test @ beta_mean_group_temp) ** 2))
        mse_cipg = float(np.mean((y_test - X_test @ beta_mean_cipg) ** 2))
        mse_cipgrp = float(np.mean((y_test - X_test @ beta_mean_cipgrp) ** 2))

        # conformal baselines (optional)
        alpha_cp = 0.10
        if compute_conformal:
            beta_ridge = fit_ridge(X_fit, y_fit, lam=1.0)
            beta_huber = fit_huber_irls(X_fit, y_fit, lam=1.0)

            t_ridge = split_conformal_interval_halfwidth(y_cal, X_cal @ beta_ridge, alpha_cp)
            t_huber = split_conformal_interval_halfwidth(y_cal, X_cal @ beta_huber, alpha_cp)
            t_q0mean = split_conformal_interval_halfwidth(y_cal, X_cal @ beta_mean_q0, alpha_cp)
            t_cipgmean = split_conformal_interval_halfwidth(y_cal, X_cal @ beta_mean_cipg, alpha_cp)
            t_cipgrpmean = split_conformal_interval_halfwidth(y_cal, X_cal @ beta_mean_cipgrp, alpha_cp)

            def cover_at(beta_hat: np.ndarray, t: float) -> float:
                return float(np.mean(np.abs(y_test - X_test @ beta_hat) <= t))

            cov_ridge = cover_at(beta_ridge, t_ridge)
            cov_huber = cover_at(beta_huber, t_huber)
            cov_q0mean = cover_at(beta_mean_q0, t_q0mean)
            cov_cipgmean = cover_at(beta_mean_cipg, t_cipgmean)
            cov_cipgrpmean = cover_at(beta_mean_cipgrp, t_cipgrpmean)

            t_mond = mondrian_halfwidth_by_group(y_cal, X_cal @ beta_ridge, g_cal, alpha_cp, n_groups=int(n_groups))
            cov_mond = float(np.mean([
                (abs(y_test[i] - X_test[i] @ beta_ridge) <= t_mond[int(g_test[i])])
                for i in range(len(y_test))
            ]))
            avg_len_mond = float(np.mean([2.0 * t_mond[int(g)] for g in g_test]))

            sc_len_ridge = float(2.0 * t_ridge)
            sc_cov_ridge = cov_ridge
            sc_len_huber = float(2.0 * t_huber)
            sc_cov_huber = cov_huber
            sc_len_q0mean = float(2.0 * t_q0mean)
            sc_cov_q0mean = cov_q0mean
            sc_len_cip_global_mean = float(2.0 * t_cipgmean)
            sc_cov_cip_global_mean = cov_cipgmean
            sc_len_cip_group_mean = float(2.0 * t_cipgrpmean)
            sc_cov_cip_group_mean = cov_cipgrpmean
            mondrian_len_ridge = avg_len_mond
            mondrian_cov_ridge = cov_mond
        else:
            sc_len_ridge = sc_cov_ridge = np.nan
            sc_len_huber = sc_cov_huber = np.nan
            sc_len_q0mean = sc_cov_q0mean = np.nan
            sc_len_cip_global_mean = sc_cov_cip_global_mean = np.nan
            sc_len_cip_group_mean = sc_cov_cip_group_mean = np.nan
            mondrian_len_ridge = mondrian_cov_ridge = np.nan

        time_total = time.perf_counter() - t0_total

        # record row
        row = {
            "rep": rep,
            "seed": int(seed),
            "K": int(K),
            "n_groups": int(n_groups),
            "group_scale_max": float(group_scale_max),
            "df": float(df),
            "hetero_strength": float(hetero_strength),
            "nonlinear_signal": float(nonlinear_signal),
            "posterior_model": str(posterior_model),
            "posterior_model_code": float({"linear": 0, "none": 0, "rff": 1, "fourier": 1, "random_fourier": 1, "relu": 2, "relu_rf": 2, "random_relu": 2, "tanh": 3, "tanh_rf": 3}.get(str(posterior_model).lower(), -1)),
            "rf_dim": float(rf_dim),
            "rf_scale": float(rf_scale),
            "rf_include_linear": float(rf_include_linear),
            "feature_dim_input": float(feature_dim_input),
            "feature_dim_posterior": float(feature_dim_posterior),
            "m_thr": int(m_thr),
            "m_proj": int(m_proj),
            "tighten_factor": float(tighten_factor),
            "S_pool": int(S_pool),
            "n_particles": int(n_particles),
            "eta_tuned": float(eta_tuned),
            "group_temp_proj_loss": float(group_temp_proj_loss),
            "eta_group_temp_min": float(np.min(eta_group_tuned)),
            "eta_group_temp_mean": float(np.mean(eta_group_tuned)),
            "eta_group_temp_max": float(np.max(eta_group_tuned)),
            "eta_group_temp_hit_max": float(np.any(eta_group_tuned >= np.max(eta_grid) - 1e-12)),
            "eta_grid_max": float(np.max(eta_grid)),
            "eta_tuned_hit_max": float(eta_tuned >= np.max(eta_grid) - 1e-12),
            "key_k": int(key_k),
            "key_alpha": float(alpha_levels[key_k]),
            "key_t": float(t_vec[key_k]),
            "eps_global": float(eps_global),
            "eps_group_min": float(eps_group.min()),
            "eps_group_max": float(eps_group.max()),
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
            "wv_group_q0": wv_q0_group,
            "wv_group_temp": wv_temp_group,
            "wv_group_group_temp": wv_group_temp_group,
            "wv_group_cip_global": wv_cipg_group,
            "wv_group_cip_group": wv_cipgrp_group,
            "mse_q0": mse_q0,
            "mse_temp": mse_temp,
            "mse_group_temp": mse_group_temp,
            "mse_cip_global": mse_cipg,
            "mse_cip_group": mse_cipgrp,
            "sc_len_ridge": sc_len_ridge,
            "sc_cov_ridge": sc_cov_ridge,
            "sc_len_huber": sc_len_huber,
            "sc_cov_huber": sc_cov_huber,
            "sc_len_q0mean": sc_len_q0mean,
            "sc_cov_q0mean": sc_cov_q0mean,
            "sc_len_cip_global_mean": sc_len_cip_global_mean,
            "sc_cov_cip_global_mean": sc_cov_cip_global_mean,
            "sc_len_cip_group_mean": sc_len_cip_group_mean,
            "sc_cov_cip_group_mean": sc_cov_cip_group_mean,
            "mondrian_len_ridge": mondrian_len_ridge,
            "mondrian_cov_ridge": mondrian_cov_ridge,
            "time_pool_sample": float(time_pool_sample),
            "time_g_compute": float(time_g_compute),
            "time_dual_global": float(time_dual_global),
            "time_dual_group": float(time_dual_group),
            "time_smc_global": float(time_smc_global),
            "time_smc_group": float(time_smc_group),
            "time_temp_tune": float(time_temp_tune),
            "time_group_temp_tune": float(time_group_temp_tune),
            "time_feature_map": float(time_feature_map),
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
            row[f"eta_group_temp_g{g}"] = float(eta_group_tuned[g])
            row[f"gm_cip_global_g{g}_key"] = float(gm_cipg[g, key_k])
            row[f"gm_cip_group_g{g}_key"] = float(gm_cipgrp[g, key_k])

        if save_tail_curves and tail_curve_q0 is not None:
            for j, tt in enumerate(np.asarray(t_grid, dtype=float)):
                row[f"tail_t_{j}"] = float(tt)
                row[f"tail_q0_{j}"] = float(tail_curve_q0[j])
                row[f"tail_temp_{j}"] = float(tail_curve_temp[j])
                row[f"tail_group_temp_{j}"] = float(tail_curve_group_temp[j])
                row[f"tail_cip_global_{j}"] = float(tail_curve_cip_global[j])
                row[f"tail_cip_group_{j}"] = float(tail_curve_cip_group[j])

        rows.append(row)

        if verbose:
            print(
                f"[rep {rep+1:02d}/{int(R)}] "
                f"K={K} G={n_groups} tf={tighten_factor:.2g} | "
                f"eta_tuned={eta_tuned:.4g} (hitmax={row['eta_tuned_hit_max']:.0f}) | "
                f"worstV: q0={wv_q0_group:.3g}, temp={wv_temp_group:.3g}, grpTemp={wv_group_temp_group:.3g}, cipGrp={wv_cipgrp_group:.3g} | "
                f"SMC uniq={info_cip_group['unique_frac']:.2f} | "
                f"time={time_total:.2f}s"
            )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    if make_figures:
        # thresholds averaged over reps
        t_mean = np.mean(np.vstack(t_list), axis=0)

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
            plt.plot(t_grid, mean_curve, label=label)

        plt.scatter(t_mean, alpha_levels, marker="x", label="Targets (t_k, alpha_k)")
        plt.xlabel("Threshold t")
        plt.ylabel(r"Test tail miscoverage $\Pr(|Y-X^\top\beta|>t)$")
        plt.title("Tail curve (global), averaged over replicates")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "tail_curve_global.png"), dpi=200)
        plt.close()

        # (2) Global miscoverage at key threshold
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
        plt.title(f"Global miscoverage at t_key (alpha={alpha_levels[key_k]:.2f})")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "miscoverage_t1.png"), dpi=200)
        plt.close()

        # (3) Groupwise miscoverage at key threshold
        plt.figure()
        methods = ["q0", "TempTune", "GroupTemp", "CIP-Global", "CIP-Group"]
        G = int(n_groups)
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
            plt.bar(x + (g - (G - 1) / 2) * width, vals, width, label=f"Group {g}")

        plt.axhline(alpha_levels[key_k], linestyle="--")
        plt.xticks(x, methods, rotation=20)
        plt.ylabel("Miscoverage at key threshold")
        plt.title("Groupwise miscoverage at t_key (mean)")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "group_miscoverage_t1.png"), dpi=200)
        plt.close()

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

        plt.xlabel("Threshold t_k (mean over reps)")
        plt.ylabel("Test miscoverage at t_k")
        plt.title("Constraint satisfaction across multiple tail points")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "miscoverage_all_constraints.png"), dpi=200)
        plt.close()

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
        plt.title("Worst-group multi-threshold violation (mean ± std)")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "worst_group_violation.png"), dpi=200)
        plt.close()

        # (6) Conformal interval lengths
        if compute_conformal:
            plt.figure()
            cols_len = [
                "sc_len_ridge", "sc_len_huber", "sc_len_q0mean",
                "sc_len_cip_global_mean", "sc_len_cip_group_mean",
                "mondrian_len_ridge",
            ]
            labels_len = [
                "SC-Ridge", "SC-Huber", "SC-q0Mean",
                "SC-CIPGlobMean", "SC-CIPGrpMean",
                "Mondrian-Ridge",
            ]
            means_len = [df_out[c].mean() for c in cols_len]
            stds_len = [df_out[c].std() for c in cols_len]
            x = np.arange(len(cols_len))
            plt.bar(x, means_len, yerr=stds_len)
            plt.xticks(x, labels_len, rotation=25, ha="right")
            plt.ylabel("Average interval length")
            plt.title("Conformal interval lengths (mean ± std)")
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, "conformal_lengths.png"), dpi=200)
            plt.close()

    if verbose:
        print(f"\n[DONE] Saved metrics to: {os.path.join(outdir, 'metrics.csv')}")
        if make_figures:
            print(f"[DONE] Saved figures to: {figdir}")

    return df_out


# ============================================================
# Summaries and plotting helpers
# ============================================================

def summarize_df(df: pd.DataFrame) -> dict:
    """
    Convert per-replicate dataframe -> one-row summary dict.
    """
    out = {}
    for col in [
        "wv_group_q0", "wv_group_temp", "wv_group_group_temp", "wv_group_cip_global", "wv_group_cip_group",
        "mse_q0", "mse_temp", "mse_group_temp", "mse_cip_global", "mse_cip_group",
        "kl_cip_global", "kl_cip_group",
        "ess_is_global", "ess_is_group",
        "smc_global_unique", "smc_group_unique",
        "smc_global_essmin", "smc_group_essmin",
        "smc_global_accept", "smc_group_accept",
        "time_pool_sample", "time_g_compute", "time_dual_global", "time_dual_group",
        "time_smc_global", "time_smc_group", "time_temp_tune", "time_group_temp_tune", "time_feature_map", "time_total",
        "eta_tuned_hit_max",
    ]:
        if col in df.columns:
            out[col + "_mean"] = float(df[col].mean())
            out[col + "_std"] = float(df[col].std())
    if "wv_group_temp" in df.columns and "wv_group_cip_group" in df.columns:
        out["adv_temp_minus_cipgrp_mean"] = float((df["wv_group_temp"] - df["wv_group_cip_group"]).mean())
        out["adv_temp_minus_cipgrp_std"] = float((df["wv_group_temp"] - df["wv_group_cip_group"]).std())
    for col in ["K", "n_groups", "group_scale_max", "df", "hetero_strength", "nonlinear_signal", "posterior_model_code", "rf_dim", "rf_scale", "rf_include_linear", "feature_dim_input", "feature_dim_posterior", "m_thr", "m_proj", "tighten_factor", "S_pool", "n_particles"]:
        if col in df.columns:
            out[col] = float(df[col].iloc[0])
    return out


def plot_line_with_error(x, ys, yerrs, labels, xlabel, ylabel, title, outpath):
    plt.figure()
    for y, e, lab in zip(ys, yerrs, labels):
        plt.errorbar(x, y, yerr=e, marker="o", label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_scatter_frontier(xy_list, labels, xlabel, ylabel, title, outpath):
    plt.figure()
    for (x, y), lab in zip(xy_list, labels):
        plt.plot(x, y, marker="o", label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_heatmap(mat, x_ticks, y_ticks, xlabel, ylabel, title, outpath):
    plt.figure()
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.yticks(np.arange(len(y_ticks)), y_ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# Paper suite: ablations + frontier + diagnostics
# ============================================================

def run_paper_suite(base_outdir: str = "paper_suite", base_seed: int = 0, fast: bool = True):
    """
    Runs the next-step experiments described in the paper discussion.
    Outputs are saved under base_outdir/.

    fast=True:
      - fewer replicates in each sub-study (good for iteration)
    fast=False:
      - more replicates (paper-quality stability)
    """
    os.makedirs(base_outdir, exist_ok=True)
    figdir = os.path.join(base_outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    # Replicate budgets
    R_main = 20 if not fast else 8
    R_ablate = 12 if not fast else 6
    R_frontier = 12 if not fast else 6
    R_scaling = 6 if not fast else 3

    # Keep threshold and projection sizes explicit.
    # m_thr_per_group controls the held-out threshold split used to choose t_k,
    # while m_per_group controls the projection split used for CIP constraints.
    m_thr_per_group = 200
    m_per_group = 200

    # -------------------------------------------
    # (A) Main run (full figures)
    # -------------------------------------------
    print("\n=== (A) Main reference run ===")
    main_out = os.path.join(base_outdir, "main")
    df_main = run_simulation(
        outdir=main_out,
        seed=base_seed,
        R=R_main,
        n_groups=4,
        m_thr=int(m_thr_per_group * 4),
        m_proj=int(m_per_group * 4),
        alpha_levels=alpha_levels_for_K(5),
        tighten_factor=0.10,
        S_pool=40000,
        n_particles=4000,
        smc_steps_group=70,
        ess_threshold=0.40,
        rejuvenate_group=6,
        make_figures=True,
        compute_conformal=True,
        verbose=True,
    )
    main_sum = summarize_df(df_main)
    pd.DataFrame([main_sum]).to_csv(os.path.join(base_outdir, "main_summary.csv"), index=False)

    # -------------------------------------------
    # (B) Ablation: K x G grid
    # -------------------------------------------
    print("\n=== (B) Ablation: K x G grid ===")
    K_list = [2, 5, 8, 10]
    G_list = [2, 4, 6]

    ab_summaries = []
    adv_mat = np.zeros((len(G_list), len(K_list)), dtype=float)

    for gi, G in enumerate(G_list):
        for ki, K in enumerate(K_list):
            out = os.path.join(base_outdir, "ablation_KG", f"G{G}_K{K}")
            df_run = run_simulation(
                outdir=out,
                seed=base_seed,         # same seed across runs for comparability
                R=R_ablate,
                n_groups=G,
                m_thr=int(m_thr_per_group * G),
                m_proj=int(m_per_group * G),
                n_test=2000,            # speed for ablations
                alpha_levels=alpha_levels_for_K(K),
                tighten_factor=0.10,
                S_pool=40000,
                n_particles=4000,
                make_figures=False,
                compute_conformal=False,
                verbose=False,
            )
            summ = summarize_df(df_run)
            summ["K"] = K
            summ["n_groups"] = G
            summ["m_proj"] = int(m_per_group * G)
            ab_summaries.append(summ)
            adv_mat[gi, ki] = summ.get("adv_temp_minus_cipgrp_mean", np.nan)

    ab_df = pd.DataFrame(ab_summaries)
    ab_df.to_csv(os.path.join(base_outdir, "ablation_KG_summary.csv"), index=False)

    plot_heatmap(
        adv_mat,
        x_ticks=[str(K) for K in K_list],
        y_ticks=[str(G) for G in G_list],
        xlabel="K (number of tail constraints)",
        ylabel="G (number of groups)",
        title="Advantage: TempTune - CIP-Group (mean worst-group violation)",
        outpath=os.path.join(figdir, "heatmap_advantage_KG.png"),
    )

    # line plot: V vs K for each G
    for method_col, label in [
        ("wv_group_temp_mean", "TempTune"),
        ("wv_group_cip_group_mean", "CIP-Group"),
        ("wv_group_cip_global_mean", "CIP-Global"),
    ]:
        ys = []
        yerrs = []
        labs = []
        for G in G_list:
            sub = ab_df[ab_df["n_groups"] == float(G)].sort_values("K")
            ys.append(sub[method_col].values)
            yerrs.append(sub[method_col.replace("_mean", "_std")].values)
            labs.append(f"{label} (G={G})")
        plot_line_with_error(
            x=K_list,
            ys=ys,
            yerrs=yerrs,
            labels=labs,
            xlabel="K (number of tail constraints)",
            ylabel=r"Worst-group violation $\widehat V(q)$",
            title=f"Worst-group violation vs K ({label})",
            outpath=os.path.join(figdir, f"ablation_V_vs_K_{safe_dirname(label)}.png"),
        )

    # -------------------------------------------
    # (C) Severity ablations: group_scale_max, df, hetero_strength
    # -------------------------------------------
    print("\n=== (C) Ablations over shift severity and tails ===")

    def run_1d_ablation(param_name: str, values: list[float], fixed_G: int = 4, fixed_K: int = 5):
        res = []
        for val in values:
            out = os.path.join(base_outdir, f"ablation_{safe_dirname(param_name)}", f"{param_name}={val}")
            kwargs = {param_name: float(val)}
            df_run = run_simulation(
                outdir=out,
                seed=base_seed,
                R=R_ablate,
                n_groups=fixed_G,
                m_thr=int(m_thr_per_group * fixed_G),
                m_proj=int(m_per_group * fixed_G),
                n_test=2000,
                alpha_levels=alpha_levels_for_K(fixed_K),
                tighten_factor=0.10,
                S_pool=40000,
                n_particles=4000,
                make_figures=False,
                compute_conformal=False,
                verbose=False,
                **kwargs,
            )
            summ = summarize_df(df_run)
            summ[param_name] = float(val)
            res.append(summ)

        dfres = pd.DataFrame(res).sort_values(param_name)
        dfres.to_csv(os.path.join(base_outdir, f"ablation_{safe_dirname(param_name)}_summary.csv"), index=False)

        x = dfres[param_name].values
        plot_line_with_error(
            x=x,
            ys=[dfres["wv_group_temp_mean"].values, dfres["wv_group_cip_group_mean"].values],
            yerrs=[dfres["wv_group_temp_std"].values, dfres["wv_group_cip_group_std"].values],
            labels=["TempTune", "CIP-Group"],
            xlabel=param_name,
            ylabel=r"Worst-group violation $\widehat V(q)$",
            title=f"Ablation: worst-group violation vs {param_name}",
            outpath=os.path.join(figdir, f"ablation_V_vs_{safe_dirname(param_name)}.png"),
        )

    run_1d_ablation("group_scale_max", [1.5, 2.0, 3.0, 4.0])
    run_1d_ablation("df", [3.0, 5.0, 10.0])
    run_1d_ablation("hetero_strength", [0.0, 0.15, 0.30])

    # -------------------------------------------
    # (D) Calibration–accuracy frontier: tighten_factor sweep
    # -------------------------------------------
    print("\n=== (D) Calibration–accuracy frontier (tighten_factor sweep) ===")
    tf_vals = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    frontier_rows = []
    for tf in tf_vals:
        out = os.path.join(base_outdir, "frontier_tighten", f"tf={tf}")
        df_run = run_simulation(
            outdir=out,
            seed=base_seed,
            R=R_frontier,
            n_groups=4,
            m_thr=int(m_thr_per_group * 4),
            m_proj=int(m_per_group * 4),
            n_test=2000,
            alpha_levels=alpha_levels_for_K(5),
            tighten_factor=float(tf),
            S_pool=40000,
            n_particles=4000,
            make_figures=False,
            compute_conformal=False,
            verbose=False,
        )
        summ = summarize_df(df_run)
        summ["tighten_factor"] = float(tf)
        frontier_rows.append(summ)

    frontier_df = pd.DataFrame(frontier_rows).sort_values("tighten_factor")
    frontier_df.to_csv(os.path.join(base_outdir, "frontier_summary.csv"), index=False)

    plot_scatter_frontier(
        xy_list=[
            (frontier_df["wv_group_temp_mean"].values, frontier_df["mse_temp_mean"].values),
            (frontier_df["wv_group_cip_group_mean"].values, frontier_df["mse_cip_group_mean"].values),
        ],
        labels=["TempTune", "CIP-Group"],
        xlabel=r"Worst-group violation $\widehat V(q)$ (smaller is better)",
        ylabel="Test MSE (smaller is better)",
        title="Calibration–accuracy frontier via tightening",
        outpath=os.path.join(figdir, "frontier_V_vs_MSE.png"),
    )

    plot_line_with_error(
        x=frontier_df["kl_cip_group_mean"].values,
        ys=[frontier_df["wv_group_cip_group_mean"].values],
        yerrs=[frontier_df["wv_group_cip_group_std"].values],
        labels=["CIP-Group"],
        xlabel=r"Approx. KL$(q^\star\|q_0)$ (pool estimate)",
        ylabel=r"Worst-group violation $\widehat V(q)$",
        title="CIP-Group: calibration vs I-projection distance",
        outpath=os.path.join(figdir, "frontier_V_vs_KL.png"),
    )

    # -------------------------------------------
    # (E) Driving violations toward zero: increase m_proj per group + tighten
    # -------------------------------------------
    print("\n=== (E) Driving violations down (m_proj per group sweep) ===")
    m_per_vals = [100, 200, 400, 800]
    tf_for_zero = [0.0, 0.10, 0.20, 0.30]
    zero_rows = []
    for tf in tf_for_zero:
        for mp in m_per_vals:
            out = os.path.join(base_outdir, "ablation_mproj", f"tf={tf}_mper={mp}")
            df_run = run_simulation(
                outdir=out,
                seed=base_seed,
                R=R_ablate,
                n_groups=4,
                m_thr=int(m_thr_per_group * 4),
                m_proj=int(mp * 4),
                n_test=2000,
                alpha_levels=alpha_levels_for_K(5),
                tighten_factor=float(tf),
                S_pool=40000,
                n_particles=4000,
                make_figures=False,
                compute_conformal=False,
                verbose=False,
            )
            summ = summarize_df(df_run)
            summ["m_per_group"] = float(mp)
            summ["tighten_factor"] = float(tf)
            zero_rows.append(summ)

    zero_df = pd.DataFrame(zero_rows)
    zero_df.to_csv(os.path.join(base_outdir, "ablation_mproj_summary.csv"), index=False)

    plt.figure()
    for tf in tf_for_zero:
        sub = zero_df[zero_df["tighten_factor"] == float(tf)].sort_values("m_per_group")
        plt.errorbar(sub["m_per_group"], sub["wv_group_cip_group_mean"], yerr=sub["wv_group_cip_group_std"], marker="o", label=f"tf={tf}")
    plt.xlabel("m_proj per group")
    plt.ylabel(r"CIP-Group worst-group violation $\widehat V(q)$")
    plt.title("Driving worst-group violation down by enlarging projection set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "violation_vs_mproj_per_group.png"), dpi=200)
    plt.close()

    # MSE vs m_proj per group (to show we do not blow up accuracy)
    plt.figure()
    for tf in tf_for_zero:
        sub = zero_df[zero_df["tighten_factor"] == float(tf)].sort_values("m_per_group")
        plt.errorbar(
            sub["m_per_group"],
            sub["mse_cip_group_mean"],
            yerr=sub["mse_cip_group_std"],
            marker="o",
            label=f"CIP-Group tf={tf}",
        )
        # also show TempTune for the same (tf) since tightening affects its objective too
        plt.errorbar(
            sub["m_per_group"],
            sub["mse_temp_mean"],
            yerr=sub["mse_temp_std"],
            marker="x",
            linestyle="--",
            label=f"TempTune tf={tf}",
        )
    plt.xlabel("m_proj per group")
    plt.ylabel("Test MSE")
    plt.title("Accuracy vs projection set size (higher m_proj reduces generalization gap)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "mse_vs_mproj_per_group.png"), dpi=200)
    plt.close()

    # Joint view: (V, MSE) as we change m_proj and tighten_factor (CIP-Group)
    plt.figure()
    for tf in tf_for_zero:
        sub = zero_df[zero_df["tighten_factor"] == float(tf)].sort_values("m_per_group")
        plt.plot(
            sub["wv_group_cip_group_mean"],
            sub["mse_cip_group_mean"],
            marker="o",
            label=f"CIP-Group tf={tf}",
        )
    plt.xlabel(r"Worst-group violation $\widehat V(q)$")
    plt.ylabel("Test MSE")
    plt.title("CIP-Group: (calibration, accuracy) vs projection size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "frontier_mproj_V_vs_MSE_cipgroup.png"), dpi=200)
    plt.close()

    # -------------------------------------------
    # (F) Algorithmic scaling diagnostics: vs S_pool, K, G
    # -------------------------------------------
    print("\n=== (F) Algorithmic scaling diagnostics ===")
    scale_rows = []

    # scaling in S_pool
    S_list = [10000, 20000, 40000, 80000] if not fast else [10000, 20000, 40000]
    for S in S_list:
        out = os.path.join(base_outdir, "scaling", f"S_pool={S}")
        df_run = run_simulation(
            outdir=out,
            seed=base_seed,
            R=R_scaling,
            n_groups=4,
            m_thr=int(m_thr_per_group * 4),
            m_proj=int(m_per_group * 4),
            n_test=1000,
            alpha_levels=alpha_levels_for_K(5),
            tighten_factor=0.10,
            S_pool=int(S),
            n_particles=4000,
            make_figures=False,
            compute_conformal=False,
            verbose=False,
        )
        summ = summarize_df(df_run)
        summ["scale_axis"] = "S_pool"
        summ["scale_value"] = float(S)
        scale_rows.append(summ)

    # scaling in K
    for K in [2, 5, 8, 10]:
        out = os.path.join(base_outdir, "scaling", f"K={K}")
        df_run = run_simulation(
            outdir=out,
            seed=base_seed,
            R=R_scaling,
            n_groups=4,
            m_thr=int(m_thr_per_group * 4),
            m_proj=int(m_per_group * 4),
            n_test=1000,
            alpha_levels=alpha_levels_for_K(K),
            tighten_factor=0.10,
            S_pool=40000,
            n_particles=4000,
            make_figures=False,
            compute_conformal=False,
            verbose=False,
        )
        summ = summarize_df(df_run)
        summ["scale_axis"] = "K"
        summ["scale_value"] = float(K)
        scale_rows.append(summ)

    # scaling in G
    for G in [2, 4, 6]:
        out = os.path.join(base_outdir, "scaling", f"G={G}")
        df_run = run_simulation(
            outdir=out,
            seed=base_seed,
            R=R_scaling,
            n_groups=G,
            m_thr=int(m_thr_per_group * G),
            m_proj=int(m_per_group * G),
            n_test=1000,
            alpha_levels=alpha_levels_for_K(5),
            tighten_factor=0.10,
            S_pool=40000,
            n_particles=4000,
            make_figures=False,
            compute_conformal=False,
            verbose=False,
        )
        summ = summarize_df(df_run)
        summ["scale_axis"] = "G"
        summ["scale_value"] = float(G)
        scale_rows.append(summ)

    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(os.path.join(base_outdir, "scaling_summary.csv"), index=False)

    def cip_time(sdf: pd.DataFrame) -> np.ndarray:
        return sdf["time_g_compute_mean"].values + sdf["time_dual_group_mean"].values + sdf["time_smc_group_mean"].values

    for axis in ["S_pool", "K", "G"]:
        sub = scale_df[scale_df["scale_axis"] == axis].sort_values("scale_value")
        x = sub["scale_value"].values

        plot_line_with_error(
            x=x,
            ys=[sub["time_total_mean"].values, cip_time(sub)],
            yerrs=[sub["time_total_std"].values, np.zeros_like(x)],
            labels=["Total time", "CIP-group core time (g + dual + SMC)"],
            xlabel=axis,
            ylabel="Seconds",
            title=f"Runtime scaling vs {axis}",
            outpath=os.path.join(figdir, f"runtime_scaling_{axis}.png"),
        )

        plot_line_with_error(
            x=x,
            ys=[sub["ess_is_group_mean"].values, sub["smc_group_unique_mean"].values],
            yerrs=[sub["ess_is_group_std"].values, sub["smc_group_unique_std"].values],
            labels=["IS ESS (group)", "SMC unique fraction (group)"],
            xlabel=axis,
            ylabel="Diagnostic value",
            title=f"Degeneracy diagnostics vs {axis}",
            outpath=os.path.join(figdir, f"diagnostics_scaling_{axis}.png"),
        )

    print(f"\n[SUITE DONE] All outputs saved under: {base_outdir}")


if __name__ == "__main__":
    # fast=True is recommended while iterating.
    # For paper-quality plots: set fast=False.
    run_paper_suite(base_outdir="paper_suite", base_seed=0, fast=True)

