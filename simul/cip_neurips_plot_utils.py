#!/usr/bin/env python3
"""Plotting utilities for CIP experiments.

The functions in this file assume the metrics produced by the CIP job-array
aggregation scripts.  They are intentionally dependency-light: only NumPy,
Pandas, and Matplotlib are required.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Color-blind friendly palette close to Okabe--Ito.
# The line styles and markers are deliberately redundant with color so that the
# figures remain legible in grayscale printouts.
@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    tail_key: str
    mcov_prefix: str
    gm_prefix: str
    wv_col: str
    mse_col: str
    color: str
    linestyle: str
    marker: str


METHODS: List[MethodSpec] = [
    MethodSpec("q0", r"$q_0$", "q0", "mcov_q0", "gm_q0", "wv_group_q0", "mse_q0", "#0072B2", "-", "o"),
    MethodSpec("TempTune", "TempTune", "temp", "mcov_temp", "gm_temp", "wv_group_temp", "mse_temp", "#E69F00", "--", "s"),
    MethodSpec("GroupTemp", "GroupTemp", "group_temp", "mcov_group_temp", "gm_group_temp", "wv_group_group_temp", "mse_group_temp", "#009E73", "-.", "^"),
    MethodSpec("CIP-Global", "CIP-Global", "cip_global", "mcov_cip_global", "gm_cip_global", "wv_group_cip_global", "mse_cip_global", "#D55E00", ":", "D"),
    MethodSpec("CIP-Group", "CIP-Group", "cip_group", "mcov_cip_group", "gm_cip_group", "wv_group_cip_group", "mse_cip_group", "#CC79A7", "-", "P"),
]

METHOD_BY_NAME = {m.name: m for m in METHODS}


def set_paper_style(base_font_size: int = 10) -> None:
    """Matplotlib defaults suitable for NeurIPS main-paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size + 1,
        "legend.fontsize": base_font_size,
        "xtick.labelsize": base_font_size,
        "ytick.labelsize": base_font_size,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        # Avoid Type 3 fonts in PDFs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _finite_mean(x: pd.Series | np.ndarray | Sequence[float]) -> float:
    vals = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def _finite_std(x: pd.Series | np.ndarray | Sequence[float]) -> float:
    vals = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig: plt.Figure, outpath_no_ext: str | Path, dpi: int = 220) -> None:
    """Save a paper-ready PDF, and optionally a PNG.

    By default the plotting scripts save PDF only.  This is the right format
    for the NeurIPS paper and avoids slow PNG rasterization for dense panels.
    Set CIP_SAVE_PNG=1 in the environment if PNG copies are desired.
    """
    outpath_no_ext = Path(outpath_no_ext)
    ensure_dir(outpath_no_ext.parent)
    fig.savefig(str(outpath_no_ext) + ".pdf")
    if os.environ.get("CIP_SAVE_PNG", "0") == "1":
        fig.savefig(str(outpath_no_ext) + ".png", dpi=dpi)
    plt.close(fig)


def read_metrics(path_or_dir: str | Path) -> pd.DataFrame:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"Could not find metrics file: {p}")
    return pd.read_csv(p)


def read_summary(path_or_dir: str | Path) -> Optional[pd.DataFrame]:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def read_group_labels(path_or_dir: str | Path, n_groups: Optional[int] = None) -> List[str]:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "group_levels.csv"
    if p.exists():
        gdf = pd.read_csv(p)
        if {"group_code", "group_label"}.issubset(set(gdf.columns)):
            return [str(x) for x in gdf.sort_values("group_code")["group_label"].tolist()]
        if "group_label" in gdf.columns:
            return [str(x) for x in gdf["group_label"].tolist()]
    if n_groups is None:
        n_groups = 0
    return [f"Group {i+1}" for i in range(int(n_groups))]


def infer_K(df: pd.DataFrame) -> int:
    if "K" in df.columns:
        vals = pd.to_numeric(df["K"], errors="coerce").dropna()
        if len(vals):
            return int(round(float(vals.iloc[0])))
    idx = []
    for c in df.columns:
        m = re.fullmatch(r"alpha_(\d+)", str(c))
        if m:
            idx.append(int(m.group(1)))
    if idx:
        return max(idx) + 1
    raise ValueError("Could not infer K from metrics columns.")


def infer_n_groups(df: pd.DataFrame) -> int:
    if "n_groups" in df.columns:
        vals = pd.to_numeric(df["n_groups"], errors="coerce").dropna()
        if len(vals):
            return int(round(float(vals.iloc[0])))
    idx = []
    for c in df.columns:
        m = re.fullmatch(r"gm_[A-Za-z0-9_]+_g(\d+)_key", str(c))
        if m:
            idx.append(int(m.group(1)))
    return max(idx) + 1 if idx else 0


def infer_key_k(df: pd.DataFrame, alpha: np.ndarray) -> int:
    if "key_k" in df.columns:
        vals = pd.to_numeric(df["key_k"], errors="coerce").dropna()
        if len(vals):
            k = int(round(float(vals.mean())))
            if 0 <= k < len(alpha):
                return k
    return int(np.nanargmin(np.abs(alpha - 0.10)))


def alpha_t_from_metrics(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    K = infer_K(df)
    alpha = np.array([_finite_mean(df[f"alpha_{k}"]) if f"alpha_{k}" in df.columns else np.nan for k in range(K)], dtype=float)
    t = np.array([_finite_mean(df[f"t_{k}"]) if f"t_{k}" in df.columns else np.nan for k in range(K)], dtype=float)
    return alpha, t


def available_methods(df: pd.DataFrame, require: str = "any") -> List[MethodSpec]:
    """Return method specs whose columns are present.

    require can be 'any', 'wv', 'mse', 'key', or 'group'.
    """
    out = []
    for m in METHODS:
        checks = []
        if require in ("any", "wv"):
            checks.append(m.wv_col in df.columns)
        if require in ("any", "mse"):
            checks.append(m.mse_col in df.columns)
        if require in ("any", "key"):
            checks.append(f"{m.mcov_prefix}_key" in df.columns)
        if require in ("any", "group"):
            checks.append(any(c.startswith(f"{m.gm_prefix}_g") for c in df.columns))
        if any(checks):
            out.append(m)
    return out


def tail_indices(df: pd.DataFrame) -> List[int]:
    idx = []
    for c in df.columns:
        m = re.fullmatch(r"tail_t_(\d+)", str(c))
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def tail_curve_tidy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return tidy tail-curve DataFrames for methods and target points."""
    alpha, t = alpha_t_from_metrics(df)
    target = pd.DataFrame({"threshold": t, "target": alpha}).replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    idx = tail_indices(df)
    methods = available_methods(df)

    if idx:
        t_grid = np.array([_finite_mean(df[f"tail_t_{j}"]) for j in idx], dtype=float)
        for m in methods:
            cols = [f"tail_{m.tail_key}_{j}" for j in idx]
            if not all(c in df.columns for c in cols):
                continue
            for j, c in enumerate(cols):
                rows.append({
                    "method": m.name,
                    "threshold": t_grid[j],
                    "tail_rate": _finite_mean(df[c]),
                })
    else:
        K = len(alpha)
        for m in methods:
            for k in range(K):
                c = f"{m.mcov_prefix}_{k}"
                if c not in df.columns:
                    continue
                rows.append({
                    "method": m.name,
                    "threshold": t[k],
                    "tail_rate": _finite_mean(df[c]),
                })

    return pd.DataFrame(rows), target


def method_metric_tidy(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Tidy mean/std by method for metric in {'wv', 'mse', 'key'}.

    'key' means global key-threshold miscoverage.
    """
    rows = []
    for m in available_methods(df):
        if metric == "wv":
            col = m.wv_col
        elif metric == "mse":
            col = m.mse_col
        elif metric == "key":
            # Real-data metrics store mcov_*_key; simulation metrics store
            # mcov_*_{key_k}.  Support both.
            col = f"{m.mcov_prefix}_key"
            if col not in df.columns:
                alpha, _t = alpha_t_from_metrics(df)
                key_k = infer_key_k(df, alpha)
                alt = f"{m.mcov_prefix}_{key_k}"
                col = alt
        else:
            raise ValueError(metric)
        if col not in df.columns:
            continue
        rows.append({
            "method": m.name,
            "mean": _finite_mean(df[col]),
            "sd": _finite_std(df[col]),
        })
    return pd.DataFrame(rows)


def frontier_tidy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in available_methods(df):
        if m.wv_col not in df.columns or m.mse_col not in df.columns:
            continue
        rows.append({
            "method": m.name,
            "wv_mean": _finite_mean(df[m.wv_col]),
            "wv_sd": _finite_std(df[m.wv_col]),
            "mse_mean": _finite_mean(df[m.mse_col]),
            "mse_sd": _finite_std(df[m.mse_col]),
        })
    return pd.DataFrame(rows)


def group_key_tidy(df: pd.DataFrame, group_labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    G = infer_n_groups(df)
    if group_labels is None or len(group_labels) < G:
        group_labels = [f"Group {i+1}" for i in range(G)]
    rows = []
    for m in available_methods(df):
        for g in range(G):
            col = f"{m.gm_prefix}_g{g}_key"
            if col not in df.columns:
                continue
            rows.append({
                "method": m.name,
                "group": str(group_labels[g]),
                "group_idx": g,
                "mean": _finite_mean(df[col]),
                "sd": _finite_std(df[col]),
            })
    return pd.DataFrame(rows)


def maybe_set_ylim_from_data(ax: plt.Axes, values: Sequence[float], target: Optional[float] = None, pad: float = 0.10, floor_zero: bool = True) -> None:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if target is not None and np.isfinite(target):
        vals = np.r_[vals, target]
    if vals.size == 0:
        return
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo = lo - pad * span
    hi = hi + pad * span
    if floor_zero:
        lo = min(0.0, lo)
    ax.set_ylim(lo, hi)


def plot_global_tail_curve(ax: plt.Axes, tail_df: pd.DataFrame, target_df: pd.DataFrame, title: str = "Global tail curve") -> None:
    for m in METHODS:
        sub = tail_df[tail_df["method"] == m.name].sort_values("threshold")
        if len(sub) == 0:
            continue
        marker = m.marker if len(sub) <= 20 else None
        markevery = max(1, len(sub) // 8) if marker is not None else None
        ax.plot(
            sub["threshold"], sub["tail_rate"],
            color=m.color, linestyle=m.linestyle, marker=marker,
            markevery=markevery, linewidth=2.2, label=m.label, alpha=0.98,
        )
    if target_df is not None and len(target_df):
        ax.scatter(target_df["threshold"], target_df["target"], marker="x", s=55, linewidths=2.0, color="black", label="Targets", zorder=5)
    ax.set_title(title)
    ax.set_xlabel(r"Threshold $t$")
    ax.set_ylabel("Test tail rate")
    ax.grid(alpha=0.25, linewidth=0.7)


def plot_method_point_range(
    ax: plt.Axes,
    metric_df: pd.DataFrame,
    ylabel: str,
    title: str,
    target: Optional[float] = None,
    show_connecting_line: bool = True,
) -> None:
    metric_df = metric_df.copy()
    order = [m.name for m in METHODS if m.name in set(metric_df["method"])]
    metric_df["method"] = pd.Categorical(metric_df["method"], categories=order, ordered=True)
    metric_df = metric_df.sort_values("method")
    xs = np.arange(len(metric_df))
    vals_for_ylim = []
    for i, r in enumerate(metric_df.itertuples(index=False)):
        m = METHOD_BY_NAME[str(r.method)]
        ax.errorbar(
            i, float(r.mean), yerr=float(r.sd),
            fmt=m.marker, color=m.color, ecolor=m.color,
            elinewidth=1.7, capsize=3.5, capthick=1.2, markersize=7.5, zorder=3,
        )
        vals_for_ylim += [float(r.mean) - float(r.sd), float(r.mean) + float(r.sd)]
    if show_connecting_line and len(metric_df) > 1:
        ax.plot(xs, metric_df["mean"].to_numpy(dtype=float), color="0.5", linewidth=1.0, alpha=0.35, zorder=1)
    if target is not None:
        ax.axhline(target, linestyle="--", color="0.25", linewidth=1.2)
    labels = [METHOD_BY_NAME[str(m)].label for m in metric_df["method"].astype(str)]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    maybe_set_ylim_from_data(ax, vals_for_ylim, target=target, pad=0.12, floor_zero=True)


def plot_group_key_profile(
    ax: plt.Axes,
    gdf: pd.DataFrame,
    target: Optional[float] = None,
    title: str = "Groupwise miscoverage at key threshold",
    max_label_len: int = 16,
) -> None:
    if gdf.empty:
        ax.text(0.5, 0.5, "No groupwise columns found", ha="center", va="center")
        ax.set_axis_off()
        return
    groups = gdf.sort_values("group_idx")["group"].drop_duplicates().tolist()
    methods = [m for m in METHODS if m.name in set(gdf["method"])]
    x0 = np.arange(len(groups), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(methods)) if len(methods) > 1 else np.array([0.0])
    vals_for_ylim = []
    for mi, m in enumerate(methods):
        sub = gdf[gdf["method"] == m.name].copy()
        sub["group"] = pd.Categorical(sub["group"], categories=groups, ordered=True)
        sub = sub.sort_values("group")
        xs = x0 + offsets[mi]
        y = sub["mean"].to_numpy(dtype=float)
        yerr = sub["sd"].to_numpy(dtype=float)
        ax.errorbar(
            xs, y, yerr=yerr, fmt=m.marker, color=m.color, ecolor=m.color,
            linestyle="none", elinewidth=1.2, capsize=2.5, markersize=6.0, label=m.label,
        )
        vals_for_ylim.extend((y - yerr).tolist())
        vals_for_ylim.extend((y + yerr).tolist())
    if target is not None:
        ax.axhline(target, linestyle="--", color="0.25", linewidth=1.2)
    labels = [g if len(g) <= max_label_len else g[: max_label_len - 1] + "…" for g in groups]
    ax.set_xticks(x0)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Group tail rate")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    maybe_set_ylim_from_data(ax, vals_for_ylim, target=target, pad=0.14, floor_zero=True)


def plot_calibration_accuracy(
    ax: plt.Axes,
    fdf: pd.DataFrame,
    title: str = "Calibration--accuracy",
    annotate: bool = True,
) -> None:
    if fdf.empty:
        ax.text(0.5, 0.5, "No frontier columns found", ha="center", va="center")
        ax.set_axis_off()
        return
    vals_x = []
    vals_y = []
    for _, r in fdf.iterrows():
        m = METHOD_BY_NAME[str(r["method"])]
        x = float(r["wv_mean"])
        y = float(r["mse_mean"])
        vals_x.append(x)
        vals_y.append(y)
        ax.scatter(x, y, s=75, color=m.color, marker=m.marker, edgecolor="black", linewidth=0.45, zorder=4)
        if annotate:
            # Small method-specific offsets to reduce overlap in common cases.
            dx, dy = {
                "q0": (5, 3),
                "TempTune": (5, -12),
                "GroupTemp": (5, 4),
                "CIP-Global": (5, 5),
                "CIP-Group": (5, -12),
            }.get(m.name, (5, 4))
            ax.annotate(m.label, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=9)
    ax.set_xlabel(r"Worst-group violation $\widehat V(q)$")
    ax.set_ylabel("Predictive MSE")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.7)
    if vals_x:
        xspan = max(vals_x) - min(vals_x)
        yspan = max(vals_y) - min(vals_y)
        if xspan == 0: xspan = max(0.01, abs(vals_x[0]) * 0.05)
        if yspan == 0: yspan = max(0.01, abs(vals_y[0]) * 0.05)
        ax.set_xlim(min(vals_x) - 0.12 * xspan, max(vals_x) + 0.20 * xspan)
        ax.set_ylim(min(vals_y) - 0.15 * yspan, max(vals_y) + 0.20 * yspan)


def add_method_legend(fig: plt.Figure, methods: Sequence[MethodSpec], y: float = 0.995, ncol: Optional[int] = None) -> None:
    handles = []
    labels = []
    for m in methods:
        handles.append(Line2D([0], [0], color=m.color, linestyle=m.linestyle, marker=m.marker, linewidth=2.0, markersize=6))
        labels.append(m.label)
    handles.append(Line2D([0], [0], color="black", linestyle="None", marker="x", markersize=7, markeredgewidth=1.8))
    labels.append("Targets")
    fig.legend(handles, labels, loc="upper center", ncol=ncol or min(6, len(labels)), frameon=False, bbox_to_anchor=(0.5, y))


def make_paper_summary_2x2(
    metrics: pd.DataFrame,
    group_labels: Optional[Sequence[str]] = None,
    title_prefix: str = "",
    include_global_key: bool = False,
) -> plt.Figure:
    """Create the main 2x2 panel used in the paper.

    Panels: global tail curve, groupwise key-threshold profile, worst-group
    violation, and calibration--accuracy frontier.
    """
    alpha, _ = alpha_t_from_metrics(metrics)
    key_k = infer_key_k(metrics, alpha)
    key_alpha = float(alpha[key_k]) if len(alpha) else 0.10
    tail_df, target_df = tail_curve_tidy(metrics)
    group_df = group_key_tidy(metrics, group_labels=group_labels)
    wv_df = method_metric_tidy(metrics, "wv")
    frontier_df = frontier_tidy(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.4))
    plot_global_tail_curve(axes[0, 0], tail_df, target_df, title="(a) Global tail curve")
    plot_group_key_profile(axes[0, 1], group_df, target=key_alpha, title="(b) Groups at key threshold")
    plot_method_point_range(axes[1, 0], wv_df, ylabel=r"$\widehat V(q)$", title="(c) Worst-group violation")
    plot_calibration_accuracy(axes[1, 1], frontier_df, title="(d) Calibration--accuracy")

    # Keep the group legend local, and use a global legend for methods.
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(handles, labels, loc="best", ncol=2, fontsize=8, frameon=True)
    methods = available_methods(metrics)
    add_method_legend(fig, methods, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def write_latex_snippet(fig_path: str, label: str, caption: str, outpath: str | Path) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)
    outpath.write_text(
        "% Auto-generated by CIP plotting script.\n"
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{fig_path}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "\\end{figure}\n",
        encoding="utf-8",
    )


def find_existing(*paths: str | Path) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def friendly_dataset_name(name: str) -> str:
    name = name.lower()
    if "diamond" in name:
        return "Diamonds"
    if "bike" in name:
        return "Bike Sharing"
    if "sim" in name or "main" in name:
        return "Simulation"
    return name

# ============================================================
# Cross-run / nonlinear-posterior comparison figures
# ============================================================

def short_method_label(method: str) -> str:
    return METHOD_BY_NAME[method].label if method in METHOD_BY_NAME else str(method)


def compact_run_label(label: object, max_len: int = 30) -> str:
    """Readable labels for comparison figures."""
    s = str(label)
    repl = {
        "nonlinear_signal_linear_posterior": "linear posterior",
        "nonlinear_signal_rff128": "RFF-128",
        "nonlinear_signal_rff256": "RFF-256",
        "diamonds_blind": "Diamonds blind",
        "diamonds_with_cut_drop1_cutprior1": r"Diamonds + cut",
        "diamonds_blind_rff256": "Diamonds blind RFF",
        "diamonds_with_cut_rff256": "Diamonds + cut RFF",
        "bike_blind": "Bike blind",
        "bike_with_group_drop1_groupPrior1": "Bike + season",
        "bike_blind_rff256": "Bike blind RFF",
        "bike_with_group_rff256": "Bike + season RFF",
    }
    if s in repl:
        return repl[s]
    # Generic cleanup.
    s = s.replace("_drop1_groupPrior1", "")
    s = s.replace("with_group", "+ group")
    s = s.replace("with_cut", "+ cut")
    s = s.replace("_rff", " RFF-")
    s = s.replace("_", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _summary_col_for_method(method: MethodSpec, metric: str) -> str:
    if metric == "wv":
        return method.wv_col
    if metric == "key":
        return f"{method.mcov_prefix}_key"
    if metric == "mse":
        return method.mse_col
    raise ValueError(f"unknown metric: {metric}")


def summary_metric_tidy(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Tidy a summary_all table for cross-run plotting.

    The aggregation scripts write columns such as wv_group_cip_group_mean and
    wv_group_cip_group_std.  This function converts them to a tidy table with
    one row per run and method.
    """
    if summary is None or len(summary) == 0:
        return pd.DataFrame()
    run_col = "run_name" if "run_name" in summary.columns else None
    if run_col is None:
        for c in ["label", "array_run_label", "setting"]:
            if c in summary.columns:
                run_col = c
                break
    if run_col is None:
        return pd.DataFrame()

    rows = []
    for _, row in summary.iterrows():
        run_name = str(row[run_col])
        posterior_model = str(row.get("posterior_model", ""))
        rf_dim = row.get("rf_dim", np.nan)
        for m in METHODS:
            base = _summary_col_for_method(m, metric)
            mean_col = f"{base}_mean"
            sd_col = f"{base}_std"
            if mean_col not in summary.columns:
                continue
            mean = pd.to_numeric(pd.Series([row.get(mean_col, np.nan)]), errors="coerce").iloc[0]
            sd = pd.to_numeric(pd.Series([row.get(sd_col, 0.0)]), errors="coerce").iloc[0] if sd_col in summary.columns else 0.0
            if not np.isfinite(mean):
                continue
            rows.append({
                "run_name": run_name,
                "run_label": compact_run_label(run_name),
                "posterior_model": posterior_model,
                "rf_dim": rf_dim,
                "method": m.name,
                "mean": float(mean),
                "sd": float(sd) if np.isfinite(sd) else 0.0,
            })
    return pd.DataFrame(rows)


def order_summary_runs(summary: pd.DataFrame) -> List[str]:
    if summary is None or len(summary) == 0:
        return []
    run_col = "run_name" if "run_name" in summary.columns else None
    if run_col is None:
        return []
    runs = [str(x) for x in summary[run_col].drop_duplicates().tolist()]
    preferred = [
        "main",
        "nonlinear_signal_linear_posterior",
        "nonlinear_signal_rff128",
        "nonlinear_signal_rff256",
        "diamonds_blind",
        "diamonds_blind_rff256",
        "diamonds_with_cut_drop1_cutprior1",
        "diamonds_with_cut_rff256",
        "bike_blind",
        "bike_blind_rff256",
        "bike_with_group_drop1_groupPrior1",
        "bike_with_group_rff256",
    ]
    out = [r for r in preferred if r in runs]
    out += [r for r in runs if r not in out]
    return out


def filter_relevant_comparison_rows(summary: pd.DataFrame, max_runs: int = 8) -> pd.DataFrame:
    """Keep the main linear/nonlinear rows, dropping large K-ablation rows by default."""
    if summary is None or summary.empty or "run_name" not in summary.columns:
        return pd.DataFrame()
    d = summary.copy()
    names = d["run_name"].astype(str)
    # Drop K-only ablations unless there are no other rows.
    non_k = d[~names.str.contains(r"_K\d+$", regex=True)].copy()
    if len(non_k):
        d = non_k
    # Prefer rows that compare linear and RFF models.
    names = d["run_name"].astype(str)
    mask = (
        names.str.contains("rff", case=False, regex=False)
        | names.str.contains("nonlinear_signal", case=False, regex=False)
        | names.isin([
            "main",
            "diamonds_blind",
            "diamonds_with_cut_drop1_cutprior1",
            "bike_blind",
            "bike_with_group_drop1_groupPrior1",
        ])
    )
    if mask.any():
        d = d[mask].copy()
    order = order_summary_runs(d)
    if order:
        d["_order"] = d["run_name"].astype(str).map({r: i for i, r in enumerate(order)})
        d = d.sort_values("_order").drop(columns=["_order"])
    if len(d) > max_runs:
        d = d.iloc[:max_runs].copy()
    return d


def plot_summary_method_comparison(
    ax: plt.Axes,
    tidy: pd.DataFrame,
    ylabel: str,
    title: str,
    run_order: Optional[Sequence[str]] = None,
) -> None:
    """Grouped mean+-sd plot across run settings and methods."""
    if tidy is None or tidy.empty:
        ax.text(0.5, 0.5, "No comparison rows", ha="center", va="center")
        ax.set_axis_off()
        return
    if run_order is None:
        run_order = tidy["run_name"].drop_duplicates().tolist()
    run_order = [r for r in run_order if r in set(tidy["run_name"])]
    methods = [m for m in METHODS if m.name in set(tidy["method"])]
    x = np.arange(len(run_order), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods)) if len(methods) > 1 else np.array([0.0])
    vals = []
    for mi, m in enumerate(methods):
        sub = tidy[tidy["method"] == m.name].copy()
        sub["run_name"] = pd.Categorical(sub["run_name"], categories=run_order, ordered=True)
        sub = sub.sort_values("run_name")
        # Align missing runs.
        by_run = {str(r.run_name): r for r in sub.itertuples(index=False)}
        xs, ys, yerrs = [], [], []
        for ri, rname in enumerate(run_order):
            if rname not in by_run:
                continue
            rr = by_run[rname]
            xs.append(x[ri] + offsets[mi])
            ys.append(float(rr.mean))
            yerrs.append(float(rr.sd))
        if not xs:
            continue
        vals.extend((np.asarray(ys) - np.asarray(yerrs)).tolist())
        vals.extend((np.asarray(ys) + np.asarray(yerrs)).tolist())
        ax.errorbar(
            xs, ys, yerr=yerrs,
            fmt=m.marker, color=m.color, ecolor=m.color,
            linestyle="none", markersize=5.6, elinewidth=1.1,
            capsize=2.2, capthick=0.9, label=m.label,
        )
    labels = [compact_run_label(r, max_len=22) for r in run_order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    maybe_set_ylim_from_data(ax, vals, pad=0.12, floor_zero=True)


def plot_summary_comparison_panels(summary: pd.DataFrame, outpath_no_ext: str | Path, title_prefix: str = "") -> None:
    """Three-panel comparison: worst-group violation, key miscoverage, MSE."""
    d = filter_relevant_comparison_rows(summary)
    if d.empty or "run_name" not in d.columns or d["run_name"].nunique() <= 1:
        return
    run_order = order_summary_runs(d)
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.25))
    plot_summary_method_comparison(
        axes[0], summary_metric_tidy(d, "wv"),
        ylabel=r"$\widehat V(q)$", title="Worst-group violation", run_order=run_order,
    )
    plot_summary_method_comparison(
        axes[1], summary_metric_tidy(d, "key"),
        ylabel="Key-threshold tail rate", title="Global key threshold", run_order=run_order,
    )
    plot_summary_method_comparison(
        axes[2], summary_metric_tidy(d, "mse"),
        ylabel="Predictive MSE", title="Prediction error", run_order=run_order,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False)
    if title_prefix:
        fig.suptitle(title_prefix, y=1.08, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_fig(fig, outpath_no_ext)


def plot_summary_frontier_small_multiples(summary: pd.DataFrame, outpath_no_ext: str | Path, title_prefix: str = "") -> None:
    """Calibration--accuracy frontiers for each linear/nonlinear run setting."""
    d = filter_relevant_comparison_rows(summary, max_runs=6)
    if d.empty or "run_name" not in d.columns or d["run_name"].nunique() <= 1:
        return
    run_order = order_summary_runs(d)
    run_order = [r for r in run_order if r in set(d["run_name"].astype(str))]
    n = len(run_order)
    ncols = 2 if n <= 4 else 3
    nrows = int(math.ceil(n / ncols))
    set_paper_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.25 * nrows), squeeze=False)
    all_front = frontier_tidy_from_summary(d)
    xvals = all_front["wv_mean"].to_numpy(dtype=float) if len(all_front) else np.array([])
    yvals = all_front["mse_mean"].to_numpy(dtype=float) if len(all_front) else np.array([])
    for idx, rname in enumerate(run_order):
        ax = axes[idx // ncols, idx % ncols]
        sub_summary = d[d["run_name"].astype(str) == rname]
        fdf = frontier_tidy_from_summary(sub_summary)
        plot_calibration_accuracy(ax, fdf, title=compact_run_label(rname), annotate=True)
        # Common axis ranges make comparisons easier.
        if len(xvals):
            xspan = max(xvals) - min(xvals)
            if xspan <= 0:
                xspan = max(0.01, abs(float(xvals[0])) * 0.05)
            ax.set_xlim(min(xvals) - 0.10 * xspan, max(xvals) + 0.22 * xspan)
        if len(yvals):
            yspan = max(yvals) - min(yvals)
            if yspan <= 0:
                yspan = max(0.01, abs(float(yvals[0])) * 0.05)
            ax.set_ylim(min(yvals) - 0.12 * yspan, max(yvals) + 0.18 * yspan)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()
    if title_prefix:
        fig.suptitle(title_prefix, y=1.02, fontsize=12)
    fig.tight_layout()
    save_fig(fig, outpath_no_ext)


def frontier_tidy_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary is None or summary.empty:
        return pd.DataFrame()
    for _, row in summary.iterrows():
        run_name = str(row.get("run_name", ""))
        for m in METHODS:
            wv_mean_col = f"{m.wv_col}_mean"
            mse_mean_col = f"{m.mse_col}_mean"
            if wv_mean_col not in summary.columns or mse_mean_col not in summary.columns:
                continue
            wv = pd.to_numeric(pd.Series([row.get(wv_mean_col, np.nan)]), errors="coerce").iloc[0]
            mse = pd.to_numeric(pd.Series([row.get(mse_mean_col, np.nan)]), errors="coerce").iloc[0]
            if not (np.isfinite(wv) and np.isfinite(mse)):
                continue
            rows.append({
                "run_name": run_name,
                "method": m.name,
                "wv_mean": float(wv),
                "wv_sd": float(row.get(f"{m.wv_col}_std", 0.0)) if f"{m.wv_col}_std" in summary.columns else 0.0,
                "mse_mean": float(mse),
                "mse_sd": float(row.get(f"{m.mse_col}_std", 0.0)) if f"{m.mse_col}_std" in summary.columns else 0.0,
            })
    return pd.DataFrame(rows)


def write_comparison_latex_snippet(fig_path: str, label: str, caption: str, outpath: str | Path) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)
    outpath.write_text(
        "% Auto-generated by CIP plotting script.\n"
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{fig_path}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "\\end{figure}\n",
        encoding="utf-8",
    )

# ============================================================
# Nonlinear / posterior-family comparison helpers
# ============================================================

def safe_dirname(x: str) -> str:
    """Filesystem-safe version of a run label."""
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(x)).strip("_")


def _get_row_value(row, key: str, default=None):
    try:
        if isinstance(row, pd.Series):
            return row.get(key, default)
        return getattr(row, key, default)
    except Exception:
        return default


def friendly_setting_label(row_or_name, dataset: str = "") -> str:
    """Short label for a row in *_summary_all.csv / nonlinear_summary.csv.

    The labels are tuned for the CIP job-array outputs but degrade gracefully for
    other run names.
    """
    if isinstance(row_or_name, (str, bytes)):
        name = str(row_or_name)
        posterior = ""
        rf_dim = None
        include_group = None
    else:
        name = str(_get_row_value(row_or_name, "run_name", _get_row_value(row_or_name, "label", _get_row_value(row_or_name, "setting_label", "setting"))))
        posterior = str(_get_row_value(row_or_name, "posterior_model", "")).lower()
        rf_dim = _get_row_value(row_or_name, "rf_dim", None)
        include_group = _get_row_value(row_or_name, "include_group_in_X", None)

    lname = name.lower()
    dname = str(dataset).lower()

    # Synthetic nonlinear suite.
    if "nonlinear_signal_linear" in lname:
        return "Linear posterior"
    m = re.search(r"nonlinear_signal_rff(\d+)", lname)
    if m:
        return f"RFF-{m.group(1)}"

    # Diamonds.
    if "diamond" in lname or "diamond" in dname:
        if "rff" in lname or posterior == "rff":
            dim = re.search(r"rff(\d+)", lname)
            dimtxt = dim.group(1) if dim else (str(int(float(rf_dim))) if rf_dim is not None and pd.notna(rf_dim) else "")
            base = f"RFF-{dimtxt}" if dimtxt else "RFF"
        else:
            base = "Linear"
        if "with_cut" in lname or "with_group" in lname or (include_group is not None and float(include_group) > 0.5):
            return f"{base}, cut included"
        if "blind" in lname:
            return f"{base}, cut excluded"
        return base

    # Bike Sharing.
    if "bike" in lname or "bike" in dname:
        if "rff" in lname or posterior == "rff":
            dim = re.search(r"rff(\d+)", lname)
            dimtxt = dim.group(1) if dim else (str(int(float(rf_dim))) if rf_dim is not None and pd.notna(rf_dim) else "")
            base = f"RFF-{dimtxt}" if dimtxt else "RFF"
        else:
            base = "Linear"
        if "with_group" in lname or (include_group is not None and float(include_group) > 0.5):
            return f"{base}, season included"
        if "blind" in lname:
            return f"{base}, season excluded"
        return base

    # Generic fallback.
    if posterior == "rff":
        try:
            return f"RFF-{int(float(rf_dim))}"
        except Exception:
            return "RFF"
    if posterior in {"linear", "none", ""}:
        if name == "main":
            return "Default"
        return name.replace("_", " ")
    return name.replace("_", " ")


def summary_metric_tidy(
    summary: pd.DataFrame,
    metric: str,
    dataset: str = "",
    methods: Optional[Sequence[MethodSpec]] = None,
) -> pd.DataFrame:
    """Tidy mean/std values from an aggregated summary file.

    Parameters
    ----------
    summary:
        A dataframe such as diamonds_summary_all.csv, bike_summary_all.csv,
        nonlinear_summary.csv, or sim_summary_all.csv.
    metric:
        One of {'wv', 'mse', 'key'}.
    """
    if methods is None:
        methods = METHODS
    rows = []
    for si, (_, r) in enumerate(summary.iterrows()):
        setting_raw = str(r.get("run_name", r.get("label", r.get("setting_label", f"setting_{si}"))))
        setting = friendly_setting_label(r, dataset=dataset)
        for m in methods:
            if metric == "wv":
                base = m.wv_col
            elif metric == "mse":
                base = m.mse_col
            elif metric == "key":
                base = f"{m.mcov_prefix}_key"
            else:
                raise ValueError(f"Unknown metric: {metric}")
            mean_col = base + "_mean"
            sd_col = base + "_std"
            if mean_col not in summary.columns:
                continue
            mean = pd.to_numeric(pd.Series([r.get(mean_col)]), errors="coerce").iloc[0]
            sd = pd.to_numeric(pd.Series([r.get(sd_col, 0.0)]), errors="coerce").iloc[0]
            if not np.isfinite(mean):
                continue
            rows.append({
                "setting_raw": setting_raw,
                "setting": setting,
                "setting_idx": si,
                "method": m.name,
                "mean": float(mean),
                "sd": float(sd) if np.isfinite(sd) else 0.0,
                "posterior_model": str(r.get("posterior_model", "")),
                "rf_dim": r.get("rf_dim", np.nan),
            })
    return pd.DataFrame(rows)


def plot_summary_metric_by_setting(
    ax: plt.Axes,
    tidy: pd.DataFrame,
    ylabel: str,
    title: str,
    target: Optional[float] = None,
    rotate_xticks: int = 25,
) -> None:
    """Grouped mean +/- sd plot from summary_metric_tidy()."""
    if tidy.empty:
        ax.text(0.5, 0.5, "No summary columns found", ha="center", va="center")
        ax.set_axis_off()
        return
    settings = tidy.sort_values("setting_idx")["setting"].drop_duplicates().tolist()
    methods = [m for m in METHODS if m.name in set(tidy["method"])]
    x0 = np.arange(len(settings), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(methods)) if len(methods) > 1 else np.array([0.0])
    vals = []
    for mi, m in enumerate(methods):
        sub = tidy[tidy["method"] == m.name].copy()
        sub["setting"] = pd.Categorical(sub["setting"], categories=settings, ordered=True)
        sub = sub.sort_values("setting")
        x = []
        y = []
        e = []
        for sj, setting in enumerate(settings):
            ss = sub[sub["setting"] == setting]
            if ss.empty:
                continue
            row = ss.iloc[0]
            x.append(x0[sj] + offsets[mi])
            y.append(float(row["mean"]))
            e.append(float(row["sd"]))
        if not x:
            continue
        x = np.asarray(x); y = np.asarray(y); e = np.asarray(e)
        vals.extend((y - e).tolist()); vals.extend((y + e).tolist())
        ax.errorbar(
            x, y, yerr=e, fmt=m.marker, color=m.color, ecolor=m.color,
            linestyle="none", elinewidth=1.2, capsize=2.5, capthick=1.0,
            markersize=6.2, label=m.label, zorder=3,
        )
        # A light connecting line makes the posterior-family trend easier to read.
        if len(x) > 1:
            ax.plot(x, y, color=m.color, linestyle=m.linestyle, linewidth=1.2, alpha=0.55, zorder=2)
    if target is not None:
        ax.axhline(target, linestyle="--", color="0.25", linewidth=1.2)
    ax.set_xticks(x0)
    ax.set_xticklabels(settings, rotation=rotate_xticks, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    maybe_set_ylim_from_data(ax, vals, target=target, pad=0.14, floor_zero=True)


def make_summary_comparison_figure(
    summary: pd.DataFrame,
    dataset: str = "",
    title_prefix: str = "",
    include_key: bool = True,
) -> Optional[plt.Figure]:
    """Posterior-family comparison figure from *_summary_all.csv.

    This is intended for the nonlinear/RFF runs.  It keeps the existing method
    colors and markers, but puts posterior family / design setting on the x-axis.
    """
    if summary is None or summary.empty:
        return None
    panels = []
    key_df = summary_metric_tidy(summary, "key", dataset=dataset) if include_key else pd.DataFrame()
    if include_key and not key_df.empty:
        panels.append((key_df, "Miscoverage at key threshold", "Key-threshold miscoverage", 0.10))
    wv_df = summary_metric_tidy(summary, "wv", dataset=dataset)
    if not wv_df.empty:
        panels.append((wv_df, r"Worst-group violation $\widehat V(q)$", "Worst-group violation", None))
    mse_df = summary_metric_tidy(summary, "mse", dataset=dataset)
    if not mse_df.empty:
        panels.append((mse_df, "Predictive MSE", "Predictive accuracy", None))
    if not panels:
        return None
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.25), squeeze=False)
    for ax, (tdf, ylabel, title, target) in zip(axes[0], panels):
        plot_summary_metric_by_setting(ax, tdf, ylabel=ylabel, title=title, target=target)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    if title_prefix:
        fig.suptitle(title_prefix, y=1.08, fontsize=plt.rcParams.get("axes.titlesize", 12) + 1)
        top = 0.82
    else:
        top = 0.84
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig

# ============================================================
# Nonlinear / approximate-posterior comparison helpers
# ============================================================

def _row_get(row, key: str, default=None):
    """Get a value from a pandas Series/dict-like row."""
    try:
        if key in row:
            return row[key]
    except Exception:
        pass
    return default


def summary_label_col(summary_df: pd.DataFrame) -> str:
    """Return the column containing the run/setting label."""
    for c in ["run_name", "label", "array_run_label", "setting_label"]:
        if c in summary_df.columns:
            return c
    raise ValueError("Could not find a run label column in the summary dataframe.")


def model_display_label(row, dataset: str = "") -> str:
    """Readable setting label for linear vs random-feature posterior comparisons."""
    label_col_val = None
    for c in ["run_name", "label", "array_run_label", "setting_label"]:
        val = _row_get(row, c, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            label_col_val = str(val)
            break
    raw = label_col_val or "setting"
    low = raw.lower()

    model = str(_row_get(row, "posterior_model", "")).lower()
    rf_dim = _row_get(row, "rf_dim", np.nan)
    try:
        rf_dim_int = int(round(float(rf_dim))) if np.isfinite(float(rf_dim)) else None
    except Exception:
        rf_dim_int = None

    # Simulation nonlinear settings.
    if "nonlinear_signal" in low or str(_row_get(row, "suite", "")).lower() == "nonlinear":
        if "rff" in low or model == "rff":
            if rf_dim_int:
                return f"RFF-{rf_dim_int}"
            return "RFF"
        return "Linear"

    # Real-data settings: keep group structure visible.
    if "blind" in low:
        base = "Blind"
    elif "with_group" in low or "with_cut" in low or bool(float(_row_get(row, "include_group_in_X", 0) or 0)):
        base = "Group-aware"
    else:
        base = "Setting"

    if "rff" in low or model == "rff":
        return f"{base} RFF-{rf_dim_int}" if rf_dim_int else f"{base} RFF"
    return f"{base} linear"


def nonlinear_subset(summary_df: pd.DataFrame, dataset: str = "") -> pd.DataFrame:
    """Select rows relevant for nonlinear-posterior comparison.

    For simulation this means suite == nonlinear.  For real datasets we keep the
    two main linear settings and their RFF counterparts, when present.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    df = summary_df.copy()
    label_col = summary_label_col(df)
    lab = df[label_col].astype(str).str.lower()
    dataset = dataset.lower()

    if "suite" in df.columns and (df["suite"].astype(str).str.lower() == "nonlinear").any():
        return df[df["suite"].astype(str).str.lower() == "nonlinear"].copy()

    if "diamond" in dataset:
        keep = lab.isin([
            "diamonds_blind",
            "diamonds_with_cut_drop1_cutprior1",
            "diamonds_blind_rff256",
            "diamonds_with_cut_rff256",
        ]) | lab.str.contains("rff")
    elif "bike" in dataset:
        keep = lab.isin([
            "bike_blind",
            "bike_with_group_drop1_groupprior1",
            "bike_blind_rff256",
            "bike_with_group_rff256",
        ]) | lab.str.contains("rff")
    else:
        keep = lab.str.contains("rff|nonlinear")

    out = df[keep].copy()
    # Drop K-ablation rows unless they are explicitly nonlinear; otherwise the
    # comparison figure becomes too crowded.
    if label_col in out.columns:
        labels = out[label_col].astype(str).str.lower()
        out = out[~labels.str.contains(r"_k\d+") | labels.str.contains("rff")].copy()
    return out


def summary_method_metric_tidy(summary_df: pd.DataFrame, metric: str, dataset: str = "") -> pd.DataFrame:
    """Tidy dataframe from one-row-per-setting summaries.

    metric in {'wv', 'mse', 'key'}.
    Output columns:
        setting, setting_raw, method, mean, sd, setting_order
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    label_col = summary_label_col(summary_df)
    rows = []
    for order_idx, (_, row) in enumerate(summary_df.reset_index(drop=True).iterrows()):
        setting_raw = str(row[label_col])
        setting = model_display_label(row, dataset=dataset)
        for m in METHODS:
            if metric == "wv":
                base = m.wv_col
            elif metric == "mse":
                base = m.mse_col
            elif metric == "key":
                base = f"{m.mcov_prefix}_key"
            else:
                raise ValueError(metric)
            mean_col = base + "_mean"
            sd_col = base + "_std"
            if mean_col not in summary_df.columns:
                continue
            mean_val = pd.to_numeric(pd.Series([row.get(mean_col, np.nan)]), errors="coerce").iloc[0]
            sd_val = pd.to_numeric(pd.Series([row.get(sd_col, 0.0)]), errors="coerce").iloc[0] if sd_col in summary_df.columns else 0.0
            if pd.isna(mean_val):
                continue
            rows.append({
                "setting": setting,
                "setting_raw": setting_raw,
                "setting_order": order_idx,
                "method": m.name,
                "mean": float(mean_val),
                "sd": float(sd_val) if pd.notna(sd_val) else 0.0,
            })
    return pd.DataFrame(rows)


def plot_summary_metric_by_setting(
    ax: plt.Axes,
    tidy_df: pd.DataFrame,
    ylabel: str,
    title: str,
    target: Optional[float] = None,
    rotate_xticks: int = 18,
) -> None:
    """Plot mean +/- sd by setting, with method colors/markers/line styles."""
    if tidy_df is None or tidy_df.empty:
        ax.text(0.5, 0.5, "No summary rows found", ha="center", va="center")
        ax.set_axis_off()
        return
    d = tidy_df.copy()
    settings = d.sort_values("setting_order")["setting"].drop_duplicates().tolist()
    x = np.arange(len(settings), dtype=float)
    vals_for_ylim = []
    for m in METHODS:
        sub = d[d["method"] == m.name].copy()
        if sub.empty:
            continue
        sub["setting"] = pd.Categorical(sub["setting"], categories=settings, ordered=True)
        sub = sub.sort_values("setting")
        y = sub["mean"].to_numpy(dtype=float)
        yerr = sub["sd"].to_numpy(dtype=float)
        xs = np.array([settings.index(str(s)) for s in sub["setting"].astype(str)], dtype=float)
        ax.errorbar(
            xs, y, yerr=yerr,
            color=m.color, linestyle=m.linestyle, marker=m.marker,
            linewidth=2.0, markersize=5.8, capsize=2.8, capthick=1.0,
            elinewidth=1.0, label=m.label,
        )
        vals_for_ylim.extend((y - yerr).tolist())
        vals_for_ylim.extend((y + yerr).tolist())
    if target is not None:
        ax.axhline(target, linestyle="--", color="0.25", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=rotate_xticks, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    maybe_set_ylim_from_data(ax, vals_for_ylim, target=target, pad=0.12, floor_zero=True)


def make_nonlinear_comparison_figure(summary_df: pd.DataFrame, dataset: str, include_key: bool = True) -> Optional[plt.Figure]:
    """Create a compact comparison across linear and nonlinear posterior families."""
    sub = nonlinear_subset(summary_df, dataset=dataset)
    if sub.empty:
        return None
    key_df = summary_method_metric_tidy(sub, "key", dataset=dataset)
    wv_df = summary_method_metric_tidy(sub, "wv", dataset=dataset)
    mse_df = summary_method_metric_tidy(sub, "mse", dataset=dataset)
    if include_key and not key_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.45))
        plot_summary_metric_by_setting(axes[0], key_df, "Key-threshold miscoverage", "(a) Global key threshold", target=0.10)
        plot_summary_metric_by_setting(axes[1], wv_df, r"Worst-group violation $\widehat V(q)$", "(b) Worst-group violation")
        plot_summary_metric_by_setting(axes[2], mse_df, "Predictive MSE", "(c) Accuracy")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45))
        plot_summary_metric_by_setting(axes[0], wv_df, r"Worst-group violation $\widehat V(q)$", "(a) Worst-group violation")
        plot_summary_metric_by_setting(axes[1], mse_df, "Predictive MSE", "(b) Accuracy")
    handles, labels = [], []
    for m in METHODS:
        if m.name in set(pd.concat([wv_df.get("method", pd.Series(dtype=str)), mse_df.get("method", pd.Series(dtype=str))], ignore_index=True)):
            handles.append(Line2D([0], [0], color=m.color, linestyle=m.linestyle, marker=m.marker, linewidth=2.0, markersize=6))
            labels.append(m.label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def make_tail_curve_grid_from_run_dirs(run_dirs: Sequence[str | Path], titles: Optional[Sequence[str]] = None) -> Optional[plt.Figure]:
    """Side-by-side tail curves for a small set of run directories."""
    pairs = []
    for rd in run_dirs:
        rd = Path(rd)
        p = rd / "metrics.csv"
        if p.exists():
            pairs.append((rd, read_metrics(p)))
    if not pairs:
        return None
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.3), sharey=True)
    if n == 1:
        axes = [axes]
    for j, (ax, (rd, metrics)) in enumerate(zip(axes, pairs)):
        tail_df, target_df = tail_curve_tidy(metrics)
        title = titles[j] if titles is not None and j < len(titles) else rd.name
        plot_global_tail_curve(ax, tail_df, target_df, title=title)
        if j > 0:
            ax.set_ylabel("")
        if j != n - 1:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.04))
        leg = axes[-1].get_legend()
        if leg is not None:
            leg.remove()
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig

# ---------------------------------------------------------------------------
# Public helpers used by the standalone drawing scripts for v4 nonlinear runs.
# These names are intentionally simple so the figure scripts can stay small.
# ---------------------------------------------------------------------------

def pretty_run_label(label: object, max_len: int = 22) -> str:
    s = str(label)
    low = s.lower()
    if low == "main":
        return "Main\nlinear"
    if "nonlinear_signal_linear" in low:
        return "Nonlinear DGP\nlinear"
    if "nonlinear_signal_rff" in low:
        dim = re.sub(r".*rff", "", low)
        return f"Nonlinear DGP\nRFF-{dim}" if dim else "Nonlinear DGP\nRFF"
    if low == "diamonds_blind":
        return "Diamonds blind\nlinear"
    if low == "diamonds_with_cut_drop1_cutprior1":
        return "Diamonds cut-in\nlinear"
    if low.startswith("diamonds_blind_rff"):
        return "Diamonds blind\nRFF-" + re.sub(r".*rff", "", low)
    if low.startswith("diamonds_with_cut_rff"):
        return "Diamonds cut-in\nRFF-" + re.sub(r".*rff", "", low)
    if low == "bike_blind":
        return "Bike blind\nlinear"
    if low == "bike_with_group_drop1_groupprior1":
        return "Bike season-in\nlinear"
    if low.startswith("bike_blind_rff"):
        return "Bike blind\nRFF-" + re.sub(r".*rff", "", low)
    if low.startswith("bike_with_group_rff"):
        return "Bike season-in\nRFF-" + re.sub(r".*rff", "", low)
    s = s.replace("_", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def read_summary_all(path_or_dir: str | Path, candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    p = Path(path_or_dir)
    if p.is_file():
        return pd.read_csv(p)
    for name in candidates:
        q = p / name
        if q.exists():
            return pd.read_csv(q)
    return None


def filter_summary_runs(summary: pd.DataFrame, keep: Optional[Sequence[str]] = None, drop_k_sensitivity: bool = True) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    out = summary.copy()
    if "run_name" not in out.columns and "label" in out.columns:
        out = out.rename(columns={"label": "run_name"})
    if "run_name" not in out.columns:
        return out
    if keep is not None:
        keep_set = {str(x) for x in keep}
        out = out[out["run_name"].astype(str).isin(keep_set)]
    elif drop_k_sensitivity:
        out = out[~out["run_name"].astype(str).str.contains(r"_K\d+$", regex=True)]
    return out.reset_index(drop=True)


def _summary_metric_tidy_v4(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    d = summary.copy()
    if "run_name" not in d.columns and "label" in d.columns:
        d = d.rename(columns={"label": "run_name"})
    if "run_name" not in d.columns:
        return pd.DataFrame()
    colmap = {
        "wv": {
            "q0": ("wv_group_q0_mean", "wv_group_q0_std"),
            "TempTune": ("wv_group_temp_mean", "wv_group_temp_std"),
            "GroupTemp": ("wv_group_group_temp_mean", "wv_group_group_temp_std"),
            "CIP-Global": ("wv_group_cip_global_mean", "wv_group_cip_global_std"),
            "CIP-Group": ("wv_group_cip_group_mean", "wv_group_cip_group_std"),
        },
        "key": {
            "q0": ("mcov_q0_key_mean", "mcov_q0_key_std"),
            "TempTune": ("mcov_temp_key_mean", "mcov_temp_key_std"),
            "GroupTemp": ("mcov_group_temp_key_mean", "mcov_group_temp_key_std"),
            "CIP-Global": ("mcov_cip_global_key_mean", "mcov_cip_global_key_std"),
            "CIP-Group": ("mcov_cip_group_key_mean", "mcov_cip_group_key_std"),
        },
        "mse": {
            "q0": ("mse_q0_mean", "mse_q0_std"),
            "TempTune": ("mse_temp_mean", "mse_temp_std"),
            "GroupTemp": ("mse_group_temp_mean", "mse_group_temp_std"),
            "CIP-Global": ("mse_cip_global_mean", "mse_cip_global_std"),
            "CIP-Group": ("mse_cip_group_mean", "mse_cip_group_std"),
        },
    }[metric]
    rows = []
    for _, r in d.iterrows():
        for method, (mc, sc) in colmap.items():
            if mc not in d.columns:
                continue
            val = pd.to_numeric(pd.Series([r.get(mc, np.nan)]), errors="coerce").iloc[0]
            if not np.isfinite(val):
                continue
            sd = pd.to_numeric(pd.Series([r.get(sc, 0.0)]), errors="coerce").iloc[0] if sc in d.columns else 0.0
            rows.append({"run_name": str(r["run_name"]), "method": method, "mean": float(val), "sd": float(sd) if np.isfinite(sd) else 0.0})
    return pd.DataFrame(rows)


def _plot_summary_metric_by_run_v4(ax: plt.Axes, summary: pd.DataFrame, metric: str, ylabel: str, title: str, run_order: Optional[Sequence[str]] = None, target: Optional[float] = None) -> None:
    tidy = _summary_metric_tidy_v4(summary, metric)
    if tidy.empty:
        ax.text(0.5, 0.5, "No summary rows", ha="center", va="center")
        ax.set_axis_off(); return
    if run_order is None:
        run_order = tidy["run_name"].drop_duplicates().tolist()
    run_order = [str(x) for x in run_order if str(x) in set(tidy["run_name"])]
    methods = [m for m in METHODS if m.name in set(tidy["method"])]
    x0 = np.arange(len(run_order), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods)) if len(methods) > 1 else np.array([0.0])
    vals = []
    for mi, m in enumerate(methods):
        sub = tidy[tidy["method"] == m.name].copy()
        sub["run_name"] = pd.Categorical(sub["run_name"], categories=run_order, ordered=True)
        sub = sub.sort_values("run_name").set_index("run_name").reindex(run_order).reset_index()
        y = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
        err = pd.to_numeric(sub["sd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        xs = x0 + offsets[mi]
        ax.errorbar(xs[mask], y[mask], yerr=err[mask], fmt=m.marker, color=m.color, ecolor=m.color,
                    linestyle="none", markersize=5.7, elinewidth=1.1, capsize=2.5, capthick=1.0, label=m.label)
        vals.extend((y[mask]-err[mask]).tolist()); vals.extend((y[mask]+err[mask]).tolist())
    if target is not None and np.isfinite(target):
        ax.axhline(float(target), linestyle="--", color="0.25", linewidth=1.2)
    ax.set_xticks(x0); ax.set_xticklabels([pretty_run_label(x) for x in run_order], rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    maybe_set_ylim_from_data(ax, vals, target=target, pad=0.13, floor_zero=True)


def plot_summary_comparison_1x3(summary: pd.DataFrame, outpath_no_ext: str | Path, run_order: Optional[Sequence[str]] = None, target: Optional[float] = 0.10) -> None:
    if summary is None or summary.empty:
        return
    if "run_name" not in summary.columns and "label" in summary.columns:
        summary = summary.rename(columns={"label": "run_name"})
    if "run_name" not in summary.columns or summary["run_name"].nunique() <= 1:
        return
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55))
    _plot_summary_metric_by_run_v4(axes[0], summary, "key", "Key-threshold tail rate", "(a) Key threshold", run_order=run_order, target=target)
    _plot_summary_metric_by_run_v4(axes[1], summary, "wv", r"$\widehat V(q)$", "(b) Worst-group violation", run_order=run_order)
    _plot_summary_metric_by_run_v4(axes[2], summary, "mse", "Predictive MSE", "(c) Predictive accuracy", run_order=run_order)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, outpath_no_ext)


def plot_cip_gain_vs_baseline(summary: pd.DataFrame, outpath_no_ext: str | Path, baseline: str = "GroupTemp") -> None:
    if summary is None or summary.empty:
        return
    if "run_name" not in summary.columns and "label" in summary.columns:
        summary = summary.rename(columns={"label": "run_name"})
    base = {
        "TempTune": ("wv_group_temp_mean", "mse_temp_mean"),
        "GroupTemp": ("wv_group_group_temp_mean", "mse_group_temp_mean"),
        "CIP-Global": ("wv_group_cip_global_mean", "mse_cip_global_mean"),
    }
    if baseline not in base or "run_name" not in summary.columns:
        return
    wv_b, mse_b = base[baseline]
    needed = [wv_b, mse_b, "wv_group_cip_group_mean", "mse_cip_group_mean"]
    if not all(c in summary.columns for c in needed):
        return
    d = summary.copy()
    d["calib_gain"] = pd.to_numeric(d[wv_b], errors="coerce") - pd.to_numeric(d["wv_group_cip_group_mean"], errors="coerce")
    d["mse_delta"] = pd.to_numeric(d["mse_cip_group_mean"], errors="coerce") - pd.to_numeric(d[mse_b], errors="coerce")
    d = d[np.isfinite(d["calib_gain"]) & np.isfinite(d["mse_delta"])].copy()
    if d.empty:
        return
    x = np.arange(len(d)); labels = [pretty_run_label(v) for v in d["run_name"]]
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    color = METHOD_BY_NAME["CIP-Group"].color; marker = METHOD_BY_NAME["CIP-Group"].marker
    axes[0].axhline(0, color="0.25", linewidth=1.0)
    axes[0].scatter(x, d["calib_gain"], color=color, marker=marker, s=65)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel(rf"$\widehat V({baseline}) - \widehat V(\mathrm{{CIP\mbox{{-}}Group}})$")
    axes[0].set_title("(a) Calibration gain")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.7)
    axes[1].axhline(0, color="0.25", linewidth=1.0)
    axes[1].scatter(x, d["mse_delta"], color=color, marker=marker, s=65)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel(rf"MSE(CIP-Group) $-$ MSE({baseline})")
    axes[1].set_title("(b) Accuracy cost")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout(); save_fig(fig, outpath_no_ext)
