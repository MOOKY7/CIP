#!/usr/bin/env python3
"""Draw figures for the synthetic simulation suite.

This script supports both the original simulation aggregation and the nonlinear
random-feature simulation outputs.  It keeps the existing figure style and adds
nonlinear comparison figures, including the missing global key-threshold
miscoverage comparison.

Examples
--------
Original full simulation:
    python draw_sim_figures_neurips.py --outdir sim_neurips_array

Nonlinear-only aggregation:
    python draw_sim_figures_neurips.py --outdir sim_neurips_array_nonlinear

Specific nonlinear run:
    python draw_sim_figures_neurips.py --outdir sim_neurips_array_nonlinear --run_label nonlinear_signal_rff256
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cip_neurips_plot_utils import (
    METHODS,
    METHOD_BY_NAME,
    ensure_dir,
    read_metrics,
    save_fig,
    set_paper_style,
    make_paper_summary_2x2,
    write_latex_snippet,
    infer_n_groups,
    tail_curve_tidy,
    alpha_t_from_metrics,
    infer_key_k,
    method_metric_tidy,
    group_key_tidy,
    frontier_tidy,
    plot_global_tail_curve,
    plot_method_point_range,
    plot_group_key_profile,
    plot_calibration_accuracy,
)

ABLATION_METHODS = [
    ("wv_group_temp_mean", "wv_group_temp_std", "TempTune", "#E69F00", "--", "s"),
    ("wv_group_group_temp_mean", "wv_group_group_temp_std", "GroupTemp", "#009E73", "-.", "^"),
    ("wv_group_cip_global_mean", "wv_group_cip_global_std", "CIP-Global", "#D55E00", ":", "D"),
    ("wv_group_cip_group_mean", "wv_group_cip_group_std", "CIP-Group", "#CC79A7", "-", "P"),
]


def safe_dirname(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(x)).strip("_")


def normalized_label(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def available_metric_runs(outdir: Path) -> List[str]:
    if not outdir.exists():
        return []
    return [p.name for p in sorted(outdir.iterdir()) if p.is_dir() and (p / "metrics.csv").exists()]


def labels_from_summaries(outdir: Path) -> List[str]:
    labels: List[str] = []
    for fname in ["nonlinear_summary.csv", "main_summary.csv", "sim_summary_all.csv"]:
        p = outdir / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        c = first_existing_col(df, ["label", "setting_label", "run_name"])
        if c is not None:
            labels.extend([str(x) for x in df[c].dropna().tolist()])
    out: List[str] = []
    seen = set()
    for x in labels:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_run_metrics(outdir: Path, run_label: str) -> Path:
    raw = str(run_label)
    candidates = [
        outdir / raw / "metrics.csv",
        outdir / safe_dirname(raw) / "metrics.csv",
        outdir / raw.replace("-", "_") / "metrics.csv",
        outdir / safe_dirname(raw.replace("-", "_")) / "metrics.csv",
    ]
    p = existing(*candidates)
    if p is not None:
        return p
    target = normalized_label(raw)
    for lab in available_metric_runs(outdir) + labels_from_summaries(outdir):
        if normalized_label(lab) == target:
            p = outdir / safe_dirname(lab) / "metrics.csv"
            if p.exists():
                return p
    runs = available_metric_runs(outdir)
    msg = [f"Could not find metrics for run_label={run_label!r} under {outdir}."]
    if runs:
        msg.append("Available completed run directories:")
        msg.extend([f"  - {r}" for r in runs[:80]])
    raise FileNotFoundError("\n".join(msg))


def resolve_main_metrics(outdir: Path, explicit: Optional[str], run_label: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    if run_label:
        return resolve_run_metrics(outdir, run_label)
    return existing(outdir / "main" / "metrics.csv", outdir / "metrics.csv")


def load_summary(outdir: Path, filename: str, suite: Optional[str] = None) -> pd.DataFrame:
    p = outdir / filename
    if p.exists():
        df = pd.read_csv(p)
    else:
        allp = outdir / "sim_summary_all.csv"
        if not allp.exists() or suite is None:
            return pd.DataFrame()
        df = pd.read_csv(allp)
    if suite is not None and "suite" in df.columns:
        df = df[df["suite"].astype(str) == str(suite)].copy()
    return df


def posterior_label(row: pd.Series, fallback: str = "") -> str:
    label = str(row.get("label", row.get("setting_label", fallback)))
    code = row.get("posterior_model_code", np.nan)
    rf_dim = row.get("rf_dim", np.nan)
    try:
        code_i = int(round(float(code)))
    except Exception:
        code_i = None
    if code_i == 0 or "linear_posterior" in label.lower():
        return "Linear posterior"
    if code_i == 1 or "rff" in label.lower():
        if pd.notna(rf_dim):
            try:
                d = int(round(float(rf_dim)))
                if d > 0:
                    return f"RFF-{d}"
            except Exception:
                pass
        m = re.search(r"rff[_-]?(\d+)", label.lower())
        if m:
            return f"RFF-{int(m.group(1))}"
        return "RFF posterior"
    return label.replace("_", " ")


def nonlinear_run_rows(outdir: Path) -> pd.DataFrame:
    df = load_summary(outdir, "nonlinear_summary.csv", "nonlinear")
    if df.empty:
        rows = []
        for lab in available_metric_runs(outdir):
            if "nonlinear" in lab.lower() or "rff" in lab.lower():
                rows.append({"label": lab, "suite": "nonlinear"})
        df = pd.DataFrame(rows)
    if df.empty:
        return df
    c = first_existing_col(df, ["label", "setting_label", "run_name"])
    if c is None:
        return pd.DataFrame()
    df = df.copy()
    df["_label"] = df[c].astype(str)
    df["_dirname"] = df["_label"].map(safe_dirname)
    df["_display"] = [posterior_label(r, str(r["_label"])) for _, r in df.iterrows()]
    df = df[(outdir / df["_dirname"] / "metrics.csv").map(lambda p: Path(p).exists())] if False else df
    keep = []
    for _, r in df.iterrows():
        keep.append((outdir / str(r["_dirname"]) / "metrics.csv").exists())
    df = df.loc[keep].copy()
    def key(r: pd.Series) -> Tuple[int, int, str]:
        disp = str(r["_display"])
        if disp.startswith("Linear"):
            return (0, 0, disp)
        m = re.search(r"RFF-(\d+)", disp)
        if m:
            return (1, int(m.group(1)), disp)
        return (2, 0, disp)
    if len(df):
        df["_sort"] = [key(r) for _, r in df.iterrows()]
        df = df.sort_values("_sort").drop(columns=["_sort"])
    return df


def draw_single_metrics(metrics_path: Path, figdir: Path, figure_path_in_latex: str, label: str = "Simulation") -> None:
    metrics = read_metrics(metrics_path)
    G = infer_n_groups(metrics)
    group_labels = [f"Group {g+1}" for g in range(G)]
    set_paper_style()
    fig = make_paper_summary_2x2(metrics, group_labels=group_labels, title_prefix=label)
    save_fig(fig, figdir / "paper_summary_2x2")

    alpha, _ = alpha_t_from_metrics(metrics)
    key_alpha = float(alpha[infer_key_k(metrics, alpha)]) if len(alpha) else 0.10
    tail_df, target_df = tail_curve_tidy(metrics)
    panels = [
        ("global_tail_curve", lambda ax: plot_global_tail_curve(ax, tail_df, target_df, title=f"{label}: global tail curve")),
        ("global_key_miscoverage", lambda ax: plot_method_point_range(ax, method_metric_tidy(metrics, "key"), ylabel="Miscoverage at key threshold", title=f"{label}: global key-threshold miscoverage", target=key_alpha)),
        ("groupwise_key_miscoverage", lambda ax: plot_group_key_profile(ax, group_key_tidy(metrics, group_labels), target=key_alpha, title=f"{label}: groupwise key-threshold miscoverage")),
        ("worst_group_violation", lambda ax: plot_method_point_range(ax, method_metric_tidy(metrics, "wv"), ylabel=r"Worst-group violation $\widehat V(q)$", title=f"{label}: worst-group violation")),
        ("calibration_accuracy_frontier", lambda ax: plot_calibration_accuracy(ax, frontier_tidy(metrics), title=f"{label}: calibration--accuracy")),
    ]
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fn(ax)
        fig.tight_layout()
        save_fig(fig, figdir / name)

    write_latex_snippet(
        figure_path_in_latex,
        "fig:sim_results",
        "Synthetic regression results. Point ranges show mean $\\pm$ one standard deviation over random replicates; GroupTemp denotes the group-specific temperature baseline.",
        figdir / "figure_snippet.tex",
    )


def nonlinear_comparison_long(outdir: Path) -> Tuple[pd.DataFrame, float]:
    runs = nonlinear_run_rows(outdir)
    rows: List[Dict[str, object]] = []
    key_targets: List[float] = []
    for _, run in runs.iterrows():
        p = outdir / str(run["_dirname"]) / "metrics.csv"
        if not p.exists():
            continue
        metrics = read_metrics(p)
        alpha, _ = alpha_t_from_metrics(metrics)
        if len(alpha):
            key_targets.append(float(alpha[infer_key_k(metrics, alpha)]))
        for metric in ["key", "wv", "mse"]:
            mdf = method_metric_tidy(metrics, metric)
            for _, r in mdf.iterrows():
                rows.append({
                    "posterior": str(run["_display"]),
                    "run_label": str(run["_label"]),
                    "metric": metric,
                    "method": str(r["method"]),
                    "mean": float(r["mean"]),
                    "sd": float(r["sd"]),
                })
    target = float(np.nanmean(key_targets)) if key_targets else 0.10
    return pd.DataFrame(rows), target


def plot_metric_by_posterior(ax: plt.Axes, long_df: pd.DataFrame, metric: str, ylabel: str, title: str, target: Optional[float] = None) -> None:
    sub = long_df[long_df["metric"] == metric].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"No {metric} data", ha="center", va="center")
        ax.set_axis_off()
        return
    posters = sub["posterior"].drop_duplicates().tolist()
    x = np.arange(len(posters), dtype=float)
    vals: List[float] = []
    for m in METHODS:
        msub = sub[sub["method"] == m.name]
        if msub.empty:
            continue
        xs, means, sds = [], [], []
        for i, post in enumerate(posters):
            row = msub[msub["posterior"] == post]
            if row.empty:
                continue
            xs.append(i)
            means.append(float(row["mean"].iloc[0]))
            sds.append(float(row["sd"].iloc[0]))
        if not means:
            continue
        means = np.asarray(means, dtype=float)
        sds = np.asarray(sds, dtype=float)
        ax.errorbar(xs, means, yerr=sds, color=m.color, linestyle=m.linestyle, marker=m.marker, linewidth=1.9, elinewidth=1.1, capsize=2.8, markersize=6.0, label=m.label)
        vals.extend((means - sds).tolist())
        vals.extend((means + sds).tolist())
    if target is not None:
        ax.axhline(target, color="0.25", linestyle="--", linewidth=1.15)
        vals.append(float(target))
    ax.set_xticks(x)
    ax.set_xticklabels(posters, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size:
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            hi = lo + max(0.01, abs(lo) * 0.1)
        pad = 0.12 * (hi - lo)
        ax.set_ylim(min(0.0, lo - pad), hi + pad)


def draw_nonlinear_comparison_figures(outdir: Path, figdir: Path) -> None:
    ndir = ensure_dir(figdir / "nonlinear")
    long_df, target = nonlinear_comparison_long(outdir)
    if long_df.empty:
        return

    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    plot_metric_by_posterior(ax, long_df, "key", "Miscoverage at key threshold", "Nonlinear simulation: global key-threshold miscoverage", target=target)
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h, l, loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    save_fig(fig, ndir / "nonlinear_global_key_miscoverage_by_posterior")

    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.65))
    plot_metric_by_posterior(axes[0], long_df, "key", "Miscoverage", "(a) Key threshold", target=target)
    plot_metric_by_posterior(axes[1], long_df, "wv", r"$\widehat V(q)$", "(b) Worst-group violation")
    plot_metric_by_posterior(axes[2], long_df, "mse", "Predictive MSE", "(c) Predictive accuracy")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(5, len(l)), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, ndir / "nonlinear_key_wv_mse_by_posterior")

    # Backward-compatible old name, now still written for existing LaTeX or notes.
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65))
    plot_metric_by_posterior(axes[0], long_df, "wv", r"$\widehat V(q)$", "(a) Worst-group violation")
    plot_metric_by_posterior(axes[1], long_df, "mse", "Predictive MSE", "(b) Predictive accuracy")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(5, len(l)), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, ndir / "nonlinear_wv_mse_by_posterior")


def draw_nonlinear_tail_curve_grid(outdir: Path, figdir: Path) -> None:
    runs = nonlinear_run_rows(outdir)
    if runs.empty:
        return
    rows = []
    for _, r in runs.iterrows():
        p = outdir / str(r["_dirname"]) / "metrics.csv"
        if p.exists():
            rows.append((str(r["_display"]), p))
    if not rows:
        return
    ndir = ensure_dir(figdir / "nonlinear")
    n = len(rows)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    set_paper_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols, 3.15 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    legend_handles, legend_labels = None, None
    for ax, (title, p) in zip(axes.ravel(), rows):
        ax.set_axis_on()
        metrics = read_metrics(p)
        tail_df, target_df = tail_curve_tidy(metrics)
        plot_global_tail_curve(ax, tail_df, target_df, title=title)
        h, l = ax.get_legend_handles_labels()
        if legend_handles is None and h:
            legend_handles, legend_labels = h, l
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=min(6, len(legend_labels)), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, ndir / "nonlinear_tail_curve_grid")


def draw_nonlinear_figures(outdir: Path, figdir: Path, also_per_run: bool = False) -> None:
    if nonlinear_run_rows(outdir).empty:
        return
    draw_nonlinear_comparison_figures(outdir, figdir)
    draw_nonlinear_tail_curve_grid(outdir, figdir)
    if also_per_run:
        for _, row in nonlinear_run_rows(outdir).iterrows():
            p = outdir / str(row["_dirname"]) / "metrics.csv"
            if not p.exists():
                continue
            this_figdir = ensure_dir(outdir / str(row["_dirname"]) / "figures_neurips")
            fig_path = f"{row['_dirname']}/figures_neurips/paper_summary_2x2.pdf"
            draw_single_metrics(p, this_figdir, fig_path, label=str(row["_display"]))
    write_latex_snippet(
        "figures_neurips/nonlinear/nonlinear_key_wv_mse_by_posterior.pdf",
        "fig:sim_nonlinear_comparison",
        "Nonlinear simulation comparison. The RFF settings use a random-feature last-layer Gibbs posterior; point ranges show mean $\\pm$ one standard deviation over random replicates.",
        figdir / "nonlinear" / "figure_snippet.tex",
    )


def plot_ablation_line(df: pd.DataFrame, xcol: str, outpath: Path, xlabel: str, title: str) -> None:
    if df.empty or xcol not in df.columns:
        return
    df = df.copy()
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df = df.dropna(subset=[xcol]).sort_values(xcol)
    if df.empty:
        return
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for mean_col, sd_col, label, color, ls, marker in ABLATION_METHODS:
        if mean_col not in df.columns:
            continue
        y = pd.to_numeric(df[mean_col], errors="coerce").to_numpy(dtype=float)
        if sd_col in df.columns:
            yerr = pd.to_numeric(df[sd_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            yerr = np.zeros_like(y)
        ax.errorbar(df[xcol], y, yerr=yerr, color=color, linestyle=ls, marker=marker, capsize=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Worst-group violation $\widehat V(q)$")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_fig(fig, outpath)


def plot_kg_grid(kg: pd.DataFrame, outpath: Path) -> None:
    if kg.empty or "K" not in kg.columns or "n_groups" not in kg.columns:
        return
    kg = kg.copy()
    kg["K"] = pd.to_numeric(kg["K"], errors="coerce")
    kg["n_groups"] = pd.to_numeric(kg["n_groups"], errors="coerce")
    groups = sorted(kg["n_groups"].dropna().unique())
    if not groups:
        return
    set_paper_style()
    fig, axes = plt.subplots(1, len(groups), figsize=(4.1 * len(groups), 3.35), squeeze=False)
    for ax, G in zip(axes.ravel(), groups):
        sub = kg[kg["n_groups"] == G].sort_values("K")
        for mean_col, sd_col, label, color, ls, marker in ABLATION_METHODS:
            if mean_col not in sub.columns:
                continue
            yerr = sub[sd_col] if sd_col in sub.columns else 0
            ax.errorbar(sub["K"], sub[mean_col], yerr=yerr, color=color, linestyle=ls, marker=marker, capsize=3, label=label)
        ax.set_title(f"G={int(G)}")
        ax.set_xlabel(r"number of thresholds $K$")
        ax.grid(alpha=0.25, linewidth=0.7)
    axes[0, 0].set_ylabel(r"$\widehat V(q)$")
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(4, len(l)), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, outpath)


def draw_ablation_figures(outdir: Path, figdir: Path) -> None:
    abldir = ensure_dir(figdir / "ablations")
    plot_ablation_line(load_summary(outdir, "frontier_summary.csv", "frontier"), "tighten_factor", abldir / "frontier_V_vs_tighten", "tightening factor", "Target tightening")
    plot_ablation_line(load_summary(outdir, "ablation_group_scale_max_summary.csv", "group_scale_max"), "group_scale_max", abldir / "ablation_V_vs_group_scale_max", "maximum group scale", "Group-scale separation")
    plot_ablation_line(load_summary(outdir, "ablation_hetero_strength_summary.csv", "hetero_strength"), "hetero_strength", abldir / "ablation_V_vs_hetero_strength", "heteroskedasticity strength", "Heteroskedasticity")
    plot_ablation_line(load_summary(outdir, "ablation_df_summary.csv", "df"), "df", abldir / "ablation_V_vs_df", "Student-t degrees of freedom", "Tail heaviness")
    kg = load_summary(outdir, "ablation_KG_summary.csv", "kg")
    plot_kg_grid(kg, abldir / "ablation_V_vs_K_by_G")
    if len(kg) and "n_groups" in kg.columns:
        ng = pd.to_numeric(kg["n_groups"], errors="coerce")
        for Gval in sorted(ng.dropna().unique()):
            plot_ablation_line(kg[ng == Gval], "K", abldir / f"ablation_V_vs_K_G{int(Gval)}", r"number of thresholds $K$", f"Threshold grid, G={int(Gval)}")


def draw(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    metrics_path = resolve_main_metrics(outdir, args.metrics, args.run_label)

    if metrics_path is not None:
        if args.run_label and args.figdir is None:
            figdir = ensure_dir(metrics_path.parent / "figures_neurips")
            figure_path = f"{metrics_path.parent.name}/figures_neurips/paper_summary_2x2.pdf"
        else:
            figdir = ensure_dir(Path(args.figdir) if args.figdir else outdir / "figures_neurips")
            figure_path = args.figure_path_in_latex
        label = "Simulation" if not args.run_label else metrics_path.parent.name.replace("_", " ")
        draw_single_metrics(metrics_path, figdir, figure_path, label=label)
        if args.run_label:
            print(f"[done] wrote figures for {args.run_label} to {figdir}")
            print(f"[info] metrics used: {metrics_path}")
            return
    else:
        figdir = ensure_dir(Path(args.figdir) if args.figdir else outdir / "figures_neurips")
        if nonlinear_run_rows(outdir).empty:
            runs = available_metric_runs(outdir)
            msg = ["Could not find main simulation metrics.csv."]
            if runs:
                msg.append("Use --run_label with one of these completed runs:")
                msg.extend([f"  - {r}" for r in runs[:40]])
            else:
                msg.append("Use --metrics explicitly or check that aggregation wrote OUTDIR/<run>/metrics.csv.")
            raise FileNotFoundError("\n".join(msg))
        print("[info] no main metrics found; drawing nonlinear comparison and per-run nonlinear panels")
        draw_nonlinear_figures(outdir, figdir, also_per_run=True)
        print(f"[done] wrote nonlinear simulation figures to {figdir}")
        print(f"[done] key-threshold comparison: {figdir / 'nonlinear' / 'nonlinear_global_key_miscoverage_by_posterior.pdf'}")
        return

    draw_ablation_figures(outdir, figdir)
    if not args.no_nonlinear:
        draw_nonlinear_figures(outdir, figdir, also_per_run=args.draw_all_nonlinear_runs)
    print(f"[done] wrote simulation figures to {figdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw NeurIPS-quality figures for the synthetic simulation.")
    p.add_argument("--outdir", default="sim_neurips_array")
    p.add_argument("--run_label", "--run-label", default=None, help="Run/setting label to draw, e.g. nonlinear_signal_rff256")
    p.add_argument("--metrics", default=None, help="Explicit path to a metrics.csv file")
    p.add_argument("--figdir", default=None, help="Output figure directory")
    p.add_argument("--figure_path_in_latex", "--figure-path-in-latex", default="figures_neurips/paper_summary_2x2.pdf")
    p.add_argument("--no_nonlinear", "--no-nonlinear", action="store_true", help="Do not draw nonlinear comparison figures")
    p.add_argument("--draw_all_nonlinear_runs", "--draw-all-nonlinear-runs", action="store_true", help="Also draw 2x2 panels for each nonlinear run in nonlinear_summary.csv")
    p.add_argument("--list_runs", "--list-runs", action="store_true", help="List completed run labels under --outdir and exit")
    args = p.parse_args()
    if args.list_runs:
        outdir = Path(args.outdir)
        labels = available_metric_runs(outdir) + labels_from_summaries(outdir)
        seen = set()
        print("Available run labels/directories:")
        for lab in labels:
            if lab in seen:
                continue
            seen.add(lab)
            print(f"  {lab}")
        raise SystemExit(0)
    return args


if __name__ == "__main__":
    draw(parse_args())
