#!/usr/bin/env python3
"""Draw final figures for the Bike Sharing experiment.

Run after Bike aggregation, for example:

    python draw_bike_figures_neurips.py --outdir bike_neurips_array

The default run is the group-aware season-included experiment:
    bike_with_group_drop1_groupPrior1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cip_neurips_plot_utils import (
    ensure_dir,
    read_metrics,
    read_group_labels,
    infer_n_groups,
    set_paper_style,
    save_fig,
    make_paper_summary_2x2,
    write_latex_snippet,
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
    make_nonlinear_comparison_figure,
    make_tail_curve_grid_from_run_dirs,
    nonlinear_subset,
    model_display_label,
    summary_label_col,
)


def resolve_run_dir(outdir: str, run_label: str) -> Path:
    out = Path(outdir)
    for p in [out / run_label, out, Path(run_label), Path(".")]:
        if (p / "metrics.csv").exists():
            return p
    return out / run_label



def draw_nonlinear_comparison(outdir: Path, base_figdir: Path) -> None:
    """Optional comparison between the linear and RFF posterior families."""
    summary_path = outdir / "bike_summary_all.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    sub = nonlinear_subset(summary, dataset="bike")
    if sub.empty or not sub.astype(str).apply(lambda x: x.str.contains("rff", case=False, regex=False)).any().any():
        return
    ndir = ensure_dir(base_figdir / "nonlinear_comparison")
    set_paper_style()
    fig = make_nonlinear_comparison_figure(sub, dataset="bike", include_key=True)
    if fig is not None:
        save_fig(fig, ndir / "bike_linear_vs_rff_summary")
    label_col = summary_label_col(sub)
    run_dirs = [outdir / str(x) for x in sub[label_col].tolist()]
    titles = [model_display_label(row, dataset="bike") for _, row in sub.iterrows()]
    fig = make_tail_curve_grid_from_run_dirs(run_dirs, titles=titles)
    if fig is not None:
        save_fig(fig, ndir / "bike_linear_vs_rff_tail_curves")
    write_latex_snippet(
        "figures_neurips/nonlinear_comparison/bike_linear_vs_rff_summary.pdf",
        "fig:bike_nonlinear_comparison",
        "Bike Sharing linear-vs-RFF posterior comparison. RFF denotes a random-feature last-layer Gibbs posterior; point ranges show mean $\\pm$ one standard deviation over random splits.",
        ndir / "figure_snippet.tex",
    )

def draw(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir) if args.run_dir else resolve_run_dir(args.outdir, args.run_label)
    metrics_path = Path(args.metrics) if args.metrics else run_dir / "metrics.csv"
    group_path = Path(args.group_levels) if args.group_levels else run_dir / "group_levels.csv"
    figdir = ensure_dir(Path(args.figdir) if args.figdir else run_dir / "figures_neurips")

    metrics = read_metrics(metrics_path)
    group_labels = read_group_labels(group_path, infer_n_groups(metrics))

    set_paper_style()
    fig = make_paper_summary_2x2(metrics, group_labels=group_labels, title_prefix="Bike Sharing")
    save_fig(fig, figdir / "paper_summary_2x2")

    alpha, _ = alpha_t_from_metrics(metrics)
    key_alpha = float(alpha[infer_key_k(metrics, alpha)])
    tail_df, target_df = tail_curve_tidy(metrics)

    panel_specs = [
        ("global_tail_curve", lambda ax: plot_global_tail_curve(ax, tail_df, target_df, title="Bike Sharing: global tail curve")),
        ("global_key_miscoverage", lambda ax: plot_method_point_range(ax, method_metric_tidy(metrics, "key"), ylabel="Miscoverage at key threshold", title="Bike Sharing: global key-threshold miscoverage", target=key_alpha)),
        ("groupwise_key_miscoverage", lambda ax: plot_group_key_profile(ax, group_key_tidy(metrics, group_labels), target=key_alpha, title="Bike Sharing: season groups at key threshold")),
        ("worst_group_violation", lambda ax: plot_method_point_range(ax, method_metric_tidy(metrics, "wv"), ylabel=r"Worst-group violation $\widehat V(q)$", title="Bike Sharing: worst-group violation")),
        ("calibration_accuracy_frontier", lambda ax: plot_calibration_accuracy(ax, frontier_tidy(metrics), title="Bike Sharing: calibration--accuracy")),
    ]
    for name, fn in panel_specs:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fn(ax)
        fig.tight_layout()
        save_fig(fig, figdir / name)

    write_latex_snippet(
        args.figure_path_in_latex,
        "fig:bike_results",
        "Bike Sharing results for the group-aware season-included design. Group labels are seasons. Point ranges show mean $\\pm$ one standard deviation over random splits; GroupTemp denotes the group-specific temperature baseline.",
        figdir / "figure_snippet.tex",
    )
    if not args.no_nonlinear:
        draw_nonlinear_comparison(Path(args.outdir), ensure_dir(Path(args.outdir) / "figures_neurips"))

    print(f"[done] wrote Bike Sharing figures to {figdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw NeurIPS-quality figures for Bike Sharing aggregation.")
    p.add_argument("--outdir", default="bike_neurips_array")
    p.add_argument("--run_label", default="bike_with_group_drop1_groupPrior1")
    p.add_argument("--run_dir", default=None, help="Explicit run directory containing metrics.csv")
    p.add_argument("--metrics", default=None, help="Explicit metrics.csv path")
    p.add_argument("--group_levels", default=None, help="Explicit group_levels.csv path")
    p.add_argument("--figdir", default=None, help="Output figure directory")
    p.add_argument("--no_nonlinear", "--no-nonlinear", action="store_true", help="Do not draw linear-vs-RFF comparison figures")
    p.add_argument("--figure_path_in_latex", default="figures/bike_paper_summary_2x2.pdf")
    return p.parse_args()


if __name__ == "__main__":
    draw(parse_args())
