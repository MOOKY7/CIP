#!/usr/bin/env python3
"""
SLURM job-array runner for the UCI Bike Sharing CIP experiment.

Default paper run:
  python run_bike_neurips_experiment_array.py --mode count --R 20 --K 5
  python run_bike_neurips_experiment_array.py --mode plan  --R 20 --K 5

With the default settings this produces 80 tasks:
  20 reps x {bike_blind, bike_with_group_drop1_groupPrior1, bike_blind_K3, bike_blind_K8}.

After the array tasks finish, run --mode collect to concatenate metrics, write
summary.csv files, and regenerate paper-ready figures with season labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_data_cip_suite_v4_nonlinear_array import (
    DatasetConfig,
    ensure_dir,
    load_bike_sharing_hourly,
    run_real_data_dataset,
    safe_dirname,
    set_mpl_style,
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    K: int
    include_group_in_X: bool
    drop_first_categorical: bool = False
    tau2_group_feature: Optional[float] = None
    posterior_model: str = "linear"
    rf_dim: int = 256
    rf_scale: float = 1.0
    rf_include_linear: bool = True


def shared_cache_dir(outdir: str) -> str:
    return os.path.join(outdir, "_cache")


def bike_config() -> DatasetConfig:
    return DatasetConfig(
        name="bike",
        loader=load_bike_sharing_hourly,
        target_col="cnt",
        group_col="season_name",
        group_top_k=None,
        y_transform="log1p",
        frac_fit=0.40,
        frac_thr=0.15,
        frac_proj=0.15,
        frac_cal=0.15,
        frac_test=0.15,
    )


def parse_int_list(s: str) -> List[int]:
    vals: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def build_specs(K: int, no_k_ablation: bool, k_ablation_values: Sequence[int], include_nonlinear: bool = False, rf_dim: int = 256, rf_scale: float = 1.0) -> List[RunSpec]:
    specs: List[RunSpec] = [
        RunSpec("bike_blind", int(K), include_group_in_X=False),
        RunSpec(
            "bike_with_group_drop1_groupPrior1",
            int(K),
            include_group_in_X=True,
            drop_first_categorical=True,
            tau2_group_feature=1.0,
        ),
    ]
    if include_nonlinear:
        # Main nonlinear approximate-posterior settings: random Fourier features
        # with a Gaussian last-layer Gibbs posterior.
        specs.append(RunSpec(
            f"bike_blind_rff{int(rf_dim)}",
            int(K),
            include_group_in_X=False,
            posterior_model="rff",
            rf_dim=int(rf_dim),
            rf_scale=float(rf_scale),
            rf_include_linear=True,
        ))
        specs.append(RunSpec(
            f"bike_with_group_rff{int(rf_dim)}",
            int(K),
            include_group_in_X=True,
            drop_first_categorical=True,
            tau2_group_feature=None,
            posterior_model="rff",
            rf_dim=int(rf_dim),
            rf_scale=float(rf_scale),
            rf_include_linear=True,
        ))
    if not no_k_ablation:
        for kk in k_ablation_values:
            kk = int(kk)
            if kk == int(K):
                continue
            specs.append(RunSpec(f"bike_blind_K{kk}", kk, include_group_in_X=False))
    return specs


def total_tasks(R: int, specs: Sequence[RunSpec]) -> int:
    return int(R) * len(specs)


def task_to_rep_spec(task_id: int, R: int, specs: Sequence[RunSpec]) -> Tuple[int, RunSpec, int]:
    n = total_tasks(R, specs)
    if task_id < 0 or task_id >= n:
        raise ValueError(f"task_id={task_id} is outside 0..{n-1}")
    rep = int(task_id) // len(specs)
    spec_idx = int(task_id) % len(specs)
    return rep, specs[spec_idx], spec_idx


def part_dir(outdir: str, spec: RunSpec, rep: int) -> str:
    return os.path.join(outdir, "_array_parts", safe_dirname(spec.label), f"rep_{int(rep):04d}")


def prepare_cache_with_lock(outdir: str, timeout_sec: int = 1800) -> None:
    """Download/cache Bike data once, using a simple fcntl lock on Linux clusters."""
    cache_dir = shared_cache_dir(outdir)
    ensure_dir(cache_dir)
    lock_path = os.path.join(cache_dir, "download.lock")
    try:
        import fcntl  # type: ignore
        with open(lock_path, "w", encoding="utf-8") as f:
            start = time.time()
            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start > timeout_sec:
                        raise TimeoutError(f"Timed out waiting for {lock_path}")
                    time.sleep(2.0)
            load_bike_sharing_hourly(cache_dir)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:
        load_bike_sharing_hourly(cache_dir)


def write_table_tex(summary: pd.DataFrame, outpath: str) -> None:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"{r['run_name']} & "
            f"{r.get('mcov_temp_key_mean', float('nan')):.3f} & "
            f"{r.get('mcov_group_temp_key_mean', float('nan')):.3f} & "
            f"{r.get('mcov_cip_group_key_mean', float('nan')):.3f} & "
            f"{r.get('wv_group_temp_mean', float('nan')):.3f} & "
            f"{r.get('wv_group_group_temp_mean', float('nan')):.3f} & "
            f"{r.get('wv_group_cip_group_mean', float('nan')):.3f} & "
            f"{r.get('mse_temp_mean', float('nan')):.3f} & "
            f"{r.get('mse_group_temp_mean', float('nan')):.3f} & "
            f"{r.get('mse_cip_group_mean', float('nan')):.3f} \\\\"  # noqa: W605
        )
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def numeric_summary(df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    row: Dict[str, float | str] = {
        "dataset": "bike",
        "run_name": spec.label,
        "include_group_in_X": float(spec.include_group_in_X),
        "drop_first_categorical": float(spec.drop_first_categorical),
        "tau2_group_feature": float(spec.tau2_group_feature) if spec.tau2_group_feature is not None else np.nan,
        "posterior_model": spec.posterior_model,
        "posterior_model_code": float({"linear": 0, "none": 0, "rff": 1, "relu": 2, "tanh": 3}.get(str(spec.posterior_model).lower(), -1)),
        "rf_dim": float(spec.rf_dim),
        "rf_scale": float(spec.rf_scale),
        "rf_include_linear": float(spec.rf_include_linear),
        "R": float(len(df)),
        "K": float(spec.K),
        "n_groups": float(df["n_groups"].iloc[0]) if "n_groups" in df.columns and len(df) else np.nan,
    }
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                row[c + "_mean"] = float(vals.mean())
                row[c + "_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
    return pd.DataFrame([row])


def read_group_labels(part_path: str) -> Optional[List[str]]:
    path = os.path.join(part_path, "group_levels.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "group_label" not in df.columns:
        return None
    return [str(x) for x in df.sort_values("group_code")["group_label"].tolist()]


def _tail_indices(df: pd.DataFrame) -> List[int]:
    idxs: List[int] = []
    for c in df.columns:
        m = re.fullmatch(r"tail_t_(\d+)", str(c))
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def _mean_cols(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return np.array([pd.to_numeric(df[c], errors="coerce").mean() for c in cols], dtype=float)


def make_figures_from_metrics(df: pd.DataFrame, run_dir: str, spec: RunSpec, group_labels: Optional[List[str]]) -> None:
    set_mpl_style()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif"})
    figdir = os.path.join(run_dir, "figures")
    ensure_dir(figdir)

    K = int(round(float(df["K"].iloc[0]))) if "K" in df.columns else int(spec.K)
    alpha = np.array([df[f"alpha_{k}"].mean() for k in range(K)], dtype=float)
    t_mean = np.array([df[f"t_{k}"].mean() for k in range(K)], dtype=float)
    key_k = int(round(float(df["key_k"].mean()))) if "key_k" in df.columns else int(np.argmin(np.abs(alpha - 0.10)))
    G = int(round(float(df["n_groups"].iloc[0]))) if "n_groups" in df.columns else 0

    methods = ["q0", "TempTune", "GroupTemp", "CIP-Global", "CIP-Group"]
    tail_keys = ["q0", "temp", "group_temp", "cip_global", "cip_group"]
    mcov_pref = ["mcov_q0", "mcov_temp", "mcov_group_temp", "mcov_cip_global", "mcov_cip_group"]
    gm_pref = ["gm_q0", "gm_temp", "gm_group_temp", "gm_cip_global", "gm_cip_group"]
    wv_cols = ["wv_group_q0", "wv_group_temp", "wv_group_group_temp", "wv_group_cip_global", "wv_group_cip_group"]
    mse_cols = ["mse_q0", "mse_temp", "mse_group_temp", "mse_cip_global", "mse_cip_group"]

    def savefig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, name + ".pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(figdir, name + ".png"), dpi=200, bbox_inches="tight")
        plt.close()

    tail_idxs = _tail_indices(df)
    plt.figure()
    if tail_idxs:
        t_grid = _mean_cols(df, [f"tail_t_{j}" for j in tail_idxs])
        for key, lab in zip(tail_keys, methods):
            cols = [f"tail_{key}_{j}" for j in tail_idxs]
            if all(c in df.columns for c in cols):
                plt.plot(t_grid, _mean_cols(df, cols), label=lab)
    else:
        for pref, lab in zip(mcov_pref, methods):
            plt.plot(t_mean, [df[f"{pref}_{k}"].mean() for k in range(K)], marker="o", label=lab)
    plt.scatter(t_mean, alpha, marker="x", label="Targets")
    plt.xlabel(r"Threshold $t$")
    plt.ylabel("Test tail rate")
    plt.title("Tail curve")
    plt.legend()
    savefig("tail_curve_global")

    plt.figure()
    x = np.arange(len(methods))
    vals = [df[f"{p}_{key_k}"].mean() for p in mcov_pref]
    errs = [df[f"{p}_{key_k}"].std(ddof=1) for p in mcov_pref]
    plt.bar(x, vals, yerr=errs)
    plt.axhline(alpha[key_k], linestyle="--")
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Miscoverage at key threshold")
    plt.title("Global key-threshold miscoverage")
    savefig("miscoverage_key_global")

    plt.figure()
    width = 0.80 / max(G, 1)
    for g in range(G):
        vals = [df[f"{gp}_g{g}_key"].mean() for gp in gm_pref]
        glab = group_labels[g] if group_labels and g < len(group_labels) else f"Group {g}"
        plt.bar(x + (g - (G - 1) / 2) * width, vals, width, label=glab)
    plt.axhline(alpha[key_k], linestyle="--")
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Group tail rate at key threshold")
    plt.title("Groupwise miscoverage at key threshold")
    plt.legend(ncol=2)
    savefig("miscoverage_key_by_group")

    plt.figure()
    for pref, lab in zip(mcov_pref, methods):
        plt.plot(t_mean, [df[f"{pref}_{k}"].mean() for k in range(K)], marker="o", label=lab)
    plt.plot(t_mean, alpha, linestyle="--", marker="x", label="Target")
    plt.xlabel(r"Threshold $t_k$ (mean over splits)")
    plt.ylabel(r"Test miscoverage at $t_k$")
    plt.title("Constraint satisfaction across tail points")
    plt.legend()
    savefig("miscoverage_all_constraints")

    plt.figure()
    v_mean = [df[c].mean() for c in wv_cols]
    v_std = [df[c].std(ddof=1) for c in wv_cols]
    plt.bar(x, v_mean, yerr=v_std)
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel(r"Worst-group violation $\widehat V(q)$")
    plt.title("Worst-group multi-threshold violation")
    savefig("worst_group_violation")

    plt.figure()
    for lab, vx, my in zip(methods, wv_cols, mse_cols):
        plt.scatter(df[vx].mean(), df[my].mean(), label=lab)
        plt.annotate(lab, (df[vx].mean(), df[my].mean()), textcoords="offset points", xytext=(4, 4))
    plt.xlabel(r"Worst-group violation $\widehat V(q)$")
    plt.ylabel("Predictive test MSE")
    plt.title("Calibration--accuracy summary")
    savefig("tradeoff_V_vs_MSE")

    cp_items = [
        ("cp_len_ridge", "SC-Ridge"),
        ("cp_len_huber", "SC-Huber"),
        ("cp_len_q0mean", "SC-q0Mean"),
        ("cp_len_group_temp_mean", "SC-GroupTempMean"),
        ("cp_len_cip_global_mean", "SC-CIPGlobMean"),
        ("cp_len_cip_group_mean", "SC-CIPGrpMean"),
        ("mondrian_len_ridge", "Mondrian-Ridge"),
    ]
    available = [(c, lab) for c, lab in cp_items if c in df.columns and df[c].notna().any()]
    if available:
        plt.figure(figsize=(7, 4))
        xx = np.arange(len(available))
        plt.bar(xx, [df[c].mean() for c, _ in available], yerr=[df[c].std(ddof=1) for c, _ in available])
        plt.xticks(xx, [lab for _, lab in available], rotation=25, ha="right")
        plt.ylabel("Average interval length")
        plt.title("Conformal interval lengths")
        savefig("conformal_lengths")

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    ax = axs[0, 0]
    if tail_idxs:
        t_grid = _mean_cols(df, [f"tail_t_{j}" for j in tail_idxs])
        for key, lab in zip(tail_keys, methods):
            cols = [f"tail_{key}_{j}" for j in tail_idxs]
            if all(c in df.columns for c in cols):
                ax.plot(t_grid, _mean_cols(df, cols), label=lab)
    else:
        for pref, lab in zip(mcov_pref, methods):
            ax.plot(t_mean, [df[f"{pref}_{k}"].mean() for k in range(K)], marker="o", label=lab)
    ax.scatter(t_mean, alpha, marker="x", label="Targets")
    ax.set_xlabel(r"Threshold $t$")
    ax.set_ylabel("Test tail rate")
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top")

    ax = axs[0, 1]
    for g in range(G):
        vals = [df[f"{gp}_g{g}_key"].mean() for gp in gm_pref]
        glab = group_labels[g] if group_labels and g < len(group_labels) else f"Group {g}"
        ax.bar(x + (g - (G - 1) / 2) * width, vals, width, label=glab)
    ax.axhline(alpha[key_k], linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Group tail rate")
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, va="top")

    ax = axs[1, 0]
    ax.bar(x, v_mean, yerr=v_std)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(r"$\widehat V(q)$")
    ax.text(0.02, 0.96, "(c)", transform=ax.transAxes, va="top")

    ax = axs[1, 1]
    for lab, vx, my in zip(methods, wv_cols, mse_cols):
        ax.scatter(df[vx].mean(), df[my].mean())
        ax.annotate(lab, (df[vx].mean(), df[my].mean()), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(r"$\widehat V(q)$")
    ax.set_ylabel("Predictive MSE")
    ax.text(0.02, 0.96, "(d)", transform=ax.transAxes, va="top")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(figdir, "paper_summary_2x2.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(figdir, "paper_summary_2x2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Build the LaTeX snippet without putting backslashes inside f-string expressions.
    fig_relpath = f"{safe_dirname(spec.label)}/figures/paper_summary_2x2.pdf"
    tex_label = f"fig:{safe_dirname(spec.label)}_bike"
    caption_label = spec.label.replace("_", r"\_")
    snippet = (
        "% Auto-generated by run_bike_neurips_experiment_array_v4_nonlinear.py\n"
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{fig_relpath}}}\n"
        f"  \\caption{{Bike Sharing results for {caption_label}. "
        "Group labels are season labels used for stratified splitting, constraints, "
        "and evaluation. GroupTemp denotes the group-specific temperature baseline.}}\n"
        f"  \\label{{{tex_label}}}\n"
        "\\end{figure}\n"
    )
    with open(os.path.join(figdir, "figure_snippet.tex"), "w", encoding="utf-8") as f:
        f.write(snippet)


def run_one_task(args: argparse.Namespace) -> None:
    specs = build_specs(args.K, args.no_k_ablation, parse_int_list(args.k_ablation_values), args.include_nonlinear, args.rf_dim, args.rf_scale)
    task_id = args.task_id
    if task_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is None:
            raise ValueError("Provide --task_id or run inside a SLURM array.")
        task_id = int(env_id)

    rep, spec, spec_idx = task_to_rep_spec(int(task_id), int(args.R), specs)
    pdir = part_dir(args.outdir, spec, rep)
    mpath = os.path.join(pdir, "metrics.csv")
    if os.path.exists(mpath) and not args.force:
        print(f"[skip] Existing metrics: {mpath}")
        return

    ensure_dir(pdir)
    prepare_cache_with_lock(args.outdir)
    job_seed = int(args.seed) + int(rep) * 10007
    print(f"[task] id={task_id} rep={rep} spec={spec.label} K={spec.K} seed={job_seed}")

    df = run_real_data_dataset(
        cfg=bike_config(),
        outdir=pdir,
        seed=job_seed,
        R=1,
        K=int(spec.K),
        S_pool=int(args.S_pool),
        n_particles=int(args.n_particles),
        S_tune=int(args.S_tune),
        S_group_tune=int(args.S_group_tune),
        group_eta_passes=int(args.group_eta_passes),
        make_figures=False,
        compute_conformal=not args.no_conformal,
        include_group_in_X=bool(spec.include_group_in_X),
        drop_first_categorical=bool(spec.drop_first_categorical),
        tau2_group_feature=spec.tau2_group_feature,
        posterior_model=spec.posterior_model,
        rf_dim=spec.rf_dim,
        rf_scale=spec.rf_scale,
        rf_include_linear=spec.rf_include_linear,
        run_label=spec.label,
        save_tail_curves=True,
        cache_dir_override=shared_cache_dir(args.outdir),
        verbose=True,
    )
    df = df.copy()
    df["rep"] = float(rep)
    df["array_task_id"] = float(task_id)
    df["array_spec_idx"] = float(spec_idx)
    df["array_run_label"] = spec.label
    df.to_csv(mpath, index=False)
    with open(os.path.join(pdir, "task.json"), "w", encoding="utf-8") as f:
        json.dump({"task_id": int(task_id), "rep": int(rep), "spec_idx": int(spec_idx), "spec": asdict(spec), "seed": job_seed}, f, indent=2)
    print(f"[task done] {mpath}")


def collect_results(args: argparse.Namespace) -> None:
    specs = build_specs(args.K, args.no_k_ablation, parse_int_list(args.k_ablation_values), args.include_nonlinear, args.rf_dim, args.rf_scale)
    ensure_dir(args.outdir)
    all_summaries: List[pd.DataFrame] = []
    manifest: List[Dict[str, object]] = []

    for spec in specs:
        frames: List[pd.DataFrame] = []
        first_part: Optional[str] = None
        missing: List[int] = []
        for rep in range(int(args.R)):
            pdir = part_dir(args.outdir, spec, rep)
            mpath = os.path.join(pdir, "metrics.csv")
            if not os.path.exists(mpath):
                missing.append(rep)
                continue
            if first_part is None:
                first_part = pdir
            df_rep = pd.read_csv(mpath)
            df_rep["rep"] = rep
            df_rep["array_run_label"] = spec.label
            frames.append(df_rep)
            manifest.append({"run_name": spec.label, "rep": rep, "metrics_path": mpath})

        if missing and not args.allow_missing:
            raise FileNotFoundError(f"Missing {len(missing)} replicate files for {spec.label}: {missing[:20]}")
        if not frames:
            print(f"[collect] no completed tasks for {spec.label}; skipping")
            continue

        run_dir = os.path.join(args.outdir, safe_dirname(spec.label))
        ensure_dir(run_dir)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

        labels = read_group_labels(first_part) if first_part is not None else None
        if labels is not None:
            pd.DataFrame({"group_code": np.arange(len(labels)), "group_label": labels}).to_csv(os.path.join(run_dir, "group_levels.csv"), index=False)

        summary = numeric_summary(df, spec)
        summary.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
        all_summaries.append(summary)
        if args.make_figures:
            make_figures_from_metrics(df, run_dir, spec, labels)
        print(f"[collect] {spec.label}: {len(df)} reps -> {run_dir}")

    if all_summaries:
        summary_all = pd.concat(all_summaries, ignore_index=True)
        summary_all.to_csv(os.path.join(args.outdir, "bike_summary_all.csv"), index=False)
        write_table_tex(summary_all, os.path.join(args.outdir, "bike_table_rows.tex"))
        print(f"[collect done] {os.path.join(args.outdir, 'bike_summary_all.csv')}")
        print(f"[collect done] {os.path.join(args.outdir, 'bike_table_rows.tex')}")
    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(args.outdir, "array_manifest.csv"), index=False)


def print_plan(args: argparse.Namespace) -> None:
    specs = build_specs(args.K, args.no_k_ablation, parse_int_list(args.k_ablation_values), args.include_nonlinear, args.rf_dim, args.rf_scale)
    n = total_tasks(args.R, specs)
    print(f"Total tasks: {n}")
    print(f"Array range: 0-{n-1}")
    for task_id in range(n):
        rep, spec, spec_idx = task_to_rep_spec(task_id, args.R, specs)
        print(f"{task_id:04d}: rep={rep:03d} spec_idx={spec_idx} label={spec.label} K={spec.K} model={spec.posterior_model}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bike Sharing CIP experiment with SLURM job-array support.")
    p.add_argument("--mode", choices=["count", "plan", "task", "collect", "prepare"], default="task")
    p.add_argument("--outdir", type=str, default="bike_neurips_array")
    p.add_argument("--task_id", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--R", type=int, default=20)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--k_ablation_values", type=str, default="3,5,8")
    p.add_argument("--S_pool", type=int, default=40000)
    p.add_argument("--n_particles", type=int, default=4000)
    p.add_argument("--S_tune", type=int, default=2000)
    p.add_argument("--S_group_tune", type=int, default=1000)
    p.add_argument("--group_eta_passes", type=int, default=2)
    p.add_argument("--include_nonlinear", action="store_true", help="add random-feature posterior settings")
    p.add_argument("--rf_dim", type=int, default=256)
    p.add_argument("--rf_scale", type=float, default=1.0)
    p.add_argument("--no_k_ablation", action="store_true")
    p.add_argument("--no_conformal", action="store_true")
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--no_figures", dest="make_figures", action="store_false")
    p.add_argument("--force", action="store_true")
    p.set_defaults(make_figures=True)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    specs = build_specs(args.K, args.no_k_ablation, parse_int_list(args.k_ablation_values), args.include_nonlinear, args.rf_dim, args.rf_scale)
    if args.mode == "count":
        print(total_tasks(args.R, specs))
    elif args.mode == "plan":
        print_plan(args)
    elif args.mode == "prepare":
        prepare_cache_with_lock(args.outdir)
        print(f"[prepare done] cache: {shared_cache_dir(args.outdir)}")
    elif args.mode == "task":
        run_one_task(args)
    elif args.mode == "collect":
        collect_results(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
