#!/usr/bin/env python3
"""SLURM job-array runner for the synthetic CIP simulation suite."""
from __future__ import annotations
import argparse, json, os, re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'serif'})

from CIP_sim_ablation_clean_split_v4_nonlinear_array import alpha_levels_for_K, run_simulation, summarize_df

def safe_dirname(x: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.=-]+', '_', str(x)).strip('_')


def json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj

@dataclass(frozen=True)
class SimSpec:
    label: str
    suite: str
    reps: int
    params: Dict[str, Any] = field(default_factory=dict)
    compute_conformal: bool = False
    save_tail_curves: bool = False

    @property
    def dirname(self) -> str:
        return safe_dirname(self.label)

def alpha(K: int):
    return alpha_levels_for_K(int(K))

def build_specs(args) -> List[SimSpec]:
    Rm, Ra, Rf, Rs, Rn = args.R_main, args.R_ablate, args.R_frontier, args.R_scaling, args.R_nonlinear
    mpg, mtg = args.m_per_group, args.m_thr_per_group
    suites = ['main'] if args.suite == 'main' else (['kg'] if args.suite == 'kg' else ['group_scale_max','df','hetero_strength'] if args.suite == 'severity' else ['nonlinear'] if args.suite == 'nonlinear' else [args.suite] if args.suite != 'all' else ['main','kg','group_scale_max','df','hetero_strength','frontier','mproj','scaling'])
    if args.include_nonlinear and 'nonlinear' not in suites:
        suites = list(suites) + ['nonlinear']
    specs: List[SimSpec] = []
    base = dict(n_fit=args.n_fit, r_cal=args.r_cal, S_pool=args.S_pool, n_particles=args.n_particles)
    def add(label, suite, reps, params, conformal=False, curves=False):
        p = dict(base); p.update(params)
        specs.append(SimSpec(label, suite, int(reps), p, bool(conformal), bool(curves)))
    if 'main' in suites:
        add('main', 'main', Rm, dict(n_groups=4, m_thr=mtg*4, m_proj=mpg*4, n_test=args.n_test, alpha_levels=alpha(5), tighten_factor=0.10), conformal=not args.no_conformal, curves=True)
    if 'kg' in suites:
        G_grid = [2,4] if args.fast else [2,4,6]
        K_grid = [2,5,8,10]
        for G in G_grid:
            for K in K_grid:
                add(f'ablation_KG_G{G}_K{K}', 'kg', Ra, dict(n_groups=G, m_thr=mtg*G, m_proj=mpg*G, n_test=args.n_test_ablate, alpha_levels=alpha(K), tighten_factor=0.10))
    fixed = dict(n_groups=4, m_thr=mtg*4, m_proj=mpg*4, n_test=args.n_test_ablate, alpha_levels=alpha(5), tighten_factor=0.10)
    if 'group_scale_max' in suites:
        for v in [1.5,2.0,3.0,4.0]:
            p=dict(fixed); p['group_scale_max']=v; add(f'ablation_group_scale_max_{v:g}', 'group_scale_max', Ra, p)
    if 'df' in suites:
        for v in [3.0,5.0,10.0]:
            p=dict(fixed); p['df']=v; add(f'ablation_df_{v:g}', 'df', Ra, p)
    if 'hetero_strength' in suites:
        for v in [0.0,0.15,0.30]:
            p=dict(fixed); p['hetero_strength']=v; add(f'ablation_hetero_strength_{v:g}', 'hetero_strength', Ra, p)
    if 'frontier' in suites:
        for tf in [0.0,0.05,0.10,0.20,0.30,0.40]:
            p=dict(fixed); p['tighten_factor']=tf; add(f'frontier_tighten_tf{tf:g}', 'frontier', Rf, p)
    if 'mproj' in suites:
        for tf in [0.0,0.10,0.20,0.30]:
            for mp in [100,200,400,800]:
                p=dict(fixed); p['tighten_factor']=tf; p['m_proj']=mp*4; add(f'ablation_mproj_tf{tf:g}_mper{mp}', 'mproj', Ra, p)
    if 'scaling' in suites:
        for S in ([10000,20000,40000] if args.fast else [10000,20000,40000,80000]):
            p=dict(fixed); p.update(n_test=args.n_test_scaling, S_pool=S); add(f'scaling_S_pool_{S}', 'scaling', Rs, p)
        for K in [2,5,8,10]:
            p=dict(fixed); p.update(n_test=args.n_test_scaling, alpha_levels=alpha(K)); add(f'scaling_K_{K}', 'scaling', Rs, p)
        for G in [2,4,6]:
            p=dict(fixed); p.update(n_groups=G, m_thr=mtg*G, m_proj=mpg*G, n_test=args.n_test_scaling); add(f'scaling_G_{G}', 'scaling', Rs, p)
    if 'nonlinear' in suites:
        # Nonlinear DGP with two posterior families.  The linear setting is a
        # misspecified baseline; the RFF setting is an approximate nonlinear
        # posterior with a Gaussian last-layer Gibbs posterior.
        p=dict(fixed)
        p.update(n_test=args.n_test, nonlinear_signal=float(args.nonlinear_signal), posterior_model='linear')
        add('nonlinear_signal_linear_posterior', 'nonlinear', Rn, p, conformal=not args.no_conformal, curves=True)
        p=dict(fixed)
        p.update(n_test=args.n_test, nonlinear_signal=float(args.nonlinear_signal), posterior_model='rff', rf_dim=int(args.rf_dim), rf_scale=float(args.rf_scale), rf_include_linear=True)
        add(f'nonlinear_signal_rff{int(args.rf_dim)}', 'nonlinear', Rn, p, conformal=not args.no_conformal, curves=True)
        if not args.fast and int(args.rf_dim) != 256:
            p=dict(fixed)
            p.update(n_test=args.n_test, nonlinear_signal=float(args.nonlinear_signal), posterior_model='rff', rf_dim=256, rf_scale=float(args.rf_scale), rf_include_linear=True)
            add('nonlinear_signal_rff256', 'nonlinear', Rn, p, conformal=not args.no_conformal, curves=True)
    return specs

def tasks(specs: List[SimSpec]) -> List[Tuple[int,int,SimSpec]]:
    out=[]
    for si,s in enumerate(specs):
        for r in range(s.reps): out.append((si,r,s))
    return out

def write_plan(outdir: Path, all_tasks):
    outdir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for tid,(si,r,s) in enumerate(all_tasks):
        row=dict(task_id=tid, spec_idx=si, rep=r, label=s.label, suite=s.suite)
        for k,v in s.params.items(): row['param_'+k] = str(v) if not np.isscalar(v) else v
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir/'array_plan.csv', index=False)

def run_task(args):
    specs=build_specs(args); all_tasks=tasks(specs)
    tid = int(args.task_id if args.task_id is not None else os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    if tid < 0 or tid >= len(all_tasks): raise SystemExit(f'task_id={tid} outside 0..{len(all_tasks)-1}')
    si, rep, spec = all_tasks[tid]
    task_out = Path(args.outdir)/spec.dirname/'reps'/f'rep_{rep:03d}'
    done = task_out/'DONE'
    metrics_path = task_out/'metrics.csv'
    seed = int(args.seed) + int(rep)*10007
    meta = json_sanitize(dict(task_id=tid, spec_idx=si, rep=rep, spec=asdict(spec), seed=seed))
    if done.exists() and not args.force:
        print(f'[skip] {done} exists'); return
    if metrics_path.exists() and not args.force:
        task_out.mkdir(parents=True, exist_ok=True)
        (task_out/'task.json').write_text(json.dumps(meta, indent=2))
        done.write_text('done\n')
        print(f'[repair] found {metrics_path}; wrote task.json and DONE')
        return
    task_out.mkdir(parents=True, exist_ok=True)
    print(f'[task {tid}] {spec.label} rep={rep} seed={seed}')
    df = run_simulation(outdir=str(task_out), seed=seed, R=1, make_figures=False, compute_conformal=spec.compute_conformal, verbose=args.verbose, save_tail_curves=spec.save_tail_curves, S_tune=args.S_tune, S_group_tune=args.S_group_tune, **spec.params)
    df=df.copy(); df['array_task_id']=tid; df['array_spec_idx']=si; df['array_rep']=rep; df['setting_label']=spec.label; df['setting_suite']=spec.suite
    df.to_csv(metrics_path, index=False)
    (task_out/'task.json').write_text(json.dumps(meta, indent=2))
    done.write_text('done\n')

def collect_spec(outdir: Path, si: int, spec: SimSpec, allow_missing: bool):
    frames=[]; missing=[]
    for rep in range(spec.reps):
        p = outdir/spec.dirname/'reps'/f'rep_{rep:03d}'/'metrics.csv'
        if not p.exists(): missing.append(rep); continue
        d=pd.read_csv(p); d['array_spec_idx']=si; d['array_rep']=rep; d['setting_label']=spec.label; d['setting_suite']=spec.suite; frames.append(d)
    if missing and not allow_missing: raise FileNotFoundError(f'Missing {len(missing)} reps for {spec.label}: {missing[:20]}')
    if not frames: return None, None, missing
    df=pd.concat(frames, ignore_index=True)
    sd=outdir/spec.dirname; sd.mkdir(parents=True, exist_ok=True)
    df.to_csv(sd/'metrics.csv', index=False)
    summ=summarize_df(df); summ.update(label=spec.label, suite=spec.suite, R_done=len(df), R_expected=spec.reps)
    for k,v in spec.params.items():
        if k=='alpha_levels': summ['K']=len(v)
        elif np.isscalar(v): summ[k]=v
    pd.DataFrame([summ]).to_csv(sd/'summary.csv', index=False)
    return df, summ, missing

def tail_idxs(df):
    return sorted(int(m.group(1)) for c in df.columns for m in [re.fullmatch(r'tail_t_(\d+)', str(c))] if m)

def mean_cols(df, cols): return np.array([pd.to_numeric(df[c], errors='coerce').mean() for c in cols])

def plot_main(outdir: Path, df: pd.DataFrame):
    figdir=outdir/'figures'; figdir.mkdir(parents=True, exist_ok=True)
    K=int(df['K'].iloc[0]); G=int(df['n_groups'].iloc[0]); key_k=int(round(df['key_k'].mean()))
    alpha_vals=np.array([df[f'alpha_{k}'].mean() for k in range(K)]); t_mean=np.array([df[f't_{k}'].mean() for k in range(K)])
    methods=[('q0','q0'),('temp','TempTune'),('group_temp','GroupTemp'),('cip_global','CIP-Global'),('cip_group','CIP-Group')]
    mcov=['mcov_q0','mcov_temp','mcov_group_temp','mcov_cip_global','mcov_cip_group']
    gm=['gm_q0','gm_temp','gm_group_temp','gm_cip_global','gm_cip_group']
    wv=['wv_group_q0','wv_group_temp','wv_group_group_temp','wv_group_cip_global','wv_group_cip_group']
    mse=['mse_q0','mse_temp','mse_group_temp','mse_cip_global','mse_cip_group']
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    ax=axes[0,0]; ti=tail_idxs(df)
    if ti:
        tg=mean_cols(df,[f'tail_t_{j}' for j in ti])
        for key,label in methods:
            cols=[f'tail_{key}_{j}' for j in ti]
            if all(c in df.columns for c in cols): ax.plot(tg, mean_cols(df, cols), label=label)
    else:
        for pref,(_,label) in zip(mcov,methods): ax.plot(t_mean,[df[f'{pref}_{k}'].mean() for k in range(K)], marker='o', label=label)
    ax.scatter(t_mean, alpha_vals, marker='x', label='Targets'); ax.set_xlabel(r'Threshold $t$'); ax.set_ylabel('Test tail rate'); ax.set_title('Global tail curve')
    ax=axes[0,1]; x=np.arange(len(methods)); width=0.8/max(G,1)
    for g in range(G): ax.bar(x+(g-(G-1)/2)*width, [df[f'{gp}_g{g}_key'].mean() for gp in gm], width, label=f'Group {g}')
    ax.axhline(alpha_vals[key_k], linestyle='--'); ax.set_xticks(x); ax.set_xticklabels([l for _,l in methods], rotation=18, ha='right'); ax.set_ylabel('Group tail rate'); ax.set_title('Key-threshold groups')
    ax=axes[1,0]; ax.bar(x,[df[c].mean() for c in wv], yerr=[df[c].std(ddof=1) for c in wv]); ax.set_xticks(x); ax.set_xticklabels([l for _,l in methods], rotation=18, ha='right'); ax.set_ylabel(r'$\widehat V(q)$'); ax.set_title('Worst-group violation')
    ax=axes[1,1]
    for label,vc,mc in zip([l for _,l in methods],wv,mse): ax.scatter(df[vc].mean(), df[mc].mean()); ax.annotate(label,(df[vc].mean(), df[mc].mean()), textcoords='offset points', xytext=(4,4), fontsize=8)
    ax.set_xlabel(r'$\widehat V(q)$'); ax.set_ylabel('Predictive MSE'); ax.set_title('Calibration--accuracy')
    handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(figdir/'paper_summary_2x2.pdf', bbox_inches='tight'); fig.savefig(figdir/'paper_summary_2x2.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    (figdir/'figure_snippet.tex').write_text('% Auto-generated by run_sim_neurips_array.py\n\\begin{figure}[t]\n  \\centering\n  \\includegraphics[width=\\textwidth]{figures/paper_summary_2x2.pdf}\n  \\caption{Synthetic regression results. GroupTemp denotes the group-specific temperature baseline.}\n  \\label{fig:sim_array_summary}\n\\end{figure}\n')

def plot_line(df, xcol, outfile):
    plt.figure(); df=df.sort_values(xcol)
    for mean,std,label in [('wv_group_temp_mean','wv_group_temp_std','TempTune'),('wv_group_group_temp_mean','wv_group_group_temp_std','GroupTemp'),('wv_group_cip_group_mean','wv_group_cip_group_std','CIP-Group')]:
        if mean in df: plt.errorbar(df[xcol], df[mean], yerr=df.get(std,0), marker='o', label=label)
    plt.xlabel(xcol); plt.ylabel(r'Worst-group violation $\widehat V(q)$'); plt.legend(); plt.tight_layout(); plt.savefig(outfile.with_suffix('.pdf'), bbox_inches='tight'); plt.savefig(outfile.with_suffix('.png'), dpi=220, bbox_inches='tight'); plt.close()

def collect(args):
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    specs=build_specs(args); write_plan(outdir, tasks(specs))
    all_s=[]; missing=[]; main_df=None
    for si,s in enumerate(specs):
        df,summ,miss=collect_spec(outdir,si,s,args.allow_missing)
        for r in miss: missing.append(dict(label=s.label,suite=s.suite,rep=r))
        if summ: all_s.append(summ)
        if s.suite=='main' and df is not None: main_df=df
    if missing: pd.DataFrame(missing).to_csv(outdir/'missing_tasks.csv', index=False)
    if not all_s: raise SystemExit('No completed tasks found')
    sdf=pd.DataFrame(all_s); sdf.to_csv(outdir/'sim_summary_all.csv', index=False)
    def save(suite, name):
        sub=sdf[sdf['suite']==suite].copy()
        if len(sub): sub.to_csv(outdir/name, index=False)
    save('main','main_summary.csv'); save('kg','ablation_KG_summary.csv'); save('group_scale_max','ablation_group_scale_max_summary.csv'); save('df','ablation_df_summary.csv'); save('hetero_strength','ablation_hetero_strength_summary.csv'); save('frontier','frontier_summary.csv'); save('mproj','ablation_mproj_summary.csv'); save('scaling','scaling_summary.csv'); save('nonlinear','nonlinear_summary.csv')
    if args.make_figures:
        if main_df is not None: plot_main(outdir, main_df)
        figdir=outdir/'figures'; figdir.mkdir(exist_ok=True)
        for suite,xcol,fname in [('group_scale_max','group_scale_max','ablation_V_vs_group_scale_max'),('df','df','ablation_V_vs_df'),('hetero_strength','hetero_strength','ablation_V_vs_hetero_strength'),('frontier','tighten_factor','frontier_V_vs_tighten')]:
            sub=sdf[sdf['suite']==suite].copy()
            if len(sub) and xcol in sub: plot_line(sub,xcol,figdir/fname)
        sub=sdf[sdf['suite']=='kg'].copy()
        if len(sub) and 'K' in sub and 'n_groups' in sub:
            for G in sorted(sub['n_groups'].dropna().unique()): plot_line(sub[sub['n_groups']==G], 'K', figdir/f'ablation_V_vs_K_G{int(G)}')
    print(f'[collect] wrote results under {outdir}')

def parse_args():
    p=argparse.ArgumentParser(description='Synthetic CIP simulation job-array driver')
    p.add_argument('--mode', choices=['count','plan','task','collect'], required=True)
    p.add_argument('--task_id', type=int, default=None); p.add_argument('--outdir', default='sim_neurips_array'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--suite', choices=['main','kg','severity','frontier','mproj','scaling','nonlinear','all'], default='all'); p.add_argument('--include_nonlinear', action='store_true'); p.add_argument('--fast', action='store_true')
    p.add_argument('--R_main', type=int, default=20); p.add_argument('--R_ablate', type=int, default=8); p.add_argument('--R_frontier', type=int, default=8); p.add_argument('--R_scaling', type=int, default=5); p.add_argument('--R_nonlinear', type=int, default=20)
    p.add_argument('--n_fit', type=int, default=400); p.add_argument('--r_cal', type=int, default=400); p.add_argument('--m_per_group', type=int, default=200); p.add_argument('--m_thr_per_group', type=int, default=200)
    p.add_argument('--n_test', type=int, default=4000); p.add_argument('--n_test_ablate', type=int, default=2000); p.add_argument('--n_test_scaling', type=int, default=1000)
    p.add_argument('--S_pool', type=int, default=40000); p.add_argument('--n_particles', type=int, default=4000); p.add_argument('--S_tune', type=int, default=2000); p.add_argument('--S_group_tune', type=int, default=1000); p.add_argument('--nonlinear_signal', type=float, default=1.0); p.add_argument('--rf_dim', type=int, default=128); p.add_argument('--rf_scale', type=float, default=1.0)
    p.add_argument('--no_conformal', action='store_true'); p.add_argument('--allow_missing', action='store_true'); p.add_argument('--no_figures', dest='make_figures', action='store_false'); p.add_argument('--force', action='store_true'); p.add_argument('--verbose', action='store_true'); p.set_defaults(make_figures=True)
    return p.parse_args()

def main():
    args=parse_args(); specs=build_specs(args); all_tasks=tasks(specs)
    if args.mode=='count': print(len(all_tasks))
    elif args.mode=='plan':
        write_plan(Path(args.outdir), all_tasks); print(f'Total tasks: {len(all_tasks)}'); print(f'Array range: 0-{len(all_tasks)-1}')
        for i,(si,r,s) in enumerate(all_tasks): print(f'{i:05d}: rep={r:03d} spec_idx={si:03d} suite={s.suite} label={s.label} model={s.params.get("posterior_model", "linear")}')
    elif args.mode=='task': run_task(args)
    elif args.mode=='collect': collect(args)
if __name__=='__main__': main()
