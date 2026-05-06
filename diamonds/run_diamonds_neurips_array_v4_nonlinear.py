#!/usr/bin/env python3
"""SLURM job-array runner for the Diamonds CIP experiments and ablations."""
from __future__ import annotations
import argparse, json, os, re, time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'serif'})

from real_data_cip_suite_v4_nonlinear_array import DatasetConfig, ensure_dir, load_diamonds, run_real_data_dataset, safe_dirname, set_mpl_style

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
    return os.path.join(outdir, '_cache')

def diamonds_config() -> DatasetConfig:
    return DatasetConfig(name='diamonds', loader=load_diamonds, target_col='price', group_col='cut', group_top_k=None, y_transform='log1p', frac_fit=0.40, frac_thr=0.15, frac_proj=0.15, frac_cal=0.15, frac_test=0.15)

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]

def build_specs(K: int, full_ablation: bool, k_ablation_values: Sequence[int], include_nonlinear: bool = False, rf_dim: int = 256, rf_scale: float = 1.0) -> List[RunSpec]:
    specs=[RunSpec('diamonds_blind', int(K), False), RunSpec('diamonds_with_cut_drop1_cutprior1', int(K), True, True, 1.0)]
    if full_ablation:
        specs.insert(1, RunSpec('diamonds_with_cut_full_onehot', int(K), True, False, None))
        specs.insert(2, RunSpec('diamonds_with_cut_drop1', int(K), True, True, None))
    if include_nonlinear:
        specs.append(RunSpec(f'diamonds_blind_rff{int(rf_dim)}', int(K), False, False, None, 'rff', int(rf_dim), float(rf_scale), True))
        specs.append(RunSpec(f'diamonds_with_cut_rff{int(rf_dim)}', int(K), True, True, None, 'rff', int(rf_dim), float(rf_scale), True))
    for kk in k_ablation_values:
        kk=int(kk)
        if kk != int(K): specs.append(RunSpec(f'diamonds_blind_K{kk}', kk, False))
    return specs

def total_tasks(R: int, specs: Sequence[RunSpec]) -> int: return int(R)*len(specs)

def task_to_rep_spec(task_id: int, R: int, specs: Sequence[RunSpec]) -> Tuple[int, RunSpec, int]:
    n=total_tasks(R,specs)
    if task_id<0 or task_id>=n: raise ValueError(f'task_id={task_id} outside 0..{n-1}')
    rep=task_id//len(specs); si=task_id%len(specs); return rep, specs[si], si

def part_dir(outdir: str, spec: RunSpec, rep: int) -> str:
    return os.path.join(outdir, '_array_parts', safe_dirname(spec.label), f'rep_{int(rep):04d}')

def prepare_cache_with_lock(outdir: str, timeout_sec: int=1800) -> None:
    cache=shared_cache_dir(outdir); ensure_dir(cache); lock_path=os.path.join(cache,'download.lock')
    try:
        import fcntl
        with open(lock_path,'w',encoding='utf-8') as f:
            start=time.time()
            while True:
                try: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.time()-start > timeout_sec: raise TimeoutError(f'Timed out waiting for {lock_path}')
                    time.sleep(2.0)
            load_diamonds(cache)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:
        load_diamonds(cache)

def numeric_summary(df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    row: Dict[str, object] = {'dataset':'diamonds','run_name':spec.label,'include_group_in_X':float(spec.include_group_in_X),'drop_first_categorical':float(spec.drop_first_categorical),'tau2_group_feature':float(spec.tau2_group_feature) if spec.tau2_group_feature is not None else np.nan,'posterior_model':spec.posterior_model,'posterior_model_code':float({'linear':0,'none':0,'rff':1,'relu':2,'tanh':3}.get(str(spec.posterior_model).lower(),-1)),'rf_dim':float(spec.rf_dim),'rf_scale':float(spec.rf_scale),'rf_include_linear':float(spec.rf_include_linear),'R':float(len(df)),'K':float(spec.K)}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            vals=pd.to_numeric(df[c], errors='coerce')
            if vals.notna().any():
                row[c+'_mean']=float(vals.mean())
                row[c+'_std']=float(vals.std(ddof=1)) if vals.notna().sum()>1 else 0.0
    return pd.DataFrame([row])

def read_group_labels(pdir: str) -> Optional[List[str]]:
    path=os.path.join(pdir,'group_levels.csv')
    if not os.path.exists(path): return None
    df=pd.read_csv(path)
    if 'group_label' not in df.columns: return None
    return [str(x) for x in df.sort_values('group_code')['group_label'].tolist()]

def tail_indices(df):
    return sorted(int(m.group(1)) for c in df.columns for m in [re.fullmatch(r'tail_t_(\d+)', str(c))] if m)

def mean_cols(df, cols): return np.array([pd.to_numeric(df[c], errors='coerce').mean() for c in cols])

def make_figures(df: pd.DataFrame, run_dir: str, spec: RunSpec, group_labels: Optional[List[str]]) -> None:
    set_mpl_style(); plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'serif'})
    figdir=os.path.join(run_dir,'figures'); ensure_dir(figdir)
    K=int(round(float(df['K'].iloc[0]))) if 'K' in df.columns else int(spec.K)
    alpha=np.array([df[f'alpha_{k}'].mean() for k in range(K)])
    t_mean=np.array([df[f't_{k}'].mean() for k in range(K)])
    key_k=int(round(float(df['key_k'].mean()))) if 'key_k' in df.columns else int(np.argmin(np.abs(alpha-0.10)))
    G=int(round(float(df['n_groups'].iloc[0]))) if 'n_groups' in df.columns else 0
    methods=['q0','TempTune','GroupTemp','CIP-Global','CIP-Group']
    tail_keys=['q0','temp','group_temp','cip_global','cip_group']
    mcov=['mcov_q0','mcov_temp','mcov_group_temp','mcov_cip_global','mcov_cip_group']
    gm=['gm_q0','gm_temp','gm_group_temp','gm_cip_global','gm_cip_group']
    wv=['wv_group_q0','wv_group_temp','wv_group_group_temp','wv_group_cip_global','wv_group_cip_group']
    mse=['mse_q0','mse_temp','mse_group_temp','mse_cip_global','mse_cip_group']
    def save(name):
        plt.tight_layout(); plt.savefig(os.path.join(figdir,name+'.pdf'), bbox_inches='tight'); plt.savefig(os.path.join(figdir,name+'.png'), dpi=220, bbox_inches='tight'); plt.close()
    ti=tail_indices(df)
    plt.figure()
    if ti:
        tg=mean_cols(df,[f'tail_t_{j}' for j in ti])
        for key,lab in zip(tail_keys,methods):
            cols=[f'tail_{key}_{j}' for j in ti]
            if all(c in df.columns for c in cols): plt.plot(tg, mean_cols(df,cols), label=lab)
    else:
        for pref,lab in zip(mcov,methods): plt.plot(t_mean,[df[f'{pref}_{k}'].mean() for k in range(K)], marker='o', label=lab)
    plt.scatter(t_mean,alpha,marker='x',label='Targets'); plt.xlabel(r'Threshold $t$'); plt.ylabel('Test tail rate'); plt.title('Tail curve'); plt.legend(); save('tail_curve_global')
    x=np.arange(len(methods))
    plt.figure(); plt.bar(x,[df[f'{p}_{key_k}'].mean() for p in mcov], yerr=[df[f'{p}_{key_k}'].std(ddof=1) for p in mcov]); plt.axhline(alpha[key_k], linestyle='--'); plt.xticks(x,methods,rotation=20,ha='right'); plt.ylabel('Miscoverage at key threshold'); plt.title('Global key-threshold miscoverage'); save('miscoverage_key_global')
    plt.figure(); width=0.8/max(G,1)
    for g in range(G):
        lab=group_labels[g] if group_labels and g < len(group_labels) else f'Group {g}'
        plt.bar(x+(g-(G-1)/2)*width,[df[f'{gp}_g{g}_key'].mean() for gp in gm],width,label=lab)
    plt.axhline(alpha[key_k], linestyle='--'); plt.xticks(x,methods,rotation=20,ha='right'); plt.ylabel('Group tail rate at key threshold'); plt.title('Groupwise miscoverage at key threshold'); plt.legend(ncol=2); save('miscoverage_key_by_group')
    plt.figure(); plt.bar(x,[df[c].mean() for c in wv], yerr=[df[c].std(ddof=1) for c in wv]); plt.xticks(x,methods,rotation=20,ha='right'); plt.ylabel(r'Worst-group violation $\widehat V(q)$'); plt.title('Worst-group multi-threshold violation'); save('worst_group_violation')
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    ax=axs[0,0]
    if ti:
        tg=mean_cols(df,[f'tail_t_{j}' for j in ti])
        for key,lab in zip(tail_keys,methods):
            cols=[f'tail_{key}_{j}' for j in ti]
            if all(c in df.columns for c in cols): ax.plot(tg, mean_cols(df,cols), label=lab)
    else:
        for pref,lab in zip(mcov,methods): ax.plot(t_mean,[df[f'{pref}_{k}'].mean() for k in range(K)], marker='o', label=lab)
    ax.scatter(t_mean,alpha,marker='x',label='Targets'); ax.set_xlabel(r'Threshold $t$'); ax.set_ylabel('Test tail rate'); ax.text(0.02,0.96,'(a)',transform=ax.transAxes,va='top')
    ax=axs[0,1]
    for g in range(G):
        lab=group_labels[g] if group_labels and g < len(group_labels) else f'Group {g}'
        ax.bar(x+(g-(G-1)/2)*width,[df[f'{gp}_g{g}_key'].mean() for gp in gm],width,label=lab)
    ax.axhline(alpha[key_k], linestyle='--'); ax.set_xticks(x); ax.set_xticklabels(methods,rotation=20,ha='right'); ax.set_ylabel('Group tail rate'); ax.text(0.02,0.96,'(b)',transform=ax.transAxes,va='top')
    ax=axs[1,0]; ax.bar(x,[df[c].mean() for c in wv], yerr=[df[c].std(ddof=1) for c in wv]); ax.set_xticks(x); ax.set_xticklabels(methods,rotation=20,ha='right'); ax.set_ylabel(r'$\widehat V(q)$'); ax.text(0.02,0.96,'(c)',transform=ax.transAxes,va='top')
    ax=axs[1,1]
    for lab,vc,mc in zip(methods,wv,mse): ax.scatter(df[vc].mean(),df[mc].mean()); ax.annotate(lab,(df[vc].mean(),df[mc].mean()),textcoords='offset points',xytext=(4,4),fontsize=8)
    ax.set_xlabel(r'$\widehat V(q)$'); ax.set_ylabel('Predictive MSE'); ax.text(0.02,0.96,'(d)',transform=ax.transAxes,va='top')
    handles,labels=axs[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False); fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(os.path.join(figdir,'paper_summary_2x2.pdf'),bbox_inches='tight'); fig.savefig(os.path.join(figdir,'paper_summary_2x2.png'),dpi=220,bbox_inches='tight'); plt.close(fig)
    caption_label=spec.label.replace('_', r'\_')
    (open(os.path.join(figdir,'figure_snippet.tex'),'w',encoding='utf-8')).write('% Auto-generated by run_diamonds_neurips_array.py\n\\begin{figure}[t]\n  \\centering\n  \\includegraphics[width=\\textwidth]{'+safe_dirname(spec.label)+'/figures/paper_summary_2x2.pdf}\n  \\caption{Diamonds results for '+caption_label+'. Group labels are cut labels. GroupTemp denotes the group-specific temperature baseline.}\n  \\label{fig:'+safe_dirname(spec.label)+'_diamonds}\n\\end{figure}\n')

def run_one_task(args):
    specs=build_specs(args.K, args.full_ablation, parse_int_list(args.k_ablation_values) if args.k_ablation else [], args.include_nonlinear, args.rf_dim, args.rf_scale)
    tid=args.task_id if args.task_id is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    rep,spec,si=task_to_rep_spec(int(tid), int(args.R), specs)
    pdir=part_dir(args.outdir,spec,rep); mpath=os.path.join(pdir,'metrics.csv')
    if os.path.exists(mpath) and not args.force: print(f'[skip] {mpath}'); return
    ensure_dir(pdir); prepare_cache_with_lock(args.outdir)
    seed=int(args.seed)+int(rep)*10007
    print(f'[task] id={tid} rep={rep} spec={spec.label} seed={seed}')
    df=run_real_data_dataset(cfg=diamonds_config(), outdir=pdir, seed=seed, R=1, K=int(spec.K), S_pool=int(args.S_pool), n_particles=int(args.n_particles), S_tune=int(args.S_tune), S_group_tune=int(args.S_group_tune), group_eta_passes=int(args.group_eta_passes), make_figures=False, compute_conformal=not args.no_conformal, include_group_in_X=spec.include_group_in_X, drop_first_categorical=spec.drop_first_categorical, tau2_group_feature=spec.tau2_group_feature, posterior_model=spec.posterior_model, rf_dim=spec.rf_dim, rf_scale=spec.rf_scale, rf_include_linear=spec.rf_include_linear, run_label=spec.label, save_tail_curves=True, cache_dir_override=shared_cache_dir(args.outdir), verbose=True)
    df=df.copy(); df['rep']=float(rep); df['array_task_id']=float(tid); df['array_spec_idx']=float(si); df['array_run_label']=spec.label; df.to_csv(mpath,index=False)
    with open(os.path.join(pdir,'task.json'),'w',encoding='utf-8') as f: json.dump({'task_id':int(tid),'rep':int(rep),'spec_idx':int(si),'spec':asdict(spec),'seed':seed},f,indent=2)

def collect_results(args):
    specs=build_specs(args.K, args.full_ablation, parse_int_list(args.k_ablation_values) if args.k_ablation else [], args.include_nonlinear, args.rf_dim, args.rf_scale)
    ensure_dir(args.outdir); all_s=[]; manifest=[]
    for spec in specs:
        frames=[]; first=None; missing=[]
        for rep in range(int(args.R)):
            pdir=part_dir(args.outdir,spec,rep); mpath=os.path.join(pdir,'metrics.csv')
            if not os.path.exists(mpath): missing.append(rep); continue
            if first is None: first=pdir
            d=pd.read_csv(mpath); d['rep']=rep; d['array_run_label']=spec.label; frames.append(d); manifest.append({'run_name':spec.label,'rep':rep,'metrics_path':mpath})
        if missing and not args.allow_missing: raise FileNotFoundError(f'Missing {len(missing)} reps for {spec.label}: {missing[:20]}')
        if not frames: print(f'[collect] no reps for {spec.label}'); continue
        run_dir=os.path.join(args.outdir,safe_dirname(spec.label)); ensure_dir(run_dir)
        df=pd.concat(frames,ignore_index=True); df.to_csv(os.path.join(run_dir,'metrics.csv'),index=False)
        labels=read_group_labels(first) if first else None
        if labels is not None: pd.DataFrame({'group_code':np.arange(len(labels)),'group_label':labels}).to_csv(os.path.join(run_dir,'group_levels.csv'),index=False)
        summ=numeric_summary(df,spec); summ.to_csv(os.path.join(run_dir,'summary.csv'),index=False); all_s.append(summ)
        if args.make_figures: make_figures(df,run_dir,spec,labels)
        print(f'[collect] {spec.label}: {len(df)} reps -> {run_dir}')
    if all_s:
        summary=pd.concat(all_s,ignore_index=True); summary.to_csv(os.path.join(args.outdir,'diamonds_summary_all.csv'),index=False)
        key_cols=['run_name','include_group_in_X','drop_first_categorical','tau2_group_feature','mcov_q0_key_mean','mcov_temp_key_mean','mcov_group_temp_key_mean','mcov_cip_global_key_mean','mcov_cip_group_key_mean','wv_group_q0_mean','wv_group_temp_mean','wv_group_group_temp_mean','wv_group_cip_global_mean','wv_group_cip_group_mean','mse_q0_mean','mse_temp_mean','mse_group_temp_mean','mse_cip_global_mean','mse_cip_group_mean']
        key_cols=[c for c in key_cols if c in summary.columns]; summary[key_cols].to_csv(os.path.join(args.outdir,'diamonds_cut_ablation_summary.csv'),index=False)
    if manifest: pd.DataFrame(manifest).to_csv(os.path.join(args.outdir,'array_manifest.csv'),index=False)

def print_plan(args):
    specs=build_specs(args.K, args.full_ablation, parse_int_list(args.k_ablation_values) if args.k_ablation else [], args.include_nonlinear, args.rf_dim, args.rf_scale)
    n=total_tasks(args.R,specs); print(f'Total tasks: {n}'); print(f'Array range: 0-{n-1}')
    for tid in range(n):
        rep,spec,si=task_to_rep_spec(tid,args.R,specs); print(f'{tid:04d}: rep={rep:03d} spec_idx={si} label={spec.label} K={spec.K} model={spec.posterior_model}')

def build_arg_parser():
    p=argparse.ArgumentParser(description='Diamonds CIP experiment with SLURM job-array support.')
    p.add_argument('--mode', choices=['count','plan','prepare','task','collect'], default='task'); p.add_argument('--outdir', default='diamonds_neurips_array'); p.add_argument('--task_id', type=int, default=None); p.add_argument('--seed', type=int, default=0); p.add_argument('--R', type=int, default=20); p.add_argument('--K', type=int, default=5)
    p.add_argument('--full_ablation', action='store_true'); p.add_argument('--k_ablation', action='store_true'); p.add_argument('--k_ablation_values', default='3,8')
    p.add_argument('--S_pool', type=int, default=40000); p.add_argument('--n_particles', type=int, default=4000); p.add_argument('--S_tune', type=int, default=2000); p.add_argument('--S_group_tune', type=int, default=1000); p.add_argument('--group_eta_passes', type=int, default=2); p.add_argument('--include_nonlinear', action='store_true', help='add random-feature posterior settings'); p.add_argument('--rf_dim', type=int, default=256); p.add_argument('--rf_scale', type=float, default=1.0)
    p.add_argument('--no_conformal', action='store_true'); p.add_argument('--allow_missing', action='store_true'); p.add_argument('--no_figures', dest='make_figures', action='store_false'); p.add_argument('--force', action='store_true'); p.set_defaults(make_figures=True); return p

def main():
    args=build_arg_parser().parse_args(); specs=build_specs(args.K,args.full_ablation,parse_int_list(args.k_ablation_values) if args.k_ablation else [], args.include_nonlinear, args.rf_dim, args.rf_scale)
    if args.mode=='count': print(total_tasks(args.R,specs))
    elif args.mode=='plan': print_plan(args)
    elif args.mode=='prepare': prepare_cache_with_lock(args.outdir); print(f'[prepare done] cache: {shared_cache_dir(args.outdir)}')
    elif args.mode=='task': run_one_task(args)
    elif args.mode=='collect': collect_results(args)
if __name__=='__main__': main()
