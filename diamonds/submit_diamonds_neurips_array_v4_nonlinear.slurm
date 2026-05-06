#!/bin/bash
#SBATCH --job-name=cip_diam_arr_nl
#SBATCH -p epycQ
#SBATCH --output=logs/cip_diam_arr_nl_%A_%a.out
#SBATCH --error=logs/cip_diam_arr_nl_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail
mkdir -p logs
OUTDIR=${OUTDIR:-diamonds_neurips_array_nonlinear}
R=${R:-20}
K=${K:-5}
FULL_ABLATION=${FULL_ABLATION:-0}
K_ABLATION=${K_ABLATION:-0}
INCLUDE_NONLINEAR=${INCLUDE_NONLINEAR:-1}
RF_DIM=${RF_DIM:-256}
RF_SCALE=${RF_SCALE:-1.0}
S_POOL=${S_POOL:-40000}
N_PARTICLES=${N_PARTICLES:-4000}
S_TUNE=${S_TUNE:-2000}
S_GROUP_TUNE=${S_GROUP_TUNE:-1000}
FULL_FLAG=""; if [[ "$FULL_ABLATION" == "1" ]]; then FULL_FLAG="--full_ablation"; fi
K_FLAG=""; if [[ "$K_ABLATION" == "1" ]]; then K_FLAG="--k_ablation"; fi
NL_FLAG=""; if [[ "$INCLUDE_NONLINEAR" == "1" ]]; then NL_FLAG="--include_nonlinear"; fi
python run_diamonds_neurips_array_v4_nonlinear.py --mode task --outdir "$OUTDIR" --R "$R" --K "$K" $FULL_FLAG $K_FLAG $NL_FLAG --rf_dim "$RF_DIM" --rf_scale "$RF_SCALE" --S_pool "$S_POOL" --n_particles "$N_PARTICLES" --S_tune "$S_TUNE" --S_group_tune "$S_GROUP_TUNE"
