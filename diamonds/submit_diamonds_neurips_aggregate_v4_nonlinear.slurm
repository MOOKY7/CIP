#!/bin/bash
#SBATCH --job-name=cip_diam_agg_nl
#SBATCH -p epycQ
#SBATCH --output=logs/cip_diam_agg_nl_%j.out
#SBATCH --error=logs/cip_diam_agg_nl_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
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
ALLOW_MISSING=${ALLOW_MISSING:-0}
FULL_FLAG=""; if [[ "$FULL_ABLATION" == "1" ]]; then FULL_FLAG="--full_ablation"; fi
K_FLAG=""; if [[ "$K_ABLATION" == "1" ]]; then K_FLAG="--k_ablation"; fi
NL_FLAG=""; if [[ "$INCLUDE_NONLINEAR" == "1" ]]; then NL_FLAG="--include_nonlinear"; fi
MISS_FLAG=""; if [[ "$ALLOW_MISSING" == "1" ]]; then MISS_FLAG="--allow_missing"; fi
python run_diamonds_neurips_array_v4_nonlinear.py --mode collect --outdir "$OUTDIR" --R "$R" --K "$K" $FULL_FLAG $K_FLAG $NL_FLAG $MISS_FLAG --rf_dim "$RF_DIM" --rf_scale "$RF_SCALE"
