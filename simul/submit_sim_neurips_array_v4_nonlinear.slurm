#!/bin/bash
#SBATCH --job-name=cip_sim_arr_nl
#SBATCH -p epycQ
#SBATCH --output=logs/cip_sim_arr_nl_%A_%a.out
#SBATCH --error=logs/cip_sim_arr_nl_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
mkdir -p logs

OUTDIR=${OUTDIR:-sim_neurips_array_nonlinear}
SUITE=${SUITE:-all}
INCLUDE_NONLINEAR=${INCLUDE_NONLINEAR:-0}
FAST=${FAST:-0}
R_MAIN=${R_MAIN:-20}
R_ABLATE=${R_ABLATE:-8}
R_FRONTIER=${R_FRONTIER:-8}
R_SCALING=${R_SCALING:-5}
R_NONLINEAR=${R_NONLINEAR:-20}
S_POOL=${S_POOL:-40000}
N_PARTICLES=${N_PARTICLES:-4000}
S_TUNE=${S_TUNE:-2000}
S_GROUP_TUNE=${S_GROUP_TUNE:-1000}
NONLINEAR_SIGNAL=${NONLINEAR_SIGNAL:-1.0}
RF_DIM=${RF_DIM:-128}
RF_SCALE=${RF_SCALE:-1.0}

FAST_FLAG=""; if [[ "$FAST" == "1" ]]; then FAST_FLAG="--fast"; fi
NL_FLAG=""; if [[ "$INCLUDE_NONLINEAR" == "1" ]]; then NL_FLAG="--include_nonlinear"; fi

python run_sim_neurips_array_v4_nonlinear.py \
  --mode task \
  --outdir "$OUTDIR" \
  --suite "$SUITE" \
  $FAST_FLAG $NL_FLAG \
  --R_main "$R_MAIN" \
  --R_ablate "$R_ABLATE" \
  --R_frontier "$R_FRONTIER" \
  --R_scaling "$R_SCALING" \
  --R_nonlinear "$R_NONLINEAR" \
  --S_pool "$S_POOL" \
  --n_particles "$N_PARTICLES" \
  --S_tune "$S_TUNE" \
  --S_group_tune "$S_GROUP_TUNE" \
  --nonlinear_signal "$NONLINEAR_SIGNAL" \
  --rf_dim "$RF_DIM" \
  --rf_scale "$RF_SCALE"
