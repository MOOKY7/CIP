#!/bin/bash
#SBATCH --job-name=cip_bike_arr_nl
#SBATCH -p epycQ
#SBATCH --output=logs/bike_arr_nl_%A_%a.out
#SBATCH --error=logs/bike_arr_nl_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
mkdir -p logs
OUTDIR=${OUTDIR:-bike_neurips_array_nonlinear}
R=${R:-20}
K=${K:-5}
S_POOL=${S_POOL:-40000}
N_PARTICLES=${N_PARTICLES:-4000}
S_TUNE=${S_TUNE:-2000}
S_GROUP_TUNE=${S_GROUP_TUNE:-1000}
GROUP_ETA_PASSES=${GROUP_ETA_PASSES:-2}
NO_K_ABLATION=${NO_K_ABLATION:-1}
INCLUDE_NONLINEAR=${INCLUDE_NONLINEAR:-1}
RF_DIM=${RF_DIM:-256}
RF_SCALE=${RF_SCALE:-1.0}
ARGS=(--mode task --outdir "${OUTDIR}" --R "${R}" --K "${K}" --S_pool "${S_POOL}" --n_particles "${N_PARTICLES}" --S_tune "${S_TUNE}" --S_group_tune "${S_GROUP_TUNE}" --group_eta_passes "${GROUP_ETA_PASSES}" --rf_dim "${RF_DIM}" --rf_scale "${RF_SCALE}")
if [[ "${NO_K_ABLATION}" == "1" ]]; then ARGS+=(--no_k_ablation); fi
if [[ "${INCLUDE_NONLINEAR}" == "1" ]]; then ARGS+=(--include_nonlinear); fi
python run_bike_neurips_experiment_array_v4_nonlinear.py "${ARGS[@]}"
