#!/bin/bash
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
MAX_CONCURRENT=${MAX_CONCURRENT:-20}
NO_K_ABLATION=${NO_K_ABLATION:-1}
INCLUDE_NONLINEAR=${INCLUDE_NONLINEAR:-1}
RF_DIM=${RF_DIM:-256}
RF_SCALE=${RF_SCALE:-1.0}
COMMON_ARGS=(--outdir "${OUTDIR}" --R "${R}" --K "${K}" --S_pool "${S_POOL}" --n_particles "${N_PARTICLES}" --S_tune "${S_TUNE}" --S_group_tune "${S_GROUP_TUNE}" --group_eta_passes "${GROUP_ETA_PASSES}" --rf_dim "${RF_DIM}" --rf_scale "${RF_SCALE}")
if [[ "${NO_K_ABLATION}" == "1" ]]; then COMMON_ARGS+=(--no_k_ablation); fi
if [[ "${INCLUDE_NONLINEAR}" == "1" ]]; then COMMON_ARGS+=(--include_nonlinear); fi
N=$(python run_bike_neurips_experiment_array_v4_nonlinear.py --mode count "${COMMON_ARGS[@]}")
LAST=$((N - 1))
ARRAY_RANGE="0-${LAST}%${MAX_CONCURRENT}"
echo "Bike nonlinear array plan: N=${N}, array=${ARRAY_RANGE}, outdir=${OUTDIR}"
python run_bike_neurips_experiment_array_v4_nonlinear.py --mode plan "${COMMON_ARGS[@]}" | head -60
EXPORTS="ALL,OUTDIR=${OUTDIR},R=${R},K=${K},S_POOL=${S_POOL},N_PARTICLES=${N_PARTICLES},S_TUNE=${S_TUNE},S_GROUP_TUNE=${S_GROUP_TUNE},GROUP_ETA_PASSES=${GROUP_ETA_PASSES},NO_K_ABLATION=${NO_K_ABLATION},INCLUDE_NONLINEAR=${INCLUDE_NONLINEAR},RF_DIM=${RF_DIM},RF_SCALE=${RF_SCALE}"
prepare_jid=$(sbatch --parsable --export="${EXPORTS}" submit_bike_neurips_download_v4_nonlinear.slurm)
echo "submitted prepare job: ${prepare_jid}"
array_jid=$(sbatch --parsable --dependency=afterok:${prepare_jid} --export="${EXPORTS}" --array=${ARRAY_RANGE} submit_bike_neurips_array_v4_nonlinear.slurm)
echo "submitted array job: ${array_jid}"
collect_jid=$(sbatch --parsable --dependency=afterok:${array_jid} --export="${EXPORTS}" submit_bike_neurips_aggregate_v4_nonlinear.slurm)
echo "submitted collect job: ${collect_jid}"
