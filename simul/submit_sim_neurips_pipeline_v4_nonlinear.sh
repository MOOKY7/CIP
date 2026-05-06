#!/bin/bash
set -euo pipefail
mkdir -p logs

OUTDIR=${OUTDIR:-sim_neurips_array_nonlinear}
SUITE=${SUITE:-nonlinear}
INCLUDE_NONLINEAR=${INCLUDE_NONLINEAR:-0}
FAST=${FAST:-0}
R_MAIN=${R_MAIN:-20}
R_ABLATE=${R_ABLATE:-8}
R_FRONTIER=${R_FRONTIER:-8}
R_SCALING=${R_SCALING:-5}
R_NONLINEAR=${R_NONLINEAR:-20}
MAX_CONCURRENT=${MAX_CONCURRENT:-50}
S_POOL=${S_POOL:-40000}
N_PARTICLES=${N_PARTICLES:-4000}
S_TUNE=${S_TUNE:-2000}
S_GROUP_TUNE=${S_GROUP_TUNE:-1000}
NONLINEAR_SIGNAL=${NONLINEAR_SIGNAL:-1.0}
RF_DIM=${RF_DIM:-128}
RF_SCALE=${RF_SCALE:-1.0}

FAST_FLAG=""; if [[ "$FAST" == "1" ]]; then FAST_FLAG="--fast"; fi
NL_FLAG=""; if [[ "$INCLUDE_NONLINEAR" == "1" ]]; then NL_FLAG="--include_nonlinear"; fi

NTASKS=$(python run_sim_neurips_array_v4_nonlinear.py --mode count --outdir "$OUTDIR" --suite "$SUITE" $FAST_FLAG $NL_FLAG --R_main "$R_MAIN" --R_ablate "$R_ABLATE" --R_frontier "$R_FRONTIER" --R_scaling "$R_SCALING" --R_nonlinear "$R_NONLINEAR" --S_pool "$S_POOL" --n_particles "$N_PARTICLES" --S_tune "$S_TUNE" --S_group_tune "$S_GROUP_TUNE" --nonlinear_signal "$NONLINEAR_SIGNAL" --rf_dim "$RF_DIM" --rf_scale "$RF_SCALE")
LAST=$((NTASKS-1))
echo "Submitting nonlinear simulation array: ${NTASKS} tasks, range 0-${LAST}, max concurrent ${MAX_CONCURRENT}"

ARRAY_JOB=$(sbatch --parsable --array=0-${LAST}%${MAX_CONCURRENT} \
  --export=ALL,OUTDIR="$OUTDIR",SUITE="$SUITE",INCLUDE_NONLINEAR="$INCLUDE_NONLINEAR",FAST="$FAST",R_MAIN="$R_MAIN",R_ABLATE="$R_ABLATE",R_FRONTIER="$R_FRONTIER",R_SCALING="$R_SCALING",R_NONLINEAR="$R_NONLINEAR",S_POOL="$S_POOL",N_PARTICLES="$N_PARTICLES",S_TUNE="$S_TUNE",S_GROUP_TUNE="$S_GROUP_TUNE",NONLINEAR_SIGNAL="$NONLINEAR_SIGNAL",RF_DIM="$RF_DIM",RF_SCALE="$RF_SCALE" \
  submit_sim_neurips_array_v4_nonlinear.slurm)

echo "Simulation array job: $ARRAY_JOB"
AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} \
  --export=ALL,OUTDIR="$OUTDIR",SUITE="$SUITE",INCLUDE_NONLINEAR="$INCLUDE_NONLINEAR",FAST="$FAST",R_MAIN="$R_MAIN",R_ABLATE="$R_ABLATE",R_FRONTIER="$R_FRONTIER",R_SCALING="$R_SCALING",R_NONLINEAR="$R_NONLINEAR",S_POOL="$S_POOL",N_PARTICLES="$N_PARTICLES",S_TUNE="$S_TUNE",S_GROUP_TUNE="$S_GROUP_TUNE",NONLINEAR_SIGNAL="$NONLINEAR_SIGNAL",RF_DIM="$RF_DIM",RF_SCALE="$RF_SCALE" \
  submit_sim_neurips_aggregate_v4_nonlinear.slurm)

echo "Aggregation job: $AGG_JOB"
