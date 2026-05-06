#!/bin/bash
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
MAX_CONCURRENT=${MAX_CONCURRENT:-500}
S_POOL=${S_POOL:-40000}
N_PARTICLES=${N_PARTICLES:-4000}
S_TUNE=${S_TUNE:-2000}
S_GROUP_TUNE=${S_GROUP_TUNE:-1000}
FULL_FLAG=""; if [[ "$FULL_ABLATION" == "1" ]]; then FULL_FLAG="--full_ablation"; fi
K_FLAG=""; if [[ "$K_ABLATION" == "1" ]]; then K_FLAG="--k_ablation"; fi
NL_FLAG=""; if [[ "$INCLUDE_NONLINEAR" == "1" ]]; then NL_FLAG="--include_nonlinear"; fi
NTASKS=$(python run_diamonds_neurips_array_v4_nonlinear.py --mode count --outdir "$OUTDIR" --R "$R" --K "$K" $FULL_FLAG $K_FLAG $NL_FLAG --rf_dim "$RF_DIM" --rf_scale "$RF_SCALE")
LAST=$((NTASKS-1))
echo "Submitting Diamonds nonlinear pipeline: ${NTASKS} tasks, range 0-${LAST}, max concurrent ${MAX_CONCURRENT}"
PREP_JOB=$(sbatch --parsable --export=ALL,OUTDIR="$OUTDIR" submit_diamonds_neurips_download_v4_nonlinear.slurm)
echo "Prepare job: $PREP_JOB"
ARRAY_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} --array=0-${LAST}%${MAX_CONCURRENT} \
  --export=ALL,OUTDIR="$OUTDIR",R="$R",K="$K",FULL_ABLATION="$FULL_ABLATION",K_ABLATION="$K_ABLATION",INCLUDE_NONLINEAR="$INCLUDE_NONLINEAR",RF_DIM="$RF_DIM",RF_SCALE="$RF_SCALE",S_POOL="$S_POOL",N_PARTICLES="$N_PARTICLES",S_TUNE="$S_TUNE",S_GROUP_TUNE="$S_GROUP_TUNE" \
  submit_diamonds_neurips_array_v4_nonlinear.slurm)
echo "Array job: $ARRAY_JOB"
AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} \
  --export=ALL,OUTDIR="$OUTDIR",R="$R",K="$K",FULL_ABLATION="$FULL_ABLATION",K_ABLATION="$K_ABLATION",INCLUDE_NONLINEAR="$INCLUDE_NONLINEAR",RF_DIM="$RF_DIM",RF_SCALE="$RF_SCALE" \
  submit_diamonds_neurips_aggregate_v4_nonlinear.slurm)
echo "Aggregation job: $AGG_JOB"
