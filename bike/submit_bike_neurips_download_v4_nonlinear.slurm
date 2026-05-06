#!/bin/bash
#SBATCH --job-name=cip_bike_prep_nl
#SBATCH --output=logs/bike_prep_nl_%j.out
#SBATCH --error=logs/bike_prep_nl_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
mkdir -p logs
OUTDIR=${OUTDIR:-bike_neurips_array_nonlinear}
python run_bike_neurips_experiment_array_v4_nonlinear.py --mode prepare --outdir "${OUTDIR}"
