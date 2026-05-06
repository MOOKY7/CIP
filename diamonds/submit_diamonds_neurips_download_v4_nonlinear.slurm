#!/bin/bash
#SBATCH --job-name=cip_diam_prep_nl
#SBATCH -p epycQ
#SBATCH --output=logs/cip_diam_prep_nl_%j.out
#SBATCH --error=logs/cip_diam_prep_nl_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -euo pipefail
mkdir -p logs
OUTDIR=${OUTDIR:-diamonds_neurips_array_nonlinear}
python run_diamonds_neurips_array_v4_nonlinear.py --mode prepare --outdir "$OUTDIR"
