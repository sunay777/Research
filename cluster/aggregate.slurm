#!/bin/bash
# aggregate.slurm — run AFTER the array finishes to collect every
# results/*/metrics.json into results/summary.csv + print the per-cell table.
#SBATCH --job-name=exo-agg
#SBATCH --partition=REPLACE_ME        # <-- any short CPU partition (`sinfo`)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge || true
module load python/3.11 || module load anaconda3 || true
source .venv/bin/activate
python -m exo_portfolio.run_grid \
  --config configs/base.yaml configs/universe_djia30.yaml \
  --grid configs/experiment_grid.yaml \
  --aggregate
