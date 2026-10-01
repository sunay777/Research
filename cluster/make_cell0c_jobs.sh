#!/usr/bin/env bash
# make_cell0c_jobs.sh — run ONCE on the login node to freeze the 50 cell0c
# training commands (1 cell x 10 seeds x 5 folds, 2,000,000 steps) into
# cluster/jobs_missing.txt, which cluster/run_missing.slurm indexes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/exo_portfolio/pybin:$PATH"
export CUDA_VISIBLE_DEVICES=
python -m exo_portfolio.run_grid \
  --config configs/base.yaml configs/universe_djia30.yaml \
  --grid configs/experiment_grid_cell0c.yaml \
  --steps 2000000 \
  --emit-jobs > cluster/jobs_missing.txt
echo "wrote cluster/jobs_missing.txt with $(wc -l < cluster/jobs_missing.txt) jobs (expect 50)"
