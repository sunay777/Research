#!/usr/bin/env bash
# make_jobs.sh — run ONCE on the login node to freeze the 250 training
# commands (5 cells x 10 seeds x 5 folds) into cluster/jobs.txt, at the full
# research budget of 2,000,000 steps. The job array indexes this file.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m exo_portfolio.run_grid \
  --config configs/base.yaml configs/universe_djia30.yaml \
  --grid configs/experiment_grid.yaml \
  --steps 2000000 \
  --emit-jobs > cluster/jobs.txt
echo "wrote cluster/jobs.txt with $(wc -l < cluster/jobs.txt) jobs"
