#!/bin/zsh
# Run the 100 normalised-input runs (cell1n, cell3n) natively on the Mac.
# Resumable: re-running skips finished runs. Keeps the Mac awake while it runs.
#   cd ~/Desktop/Honours/Research/ProjectCode/exo_portfolio
#   zsh cluster/run_norm_mac.sh            # default 8 parallel runs
#   zsh cluster/run_norm_mac.sh 6          # or choose the number
set -e
cd "$(dirname "$0")/.."
WORKERS=${1:-8}
if [ ! -d .venv-norm ]; then
  python3 -m venv .venv-norm
  .venv-norm/bin/pip install -q --upgrade pip
  .venv-norm/bin/pip install -q -e ".[rl]"
fi
mkdir -p logs
echo "Starting $WORKERS workers; progress in logs/norm_queue.log"
caffeinate -i .venv-norm/bin/python cluster/run_local_queue.py --workers "$WORKERS" 2>&1 | tee -a logs/norm_queue.log
