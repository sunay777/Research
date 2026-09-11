#!/usr/bin/env bash
# trial_run.sh — short laptop smoke-test of the training pipeline.
#
# Wraps `python -m exo_portfolio.train` with the offline configs, the repo's
# .venv, and a throwaway results dir, so you can sanity-check a cell without
# typing the full command. NOT a research run — see --steps note below.
#
# Usage:
#   ./trial_run.sh                       # cell4, 4096 steps, seed 0, fold 0
#   ./trial_run.sh cell1                 # pick a cell (cell0..cell4)
#   ./trial_run.sh cell4 20480           # cell + step budget
#   ./trial_run.sh cell2 8192 3 1        # cell + steps + seed + fold
#
# Cells: cell0/cell1 = SB3 (PPO), cell2/cell3/cell4 = custom Exo-PPO.
# Steps: 4096 ≈ 80s on a laptop; the real research budget is 2,000,000 (cluster).
set -euo pipefail

# Always run from the directory this script lives in (the working copy).
cd "$(dirname "$0")"

CELL="${1:-cell4}"
STEPS="${2:-4096}"
SEED="${3:-0}"
FOLD="${4:-0}"
RESULTS="${RESULTS_DIR:-results_smoke}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"   # fall back to whatever python is on PATH

echo "== trial run: cell=$CELL steps=$STEPS seed=$SEED fold=$FOLD -> $RESULTS/ =="
time "$PY" -m exo_portfolio.train \
  --config configs/base.yaml configs/universe_djia30.yaml \
  --cell "$CELL" --seed "$SEED" --fold "$FOLD" --steps "$STEPS" \
  --results "$RESULTS"

RUN_DIR="$RESULTS/${CELL}_seed${SEED}_fold${FOLD}"
echo
echo "== outputs in $RUN_DIR/ =="
ls -1 "$RUN_DIR"
if [ -f "$RUN_DIR/train_log.csv" ]; then
  echo
  echo "== PPO learning curve (train_log.csv) =="
  column -s, -t "$RUN_DIR/train_log.csv" | cut -c1-100
fi
