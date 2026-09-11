#!/usr/bin/env bash
# setup_env.sh — run ONCE on the LOGIN NODE (has internet) to build the
# cluster Python environment. Do NOT reuse the .venv from your Mac — it is
# macOS/arm64 and will not run on the Linux compute nodes.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

# 1. Load a Python >= 3.11. Names vary per cluster — check `module avail`
#    and replace the line below with whatever it lists (e.g. python/3.11,
#    Python/3.11.4, anaconda3).
module purge || true
module load python/3.11 || module load anaconda3 || true
python --version

# 2. Fresh virtualenv in the repo (a stale Mac .venv would break imports).
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Install the project + RL extras (torch/SB3/tensorboard). CPU wheels are
#    fine — this project does not need a GPU.
pip install -e ".[rl]"

echo
echo "Environment ready. data_cache present? ->"
ls -1 data_cache/ 2>/dev/null || echo "  MISSING: copy data_cache/ up (see cluster/README.md) so compute nodes never call yfinance."
