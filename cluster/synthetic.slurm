#!/bin/bash
# synthetic.slurm — full-budget Part-I P_exo-shift experiment
# (10 seeds x 100k steps, cells 1/2/4) -> results/tables/synthetic_shift.csv.
#
# BEFORE SUBMITTING: delete results/tables/synthetic_shift.csv. stage_synthetic
# skips if that file exists, and the current one is a SMOKE-budget result.
#
#     rm -f results/tables/synthetic_shift.csv
#     sbatch cluster/synthetic.slurm
#SBATCH --job-name=exo-syn
#SBATCH --partition=stampede
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=cluster/logs/syn_%j.out
#SBATCH --error=cluster/logs/syn_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/exo_portfolio/pybin:$PATH"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "START $(date '+%F %T') on $(hostname)"
python -m exo_portfolio.reproduce \
  --config configs/base.yaml configs/universe_djia30.yaml \
  --budget full --stages synthetic
echo "exit rc=$? $(date '+%F %T')"
