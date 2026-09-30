#!/bin/bash
#SBATCH --job-name=exo-miss
#SBATCH --partition=stampede
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/exo_portfolio/pybin:$PATH"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
CMD="$(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster/jobs_missing.txt)"
[ -n "$CMD" ] || { echo "no line $SLURM_ARRAY_TASK_ID"; exit 0; }
echo "RUN: $CMD  ($(date '+%F %T') on $(hostname))"
timeout -k 1m 6h bash -c "$CMD"
echo "exit rc=$?"
