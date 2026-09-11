#!/bin/bash
# train_array.slurm — one Slurm job ARRAY task per (cell, seed, fold).
# Submit from the repo root:  sbatch cluster/train_array.slurm
# (or size the array to the job file:
#     sbatch --array=1-$(wc -l < cluster/jobs.txt)%20 cluster/train_array.slurm )
#
#SBATCH --job-name=exo-grid
#SBATCH --partition=REPLACE_ME        # <-- run `sinfo`; pick a CPU partition whose TIMELIMIT covers a full run
#SBATCH --array=1-250%20              # 250 runs; %20 = at most 20 at a time (be a good cluster citizen)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2             # PPO is CPU-bound; start small, raise only if a run is CPU-starved
#SBATCH --mem=4G                      # data is tiny (~2 MB); 4G is generous
#SBATCH --time=24:00:00               # est. ~11 h/run from your laptop smoke test — MEASURE one first
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- environment (match the module you used in setup_env.sh) ---
module purge || true
module load python/3.11 || module load anaconda3 || true
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"   # keep torch from grabbing every core

# --- pick this task's command from the frozen job list ---
JOB_FILE="cluster/jobs.txt"
CMD="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_FILE")"
[ -n "$CMD" ] || { echo "no command at line ${SLURM_ARRAY_TASK_ID} of $JOB_FILE"; exit 1; }

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID}"
echo "cmd = $CMD"
srun bash -c "$CMD"
