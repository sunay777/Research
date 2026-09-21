#!/bin/bash
#SBATCH --job-name=exo-grid
#SBATCH --partition=stampede
#SBATCH --array=0-12%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=71:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Grouped, self-running grid driver for the exo_portfolio ablation study on mscluster.
#
# Why this exists: stampede QOS caps you at MaxSubmit=20 queued and MaxJobs=2 running,
# so a 250-task array is rejected. Instead we submit 13 "group" tasks (array 0-12), 2 run
# at a time (%2), and each group runs PER=20 jobs.txt lines back-to-back. One `sbatch`,
# hands-off for the whole grid.
#
# Idempotent: a run whose results/<cell>_seed<s>_fold<f>/metrics.json exists is skipped,
# so after any interruption/timeout you just re-run `sbatch cluster/run_grid_grouped.slurm`
# and it resumes, redoing only what's missing.
#
# Hang protection: individual runs occasionally hang (mscluster filesystem-I/O stalls),
# and a hung run would otherwise block its whole group until the wall clock. `timeout`
# kills any single run that exceeds 6h (~6x the normal ~1h) and moves on.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/exo_portfolio/pybin:$PATH"     # pybin/python -> apptainer anaconda container
export CUDA_VISIBLE_DEVICES=                        # CPU-only
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PER=20
JOBS="cluster/jobs.txt"
TOTAL=$(wc -l < "$JOBS")
start=$(( SLURM_ARRAY_TASK_ID * PER + 1 ))
end=$(( start + PER - 1 )); [ "$end" -gt "$TOTAL" ] && end=$TOTAL

echo "group $SLURM_ARRAY_TASK_ID -> jobs $start..$end on $(hostname)"
for i in $(seq "$start" "$end"); do
  CMD="$(sed -n "${i}p" "$JOBS")"
  [ -n "$CMD" ] || continue
  cell=$(sed -n 's/.*--cell \([^ ]*\).*/\1/p' <<< "$CMD")
  seed=$(sed -n 's/.*--seed \([^ ]*\).*/\1/p' <<< "$CMD")
  fold=$(sed -n 's/.*--fold \([^ ]*\).*/\1/p' <<< "$CMD")
  if [ -f "results/${cell}_seed${seed}_fold${fold}/metrics.json" ]; then
    echo "[$i] skip ${cell}_seed${seed}_fold${fold} (done)"; continue
  fi
  echo "[$i] RUN ${cell}_seed${seed}_fold${fold} ($(date '+%F %T'))"
  timeout -k 1m 6h bash -c "$CMD"
  rc=$?
  if [ "$rc" -eq 124 ]; then echo "[$i] KILLED after 6h (hung) — moving on"
  elif [ "$rc" -ne 0 ]; then echo "[$i] FAILED rc=$rc — moving on"; fi
done
echo "group $SLURM_ARRAY_TASK_ID finished"
