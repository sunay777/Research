# Running the exo_portfolio grid on mscluster

The full ablation grid is 5 cells x 10 seeds x 5 folds = **250 independent
runs**, each 2,000,000 steps. Each run is one Slurm array task. CPU only —
no GPU needed.

Golden rule from the workshop: the **login node is for preparation only**
(edit, scp, submit). Never run training on it. Compute nodes do the work.

## 0. Log in and read the MOTD
    ssh <your-username>@<login-node>      # host + credentials are in your access email
Read the Message of the Day, then see what exists:
    sinfo            # partition names + time limits  -> fill REPLACE_ME in the .slurm files
    module avail     # exact Python module name       -> fix the `module load` lines

## 1. Get the code + data up (from your Mac, in ProjectCode/)
    rsync -av --exclude='.venv' --exclude='results/*' \
      exo_portfolio/ <user>@<login-node>:~/exo_portfolio/
`data_cache/` (the offline price/macro CSVs) rides along so compute nodes
never have to call yfinance.

## 2. Build the environment (login node, once)
    cd ~/exo_portfolio
    bash cluster/setup_env.sh        # edit the module name inside first if needed

## 3. Smoke-test ONE run before scaling (workshop: start small, test first)
    source .venv/bin/activate
    ./trial_run.sh cell4 20480       # ~a few minutes; confirms imports + data
Then time a realistic slice as a single array task to size --time:
    ./make_jobs.sh                   # writes cluster/jobs.txt (250 lines, 2M steps each)
    sbatch --array=1-1 cluster/train_array.slurm
    squeue -u $USER                  # watch it; check logs/exo-grid_*_1.out

## 4. Launch the full grid
    sbatch --array=1-$(wc -l < cluster/jobs.txt)%20 cluster/train_array.slurm
`%20` caps concurrent tasks at 20 (good citizenship). Monitor / cancel:
    squeue -u $USER
    scancel <jobid>                  # or  scancel <jobid>_<taskid>  for one task

## 5. Collect results
    sbatch cluster/aggregate.slurm   # -> results/summary.csv + per-cell table
Then pull them back to your Mac:
    rsync -av <user>@<login-node>:~/exo_portfolio/results/ ProjectCode/exo_portfolio/results/

## Notes
- Each task is independent, so failures are per-run: just resubmit those
  array indices, e.g. `sbatch --array=7,42,88 cluster/train_array.slurm`.
- Runtime estimate from your laptop smoke test (4096 steps ~= 80 s) is
  ~11 h per 2M-step run. If that x 250 is too much wall-clock, cut seeds or
  folds in experiment_grid.yaml first, or raise the `%N` concurrency.
- REPLACE_ME (partition) appears in train_array.slurm and aggregate.slurm.
