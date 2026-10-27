"""Resumable local job queue for the normalised-input runs (cell1n, cell3n).

Runs every (cell, seed, fold) whose results/<run_id>/metrics.json is missing,
N at a time, one thread each. Safe to stop and restart: finished runs are
skipped, and a run interrupted midway simply starts again.

    python cluster/run_local_queue.py --workers 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ["configs/base.yaml", "configs/universe_djia30.yaml"]


def jobs(cells):
    # interleave the cheap SB3 cell with the slow custom cell so the queue
    # finishes evenly
    out = []
    for fold in range(5):
        for seed in range(10):
            for cell in cells:
                out.append((cell, seed, fold))
    return out


def done(cell, seed, fold):
    return (ROOT / "results" / f"{cell}_seed{seed}_fold{fold}" / "metrics.json").exists()


def run(job):
    cell, seed, fold = job
    if done(*job):
        return job, "skip"
    log = ROOT / "logs" / f"{cell}_seed{seed}_fold{fold}.log"
    log.parent.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    t = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call([sys.executable, "-m", "exo_portfolio.experiments.norm_flat",
                              "--config", *CONFIG, "--cell", cell,
                              "--seed", str(seed), "--fold", str(fold)],
                             cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return job, f"rc={rc} {(time.time() - t) / 3600:.2f}h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cells", nargs="+", default=["cell1n", "cell3n"])
    a = ap.parse_args()
    todo = [j for j in jobs(a.cells) if not done(*j)]
    print(f"{len(todo)} runs to do", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for job, status in ex.map(run, todo):
            print(time.strftime("%Y-%m-%d %H:%M"), *job, status, flush=True)


if __name__ == "__main__":
    main()
