"""Launch the full experiment matrix (Manual Part H, milestone M6).

The grid is cells × seeds × folds from configs/experiment_grid.yaml. The
optimiser, data, folds and step budget are IDENTICAL across cells — only the
architecture switches differ (Manual A.1.6); this module enforces that the
YAML's cell definitions match the presets compiled into the agent.

Modes:
  --emit-jobs   print one `python -m exo_portfolio.train ...` command per run
                — each (cell, seed, fold) is an independent job for the
                Mathematical Sciences Cluster (Manual L)
  --run         execute runs sequentially in-process, skipping runs whose
                metrics.json already exists (resumable)
  --aggregate   collect results/*/metrics.json into results/summary.csv and
                print the per-(cell, split) mean ± std table

Usage:
    python -m exo_portfolio.run_grid --config configs/base.yaml \
        configs/universe_djia30.yaml --grid configs/experiment_grid.yaml \
        --emit-jobs [--steps 2000000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from exo_portfolio.config import Config


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def load_grid(path: str | Path) -> dict:
    with open(path) as fh:
        grid = yaml.safe_load(fh)
    _validate_cells(grid["cells"])
    return grid


def _validate_cells(cells: dict) -> None:
    """The YAML documents the switches; the agent compiles them. They must
    never diverge silently."""
    from exo_portfolio.models.presets import CELL_PRESETS

    for name, switches in cells.items():
        preset = CELL_PRESETS.get(name)
        assert preset is not None, f"unknown cell in grid: {name}"
        assert dict(switches) == preset, (
            f"{name}: experiment_grid.yaml {switches} != agent preset {preset}")


def expand(grid: dict) -> list[tuple[str, int, int]]:
    """-> [(cell, seed, fold), ...] in deterministic order."""
    return [(cell, seed, fold)
            for cell in grid["cells"]
            for seed in grid["seeds"]
            for fold in grid["folds"]]


def job_command(config_paths: list[str], cell: str, seed: int, fold: int,
                steps: int | None = None,
                results_root: str = "results") -> str:
    cmd = (f"python -m exo_portfolio.train --config {' '.join(config_paths)} "
           f"--cell {cell} --seed {seed} --fold {fold} --results {results_root}")
    if steps:
        cmd += f" --steps {steps}"
    return cmd


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results_root: str | Path = "results") -> pd.DataFrame:
    """Collect every results/*/metrics.json into one long-format frame:
    one row per (run_id, split) with cell/seed/fold and all metrics."""
    rows = []
    for metrics_path in sorted(Path(results_root).glob("*/metrics.json")):
        run_dir = metrics_path.parent
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        metrics = json.loads(metrics_path.read_text())
        for split, m in metrics.items():
            rows.append({"run_id": run_dir.name, "cell": cfg["cell"],
                         "seed": cfg["seed"], "fold": cfg["fold"],
                         "split": split, **m})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(Path(results_root) / "summary.csv", index=False)
    return df


def summary_table(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Mean ± std across (seed, fold) per cell for the given split."""
    sub = df[df["split"] == split]
    metrics = ["sharpe", "sortino", "cum_log_return", "max_drawdown",
               "mean_turnover"]
    agg = sub.groupby("cell")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--grid", default="configs/experiment_grid.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--emit-jobs", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args(argv)

    grid = load_grid(args.grid)
    combos = expand(grid)

    if args.emit_jobs:
        for cell, seed, fold in combos:
            print(job_command(args.config, cell, seed, fold,
                              args.steps, args.results))
        return

    if args.run:
        from exo_portfolio.train import run as train_run

        for i, (cell, seed, fold) in enumerate(combos):
            cfg = Config.from_yaml(*args.config)
            cfg.cell, cfg.seed, cfg.fold = cell, seed, fold
            run_dir = Path(args.results) / cfg.run_id
            if is_complete(run_dir):
                print(f"[{i + 1}/{len(combos)}] {cfg.run_id}: complete, skipping")
                continue
            print(f"[{i + 1}/{len(combos)}] {cfg.run_id}: running")
            train_run(cfg, total_steps=args.steps, results_root=args.results)

    if args.aggregate or args.run:
        df = aggregate(args.results)
        if len(df):
            print(f"\nsummary.csv written ({len(df)} rows). "
                  f"Test-split mean ± std across seeds/folds:")
            print(summary_table(df).round(3).to_string())
        else:
            print("no completed runs found to aggregate")


if __name__ == "__main__":
    main(sys.argv[1:])
