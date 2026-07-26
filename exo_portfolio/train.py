"""Single-run entrypoint (Manual C.1): (cell, seed, fold) -> results/{run_id}/.

Usage:
    python -m exo_portfolio.train --config configs/base.yaml configs/universe_djia30.yaml \
        --cell cell1 --seed 0 --fold 0 [--steps 200000] [--algo ppo]

Writes into results/{run_id}/:
    config.yaml    the exact resolved config (reproducibility, Manual L)
    metrics.json   train/val/test metric rows
    model.zip      the trained SB3 model (cells 0-1)
    series_test.csv  per-day test log-returns + turnover (for M7 regime slicing)

Cells 0-1 run on SB3 (this milestone). Cells 2-4 dispatch to the custom
Exo-agent from M5 onward.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from exo_portfolio.config import Config, seed_everything
from exo_portfolio.data.align import build_features
from exo_portfolio.data.loaders import load_raw_bundle
from exo_portfolio.data.splits import rolling_origin_folds


SB3_CELLS = ("cell0", "cell1")


def run(cfg: Config, total_steps: int | None = None,
        results_root: str | Path = "results") -> dict:
    from exo_portfolio.baselines.sb3_baselines import (evaluate_policy_window,
                                                       train_sb3)

    run_dir = Path(results_root) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    seed_everything(cfg.seed)

    features = build_features(**load_raw_bundle(cfg), cfg=cfg)
    fold = rolling_origin_folds(features.dates, n_folds=5)[cfg.fold]

    if cfg.cell not in SB3_CELLS:
        raise NotImplementedError(
            f"{cfg.cell} needs the custom Exo-agent (milestone M5); "
            "cells 0-1 are the SB3 baselines.")

    model = train_sb3(cfg, features,
                      int(fold.train_idx[0]), int(fold.train_idx[-1]),
                      level=cfg.cell, run_dir=run_dir,
                      total_steps=total_steps)

    metrics = {}
    series = {}
    for split_name, idx in (("train", fold.train_idx), ("val", fold.val_idx),
                            ("test", fold.test_idx)):
        out = evaluate_policy_window(model, features, cfg,
                                     int(idx[0]), int(idx[-1]), cfg.cell)
        metrics[split_name] = out["metrics"]
        series[split_name] = out

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "date": series["test"]["dates"],
        "log_return": series["test"]["log_returns"],
        "turnover": series["test"]["turnover"],
    }).to_csv(run_dir / "series_test.csv", index=False)

    print(f"[{cfg.run_id}] fold={fold.name}")
    for k, m in metrics.items():
        print(f"  {k:>5}: sharpe={m['sharpe']:+.3f} cumlog={m['cum_log_return']:+.4f} "
              f"maxDD={m['max_drawdown']:.1%} turn={m['mean_turnover']:.4f}")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--cell", default="cell1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--algo", default=None, help="override cfg.train.algo")
    ap.add_argument("--steps", type=int, default=None,
                    help="override cfg.train.total_steps")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(*args.config)
    cfg.cell, cfg.seed, cfg.fold = args.cell, args.seed, args.fold
    if args.algo:
        cfg.train.algo = args.algo
    run(cfg, total_steps=args.steps, results_root=args.results)


if __name__ == "__main__":
    main(sys.argv[1:])
