"""Normalised-input flat agents (supervisor review, 2026-10).

Question: do the concatenating agents (cell1, cell3) fail because their raw
inputs mix scales (VIX levels ~10-80 beside daily returns ~0.01), rather than
because they concatenate? cell1n / cell3n are cell1 / cell3 with every input
z-scored using statistics from the fold's TRAINING window only (validation and
test rows never touch the statistics):

  * exo_actor and exo_critic_extra: per-column (x - mean) / std.
  * endo: the last-K portfolio-return window is multiplied by
    1 / std(daily log return of the equal-weight portfolio) over the training
    window (env.endo_return_scale). Portfolio weights and the cash fraction
    are already in [0, 1] and are left as they are.

Everything else matches the original grid exactly: the same feature calendar
and macro alignment the grid was trained on (eval.macro_lag_reeval legacy
build), the same folds, seeds, PPO settings and 2M-step budget. So
cell1n - cell1 and cell3n - cell3 isolate the effect of input scaling.

    python -m exo_portfolio.experiments.norm_flat --config configs/base.yaml \\
        configs/universe_djia30.yaml --cell cell3n --seed 0 --fold 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exo_portfolio.config import Config
from exo_portfolio.data.align import FeatureSet
from exo_portfolio.data.splits import rolling_origin_folds

NORM_CELLS = ("cell1n", "cell3n")
STD_FLOOR = 1e-8


def grid_features(cfg: Config) -> FeatureSet:
    """The exact actor features the original grid used, plus the matching
    critic-only copy (macro at its FRED stamp date with no lag)."""
    from exo_portfolio.data.loaders import load_raw_bundle
    from exo_portfolio.eval.macro_lag_reeval import (MACRO_COLS,
                                                     build_feature_pair,
                                                     legacy_macro_lagged)
    legacy, _ = build_feature_pair(cfg)
    raw = load_raw_bundle(cfg)
    crit = legacy_macro_lagged(raw["macro"], {}, raw["prices"].index)
    crit = crit[list(MACRO_COLS)].loc[legacy.dates]
    crit.columns = [f"{c}_unlagged" for c in MACRO_COLS]
    return FeatureSet(dates=legacy.dates, prices=legacy.prices,
                      exo_actor=legacy.exo_actor, exo_critic_extra=crit)


def normalise(features: FeatureSet, train_idx: np.ndarray) -> tuple[FeatureSet, dict]:
    rows = features.dates[train_idx]

    def z(df: pd.DataFrame):
        mu = df.loc[rows].mean()
        sd = df.loc[rows].std(ddof=0).clip(lower=STD_FLOOR)
        return (df - mu) / sd, mu, sd

    exo, mu_a, sd_a = z(features.exo_actor)
    crit, mu_c, sd_c = z(features.exo_critic_extra)
    p = features.prices.loc[rows].values
    ew = np.log((p[1:] / p[:-1]).mean(axis=1))
    endo_scale = float(1.0 / max(ew.std(ddof=0), STD_FLOOR))
    stats = {"train_start": str(rows[0].date()), "train_end": str(rows[-1].date()),
             "endo_return_scale": endo_scale,
             "exo_actor_std_min": float(sd_a.min()), "exo_actor_std_max": float(sd_a.max()),
             "critic_mean": mu_c.round(6).to_dict(), "critic_std": sd_c.round(6).to_dict(),
             "market_macro_mean": mu_a.iloc[-4:].round(6).to_dict(),
             "market_macro_std": sd_a.iloc[-4:].round(6).to_dict()}
    return FeatureSet(dates=features.dates, prices=features.prices,
                      exo_actor=exo, exo_critic_extra=crit), stats


def run_one(cfg: Config, total_steps: int | None, results_root: str) -> dict:
    from exo_portfolio.train import run
    assert cfg.cell in NORM_CELLS, cfg.cell
    feats = grid_features(cfg)
    fold = rolling_origin_folds(feats.dates, n_folds=5)[cfg.fold]
    feats, stats = normalise(feats, fold.train_idx)
    cfg.env.endo_return_scale = stats["endo_return_scale"]
    run_dir = Path(results_root) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    return run(cfg, total_steps=total_steps, results_root=results_root,
               features=feats)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--cell", required=True, choices=NORM_CELLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    cfg = Config.from_yaml(*a.config)
    cfg.cell, cfg.seed, cfg.fold = a.cell, a.seed, a.fold
    run_one(cfg, a.steps, a.results)


if __name__ == "__main__":
    main(sys.argv[1:])
