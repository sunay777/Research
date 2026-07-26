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
    run_dir = Path(results_root) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    seed_everything(cfg.seed)

    features = build_features(**load_raw_bundle(cfg), cfg=cfg)
    fold = rolling_origin_folds(features.dates, n_folds=5)[cfg.fold]
    t0, t1 = int(fold.train_idx[0]), int(fold.train_idx[-1])

    if cfg.cell in SB3_CELLS:
        from exo_portfolio.baselines.sb3_baselines import (
            evaluate_policy_window, train_sb3)

        model = train_sb3(cfg, features, t0, t1, level=cfg.cell,
                          run_dir=run_dir, total_steps=total_steps)
        evaluate = lambda s, e: evaluate_policy_window(
            model, features, cfg, s, e, cfg.cell)
    else:                                   # cells 2-4: the custom Exo-agent
        import json as _json

        import numpy as np_
        import torch

        from exo_portfolio.algos.ppo import PPO, evaluate_agent_window
        from exo_portfolio.envs.portfolio_env import PortfolioEnv
        from exo_portfolio.models.agent import (ExoActorCritic,
                                                model_cfg_for_cell)

        env = PortfolioEnv(features, cfg, t0, t1)
        obs_dims = {k: int(np_.prod(s.shape))
                    for k, s in env.observation_space.spaces.items()}
        agent = ExoActorCritic(obs_dims, features.prices.shape[1] + 1,
                               model_cfg_for_cell(cfg.cell, cfg.model),
                               n_assets=features.prices.shape[1],
                               price_window=cfg.data.price_window)
        # capacity control (Manual L): record parameter counts per cell
        (run_dir / "param_counts.json").write_text(
            _json.dumps(agent.param_counts(), indent=2))

        ppo = PPO(env, agent, cfg, run_dir=run_dir)
        ppo.learn(int(total_steps or cfg.train.total_steps))
        torch.save(agent.state_dict(), run_dir / "model.pt")
        evaluate = lambda s, e: evaluate_agent_window(agent, features, cfg, s, e)

    metrics = {}
    series = {}
    for split_name, idx in (("train", fold.train_idx), ("val", fold.val_idx),
                            ("test", fold.test_idx)):
        out = evaluate(int(idx[0]), int(idx[-1]))
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
