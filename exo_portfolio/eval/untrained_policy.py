"""Untrained-policy baseline (supervisor review, 2026-10).

Every agent starts from a near-zero mean logit (output layer initialised with
orthogonal gain 0.01), so before training its deterministic policy is close to
a uniform portfolio over the 29 stocks and cash. This evaluates that starting
point: each cell's agent is built exactly as train.run builds it (same seed,
same features the grid used), NOT trained, and rolled out deterministically on
its fold's test window. If the untrained policy scores close to equal weight,
training moved the agents away from their best result.

    python -m exo_portfolio.eval.untrained_policy --config configs/base.yaml \\
        configs/universe_djia30.yaml --cells cell0 cell1 cell4 --out results/untrained
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exo_portfolio.config import Config, seed_everything
from exo_portfolio.data.splits import rolling_origin_folds
from exo_portfolio.eval.metrics import summarize

SB3_LEVEL = {"cell0": "cell0", "cell1": "cell1"}


def rollout(env, act):
    obs, _ = env.reset(seed=0)
    rhos, taus, expo, dates = [], [], [], []
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(act(obs))
        rhos.append(info["step_log_return"]); taus.append(info["turnover"])
        expo.append(1.0 - float(info["weights"][0])); dates.append(info["date"])
        done = term or trunc
    return np.array(rhos), np.array(taus), np.array(expo), dates


def untrained_actor(cfg: Config, features, t0: int, t1: int):
    seed_everything(cfg.seed)
    if cfg.cell in SB3_LEVEL:
        from stable_baselines3 import PPO
        from exo_portfolio.baselines.sb3_baselines import make_env
        env = make_env(features, cfg, t0, t1, SB3_LEVEL[cfg.cell])
        model = PPO("MlpPolicy", env, seed=cfg.seed, verbose=0,
                    learning_rate=cfg.train.lr, gamma=cfg.train.gamma,
                    gae_lambda=cfg.train.gae_lambda, clip_range=cfg.train.clip)
        make = lambda f, s, e: make_env(f, cfg, s, e, SB3_LEVEL[cfg.cell])
        return make, lambda o: model.predict(o, deterministic=True)[0]
    import torch
    from exo_portfolio.algos.ppo import _to_tensors
    from exo_portfolio.envs.portfolio_env import PortfolioEnv
    from exo_portfolio.models.agent import ExoActorCritic, model_cfg_for_cell
    env = PortfolioEnv(features, cfg, t0, t1)
    dims = {k: int(np.prod(s.shape)) for k, s in env.observation_space.spaces.items()}
    agent = ExoActorCritic(dims, features.prices.shape[1] + 1,
                           model_cfg_for_cell(cfg.cell, cfg.model),
                           n_assets=features.prices.shape[1],
                           price_window=cfg.data.price_window).eval()

    def act(o):
        with torch.no_grad():
            a, _, _ = agent.act(_to_tensors(o, "cpu"), deterministic=True)
        return a.squeeze(0).numpy()
    return (lambda f, s, e: PortfolioEnv(f, cfg, s, e)), act


def main(argv=None):
    from exo_portfolio.experiments.norm_flat import grid_features
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", default=["cell0", "cell1", "cell4"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="results/untrained")
    a = ap.parse_args(argv)
    base = Config.from_yaml(*a.config)
    feats = grid_features(base)
    folds = rolling_origin_folds(feats.dates, n_folds=5)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in a.cells:
        for seed in a.seeds:
            for k, fold in enumerate(folds):
                cfg = Config.from_yaml(*a.config)
                cfg.cell, cfg.seed, cfg.fold = cell, seed, k
                t0, t1 = int(fold.train_idx[0]), int(fold.train_idx[-1])
                make, act = untrained_actor(cfg, feats, t0, t1)
                s, e = int(fold.test_idx[0]), int(fold.test_idx[-1])
                rho, tau, expo, dates = rollout(make(feats, s, e), act)
                m = summarize(rho, tau)
                rows.append({"cell": f"untrained_{cell}", "seed": seed, "fold": k,
                             **m, "mean_exposure": float(expo.mean())})
                pd.DataFrame({"date": dates, "log_return": rho, "turnover": tau,
                              "exposure": expo}).to_csv(
                    out / f"untrained_{cell}_seed{seed}_fold{k}.csv", index=False)
        print(cell, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "untrained_runs.csv", index=False)
    print(df.groupby("cell")[["sharpe", "sortino", "cum_log_return", "max_drawdown",
                              "mean_turnover", "mean_exposure"]].mean().round(4))


if __name__ == "__main__":
    main(sys.argv[1:])
