"""Test-time exo ablation: does a TRAINED agent use its exogenous inputs?

Complements the retraining ablation axis (exo_portfolio.exo_ablation), which
asks whether an input helps when the agent learns without it. Here nothing is
retrained: every saved model is rolled out on its fold's test window once
intact and once per (group, mode) with one named exo_actor group masked, and
each masked rollout is paired against the SAME model intact. With 50 models
per cell this has real power where a 1-seed retraining sweep has none.

Masking (observations only — prices, and so P_exo, are untouched):
  permute  PRIMARY. Rows of the group's block are permuted in time WITHIN the
           test window: the marginal distribution is exactly preserved, only
           the alignment with the date is destroyed (permutation importance).
           `n_perm` draws per model; the draw depends only on (fold, group,
           draw), so every cell and seed sees identical permutations.
  zero     SECONDARY. Block set to 0 — out of distribution for level features
           such as VIX, so it mixes "information removed" with "input shift".

Per masked rollout, relative to the intact one: Δ sharpe / cum_log_return /
max_drawdown / mean exposure, and `policy_shift` = mean_t ½‖w_masked − w_intact‖₁
(the fraction of the portfolio reallocated) — nonzero iff the policy reads the
group, whatever that does to returns.

    python -m exo_portfolio.eval.test_time_ablation --config configs/base.yaml \\
        configs/universe_djia30.yaml --cells cell4 --folds 0 --out OUT.csv
    python -m exo_portfolio.eval.test_time_ablation --summarize OUT_DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from exo_portfolio.config import Config
from exo_portfolio.data.align import (EXO_GROUPS, FeatureSet,
                                      exo_group_slices)

MODES = ("permute", "zero")
METRICS = ("sharpe", "cum_log_return", "max_drawdown", "mean_exposure",
           "policy_shift")


def mask_window(features: FeatureSet, group: str, mode: str, start: int,
                end: int, price_window: int,
                rng: np.random.Generator | None = None) -> FeatureSet:
    """Copy of `features` with `group` masked on rows [start, end] only."""
    exo = features.exo_actor.values.astype(np.float32).copy()
    sl = exo_group_slices(features.prices.shape[1], price_window,
                          exo.shape[1])[group]
    rows = np.arange(start, end + 1)
    if mode == "zero":
        exo[start:end + 1, sl] = 0.0
    elif mode == "permute":
        exo[start:end + 1, sl] = exo[rng.permutation(rows), sl]
    else:
        raise ValueError(mode)
    return FeatureSet(dates=features.dates, prices=features.prices,
                      exo_actor=pd.DataFrame(exo, index=features.exo_actor.index,
                                             columns=features.exo_actor.columns),
                      exo_critic_extra=features.exo_critic_extra)


def perm_rng(fold: int, group: str, draw: int) -> np.random.Generator:
    """Common random numbers: same permutation for every cell and seed."""
    return np.random.default_rng([fold, EXO_GROUPS.index(group), draw, 7919])


def load_actor(run_dir: Path, cfg: Config, features: FeatureSet):
    """-> rollout(features, start, end) for a saved SB3 or custom run."""
    if (run_dir / "model.zip").exists():
        from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

        from exo_portfolio.baselines.sb3_baselines import make_env

        algo = {"ppo": PPO, "sac": SAC, "a2c": A2C,
                "ddpg": DDPG, "td3": TD3}[cfg.train.algo]
        model = algo.load(run_dir / "model.zip", device="cpu")
        return lambda f, s, e: _rollout(
            make_env(f, cfg, s, e, cfg.cell),
            lambda o: model.predict(o, deterministic=True)[0], cfg.seed)

    import torch

    from exo_portfolio.algos.ppo import _to_tensors
    from exo_portfolio.envs.portfolio_env import PortfolioEnv
    from exo_portfolio.models.agent import ExoActorCritic
    from exo_portfolio.models.presets import model_cfg_for_cell

    env0 = PortfolioEnv(features, cfg)
    obs_dims = {k: int(np.prod(sp.shape))
                for k, sp in env0.observation_space.spaces.items()}
    agent = ExoActorCritic(obs_dims, features.prices.shape[1] + 1,
                           model_cfg_for_cell(cfg.cell, cfg.model),
                           n_assets=features.prices.shape[1],
                           price_window=cfg.data.price_window)
    agent.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    agent.eval()

    def act(o):
        with torch.no_grad():
            a, _, _ = agent.act(_to_tensors(o, "cpu"), deterministic=True)
        return a.squeeze(0).cpu().numpy()

    return lambda f, s, e: _rollout(PortfolioEnv(f, cfg, s, e), act, cfg.seed)


def _rollout(env, act, seed: int) -> dict:
    """Deterministic rollout keeping the weight path (mirrors the evaluators
    in algos.ppo / baselines.sb3_baselines, which drop it)."""
    from exo_portfolio.eval.metrics import summarize

    obs, _ = env.reset(seed=seed)
    rhos, taus, weights = [], [], []
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(obs))
        rhos.append(info["step_log_return"])
        taus.append(info["turnover"])
        weights.append(np.asarray(info["weights"], float))
        done = terminated or truncated
    w = np.stack(weights)
    row = summarize(np.array(rhos), np.array(taus))
    row["mean_exposure"] = float(np.mean(1.0 - w[:, 0]))
    return {"metrics": row, "weights": w}


def evaluate_run(run_dir: Path, features: FeatureSet, folds,
                 n_perm: int = 3, groups=EXO_GROUPS, modes=MODES) -> list[dict]:
    cfg = Config.from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    fold = folds[cfg.fold]
    s, e = int(fold.test_idx[0]), int(fold.test_idx[-1])
    rollout = load_actor(run_dir, cfg, features)
    base = rollout(features, s, e)
    key = {"run_id": run_dir.name, "cell": cfg.cell, "seed": cfg.seed,
           "fold": cfg.fold}
    rows = [{**key, "group": "none", "mode": "intact", "draw": 0,
             **base["metrics"], "policy_shift": 0.0}]
    for group in groups:
        for mode in modes:
            for draw in range(n_perm if mode == "permute" else 1):
                f = mask_window(features, group, mode, s, e,
                                cfg.data.price_window,
                                perm_rng(cfg.fold, group, draw))
                out = rollout(f, s, e)
                shift = 0.5 * np.abs(out["weights"] - base["weights"]).sum(1).mean()
                rows.append({**key, "group": group, "mode": mode, "draw": draw,
                             **out["metrics"], "policy_shift": float(shift)})
    return rows


def summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Paired (masked − intact) per model, permutation draws averaged first;
    mean ± 95% CI across models, paired test, Holm across groups within each
    (cell, mode, metric)."""
    from exo_portfolio.eval.stats import holm_correction, paired_test

    intact = df[df["mode"] == "intact"].set_index("run_id")
    masked = (df[df["mode"] != "intact"]
              .groupby(["run_id", "cell", "group", "mode"])[list(METRICS)]
              .mean().reset_index())
    rows = []
    for (cell, group, mode), g in masked.groupby(["cell", "group", "mode"]):
        g = g.set_index("run_id")
        base = intact.loc[g.index]
        for metric in METRICS:
            a, b = g[metric].values, base[metric].values
            d = a - b
            res = paired_test(a, b) if len(d) >= 3 else {"pvalue": np.nan}
            rows.append({"cell": cell, "mode": mode, "group": group,
                         "metric": metric, "n_models": len(d),
                         "mean_diff": d.mean(),
                         "ci95": 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                         if len(d) > 1 else np.nan,
                         "intact_mean": b.mean(), "pvalue": res["pvalue"]})
    out = pd.DataFrame(rows)
    out["pvalue_holm"] = np.nan
    for _, idx in out.groupby(["cell", "mode", "metric"]).groups.items():
        p = out.loc[idx, "pvalue"]
        if p.notna().all():
            out.loc[idx, "pvalue_holm"] = holm_correction(p.tolist())
    out["significant_5pct"] = out["pvalue_holm"] < 0.05
    out["mode"] = pd.Categorical(out["mode"], ["permute", "zero"])
    return out.sort_values(["mode", "metric", "cell", "group"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cells", nargs="+", default=["cell0", "cell1", "cell2", "cell4"])
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--out", help="per-rollout CSV for this chunk")
    ap.add_argument("--summarize", metavar="DIR",
                    help="combine DIR/*.csv into results/tables/")
    args = ap.parse_args(argv)

    if args.summarize:
        df = pd.concat([pd.read_csv(p) for p in sorted(Path(args.summarize).glob("*.csv"))])
        tables = Path(args.results) / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        df.to_csv(tables / "test_time_ablation_runs.csv", index=False)
        summ = summarize_ablation(df)
        summ.to_csv(tables / "test_time_ablation.csv", index=False)
        print(f"{len(df)} rollouts, {df['run_id'].nunique()} models -> "
              f"{tables / 'test_time_ablation.csv'}")
        return

    from exo_portfolio.data.align import build_features
    from exo_portfolio.data.loaders import load_raw_bundle
    from exo_portfolio.data.splits import rolling_origin_folds

    cfg0 = Config.from_yaml(*args.config)
    features = build_features(**load_raw_bundle(cfg0), cfg=cfg0)
    folds = rolling_origin_folds(features.dates, n_folds=5)
    rows = []
    for cell in args.cells:
        for fold in args.folds:
            for run_dir in sorted(Path(args.results).glob(f"{cell}_seed*_fold{fold}")):
                rows += evaluate_run(run_dir, features, folds, args.n_perm)
                print(f"[tta] {run_dir.name} done", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
