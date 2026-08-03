"""M9 — one command regenerates every table and figure (Manual A.3 / F).

    python -m exo_portfolio.reproduce --config configs/base.yaml \
        configs/universe_djia30.yaml --budget smoke|full [--stages ...]

Stages (run in order; each is idempotent/resumable):
  data       ensure the raw cache + aligned features exist
  grid       the ablation grid (resumable; skips completed runs)
  classical  Part-K baselines per fold -> tables/classical.csv
  synthetic  Part-I P_exo-shift experiment -> synthetic_shift.csv
  report     summary, regime-conditional, paired-stats, cost-sensitivity tables
  figures    every write-up figure -> results/figures/

Budgets:
  smoke  1 seed x 1 fold x 4096 steps, 2 synthetic seeds — end-to-end wiring
         check on a laptop (~10 min)
  full   the Manual's budgets (10 seeds x 5 folds x 2M steps, 10 synthetic
         seeds) — cluster scale; identical code path
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
from exo_portfolio.data.loaders import load_raw_bundle, fetch_single
from exo_portfolio.data.splits import folds_cover, rolling_origin_folds

BUDGETS = {
    "smoke": dict(seeds=[0], folds=[0], steps=4096, syn_seeds=2,
                  syn_steps=20_000, syn_eval_eps=10),
    "full": dict(seeds=list(range(10)), folds=list(range(5)), steps=None,
                 syn_seeds=10, syn_steps=100_000, syn_eval_eps=20),
}
COSTS_J5 = (0.0005, 0.001, 0.002)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_data(cfg: Config):
    bundle = load_raw_bundle(cfg)
    features = build_features(**bundle, cfg=cfg)
    folds = rolling_origin_folds(features.dates, n_folds=5)
    assert folds_cover(features.dates, folds), \
        "stress windows not covered by test blocks (Manual D.5)"
    print(f"[data] {len(features.dates)} aligned days "
          f"[{features.dates[0].date()} .. {features.dates[-1].date()}], "
          f"{features.prices.shape[1]} assets, folds OK")
    return bundle, features, folds


def stage_grid(cfg_paths, budget, results: Path):
    from exo_portfolio.run_grid import aggregate, expand, is_complete
    from exo_portfolio.train import run as train_run

    combos = [(c, s, f) for c in ("cell0", "cell1", "cell2", "cell3", "cell4")
              for s in budget["seeds"] for f in budget["folds"]]
    for i, (cell, seed, fold) in enumerate(combos):
        cfg = Config.from_yaml(*cfg_paths)
        cfg.cell, cfg.seed, cfg.fold = cell, seed, fold
        if is_complete(results / cfg.run_id):
            print(f"[grid {i + 1}/{len(combos)}] {cfg.run_id}: done, skip")
            continue
        print(f"[grid {i + 1}/{len(combos)}] {cfg.run_id}")
        train_run(cfg, total_steps=budget["steps"], results_root=results)
    return aggregate(results)


def stage_classical(cfg: Config, features, folds, results: Path) -> pd.DataFrame:
    from exo_portfolio.baselines.classical import (buy_and_hold_index,
                                                   equal_weight_targets,
                                                   mean_variance_targets,
                                                   simulate_target_weights,
                                                   vol_overlay_targets)
    from exo_portfolio.eval.metrics import summarize

    index = fetch_single(cfg.data.index_ticker, cfg.data.start, cfg.data.end,
                         cfg.data.cache_dir, "gspc")
    rows = []
    for fold_id, fold in enumerate(folds):
        prices = features.prices.iloc[fold.test_idx]
        T, N = prices.shape
        runs = {
            "equal_weight": simulate_target_weights(
                prices, equal_weight_targets(T, N), cfg.env.transaction_cost),
            "vol_overlay": simulate_target_weights(
                prices, vol_overlay_targets(prices), cfg.env.transaction_cost),
            "mean_variance": simulate_target_weights(
                prices, mean_variance_targets(prices), cfg.env.transaction_cost),
            "buy_and_hold_index": buy_and_hold_index(index.loc[prices.index]),
        }
        for name, r in runs.items():
            rows.append({"cell": name, "fold": fold_id, "split": "test",
                         **summarize(r["log_returns"], r["turnover"])})
    df = pd.DataFrame(rows)
    out = results / "tables" / "classical.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[classical] {len(df)} rows -> {out}")
    return df


def stage_synthetic(budget, results: Path) -> pd.DataFrame:
    from exo_portfolio.envs.synthetic_exo import run_shift_experiment

    out = results / "tables" / "synthetic_shift.csv"
    if out.exists():
        print(f"[synthetic] exists, skip ({out})")
        return pd.read_csv(out)
    df = run_shift_experiment(cells=("cell1", "cell2", "cell4"),
                              seeds=range(budget["syn_seeds"]),
                              train_steps=budget["syn_steps"],
                              n_eval_episodes=budget["syn_eval_eps"],
                              out_csv=str(out))
    print(f"[synthetic] {len(df)} rows -> {out}")
    return df


def _load_policy(run_dir: Path, features, cfg: Config):
    """Rebuild the evaluate(start, end, cfg) callable for a saved run."""
    if (run_dir / "model.zip").exists():                    # SB3 cells
        from stable_baselines3 import A2C, PPO, SAC

        algo = {"ppo": PPO, "sac": SAC, "a2c": A2C}[cfg.train.algo]
        model = algo.load(run_dir / "model.zip", device="cpu")
        from exo_portfolio.baselines.sb3_baselines import evaluate_policy_window

        return lambda s, e, c: evaluate_policy_window(model, features, c, s, e,
                                                      c.cell)
    if (run_dir / "model.pt").exists():                     # custom cells
        import torch

        from exo_portfolio.algos.ppo import evaluate_agent_window
        from exo_portfolio.envs.portfolio_env import PortfolioEnv
        from exo_portfolio.models.agent import ExoActorCritic
        from exo_portfolio.models.presets import model_cfg_for_cell

        env = PortfolioEnv(features, cfg)
        obs_dims = {k: int(np.prod(sp.shape))
                    for k, sp in env.observation_space.spaces.items()}
        agent = ExoActorCritic(obs_dims, features.prices.shape[1] + 1,
                               model_cfg_for_cell(cfg.cell, cfg.model),
                               n_assets=features.prices.shape[1],
                               price_window=cfg.data.price_window)
        agent.load_state_dict(torch.load(run_dir / "model.pt",
                                         weights_only=True))
        agent.eval()
        return lambda s, e, c: evaluate_agent_window(agent, features, c, s, e)
    return None


def stage_report(cfg0: Config, features, folds, results: Path):
    from exo_portfolio.eval.diagnostics import diagnostics_report
    from exo_portfolio.eval.regimes import (label_stressed_vix,
                                            regime_conditional_metrics)
    from exo_portfolio.eval.stats import compare_cells, seed_fold_dispersion
    from exo_portfolio.run_grid import aggregate, summary_table

    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    vix = fetch_single(cfg0.data.vix_ticker, cfg0.data.start, cfg0.data.end,
                       cfg0.data.cache_dir, "vix")
    stressed = label_stressed_vix(vix)

    df = aggregate(results)
    if df.empty:
        print("[report] no grid runs found — run the grid stage first")
        return
    summary_table(df).to_csv(tables / "grid_summary_test.csv")

    # paired stats (needs >=3 matched pairs)
    n_pairs = df[df["split"] == "test"].groupby("cell").size().min()
    if n_pairs >= 3:
        compare_cells(df).to_csv(tables / "cell_comparisons_holm.csv", index=False)
        seed_fold_dispersion(df).to_csv(tables / "dispersion.csv")
    else:
        print(f"[report] only {n_pairs} matched pairs — stats need >=3, skipped")

    # regime-conditional + diagnostics per run; cost sensitivity (J.5)
    regime_rows, diag_rows, betas, cost_rows, responses = [], [], [], [], {}
    for run_dir in sorted(results.glob("cell*_seed*_fold*")):
        s_path = run_dir / "series_test.csv"
        if not s_path.exists():
            continue
        run_cfg = Config.from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
        s = pd.read_csv(s_path, parse_dates=["date"]).set_index("date")
        rt = regime_conditional_metrics(s["log_return"], s["turnover"], stressed)
        for regime in ("pooled", "calm", "stressed"):
            regime_rows.append({"run_id": run_dir.name, "cell": run_cfg.cell,
                                "seed": run_cfg.seed, "fold": run_cfg.fold,
                                "regime": regime, **rt[regime]})
        if "exposure" in s.columns:
            d = diagnostics_report(s["exposure"], vix)
            diag_rows.append({"run_id": run_dir.name, "cell": run_cfg.cell,
                              **d["exposure_vix"],
                              "ir_n_spikes": d["impulse_response"]["n_spikes"],
                              "ir_cum10d": d["impulse_response"]["cumulative_10d"]})
            if run_cfg.seed == 0:
                betas.append({"cell": run_cfg.cell,
                              "beta": d["exposure_vix"]["beta"],
                              "se": abs(d["exposure_vix"]["beta"]) /
                                    max(np.sqrt(d["exposure_vix"]["n"]), 1)})
                responses[run_cfg.cell] = d["impulse_response"]

        # J.5 cost sensitivity: re-evaluate the saved policy on its fold's
        # test window at three cost levels
        policy = _load_policy(run_dir, features, run_cfg)
        if policy is not None:
            fold = folds[run_cfg.fold]
            for cost in COSTS_J5:
                c = Config.from_dict(run_cfg.to_dict())
                c.env.transaction_cost = cost
                out = policy(int(fold.test_idx[0]), int(fold.test_idx[-1]), c)
                cost_rows.append({"run_id": run_dir.name, "cell": run_cfg.cell,
                                  "seed": run_cfg.seed, "fold": run_cfg.fold,
                                  "transaction_cost": cost, **out["metrics"]})

    pd.DataFrame(regime_rows).to_csv(tables / "regime_conditional.csv", index=False)
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(tables / "diagnostics.csv", index=False)
    if cost_rows:
        pd.DataFrame(cost_rows).to_csv(tables / "cost_sensitivity.csv", index=False)
    (tables / "_diag_cache.json").write_text(json.dumps(
        {"betas": betas, "responses": responses}, default=float))
    print(f"[report] tables -> {tables} "
          f"({len(regime_rows)} regime rows, {len(cost_rows)} cost rows)")


def stage_figures(cfg0: Config, results: Path):
    from exo_portfolio.eval import figures as F
    from exo_portfolio.eval.regimes import label_stressed_vix

    figs = results / "figures"
    vix = fetch_single(cfg0.data.vix_ticker, cfg0.data.start, cfg0.data.end,
                       cfg0.data.cache_dir, "vix")
    made = []
    made.append(F.fig_learning_curves(results, figs / "learning_curves.png"))
    made.append(F.fig_synthetic_shift(results / "tables" / "synthetic_shift.csv",
                                      figs / "synthetic_shift.png"))
    made.append(F.fig_equity_curves(results, figs / "equity_curves_fold0.png",
                                    stressed=label_stressed_vix(vix)))
    cache = results / "tables" / "_diag_cache.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        made.append(F.fig_exposure_vix_betas(pd.DataFrame(d["betas"]),
                                             figs / "exposure_vix_betas.png"))
        made.append(F.fig_impulse_response(d["responses"],
                                           figs / "impulse_response.png"))
    made = [m for m in made if m]
    print(f"[figures] {len(made)} figures -> {figs}")
    return made


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_STAGES = ("data", "grid", "classical", "synthetic", "report", "figures")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--budget", choices=list(BUDGETS), default="smoke")
    ap.add_argument("--stages", nargs="+", default=list(ALL_STAGES),
                    choices=list(ALL_STAGES))
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)

    budget = BUDGETS[args.budget]
    results = Path(args.results)
    cfg0 = Config.from_yaml(*args.config)
    seed_everything(cfg0.seed)

    bundle, features, folds = stage_data(cfg0)      # always needed downstream
    if "grid" in args.stages:
        stage_grid(args.config, budget, results)
    if "classical" in args.stages:
        stage_classical(cfg0, features, folds, results)
    if "synthetic" in args.stages:
        stage_synthetic(budget, results)
    if "report" in args.stages:
        stage_report(cfg0, features, folds, results)
    if "figures" in args.stages:
        stage_figures(cfg0, results)
    print("[reproduce] done")


if __name__ == "__main__":
    main(sys.argv[1:])
