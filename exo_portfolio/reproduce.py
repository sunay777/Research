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


def _write_baseline_run(results: Path, cfg0: Config, name: str, seed: int,
                        fold: int, series: dict, meta: dict) -> None:
    """Emit a baseline run under the standard {name}_seed{seed}_fold{fold} dir
    so it flows through stage_report (regime-conditional + J.5) and
    run_grid.aggregate exactly like an RL cell (Manual M10). The baseline name
    goes in the `cell` column, as stage_classical already did."""
    d = results / f"{name}_seed{seed}_fold{fold}"
    d.mkdir(parents=True, exist_ok=True)
    cfg_dict = cfg0.to_dict()
    cfg_dict.update(cell=name, seed=seed, fold=fold)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
    (d / "baseline.json").write_text(json.dumps(meta))
    (d / "metrics.json").write_text(json.dumps({"test": series["metrics"]}, indent=2))
    pd.DataFrame({
        "date": series["dates"],
        "log_return": series["log_returns"],
        "turnover": series["turnover"],
        "exposure": series["exposure"],
    }).to_csv(d / "series_test.csv", index=False)


def stage_classical(cfg: Config, features, folds, results: Path,
                    budget: dict | None = None) -> pd.DataFrame:
    """Every non-RL baseline (Manual K + M10) per fold, emitted as run-dirs with
    a per-day series_test.csv so they reach the same tables/stats as the RL
    agents. Deterministic baselines get one run per fold (paired by FOLD; no
    across-seed spread); the random baselines get one per (seed, fold) so they
    carry a genuine across-seed distribution."""
    from exo_portfolio.baselines.classical import (buy_and_hold_index,
                                                   deterministic_target_baselines,
                                                   random_policy_in_env,
                                                   random_target_baselines,
                                                   simulate_baseline_series)
    from exo_portfolio.eval.metrics import summarize

    budget = budget or BUDGETS["smoke"]
    seeds = budget["seeds"]
    index = fetch_single(cfg.data.index_ticker, cfg.data.start, cfg.data.end,
                         cfg.data.cache_dir, "gspc")
    rows = []
    for fold_id in budget["folds"]:                # align with the grid's folds
        fold = folds[fold_id]
        prices = features.prices.iloc[fold.test_idx]
        t0, t1 = int(fold.test_idx[0]), int(fold.test_idx[-1])

        # deterministic simplex baselines (classical + M10 traditional), seed 0
        det = deterministic_target_baselines(prices, cfg)
        det_series = {name: simulate_baseline_series(
            prices, tg, cfg.env.transaction_cost) for name, tg in det.items()}
        # buy-and-hold the index: cost-invariant, fully invested
        bh = buy_and_hold_index(index.loc[prices.index])
        bh_row = summarize(bh["log_returns"], bh["turnover"])
        bh_row["final_value"] = bh["final_value"]
        det_series["buy_and_hold_index"] = {
            "metrics": bh_row, "log_returns": bh["log_returns"],
            "turnover": bh["turnover"], "dates": list(prices.index[1:]),
            "exposure": np.ones(len(bh["log_returns"]))}

        for name, ser in det_series.items():
            kind = "index" if name == "buy_and_hold_index" else "deterministic"
            _write_baseline_run(results, cfg, name, 0, fold_id, ser,
                                {"name": name, "kind": kind, "seed": 0})
            rows.append({"cell": name, "fold": fold_id, "split": "test",
                         **ser["metrics"]})

        # random baselines: one run per (seed, fold) — genuine across-seed spread
        for seed in seeds:
            for name, tg in random_target_baselines(prices, seed).items():
                ser = simulate_baseline_series(prices, tg, cfg.env.transaction_cost)
                _write_baseline_run(results, cfg, name, seed, fold_id, ser,
                                    {"name": name, "kind": "random_target", "seed": seed})
            ser = random_policy_in_env(features, cfg, t0, t1, seed=seed)
            _write_baseline_run(results, cfg, "random_action", seed, fold_id, ser,
                                {"name": "random_action", "kind": "random_action", "seed": seed})

    df = pd.DataFrame(rows)
    out = results / "tables" / "classical.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[classical] {len(df)} deterministic rows -> {out}; "
          f"baseline run-dirs emitted (random seeds: {list(seeds)})")
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


def _baseline_policy(run_dir: Path, features):
    """Re-evaluate callable for a non-RL baseline run — recomputes its target
    rows on the fold window at the requested cost, so baselines get J.5 cost
    sensitivity too (buy-and-hold is cost-invariant: its stored series is
    reused)."""
    from exo_portfolio.baselines.classical import (
        deterministic_target_baselines, random_policy_in_env,
        random_target_baselines, simulate_baseline_series)
    from exo_portfolio.eval.metrics import summarize

    meta = json.loads((run_dir / "baseline.json").read_text())
    name, kind, seed = meta["name"], meta["kind"], int(meta.get("seed", 0))

    def evaluate(s, e, c):
        prices = features.prices.iloc[s:e + 1]
        if kind == "deterministic":
            tg = deterministic_target_baselines(prices, c)[name]
            return simulate_baseline_series(prices, tg, c.env.transaction_cost)
        if kind == "random_target":
            tg = random_target_baselines(prices, seed)[name]
            return simulate_baseline_series(prices, tg, c.env.transaction_cost)
        if kind == "random_action":
            return random_policy_in_env(features, c, s, e, seed=seed)
        # index baseline (buy-and-hold): cost-invariant, reuse stored series
        sdf = pd.read_csv(run_dir / "series_test.csv", parse_dates=["date"])
        row = summarize(sdf["log_return"].values, sdf["turnover"].values)
        row["final_value"] = float(np.exp(sdf["log_return"].sum()))
        return {"metrics": row, "log_returns": sdf["log_return"].values,
                "turnover": sdf["turnover"].values, "dates": list(sdf["date"]),
                "exposure": sdf["exposure"].values}

    return evaluate


def _load_policy(run_dir: Path, features, cfg: Config):
    """Rebuild the evaluate(start, end, cfg) callable for a saved run."""
    if (run_dir / "baseline.json").exists():                # non-RL baselines
        return _baseline_policy(run_dir, features)
    if (run_dir / "model.zip").exists():                    # SB3 cells
        from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

        algo = {"ppo": PPO, "sac": SAC, "a2c": A2C,
                "ddpg": DDPG, "td3": TD3}[cfg.train.algo]
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
    from exo_portfolio.eval.diagnostics import (diagnostics_report,
                                                pool_impulse_by_event,
                                                summarize_exposure_vix)
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

    # paired stats across the architecture grid (matched by seed+fold)
    # (cell0c = cell0 on the custom PPO; absent until its runs exist, and
    # compare_cells skips any baseline with no results, so this is a no-op then)
    grid_cells = [c for c in ("cell0", "cell0c", "cell1", "cell2", "cell3", "cell4")
                  if c in df["cell"].values]
    # a partially-synced cell0c must not gate the main grid's stats
    gate_cells = [c for c in grid_cells if c != "cell0c"]
    n_pairs = (df[df["split"] == "test"]
               .groupby("cell").size().reindex(gate_cells).min()
               if gate_cells else 0)
    if n_pairs >= 3:
        compare_cells(df, baselines=("cell0", "cell0c", "cell1", "cell2", "cell3")
                      ).to_csv(tables / "cell_comparisons_holm.csv", index=False)
        seed_fold_dispersion(df).to_csv(tables / "dispersion.csv")
        # implementation control: does switching SB3 -> custom PPO alone move cell0?
        if "cell0c" in df["cell"].values and "cell0" in df["cell"].values:
            vs_impl = compare_cells(df, target="cell0c", baselines=("cell0",))
            if len(vs_impl):
                vs_impl.to_csv(tables / "cell_comparisons_vs_cell0c_holm.csv",
                               index=False)
    else:
        print(f"[report] only {n_pairs} matched grid pairs — stats need >=3, skipped")

    # cell4 vs the non-RL baselines, paired by FOLD (deterministic baselines
    # have no across-seed spread — Manual M10 / K credibility bar)
    baseline_cells = tuple(c for c in ("equal_weight", "vol_overlay",
                                       "mean_variance", "inverse_vol",
                                       "min_variance", "max_sharpe", "momentum",
                                       "buy_and_hold_index", "random_weight",
                                       "random_buy_and_hold", "random_action")
                           if c in df["cell"].values)
    n_folds = df[df["split"] == "test"]["fold"].nunique()
    if "cell4" in df["cell"].values and baseline_cells and n_folds >= 3:
        vs_base = compare_cells(df, target="cell4", baselines=baseline_cells,
                                pair_on=("fold",))
        if len(vs_base):
            vs_base.to_csv(tables / "cell4_vs_baselines_holm.csv", index=False)

    # regime-conditional + diagnostics per run; cost sensitivity (J.5).
    # Glob is `*_seed*_fold*` (not `cell*_...`) so the non-RL baseline run-dirs
    # emitted by stage_classical flow through here too (Manual M10).
    regime_rows, diag_rows, betas, cost_rows, responses = [], [], [], [], {}
    for run_dir in sorted(results.glob("*_seed*_fold*")):
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
                              "seed": run_cfg.seed, "fold": run_cfg.fold,
                              **d["exposure_vix"],
                              "ir_n_spikes": d["impulse_response"]["n_spikes"],
                              "ir_cum10d": d["impulse_response"]["cumulative_10d"]})
            # every seed x fold; pooled after the loop (seeds averaged first —
            # they are replicate policies on the same market data)
            betas.append({"cell": run_cfg.cell, "seed": run_cfg.seed,
                          "fold": run_cfg.fold,
                          "beta": d["exposure_vix"]["beta"]})
            responses.setdefault(run_cfg.cell, []).append(
                d["impulse_response"])

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
        diag_df = pd.DataFrame(diag_rows)
        diag_df.to_csv(tables / "diagnostics.csv", index=False)
        summarize_exposure_vix(diag_df).to_csv(tables / "mechanism_summary.csv")
    if cost_rows:
        pd.DataFrame(cost_rows).to_csv(tables / "cost_sensitivity.csv", index=False)
    # all seeds x folds, clustered by spike event
    responses = {cell: pool_impulse_by_event(rs)
                 for cell, rs in responses.items()}
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
        stage_classical(cfg0, features, folds, results, budget)
    if "synthetic" in args.stages:
        stage_synthetic(budget, results)
    if "report" in args.stages:
        stage_report(cfg0, features, folds, results)
    if "figures" in args.stages:
        stage_figures(cfg0, results)
    print("[reproduce] done")


if __name__ == "__main__":
    main(sys.argv[1:])
