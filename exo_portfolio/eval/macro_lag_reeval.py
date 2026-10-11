"""Re-evaluate the TRAINED grid models with correctly lagged macro inputs.

Background: until 2026-10-11 the macro features reached the actor too early.
FRED stamps a monthly value with the first day of its reference month and the
lag was added to that stamp; on top of that CPI_YOY got a 0-day lag because the
lag was looked up by the derived column name (the config is keyed by
CPIAUCSL). See DESIGN_DECISIONS.md §3.4 / §14.1. data/align.py is now fixed.

Nothing is retrained here. Each saved model is rolled out on its fold's test
window twice, on the SAME dates and folds the grid used:
  legacy  the exact features the models were trained and evaluated with
          (rebuilt with the old alignment, below). Must reproduce every run's
          saved test metrics -- that is the validation.
  fixed   identical, except the actor's two macro columns are replaced by the
          correctly lagged ones (data/align.apply_publication_lag).
Only the actor's macro inputs change; prices, the other exo columns and the
fold windows are untouched. The critic is not used at test time.

This bounds the TEST-TIME dependence on leaked macro values. It does not undo
any training-time effect: the policies were still shaped by future macro data.

    python -m exo_portfolio.eval.macro_lag_reeval --config configs/base.yaml \\
        configs/universe_djia30.yaml --cells cell1 --out OUT_DIR/cell1.csv
    python -m exo_portfolio.eval.macro_lag_reeval --summarize OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from exo_portfolio.config import Config
from exo_portfolio.data.align import (FeatureSet, _ffill_onto,
                                      apply_publication_lag, transform_macro)

MACRO_COLS = ("FEDFUNDS", "CPI_YOY")
METRICS = ("sharpe", "sortino", "cum_log_return", "max_drawdown",
           "mean_turnover", "mean_exposure")


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

def legacy_macro_lagged(macro: pd.DataFrame, lags: dict,
                        calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """The pre-fix alignment, reproduced verbatim so the trained models can be
    re-run on exactly the inputs they saw: CPI YoY with padded gaps, lag added
    to the FRED (start-of-month) stamp, lag looked up by DERIVED column name."""
    out = macro.copy()
    out["CPI_YOY"] = out["CPIAUCSL"].ffill().pct_change(12)
    out = out.drop(columns=["CPIAUCSL"])
    cols = {}
    for col in out.columns:
        s = out[col].dropna()
        s = pd.Series(s.values, index=s.index + pd.Timedelta(days=lags.get(col, 0)))
        cols[col] = _ffill_onto(s, calendar)
    return pd.DataFrame(cols, index=calendar)


def build_feature_pair(cfg: Config) -> tuple[FeatureSet, FeatureSet]:
    """(legacy, fixed) feature sets on the legacy trading-date index."""
    from exo_portfolio.data.align import build_features
    from exo_portfolio.data.loaders import load_raw_bundle

    raw = load_raw_bundle(cfg)
    fixed_full = build_features(**raw, cfg=cfg)          # fixed alignment

    # Legacy calendar: the old code's first valid day was set by CPI_YOY with
    # a 0-day lag. Rebuild the legacy exo_actor on the full price calendar,
    # then trim exactly as the old build did.
    cal = raw["prices"].index
    leg_macro = legacy_macro_lagged(raw["macro"],
                                    cfg.data.macro_publication_lag_days, cal)
    exo_cols = [c for c in fixed_full.exo_actor.columns
                if not c.endswith("_lagged")]
    # non-macro exo columns are unaffected by the fix: take them from a build
    # on the full calendar (reindex fixed_full, whose rows start later, would
    # lose early rows), so rebuild them via build_features' own code path:
    nonmacro = _nonmacro_exo(raw, cfg)[exo_cols]
    legacy_exo = pd.concat([nonmacro, leg_macro.add_suffix("_lagged")], axis=1)
    first = legacy_exo.dropna().index.min()
    dates = cal[cal >= first]
    assert dates[0] == pd.Timestamp("2011-01-03") and len(dates) == 3894, \
        "legacy feature calendar does not match the one the grid was run on"

    legacy = FeatureSet(dates=dates, prices=raw["prices"].loc[dates],
                        exo_actor=legacy_exo.loc[dates],
                        exo_critic_extra=pd.DataFrame(
                            np.zeros((len(dates), 2)), index=dates,
                            columns=[f"{c}_unlagged" for c in MACRO_COLS]))

    fixed_macro = apply_publication_lag(transform_macro(raw["macro"]),
                                        cfg.data.macro_publication_lag_days, cal)
    fixed_exo = legacy.exo_actor.copy()
    for c in MACRO_COLS:
        fixed_exo[f"{c}_lagged"] = fixed_macro[c].loc[dates].values
    # Rows before the fixed macro exists (Jan-Feb 2011, training region only)
    # take the first published value; test windows start in 2018 and are
    # asserted NaN-free below before any fill.
    fixed_exo = fixed_exo.bfill()
    fixed = FeatureSet(dates=dates, prices=legacy.prices, exo_actor=fixed_exo,
                       exo_critic_extra=legacy.exo_critic_extra)
    return legacy, fixed


def _nonmacro_exo(raw: dict, cfg: Config) -> pd.DataFrame:
    """Asset-window + index + VIX columns on the full price calendar, computed
    exactly as data/align.build_features does (that code is unchanged by the
    fix; only the macro block changed)."""
    from numpy.lib.stride_tricks import sliding_window_view

    prices, W = raw["prices"], cfg.data.price_window
    cal = prices.index
    lr = np.log(prices.values[1:] / prices.values[:-1])
    win = sliding_window_view(lr, W, axis=0)
    cols = [f"{t}_lr_t-{W - 1 - k}" for t in prices.columns for k in range(W)]
    asset = pd.DataFrame(win.reshape(len(cal) - W, -1), index=cal[W:],
                         columns=cols).reindex(cal)
    idx = _ffill_onto(raw["index"].dropna(), cal)
    idx_lr = np.log(idx / idx.shift(1)).rename("gspc_lr")
    vix = _ffill_onto(raw["vix"].dropna(), cal).rename("vix")
    return pd.concat([asset, idx_lr, vix], axis=1)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def rollout_series(env, act, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    rhos, taus, weights = [], [], []
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(obs))
        rhos.append(info["step_log_return"])
        taus.append(info["turnover"])
        weights.append(np.asarray(info["weights"], float))
        done = terminated or truncated
    return {"rho": np.array(rhos), "tau": np.array(taus),
            "w": np.stack(weights)}


def make_actor(run_dir: Path, cfg: Config, features: FeatureSet):
    """-> run(features, start, end) for a saved SB3 or custom run (same model
    loading as eval/test_time_ablation.load_actor, keeping the full series)."""
    if (run_dir / "model.zip").exists():
        from stable_baselines3 import PPO

        from exo_portfolio.baselines.sb3_baselines import make_env

        model = PPO.load(run_dir / "model.zip", device="cpu")
        return lambda f, s, e: rollout_series(
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

    return lambda f, s, e: rollout_series(PortfolioEnv(f, cfg, s, e), act,
                                          cfg.seed)


def evaluate_run(run_dir: Path, legacy: FeatureSet, fixed: FeatureSet,
                 folds) -> tuple[list[dict], dict]:
    from exo_portfolio.eval.metrics import summarize

    cfg = Config.from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    fold = folds[cfg.fold]
    s, e = int(fold.test_idx[0]), int(fold.test_idx[-1])
    assert not fixed.exo_actor.iloc[s - 1:e + 1].isna().any().any()
    run = make_actor(run_dir, cfg, legacy)
    saved = json.loads((run_dir / "metrics.json").read_text())["test"]
    key = {"run_id": run_dir.name, "cell": cfg.cell, "seed": cfg.seed,
           "fold": cfg.fold}
    rows, series = [], {}
    base_w = None
    for name, feats in (("legacy", legacy), ("fixed", fixed)):
        out = run(feats, s, e)
        m = summarize(out["rho"], out["tau"])
        m["mean_exposure"] = float(np.mean(1.0 - out["w"][:, 0]))
        if name == "legacy":
            base_w = out["w"]
            m["max_abs_err_vs_saved"] = max(
                abs(m[k] - saved[k]) for k in ("sharpe", "cum_log_return",
                                               "max_drawdown", "mean_turnover"))
        m["policy_shift"] = float(0.5 * np.abs(out["w"] - base_w).sum(1).mean())
        rows.append({**key, "features": name, **m})
        series[name] = pd.DataFrame(
            {"log_return": out["rho"], "turnover": out["tau"],
             "exposure": 1.0 - out["w"][:, 0]},
            index=legacy.dates[s + 1:s + 1 + len(out["rho"])])
    return rows, series


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_reeval(df: pd.DataFrame, results: Path) -> dict[str, pd.DataFrame]:
    from exo_portfolio.eval.stats import compare_cells, paired_test
    from exo_portfolio.run_grid import aggregate

    leg = df[df.features == "legacy"].set_index("run_id")
    fix = df[df.features == "fixed"].set_index("run_id")

    # 1) per cell: legacy vs fixed, paired by model
    rows = []
    for cell, g in fix.groupby("cell"):
        b = leg.loc[g.index]
        for metric in (*METRICS, "policy_shift"):
            a_, b_ = g[metric].values, b[metric].values
            d = a_ - b_
            res = (paired_test(a_, b_) if metric != "policy_shift"
                   else {"pvalue": np.nan})
            rows.append({"cell": cell, "metric": metric, "n_models": len(d),
                         "legacy_mean": b_.mean(), "fixed_mean": a_.mean(),
                         "mean_diff": d.mean(),
                         "ci95": 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                         "pvalue": res["pvalue"]})
    per_cell = pd.DataFrame(rows)

    # 2) the headline comparisons, re-run with the fixed cells. Cells without
    # macro (cell0, cell0c, cell0cw, baselines) keep their saved metrics.
    agg = aggregate(results)
    agg = agg[agg.split == "test"]
    fixed_rows = fix.reset_index().assign(split="test")
    redone = set(fixed_rows.cell)
    merged = pd.concat([agg[~agg.cell.isin(redone)],
                        fixed_rows[[c for c in agg.columns if c in fixed_rows]]],
                       ignore_index=True)
    vs_cells = compare_cells(merged, target="cell4",
                             baselines=("cell0", "cell0c", "cell0cw", "cell1",
                                        "cell2", "cell3"))
    vs_base = compare_cells(merged, target="cell4",
                            baselines=("equal_weight", "vol_overlay",
                                       "inverse_vol", "min_variance"),
                            pair_on=("fold",))
    return {"macro_lag_reeval_by_cell": per_cell,
            "macro_lag_reeval_cell_comparisons_holm": vs_cells,
            "macro_lag_reeval_vs_baselines_holm": vs_base}


def regime_table(series_dir: Path, results: Path, cfg: Config) -> pd.DataFrame:
    """Regime-conditional Sharpe (fixed features) for re-evaluated cells, plus
    the saved series for the cells that do not see macro."""
    from exo_portfolio.data.loaders import fetch_single
    from exo_portfolio.eval.regimes import (label_stressed_vix,
                                            regime_conditional_metrics)

    vix = fetch_single(cfg.data.vix_ticker, cfg.data.start, cfg.data.end,
                       cfg.data.cache_dir, "vix")
    stressed = label_stressed_vix(vix)
    rows = []
    paths = [(p, "fixed") for p in sorted(series_dir.glob("*__fixed.csv"))]
    redone = {p.name.split("_seed")[0] for p, _ in paths}
    for c in ("cell0", "cell0c", "cell0cw", "equal_weight"):
        if c not in redone:
            paths += [(p / "series_test.csv", "saved")
                      for p in sorted(results.glob(f"{c}_seed*_fold*"))]
    for p, src in paths:
        name = p.parent.name if src == "saved" else p.name.split("__")[0]
        cell = name.split("_seed")[0]
        s = pd.read_csv(p, parse_dates=[0], index_col=0)
        rt = regime_conditional_metrics(s["log_return"], s["turnover"], stressed)
        for regime in ("pooled", "calm", "stressed"):
            rows.append({"run_id": name, "cell": cell, "regime": regime,
                         "source": src, **rt[regime]})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/base.yaml",
                                                    "configs/universe_djia30.yaml"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--cells", nargs="+", default=["cell1", "cell2", "cell3", "cell4"])
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--out", help="per-run CSV for this chunk")
    ap.add_argument("--summarize", metavar="DIR")
    args = ap.parse_args(argv)
    results = Path(args.results)
    cfg0 = Config.from_yaml(*args.config)

    if args.summarize:
        d = Path(args.summarize)
        df = pd.concat([pd.read_csv(p) for p in sorted(d.glob("*.csv"))])
        tables = results / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        df.to_csv(tables / "macro_lag_reeval_runs.csv", index=False)
        for name, t in summarize_reeval(df, results).items():
            t.to_csv(tables / f"{name}.csv", index=False)
        regime_table(d / "series", results, cfg0).to_csv(
            tables / "macro_lag_reeval_regimes.csv", index=False)
        print(f"{len(df)} rollouts -> {tables}/macro_lag_reeval_*.csv")
        return

    from exo_portfolio.data.splits import rolling_origin_folds

    legacy, fixed = build_feature_pair(cfg0)
    folds = rolling_origin_folds(legacy.dates, n_folds=5)
    out = Path(args.out)
    (out.parent / "series").mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in args.cells:
        for fold in args.folds:
            for run_dir in sorted(results.glob(f"{cell}_seed*_fold{fold}")):
                r, series = evaluate_run(run_dir, legacy, fixed, folds)
                rows += r
                series["fixed"].to_csv(out.parent / "series" /
                                       f"{run_dir.name}__fixed.csv")
                print(f"[reeval] {run_dir.name} saved-err="
                      f"{r[0]['max_abs_err_vs_saved']:.2e} "
                      f"dSharpe={r[1]['sharpe'] - r[0]['sharpe']:+.3f}",
                      flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
