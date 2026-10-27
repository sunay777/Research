"""Publication-dated macro re-evaluation for the normalised flat agents.

Same check as eval.macro_lag_reeval, for cell1n / cell3n: each saved policy is
rolled out on its test window twice, with the grid's macro features (as
trained) and with publication-dated macro features. Both feature sets are
normalised with the statistics the run was trained with (training window of
the legacy features), so only the macro timing differs.

    python -m exo_portfolio.eval.norm_flat_reeval --cells cell1n --out results/macro_lag_reeval_norm/cell1n.csv
    python -m exo_portfolio.eval.norm_flat_reeval --summarize results/macro_lag_reeval_norm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from exo_portfolio.config import Config
from exo_portfolio.data.align import FeatureSet
from exo_portfolio.data.splits import rolling_origin_folds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/base.yaml",
                                                    "configs/universe_djia30.yaml"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--cells", nargs="+", default=["cell1n", "cell3n"])
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--out")
    ap.add_argument("--summarize", metavar="DIR")
    a = ap.parse_args(argv)
    results = Path(a.results)

    if a.summarize:
        df = pd.concat([pd.read_csv(p) for p in sorted(Path(a.summarize).glob("*.csv"))])
        rows = []
        for cell, g in df.groupby("cell"):
            p = g.pivot_table(index="run_id", columns="features", values="sharpe")
            fm = g.groupby(["fold", "features"]).sharpe.mean().unstack()
            rows.append({"cell": cell, "n_runs": len(p),
                         "mean_sharpe_legacy": p.legacy.mean(),
                         "mean_sharpe_fixed": p.fixed.mean(),
                         "mean_diff": (p.fixed - p.legacy).mean(),
                         "max_abs_fold_diff": (fm.fixed - fm.legacy).abs().max(),
                         "max_err_vs_saved": g.max_abs_err_vs_saved.max()})
        out = pd.DataFrame(rows)
        (results / "tables").mkdir(parents=True, exist_ok=True)
        out.to_csv(results / "tables" / "macro_lag_reeval_norm.csv", index=False)
        print(out.to_string())
        return

    from exo_portfolio.eval.macro_lag_reeval import build_feature_pair, evaluate_run
    from exo_portfolio.experiments.norm_flat import grid_features, normalise

    cfg0 = Config.from_yaml(*a.config)
    grid = grid_features(cfg0)
    _, fixed_raw = build_feature_pair(cfg0)
    folds = rolling_origin_folds(grid.dates, n_folds=5)
    rows = []
    for fold in a.folds:
        tr = grid.dates[folds[fold].train_idx]
        leg_n, _ = normalise(grid, folds[fold].train_idx)
        mu = grid.exo_actor.loc[tr].mean()
        sd = grid.exo_actor.loc[tr].std(ddof=0).clip(lower=1e-8)
        fix_n = FeatureSet(dates=grid.dates, prices=grid.prices,
                           exo_actor=(fixed_raw.exo_actor - mu) / sd,
                           exo_critic_extra=leg_n.exo_critic_extra)
        for cell in a.cells:
            for run_dir in sorted(results.glob(f"{cell}_seed*_fold{fold}")):
                r, _ = evaluate_run(run_dir, leg_n, fix_n, folds)
                rows += r
                print(f"[reeval] {run_dir.name} err={r[0]['max_abs_err_vs_saved']:.1e} "
                      f"dSharpe={r[1]['sharpe'] - r[0]['sharpe']:+.4f}", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
