"""Fold-level paired tests (supervisor review, 2026-10).

The 10 seeds in a fold share one test period, and fold-to-fold variation is
much larger than seed-to-seed variation, so the 50 (seed, fold) runs of a cell
are not 50 independent results. Every comparison here first averages each
strategy over seeds within a fold and then pairs the five fold means
(n = 5). With n = 5 a Wilcoxon signed-rank test cannot go below p = 0.0625,
so the paired t-test is used throughout; Holm correction is applied within
each family.

Writes results/tables/fold_level/*.csv. Strategies that are missing (e.g.
cell1n/cell3n before their runs exist) are skipped.

    python -m exo_portfolio.eval.fold_level_stats
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path("results")
OUT = RES / "tables" / "fold_level"
METRICS = ["sharpe", "sortino", "cum_log_return", "max_drawdown"]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def paired(a: pd.Series, b: pd.Series):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = j.a - j.b
    if len(d) < 2 or np.allclose(d, d.iloc[0]):
        p = 0.0 if len(d) and abs(d.mean()) > 0 else 1.0
    else:
        p = float(stats.ttest_rel(j.a, j.b).pvalue)
    return float(d.mean()), p, int(len(d)), int((d > 0).sum())


def family(fm: pd.DataFrame, target: str, others: list[str], metrics=METRICS):
    rows = []
    for o in others:
        if o not in fm.index.get_level_values(0):
            continue
        for m in metrics:
            diff, p, n, npos = paired(fm.loc[target][m], fm.loc[o][m])
            rows.append({"strategy": target, "vs": o, "metric": m, "mean_diff": diff,
                         "pvalue": p, "n_folds": n, "folds_target_higher": npos})
    df = pd.DataFrame(rows)
    if len(df):
        df["pvalue_holm"] = holm(df.pvalue)
    return df


def load_runs() -> pd.DataFrame:
    s = pd.read_csv(RES / "summary.csv")
    s = s[s.split == "test"][["cell", "seed", "fold"] + METRICS + ["mean_turnover"]]
    extra = []
    for c in ("cell1n", "cell3n"):
        for d in sorted(RES.glob(f"{c}_seed*_fold*")):
            mj = d / "metrics.json"
            if mj.exists():
                import json
                m = json.loads(mj.read_text())["test"]
                seed = int(d.name.split("seed")[1].split("_")[0]); fold = int(d.name[-1])
                extra.append({"cell": c, "seed": seed, "fold": fold,
                              **{k: m[k] for k in METRICS + ["mean_turnover"]}})
    u = RES / "untrained" / "untrained_runs.csv"
    if u.exists():
        uu = pd.read_csv(u)[["cell", "seed", "fold"] + METRICS + ["mean_turnover"]]
        extra += uu.to_dict("records")
    return pd.concat([s, pd.DataFrame(extra)], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    fm = runs.groupby(["cell", "fold"])[METRICS + ["mean_turnover"]].mean()

    # summary: mean and spread of the five fold means
    summ = fm.groupby(level=0).agg(["mean", "std"])
    summ.columns = [f"{a}_{b}_over_folds" for a, b in summ.columns]
    summ.to_csv(OUT / "summary_fold_means.csv")

    cells = ["cell0", "cell0c", "cell0cw", "cell1", "cell2", "cell3"]
    family(fm, "cell4", cells).to_csv(OUT / "cell4_vs_cells.csv", index=False)
    pd.concat([family(fm, "cell0c", ["cell0"]),
               family(fm, "cell0cw", ["cell0c"])]).to_csv(OUT / "controls.csv", index=False)
    base = ["equal_weight", "vol_overlay", "mean_variance", "inverse_vol",
            "min_variance", "max_sharpe", "momentum", "buy_and_hold_index",
            "random_weight", "random_constant_mix", "random_action"]
    family(fm, "cell4", base).to_csv(OUT / "cell4_vs_baselines.csv", index=False)

    # trained vs untrained, and untrained vs equal weight
    rows = []
    for c in ["cell0", "cell0c", "cell0cw", "cell1", "cell2", "cell3", "cell4"]:
        if f"untrained_{c}" in fm.index.get_level_values(0):
            rows.append(family(fm, c, [f"untrained_{c}"], ["sharpe", "cum_log_return"]))
    if rows:
        t = pd.concat(rows); t["pvalue_holm"] = holm(t.pvalue)
        t.to_csv(OUT / "trained_vs_untrained.csv", index=False)
        family(fm, "untrained_cell4", ["equal_weight"]).to_csv(
            OUT / "untrained_vs_equal_weight.csv", index=False)

    # normalised flat agents (when present)
    # One family of 16 tests: cell4 vs each normalised agent, and each
    # normalised agent vs its unnormalised counterpart (4 metrics each).
    norm = []
    for c, ref in (("cell1n", "cell1"), ("cell3n", "cell3")):
        if c in fm.index.get_level_values(0):
            norm.append(family(fm, "cell4", [c]))
            norm.append(family(fm, c, [ref]))
    if norm:
        t = pd.concat(norm); t["pvalue_holm"] = holm(t.pvalue)
        t.to_csv(OUT / "normalised_flat.csv", index=False)

    # calm / stressed regimes
    r = pd.read_csv(RES / "tables" / "regime_conditional.csv")
    rows = []
    for reg in ("calm", "stressed"):
        rf = r[r.regime == reg].groupby(["cell", "fold"])[["sharpe", "cum_log_return"]].mean()
        f = family(rf, "cell4", ["cell0", "cell0c", "cell0cw", "cell1", "cell2", "equal_weight"],
                   ["sharpe"])
        f.insert(0, "regime", reg); rows.append(f)
        rf.groupby(level=0).mean().add_prefix(f"{reg}_").to_csv(OUT / f"regime_means_{reg}.csv")
    reg = pd.concat(rows); reg["pvalue_holm"] = holm(reg.pvalue)
    reg.to_csv(OUT / "regimes.csv", index=False)

    # test-time feature ablation: change per model averaged over draws, then
    # averaged over seeds within each fold, one-sample t-test over 5 folds
    ab = pd.read_csv(RES / "tables" / "test_time_ablation_runs.csv")
    intact = ab[ab["mode"] == "intact"].set_index(["cell", "seed", "fold"])
    rows = []
    for (cell, mode, grp), g in ab[ab["mode"] != "intact"].groupby(["cell", "mode", "group"]):
        g = g.groupby(["seed", "fold"])[["sharpe", "cum_log_return", "max_drawdown",
                                         "mean_exposure", "policy_shift"]].mean()
        base_ = intact.loc[cell][["sharpe", "cum_log_return", "max_drawdown", "mean_exposure"]]
        for m in ["sharpe", "cum_log_return", "max_drawdown", "mean_exposure", "policy_shift"]:
            d = g[m] - (base_[m] if m != "policy_shift" else 0.0)
            dfold = d.groupby(level="fold").mean()
            p = (1.0 if np.allclose(dfold, 0) else float(stats.ttest_1samp(dfold, 0).pvalue))
            rows.append({"cell": cell, "mode": mode, "group": grp, "metric": m,
                         "mean_diff": float(dfold.mean()), "pvalue": p, "n_folds": len(dfold)})
    ab_t = pd.DataFrame(rows)
    ab_t["pvalue_holm"] = ab_t.groupby(["cell", "mode", "metric"]).pvalue.transform(holm)
    ab_t.to_csv(OUT / "test_time_ablation.csv", index=False)
    print("wrote", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
