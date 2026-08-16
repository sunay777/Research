"""Paired significance tests + Holm correction (Manual J.3).

Per-fold, per-seed metric matrices: compare cell4 (or any target) against
each baseline with a PAIRED test — pairs share (seed, fold), so the market
path is identical and only the architecture differs.

Test choice: Shapiro normality check on the paired differences; normal ->
scipy ttest_rel, else the Wilcoxon signed-rank test. (The manual mentions
mannwhitneyu, but that is an unpaired test — Wilcoxon signed-rank is the
correct nonparametric analogue for paired samples; deviation documented.)

Multiple comparisons: Holm step-down over the baseline x metric family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

DEFAULT_METRICS = ("sharpe", "sortino", "cum_log_return", "max_drawdown")


def paired_test(a: np.ndarray, b: np.ndarray, alpha_normal: float = 0.05) -> dict:
    """Paired comparison of matched samples a vs b -> test name, stat, p.

    Positive mean_diff means a > b on average.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) >= 3
    diff = a - b
    if np.allclose(diff, 0):
        return {"test": "degenerate", "stat": 0.0, "pvalue": 1.0,
                "mean_diff": 0.0, "n": len(a)}
    # Zero-variance guard (Manual M10): a deterministic baseline paired by fold
    # can give a CONSTANT non-zero difference. Shapiro/ttest would divide by a
    # zero standard deviation — instead report the perfectly-consistent
    # separation directly (no division by zero).
    if np.std(diff) == 0:
        return {"test": "constant_diff",
                "stat": float(np.sign(diff.mean()) * np.inf),
                "pvalue": 0.0, "mean_diff": float(diff.mean()), "n": len(a)}
    normal_p = sps.shapiro(diff).pvalue
    if normal_p > alpha_normal:
        res = sps.ttest_rel(a, b)
        name = "ttest_rel"
    else:
        res = sps.wilcoxon(a, b)
        name = "wilcoxon"
    return {"test": name, "stat": float(res.statistic),
            "pvalue": float(res.pvalue), "mean_diff": float(diff.mean()),
            "n": len(a)}


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def compare_cells(df: pd.DataFrame, target: str = "cell4",
                  baselines: tuple[str, ...] = ("cell0", "cell1", "cell2", "cell3"),
                  metrics: tuple[str, ...] = DEFAULT_METRICS,
                  split: str = "test",
                  pair_on: tuple[str, ...] = ("seed", "fold")) -> pd.DataFrame:
    """Paired comparisons of `target` vs each baseline, Holm-corrected across
    the baseline x metric family.

    `df` is the long-format frame from run_grid.aggregate().
    Pairs share the `pair_on` keys; metrics are averaged within each pairing
    key first (idempotent for the default (seed, fold), and the way to pair a
    DETERMINISTIC baseline — which has no across-seed spread — by FOLD alone
    (Manual M10): pass pair_on=("fold",)). Returns one row per (baseline,
    metric) with raw and corrected p-values.
    """
    sub = (df[df["split"] == split]
           .groupby(["cell", *pair_on])[list(metrics)].mean().sort_index())
    rows = []
    for baseline in baselines:
        if baseline not in df["cell"].values or target not in df["cell"].values:
            continue
        t = sub.loc[target]
        b = sub.loc[baseline]
        common = t.index.intersection(b.index)      # matched pairs
        if len(common) < 3:                         # paired tests need >=3
            continue
        for metric in metrics:
            res = paired_test(t.loc[common, metric].values,
                              b.loc[common, metric].values)
            rows.append({"baseline": baseline, "metric": metric,
                         "pair_on": "+".join(pair_on), **res})
    out = pd.DataFrame(rows)
    if len(out):
        out["pvalue_holm"] = holm_correction(out["pvalue"].tolist())
        out["significant_5pct"] = out["pvalue_holm"] < 0.05
    return out


def seed_fold_dispersion(df: pd.DataFrame, metric: str = "sharpe",
                         split: str = "test") -> pd.DataFrame:
    """Mean ± std across seeds and across folds SEPARATELY (Manual J.3)."""
    sub = df[df["split"] == split]
    over_seeds = sub.groupby(["cell", "fold"])[metric].mean() \
        .groupby("cell").agg(["mean", "std"]).add_prefix("across_folds_")
    over_folds = sub.groupby(["cell", "seed"])[metric].mean() \
        .groupby("cell").agg(["mean", "std"]).add_prefix("across_seeds_")
    return over_seeds.join(over_folds)
