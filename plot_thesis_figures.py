"""Thesis figures: market regimes and walk-forward out-of-sample equity curves.

    python plot_thesis_figures.py [--results results] [--cache data_cache]

Writes into <results>/figures/:
  market_regimes.png      S&P 500 over the full sample, days in the causal
                          VIX-stress regime shaded (same label as the results
                          tables: eval.regimes.label_stressed_vix), the five
                          walk-forward test folds marked, key events annotated.
  oos_equity_curves.png   The five contiguous test folds stitched into one
                          walk-forward out-of-sample path (2018-09 -> 2026-06).
                          RL cells: median across 10 seeds, with a 10th-90th
                          percentile seed band for cell4. Baselines are
                          deterministic (one path).
  oos_final_by_seed.png   Final stitched wealth of every seed per model, vs
                          the equal-weight (1/N) benchmark line.

and into <results>/tables/:
  oos_stitched_by_seed.csv   one row per (model, seed) stitched path: final
                             wealth, annual return, vol, Sharpe, max drawdown,
                             mean exposure / turnover, beats_equal_weight
  oos_stitched_summary.csv   one row per model: median across seeds (min/max
                             final wealth, number of seeds beating equal weight)

Benchmark: equal weight (1/N) on the SAME 29 stocks — same universe, costs and
survivorship bias as the agents. The S&P 500 (^GSPC) is a price index (no
dividends, no costs, different universe) and is drawn as context only.

Style (palette, axes, surface) is shared with exo_portfolio.eval.figures so
these sit alongside the pipeline's figures without looking different.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from exo_portfolio.eval.figures import (CELL_COLORS, CELL_LINESTYLES, INK,
                                        MUTED, SERIES_LW, SURFACE, _save,
                                        _style_ax)
from exo_portfolio.eval.regimes import label_stressed_vix

STRESS_FILL = "#f3d9cf"          # soft warm tint: regime context, not a series
FOLD_LINE = "#9d9c94"

CELL_LABELS = {
    "cell0": "cell0 (no exo)",
    "cell0c": "cell0c (no exo, custom PPO)",
    "cell0cw": "cell0cw (no exo, custom PPO, cell4-size)",
    "cell1": "cell1 (monolithic)",
    "cell2": "cell2",
    "cell3": "cell3",
    "cell4": "cell4",
}
# baselines: neutral inks, told apart by dash pattern. Equal weight on the
# same 29 stocks is the headline benchmark; the S&P line is context only.
BENCHMARK = "equal_weight"
BASELINES = {
    "equal_weight": ("Equal weight 1/N (benchmark)", "#3d3c38", "-"),
    "inverse_vol": ("Inverse volatility", "#6f6e68", (0, (1, 1.6))),
    "buy_and_hold_index": ("S&P 500 price index, no dividends (context)",
                           "#a3a29b", (0, (5, 2))),
}
# display names for the remaining (non-plotted-as-line) baselines
BASELINE_NAMES = {
    "random_constant_mix": "Random constant-mix",
    "random_weight": "Random weights (daily)",
    "random_action": "Random actions",
    "min_variance": "Min variance",
    "max_sharpe": "Max Sharpe",
    "mean_variance": "Mean-variance",
    "momentum": "Momentum (top 10)",
    "vol_overlay": "Vol overlay (1/N, 10% target)",
}
BANDED = ("cell4",)   # one band: overlapping bands turn to mud

EVENTS = [
    ("2011-08-08", "US downgrade"),
    ("2015-08-24", "China deval."),
    ("2018-12-24", "Q4-18 selloff"),
    ("2020-03-23", "COVID low"),
    ("2022-03-16", "Fed hikes begin"),
    ("2023-03-10", "SVB"),
    ("2025-04-08", "Tariff shock"),
]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_series(results: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(results.glob("*_seed*_fold*")):
        m = re.match(r"(.+)_seed(\d+)_fold(\d+)$", d.name)
        p = d / "series_test.csv"
        if not m or not p.exists():
            continue
        s = pd.read_csv(p, parse_dates=["date"])
        s["cell"], s["seed"], s["fold"] = m[1], int(m[2]), int(m[3])
        rows.append(s)
    if not rows:
        raise SystemExit(f"no series_test.csv under {results}")
    return pd.concat(rows, ignore_index=True)


def stitched_paths(df: pd.DataFrame, cell: str) -> pd.DataFrame:
    """date x seed matrix of cumulative wealth over the stitched test folds.
    Each seed's path chains that seed's agent from fold 0..4 (walk-forward:
    each fold is a separately trained agent tested on the next window)."""
    sub = df[df["cell"] == cell]
    lr = sub.pivot_table(index="date", columns="seed", values="log_return")
    lr = lr.sort_index()
    return np.exp(lr.cumsum())


def fold_starts(df: pd.DataFrame) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    ref = df[df["cell"] == df["cell"].iloc[0]]
    g = ref.groupby("fold")["date"]
    return [(f, g.min()[f], g.max()[f]) for f in sorted(g.groups)]


def shade_regime(ax, stressed: pd.Series, lo=None, hi=None):
    s = stressed.copy()
    if lo is not None:
        s = s[s.index >= lo]
    if hi is not None:
        s = s[s.index <= hi]
    # contiguous stressed blocks -> spans
    blk = (s != s.shift()).cumsum()
    for _, run in s.groupby(blk):
        if run.iloc[0]:
            ax.axvspan(run.index[0], run.index[-1] + pd.Timedelta(days=1),
                       color=STRESS_FILL, lw=0, zorder=0)


def log_axis(ax, lo, hi):
    ax.set_yscale("log")
    cands = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
    ticks = [t for t in cands if lo * 0.97 <= t <= hi * 1.03]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))


def direct_labels(ax, items, x, min_gap_frac=0.045):
    """items: (y_value, text). Nudge apart in log space so labels never
    collide; text in ink (identity is carried by the marker beside it)."""
    y0, y1 = (np.log(v) for v in ax.get_ylim())
    gap = (y1 - y0) * min_gap_frac
    items = sorted(items, key=lambda t: t[0])
    ys = [np.log(v) for v, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (v, text, color), y in zip(items, ys):
        ax.plot([x], [v], marker="o", ms=3.5, color=color, mec=SURFACE,
                mew=0.8, zorder=6, clip_on=False)
        ax.annotate(text, xy=(x, np.exp(y)), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK, annotation_clip=False)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_market_regimes(gspc: pd.Series, stressed: pd.Series, folds, out: Path):
    px = gspc.dropna()
    wealth = px / px.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.1))
    shade_regime(ax, stressed, px.index[0], px.index[-1])

    t0 = folds[0][1]
    ax.axvspan(px.index[0], t0, color="#000000", alpha=0.035, lw=0, zorder=0)
    ax.plot(wealth.index, wealth.values, color=INK, lw=SERIES_LW, zorder=3)
    lo, hi = wealth.min(), wealth.max()
    log_axis(ax, lo, hi)
    ax.set_ylim(lo * 0.9, hi * 1.12)

    for f, a, b in folds:
        ax.axvline(a, color=FOLD_LINE, lw=0.7, ls=(0, (4, 3)), zorder=2)
        mid = a + (b - a) / 2
        ax.text(mid, 0.035, f"test F{f}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=7.5, color=MUTED)
    ax.text(px.index[0] + (t0 - px.index[0]) / 2, 0.035,
            "training / validation history", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=7.5, color=MUTED)

    for i, (d, name) in enumerate(EVENTS):
        d = pd.Timestamp(d)
        if not (px.index[0] <= d <= px.index[-1]):
            continue
        ax.axvline(d, color=INK, lw=0.6, alpha=0.35, ls=":", zorder=2)
        ax.text(d, 1.015 + 0.045 * (i % 2), name,
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=INK)

    ax.set_ylabel("S&P 500 growth of $1 (log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(px.index[0], px.index[-1])
    _style_ax(ax)
    ax.grid(axis="x", visible=False)
    ax.set_title("Market path and the VIX stress regime used for evaluation",
                 pad=30, loc="left")
    handles = [Patch(color=STRESS_FILL, label="Stressed (VIX ≥ trailing-252d 80th pct.)"),
               Line2D([], [], color=FOLD_LINE, ls=(0, (4, 3)), label="Test-fold boundary")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
              labelcolor=INK)
    return _save(fig, out)


def fig_oos_equity(df: pd.DataFrame, stressed: pd.Series, folds, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    a0, b1 = folds[0][1], folds[-1][2]
    shade_regime(ax, stressed, a0, b1)
    for f, a, b in folds[1:]:
        ax.axvline(a, color=FOLD_LINE, lw=0.7, ls=(0, (4, 3)), zorder=2)
    for f, a, b in folds:
        ax.text(a + (b - a) / 2, 0.025, f"F{f}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=7.5, color=MUTED)

    ends, allv = [], []
    for cell, (label, color, ls) in BASELINES.items():
        if cell not in set(df["cell"]):
            continue
        w = stitched_paths(df, cell).iloc[:, 0]
        ax.plot(w.index, w.values, color=color, lw=SERIES_LW, ls=ls, zorder=3,
                label=label)
        ends.append((w.iloc[-1], label, color))
        allv.append(w.values)

    for cell in CELL_COLORS:
        if cell not in set(df["cell"]):
            continue
        W = stitched_paths(df, cell)
        med = W.median(axis=1)
        c = CELL_COLORS[cell]
        if cell in BANDED:
            q10, q90 = W.quantile(0.1, axis=1), W.quantile(0.9, axis=1)
            ax.fill_between(W.index, q10, q90, color=c, alpha=0.18, lw=0,
                            zorder=1, label=f"{cell} seed range (10th–90th pct.)")
            allv += [q10.values, q90.values]
        ax.plot(med.index, med.values, color=c, lw=SERIES_LW, zorder=4,
                ls=CELL_LINESTYLES.get(cell, "-"),
                label=CELL_LABELS.get(cell, cell))
        ends.append((med.iloc[-1], cell, c))
        allv.append(med.values)

    allv = np.concatenate(allv)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    log_axis(ax, lo, hi)
    ax.set_ylim(lo * 0.9, hi * 1.12)
    ax.axhline(1, color="#c3c2b7", lw=1, zorder=1)
    ax.set_xlim(a0, b1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_ax(ax)
    ax.grid(axis="x", visible=False)
    direct_labels(ax, [(v, f"{t}  {v:.2f}×", c) for v, t, c in ends], b1)

    ax.set_ylabel("Growth of $1, walk-forward out-of-sample (log scale)")
    ax.set_title("Walk-forward out-of-sample equity curves, test folds F0–F4 "
                 "stitched (RL cells: median of 10 seeds)", loc="left", pad=12)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=STRESS_FILL, label="Stressed regime"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
              labelcolor=INK, ncol=2)
    return _save(fig, out)


def stitched_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, seed) for the stitched walk-forward test path."""
    rows = []
    for cell, sub in df.groupby("cell"):
        lr = sub.pivot_table(index="date", columns="seed",
                             values="log_return").sort_index()
        years = len(lr) / 252
        W = np.exp(lr.cumsum())
        exp = sub.groupby("seed")["exposure"].mean()
        turn = sub.groupby("seed")["turnover"].mean()
        for seed in lr.columns:
            r = lr[seed]
            rows.append({
                "model": cell, "seed": seed,
                "final_wealth": W[seed].iloc[-1],
                "annual_return": np.exp(r.sum() / years) - 1,
                "annual_vol": r.std() * np.sqrt(252),
                "sharpe": r.mean() / r.std() * np.sqrt(252),
                "max_drawdown": (W[seed] / W[seed].cummax() - 1).min(),
                "mean_exposure": exp[seed], "mean_daily_turnover": turn[seed],
            })
    out = pd.DataFrame(rows)
    bench = out.loc[out["model"] == BENCHMARK, "final_wealth"]
    if len(bench):
        out["beats_equal_weight"] = out["final_wealth"] > bench.iloc[0]
    return out


def stitched_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    g = by_seed.groupby("model")
    out = g.median(numeric_only=True).drop(
        columns=["seed", "beats_equal_weight"], errors="ignore")
    out.insert(0, "n_seeds", g.size())
    out.insert(2, "final_wealth_min", g["final_wealth"].min())
    out.insert(3, "final_wealth_max", g["final_wealth"].max())
    if "beats_equal_weight" in by_seed:
        out["seeds_beating_equal_weight"] = (g["beats_equal_weight"].sum()
                                             .astype(int))
        out.loc[BENCHMARK, "seeds_beating_equal_weight"] = np.nan
    return out.sort_values("final_wealth", ascending=False)


def fig_final_by_seed(by_seed: pd.DataFrame, out: Path):
    """Strip plot: each seed's final stitched wealth per model."""
    order = (by_seed.groupby("model")["final_wealth"].median()
             .sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(order) + 1.2))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = by_seed.loc[by_seed["model"] == m, "final_wealth"].values
        c = CELL_COLORS.get(m, MUTED)
        jitter = rng.uniform(-0.12, 0.12, len(v)) if len(v) > 1 else 0
        ax.scatter(v, i + jitter, s=14, color=c, alpha=0.85, lw=0, zorder=3)
        ax.plot([np.median(v)] * 2, [i - 0.28, i + 0.28], color=INK,
                lw=SERIES_LW, zorder=4)
    bench = by_seed.loc[by_seed["model"] == BENCHMARK, "final_wealth"]
    if len(bench):
        ax.axvline(bench.iloc[0], color=INK, lw=0.7, ls=(0, (4, 3)), zorder=2)
        ax.text(bench.iloc[0], len(order) - 0.4, " equal weight (benchmark)",
                fontsize=7.5, color=INK, va="bottom")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([CELL_LABELS.get(m) or BASELINE_NAMES.get(m)
                        or BASELINES.get(m, (m,))[0] for m in order])
    ax.set_xlabel("Final growth of $1, walk-forward out-of-sample (log scale)")
    ax.set_title("Final wealth per seed (dots) and median (bar)", loc="left")
    _style_ax(ax)
    ax.grid(axis="y", visible=False)
    return _save(fig, out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--cache", default="data_cache")
    args = ap.parse_args(argv)
    results, cache = Path(args.results), Path(args.cache)

    df = load_series(results)
    gspc = pd.read_csv(cache / "gspc.csv", parse_dates=["Date"],
                       index_col="Date")["gspc"]
    vix = pd.read_csv(cache / "vix.csv", parse_dates=["Date"],
                      index_col="Date")["vix"]
    stressed = label_stressed_vix(vix)
    folds = fold_starts(df)

    figs, tables = results / "figures", results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    by_seed = stitched_stats(df)
    by_seed.to_csv(tables / "oos_stitched_by_seed.csv", index=False)
    stitched_summary(by_seed).to_csv(tables / "oos_stitched_summary.csv")
    print("wrote", tables / "oos_stitched_by_seed.csv")
    print("wrote", tables / "oos_stitched_summary.csv")
    for p in (fig_market_regimes(gspc, stressed, folds, figs / "market_regimes.png"),
              fig_oos_equity(df, stressed, folds, figs / "oos_equity_curves.png"),
              fig_final_by_seed(by_seed, figs / "oos_final_by_seed.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
