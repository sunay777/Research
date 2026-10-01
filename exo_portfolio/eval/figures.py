"""Write-up figures (Manual M9) — matplotlib, one style for every figure.

Style follows the project's chart conventions: fixed categorical color per
cell (identity never repainted), one axis per panel, thin marks, recessive
grid, text in ink colors (never series colors). Palette validated for CVD
safety (adjacent-pair ΔE ≥ 8) with the skill validator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical slots (validated): color follows the cell, everywhere
CELL_COLORS = {
    "cell0": "#2a78d6",   # blue
    "cell0c": "#2a78d6",  # cell0 on the custom PPO: same blue, dashed
    "cell0cw": "#2a78d6", # cell0c at cell4's size: same blue, dotted
    "cell1": "#eb6834",   # orange
    "cell2": "#1baf7a",   # aqua
    "cell3": "#eda100",   # yellow
    "cell4": "#e87ba4",   # magenta
}
# a cell that shares another's color is told apart by dash pattern
CELL_LINESTYLES = {"cell0c": "--", "cell0cw": ":"}
CLASSICAL_COLOR = "#898781"      # muted — classical baselines are context
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
STRESS_SHADE = "#f0efec"


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK)
        lbl.set_fontsize(9)
    ax.title.set_color(INK)
    ax.title.set_fontsize(10)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_learning_curves(results_root: str | Path, out: Path) -> Path | None:
    """Mean rollout reward vs env steps per cell (mean ± std over seeds).
    Claim A evidence: sample efficiency / stability. Uses the custom cells'
    train_log.csv (SB3 cells log to TensorBoard instead)."""
    frames = []
    for p in Path(results_root).glob("*/train_log.csv"):
        cell = p.parent.name.split("_")[0]
        df = pd.read_csv(p)
        df["cell"] = cell
        frames.append(df[["cell", "env_steps", "mean_reward"]])
    if not frames:
        return None
    df = pd.concat(frames)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for cell, g in df.groupby("cell"):
        agg = g.groupby("env_steps")["mean_reward"].agg(["mean", "std"])
        c = CELL_COLORS.get(cell, MUTED)
        ax.plot(agg.index, agg["mean"], color=c, linewidth=2, label=cell,
                linestyle=CELL_LINESTYLES.get(cell, "-"))
        if agg["std"].notna().any():
            ax.fill_between(agg.index, agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"], color=c, alpha=0.15,
                            linewidth=0)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean rollout reward")
    ax.set_title("Learning curves (mean ± std across seeds)")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, out)


def fig_synthetic_shift(csv_path: str | Path, out: Path) -> Path | None:
    """Degradation vs P_exo shift: raw reward and oracle-regret panels."""
    p = Path(csv_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    has_regret = "regret" in df.columns
    ncols = 2 if has_regret else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 3.4))
    axes = np.atleast_1d(axes)

    for ax, col, title in zip(
            axes, ["mean_reward", "regret"][:ncols],
            ["Test reward vs P_exo shift",
             "Regret vs oracle (lower = more robust)"][:ncols]):
        for cell, g in df.groupby("cell"):
            agg = g.groupby("shift")[col].agg(["mean", "std", "count"])
            c = CELL_COLORS.get(cell, MUTED)
            ci = 1.96 * agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.plot(agg.index, agg["mean"], color=c, linewidth=2,
                    linestyle=CELL_LINESTYLES.get(cell, "-"),
                    marker="o", markersize=4, label=cell)
            if agg["std"].notna().any():
                ax.fill_between(agg.index, agg["mean"] - ci, agg["mean"] + ci,
                                color=c, alpha=0.15, linewidth=0)
        if col == "mean_reward" and "oracle_mean_reward" in df.columns:
            o = df.groupby("shift")["oracle_mean_reward"].mean()
            ax.plot(o.index, o.values, color=MUTED, linewidth=1.5,
                    linestyle="--", label="oracle")
        ax.set_xlabel("shift magnitude s")
        ax.set_ylabel(col.replace("_", " "))
        ax.set_title(title)
        _style_ax(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, out)


def fig_equity_curves(results_root: str | Path, out: Path,
                      stressed: pd.Series | None = None,
                      seed: int = 0, fold: int = 0) -> Path | None:
    """Cumulative test log-return per cell on one fold, stressed spans shaded."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = False
    for cell in CELL_COLORS:
        p = Path(results_root) / f"{cell}_seed{seed}_fold{fold}" / "series_test.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        ax.plot(s.index, s["log_return"].cumsum(), linewidth=2,
                color=CELL_COLORS[cell], label=cell,
                linestyle=CELL_LINESTYLES.get(cell, "-"))
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    if stressed is not None:
        xs = ax.get_xlim()
        sub = stressed.reindex(pd.date_range(*ax.get_lines()[0].get_xdata()[[0, -1]],
                                             freq="D")).fillna(False)
        in_block = False
        for d, flag in sub.items():
            if flag and not in_block:
                start, in_block = d, True
            elif not flag and in_block:
                ax.axvspan(start, d, color=STRESS_SHADE, zorder=0)
                in_block = False
        if in_block:
            ax.axvspan(start, sub.index[-1], color=STRESS_SHADE, zorder=0)
        ax.set_xlim(xs)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative log return")
    ax.set_title(f"Test equity curves — fold {fold} (stressed days shaded)")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, out)


def fig_exposure_vix_betas(betas: pd.DataFrame, out: Path) -> Path | None:
    """Bar chart of exposure~VIX beta per cell (J.4 mechanism evidence).
    `betas` columns: cell, beta, and either fold (one row per run: seeds are
    averaged within each fold, error bar = 95% CI across folds) or se."""
    if betas.empty:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    order = [c for c in CELL_COLORS if c in set(betas["cell"])]
    n_seeds = betas["seed"].nunique() if "seed" in betas else 1
    if "fold" in betas and betas.groupby("cell").size().max() > 1:
        # seeds are replicate policies on the same market window: average
        # them within each fold, then the SE is across folds
        per_fold = betas.groupby(["cell", "fold"])["beta"].mean()
        g = per_fold.groupby("cell")
        sub = pd.DataFrame({"beta": g.mean(),
                            "se": g.std(ddof=1) / np.sqrt(g.size())}).loc[order]
    else:
        sub = betas.set_index("cell").loc[order]
    colors = [CELL_COLORS[c] for c in order]
    err = 1.96 * sub["se"] if "se" in sub else None
    bars = ax.bar(order, sub["beta"], color=colors, width=0.6,
                  yerr=err, ecolor=MUTED, capsize=3, error_kw={"linewidth": 1})
    for bar, c in zip(bars, order):     # shared-color cells: hatched bar
        if c in CELL_LINESTYLES:
            bar.set_hatch("///" if CELL_LINESTYLES[c] == "--" else "...")
            bar.set_edgecolor(SURFACE)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_ylabel("exposure ~ VIX slope (β)")
    ax.set_title("De-risking response to VIX by cell"
                 + (f"\n({n_seeds} seeds averaged per fold; ±95% CI across folds)"
                    if n_seeds > 1 else ""))
    _style_ax(ax)
    return _save(fig, out)


def fig_impulse_response(responses: dict[str, dict], out: Path) -> Path | None:
    """Mean exposure change after VIX spikes per cell, pooled over every
    seed x fold run and clustered by spike event. `responses[cell]` = output
    of pool_impulse_by_event (vix_spike_impulse_response's shape + n_runs)."""
    # RL cells only: baselines are in the cache/tables, but 11 grey lines (and
    # vol_overlay's mechanical de-risking) would swamp the cells' scale
    valid = {c: responses[c] for c in CELL_COLORS
             if responses.get(c) and responses[c].get("mean_response") is not None}
    if not valid:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for cell, r in valid.items():
        days = np.arange(len(r["mean_response"]))
        c = CELL_COLORS.get(cell, MUTED)
        ax.plot(days, r["mean_response"], color=c, linewidth=2,
                linestyle=CELL_LINESTYLES.get(cell, "-"),
                marker="o", markersize=3.5,
                label=f"{cell} (n={r['n_spikes']} events)")
        if r.get("se_response"):
            se = np.asarray(r["se_response"])
            m = np.asarray(r["mean_response"])
            ax.fill_between(days, m - 1.96 * se, m + 1.96 * se,
                            color=c, alpha=0.15, linewidth=0)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xlabel("days after VIX spike")
    ax.set_ylabel("Δ exposure vs day −1")
    n_runs = max((r.get("n_runs", 1) for r in valid.values()), default=1)
    ax.set_title("Impulse response of exposure to VIX spikes"
                 + (f"\n({n_runs} runs per cell, seed-averaged per event; "
                    "±95% CI across events)" if n_runs > 1 else ""))
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    return _save(fig, out)
