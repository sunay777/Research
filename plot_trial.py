"""Quick visualisation of a trial run's results.

Reads a results/{run_id}/ directory (train_log.csv, series_test.csv,
metrics.json) and writes PNGs into {run_dir}/figures/.

Usage:
    .venv/bin/python plot_trial.py results_smoke/cell4_seed0_fold0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results_smoke/cell4_seed0_fold0")
fig_dir = run_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
run_id = run_dir.name

# --------------------------------------------------------------------------
# 1. PPO learning curve — four diagnostics vs environment steps
# --------------------------------------------------------------------------
log = pd.read_csv(run_dir / "train_log.csv")
x = log["env_steps"]
panels = [
    ("explained_variance", "Critic explained variance (R²)", "C0"),
    ("value_loss", "Value loss", "C3"),
    ("mean_reward", "Mean reward (rolling Sharpe)", "C2"),
    ("approx_kl", "Approx KL (policy step size)", "C1"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (col, title, c) in zip(axes.ravel(), panels):
    ax.plot(x, log[col], "-o", color=c, ms=4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("environment steps")
    ax.grid(alpha=0.3)
fig.suptitle(f"PPO learning curve — {run_id}", fontsize=13, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
p1 = fig_dir / "learning_curve.png"
fig.savefig(p1, dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------
# 2. Test-window equity curve + market exposure
# --------------------------------------------------------------------------
ser = pd.read_csv(run_dir / "series_test.csv", parse_dates=["date"])
equity = np.exp(ser["log_return"].cumsum())
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(ser["date"], equity, color="C0", lw=1.6)
ax1.axhline(1.0, color="grey", ls="--", lw=0.8)
ax1.set_ylabel("equity (start = 1.0)")
ax1.set_title(f"Test-window equity curve — {run_id}", fontsize=12, weight="bold")
ax1.grid(alpha=0.3)
ax2.fill_between(ser["date"], ser["exposure"], color="C1", alpha=0.5)
ax2.set_ylabel("market exposure\n(1 − cash)")
ax2.set_xlabel("date")
ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.3)
fig.tight_layout()
p2 = fig_dir / "equity_curve.png"
fig.savefig(p2, dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------
# 3. Train / val / test metric bars
# --------------------------------------------------------------------------
metrics = json.loads((run_dir / "metrics.json").read_text())
splits = ["train", "val", "test"]
show = [("sharpe", "Sharpe"), ("cum_log_return", "Cum. log-return"),
        ("max_drawdown", "Max drawdown"), ("mean_turnover", "Mean turnover")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (key, label) in zip(axes, show):
    vals = [metrics[s][key] for s in splits]
    ax.bar(splits, vals, color=["C0", "C2", "C3"])
    ax.set_title(label, fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
fig.suptitle(f"Metrics by split — {run_id}", fontsize=13, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
p3 = fig_dir / "metrics_by_split.png"
fig.savefig(p3, dpi=130)
plt.close(fig)

print("wrote:")
for p in (p1, p2, p3):
    print(" ", p)
