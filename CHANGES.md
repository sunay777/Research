# Methodology changes after the main ablation grid

The main grid (cell0–cell4 × 10 seeds × 5 folds, 2,000,000 PPO steps per run,
250 runs) finished on 2026-10-01. The 250 trained models were never retrained
or edited. Every change below either adds runs or changes how existing runs
are evaluated, pooled or reported. Dates are when each change was made.

## 1. Additional control cells

**1.1 cell0c: implementation control (2026-10-01).**
cell0 and cell1 are trained with Stable-Baselines3 PPO, while cell2–cell4 use
the project's own PPO (`exo_portfolio/algos/ppo.py`). So "cell4 vs cell0" mixed
up the architecture with the PPO implementation. cell0c has cell0's
switches (single encoder, symmetric critic, no exogenous input) but trains with
the custom PPO. It ran over the same 10 seeds × 5 folds at 2,000,000 steps
(`configs/experiment_grid_cell0c.yaml`). It has 69,821 parameters, against 17,021
for cell0 and 214,205 for cell4.

**1.2 cell0cw: capacity control (2026-10-01).**
cell0c widened to cell4's size. The single-encoder MLP's hidden width goes
from 128 to 468 (new `ModelCfg.stream_hidden`, default 128, so every existing
cell is unchanged). Width is the only difference from cell0c. That gives
104,236 actor / 213,981 total parameters, against cell4's 104,284 / 214,205
(−0.1%). Same 50 seed × fold runs (`configs/experiment_grid_cell0cw.yaml`).

**1.3 Statistical families.** cell4's Holm-corrected comparisons
(`cell_comparisons_holm.csv`) now also test against cell0c and cell0cw, so the
family grows from 16 to 24 tests (4 metrics × 6 baselines). Two new tables
isolate each control:
- `cell_comparisons_vs_cell0c_holm.csv`: cell0c vs cell0, the implementation effect alone.
- `cell_comparisons_vs_cell0cw_holm.csv`: cell0cw vs cell0c, the network-size effect alone.

All three use paired tests across matched (seed, fold) pairs.

## 2. Mechanism diagnostics (exposure–VIX regression, VIX-spike impulse response)

**2.1 Pooling across folds (2026-10-01).** Previously, only one fold's
seed-0 impulse response was kept: each fold overwrote the last, so the figure
showed fold 4 alone. Responses are now pooled across all five folds.

**2.2 All seeds, clustered by event or fold (2026-10-01).** The diagnostics
now use all 10 seeds × 5 folds per cell instead of seed 0 only. Every seed in
a fold sees the same VIX spikes, so stacking runs would count each event once
per seed and make the standard errors too small. To avoid this:
- **Impulse response:** each spike event's response path is averaged over the
  runs that saw it, then the mean and SE are taken across the distinct events
  (84 per cell).
- **Exposure–VIX β:** averaged over seeds within each fold, then mean ± SE
  across the five folds.
- **New table:** `mechanism_summary.csv` also reports the SE across seeds, and
  the share of runs whose own HAC-robust test is significantly negative or
  positive at 5%.

**2.3 Figure scope.** The impulse-response figure shows RL cells only. Baseline
responses are still in `diagnostics.csv` and `_diag_cache.json`.

## 3. Synthetic P_exo-shift experiment at full budget (2026-10-01)

The published `synthetic_shift.csv` was a smoke-budget result (2 seeds, 20,000
steps). It was regenerated at the full budget: 10 seeds, 100,000 steps,
20 evaluation episodes, cells 1/2/4. The code was unchanged.
Interpretation notes for the write-up:
- The "oracle" is a frozen training-world rule that reads the true regime. At
  shifts above about 0.5 it works against the shifted world, so negative
  regret means beating an anti-optimal reference, not robustness.
- Shift = 1 is a relabelled copy of shift = 0 (regime means, volatilities and
  persistence swapped).

## 4. Test-time exogenous-feature ablation (2026-10-01)

New analysis in `exo_portfolio/eval/test_time_ablation.py`. Nothing is retrained.
- **Models:** all 50 trained models of cell0 (null control), cell1, cell2 and cell4.
- **Rollouts:** each model is run on its fold's test window once intact, and once with
  one named exogenous group masked. The groups are asset returns, index return,
  VIX and macro.
- **Masking:** changes only what the agent observes; prices are untouched.
  - **Permute (primary):** the group's rows are shuffled in time *within the
    test window*. This keeps the group's distribution but breaks its link to
    the date. 3 permutations per model, identical across cells and seeds.
  - **Zero (secondary):** the group is set to 0. This is out of distribution for
    level features such as VIX.
- **Metrics:** change (masked − intact, paired by model) in Sharpe, cumulative
  log return, max drawdown and mean exposure, plus `policy_shift`: the mean
  fraction of the portfolio reallocated, ½‖w_masked − w_intact‖₁.
- **Statistics:** permutation draws averaged per model, then paired tests
  across the 50 models, Holm-corrected across the 4 groups within each
  (cell, mode, metric).
- **Validation:** intact rollouts reproduce every run's saved test metrics
  exactly, and cell0 shows exactly zero change under every mask.
- **Outputs:** `test_time_ablation.csv` (summary) and `test_time_ablation_runs.csv`
  (all 3,400 rollouts).

The retraining ablation (`exo_portfolio/exo_ablation.py`,
`configs/exo_ablation.yaml`) was **not run**. As configured it is one seed × one fold,
too few to detect plausible effects, and its 13 cell4 runs would cost about 60 CPU-hours.

## 5. Classical baselines and benchmark (2026-10-02)

**5.1 Pre-fold price history for rolling-window baselines.** Inverse
volatility, min variance, max Sharpe, mean–variance, momentum (60-day
lookbacks) and vol overlay (20-day) used to be computed only from prices inside
each test window. So they sat fully in cash (exposure 0) for their first
60 / 20 days of every fold, about 15% of the test period. They now take their
lookback from up to 61 trading days *before* the fold
(`baselines.classical.deterministic_window_targets`). This doesn't break
causality: every rule uses only data dated ≤ t, and the result equals running
the rule over the full history and keeping the test-window rows (unit-tested).
The RL agents' observation windows already reached back before the fold.
Portfolios still start each fold all-cash, as the agents do. Effect on the
stitched out-of-sample final wealth:

| Baseline | before → after |
|---|---|
| inverse volatility | 2.39× → 2.39× |
| min variance | 2.01× → 2.02× |
| vol overlay | 1.72× → 1.77× |
| mean–variance | 1.78× → 1.69× |
| max Sharpe | 1.78× → 1.59× |
| momentum | 2.03× → 1.67× |

The cost-sensitivity re-evaluation (J.5) uses the same construction.

**5.2 Renamed `random_buy_and_hold` → `random_constant_mix`.** The rule draws
one random long-only weight vector (cash + 29 stocks) and **rebalances back to
it every day**. That is a constant-mix portfolio, not buy-and-hold. Its random
stream is unchanged, so the numbers are identical. The superseded run
directories were moved to `results_superseded/`.

**5.3 Headline benchmark = equal weight (1/N) on the same 29 stocks.** It has the
same universe, costs, rebalancing simulator and survivorship bias as the
agents. Figures and the stitched tables now compare against it:
`beats_equal_weight` in `oos_stitched_by_seed.csv`, and
`seeds_beating_equal_weight` in `oos_stitched_summary.csv`. The S&P 500 (^GSPC)
line is kept as **context only**, labelled as a price index without dividends.
It is a different, broader universe, pays no costs, and excludes dividends,
whereas the stock prices are dividend-adjusted.

## 6. Reporting additions (no change to any result)

- Stitched walk-forward out-of-sample tables per (model, seed) and per model,
  plus a final-wealth-per-seed strip plot (`plot_thesis_figures.py`).
- Thinner lines and markers in all figures.

## 7. Compute

The main grid ran on the Wits Mathematical Sciences cluster (stampede
partition). cell0c, cell0cw, the full-budget synthetic experiment and every
re-evaluation above ran locally on an Apple M4 (one thread per run,
`OMP_NUM_THREADS=1`), using the same code and configuration.

## Known limitations (unchanged, state in the thesis)

- **Survivorship bias:** the universe is the 2024 DJIA membership applied back
  to 2010 (29 names; DOW excluded for missing pre-2019 history). It flatters
  the absolute returns of every strategy on these stocks, agents and baselines
  alike; comparisons between them are unaffected.
- **Cash earns 0%.**
- **Privileged critic input** (cell3 and cell4) is only the two macro series
  (Fed funds rate, CPI year on year) without their 1- and 14-day publication lags.
