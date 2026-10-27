# Methodology changes after the main ablation grid

The main grid (cell0–cell4 × 10 seeds × 5 folds, 2,000,000 PPO steps per run,
250 runs) finished on 2026-10-01. The 250 trained models were never retrained
or edited. §8 records a data-alignment bug (macro look-ahead) found on
2026-10-11, the fix, and its measured effect on the trained models. Every change below either adds runs or changes how existing runs
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

## 8. Macro publication-lag fix and re-evaluation (2026-10-11)

**8.1 The bug.** The actor's macro features became visible before they were
published, in two ways:
- FRED stamps a monthly value with the *first* day of its reference month, and
  the lag was added to that stamp. So the Fed funds average for month m was
  visible from day 2 of month m, before most of the days it averages.
- The lag config is keyed by the raw series (`CPIAUCSL`), but it was looked up
  by the derived column name (`CPI_YOY`), which silently fell back to **0 days**.
  CPI inflation for month m was visible from the 1st of month m, about six
  weeks before its release.

Example: on 2 March 2020 the actor already saw March's Fed funds average (0.65%,
reflecting the emergency cuts of 3 and 15 March) and March's CPI inflation.
Because the CPI lag was 0, the actor's and the critic's CPI columns were
identical, and the "privileged" critic input differed from the actor's only by
one day of Fed funds. Every cell that sees macro (cells 1–4) was trained and
evaluated with these inputs.

**8.2 The fix** (`exo_portfolio/data/align.py`, `configs/base.yaml`):
- Lags now count from the **end** of the reference month.
- Derived features inherit their source series' lag (`MACRO_SOURCE`), and a
  column with no configured lag raises an error instead of defaulting to 0.
- CPI lag is now **31 days** after month end (was 14). Normal releases come
  10–17 days after month end, but shutdown-delayed releases came later (e.g.
  September 2025 CPI on 24 October 2025); 31 days rules out any leak.
- The missing October 2025 CPI (not collected during the US government
  shutdown) is no longer padded from September; that month adds no new value.
- The critic's privileged copy is now visible from the end of its reference
  month: it never describes days after t, but is still ahead of publication.
  For Fed funds this equals the actor's value on 98% of days, so the critic's
  privilege is now almost entirely the CPI release delay.
- Three regression tests in `tests/test_no_lookahead.py`; the existing lag test
  now checks the end-of-month rule. 110 tests.
- Side effect: the first usable day moves from 2011-01-03 to 2011-03-03, so
  any NEW run would get test windows shifted by about 3–4 weeks
  (fold0 test would start 2018-10-25 instead of 2018-09-27). Pin the folds to
  the old dates before retraining so new runs stay paired with the old ones.

**8.3 Re-evaluation of the trained models** (`exo_portfolio/eval/macro_lag_reeval.py`).
Nothing was retrained. All 250 models of cells 0–4 were rolled out on their
original test windows twice: once on the exact inputs they were trained with
(rebuilt with the old alignment; every run reproduces its saved test metrics,
max abs error 1.4e-5, from a different torch version), and once with only the
actor's two macro columns replaced by correctly lagged ones.

| Cell | Test Sharpe, old inputs → corrected | Paired p | Portfolio reallocated |
|---|---|---|---|
| cell0 (null control) | 0.425 → 0.425 | 1.0 | 0% |
| cell1 | −0.514 → −0.508 | 0.11 | 1.9% |
| cell2 | 0.716 → 0.713 | 0.84 | 1.1% |
| cell3 | −0.564 → −0.562 | 0.69 | 1.5% |
| cell4 | 0.715 → 0.706 | 0.51 | 1.5% |

Effect on the headline comparisons (cell4 vs each cell, Holm across the same
24-test family; cells without macro keep their saved metrics):

| cell4 vs | Sharpe Δ, Holm p (old → corrected) | Sortino Δ, Holm p (old → corrected) |
|---|---|---|
| cell0cw | +0.289, 0.046 → +0.281, **0.062** | +0.438, 0.035 → +0.425, **0.047** |
| cell0 | +0.289, 0.047 → +0.281, **0.065** | +0.427, 0.047 → +0.414, **0.065** |
| cell0c | +0.139, 0.34 → +0.131, 0.46 | +0.217, 0.29 → +0.204, 0.39 |

Everything else keeps its conclusion: cell4 ≈ cell2 (Sharpe −0.007), cell4
beats cell1/cell3, cell4 does not beat equal weight (Sharpe −0.22, n = 5),
and stressed-day Sharpe is unchanged (cell4 −2.56 vs cell0 −1.93, paired
difference −0.63, p = 0.003; vs cell0cw −0.19, p = 0.31).

**Interpretation for the write-up.** The trained policies barely use the leaked
macro values at test time, so the test-time effect of the leak is small. It does
not remove the training-time effect: cells 1–4 were trained on inputs that
contained future information. The cell4-vs-cell0cw Sharpe result was borderline
before and is no longer significant after Holm correction; Sortino stays just
under 0.05. Outputs: `results/tables/macro_lag_reeval_*.csv`.

## Known limitations (unchanged, state in the thesis)

- **Survivorship bias:** the universe is the 2024 DJIA membership applied back
  to 2010 (29 names; DOW excluded for missing pre-2019 history). It flatters
  the absolute returns of every strategy on these stocks, agents and baselines
  alike; comparisons between them are unaffected.
- **Cash earns 0%.**
- **Macro look-ahead in the trained models** (§8): cells 1–4 were trained with
  macro inputs that were visible before publication. Fixed in the code on
  2026-10-11 and bounded at test time by re-evaluation; removing the
  training-time effect needs retraining cells 1–4.
- **Privileged critic input** (cell3 and cell4) is only the two macro series
  (Fed funds rate, CPI year on year). In the trained models it was effectively
  the same as the actor's input (§8.1). After the fix it is each value from the
  end of its reference month, before publication; for a meaningful privileged
  signal at retraining, use the daily Fed funds rate (FRED `DFF`).

## 9. Supervisor review of draft 1 (2026-10-24)

**9.1 Fold-level statistics.** The 10 seeds in a fold share one test period,
so the 50 runs of a cell are not independent. All paired tests now average
over seeds within each fold and compare the five fold means with a paired
t-test (a Wilcoxon test cannot reach p < 0.05 with n = 5), Holm-corrected
within each family (`eval/fold_level_stats.py` ->
`results/tables/fold_level/`). Only the factored-vs-flat comparisons survive;
cell4 vs cell0cw Sharpe is +0.29, p = 0.068 unadjusted, 0.88 Holm. Re-running
with publication-dated macro inputs changes none of the fold-level
conclusions.

**9.2 Untrained policy** (`eval/untrained_policy.py` ->
`results/untrained/`). Every cell, built exactly as in training but not
trained, scores test Sharpe 0.926 (equal weight 0.923): the near-zero
initial logits give 1/30 to each stock and cash. Every trained cell's mean
Sharpe is below its untrained starting point (cell4 -0.21, Holm p = 0.57;
flat cells about -1.45, Holm p ~ 0.05).

**9.3 Normalised flat agents** (`experiments/norm_flat.py`, presets
cell1n / cell3n). cell1 / cell3 with every input z-scored on the fold's
training window only; same features, folds, seeds and budget as the grid.
Runs natively on the Mac with `zsh cluster/run_norm_mac.sh` (resumable).
Result (100 runs, 2026-10-24): normalisation makes both agents worse in every
fold. Test Sharpe cell1n -1.02 (cell1 -0.51), cell3n -1.20 (cell3 -0.56);
turnover rises to about 1.5/day; training Sharpe stays 4.6-6.7. cell4 beats
both (Holm p = 0.030, 0.033; family of 16 in
`tables/fold_level/normalised_flat.csv`). Re-evaluating with
publication-dated macro (`eval/norm_flat_reeval.py`) moves mean Sharpe by
< 0.005 and changes no conclusion. `eval/macro_lag_reeval.make_actor` now
maps cell1n to the cell1 SB3 observation level.

**9.4 First-step reward.** Episodes start in cash, so the first return is the
opening trade's cost with zero variance; the first reward is about -2 (cell4)
to -20 (cell1), against typical |r| < 1. One step per episode; documented in
the paper, not changed (changing it would break comparability with the grid).

**9.5 Figures** use the paper's plain agent names (`eval/figures.py`
`DISPLAY_NAMES`, `plot_thesis_figures.CELL_LABELS`).
