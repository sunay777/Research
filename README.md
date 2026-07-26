# exo_portfolio — Exploiting Exogenous Structure in RL for Portfolio Optimisation

Honours research project, Sunay Master (2677874). Built strictly to
`Development_Manual.md` (the *how*) and `Development_Design_Notes.md` (the *why*)
in the parent Research folder.

## The two claims the code must keep separate (Manual Part B)

- **Claim A (sample efficiency / stability):** the factored agent reaches a
  performance threshold in fewer env steps and with lower seed variance than the
  monolithic baseline. Evidence: learning curves, steps-to-threshold, seed variance.
- **Claim B (robustness under `P_exo` shift):** better risk-adjusted performance
  in *stressed* out-of-sample windows. Evidence: regime-conditional metrics on
  real data **and** the synthetic `P_exo`-shift experiment — an inductive-bias
  argument, *not* a corollary of the Exo-MDP regret bounds.

## Layout

```
configs/            base.yaml, universe_djia30.yaml, experiment_grid.yaml
exo_portfolio/
  config.py         typed config schema; seed_everything
  data/             loaders (yfinance/FRED) · align (lags, NaN policy) · splits (folds)
  envs/             portfolio_env (Exo-MDP env) · synthetic_exo (Part I, M8)
  models/           encoders · heads · agent (ablation switches)      [M5]
  algos/            custom PPO (dict obs + privileged critic)         [M5]
  baselines/        classical (1/N, MV, vol overlay) [M3] · SB3       [M4]
  eval/             metrics [M3] · regimes · stats · diagnostics      [M7]
  train.py          single-run entrypoint (cell, seed, fold)          [M4+]
  run_grid.py       full experiment matrix                            [M6]
tests/              env math · metrics · no-lookahead · splits
results/            one directory per run_id = {cell}_seed{seed}_fold{fold}
```

## Setup

```bash
pip install -e ".[dev]"        # M0-M3 (data + env + classical baselines)
pip install -e ".[rl,dev]"     # from M4 (torch, SB3, tensorboard)
pytest -q                      # acceptance gates — must stay green
```

## Fetching real data (M1)

```bash
python -m exo_portfolio.data.loaders --config configs/base.yaml configs/universe_djia30.yaml
```

Downloads DJIA-30 adjusted OHLCV, ^GSPC, ^VIX (yfinance) and FEDFUNDS/CPIAUCSL
(FRED) into `data_cache/`, then builds the aligned, lagged, NaN-free feature
matrices. Requires network access; all tests run on synthetic fixtures and do
**not** need the download.

FRED access uses the official API when a key is present — put
`FRED_API_KEY=...` in a `.env` file at the repo root (gitignored) or export it
as an environment variable. Without a key it falls back to the public
fredgraph CSV endpoint. yfinance needs no key. Run once on a machine with
internet; the CSV cache makes everything reproducible offline afterwards.

## Milestone status (Manual Part F)

| M | Deliverable | Status |
|---|---|---|
| M0 | Repo skeleton, config, CI runs pytest | ✅ done |
| M1 | Data layer (loaders, alignment/lags, folds) | ✅ done (tests green) |
| M2 | Exo-MDP environment (E.3 dynamics, E.4 reward) | ✅ done (tests green) |
| M3 | Metrics (hand-computed tests) + classical baselines (1/N, B&H, MV, vol overlay) | ✅ done (tests green) |
| M4 | SB3 baselines (cells 0–1) + train.py entrypoint | ✅ code done; smoke-validated on real data |
| M5 | Encoders + ExoActorCritic (5-cell switches) + custom PPO | ✅ done (overfit gate green) |
| M6 | Ablation grid runner | ⏳ next |
| M7-M9 | eval/stats/diagnostics → synthetic → repro | pending |

**M4 note (honest scope):** the "reproduce a known FinRL-style number" acceptance
requires the full 2M-step budget per run — that belongs on the Mathematical
Sciences Cluster (Manual L). In-session validation: a 50k-step PPO cell1 run on
real fold0 executes end-to-end and yields plausible under-trained metrics
(train Sharpe ≈ 1.1, COVID-window test Sharpe ≈ 0.06, maxDD 32%). Full-budget
runs: `python -m exo_portfolio.train --config configs/base.yaml configs/universe_djia30.yaml --cell cell1 --seed 0 --fold 0`.

## Guardrails (never break these)

1. **No lookahead** — any feature at day *t* was knowable at the close of day *t*;
   macro is published with a lag, and the lag is applied. `tests/test_no_lookahead.py`
   is the gate.
2. **`exo_critic_extra` discipline** — un-lagged macro is privileged input for the
   asymmetric critic only; the actor never touches it; nothing is ever dated > *t*.
3. **One config object**, everything seeded, deterministic env step.
4. **The ablation grid stays clean** — identical optimiser/seeds/data/budget
   across cells; only architecture switches differ.
