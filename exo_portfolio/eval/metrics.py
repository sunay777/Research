"""Performance metrics (Manual J.1). The rolling-Sharpe reward (E.4) lives
here so the env reward and the evaluation metric are literally the same code.

Every metric is a small pure function of a vector of step log-returns (and,
for turnover, per-step turnovers), unit-tested against hand-computed values
on toy series (M3 acceptance).

Conventions (documented so the thesis tables are unambiguous):
- Sharpe/Sortino are annualised with sqrt(periods_per_year), 252 by default.
- Sharpe uses the sample std (ddof=1).
- Sortino's downside deviation is sqrt(mean over ALL n observations of
  min(r, 0)^2) — the zero-target full-length convention.
- Max drawdown is reported as a positive fraction of the running peak.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def rolling_sharpe(returns, eps: float = 1e-8) -> float:
    """Mean / sqrt(var + eps) over the given step log-returns (Manual E.4).

    This is the env's reward. For fewer than K observations the caller passes
    the available prefix. An empty prefix (at reset) scores 0.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    return float(r.mean() / np.sqrt(r.var() + eps))


def cumulative_log_return(returns) -> float:
    return float(np.sum(np.asarray(returns, dtype=np.float64)))


def annualized_sharpe(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside == 0:
        return float("inf") if r.mean() > 0 else 0.0
    return float(r.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    """Largest peak-to-trough decline of the equity curve implied by the step
    log-returns, as a positive fraction (0.25 = -25% from peak)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    values = np.exp(np.cumsum(r))
    peaks = np.maximum.accumulate(np.concatenate([[1.0], values]))[1:]
    return float(np.max(1.0 - values / peaks))


def mean_turnover(turnovers) -> float:
    t = np.asarray(turnovers, dtype=np.float64)
    return float(t.mean()) if t.size else 0.0


def summarize(returns, turnovers=None,
              periods_per_year: int = TRADING_DAYS) -> dict:
    """The standard results row (Manual J.1)."""
    return {
        "cum_log_return": cumulative_log_return(returns),
        "sharpe": annualized_sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "mean_turnover": mean_turnover(turnovers if turnovers is not None else []),
        "n_steps": int(np.asarray(returns).size),
    }
