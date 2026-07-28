"""Causal calm/stressed regime labelling (Manual J.2).

Claim B lives in the stressed slice, so the labels must be beyond reproach:
the label for day t uses ONLY information available at the close of day t.
No future-derived labels, ever (Manual L: privileged-info discipline applies
to evaluation labels too).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exo_portfolio.eval.metrics import summarize


def label_stressed_vix(vix: pd.Series, percentile: float = 0.80,
                       lookback: int = 252,
                       min_periods: int = 60) -> pd.Series:
    """Day t is 'stressed' iff VIX_t >= the `percentile` quantile of VIX over
    the trailing `lookback` days *up to and including t*.

    Including t is causal — VIX_t is known at the close of t. The first
    `min_periods` days are labelled calm (not enough history to judge).
    """
    thresh = vix.rolling(lookback, min_periods=min_periods).quantile(percentile)
    return (vix >= thresh).fillna(False)


def label_stressed_realized_vol(portfolio_log_returns: pd.Series,
                                percentile: float = 0.80,
                                vol_window: int = 20,
                                lookback: int = 252,
                                min_periods: int = 60) -> pd.Series:
    """Alternative label: trailing realised vol of a reference portfolio
    (e.g. 1/N) above its own trailing quantile. Same causality convention."""
    vol = portfolio_log_returns.rolling(vol_window).std()
    thresh = vol.rolling(lookback, min_periods=min_periods).quantile(percentile)
    return (vol >= thresh).fillna(False)


def regime_conditional_metrics(log_returns: pd.Series, turnover: pd.Series,
                               stressed: pd.Series) -> dict[str, dict]:
    """The J.2 report row: pooled + calm + stressed metric blocks.

    `log_returns`/`turnover` are indexed by date; `stressed` is a boolean
    series covering (at least) those dates.
    """
    s = stressed.reindex(log_returns.index).fillna(False).astype(bool)
    out = {"pooled": summarize(log_returns.values, turnover.values)}
    for name, mask in (("calm", ~s), ("stressed", s)):
        r = log_returns[mask.values]
        t = turnover[mask.values]
        out[name] = summarize(r.values, t.values)
    out["n_stressed_days"] = int(s.sum())
    out["frac_stressed"] = float(s.mean()) if len(s) else 0.0
    return out
