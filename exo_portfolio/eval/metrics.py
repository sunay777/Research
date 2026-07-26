"""Performance metrics (Manual J.1). The rolling-Sharpe reward (E.4) lives
here so the env reward and the evaluation metric are literally the same code.

Full metric suite (cumulative log-return, annualised Sharpe, Sortino, MaxDD,
turnover aggregation) is completed at milestone M3 with hand-computed tests.
"""

from __future__ import annotations

import numpy as np


def rolling_sharpe(returns, eps: float = 1e-8) -> float:
    """Mean / sqrt(var + eps) over the given step log-returns (Manual E.4).

    For fewer than K observations the caller passes the available prefix.
    An empty prefix (at reset, before any step) scores 0.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    return float(r.mean() / np.sqrt(r.var() + eps))
