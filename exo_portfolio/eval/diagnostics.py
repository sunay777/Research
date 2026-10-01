"""Mechanistic diagnostics (Manual J.4) — the evidence for Claim B.

Performance alone can be luck; these show the *mechanism*: the exo-aware
agent de-risks when exogenous risk rises.

1. exposure ~ VIX regression: realised equity exposure (1 - cash weight)
   regressed on contemporaneous VIX. Expect a significantly negative slope
   for exo-aware agents (cell2/4), weaker or absent for cell0/1.
2. VIX-spike impulse response: mean change in exposure in the days after a
   large VIX jump. A de-risking response is the thesis's claimed mechanism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def exposure_vix_regression(exposure: pd.Series, vix: pd.Series) -> dict:
    """OLS: exposure_t = alpha + beta * VIX_t (+ HAC-robust errors).

    Both series indexed by date; aligned on their intersection.
    Returns beta, its p-value (Newey-West, maxlags=5), and R^2.
    """
    import statsmodels.api as sm

    idx = exposure.index.intersection(vix.index)
    y = exposure.loc[idx].astype(float)
    x = sm.add_constant(vix.loc[idx].astype(float).rename("vix"))
    fit = sm.OLS(y.values, x.values).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": 5})
    return {"beta": float(fit.params[1]), "pvalue": float(fit.pvalues[1]),
            "r2": float(fit.rsquared), "n": int(len(idx))}


def find_vix_spikes(vix: pd.Series, z: float = 2.0, lookback: int = 63,
                    min_jump: float = 1.0) -> pd.DatetimeIndex:
    """Days where the 1-day VIX change exceeds `z` trailing stds of daily
    VIX changes (threshold computed causally from the past `lookback` days)
    AND at least `min_jump` VIX points in absolute terms — a 2-sigma move in
    a dead-calm tape is not an economically meaningful spike."""
    dv = vix.diff()
    sd = dv.rolling(lookback, min_periods=20).std().shift(1)  # past-only
    spikes = (dv > z * sd) & (dv > min_jump)
    return vix.index[spikes.fillna(False)]


def vix_spike_impulse_response(exposure: pd.Series, vix: pd.Series,
                               z: float = 2.0, horizon: int = 10) -> dict:
    """Average exposure change (relative to the day before the spike) over
    each day 0..horizon following VIX spikes inside the exposure window."""
    spikes = find_vix_spikes(vix, z=z)
    spikes = spikes[(spikes >= exposure.index[1])
                    & (spikes <= exposure.index[-1])]
    paths = []
    pos = exposure.index.get_indexer(spikes)
    for p in pos:
        if p < 1 or p + horizon >= len(exposure):
            continue
        base = exposure.iloc[p - 1]
        paths.append(exposure.iloc[p:p + horizon + 1].values - base)
    if not paths:
        return {"n_spikes": 0, "mean_response": None,
                "cumulative_10d": None}
    arr = np.stack(paths)
    mean_resp = arr.mean(axis=0)
    return {"n_spikes": len(paths),
            "mean_response": mean_resp.tolist(),
            "cumulative_10d": float(mean_resp[min(horizon, len(mean_resp) - 1)]),
            "se_response": (arr.std(axis=0, ddof=1) / np.sqrt(len(paths))).tolist()
            if len(paths) > 1 else None}


def pool_impulse_responses(responses: list[dict]) -> dict:
    """Pool several vix_spike_impulse_response outputs (e.g. one run per
    test fold) into one, exactly as if every spike path had been stacked.

    Each run's (n, mean, se) recovers its per-spike sum of squares
    (s^2 = se^2 * n), so the pooled per-spike variance is the usual
    within-run + between-run decomposition and the pooled SE is per-spike,
    over all N spikes. Runs with no spikes are skipped; a run with one spike
    (se None) contributes its mean and zero within-run variance."""
    runs = [r for r in responses if r and r.get("mean_response") is not None]
    if not runs:
        return {"n_spikes": 0, "mean_response": None, "cumulative_10d": None,
                "n_runs": 0}
    n = np.array([r["n_spikes"] for r in runs], dtype=float)
    m = np.stack([np.asarray(r["mean_response"], dtype=float) for r in runs])
    se = np.stack([np.asarray(r["se_response"], dtype=float)
                   if r.get("se_response") is not None
                   else np.zeros(m.shape[1]) for r in runs])
    N = n.sum()
    mean = (n[:, None] * m).sum(axis=0) / N
    within = ((n - 1)[:, None] * se ** 2 * n[:, None]).sum(axis=0)
    between = (n[:, None] * (m - mean) ** 2).sum(axis=0)
    out = {"n_spikes": int(N), "mean_response": mean.tolist(),
           "cumulative_10d": float(mean[-1]), "n_runs": len(runs),
           "se_response": None}
    if N > 1:
        var = (within + between) / (N - 1)
        out["se_response"] = np.sqrt(var / N).tolist()
    return out


def diagnostics_report(exposure: pd.Series, vix: pd.Series) -> dict:
    """Both J.4 diagnostics in one row (per run, computed on the test window)."""
    return {"exposure_vix": exposure_vix_regression(exposure, vix),
            "impulse_response": vix_spike_impulse_response(exposure, vix)}
