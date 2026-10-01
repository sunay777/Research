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
    paths, dates = [], []
    pos = exposure.index.get_indexer(spikes)
    for d, p in zip(spikes, pos):
        if p < 1 or p + horizon >= len(exposure):
            continue
        base = exposure.iloc[p - 1]
        paths.append(exposure.iloc[p:p + horizon + 1].values - base)
        dates.append(pd.Timestamp(d).strftime("%Y-%m-%d"))
    if not paths:
        return {"n_spikes": 0, "mean_response": None,
                "cumulative_10d": None, "spike_dates": [], "paths": []}
    arr = np.stack(paths)
    mean_resp = arr.mean(axis=0)
    return {"n_spikes": len(paths),
            "mean_response": mean_resp.tolist(),
            "cumulative_10d": float(mean_resp[min(horizon, len(mean_resp) - 1)]),
            "se_response": (arr.std(axis=0, ddof=1) / np.sqrt(len(paths))).tolist()
            if len(paths) > 1 else None,
            # per-spike detail, so runs sharing the same spikes (seeds of one
            # fold) can be pooled by event — see pool_impulse_by_event
            "spike_dates": dates, "paths": arr.tolist()}


def pool_impulse_by_event(responses: list[dict]) -> dict:
    """Pool vix_spike_impulse_response outputs from many runs of one cell
    (seeds x folds) into one response, clustered by spike EVENT.

    Every seed of a fold sees the same VIX spikes, so stacking all runs' paths
    would count each event once per seed and shrink the SE by ~sqrt(n_seeds)
    spuriously. Instead each event's path is first averaged over the runs
    that observed it (seeds = replicate policies on the same market), then the
    mean and SE are taken across the distinct events (the market sample)."""
    by_event: dict[str, list] = {}
    n_runs = 0
    for r in responses:
        if not r or not r.get("paths"):
            continue
        n_runs += 1
        for d, path in zip(r["spike_dates"], r["paths"]):
            by_event.setdefault(d, []).append(path)
    if not by_event:
        return {"n_spikes": 0, "mean_response": None, "cumulative_10d": None,
                "se_response": None, "n_runs": n_runs}
    ev = np.stack([np.mean(v, axis=0) for _, v in sorted(by_event.items())])
    mean = ev.mean(axis=0)
    return {"n_spikes": len(ev), "mean_response": mean.tolist(),
            "cumulative_10d": float(mean[-1]),
            "se_response": (ev.std(axis=0, ddof=1) / np.sqrt(len(ev))).tolist()
            if len(ev) > 1 else None,
            "n_runs": n_runs,
            "runs_per_event": float(np.mean([len(v) for v in by_event.values()]))}


def summarize_exposure_vix(diag: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Per-cell summary of the per-run exposure~VIX regressions.

    `diag` has one row per run (cell, seed, fold, beta, pvalue). beta_mean is
    over all runs; beta_se_folds averages seeds within each fold first and
    takes the SE across folds (market windows); beta_se_seeds averages folds
    within each seed and takes the SE across seeds (training replicates on
    the same market history). frac_* count runs whose HAC test rejects at
    `alpha` in each direction."""
    rows = []
    for cell, g in diag.groupby("cell"):
        by_fold = g.groupby("fold")["beta"].mean()
        by_seed = g.groupby("seed")["beta"].mean()
        sig = g["pvalue"] < alpha
        rows.append({
            "cell": cell, "n_runs": len(g), "n_seeds": g["seed"].nunique(),
            "n_folds": g["fold"].nunique(),
            "beta_mean": g["beta"].mean(),
            "beta_se_folds": by_fold.std(ddof=1) / np.sqrt(len(by_fold))
            if len(by_fold) > 1 else np.nan,
            "beta_se_seeds": by_seed.std(ddof=1) / np.sqrt(len(by_seed))
            if len(by_seed) > 1 else np.nan,
            "frac_beta_negative": float((g["beta"] < 0).mean()),
            "frac_sig_negative": float((sig & (g["beta"] < 0)).mean()),
            "frac_sig_positive": float((sig & (g["beta"] > 0)).mean()),
            "folds_mean_negative": int((by_fold < 0).sum()),
        })
    return pd.DataFrame(rows).set_index("cell")


def diagnostics_report(exposure: pd.Series, vix: pd.Series) -> dict:
    """Both J.4 diagnostics in one row (per run, computed on the test window)."""
    return {"exposure_vix": exposure_vix_regression(exposure, vix),
            "impulse_response": vix_spike_impulse_response(exposure, vix)}
