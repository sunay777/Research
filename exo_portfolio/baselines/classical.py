"""Non-RL baselines (Manual Part K) — the credibility bar.

All portfolio baselines are simulated with the SAME closed-form dynamics as
the environment (E.3 steps 1-7), so their numbers are directly comparable to
the RL agents': same cost model, same timing convention (the target chosen at
day t is implemented against the price-drifted weights using y_{t+1}).

Every target-weight rule is CAUSAL: row t is a function of data dated <= t.

Baselines:
- equal_weight   : 1/N across assets, rebalanced every step
- buy_and_hold_index : the ^GSPC index itself (no costs; pure price series)
- mean_variance  : rolling Markowitz estimate, long-only heuristic
                   (w ∝ Σ̂⁻¹μ̂ with ridge shrinkage, negatives clipped,
                   renormalised; all-cash if nothing is attractive)
- vol_overlay    : 1/N scaled by min(1, target_vol / trailing realised vol),
                   remainder in cash — the one-line heuristic the learned
                   agent must beat for the thesis to mean anything (K).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exo_portfolio.eval.metrics import summarize, TRADING_DAYS


# ---------------------------------------------------------------------------
# The shared simulator (E.3, identical arithmetic to PortfolioEnv.step)
# ---------------------------------------------------------------------------

def simulate_target_weights(prices: pd.DataFrame, targets: np.ndarray,
                            transaction_cost: float) -> dict:
    """Run target-weight rows through the E.3 dynamics.

    prices : (T, N) adjusted closes.
    targets: (T-1, N+1) rows sum to 1, index 0 = cash. Row t is the target
             chosen at day t (from data <= t), implemented against the
             drifted weights using y_{t+1} — exactly PortfolioEnv.step.

    Returns step log-returns, per-step turnover, and the equity curve.
    """
    p = prices.values.astype(np.float64)
    T, N = p.shape
    assert targets.shape == (T - 1, N + 1)
    assert np.allclose(targets.sum(axis=1), 1.0) and (targets >= 0).all()

    w = np.zeros(N + 1)
    w[0] = 1.0                                   # start all-cash (E.5)
    v = 1.0
    rhos = np.empty(T - 1)
    taus = np.empty(T - 1)
    values = np.empty(T - 1)

    for t in range(T - 1):
        y = np.concatenate([[1.0], p[t + 1] / p[t]])
        g = w @ y                                # 1
        w_drift = (y * w) / g                    # 2
        w_new = targets[t]                       # 3
        tau = np.abs(w_new - w_drift).sum()      # 4
        mu = transaction_cost * tau              # 5
        v_new = v * g * (1.0 - mu)               # 6
        rhos[t] = np.log(v_new / v)              # 7
        taus[t] = tau
        values[t] = v_new
        w, v = w_new, v_new

    return {"log_returns": rhos, "turnover": taus, "values": values,
            "final_value": float(v)}


# ---------------------------------------------------------------------------
# Target-weight rules (all causal)
# ---------------------------------------------------------------------------

def equal_weight_targets(T: int, N: int) -> np.ndarray:
    """1/N across assets, zero cash, every step."""
    targets = np.full((T - 1, N + 1), 1.0 / N)
    targets[:, 0] = 0.0
    return targets


def vol_overlay_targets(prices: pd.DataFrame, window: int = 20,
                        target_vol: float = 0.10,
                        periods_per_year: int = TRADING_DAYS) -> np.ndarray:
    """1/N scaled by min(1, target_vol / trailing annualised realised vol of
    the 1/N portfolio); remainder in cash. Row t uses returns dated <= t only.
    Before `window` observations exist, exposure defaults to 1 (fully 1/N).
    """
    p = prices.values.astype(np.float64)
    T, N = p.shape
    port_lr = np.log((p[1:] / p[:-1]).mean(axis=1))   # 1/N daily log-return; row i -> day i+1

    exposure = np.ones(T - 1)
    for t in range(T - 1):
        hist = port_lr[max(0, t - window):t]          # returns knowable at day t
        if len(hist) >= window:
            ann_vol = hist.std(ddof=1) * np.sqrt(periods_per_year)
            if ann_vol > 0:
                exposure[t] = min(1.0, target_vol / ann_vol)

    targets = np.empty((T - 1, N + 1))
    targets[:, 1:] = (exposure / N)[:, None]
    targets[:, 0] = 1.0 - exposure
    return targets


def mean_variance_targets(prices: pd.DataFrame, window: int = 60,
                          ridge: float = 1e-4) -> np.ndarray:
    """Rolling Markowitz: w ∝ (Σ̂ + ridge·I)⁻¹ μ̂ on the trailing `window` of
    daily log-returns (data <= t only), negatives clipped to zero and the rest
    renormalised (long-only heuristic); all-cash when no asset is attractive
    or the window is not yet full. Documented simplification, not an exact
    constrained optimiser.
    """
    p = prices.values.astype(np.float64)
    T, N = p.shape
    lr = np.log(p[1:] / p[:-1])                       # row i -> day i+1

    targets = np.zeros((T - 1, N + 1))
    targets[:, 0] = 1.0                               # default: cash
    for t in range(T - 1):
        hist = lr[max(0, t - window):t]               # returns <= day t
        if len(hist) < window:
            continue
        mu = hist.mean(axis=0)
        sigma = np.cov(hist, rowvar=False) + ridge * np.eye(N)
        raw = np.linalg.solve(sigma, mu)
        raw = np.clip(raw, 0.0, None)
        s = raw.sum()
        if s > 0:
            targets[t, 1:] = raw / s
            targets[t, 0] = 0.0
    return targets


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def buy_and_hold_index(index: pd.Series) -> dict:
    """Metrics of holding the broad index itself (no costs) — Manual K."""
    lr = np.log(index.values[1:] / index.values[:-1])
    return {"log_returns": lr, "turnover": np.zeros(len(lr)),
            "values": np.exp(np.cumsum(lr)),
            "final_value": float(np.exp(lr.sum()))}


def run_classical_baselines(prices: pd.DataFrame, index: pd.Series,
                            transaction_cost: float) -> dict[str, dict]:
    """All Part-K baselines -> {name: summarize(...) row}."""
    T, N = prices.shape
    runs = {
        "equal_weight": simulate_target_weights(
            prices, equal_weight_targets(T, N), transaction_cost),
        "vol_overlay": simulate_target_weights(
            prices, vol_overlay_targets(prices), transaction_cost),
        "mean_variance": simulate_target_weights(
            prices, mean_variance_targets(prices), transaction_cost),
        "buy_and_hold_index": buy_and_hold_index(index.loc[prices.index]),
    }
    return {name: summarize(r["log_returns"], r["turnover"])
            for name, r in runs.items()}
