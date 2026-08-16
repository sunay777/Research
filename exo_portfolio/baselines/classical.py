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

from exo_portfolio.config import Config
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
# M10 — extra traditional (non-RL) weight-rule baselines (all causal)
# ---------------------------------------------------------------------------

def _rolling_lr(prices: pd.DataFrame) -> np.ndarray:
    """Daily log-returns, row i -> day i+1 (same convention as the others)."""
    p = prices.values.astype(np.float64)
    return np.log(p[1:] / p[:-1])


def inverse_vol_targets(prices: pd.DataFrame, window: int = 60) -> np.ndarray:
    """Risk parity / inverse-volatility: w_i ∝ 1 / σ̂_i on the trailing `window`
    of daily log-returns (data <= t only), zero cash. Ignores both the mean and
    the cross-asset correlations — the classic naive risk-parity heuristic.
    All-cash before the window is full.
    """
    T, N = prices.shape
    lr = _rolling_lr(prices)
    targets = np.zeros((T - 1, N + 1))
    targets[:, 0] = 1.0
    for t in range(T - 1):
        hist = lr[max(0, t - window):t]                # returns <= day t
        if len(hist) < window:
            continue
        vol = hist.std(axis=0, ddof=1)
        inv = np.where(vol > 0, 1.0 / vol, 0.0)
        s = inv.sum()
        if s > 0:
            targets[t, 1:] = inv / s
            targets[t, 0] = 0.0
    return targets


def min_variance_targets(prices: pd.DataFrame, window: int = 60,
                         ridge: float = 1e-4) -> np.ndarray:
    """Global minimum-variance portfolio: w ∝ (Σ̂ + ridge·I)⁻¹ 1, negatives
    clipped and renormalised (long-only heuristic); all-cash when the window is
    not yet full or nothing is positive. Uses the SAME ridge-regularised
    covariance as `mean_variance_targets` but ignores expected returns.
    """
    T, N = prices.shape
    lr = _rolling_lr(prices)
    ones = np.ones(N)
    targets = np.zeros((T - 1, N + 1))
    targets[:, 0] = 1.0
    for t in range(T - 1):
        hist = lr[max(0, t - window):t]
        if len(hist) < window:
            continue
        sigma = np.cov(hist, rowvar=False) + ridge * np.eye(N)
        raw = np.clip(np.linalg.solve(sigma, ones), 0.0, None)
        s = raw.sum()
        if s > 0:
            targets[t, 1:] = raw / s
            targets[t, 0] = 0.0
    return targets


def max_sharpe_targets(prices: pd.DataFrame, window: int = 60,
                       ridge: float = 1e-4) -> np.ndarray:
    """Naive (diagonal-covariance) tangency / maximum-Sharpe portfolio:
    w_i ∝ (μ̂_i / (σ̂_i² + ridge))₊ over the trailing `window` (data <= t only),
    renormalised (long-only). This is the Sharpe-maximising portfolio under a
    diagonal covariance model — it tilts toward high per-asset return/variance
    and, unlike the full-covariance `mean_variance_targets`, ignores cross-asset
    correlations (so the two are genuinely distinct baseline rows). All-cash
    before the window is full or when no asset has positive expected return.
    """
    T, N = prices.shape
    lr = _rolling_lr(prices)
    targets = np.zeros((T - 1, N + 1))
    targets[:, 0] = 1.0
    for t in range(T - 1):
        hist = lr[max(0, t - window):t]
        if len(hist) < window:
            continue
        mu = hist.mean(axis=0)
        var = hist.var(axis=0, ddof=1) + ridge
        raw = np.clip(mu / var, 0.0, None)
        s = raw.sum()
        if s > 0:
            targets[t, 1:] = raw / s
            targets[t, 0] = 0.0
    return targets


def momentum_targets(prices: pd.DataFrame, lookback: int = 60,
                     top_k: int = 10) -> np.ndarray:
    """Cross-sectional momentum: hold the `top_k` assets by trailing `lookback`
    cumulative log-return (data <= t only), equal-weighted, zero cash. All-cash
    before the window is full. `k` (and `lookback`) come from cfg.baselines.
    """
    T, N = prices.shape
    lr = _rolling_lr(prices)
    k = int(min(max(top_k, 1), N))
    targets = np.zeros((T - 1, N + 1))
    targets[:, 0] = 1.0
    for t in range(T - 1):
        hist = lr[max(0, t - lookback):t]
        if len(hist) < lookback:
            continue
        trailing = hist.sum(axis=0)                    # cumulative trailing return
        winners = np.argsort(trailing)[-k:]            # top-k performers
        targets[t, 1:][winners] = 1.0 / k
        targets[t, 0] = 0.0
    return targets


# ---------------------------------------------------------------------------
# M10 — random baselines (all reproducible from a seed)
# ---------------------------------------------------------------------------

# Fixed per-baseline stream indices so each random baseline is independently
# reproducible from a single integer seed (SeedSequence-style spawning).
RANDOM_BASELINE_STREAM = {"random_weight": 1, "random_buy_and_hold": 2,
                          "random_action": 3}


def random_baseline_rng(seed: int, name: str) -> np.random.Generator:
    """Independent, reproducible RNG per (seed, random-baseline name)."""
    return np.random.default_rng([int(seed), RANDOM_BASELINE_STREAM[name]])


def random_weight_targets(T: int, N: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Long-only random weights each step: a fresh Dirichlet draw over the
    N assets + cash (N+1 simplex) at every day. Rows sum to 1, non-negative.
    """
    return rng.dirichlet(np.ones(N + 1), size=T - 1)


def random_buy_and_hold_targets(T: int, N: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Draw one long-only weight vector (Dirichlet over N assets + cash) and
    hold it — every row identical (a random constant allocation)."""
    w = rng.dirichlet(np.ones(N + 1))
    return np.tile(w, (T - 1, 1))


def random_policy_in_env(features, cfg: Config, start: int, end: int,
                         seed: int | None = None,
                         deterministic: bool = False) -> dict:
    """Random-action baseline: sample `env.action_space` each step and roll out.

    Same return shape as sb3_baselines.evaluate_policy_window /
    algos.ppo.evaluate_agent_window. CRITICAL: `env.reset(seed)` only seeds the
    observation RNG, so the action sampler is seeded EXPLICITLY via
    `action_space.seed(...)` (Manual A.1.3) — otherwise the rollout is not
    reproducible.
    """
    from exo_portfolio.envs.portfolio_env import PortfolioEnv

    seed = cfg.seed if seed is None else seed
    env = PortfolioEnv(features, cfg, start, end)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(int(seed))                   # seed the action sampler
    rhos, taus, dates, exposures = [], [], [], []
    done = False
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        rhos.append(info["step_log_return"])
        taus.append(info["turnover"])
        dates.append(info["date"])
        exposures.append(1.0 - float(info["weights"][0]))
        done = terminated or truncated
    row = summarize(np.array(rhos), np.array(taus))
    row["final_value"] = float(np.exp(np.sum(rhos)))
    return {"metrics": row, "log_returns": np.array(rhos),
            "turnover": np.array(taus), "dates": dates,
            "exposure": np.array(exposures)}


# ---------------------------------------------------------------------------
# Registries + unified series builder (so every baseline flows through the
# shared E.3 simulator and reaches the same tables as the RL agents)
# ---------------------------------------------------------------------------

def deterministic_target_baselines(prices: pd.DataFrame,
                                   cfg: Config) -> dict[str, np.ndarray]:
    """name -> (T-1, N+1) target rows for every deterministic simplex baseline
    (classical + M10 traditional). Hyper-parameters come from cfg.baselines."""
    T, N = prices.shape
    b = cfg.baselines
    return {
        "equal_weight": equal_weight_targets(T, N),
        "vol_overlay": vol_overlay_targets(prices),
        "mean_variance": mean_variance_targets(prices),
        "inverse_vol": inverse_vol_targets(prices, b.trad_window),
        "min_variance": min_variance_targets(prices, b.trad_window, b.trad_ridge),
        "max_sharpe": max_sharpe_targets(prices, b.trad_window, b.trad_ridge),
        "momentum": momentum_targets(prices, b.momentum_lookback, b.momentum_top_k),
    }


def random_target_baselines(prices: pd.DataFrame,
                            seed: int) -> dict[str, np.ndarray]:
    """name -> (T-1, N+1) target rows for the seeded random target baselines."""
    T, N = prices.shape
    return {
        "random_weight": random_weight_targets(
            T, N, random_baseline_rng(seed, "random_weight")),
        "random_buy_and_hold": random_buy_and_hold_targets(
            T, N, random_baseline_rng(seed, "random_buy_and_hold")),
    }


def simulate_baseline_series(prices: pd.DataFrame, targets: np.ndarray,
                             transaction_cost: float) -> dict:
    """Run target rows through the shared simulator and package them in the
    same shape as the RL evaluators (metrics + per-day series), so baselines
    reach the regime-conditional and cost-sensitivity tables.

    exposure = 1 - cash weight, dated at the day the return is realised
    (prices.index[1:]) — matching PortfolioEnv's info["date"] convention.
    """
    sim = simulate_target_weights(prices, targets, transaction_cost)
    row = summarize(sim["log_returns"], sim["turnover"])
    row["final_value"] = sim["final_value"]
    return {"metrics": row, "log_returns": sim["log_returns"],
            "turnover": sim["turnover"], "dates": list(prices.index[1:]),
            "exposure": 1.0 - targets[:, 0]}


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
