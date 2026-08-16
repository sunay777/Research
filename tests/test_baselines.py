"""M3 acceptance: classical baselines run end-to-end, are causal, and the
shared simulator agrees exactly with the PortfolioEnv dynamics (Manual K/F).
"""

import numpy as np
import pandas as pd
import pytest

from exo_portfolio.baselines.classical import (buy_and_hold_index,
                                               deterministic_target_baselines,
                                               equal_weight_targets,
                                               inverse_vol_targets,
                                               max_sharpe_targets,
                                               mean_variance_targets,
                                               min_variance_targets,
                                               momentum_targets,
                                               random_baseline_rng,
                                               random_buy_and_hold_targets,
                                               random_policy_in_env,
                                               random_weight_targets,
                                               run_classical_baselines,
                                               simulate_baseline_series,
                                               simulate_target_weights,
                                               vol_overlay_targets)
from exo_portfolio.config import Config
from exo_portfolio.data.align import build_features
from exo_portfolio.envs.portfolio_env import PortfolioEnv

from tests.conftest import make_bundle


@pytest.fixture(scope="module")
def market():
    bundle = make_bundle(seed=4, start="2014-01-01", end="2016-12-31", n_assets=4)
    return bundle


def test_equal_weight_end_to_end(market):
    prices = market["prices"]
    T, N = prices.shape
    out = simulate_target_weights(prices, equal_weight_targets(T, N), 0.001)
    assert len(out["log_returns"]) == T - 1
    assert np.isfinite(out["log_returns"]).all()
    assert out["final_value"] > 0
    # first step: all-cash -> 1/N is a full rebalance, turnover 2*? no —
    # |w_new - w_drift|: cash 1->0 plus N legs 0->1/N: 1 + N*(1/N) = 2
    assert out["turnover"][0] == pytest.approx(2.0)


def test_simulator_matches_env_exactly(market):
    """The baseline simulator and PortfolioEnv must produce the SAME wealth
    path for the same targets — one E.3 implementation cross-checks the other."""
    cfg = Config()
    cfg.data.price_window = 10
    cfg.env.reward_window_K = 5
    fs = build_features(**market, cfg=cfg)
    prices = fs.prices
    T, N = prices.shape

    targets = equal_weight_targets(T, N)
    sim = simulate_target_weights(prices, targets, cfg.env.transaction_cost)

    env = PortfolioEnv(fs, cfg)
    env.reset(seed=0)
    tiny = 1e-300                                  # log(w+tiny): exact for w>0
    for t in range(min(200, T - 1)):
        a = np.log(targets[t] + tiny)
        _, _, term, _, info = env.step(a)
        assert info["v"] == pytest.approx(sim["values"][t], rel=1e-9), f"step {t}"
        assert info["turnover"] == pytest.approx(sim["turnover"][t], rel=1e-6, abs=1e-9)
        if term:
            break


def test_vol_overlay_causal_and_bounded(market):
    prices = market["prices"]
    targets = vol_overlay_targets(prices)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert (targets >= 0).all() and (targets[:, 0] <= 1).all()
    # causality: perturbing prices after day t must not change rows <= t
    prices2 = prices.copy()
    cut = 300
    prices2.iloc[cut + 1:] *= 1.7
    targets2 = vol_overlay_targets(prices2)
    assert np.array_equal(targets[:cut + 1], targets2[:cut + 1])


def test_mean_variance_causal_and_valid(market):
    prices = market["prices"]
    targets = mean_variance_targets(prices)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert (targets >= 0).all()                    # long-only
    prices2 = prices.copy()
    cut = 300
    prices2.iloc[cut + 1:] *= 0.6
    targets2 = mean_variance_targets(prices2)
    assert np.array_equal(targets[:cut + 1], targets2[:cut + 1])


def test_buy_and_hold_index(market):
    out = buy_and_hold_index(market["index"])
    # pure price series: final value = last/first exactly
    expected = market["index"].iloc[-1] / market["index"].iloc[0]
    assert out["final_value"] == pytest.approx(expected, rel=1e-12)
    assert (out["turnover"] == 0).all()


def test_run_all_baselines(market):
    table = run_classical_baselines(market["prices"], market["index"], 0.001)
    assert set(table) == {"equal_weight", "vol_overlay", "mean_variance",
                          "buy_and_hold_index"}
    for name, row in table.items():
        assert np.isfinite(row["cum_log_return"]), name
        assert 0 <= row["max_drawdown"] <= 1, name


# --------------------------------------------------------------------------
# M10 — extra traditional weight-rule baselines: valid simplex + causal
# --------------------------------------------------------------------------

def _assert_valid_simplex(targets, T, N):
    assert targets.shape == (T - 1, N + 1)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert (targets >= 0).all()


def _assert_causal(fn, prices, cut=300):
    """Perturbing prices strictly after `cut` must not change rows <= cut."""
    targets = fn(prices)
    prices2 = prices.copy()
    prices2.iloc[cut + 1:] *= 1.7
    targets2 = fn(prices2)
    assert np.array_equal(targets[:cut + 1], targets2[:cut + 1])


@pytest.mark.parametrize("fn", [inverse_vol_targets, min_variance_targets,
                                max_sharpe_targets, momentum_targets])
def test_traditional_baselines_valid_and_causal(market, fn):
    prices = market["prices"]
    T, N = prices.shape
    _assert_valid_simplex(fn(prices), T, N)
    _assert_causal(fn, prices)


def test_momentum_holds_exactly_top_k(market):
    prices = market["prices"]
    T, N = prices.shape
    targets = momentum_targets(prices, lookback=40, top_k=2)
    invested = targets[targets[:, 0] == 0.0][:, 1:]
    assert invested.size, "no rows ever invested"
    # each invested row is equal-weight over exactly k=2 names
    assert np.allclose(invested[invested > 0], 0.5)
    assert (np.count_nonzero(invested, axis=1) == 2).all()


def test_traditional_run_through_simulator(market):
    prices = market["prices"]
    T, N = prices.shape
    out = simulate_target_weights(prices, inverse_vol_targets(prices), 0.001)
    assert len(out["log_returns"]) == T - 1
    assert np.isfinite(out["log_returns"]).all() and out["final_value"] > 0


# --------------------------------------------------------------------------
# M10 — random baselines: reproducible from a seed + valid + right shape
# --------------------------------------------------------------------------

def test_random_weight_deterministic_and_valid(market):
    prices = market["prices"]
    T, N = prices.shape
    a = random_weight_targets(T, N, random_baseline_rng(0, "random_weight"))
    b = random_weight_targets(T, N, random_baseline_rng(0, "random_weight"))
    c = random_weight_targets(T, N, random_baseline_rng(1, "random_weight"))
    _assert_valid_simplex(a, T, N)
    assert np.array_equal(a, b), "same seed must reproduce exactly"
    assert not np.array_equal(a, c), "different seed must differ"
    out = simulate_target_weights(prices, a, 0.001)
    assert np.isfinite(out["log_returns"]).all()


def test_random_buy_and_hold_constant_and_valid(market):
    prices = market["prices"]
    T, N = prices.shape
    a = random_buy_and_hold_targets(T, N, random_baseline_rng(0, "random_buy_and_hold"))
    b = random_buy_and_hold_targets(T, N, random_baseline_rng(0, "random_buy_and_hold"))
    _assert_valid_simplex(a, T, N)
    assert np.array_equal(a, b)
    assert np.allclose(a, a[0]), "buy-and-hold must be a constant allocation"
    # independent stream from random_weight (first rows differ)
    rw = random_weight_targets(T, N, random_baseline_rng(0, "random_weight"))
    assert not np.array_equal(a[0], rw[0])


def test_random_policy_in_env_deterministic_and_valid(market):
    cfg = Config()
    cfg.data.price_window = 10
    cfg.env.reward_window_K = 5
    fs = build_features(**market, cfg=cfg)
    a = random_policy_in_env(fs, cfg, 0, 120, seed=7)
    b = random_policy_in_env(fs, cfg, 0, 120, seed=7)
    d = random_policy_in_env(fs, cfg, 0, 120, seed=8)
    # explicit action_space seeding makes the rollout reproducible
    assert np.array_equal(a["log_returns"], b["log_returns"])
    assert not np.array_equal(a["log_returns"], d["log_returns"])
    assert len(a["log_returns"]) == 120 and len(a["dates"]) == 120
    assert ((a["exposure"] >= 0) & (a["exposure"] <= 1)).all()
    m = a["metrics"]
    assert np.isfinite(m["sharpe"]) and 0 <= m["max_drawdown"] <= 1


def test_baseline_series_shape_and_registry(market):
    cfg = Config()
    prices = market["prices"]
    T, N = prices.shape
    det = deterministic_target_baselines(prices, cfg)
    assert set(det) == {"equal_weight", "vol_overlay", "mean_variance",
                        "inverse_vol", "min_variance", "max_sharpe", "momentum"}
    ser = simulate_baseline_series(prices, det["equal_weight"],
                                   cfg.env.transaction_cost)
    for key in ("log_returns", "turnover", "dates", "exposure"):
        assert len(ser[key]) == T - 1, key
    assert ((ser["exposure"] >= 0) & (ser["exposure"] <= 1)).all()
    assert np.isfinite(ser["metrics"]["sharpe"])
