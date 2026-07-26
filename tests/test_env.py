"""M2 acceptance tests for the Exo-MDP environment (Manual E.6)."""

import numpy as np
import pytest

from exo_portfolio.config import Config
from exo_portfolio.data.align import build_features
from exo_portfolio.envs.portfolio_env import PortfolioEnv, softmax
from exo_portfolio.eval.metrics import rolling_sharpe

from tests.conftest import make_bundle


def make_env(cost=0.001, seed=0, n_assets=3, K=5):
    cfg = Config()
    cfg.data.price_window = 10
    cfg.env.reward_window_K = K
    cfg.env.transaction_cost = cost
    fs = build_features(**make_bundle(seed=seed, start="2015-01-01",
                                      end="2016-12-31", n_assets=n_assets), cfg=cfg)
    return PortfolioEnv(fs, cfg), fs


def drifted(env):
    """The w' the NEXT step will produce, computable from public state."""
    y = np.ones(env.N + 1)
    y[1:] = env.prices[env.t + 1] / env.prices[env.t]
    g = env.w @ y
    return (y * env.w) / g, g


def test_reset_state():
    env, _ = make_env()
    obs, info = env.reset(seed=0)
    assert info["v"] == 1.0
    w = obs["endo"][: env.N + 1]
    assert w[0] == 1.0 and np.all(w[1:] == 0)          # all cash (E.5)
    assert obs["endo"][env.N + 1] == 1.0               # cash fraction
    assert np.all(obs["endo"][-env.K:] == 0)           # empty return window


def test_conservation_no_cost_no_rebalance():
    """c=0 and a_t reproducing w_t' => v_{t+1} = v_t * g exactly (E.6)."""
    env, _ = make_env(cost=0.0)
    env.reset(seed=0)
    env.step(np.zeros(env.N + 1))                      # move to interior weights
    for _ in range(5):
        w_prime, g_expected = drifted(env)
        v_before = env.v
        a = np.log(w_prime)                            # softmax(log w') == w'
        _, _, _, _, info = env.step(a)
        assert info["turnover"] == pytest.approx(0.0, abs=1e-12)
        assert info["g"] == pytest.approx(g_expected, rel=1e-15)
        assert env.v == v_before * info["g"]           # exact: mu == 0


def test_cost_monotonicity():
    """Fixed g (same state, same prices): higher turnover => lower v (E.6)."""
    results = {}
    for label, target in {"low": None, "high": "far"}.items():
        env, _ = make_env(cost=0.002)
        env.reset(seed=0)
        env.step(np.zeros(env.N + 1))
        w_prime, _ = drifted(env)
        if target is None:                             # tiny rebalance
            a = np.log(w_prime)
        else:                                          # slam everything to cash
            a = np.zeros(env.N + 1); a[0] = 10.0
        _, _, _, _, info = env.step(a)
        results[label] = info
    assert results["high"]["g"] == pytest.approx(results["low"]["g"])  # g is action-free
    assert results["high"]["turnover"] > results["low"]["turnover"]
    assert results["high"]["v"] < results["low"]["v"]


def test_determinism():
    """Same (seed, actions) => identical trajectory (E.6)."""
    rng = np.random.default_rng(7)
    actions = rng.normal(size=(40, 4))
    traj = []
    for _ in range(2):
        env, _ = make_env(seed=3)
        obs, _ = env.reset(seed=11)
        vs, rewards, endos = [], [], [obs["endo"]]
        for a in actions:
            obs, r, term, _, info = env.step(a)
            vs.append(info["v"]); rewards.append(r); endos.append(obs["endo"])
        traj.append((np.array(vs), np.array(rewards), np.stack(endos)))
    for x, y in zip(traj[0], traj[1]):
        assert np.array_equal(x, y)


def test_markov_reward_from_endo_window():
    """Once K steps have elapsed, the reward must be reconstructable from the
    endo return-window alone (E.6). This is why the window is in the state."""
    env, _ = make_env(K=5)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for i in range(12):
        obs, r, _, _, _ = env.step(rng.normal(size=env.N + 1))
        if i + 1 >= env.K:
            window = obs["endo"][-env.K:]
            assert r == pytest.approx(rolling_sharpe(window, env.eps), rel=1e-5)


def test_no_lookahead_in_env():
    """Obs and rewards up to pointer t must not depend on prices > t (E.6)."""
    env_a, fs = make_env(seed=5)
    # clone features, corrupt strictly-future prices/exo rows
    import copy
    fs_b = copy.deepcopy(fs)
    cut = 30
    fs_b.prices.iloc[cut + 1:] *= 3.14
    fs_b.exo_actor.iloc[cut + 1:] += 1.0
    fs_b.exo_critic_extra.iloc[cut + 1:] += 1.0
    env_b = PortfolioEnv(fs_b, env_a.cfg)

    rng = np.random.default_rng(2)
    actions = rng.normal(size=(cut, env_a.N + 1))
    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    assert np.array_equal(obs_a["endo"], obs_b["endo"])
    for i, a in enumerate(actions):
        obs_a, ra, _, _, ia = env_a.step(a)
        obs_b, rb, _, _, ib = env_b.step(a)
        assert ra == rb and ia["v"] == ib["v"], f"future prices leaked at step {i}"
        if i < cut - 1:   # obs at pointer <= cut must still match
            for k in ("endo", "exo_actor", "exo_critic_extra"):
                assert np.array_equal(obs_a[k], obs_b[k]), f"{k} leaked at step {i}"


def test_softmax_valid_weights():
    for x in (np.zeros(5), np.array([1000., -1000., 0., 3., 2.]),
              np.random.default_rng(0).normal(size=8) * 50):
        w = softmax(x)
        assert w.sum() == pytest.approx(1.0) and (w >= 0).all()


def test_episode_terminates_at_window_end():
    env, fs = make_env()
    env2 = PortfolioEnv(fs, env.cfg, start=0, end=25)
    env2.reset(seed=0)
    term, steps = False, 0
    while not term:
        _, _, term, _, _ = env2.step(np.zeros(env2.N + 1))
        steps += 1
    assert steps == 25
    with pytest.raises(AssertionError):
        env2.step(np.zeros(env2.N + 1))
