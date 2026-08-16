"""M4 acceptance (smoke level): SB3 baselines run end-to-end at both
information levels, deterministically, on synthetic data. The full
"reproduce a known FinRL-style number" check requires the real 2M-step
budget and runs on the cluster — see README (M4 notes).
"""

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")

from exo_portfolio.baselines.sb3_baselines import (InfoLevelWrapper,
                                                   evaluate_policy_window,
                                                   make_env, train_sb3)
from exo_portfolio.config import Config
from exo_portfolio.data.align import build_features
from exo_portfolio.envs.portfolio_env import PortfolioEnv

from tests.conftest import make_bundle


@pytest.fixture(scope="module")
def setup():
    cfg = Config()
    cfg.data.price_window = 10
    cfg.env.reward_window_K = 5
    fs = build_features(**make_bundle(seed=2, start="2014-01-01",
                                      end="2016-12-31", n_assets=3), cfg=cfg)
    return cfg, fs


def test_wrapper_dims_and_content(setup):
    cfg, fs = setup
    base = PortfolioEnv(fs, cfg)
    d_endo = base.observation_space["endo"].shape[0]
    d_exo = base.observation_space["exo_actor"].shape[0]

    w0 = InfoLevelWrapper(PortfolioEnv(fs, cfg), "cell0")
    w1 = InfoLevelWrapper(PortfolioEnv(fs, cfg), "cell1")
    o0, _ = w0.reset(seed=0)
    o1, _ = w1.reset(seed=0)
    assert o0.shape == (d_endo,)
    assert o1.shape == (d_endo + d_exo,)
    assert np.array_equal(o1[:d_endo], o0)          # endo block identical
    # privileged block must NOT be reachable in either level
    d_priv = base.observation_space["exo_critic_extra"].shape[0]
    assert d_priv > 0 and o1.shape[0] == d_endo + d_exo  # no room for it


def test_smoke_train_and_eval(setup):
    """A tiny PPO run must execute, save, and evaluate to finite metrics."""
    cfg, fs = setup
    cfg.train.algo = "ppo"
    model = train_sb3(cfg, fs, 0, 120, level="cell1",
                      run_dir="/tmp/sb3_smoke", total_steps=512)
    out = evaluate_policy_window(model, fs, cfg, 120, 200, "cell1")
    m = out["metrics"]
    assert np.isfinite(m["cum_log_return"]) and np.isfinite(m["sharpe"])
    assert 0 <= m["max_drawdown"] <= 1
    assert len(out["log_returns"]) == 80


def test_eval_deterministic(setup):
    cfg, fs = setup
    model = train_sb3(cfg, fs, 0, 120, level="cell0",
                      run_dir="/tmp/sb3_smoke0", total_steps=256)
    a = evaluate_policy_window(model, fs, cfg, 120, 180, "cell0")
    b = evaluate_policy_window(model, fs, cfg, 120, 180, "cell0")
    assert np.array_equal(a["log_returns"], b["log_returns"])


def test_algo_class_covers_finrl_family():
    from exo_portfolio.baselines.sb3_baselines import (OFF_POLICY, ON_POLICY,
                                                       _algo_class)
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    assert _algo_class("ddpg") is DDPG and _algo_class("td3") is TD3
    assert _algo_class("ppo") is PPO and _algo_class("sac") is SAC
    assert _algo_class("a2c") is A2C
    assert set(ON_POLICY) == {"ppo", "a2c"}
    assert set(OFF_POLICY) == {"sac", "ddpg", "td3"}


@pytest.mark.parametrize("algo", ["ddpg", "td3"])
def test_smoke_offpolicy_train_and_eval(setup, algo):
    """An off-policy (replay-buffer) SB3 baseline trains end-to-end at a tiny
    budget and evaluates to finite metrics — the guard must not feed it the
    on-policy-only gae_lambda / clip hyper-parameters."""
    cfg, fs = setup
    cfg.train.algo = algo
    model = train_sb3(cfg, fs, 0, 120, level="cell1",
                      run_dir=f"/tmp/sb3_{algo}", total_steps=200)
    out = evaluate_policy_window(model, fs, cfg, 120, 180, "cell1")
    m = out["metrics"]
    assert np.isfinite(m["cum_log_return"]) and np.isfinite(m["sharpe"])
    assert 0 <= m["max_drawdown"] <= 1
    assert len(out["log_returns"]) == 60
