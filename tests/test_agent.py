"""M5 acceptance tests: ablation switches, privileged-info discipline,
GAE correctness, and the overfit sanity gate (Manual G / F)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from exo_portfolio.algos.ppo import PPO, _to_tensors
from exo_portfolio.algos.rollout import RolloutBuffer
from exo_portfolio.config import Config
from exo_portfolio.data.align import build_features
from exo_portfolio.envs.portfolio_env import PortfolioEnv
from exo_portfolio.models.agent import (CELL_PRESETS, ExoActorCritic,
                                        model_cfg_for_cell)

from tests.conftest import make_bundle


N_ASSETS, WINDOW = 3, 10


@pytest.fixture(scope="module")
def setup():
    cfg = Config()
    cfg.data.price_window = WINDOW
    cfg.env.reward_window_K = 5
    cfg.model.d_endo, cfg.model.d_exo = 16, 32          # small for tests
    fs = build_features(**make_bundle(seed=8, start="2014-01-01",
                                      end="2016-12-31", n_assets=N_ASSETS),
                        cfg=cfg)
    env = PortfolioEnv(fs, cfg)
    obs_dims = {k: int(np.prod(s.shape))
                for k, s in env.observation_space.spaces.items()}
    return cfg, fs, env, obs_dims


def _agent(cell, cfg, obs_dims):
    torch.manual_seed(0)
    return ExoActorCritic(obs_dims, N_ASSETS + 1,
                          model_cfg_for_cell(cell, cfg.model),
                          n_assets=N_ASSETS, price_window=WINDOW)


def _obs_batch(env, seed=0):
    obs, _ = env.reset(seed=seed)
    return _to_tensors(obs, "cpu")


def test_all_cells_instantiate_and_act(setup):
    cfg, fs, env, obs_dims = setup
    counts = {}
    for cell in CELL_PRESETS:
        agent = _agent(cell, cfg, obs_dims)
        obs = _obs_batch(env)
        a, logp, v = agent.act(obs)
        assert a.shape == (1, N_ASSETS + 1)
        assert torch.isfinite(a).all() and torch.isfinite(v).all()
        counts[cell] = agent.param_counts()
    # capacity control (Manual L): same-architecture pairs must match exactly
    assert counts["cell1"]["actor"] == counts["cell3"]["actor"]
    assert counts["cell2"]["actor"] == counts["cell4"]["actor"]


def test_actor_never_sees_privileged_info(setup):
    """Perturbing exo_critic_extra must never move the policy, in ANY cell."""
    cfg, fs, env, obs_dims = setup
    for cell in CELL_PRESETS:
        agent = _agent(cell, cfg, obs_dims)
        obs = _obs_batch(env)
        obs2 = {k: v.clone() for k, v in obs.items()}
        obs2["exo_critic_extra"] += 123.0
        d1, d2 = agent.distribution(obs), agent.distribution(obs2)
        assert torch.equal(d1.mean, d2.mean), f"{cell}: actor saw privilege"


def test_critic_privilege_only_when_asymmetric(setup):
    cfg, fs, env, obs_dims = setup
    for cell, preset in CELL_PRESETS.items():
        agent = _agent(cell, cfg, obs_dims)
        obs = _obs_batch(env)
        obs2 = {k: v.clone() for k, v in obs.items()}
        obs2["exo_critic_extra"] += 123.0
        v1, v2 = agent.forward_critic(obs), agent.forward_critic(obs2)
        if preset["critic"] == "asymmetric":
            assert not torch.equal(v1, v2), f"{cell}: privilege ignored"
        else:
            assert torch.equal(v1, v2), f"{cell}: symmetric critic leaked"


def test_cell0_ignores_exo_actor(setup):
    cfg, fs, env, obs_dims = setup
    agent = _agent("cell0", cfg, obs_dims)
    obs = _obs_batch(env)
    obs2 = {k: v.clone() for k, v in obs.items()}
    obs2["exo_actor"] += 7.0
    assert torch.equal(agent.distribution(obs).mean,
                       agent.distribution(obs2).mean)
    # while cell1 (concat) must react to exogenous input
    agent1 = _agent("cell1", cfg, obs_dims)
    assert not torch.equal(agent1.distribution(obs).mean,
                           agent1.distribution(obs2).mean)


def test_gae_hand_computed():
    buf = RolloutBuffer(3, {"x": 1}, 1, gamma=0.5, gae_lambda=0.5)
    for r, v in zip([1.0, 0.0, 2.0], [0.5, 0.4, 0.3]):
        buf.add({"x": np.zeros(1)}, np.zeros(1), 0.0, r, v, 0.0)
    buf.compute_gae(last_value=0.1, last_done=0.0)
    # delta2 = 2 + .5*.1 - .3 = 1.75             ; gae2 = 1.75
    # delta1 = 0 + .5*.3 - .4 = -0.25            ; gae1 = -.25 + .25*1.75 = .1875
    # delta0 = 1 + .5*.4 - .5 = 0.7              ; gae0 = .7 + .25*.1875 = .746875
    assert buf.advantages == pytest.approx([0.746875, 0.1875, 1.75])
    assert buf.returns == pytest.approx([1.246875, 0.5875, 2.05])


def test_gae_resets_at_done():
    buf = RolloutBuffer(2, {"x": 1}, 1, gamma=0.9, gae_lambda=0.9)
    buf.add({"x": np.zeros(1)}, np.zeros(1), 0.0, 1.0, 0.5, 0.0)
    buf.add({"x": np.zeros(1)}, np.zeros(1), 0.0, 1.0, 0.5, 1.0)  # new episode
    buf.compute_gae(last_value=9.9, last_done=0.0)
    # step0's future is cut by the done flag before step1:
    # delta0 = 1 + 0 - 0.5 = 0.5 ; gae0 = 0.5 (no bootstrap across the reset)
    assert buf.advantages[0] == pytest.approx(0.5)


def _det_eval(agent, fs, cfg, start=0, end=60):
    """Mean rolling-Sharpe reward of the DETERMINISTIC policy on the window."""
    env = PortfolioEnv(fs, cfg, start=start, end=end)
    obs, _ = env.reset(seed=0)
    rewards, done = [], False
    while not done:
        with torch.no_grad():
            a, _, _ = agent.act(_to_tensors(obs, "cpu"), deterministic=True)
        obs, r, done, _, _ = env.step(a.squeeze(0).numpy())
        rewards.append(r)
    return float(np.mean(rewards))


def test_overfit_tiny_window(setup):
    """M5 acceptance gate (Manual G.3): on a single fixed ~60-day window the
    agent's in-sample rolling Sharpe must increase — proves the plumbing
    learns before any generalisation question.

    The window has real structure (one steadily trending asset), so the
    optimal in-sample policy is unambiguous and the gradient signal strong —
    this tests the learning loop, not luck on a random walk.
    """
    cfg, fs, env, obs_dims = setup
    cfg = Config()
    cfg.data.price_window = WINDOW
    cfg.env.reward_window_K = 5
    cfg.model.d_endo, cfg.model.d_exo = 16, 32
    cfg.train.lr = 1e-3

    bundle = make_bundle(seed=8, start="2014-01-01", end="2015-12-31",
                         n_assets=N_ASSETS)
    rng = np.random.default_rng(0)
    T = len(bundle["prices"])
    t = np.arange(T)
    bundle["prices"].iloc[:, 0] = 100 * np.exp(0.003 * t + rng.normal(0, 5e-4, T))
    bundle["prices"].iloc[:, 1] = 100 * np.exp(rng.normal(0, 3e-3, T).cumsum())
    bundle["prices"].iloc[:, 2] = 100 * np.exp(-0.001 * t + rng.normal(0, 3e-3, T).cumsum())
    fs2 = build_features(**bundle, cfg=cfg)

    env60 = PortfolioEnv(fs2, cfg, start=0, end=60)
    agent = _agent("cell4", cfg, obs_dims)
    before = _det_eval(agent, fs2, cfg)
    ppo = PPO(env60, agent, cfg, n_steps=240, batch_size=120, n_epochs=8)
    rows = ppo.learn(total_steps=240 * 16)
    after = _det_eval(agent, fs2, cfg)

    assert after > before, f"no learning: before={before:.4f} after={after:.4f}"
    assert after > 0, f"in-sample Sharpe should turn positive, got {after:.4f}"
    assert np.isfinite(rows[-1]["explained_variance"])
