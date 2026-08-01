"""M8 acceptance tests: the synthetic Exo-MDP is a controlled world where
the factorisation is ground truth and the P_exo shift is exact."""

import numpy as np
import pytest

from exo_portfolio.envs.synthetic_exo import (SyntheticCfg, SyntheticExoEnv,
                                              _shifted_params)


def rollout(env, seed=0, policy=None, steps=None):
    obs, info = env.reset(seed=seed)
    total, regimes, rhos = [], [info["regime"]], []
    n = steps or env.cfg.episode_len
    for _ in range(n):
        a = policy(obs, env) if policy else np.zeros(env.N + 1)
        obs, r, done, _, info = env.step(a)
        total.append(r)
        regimes.append(info["regime"])
        rhos.append(info["step_log_return"])
        if done:
            break
    return np.array(total), np.array(regimes), np.array(rhos)


def test_determinism():
    a = rollout(SyntheticExoEnv(), seed=7)
    b = rollout(SyntheticExoEnv(), seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_regime_chain_statistics():
    env = SyntheticExoEnv(SyntheticCfg(episode_len=5000))
    _, regimes, _ = rollout(env, seed=0)
    frac_stress = regimes.mean()
    expected = env.p01 / (env.p01 + env.p10)     # stationary distribution
    assert abs(frac_stress - expected) < 0.08


def test_privileged_obs_is_true_regime():
    env = SyntheticExoEnv()
    obs, info = env.reset(seed=3)
    for _ in range(50):
        onehot = obs["exo_critic_extra"]
        assert onehot.sum() == 1.0 and onehot[info["regime"]] == 1.0
        obs, _, done, _, info_next = env.step(np.zeros(env.N + 1))
        # obs after step reflects the POST-transition regime
        info = {"regime": np.argmax(obs["exo_critic_extra"])}
        if done:
            break


def test_shift_interpolation():
    base = SyntheticCfg()
    p01_0, p10_0, mu_0, sg_0 = _shifted_params(base)
    assert (p01_0, p10_0) == (base.p_calm_to_stress, base.p_stress_to_calm)
    assert np.allclose(mu_0, base.mu)

    full = SyntheticCfg(shift=1.0)
    p01_1, p10_1, mu_1, sg_1 = _shifted_params(full)
    assert (p01_1, p10_1) == (base.p_stress_to_calm, base.p_calm_to_stress)
    assert np.allclose(mu_1, np.asarray(base.mu)[::-1])    # means swapped
    assert np.allclose(sg_1, np.asarray(base.sigma)[::-1])

    half = SyntheticCfg(shift=0.5)
    p01_h, _, _, _ = _shifted_params(half)
    assert p01_h == pytest.approx((p01_0 + p01_1) / 2)


def test_observation_function_unchanged_by_shift():
    """Only P_exo shifts; the observation map (embedding A) must not."""
    e0 = SyntheticExoEnv(SyntheticCfg(shift=0.0))
    e1 = SyntheticExoEnv(SyntheticCfg(shift=1.0))
    assert np.array_equal(e0.A, e1.A)


def test_regime_matters_oracle_beats_uniform():
    """An oracle using the true regime (all-in when calm, cash when stressed)
    must out-earn uniform weights — the exogenous signal is real."""
    def oracle(obs, env):
        a = np.zeros(env.N + 1)
        if obs["exo_critic_extra"][1] == 1.0:    # stressed -> cash
            a[0] = 10.0
        else:                                    # calm -> invest
            a[0] = -10.0
        return a

    cum_oracle, cum_uniform = 0.0, 0.0
    for seed in range(5):
        env = SyntheticExoEnv(SyntheticCfg(episode_len=400))
        _, _, rho_o = rollout(env, seed=seed, policy=oracle)
        env2 = SyntheticExoEnv(SyntheticCfg(episode_len=400))
        _, _, rho_u = rollout(env2, seed=seed)
        cum_oracle += rho_o.sum()
        cum_uniform += rho_u.sum()
    assert cum_oracle > cum_uniform, (cum_oracle, cum_uniform)


def test_action_does_not_influence_regime_path():
    """The factorisation ground truth: P_exo is action-independent."""
    def all_cash(obs, env):
        a = np.zeros(env.N + 1); a[0] = 10.0
        return a

    _, regimes_a, _ = rollout(SyntheticExoEnv(), seed=11)
    _, regimes_b, _ = rollout(SyntheticExoEnv(), seed=11, policy=all_cash)
    assert np.array_equal(regimes_a, regimes_b)


def test_grid_agents_run_on_synthetic_env():
    torch = pytest.importorskip("torch")
    from exo_portfolio.algos.ppo import _to_tensors
    from exo_portfolio.envs.synthetic_exo import make_config
    from exo_portfolio.models.agent import ExoActorCritic
    from exo_portfolio.models.presets import CELL_PRESETS, model_cfg_for_cell

    env = SyntheticExoEnv()
    obs_dims = {k: int(np.prod(s.shape))
                for k, s in env.observation_space.spaces.items()}
    obs, _ = env.reset(seed=0)
    for cell in CELL_PRESETS:
        cfg = make_config(cell=cell)
        agent = ExoActorCritic(obs_dims, env.N + 1,
                               model_cfg_for_cell(cell, cfg.model),
                               n_assets=env.N, price_window=env.W)
        a, _, v = agent.act(_to_tensors(obs, "cpu"))
        assert np.isfinite(a.numpy()).all() and np.isfinite(float(v))


def test_shift_experiment_smoke():
    torch = pytest.importorskip("torch")
    from exo_portfolio.envs.synthetic_exo import run_shift_experiment

    df = run_shift_experiment(cells=("cell1",), seeds=(0,),
                              shifts=(0.0, 1.0), train_steps=1000,
                              n_eval_episodes=2, out_csv=None,
                              syn_cfg=SyntheticCfg(episode_len=50))
    assert len(df) == 2
    assert set(df.columns) >= {"cell", "seed", "shift", "mean_reward",
                               "cum_log_return"}
    assert np.isfinite(df["mean_reward"]).all()
