"""SB3 PPO/SAC/A2C baselines at cell0/cell1 information levels (Manual K, M4).

These provide the *algorithm-family* baselines. They never receive privileged
information: `exo_critic_extra` is stripped by the wrapper, so an SB3 critic
sees exactly what its actor sees (symmetric by construction).

- cell0: endo only — no exogenous info at all.
- cell1: endo ⊕ exo_actor flat-concatenated — the monolithic MDP.

The custom Exo-agent (cells 2-4, dual encoder / asymmetric critic) is NOT
built on SB3 — see algos/ppo.py from M5 (SB3 cannot cleanly support a
privileged critic; Manual A.2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from exo_portfolio.config import Config, seed_everything
from exo_portfolio.data.align import (FeatureSet, exo_group_slices,
                                      mask_exo_array)
from exo_portfolio.envs.portfolio_env import PortfolioEnv
from exo_portfolio.eval.metrics import summarize


class InfoLevelWrapper(gym.ObservationWrapper):
    """Flatten the Dict observation to the cell's information level.

    cell0 -> endo only; cell1 -> endo ⊕ exo_actor.
    exo_critic_extra is always dropped (no privilege for SB3 cells).
    """

    def __init__(self, env: gym.Env, level: str):
        super().__init__(env)
        assert level in ("cell0", "cell1"), level
        self.level = level
        d_endo = env.observation_space["endo"].shape[0]
        d_exo = env.observation_space["exo_actor"].shape[0]
        dim = d_endo if level == "cell0" else d_endo + d_exo
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)

    def observation(self, obs: dict) -> np.ndarray:
        if self.level == "cell0":
            return obs["endo"]
        return np.concatenate([obs["endo"], obs["exo_actor"]])


class ExoMaskWrapper(gym.ObservationWrapper):
    """Mask a NAMED group of the actor's exogenous features (M10 exo-ablation
    axis) — analogous to InfoLevelWrapper, but it degrades a feature group
    in-place (dimension-preserving) instead of choosing an information level.

    Modes {keep, zero, permute, noise} match data.align.mask_exo_array. The
    whole-dataset masked array is precomputed once (so permute-in-time is a
    single deterministic reordering), then the row for the env's current step
    is served. Observation dimensionality is never changed, so the custom
    dual-encoder cells keep their fixed ExoEncoder layout (Manual G.1).
    """

    def __init__(self, env: gym.Env, group: str, mode: str, seed: int = 0):
        super().__init__(env)
        self.group = group
        self.mode = mode
        base = env.unwrapped
        if group == "none" or mode == "keep":
            self._masked = base.exo_actor
        else:
            sl = exo_group_slices(base.N, base.cfg.data.price_window,
                                  base.exo_actor.shape[1])[group]
            self._masked = mask_exo_array(base.exo_actor, sl, mode,
                                          np.random.default_rng(seed))

    def observation(self, obs: dict) -> dict:
        obs = dict(obs)
        obs["exo_actor"] = self._masked[self.env.unwrapped.t]
        return obs


def make_env(features: FeatureSet, cfg: Config, start: int, end: int,
             level: str) -> gym.Env:
    from stable_baselines3.common.monitor import Monitor

    return Monitor(InfoLevelWrapper(PortfolioEnv(features, cfg, start, end), level))


# The FinRL algorithm family, apples-to-apples in THIS env (Manual K / A.2):
# on-policy PPO/A2C, off-policy SAC/DDPG/TD3. (FinRL-Meta's DataOps pipeline is
# a separate concern — we only borrow the algorithm set.)
ON_POLICY = ("ppo", "a2c")
OFF_POLICY = ("sac", "ddpg", "td3")


def _algo_class(name: str):
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    return {"ppo": PPO, "sac": SAC, "a2c": A2C,
            "ddpg": DDPG, "td3": TD3}[name.lower()]


def train_sb3(cfg: Config, features: FeatureSet, train_start: int,
              train_end: int, level: str, run_dir: str | Path,
              total_steps: int | None = None):
    """Train one SB3 baseline on the given train window; returns the model.

    Optimiser hyper-parameters come from cfg.train so SB3 cells and the
    custom-agent cells share them (Manual A.1.6). The seed is cfg.seed.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    algo_cls = _algo_class(cfg.train.algo)
    env = make_env(features, cfg, train_start, train_end, level)

    # Shared optimiser hyper-parameters (Manual A.1.6), guarded by algorithm
    # family: gae_lambda is on-policy-only; clip is PPO-only; the off-policy
    # replay learners take only lr/gamma from cfg.
    algo = cfg.train.algo.lower()
    kwargs = dict(seed=cfg.seed, verbose=0,
                  tensorboard_log=str(run_dir / "tb"),
                  learning_rate=cfg.train.lr, gamma=cfg.train.gamma)
    if algo in ON_POLICY:
        kwargs.update(gae_lambda=cfg.train.gae_lambda)
    if algo == "ppo":
        kwargs.update(clip_range=cfg.train.clip)
    assert algo in ON_POLICY or algo in OFF_POLICY, f"unknown algo {algo}"

    model = algo_cls("MlpPolicy", env, **kwargs)
    model.learn(total_timesteps=int(total_steps or cfg.train.total_steps),
                progress_bar=False)
    model.save(run_dir / "model")
    return model


def evaluate_policy_window(model, features: FeatureSet, cfg: Config,
                           start: int, end: int, level: str,
                           deterministic: bool = True) -> dict:
    """Deterministic rollout over [start, end]; returns the standard metrics
    row plus the step-level series (for regime slicing at M7)."""
    env = make_env(features, cfg, start, end, level)
    obs, _ = env.reset(seed=cfg.seed)
    rhos, taus, dates, exposures = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        rhos.append(info["step_log_return"])
        taus.append(info["turnover"])
        dates.append(info["date"])
        exposures.append(1.0 - float(info["weights"][0]))   # 1 - cash (J.4)
        done = terminated or truncated
    row = summarize(np.array(rhos), np.array(taus))
    row["final_value"] = float(np.exp(np.sum(rhos)))
    return {"metrics": row, "log_returns": np.array(rhos),
            "turnover": np.array(taus), "dates": dates,
            "exposure": np.array(exposures)}
