"""Controlled synthetic Exo-MDP with an injected P_exo shift (Manual Part I).

A world where the factorisation is ground truth:

- ENDOGENOUS: the same closed-form weight/wealth update as the real env
  (E.3 steps 1-7) over `n_assets` synthetic assets + cash. Deterministic
  given (state, action, returns).
- EXOGENOUS: a 2-state Markov regime e_t (0 = calm, 1 = stressed). The
  regime during (t, t+1] drives that step's asset returns:
      r ~ N(mu[e_t], sigma[e_t])   per asset
  Calm has positive drift / low vol; stressed negative drift / high vol —
  so the optimal policy is "invest when calm, de-risk when stressed",
  exactly the mechanism the thesis claims on real markets.
- OBSERVATION (same Dict contract as PortfolioEnv, so every grid cell runs
  unchanged):
    endo             : weights + cash fraction + last K portfolio log-returns
    exo_actor        : per-asset return windows (asset block) ⊕ a NOISY
                       high-dim linear embedding of the current regime plus
                       pure-noise clutter dims (the "irrelevant exogenous
                       clutter" of Efroni/Wan)
    exo_critic_extra : the TRUE regime one-hot (privileged, exact)
- INJECTED SHIFT (test time): `shift` ∈ [0, 1] interpolates the transition
  probabilities AND the regime return-means toward their swapped
  configuration; shift=0 is the training world, shift=1 swaps calm and
  stressed persistence/means entirely — a P_exo unseen in training.

Measure (Part I): train factored vs monolithic cells on shift=0, then plot
test performance vs shift magnitude. Hypothesis (Claim B): factored degrades
more gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from exo_portfolio.config import Config
from exo_portfolio.eval.metrics import rolling_sharpe


@dataclass
class SyntheticCfg:
    n_assets: int = 2
    price_window: int = 10          # per-asset return window in exo_actor
    episode_len: int = 200
    reward_window_K: int = 5
    transaction_cost: float = 0.001
    reward_eps: float = 1e-8
    # regime process (training configuration)
    p_calm_to_stress: float = 0.05
    p_stress_to_calm: float = 0.10
    mu: tuple = ((0.0012, 0.0008), (-0.0015, -0.0010))   # [regime][asset]
    sigma: tuple = ((0.006, 0.005), (0.018, 0.015))
    # observation noise / clutter
    embed_dim: int = 8              # noisy linear embedding of the regime
    clutter_dim: int = 8            # pure-noise exogenous dims
    embed_noise: float = 0.5
    # injected shift (0 = training world)
    shift: float = 0.0
    embed_seed: int = 1234          # fixes the embedding matrix A across envs


def _shifted_params(cfg: SyntheticCfg):
    """Interpolate transitions and regime means toward the swapped world."""
    s = float(cfg.shift)
    p01 = (1 - s) * cfg.p_calm_to_stress + s * cfg.p_stress_to_calm
    p10 = (1 - s) * cfg.p_stress_to_calm + s * cfg.p_calm_to_stress
    mu = np.asarray(cfg.mu, dtype=np.float64)
    mu_shift = (1 - s) * mu + s * mu[::-1]               # swap regime means
    sigma = np.asarray(cfg.sigma, dtype=np.float64)
    sigma_shift = (1 - s) * sigma + s * sigma[::-1]
    return p01, p10, mu_shift, sigma_shift


class SyntheticExoEnv(gym.Env):
    """Gymnasium env with the PortfolioEnv observation contract."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: SyntheticCfg | None = None):
        super().__init__()
        self.cfg = cfg or SyntheticCfg()
        c = self.cfg
        self.N = c.n_assets
        self.K = c.reward_window_K
        self.W = c.price_window
        self.p01, self.p10, self.mu, self.sigma = _shifted_params(c)

        # fixed regime-embedding matrix (identical across shift levels so the
        # OBSERVATION function is unchanged — only P_exo shifts)
        emb_rng = np.random.default_rng(c.embed_seed)
        self.A = emb_rng.normal(0, 1, (c.embed_dim, 2))

        n_w = self.N + 1
        d_exo = self.N * self.W + c.embed_dim + c.clutter_dim
        self.observation_space = spaces.Dict({
            "endo": spaces.Box(-np.inf, np.inf, (n_w + 1 + self.K,), np.float32),
            "exo_actor": spaces.Box(-np.inf, np.inf, (d_exo,), np.float32),
            "exo_critic_extra": spaces.Box(0.0, 1.0, (2,), np.float32),
        })
        self.action_space = spaces.Box(-10.0, 10.0, (n_w,), np.float32)

    # ------------------------------------------------------------------ obs
    def _obs(self) -> dict:
        window = np.zeros(self.K, dtype=np.float32)
        hist = self._ret_history[-self.K:]
        if hist:
            window[-len(hist):] = hist
        endo = np.concatenate([self.w.astype(np.float32),
                               np.float32([self.w[0]]), window])

        asset_block = self._asset_windows.reshape(-1).astype(np.float32)
        onehot = np.zeros(2)
        onehot[self.e] = 1.0
        embed = self.A @ onehot + self.np_random.normal(
            0, self.cfg.embed_noise, self.cfg.embed_dim)
        clutter = self.np_random.normal(0, 1, self.cfg.clutter_dim)
        exo = np.concatenate([asset_block, embed, clutter]).astype(np.float32)

        return {"endo": endo, "exo_actor": exo,
                "exo_critic_extra": onehot.astype(np.float32)}

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.v = 1.0
        self.w = np.zeros(self.N + 1, dtype=np.float64)
        self.w[0] = 1.0
        self._ret_history: list[float] = []
        self._asset_windows = np.zeros((self.N, self.W), dtype=np.float64)
        # start in the stationary distribution of the (shifted) chain
        p_stress = self.p01 / (self.p01 + self.p10)
        self.e = int(self.np_random.random() < p_stress)
        return self._obs(), {"v": self.v, "regime": self.e}

    def step(self, action):
        assert self.t < self.cfg.episode_len, "episode already terminated"
        a = np.asarray(action, dtype=np.float64).reshape(self.N + 1)

        # exogenous: regime e_t drives this step's returns (P_exo)
        r = self.np_random.normal(self.mu[self.e], self.sigma[self.e])
        y = np.concatenate([[1.0], np.exp(r)])

        # endogenous: E.3 steps 1-7, identical arithmetic to PortfolioEnv
        g = float(self.w @ y)
        w_drift = (y * self.w) / g
        z = a - np.max(a)
        w_new = np.exp(z) / np.exp(z).sum()
        tau = float(np.abs(w_new - w_drift).sum())
        mu_cost = self.cfg.transaction_cost * tau
        v_new = self.v * g * (1.0 - mu_cost)
        rho = float(np.log(v_new / self.v))

        self._ret_history.append(rho)
        reward = rolling_sharpe(self._ret_history[-self.K:], self.cfg.reward_eps)

        # roll the per-asset return windows (observable history)
        self._asset_windows = np.roll(self._asset_windows, -1, axis=1)
        self._asset_windows[:, -1] = r

        # exogenous transition — INDEPENDENT of the action (the factorisation)
        flip_p = self.p01 if self.e == 0 else self.p10
        prev_regime = self.e
        if self.np_random.random() < flip_p:
            self.e = 1 - self.e

        self.t += 1
        self.w = w_new
        self.v = v_new
        terminated = self.t >= self.cfg.episode_len
        info = {"v": self.v, "g": g, "turnover": tau, "step_log_return": rho,
                "weights": self.w.copy(), "regime": prev_regime}
        return self._obs(), reward, terminated, False, info


# ---------------------------------------------------------------------------
# The Part-I experiment: degradation vs shift magnitude
# ---------------------------------------------------------------------------

def make_config(seed: int = 0, cell: str = "cell4",
                total_steps: int = 100_000) -> Config:
    """A Config wired for the synthetic experiment (small dims, shared PPO)."""
    cfg = Config()
    cfg.seed = seed
    cfg.cell = cell
    cfg.train.total_steps = total_steps
    cfg.model.d_endo, cfg.model.d_exo = 16, 32
    return cfg


def run_shift_experiment(cells=("cell1", "cell2", "cell4"),
                         seeds=range(10),
                         shifts=(0.0, 0.25, 0.5, 0.75, 1.0),
                         train_steps: int = 100_000,
                         n_eval_episodes: int = 20,
                         syn_cfg: SyntheticCfg | None = None,
                         out_csv: str | None = "results/synthetic_shift.csv"):
    """Train each (cell, seed) on shift=0, evaluate deterministically on every
    shift level -> long DataFrame (cell, seed, shift, mean_reward, cum_log_return).

    Each row also carries `oracle_mean_reward`: the true-regime oracle
    (all-in when calm, all-cash when stressed) evaluated on the SAME shifted
    worlds. Report degradation both raw and as regret (oracle - agent) — at
    intermediate shifts the interpolated regime means move toward zero, so
    the world is intrinsically less profitable and raw curves confound
    brittleness with achievable reward.

    NOTE (interpretation): the oracle applies the TRAINING-world mapping
    (invest-when-calm), so under large shifts it becomes anti-optimal and
    regret can go negative — it is a frozen-expert reference, not an upper
    bound. State this in the write-up when presenting regret curves.

    Full acceptance run (>=10 seeds) belongs on the cluster; smaller calls of
    this same function are used for smoke tests.
    """
    import pandas as pd
    import torch

    from exo_portfolio.algos.ppo import PPO, _to_tensors
    from exo_portfolio.models.agent import ExoActorCritic
    from exo_portfolio.models.presets import model_cfg_for_cell

    base = syn_cfg or SyntheticCfg()

    def _oracle_reward(scfg: SyntheticCfg) -> float:
        env = SyntheticExoEnv(scfg)
        rews = []
        for ep in range(n_eval_episodes):
            obs, _ = env.reset(seed=10_000 + ep)
            done, ep_r = False, []
            while not done:
                a = np.zeros(env.N + 1)
                a[0] = 10.0 if obs["exo_critic_extra"][1] == 1.0 else -10.0
                obs, r, done, _, _ = env.step(a)
                ep_r.append(r)
            rews.append(np.mean(ep_r))
        return float(np.mean(rews))

    oracle = {}
    for shift in shifts:
        oracle[shift] = _oracle_reward(
            SyntheticCfg(**{**base.__dict__, "shift": shift}))

    rows = []
    for cell in cells:
        for seed in seeds:
            cfg = make_config(seed=seed, cell=cell, total_steps=train_steps)
            train_env = SyntheticExoEnv(base)
            obs_dims = {k: int(np.prod(s.shape))
                        for k, s in train_env.observation_space.spaces.items()}
            torch.manual_seed(seed)
            agent = ExoActorCritic(obs_dims, base.n_assets + 1,
                                   model_cfg_for_cell(cell, cfg.model),
                                   n_assets=base.n_assets,
                                   price_window=base.price_window)
            ppo = PPO(train_env, agent, cfg, n_steps=1000, batch_size=250,
                      n_epochs=6)
            ppo.learn(train_steps)

            for shift in shifts:
                scfg = SyntheticCfg(**{**base.__dict__, "shift": shift})
                env = SyntheticExoEnv(scfg)
                rews, cums = [], []
                for ep in range(n_eval_episodes):
                    obs, _ = env.reset(seed=10_000 + ep)   # same eval worlds
                    done, ep_r, ep_c = False, [], 0.0
                    while not done:
                        with torch.no_grad():
                            a, _, _ = agent.act(_to_tensors(obs, "cpu"),
                                                deterministic=True)
                        obs, r, done, _, info = env.step(a.squeeze(0).numpy())
                        ep_r.append(r)
                        ep_c += info["step_log_return"]
                    rews.append(np.mean(ep_r))
                    cums.append(ep_c)
                rows.append({"cell": cell, "seed": seed, "shift": shift,
                             "mean_reward": float(np.mean(rews)),
                             "cum_log_return": float(np.mean(cums)),
                             "oracle_mean_reward": oracle[shift],
                             "regret": oracle[shift] - float(np.mean(rews))})
                print(f"[{cell} seed{seed}] shift={shift:.2f} "
                      f"reward={rows[-1]['mean_reward']:+.4f}")

    df = pd.DataFrame(rows)
    if out_csv:
        from pathlib import Path

        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["cell1", "cell2", "cell4"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--train-steps", type=int, default=100_000)
    ap.add_argument("--out", default="results/synthetic_shift.csv")
    args = ap.parse_args()
    run_shift_experiment(cells=args.cells, seeds=range(args.seeds),
                         train_steps=args.train_steps, out_csv=args.out)
