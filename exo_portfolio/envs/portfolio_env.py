"""The Exo-MDP portfolio environment (Manual Part E). The heart of the project.

Factorisation this env makes explicit:
- ``P_endo`` — the wealth/weight transition — is DETERMINISTIC in closed form
  given (state, action, prices): implemented literally as E.3 steps 1-7.
- ``P_exo``  — the price/feature path — is exogenous: the agent's actions never
  influence it (we are a small investor; no market impact).

Observation is a Gymnasium ``Dict`` (E.2):
- ``endo``             : current weights (N+1, cash first) + cash fraction +
                         last K portfolio step log-returns (zero-padded before
                         K steps have elapsed). The K-window makes the
                         path-dependent rolling-Sharpe reward Markov.
- ``exo_actor``        : lagged, deployment-realistic exogenous features.
- ``exo_critic_extra`` : privileged (un-lagged) features. Present in the space
                         always; ONLY the asymmetric critic may consume it —
                         the actor never touches it (Manual L).

Timing convention (E.3, portfolio-vector-memory style): at time t the agent
holds w_t (set by a_{t-1}). Over (t, t+1] prices move by y_{t+1}; the holdings
drift to w_t'; the action a_t is then implemented as new target weights
w_{t+1} = softmax(a_t), with proportional cost on |w_{t+1} - w_t'|.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from exo_portfolio.config import Config
from exo_portfolio.data.align import FeatureSet
from exo_portfolio.eval.metrics import rolling_sharpe


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


class PortfolioEnv(gym.Env):
    """Long-only portfolio over N assets + cash (index 0 = cash)."""

    metadata = {"render_modes": []}

    def __init__(self, features: FeatureSet, cfg: Config,
                 start: int = 0, end: int | None = None):
        """`start`/`end` are positional bounds into `features.dates`
        (half-open on steps: the final observation sits at `end`), so a fold's
        train/val/test indices map directly onto env windows (Manual E.5).
        """
        super().__init__()
        self.cfg = cfg
        self.features = features
        self.prices = features.prices.values.astype(np.float64)     # (T, N)
        self.exo_actor = features.exo_actor.values.astype(np.float32)
        self.exo_critic_extra = features.exo_critic_extra.values.astype(np.float32)

        self.start = start
        self.end = len(features.dates) - 1 if end is None else end
        assert 0 <= self.start < self.end < len(features.dates)

        self.N = self.prices.shape[1]
        self.K = cfg.env.reward_window_K
        self.cost = cfg.env.transaction_cost
        self.eps = cfg.env.reward_eps

        n_w = self.N + 1
        self.observation_space = spaces.Dict({
            "endo": spaces.Box(-np.inf, np.inf, (n_w + 1 + self.K,), np.float32),
            "exo_actor": spaces.Box(-np.inf, np.inf,
                                    (self.exo_actor.shape[1],), np.float32),
            "exo_critic_extra": spaces.Box(-np.inf, np.inf,
                                           (self.exo_critic_extra.shape[1],),
                                           np.float32),
        })
        self.action_space = spaces.Box(-10.0, 10.0, (n_w,), np.float32)

    # ------------------------------------------------------------------ obs
    def _obs(self) -> dict:
        window = np.zeros(self.K, dtype=np.float32)
        hist = self._ret_history[-self.K:]
        if hist:
            window[-len(hist):] = hist        # most recent return is last
        endo = np.concatenate([self.w.astype(np.float32),
                               np.float32([self.w[0]]),      # cash fraction
                               window])
        return {
            "endo": endo,
            "exo_actor": self.exo_actor[self.t],
            "exo_critic_extra": self.exo_critic_extra[self.t],
        }

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = self.start
        self.v = 1.0
        self.w = np.zeros(self.N + 1, dtype=np.float64)
        self.w[0] = 1.0                        # all cash (E.5)
        self._ret_history: list[float] = []
        return self._obs(), {"v": self.v, "date": self.features.dates[self.t]}

    def step(self, action: np.ndarray):
        assert self.t < self.end, "episode already terminated"
        a = np.asarray(action, dtype=np.float64).reshape(self.N + 1)

        # price relatives over (t, t+1]; cash entry = 1 (E.3)
        y = np.empty(self.N + 1, dtype=np.float64)
        y[0] = 1.0
        y[1:] = self.prices[self.t + 1] / self.prices[self.t]

        # E.3 steps 1-7, implemented literally
        g = float(self.w @ y)                          # 1. gross growth
        w_drift = (y * self.w) / g                     # 2. price-drifted weights
        w_new = softmax(a)                             # 3. new target weights
        tau = float(np.abs(w_new - w_drift).sum())     # 4. turnover
        mu = self.cost * tau                           # 5. cost multiplier
        v_new = self.v * g * (1.0 - mu)                # 6. value update
        rho = float(np.log(v_new / self.v))            # 7. step log-return

        self._ret_history.append(rho)
        reward = rolling_sharpe(self._ret_history[-self.K:], self.eps)  # E.4

        self.t += 1
        self.w = w_new
        self.v = v_new

        terminated = self.t >= self.end
        info = {"v": self.v, "g": g, "turnover": tau, "step_log_return": rho,
                "weights": self.w.copy(), "w_drift": w_drift,
                "date": self.features.dates[self.t]}
        return self._obs(), reward, terminated, False, info
