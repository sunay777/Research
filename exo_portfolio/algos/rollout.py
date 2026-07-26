"""Rollout buffer with dict observations and GAE (Manual G.3)."""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dims: dict[str, int], n_actions: int,
                 gamma: float, gae_lambda: float, device: str = "cpu"):
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs = {k: np.zeros((n_steps, d), dtype=np.float32)
                    for k, d in obs_dims.items()}
        self.actions = np.zeros((n_steps, n_actions), dtype=np.float32)
        self.logprobs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)   # done BEFORE step t
        self.ptr = 0

    def add(self, obs: dict[str, np.ndarray], action: np.ndarray,
            logprob: float, reward: float, value: float, done: float):
        t = self.ptr
        for k in self.obs:
            self.obs[k][t] = obs[k]
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.rewards[t] = reward
        self.values[t] = value
        self.dones[t] = done
        self.ptr += 1

    def compute_gae(self, last_value: float, last_done: float):
        assert self.ptr == self.n_steps, "buffer not full"
        adv = np.zeros(self.n_steps, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_nonterminal = 1.0 - last_done
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = (self.rewards[t]
                     + self.gamma * next_value * next_nonterminal
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values

    def batches(self, batch_size: int, rng: np.random.Generator):
        idx = rng.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            b = idx[start:start + batch_size]
            yield {
                "obs": {k: torch.as_tensor(v[b], device=self.device)
                        for k, v in self.obs.items()},
                "actions": torch.as_tensor(self.actions[b], device=self.device),
                "logprobs": torch.as_tensor(self.logprobs[b], device=self.device),
                "advantages": torch.as_tensor(self.advantages[b], device=self.device),
                "returns": torch.as_tensor(self.returns[b], device=self.device),
                "values": torch.as_tensor(self.values[b], device=self.device),
            }

    def reset(self):
        self.ptr = 0
