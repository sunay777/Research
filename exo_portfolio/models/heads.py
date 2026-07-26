"""ActorHead / CriticHead (Manual G).

Actor: diagonal Gaussian over pre-softmax weight logits (the env applies the
softmax; Manual G.3). The log-std is a state-independent learned parameter.

Critic: small MLP to a scalar value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from exo_portfolio.models.encoders import mlp


class ActorHead(nn.Module):
    def __init__(self, d_latent: int, n_actions: int,
                 init_log_std: float = -0.5):
        super().__init__()
        self.mean = nn.Linear(d_latent, n_actions)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)   # near-uniform start
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((n_actions,), init_log_std))

    def forward(self, z: torch.Tensor) -> Independent:
        return Independent(Normal(self.mean(z), self.log_std.exp()), 1)


class CriticHead(nn.Module):
    def __init__(self, d_latent: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(*mlp([d_latent, hidden], out_act=True),
                                 nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
