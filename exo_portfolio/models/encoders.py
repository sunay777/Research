"""EndoEncoder / ExoEncoder (Manual G.1).

EndoEncoder: small MLP — the endogenous state (weights, cash, K-return
window) is low-dimensional and needs little capacity.

ExoEncoder: per-asset Conv1d over the price window (shared across assets),
producing one embedding per asset; embeddings are concatenated with the
market-level features (index return, VIX, lagged macro) and passed through
an MLP to R^{d_exo}.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes: list[int], act=nn.Tanh, out_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(act())
    return nn.Sequential(*layers)


class EndoEncoder(nn.Module):
    """endo (B, in_dim) -> z_endo (B, d_endo)."""

    def __init__(self, in_dim: int, d_endo: int, hidden: int = 64):
        super().__init__()
        self.net = mlp([in_dim, hidden, d_endo])

    def forward(self, endo: torch.Tensor) -> torch.Tensor:
        return self.net(endo)


class ExoEncoder(nn.Module):
    """exo_actor (B, n_assets*price_window + n_market) -> z_exo (B, d_exo).

    Layout contract (matches data/align.build_features): the first
    n_assets*price_window entries are the flattened per-asset return windows
    (asset-major), the remaining n_market entries are market-level features.
    """

    def __init__(self, n_assets: int, price_window: int, n_market: int,
                 d_exo: int, d_asset: int = 8, conv_channels: int = 8):
        super().__init__()
        self.n_assets = n_assets
        self.price_window = price_window
        self.n_market = n_market

        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.asset_proj = nn.Linear(conv_channels, d_asset)
        self.head = mlp([n_assets * d_asset + n_market, 2 * d_exo, d_exo])

    def forward(self, exo: torch.Tensor) -> torch.Tensor:
        B = exo.shape[0]
        n_aw = self.n_assets * self.price_window
        windows = exo[:, :n_aw].reshape(B * self.n_assets, 1, self.price_window)
        market = exo[:, n_aw:]

        emb = self.conv(windows).squeeze(-1)              # (B*N, C)
        emb = torch.tanh(self.asset_proj(emb))            # (B*N, d_asset)
        emb = emb.reshape(B, -1)                          # (B, N*d_asset)
        return self.head(torch.cat([emb, market], dim=1))
