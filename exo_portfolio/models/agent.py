"""ExoActorCritic — the single agent class with the three ablation switches
(Manual G.2). The five grid cells are presets of these switches (Part H).

    model.encoder  ∈ {single, dual}
    model.critic   ∈ {symmetric, asymmetric}
    model.exo_mode ∈ {none, concat, split}

Invariants enforced here:
- forward_actor / the action distribution use ONLY obs["endo"] and
  obs["exo_actor"] (deployment-realistic) — never obs["exo_critic_extra"].
- A symmetric critic sees exactly the actor's inputs; an asymmetric critic
  additionally consumes obs["exo_critic_extra"] (privileged, training-only).
  This reduces value-estimate variance without biasing the policy gradient,
  valid because P_exo is conditionally independent of the action.
- Actor and critic have separate parameters (no shared trunk), so privileged
  gradients cannot leak into the actor through a shared encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from exo_portfolio.config import ModelCfg
from exo_portfolio.models.encoders import EndoEncoder, ExoEncoder, mlp
from exo_portfolio.models.heads import ActorHead, CriticHead

# Part H — the ablation grid as presets of the three switches.
CELL_PRESETS: dict[str, dict] = {
    "cell0": dict(encoder="single", critic="symmetric", exo_mode="none"),
    "cell1": dict(encoder="single", critic="symmetric", exo_mode="concat"),
    "cell2": dict(encoder="dual", critic="symmetric", exo_mode="split"),
    "cell3": dict(encoder="single", critic="asymmetric", exo_mode="concat"),
    "cell4": dict(encoder="dual", critic="asymmetric", exo_mode="split"),
}

VALID_COMBOS = {("single", "none"), ("single", "concat"), ("dual", "split")}


def model_cfg_for_cell(cell: str, base: ModelCfg | None = None) -> ModelCfg:
    base = base or ModelCfg()
    preset = CELL_PRESETS[cell]
    return ModelCfg(encoder=preset["encoder"], critic=preset["critic"],
                    exo_mode=preset["exo_mode"],
                    d_endo=base.d_endo, d_exo=base.d_exo)


class _FeatureStream(nn.Module):
    """One feature extractor at the actor's information level.

    single+none   : MLP(endo) -> d_lat
    single+concat : MLP(endo ⊕ exo_actor) -> d_lat            (monolithic)
    dual+split    : EndoEncoder(endo) ⊕ ExoEncoder(exo_actor) (factored)

    d_lat = d_endo + d_exo in every mode, so downstream capacity is
    identical across cells (Manual G.2: comparable parameter counts —
    residual differences live in the extractor and are logged).
    """

    def __init__(self, cfg: ModelCfg, d_endo_in: int, d_exo_in: int,
                 n_assets: int, price_window: int):
        super().__init__()
        assert (cfg.encoder, cfg.exo_mode) in VALID_COMBOS, \
            f"invalid switch combo: {cfg.encoder}/{cfg.exo_mode}"
        self.mode = (cfg.encoder, cfg.exo_mode)
        self.d_lat = cfg.d_endo + cfg.d_exo

        if self.mode == ("single", "none"):
            self.net = mlp([d_endo_in, 128, self.d_lat])
        elif self.mode == ("single", "concat"):
            self.net = mlp([d_endo_in + d_exo_in, 128, self.d_lat])
        else:                                   # ("dual", "split")
            n_market = d_exo_in - n_assets * price_window
            assert n_market > 0, "exo_actor smaller than the asset block"
            self.endo_enc = EndoEncoder(d_endo_in, cfg.d_endo)
            self.exo_enc = ExoEncoder(n_assets, price_window, n_market,
                                      cfg.d_exo)

    def forward(self, endo: torch.Tensor, exo: torch.Tensor) -> torch.Tensor:
        if self.mode == ("single", "none"):
            return self.net(endo)
        if self.mode == ("single", "concat"):
            return self.net(torch.cat([endo, exo], dim=1))
        return torch.cat([self.endo_enc(endo), self.exo_enc(exo)], dim=1)


class ExoActorCritic(nn.Module):
    def __init__(self, obs_dims: dict[str, int], n_actions: int,
                 cfg: ModelCfg, n_assets: int, price_window: int):
        """obs_dims: {"endo": d, "exo_actor": d, "exo_critic_extra": d}."""
        super().__init__()
        self.cfg = cfg
        self.asymmetric = cfg.critic == "asymmetric"
        d_priv = obs_dims["exo_critic_extra"]

        self.actor_stream = _FeatureStream(cfg, obs_dims["endo"],
                                           obs_dims["exo_actor"],
                                           n_assets, price_window)
        self.critic_stream = _FeatureStream(cfg, obs_dims["endo"],
                                            obs_dims["exo_actor"],
                                            n_assets, price_window)
        self.actor_head = ActorHead(self.actor_stream.d_lat, n_actions)
        d_critic_in = self.critic_stream.d_lat + (d_priv if self.asymmetric else 0)
        self.critic_head = CriticHead(d_critic_in)

    # ---------------------------------------------------------------- actor
    def distribution(self, obs: dict[str, torch.Tensor]):
        """ALWAYS uses only endo + exo_actor (deployment-realistic)."""
        z = self.actor_stream(obs["endo"], obs["exo_actor"])
        return self.actor_head(z)

    def forward_actor(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.distribution(obs).mean

    # --------------------------------------------------------------- critic
    def forward_critic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.critic_stream(obs["endo"], obs["exo_actor"])
        if self.asymmetric:
            z = torch.cat([z, obs["exo_critic_extra"]], dim=1)
        return self.critic_head(z)

    # ------------------------------------------------------------- PPO API
    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action), self.forward_critic(obs)

    def evaluate_actions(self, obs: dict[str, torch.Tensor],
                         actions: torch.Tensor):
        dist = self.distribution(obs)
        return (dist.log_prob(actions), dist.entropy(),
                self.forward_critic(obs))

    # ---------------------------------------------------------- diagnostics
    def param_counts(self) -> dict[str, int]:
        """Log these per cell (Manual L: capacity control)."""
        count = lambda m: sum(p.numel() for p in m.parameters())
        return {
            "actor": count(self.actor_stream) + count(self.actor_head),
            "critic": count(self.critic_stream) + count(self.critic_head),
            "total": count(self),
        }
