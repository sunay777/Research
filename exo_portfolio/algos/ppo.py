"""Custom clipped PPO with GAE (Manual G.3).

Two non-standard requirements SB3 cannot cleanly provide:
1. Dict observations (endo / exo_actor / exo_critic_extra).
2. An asymmetric critic: the value net receives exo_critic_extra; the policy
   never does (enforced structurally in ExoActorCritic).

Logged per update (Manual G.3): policy loss, value loss, entropy, approx KL,
clip fraction, explained variance, mean reward, mean turnover — to
{run_dir}/train_log.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from exo_portfolio.algos.rollout import RolloutBuffer
from exo_portfolio.config import Config, seed_everything
from exo_portfolio.models.agent import ExoActorCritic


def _to_tensors(obs: dict[str, np.ndarray], device: str):
    return {k: torch.as_tensor(v, dtype=torch.float32,
                               device=device).unsqueeze(0)
            for k, v in obs.items()}


class PPO:
    def __init__(self, env, agent: ExoActorCritic, cfg: Config,
                 run_dir: str | Path | None = None, n_steps: int = 2048,
                 batch_size: int = 256, n_epochs: int = 10,
                 ent_coef: float = 0.0, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, device: str = "cpu"):
        self.env = env
        self.agent = agent.to(device)
        self.cfg = cfg
        self.device = device
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.train.lr)
        obs_dims = {k: int(np.prod(s.shape))
                    for k, s in env.observation_space.spaces.items()}
        n_actions = int(np.prod(env.action_space.shape))
        self.buffer = RolloutBuffer(n_steps, obs_dims, n_actions,
                                    cfg.train.gamma, cfg.train.gae_lambda,
                                    device)
        self.rng = np.random.default_rng(cfg.seed)
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_rows: list[dict] = []

        seed_everything(cfg.seed)
        self._obs, _ = env.reset(seed=cfg.seed)
        self._done = 0.0

    # ------------------------------------------------------------- rollouts
    def collect(self) -> dict:
        self.buffer.reset()
        ep_rewards, ep_turnovers = [], []
        for _ in range(self.n_steps):
            with torch.no_grad():
                a, logp, v = self.agent.act(_to_tensors(self._obs, self.device))
            action = a.squeeze(0).cpu().numpy()
            next_obs, reward, term, trunc, info = self.env.step(action)
            self.buffer.add(self._obs, action, float(logp), float(reward),
                            float(v), self._done)
            ep_rewards.append(float(reward))
            ep_turnovers.append(float(info.get("turnover", np.nan)))
            self._done = float(term or trunc)
            if term or trunc:
                next_obs, _ = self.env.reset()
            self._obs = next_obs

        with torch.no_grad():
            last_v = float(self.agent.forward_critic(
                _to_tensors(self._obs, self.device)))
        self.buffer.compute_gae(last_v, self._done)
        return {"mean_reward": float(np.mean(ep_rewards)),
                "mean_turnover": float(np.nanmean(ep_turnovers))}

    # -------------------------------------------------------------- updates
    def update(self) -> dict:
        pol_losses, val_losses, entropies, kls, clipfracs = [], [], [], [], []
        for _ in range(self.n_epochs):
            for batch in self.buffer.batches(self.batch_size, self.rng):
                logp, entropy, values = self.agent.evaluate_actions(
                    batch["obs"], batch["actions"])
                adv = batch["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = (logp - batch["logprobs"]).exp()
                clip = self.cfg.train.clip
                pg1 = -adv * ratio
                pg2 = -adv * ratio.clamp(1 - clip, 1 + clip)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = 0.5 * (values - batch["returns"]).pow(2).mean()
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(),
                                               self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    pol_losses.append(float(policy_loss))
                    val_losses.append(float(value_loss))
                    entropies.append(float(entropy.mean()))
                    kls.append(float((batch["logprobs"] - logp).mean()))
                    clipfracs.append(float(((ratio - 1).abs() > clip)
                                           .float().mean()))

        y, yhat = self.buffer.returns, self.buffer.values
        var_y = np.var(y)
        explained = float(1 - np.var(y - yhat) / var_y) if var_y > 0 else 0.0
        return {"policy_loss": float(np.mean(pol_losses)),
                "value_loss": float(np.mean(val_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(kls)),
                "clip_frac": float(np.mean(clipfracs)),
                "explained_variance": explained}

    # ---------------------------------------------------------------- learn
    def learn(self, total_steps: int, log_every: int = 1) -> list[dict]:
        n_updates = max(1, total_steps // self.n_steps)
        for u in range(n_updates):
            roll = self.collect()
            stats = self.update()
            row = {"update": u, "env_steps": (u + 1) * self.n_steps,
                   **roll, **stats}
            self._log_rows.append(row)
            if self.run_dir and (u % log_every == 0 or u == n_updates - 1):
                self._flush_log()
        return self._log_rows

    def _flush_log(self):
        path = self.run_dir / "train_log.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self._log_rows[0]))
            writer.writeheader()
            writer.writerows(self._log_rows)

    # ------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, obs: dict[str, np.ndarray],
                deterministic: bool = True) -> np.ndarray:
        a, _, _ = self.agent.act(_to_tensors(obs, self.device),
                                 deterministic=deterministic)
        return a.squeeze(0).cpu().numpy()


def evaluate_agent_window(agent: ExoActorCritic, features, cfg: Config,
                          start: int, end: int,
                          deterministic: bool = True,
                          device: str = "cpu") -> dict:
    """Deterministic rollout of the custom agent over [start, end] — mirrors
    baselines.sb3_baselines.evaluate_policy_window (same return shape)."""
    from exo_portfolio.envs.portfolio_env import PortfolioEnv
    from exo_portfolio.eval.metrics import summarize

    env = PortfolioEnv(features, cfg, start, end)
    obs, _ = env.reset(seed=cfg.seed)
    rhos, taus, dates = [], [], []
    done = False
    while not done:
        with torch.no_grad():
            a, _, _ = agent.act(_to_tensors(obs, device),
                                deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(a.squeeze(0).cpu().numpy())
        rhos.append(info["step_log_return"])
        taus.append(info["turnover"])
        dates.append(info["date"])
        done = terminated or truncated
    row = summarize(np.array(rhos), np.array(taus))
    row["final_value"] = float(np.exp(np.sum(rhos)))
    return {"metrics": row, "log_returns": np.array(rhos),
            "turnover": np.array(taus), "dates": dates}
