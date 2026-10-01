"""Typed configuration schema (Development Manual, Part C.3).

Every experiment is driven by one `Config` object. No magic numbers in code
bodies — anything tunable lives here and is overridable from YAML.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataCfg:
    universe: str = "djia30"
    start: str = "2010-01-01"
    end: str = "2026-06-30"
    tickers: list[str] = field(default_factory=list)  # filled from universe yaml
    index_ticker: str = "^GSPC"
    vix_ticker: str = "^VIX"
    macro_series: list[str] = field(default_factory=lambda: ["FEDFUNDS", "CPIAUCSL"])
    # Publication lag in *calendar days* per FRED series: the value for a
    # reference period only becomes visible this many days after its date.
    macro_publication_lag_days: dict[str, int] = field(
        default_factory=lambda: {"FEDFUNDS": 1, "CPIAUCSL": 14}
    )
    price_window: int = 30          # lookback window of per-asset log-returns
    cache_dir: str = "data_cache"   # raw downloads cached here as CSV


@dataclass
class EnvCfg:
    transaction_cost: float = 0.001  # proportional, per unit turnover
    reward_window_K: int = 20        # rolling Sharpe window
    reward_eps: float = 1e-8
    allow_short: bool = False        # v1: long-only + cash


@dataclass
class ModelCfg:
    encoder: str = "dual"            # {"single", "dual"}
    critic: str = "asymmetric"       # {"symmetric", "asymmetric"}
    exo_mode: str = "split"          # {"none", "concat", "split"}
    d_endo: int = 32
    d_exo: int = 128
    # hidden width of the single-encoder MLP stream (single+none / +concat);
    # widened only by capacity-control presets (cell0cw)
    stream_hidden: int = 128


@dataclass
class ExoAblationCfg:
    """Exo-feature ablation axis (M10). A SEPARATE experimental axis from the
    architecture grid — like the J.5 cost sweep — that masks a named group of
    the actor's exogenous features. Default (keep/none) is a no-op.

    group ∈ {none, asset_returns, index, vix, macro} — the NAMED exo_actor
           groups (data/align.build_features layout).
    mode  ∈ {keep, zero, permute, noise} — how the group is degraded. All modes
           are dimension-preserving (in-place), so the custom dual-encoder cells
           (cell2/cell4) keep their fixed ExoEncoder layout (Manual G.1).
    """
    group: str = "none"
    mode: str = "keep"
    seed: int = 0                    # deterministic permutation / noise draw


@dataclass
class BaselinesCfg:
    """Hyper-parameters for the non-RL baselines (Manual K), so nothing is a
    magic number in a function body (Manual A.1.2). The M3 baselines keep their
    own defaults; these drive the M10 traditional weight-rules."""
    trad_window: int = 60           # trailing window for inverse-vol/min-var/max-Sharpe
    trad_ridge: float = 1e-4        # covariance ridge (shared with mean_variance style)
    momentum_lookback: int = 60     # trailing window for cross-sectional momentum
    momentum_top_k: int = 10        # hold the top-k trailing performers, equal-weighted


@dataclass
class TrainCfg:
    algo: str = "ppo"
    total_steps: int = 2_000_000
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    baselines: BaselinesCfg = field(default_factory=BaselinesCfg)
    exo_ablation: ExoAblationCfg = field(default_factory=ExoAblationCfg)
    seed: int = 0
    fold: int = 0
    cell: str = "cell4"

    # -- identity ----------------------------------------------------------
    @property
    def run_id(self) -> str:
        return f"{self.cell}_seed{self.seed}_fold{self.fold}"

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return cls(
            data=DataCfg(**d.get("data", {})),
            env=EnvCfg(**d.get("env", {})),
            model=ModelCfg(**d.get("model", {})),
            train=TrainCfg(**d.get("train", {})),
            baselines=BaselinesCfg(**d.get("baselines", {})),
            exo_ablation=ExoAblationCfg(**d.get("exo_ablation", {})),
            **{k: v for k, v in d.items()
               if k in ("seed", "fold", "cell")},
        )

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> "Config":
        """Load one or more YAML files; later files override earlier ones."""
        merged: dict[str, Any] = {}
        for p in paths:
            with open(p) as fh:
                _deep_update(merged, yaml.safe_load(fh) or {})
        return cls.from_dict(merged)


def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# Seeding (Manual A.1.3): seed everything, log the seed.
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed `random`, `numpy` and, when available, `torch` (+ CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is only required from M4/M5 onward
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
