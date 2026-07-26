"""M0 acceptance: config loads, merges YAMLs, seeds reproducibly."""

from pathlib import Path

import numpy as np

from exo_portfolio.config import Config, seed_everything

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_defaults():
    cfg = Config()
    assert cfg.env.transaction_cost == 0.001
    assert cfg.env.reward_window_K == 20
    assert cfg.model.encoder == "dual"
    assert cfg.run_id == "cell4_seed0_fold0"


def test_yaml_merge():
    cfg = Config.from_yaml(CONFIGS / "base.yaml", CONFIGS / "universe_djia30.yaml")
    assert cfg.data.universe == "djia30"
    assert len(cfg.data.tickers) == 29          # DOW excluded (no pre-2019 data)
    assert cfg.data.macro_publication_lag_days["CPIAUCSL"] == 14
    # base.yaml values survive the merge
    assert cfg.train.gamma == 0.99


def test_seed_everything_reproducible():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
