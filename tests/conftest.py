"""Shared synthetic fixtures — tests never hit the network (Manual D)."""

import numpy as np
import pandas as pd
import pytest

from exo_portfolio.config import Config


def make_bundle(seed: int = 0, start: str = "2010-01-01", end: str = "2024-12-31",
                n_assets: int = 4):
    """Synthetic raw-data bundle shaped exactly like loaders.load_raw_bundle."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, end)                      # proxy trading calendar
    T, N = len(cal), n_assets

    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(2e-4, 0.01, (T, N)), axis=0)),
        index=cal, columns=[f"AST{i}" for i in range(N)])
    index = pd.Series(
        1000 * np.exp(np.cumsum(rng.normal(2e-4, 0.008, T))), index=cal, name="gspc")
    vix = pd.Series(np.clip(rng.normal(18, 6, T), 9, None), index=cal, name="vix")

    months = pd.date_range(start, end, freq="MS")         # monthly reference dates
    macro = pd.DataFrame({
        "FEDFUNDS": np.clip(np.cumsum(rng.normal(0, 0.1, len(months))) + 2, 0, None),
        "CPIAUCSL": 200 * np.exp(np.cumsum(rng.normal(0.002, 0.001, len(months)))),
    }, index=months)

    return {"prices": prices, "index": index, "vix": vix, "macro": macro}


@pytest.fixture
def bundle():
    return make_bundle()


@pytest.fixture
def cfg():
    c = Config()
    c.data.price_window = 10        # small windows keep tests fast
    c.env.reward_window_K = 5
    return c
