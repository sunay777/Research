"""Test-time exo ablation: masking is window-local, permutation preserves
the window's values, draws are common across cells, and the summary pairs
each masked rollout with the same model intact."""

import numpy as np
import pandas as pd
import pytest

from exo_portfolio.config import Config
from exo_portfolio.data.align import build_features, exo_group_slices
from exo_portfolio.eval.test_time_ablation import (mask_window, perm_rng,
                                                   summarize_ablation)

from tests.conftest import make_bundle


@pytest.fixture(scope="module")
def features():
    cfg = Config()
    cfg.data.price_window = 10
    fs = build_features(**make_bundle(seed=3, start="2014-01-01",
                                      end="2016-12-31", n_assets=4), cfg=cfg)
    return cfg, fs


@pytest.mark.parametrize("group", ["asset_returns", "vix", "macro"])
def test_permute_is_window_local_and_value_preserving(features, group):
    cfg, fs = features
    s, e = 100, 220
    out = mask_window(fs, group, "permute", s, e, cfg.data.price_window,
                      perm_rng(0, group, 0))
    a = fs.exo_actor.values.astype(np.float32)        # env reads float32
    b = out.exo_actor.values
    sl = exo_group_slices(fs.prices.shape[1], cfg.data.price_window, a.shape[1])[group]
    other = np.ones(a.shape[1], bool)
    other[sl] = False
    np.testing.assert_array_equal(a[:s], b[:s])            # outside window
    np.testing.assert_array_equal(a[e + 1:], b[e + 1:])
    np.testing.assert_array_equal(a[:, other], b[:, other])  # other groups
    win_a = np.sort(a[s:e + 1, sl].round(6), axis=0)
    win_b = np.sort(b[s:e + 1, sl].round(6), axis=0)
    np.testing.assert_array_equal(win_a, win_b)            # same values
    assert not np.array_equal(a[s:e + 1, sl], b[s:e + 1, sl])
    pd.testing.assert_frame_equal(out.prices, fs.prices)   # P_exo untouched


def test_zero_mask_and_common_draws(features):
    cfg, fs = features
    out = mask_window(fs, "vix", "zero", 50, 60, cfg.data.price_window)
    sl = exo_group_slices(fs.prices.shape[1], cfg.data.price_window,
                          fs.exo_actor.shape[1])["vix"]
    assert (out.exo_actor.values[50:61, sl] == 0).all()
    np.testing.assert_allclose(out.exo_actor.values[61:, sl],
                               fs.exo_actor.values[61:, sl], rtol=1e-6)
    p1 = perm_rng(2, "vix", 1).permutation(100)
    p2 = perm_rng(2, "vix", 1).permutation(100)
    np.testing.assert_array_equal(p1, p2)


def test_summary_pairs_with_intact():
    rows = []
    for i in range(6):
        base = {"run_id": f"r{i}", "cell": "c", "seed": i, "fold": 0}
        m = dict(sharpe=1.0 + i, cum_log_return=0.1, max_drawdown=0.2,
                 mean_exposure=0.9, policy_shift=0.0)
        rows.append({**base, "group": "none", "mode": "intact", "draw": 0, **m})
        for d in range(2):                      # two draws averaged first
            rows.append({**base, "group": "vix", "mode": "permute", "draw": d,
                         **{**m, "sharpe": m["sharpe"] - 0.5 - 0.2 * d,
                            "policy_shift": 0.1}})
    s = summarize_ablation(pd.DataFrame(rows)).set_index("metric")
    assert np.isclose(s.loc["sharpe", "mean_diff"], -0.6)
    assert np.isclose(s.loc["policy_shift", "mean_diff"], 0.1)
    assert s.loc["sharpe", "n_models"] == 6
