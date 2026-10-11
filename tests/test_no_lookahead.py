"""M1 acceptance gate (Manual D.5 / L): every feature at day t is a pure
function of data dated <= t, and macro publication lags are respected."""

import numpy as np
import pandas as pd

from exo_portfolio.data.align import build_features

from tests.conftest import make_bundle


def _perturb_after(bundle, t: pd.Timestamp, rng):
    """Return a copy of the bundle where everything strictly after t is changed."""
    b = {k: v.copy() for k, v in bundle.items()}
    for key in ("prices", "index", "vix", "macro"):
        obj = b[key]
        mask = obj.index > t
        noise = rng.uniform(1.5, 2.5, size=(int(mask.sum()),) + obj.values.shape[1:] if obj.ndim > 1 else int(mask.sum()))
        obj.loc[mask] = obj.loc[mask].values * noise
    return b


def test_features_pure_function_of_past(bundle, cfg):
    """Perturbing all raw data strictly after t must leave rows <= t untouched."""
    fs = build_features(**bundle, cfg=cfg)
    rng = np.random.default_rng(1)
    ts = rng.choice(len(fs.dates) - 40, size=5, replace=False) + 20  # random t's

    for pos in ts:
        t = fs.dates[pos]
        fs2 = build_features(**_perturb_after(bundle, t, rng), cfg=cfg)
        upto = fs.dates[fs.dates <= t]
        for name in ("exo_actor", "exo_critic_extra", "prices"):
            a = getattr(fs, name).loc[upto].values
            b = getattr(fs2, name).loc[upto].values
            assert np.array_equal(a, b), f"lookahead leak in {name} at t={t}"


def test_macro_lag_applied(bundle, cfg):
    """A macro jump in reference month m must reach exo_actor only at
    end-of-month(m) + lag, and exo_critic_extra (privileged) at end-of-month(m)
    -- never earlier, because before then month m has not finished."""
    lag = cfg.data.macro_publication_lag_days["FEDFUNDS"] = 30
    b = {k: v.copy() for k, v in bundle.items()}
    d = b["macro"].index[100]                         # FRED stamp: 1st of month m
    b["macro"].loc[d:, "FEDFUNDS"] += 100.0           # big jump from month m on

    fs = build_features(**b, cfg=cfg)
    actor = fs.exo_actor["FEDFUNDS_lagged"]
    critic = fs.exo_critic_extra["FEDFUNDS_unlagged"]
    month_end = d + pd.offsets.MonthEnd(0)
    pub = month_end + pd.Timedelta(days=lag)

    assert (actor[fs.dates < pub] < 50).all(), "jump visible to actor before publication"
    assert (actor[fs.dates >= pub] > 50).all(), "jump missing after publication"
    assert (critic[fs.dates < month_end] < 50).all(), \
        "privileged critic saw month m before month m was over"
    assert (critic[(fs.dates >= month_end) & (fs.dates < pub)] > 50).all(), \
        "privileged critic should see month m from its end, before publication"


def test_cpi_yoy_inherits_cpi_lag(bundle, cfg):
    """Regression: CPI_YOY is derived from CPIAUCSL and must get CPIAUCSL's lag
    (it used to fall through to a 0-day lag because the lag lookup used the
    derived column name)."""
    lag = cfg.data.macro_publication_lag_days["CPIAUCSL"] = 31
    b = {k: v.copy() for k, v in bundle.items()}
    d = b["macro"].index[100]
    b["macro"].loc[d:, "CPIAUCSL"] *= 3.0             # YoY jumps in month m

    fs = build_features(**b, cfg=cfg)
    actor = fs.exo_actor["CPI_YOY_lagged"]
    pub = d + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)
    assert (actor[fs.dates < pub] < 1.0).all(), "CPI_YOY visible before publication"
    assert (actor[(fs.dates >= pub) & (fs.dates < pub + pd.Timedelta(days=20))] > 1.0).all()


def test_unconfigured_macro_lag_is_an_error(bundle, cfg):
    """A macro column with no configured lag must fail loudly, not default to 0."""
    import pytest
    del cfg.data.macro_publication_lag_days["CPIAUCSL"]
    with pytest.raises(KeyError):
        build_features(**bundle, cfg=cfg)


def test_missing_cpi_month_not_padded(bundle, cfg):
    """A missing CPI month (e.g. Oct 2025) must not be filled from the previous
    month's level: no new YoY value is published for it."""
    from exo_portfolio.data.align import transform_macro
    m = bundle["macro"].copy()
    gap = m.index[60]
    m.loc[gap, "CPIAUCSL"] = np.nan
    yoy = transform_macro(m)["CPI_YOY"]
    assert np.isnan(yoy.loc[gap])


def test_final_matrix_nan_free(bundle, cfg):
    fs = build_features(**bundle, cfg=cfg)
    assert not fs.exo_actor.isna().any().any()
    assert not fs.exo_critic_extra.isna().any().any()
    assert not fs.prices.isna().any().any()
    assert len(fs.dates) > 3000        # ~15y of business days survive the trim


def test_asset_window_ends_at_t(bundle, cfg):
    """Most recent entry of each asset window must equal ln(p_t / p_{t-1})."""
    fs = build_features(**bundle, cfg=cfg)
    t = fs.dates[50]
    prices = bundle["prices"]
    pos = prices.index.get_loc(t)
    for tkr in prices.columns:
        expected = np.log(prices[tkr].iloc[pos] / prices[tkr].iloc[pos - 1])
        got = fs.exo_actor.loc[t, f"{tkr}_lr_t-0"]
        assert np.isclose(got, expected)
