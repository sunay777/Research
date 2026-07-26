"""Regression test for the cache-coverage bug: a cache built for an earlier
end date must NOT silently satisfy a request for a later end date."""

import numpy as np
import pandas as pd
import pytest

from exo_portfolio.data import loaders


def _write_price_cache(tmp_path, end):
    dates = pd.bdate_range("2010-01-04", end)
    df = pd.DataFrame({"AAA": np.linspace(10, 20, len(dates)),
                       "BBB": np.linspace(30, 40, len(dates))}, index=dates)
    (tmp_path / "prices.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path / "prices.csv")
    return df


def test_stale_price_cache_triggers_refetch(tmp_path, monkeypatch):
    _write_price_cache(tmp_path, "2024-12-30")          # stale: ends 2024

    called = {}
    fresh_dates = pd.bdate_range("2010-01-04", "2026-06-29")
    fresh = pd.DataFrame({"AAA": 1.0, "BBB": 2.0}, index=fresh_dates)

    class FakeYF:
        @staticmethod
        def download(tickers, **kw):
            called["yes"] = True
            return {"Close": fresh}

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = loaders.fetch_prices(["AAA", "BBB"], "2010-01-01", "2026-06-30", tmp_path)
    assert called.get("yes"), "stale cache should force a re-download"
    assert out.index.max() >= pd.Timestamp("2026-06-01")
    # and the refreshed cache was persisted
    again = pd.read_csv(tmp_path / "prices.csv", index_col=0, parse_dates=True)
    assert again.index.max() >= pd.Timestamp("2026-06-01")


def test_fresh_price_cache_is_used_without_network(tmp_path, monkeypatch):
    _write_price_cache(tmp_path, "2026-06-29")          # covers request

    class ExplodingYF:
        @staticmethod
        def download(*a, **kw):
            raise AssertionError("network must not be touched on a fresh cache")

    monkeypatch.setitem(__import__("sys").modules, "yfinance", ExplodingYF)
    out = loaders.fetch_prices(["AAA", "BBB"], "2010-01-01", "2026-06-30", tmp_path)
    assert out.index.max() == pd.Timestamp("2026-06-29")


def test_macro_cache_slack_allows_monthly_lag(tmp_path):
    """Monthly macro ending ~2 months before the requested end is still fresh."""
    months = pd.date_range("2010-01-01", "2026-05-01", freq="MS")
    pd.DataFrame({"FEDFUNDS": 2.0, "CPIAUCSL": 300.0}, index=months) \
        .to_csv(tmp_path / "macro.csv")
    out = loaders.fetch_macro(["FEDFUNDS", "CPIAUCSL"],
                              "2010-01-01", "2026-06-30", tmp_path)
    assert out.index.max() == pd.Timestamp("2026-05-01")


def test_cache_covers_helper():
    idx = pd.bdate_range("2010-01-04", "2024-12-30")
    assert loaders._cache_covers(idx, "2010-01-01", "2024-12-31")
    assert not loaders._cache_covers(idx, "2010-01-01", "2026-06-30")
    assert not loaders._cache_covers(idx, "2008-01-01", "2024-12-31")
    assert not loaders._cache_covers(pd.DatetimeIndex([]), "2010-01-01", "2024-12-31")
