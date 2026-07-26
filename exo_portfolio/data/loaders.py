"""Raw data fetch + CSV cache (Manual D.1).

Sources:
- Prices (daily, auto-adjusted close) per constituent: yfinance
- Broad index ^GSPC level, ^VIX close: yfinance
- Macro: FRED FEDFUNDS (policy rate), CPIAUCSL (CPI level -> YoY inflation
  computed downstream in align.py)

Everything downloaded is cached to `cfg.data.cache_dir` as CSV so experiments
are reproducible offline. All tests run on synthetic fixtures and never hit
the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from exo_portfolio.config import Config


# ---------------------------------------------------------------------------
# Cached fetchers
# ---------------------------------------------------------------------------

def fetch_prices(tickers: list[str], start: str, end: str,
                 cache_dir: str | Path) -> pd.DataFrame:
    """Adjusted daily close for `tickers`, columns = tickers, index = dates."""
    cache = Path(cache_dir) / "prices.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df) and set(tickers).issubset(df.columns):
            return df[tickers].loc[start:end]

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    if isinstance(raw, pd.Series):           # single ticker edge case
        raw = raw.to_frame(tickers[0])
    raw = raw[tickers]
    if raw.dropna(how="all").empty:
        raise RuntimeError("Empty price download — check network access; not caching.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cache)
    return raw


def fetch_single(ticker: str, start: str, end: str,
                 cache_dir: str | Path, name: str) -> pd.Series:
    """One yfinance close series (used for ^GSPC and ^VIX)."""
    cache = Path(cache_dir) / f"{name}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df):
            return df.iloc[:, 0].loc[start:end]

    import yfinance as yf

    s = yf.download(ticker, start=start, end=end, auto_adjust=True,
                    progress=False)["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = name
    if s.dropna().empty:
        raise RuntimeError(f"Empty download for {ticker} — not caching.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(cache)
    return s


def _fred_api_key() -> str | None:
    """FRED API key from the FRED_API_KEY env var or an untracked .env file.

    The key is a personal credential: it lives in .env (gitignored) or the
    environment — NEVER in config YAML or anything committed.
    """
    import os

    if os.environ.get("FRED_API_KEY"):
        return os.environ["FRED_API_KEY"]
    for candidate in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                if line.strip().startswith("FRED_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


def _fetch_fred_api(series: list[str], start: str, end: str,
                    api_key: str) -> pd.DataFrame:
    """Official FRED API (api.stlouisfed.org). Values are indexed by their
    *reference-period* date, matching pandas-datareader's fredgraph output."""
    import json
    import urllib.request

    cols = {}
    for sid in series:
        url = ("https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={sid}&api_key={api_key}&file_type=json"
               f"&observation_start={start}&observation_end={end}")
        with urllib.request.urlopen(url, timeout=30) as r:
            obs = json.load(r)["observations"]
        s = pd.Series({pd.Timestamp(o["date"]): float(o["value"])
                       for o in obs if o["value"] != "."}, name=sid)
        cols[sid] = s
    return pd.DataFrame(cols).sort_index()


def fetch_macro(series: list[str], start: str, end: str,
                cache_dir: str | Path) -> pd.DataFrame:
    """FRED series, columns = series codes, index = reference-period dates.

    Uses the official FRED API when a key is available (env FRED_API_KEY or
    .env), else falls back to pandas-datareader's fredgraph CSV endpoint.

    NOTE: the index carries the *reference period* date (e.g. 2020-03-01 for
    March CPI). Publication lag is applied later, in align.apply_publication_lag
    — never here, so the cache stays raw.
    """
    cache = Path(cache_dir) / "macro.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df) and set(series).issubset(df.columns):
            return df[series].loc[start:end]

    key = _fred_api_key()
    if key:
        df = _fetch_fred_api(series, start, end, key)
    else:
        import pandas_datareader.data as web

        df = web.DataReader(series, "fred", start, end)
    if df.dropna(how="all").empty:
        raise RuntimeError("Empty FRED download — not caching.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


# ---------------------------------------------------------------------------
# One-shot bundle
# ---------------------------------------------------------------------------

def load_raw_bundle(cfg: Config) -> dict:
    """Fetch (or read from cache) everything the feature builder needs."""
    d = cfg.data
    return {
        "prices": fetch_prices(d.tickers, d.start, d.end, d.cache_dir),
        "index": fetch_single(d.index_ticker, d.start, d.end, d.cache_dir, "gspc"),
        "vix": fetch_single(d.vix_ticker, d.start, d.end, d.cache_dir, "vix"),
        "macro": fetch_macro(d.macro_series, d.start, d.end, d.cache_dir),
    }


if __name__ == "__main__":
    # python -m exo_portfolio.data.loaders --config configs/base.yaml configs/universe_djia30.yaml
    args = [a for a in sys.argv[1:] if a != "--config"]
    cfg = Config.from_yaml(*args) if args else Config()
    bundle = load_raw_bundle(cfg)
    for k, v in bundle.items():
        print(f"{k}: shape={getattr(v, 'shape', None)}, "
              f"range=[{v.index.min().date()} .. {v.index.max().date()}]")
