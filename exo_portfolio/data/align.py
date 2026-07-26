"""Time alignment, publication lags, and feature building (Manual D.3-D.4).

Cardinal rule (Manual A.1.4): any value visible at observation time `t` must
have been knowable at the market close of day `t`.

Two aligned macro versions are produced:
- ``exo_actor``        — publication-lagged (deployment-realistic)
- ``exo_critic_extra`` — un-lagged reference-period values (privileged,
  consumed ONLY by the asymmetric critic). Still causal: nothing dated > t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from exo_portfolio.config import Config


# ---------------------------------------------------------------------------
# Macro helpers
# ---------------------------------------------------------------------------

def transform_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """Turn raw FRED levels into model features.

    CPIAUCSL (CPI level) -> CPI_YOY, year-on-year inflation. The YoY value for
    reference month m is computable the moment month m's CPI is published, so
    it inherits CPIAUCSL's publication lag unchanged.
    """
    out = macro.copy()
    if "CPIAUCSL" in out.columns:
        out["CPI_YOY"] = out["CPIAUCSL"].pct_change(12)
        out = out.drop(columns=["CPIAUCSL"])
    return out


def apply_publication_lag(macro: pd.DataFrame, lags: dict[str, int],
                          calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Shift each series to its publication date, then forward-fill onto the
    trading calendar so day t carries the most recent value *published on or
    before* t (Manual D.4). Lag is in calendar days.
    """
    cols = {}
    for col in macro.columns:
        lag = lags.get(col, 0)
        s = macro[col].dropna()
        s = pd.Series(s.values, index=s.index + pd.Timedelta(days=lag), name=col)
        cols[col] = _ffill_onto(s, calendar)
    return pd.DataFrame(cols, index=calendar)


def align_unlagged(macro: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Un-lagged (privileged) version: value visible from its reference date.

    Still causal — the value carried at day t is dated <= t.
    """
    return pd.DataFrame(
        {c: _ffill_onto(macro[c].dropna(), calendar) for c in macro.columns},
        index=calendar,
    )


def _ffill_onto(s: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a sparsely-dated series onto `calendar` using only past
    values (an as-of join; never interpolates from the future)."""
    union = s.index.union(calendar)
    return s.reindex(union).ffill().reindex(calendar)


# ---------------------------------------------------------------------------
# Feature bundle
# ---------------------------------------------------------------------------

@dataclass
class FeatureSet:
    """Aligned, NaN-free matrices, all indexed by `dates` (trading days)."""
    dates: pd.DatetimeIndex
    prices: pd.DataFrame            # (T, N) adjusted close, env consumes this
    exo_actor: pd.DataFrame         # (T, N*W + 2 + n_macro) lagged, actor-visible
    exo_critic_extra: pd.DataFrame  # (T, n_macro) un-lagged, critic-only

    def __post_init__(self):
        for name in ("prices", "exo_actor", "exo_critic_extra"):
            df = getattr(self, name)
            assert not df.isna().any().any(), f"NaNs in {name} (Manual D.3)"
            assert df.index.equals(self.dates)


def build_features(prices: pd.DataFrame, index: pd.Series, vix: pd.Series,
                   macro: pd.DataFrame, cfg: Config) -> FeatureSet:
    """Assemble the aligned feature matrices.

    - Trading calendar = the price index (NYSE trading days as observed in the
      price data itself).
    - Per-asset block: the last `price_window` daily log-returns, flattened
      (N * W columns). The window ends at day t: ln(p_t / p_{t-1}) is the most
      recent entry — knowable at the close of t.
    - Market block: index 1-day log-return + VIX close (level).
    - Macro block: publication-lagged (actor) / un-lagged (critic extra).
    - NaN policy: drop leading rows until every column is valid, then assert
      zero NaNs (Manual D.3).
    """
    W = cfg.data.price_window
    calendar = prices.index

    # --- per-asset log-return windows -------------------------------------
    lr = np.log(prices.values[1:] / prices.values[:-1])      # rows -> calendar[1:]
    win = sliding_window_view(lr, W, axis=0)                 # (T-W, N, W)
    win_dates = calendar[W:]                                 # window ends at t
    asset_cols = [f"{tkr}_lr_t-{W - 1 - k}" for tkr in prices.columns
                  for k in range(W)]
    asset_block = pd.DataFrame(win.reshape(len(win_dates), -1),
                               index=win_dates, columns=asset_cols)
    asset_block = asset_block.reindex(calendar)   # leading NaNs trimmed below

    # --- market block (sampled at the NYSE close, ffill past-only) --------
    idx_aligned = _ffill_onto(index.dropna(), calendar)
    idx_lr = np.log(idx_aligned / idx_aligned.shift(1)).rename("gspc_lr")
    vix_aligned = _ffill_onto(vix.dropna(), calendar).rename("vix")

    # --- macro blocks ------------------------------------------------------
    macro_t = transform_macro(macro)
    macro_lagged = apply_publication_lag(
        macro_t, cfg.data.macro_publication_lag_days, calendar)
    macro_unlagged = align_unlagged(macro_t, calendar)

    exo_actor = pd.concat([asset_block, idx_lr, vix_aligned,
                           macro_lagged.add_suffix("_lagged")], axis=1)
    exo_critic_extra = macro_unlagged.add_suffix("_unlagged")

    # --- NaN policy: drop leading NaNs, assert none remain -----------------
    first_valid = max(df.dropna().index.min()
                      for df in (exo_actor, exo_critic_extra))
    dates = calendar[calendar >= first_valid]

    return FeatureSet(
        dates=dates,
        prices=prices.loc[dates],
        exo_actor=exo_actor.loc[dates],
        exo_critic_extra=exo_critic_extra.loc[dates],
    )
