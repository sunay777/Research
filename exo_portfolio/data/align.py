"""Time alignment, publication lags, and feature building (Manual D.3-D.4).

Cardinal rule (Manual A.1.4): any value visible at observation time `t` must
have been knowable at the market close of day `t`.

Two aligned macro versions are produced:
- ``exo_actor``        — publication-lagged (deployment-realistic)
- ``exo_critic_extra`` — values visible from the end of their reference
  period, without the publication delay (privileged, consumed ONLY by the
  asymmetric critic). Still causal: never describes days after t.
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

# Each derived feature inherits the publication lag of the raw FRED series it
# is computed from (the lag config is keyed by raw FRED code).
MACRO_SOURCE = {"CPI_YOY": "CPIAUCSL"}


def transform_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """Turn raw FRED levels into model features.

    CPIAUCSL (CPI level) -> CPI_YOY, year-on-year inflation. The YoY value for
    reference month m is computable the moment month m's CPI is published, so
    it inherits CPIAUCSL's publication lag (see MACRO_SOURCE).

    A missing reference month (e.g. October 2025 CPI, never collected during
    the US government shutdown) stays missing: it is NOT padded from the
    previous month, so that month simply carries no new value.
    """
    out = macro.copy()
    if "CPIAUCSL" in out.columns:
        out["CPI_YOY"] = out["CPIAUCSL"].pct_change(12, fill_method=None)
        out = out.drop(columns=["CPIAUCSL"])
    return out


def reference_period_end(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """FRED stamps a monthly observation with the FIRST day of its reference
    month, but a monthly average or index level is only complete at the END of
    that month. Map each stamp to the last calendar day of its month."""
    return index + pd.offsets.MonthEnd(0)


def macro_lag(col: str, lags: dict[str, int]) -> int:
    """Publication lag for a (possibly derived) macro column. A column with no
    configured lag is an error, never a silent zero."""
    key = MACRO_SOURCE.get(col, col)
    if key not in lags:
        raise KeyError(f"no publication lag configured for macro column "
                       f"{col!r} (looked up {key!r}); add it to "
                       f"data.macro_publication_lag_days")
    return int(lags[key])


def apply_publication_lag(macro: pd.DataFrame, lags: dict[str, int],
                          calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Shift each series to its publication date, then forward-fill onto the
    trading calendar so day t carries the most recent value *published on or
    before* t (Manual D.4).

    Publication date = end of the reference month + lag (calendar days), so a
    lag is "days after the reference period closes", matching how statistical
    agencies schedule releases.
    """
    cols = {}
    for col in macro.columns:
        lag = macro_lag(col, lags)
        s = macro[col].dropna()
        pub = reference_period_end(s.index) + pd.Timedelta(days=lag)
        cols[col] = _ffill_onto(pd.Series(s.values, index=pub, name=col), calendar)
    return pd.DataFrame(cols, index=calendar)


def align_unlagged(macro: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Privileged (critic-only) version: each value becomes visible at the END
    of its reference month, i.e. as soon as the period it measures is over,
    without waiting for the publication delay.

    It never contains information about days after t (the period it describes
    has fully elapsed); it is privileged only because the deployed actor has to
    wait for the official release.
    """
    cols = {}
    for c in macro.columns:
        s = macro[c].dropna()
        s = pd.Series(s.values, index=reference_period_end(s.index), name=c)
        cols[c] = _ffill_onto(s, calendar)
    return pd.DataFrame(cols, index=calendar)


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


# ---------------------------------------------------------------------------
# Exo feature-ablation axis (M10) — mask NAMED groups of exo_actor.
#
# A SEPARATE experimental axis from the architecture grid (like the J.5 cost
# sweep): it degrades one named group of the ACTOR's exogenous features while
# leaving obs dimensionality untouched, so the custom dual-encoder cells keep
# their fixed ExoEncoder layout (Manual G.1). Physical column drop is never
# used — every mode is in-place.
# ---------------------------------------------------------------------------

EXO_GROUPS = ("asset_returns", "index", "vix", "macro")
EXO_MASK_MODES = ("keep", "zero", "permute", "noise")


def exo_group_slices(n_assets: int, price_window: int,
                     n_exo_cols: int) -> dict[str, slice]:
    """Column ranges of each NAMED exo_actor group, matching the build_features
    layout: [ N*price_window asset-return cols ] + gspc_lr + vix + macro...
    """
    n_aw = n_assets * price_window
    return {
        "asset_returns": slice(0, n_aw),
        "index": slice(n_aw, n_aw + 1),
        "vix": slice(n_aw + 1, n_aw + 2),
        "macro": slice(n_aw + 2, n_exo_cols),
    }


def mask_exo_array(exo: np.ndarray, sl: slice, mode: str,
                   rng: np.random.Generator) -> np.ndarray:
    """Return a copy of `exo` (T, d) with columns `sl` degraded per `mode`.

    - keep    : identity.
    - zero    : columns set to 0 (the feature is removed but its slot remains).
    - permute : rows of the block permuted along TIME. The permutation is drawn
                from a fixed seed, independent of the price path, so it injects
                no predictive (lookahead) signal — it only destroys the
                feature's temporal alignment. The column value multiset is
                preserved (a genuine reordering), which is what makes it a
                causal-safe ablation rather than a data leak.
    - noise   : columns replaced by standard-normal noise (deterministic seed).
    """
    assert mode in EXO_MASK_MODES, mode
    out = exo.copy()
    if mode == "keep":
        return out
    block = out[:, sl]
    if mode == "zero":
        out[:, sl] = 0.0
    elif mode == "permute":
        out[:, sl] = block[rng.permutation(block.shape[0])]
    elif mode == "noise":
        out[:, sl] = rng.standard_normal(block.shape).astype(out.dtype)
    return out


def apply_exo_ablation(features: "FeatureSet", cfg: Config) -> "FeatureSet":
    """Return a FeatureSet with cfg.exo_ablation applied to exo_actor.

    A no-op when group is 'none' or mode is 'keep'. Dimension-preserving:
    columns/index are untouched so every cell (classical, SB3, custom) consumes
    the same shape (Manual G.1 dimensionality rule for cell2/cell4).
    """
    ab = cfg.exo_ablation
    if ab.group == "none" or ab.mode == "keep":
        return features
    assert ab.group in EXO_GROUPS, f"unknown exo group: {ab.group}"

    exo = features.exo_actor.values.astype(np.float32)
    n_assets = features.prices.shape[1]
    sl = exo_group_slices(n_assets, cfg.data.price_window,
                          exo.shape[1])[ab.group]
    masked = mask_exo_array(exo, sl, ab.mode, np.random.default_rng(ab.seed))
    exo_actor = pd.DataFrame(masked, index=features.exo_actor.index,
                             columns=features.exo_actor.columns)
    return FeatureSet(dates=features.dates, prices=features.prices,
                      exo_actor=exo_actor,
                      exo_critic_extra=features.exo_critic_extra)


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
