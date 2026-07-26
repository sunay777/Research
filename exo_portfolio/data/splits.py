"""Chronological splits and rolling-origin walk-forward folds (Manual D.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray   # positional indices into the aligned date axis
    val_idx: np.ndarray
    test_idx: np.ndarray
    name: str

    def __post_init__(self):
        assert len(self.train_idx) and len(self.val_idx) and len(self.test_idx)
        # strict chronology, no overlap (Manual D.5 acceptance)
        assert self.train_idx[-1] < self.val_idx[0] <= self.val_idx[-1] < self.test_idx[0]


def chrono_split(n: int, train_frac: float = 0.70,
                 val_frac: float = 0.15) -> Fold:
    """Base chronological 70/15/15 split over n aligned days."""
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))
    return Fold(np.arange(0, t_end), np.arange(t_end, v_end),
                np.arange(v_end, n), name="chrono")


def rolling_origin_folds(dates: pd.DatetimeIndex, n_folds: int = 5,
                         first_test_frac: float = 0.50,
                         val_frac_of_train: float = 0.15) -> list[Fold]:
    """>=5 walk-forward folds: the region after `first_test_frac` is cut into
    `n_folds` consecutive test blocks; each fold trains on everything before
    its block (minus a trailing validation slice).

    With 2010-2024 data and defaults, the test blocks tile ~2017-2024, so the
    2020 COVID window and the 2022 inflation window each fall inside a test
    block (assert with `folds_cover`).
    """
    n = len(dates)
    test_start0 = int(n * first_test_frac)
    edges = np.linspace(test_start0, n, n_folds + 1, dtype=int)

    folds = []
    for k in range(n_folds):
        ts, te = edges[k], edges[k + 1]
        n_val = max(1, int(ts * val_frac_of_train))
        folds.append(Fold(
            train_idx=np.arange(0, ts - n_val),
            val_idx=np.arange(ts - n_val, ts),
            test_idx=np.arange(ts, te),
            name=f"fold{k}_{dates[ts].date()}_{dates[te - 1].date()}",
        ))
    return folds


def folds_cover(dates: pd.DatetimeIndex, folds: list[Fold],
                windows: tuple[tuple[str, str], ...] = (
                    ("2020-02-15", "2020-04-30"),   # COVID crash
                    ("2022-01-01", "2022-10-31"),   # inflation drawdown
                )) -> bool:
    """True iff every required stress window intersects >=1 fold's test block."""
    for lo, hi in windows:
        lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
        if not any(((dates[f.test_idx] >= lo_t) & (dates[f.test_idx] <= hi_t)).any()
                   for f in folds):
            return False
    return True
