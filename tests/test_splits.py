"""M1 acceptance: train precedes val precedes test, no overlap; stress
windows (COVID 2020, inflation 2022) covered by >=1 test block."""

import numpy as np
import pandas as pd
import pytest

from exo_portfolio.data.splits import (Fold, chrono_split, folds_cover,
                                       rolling_origin_folds)

DATES = pd.bdate_range("2010-01-01", "2024-12-31")


def _assert_clean(fold: Fold, n: int):
    all_idx = np.concatenate([fold.train_idx, fold.val_idx, fold.test_idx])
    assert len(np.unique(all_idx)) == len(all_idx), "index overlap"
    assert fold.train_idx[-1] < fold.val_idx[0]
    assert fold.val_idx[-1] < fold.test_idx[0]
    assert all_idx.max() < n


def test_chrono_split_70_15_15():
    n = len(DATES)
    f = chrono_split(n)
    _assert_clean(f, n)
    assert abs(len(f.train_idx) / n - 0.70) < 0.01
    assert abs(len(f.val_idx) / n - 0.15) < 0.01


def test_rolling_origin_folds_clean():
    folds = rolling_origin_folds(DATES, n_folds=5)
    assert len(folds) >= 5
    for f in folds:
        _assert_clean(f, len(DATES))
    # origins actually roll forward
    starts = [f.test_idx[0] for f in folds]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    # test blocks tile the tail contiguously (collectively many regimes)
    for a, b in zip(folds[:-1], folds[1:]):
        assert a.test_idx[-1] + 1 == b.test_idx[0]


def test_stress_windows_covered():
    folds = rolling_origin_folds(DATES, n_folds=5)
    assert folds_cover(DATES, folds), \
        "COVID-2020 and inflation-2022 windows must appear in test blocks (D.5)"


def test_fold_rejects_overlap():
    with pytest.raises(AssertionError):
        Fold(np.arange(0, 10), np.arange(9, 12), np.arange(12, 20), "bad")
