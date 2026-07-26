"""M3 acceptance: every metric matches a hand-computed value on a toy series
(Manual J.1 / F). Expected numbers below are worked out by hand in the
comments — they are NOT derived by calling the code under test.
"""

import math

import numpy as np
import pytest

from exo_portfolio.eval.metrics import (annualized_sharpe,
                                        cumulative_log_return, max_drawdown,
                                        mean_turnover, rolling_sharpe,
                                        sortino, summarize)

# Toy series used throughout: three daily log-returns
R = [0.01, -0.02, 0.03]
# mean = (0.01 - 0.02 + 0.03) / 3 = 0.02/3            = 0.006666...
M = 0.02 / 3


def test_cumulative_log_return():
    # 0.01 - 0.02 + 0.03 = 0.02
    assert cumulative_log_return(R) == pytest.approx(0.02)


def test_annualized_sharpe_hand_computed():
    # sample var (ddof=1) = [(0.01-M)^2 + (-0.02-M)^2 + (0.03-M)^2] / 2
    #   deviations: 0.0033333, -0.0266667, 0.0233333
    #   squares:    1.11111e-05, 7.11111e-04, 5.44444e-04 ; sum = 1.266667e-03
    #   var = 6.333333e-04 ; sd = 0.02516611
    sd = math.sqrt(((0.01 - M) ** 2 + (-0.02 - M) ** 2 + (0.03 - M) ** 2) / 2)
    expected = M / sd * math.sqrt(252)          # = 4.20535...
    assert annualized_sharpe(R) == pytest.approx(expected, rel=1e-12)
    assert annualized_sharpe(R) == pytest.approx(4.2053, abs=1e-3)


def test_sortino_hand_computed():
    # downside deviation over ALL n: sqrt((0 + 0.02^2 + 0) / 3)
    #   = sqrt(4e-4 / 3) = 0.01154701
    dd = math.sqrt((0.02 ** 2) / 3)
    expected = M / dd * math.sqrt(252)          # = 9.16515...
    assert sortino(R) == pytest.approx(expected, rel=1e-12)
    assert sortino(R) == pytest.approx(9.1651, abs=1e-3)
    # all-positive returns: no downside -> +inf by convention
    assert sortino([0.01, 0.02]) == float("inf")


def test_max_drawdown_hand_computed():
    # equity curve exp(cumsum(r)) for r = [ln 1.2, ln(0.75), ln(1.1/0.9)]:
    #   values 1.2, 0.9, 1.1 ; running peaks 1.2, 1.2, 1.2
    #   drawdowns 0, 0.25, 1 - 1.1/1.2 = 0.08333 ; max = 0.25
    r = [math.log(1.2), math.log(0.9 / 1.2), math.log(1.1 / 0.9)]
    assert max_drawdown(r) == pytest.approx(0.25, rel=1e-12)
    # monotone growth -> zero drawdown
    assert max_drawdown([0.01, 0.02, 0.005]) == pytest.approx(0.0)


def test_mean_turnover():
    assert mean_turnover([0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert mean_turnover([]) == 0.0


def test_rolling_sharpe_hand_computed():
    # population var (ddof=0) = 1.266667e-03 / 3 = 4.222222e-04
    var = ((0.01 - M) ** 2 + (-0.02 - M) ** 2 + (0.03 - M) ** 2) / 3
    expected = M / math.sqrt(var + 1e-8)
    assert rolling_sharpe(R) == pytest.approx(expected, rel=1e-12)
    assert rolling_sharpe([]) == 0.0


def test_degenerate_inputs():
    assert annualized_sharpe([0.01]) == 0.0          # < 2 observations
    assert annualized_sharpe([0.01, 0.01]) == 0.0    # zero variance
    assert max_drawdown([]) == 0.0
    assert sortino([]) == 0.0


def test_summarize_keys():
    row = summarize(R, [0.1, 0.2, 0.3])
    assert set(row) == {"cum_log_return", "sharpe", "sortino", "max_drawdown",
                        "mean_turnover", "n_steps"}
    assert row["n_steps"] == 3
    assert row["mean_turnover"] == pytest.approx(0.2)
