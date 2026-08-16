"""M7 acceptance: causal regime labels, regime-conditional metrics,
paired stats with Holm correction, and the two J.4 diagnostics."""

import numpy as np
import pandas as pd
import pytest

from exo_portfolio.eval.diagnostics import (exposure_vix_regression,
                                            find_vix_spikes,
                                            vix_spike_impulse_response)
from exo_portfolio.eval.regimes import (label_stressed_vix,
                                        regime_conditional_metrics)
from exo_portfolio.eval.stats import (compare_cells, holm_correction,
                                      paired_test, seed_fold_dispersion)

DATES = pd.bdate_range("2015-01-01", "2020-12-31")
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ regimes

def _vix():
    base = np.clip(RNG.normal(16, 4, len(DATES)), 9, None)
    base[900:960] += 40                       # a crisis block
    return pd.Series(base, index=DATES)


def test_regime_labels_causal():
    vix = _vix()
    lab = label_stressed_vix(vix)
    cut = 700
    vix2 = vix.copy()
    vix2.iloc[cut + 1:] += 100.0              # future explosion
    lab2 = label_stressed_vix(vix2)
    assert lab.iloc[:cut + 1].equals(lab2.iloc[:cut + 1]), \
        "future VIX changed past labels — lookahead in regime labelling"


def test_regime_labels_catch_crisis():
    lab = label_stressed_vix(_vix())
    assert lab.iloc[900:960].mean() > 0.9      # crisis block flagged
    assert 0.05 < lab.mean() < 0.45            # neither empty nor everything


def test_regime_conditional_metrics_partition():
    n = 300
    idx = DATES[:n]
    r = pd.Series(RNG.normal(0, 0.01, n), index=idx)
    tau = pd.Series(np.abs(RNG.normal(0, 0.02, n)), index=idx)
    stressed = pd.Series([i % 5 == 0 for i in range(n)], index=idx)
    out = regime_conditional_metrics(r, tau, stressed)
    assert out["n_stressed_days"] == 60
    # calm + stressed partition the pooled window exactly
    assert (out["calm"]["n_steps"] + out["stressed"]["n_steps"]
            == out["pooled"]["n_steps"])
    total = out["calm"]["cum_log_return"] + out["stressed"]["cum_log_return"]
    assert total == pytest.approx(out["pooled"]["cum_log_return"])


# -------------------------------------------------------------------- stats

def test_holm_hand_computed():
    # sorted p: .01 -> *3 = .03 ; .03 -> *2 = .06 ; .04 -> *1 = .04 -> monotone .06
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([0.5]) == pytest.approx([0.5])


def test_paired_test_detects_shift():
    a = RNG.normal(1.0, 0.1, 30)
    res = paired_test(a + 0.5, a)
    assert res["pvalue"] < 1e-6 and res["mean_diff"] == pytest.approx(0.5)
    res_null = paired_test(a, a + RNG.normal(0, 1e-6, 30))
    assert res_null["pvalue"] > 0.01 or abs(res_null["mean_diff"]) < 1e-6


def _fake_long_df():
    rows = []
    for cell, edge in (("cell1", 0.0), ("cell4", 0.4)):
        for seed in range(6):
            for fold in range(3):
                rng = np.random.default_rng(seed * 10 + fold)
                base = rng.normal(0.5, 0.05)
                rows.append({"cell": cell, "seed": seed, "fold": fold,
                             "split": "test", "sharpe": base + edge,
                             "sortino": base + edge, "cum_log_return": 0.1 + edge,
                             "max_drawdown": 0.3 - edge / 4,
                             "mean_turnover": 0.01})
    return pd.DataFrame(rows)


def test_compare_cells_pairing_and_holm():
    df = _fake_long_df()
    out = compare_cells(df, target="cell4", baselines=("cell1",))
    assert len(out) == 4                                   # 4 metrics
    sharpe_row = out[out["metric"] == "sharpe"].iloc[0]
    assert sharpe_row["n"] == 18                           # 6 seeds x 3 folds
    assert sharpe_row["mean_diff"] == pytest.approx(0.4)
    assert sharpe_row["pvalue_holm"] < 0.05
    disp = seed_fold_dispersion(df)
    assert set(disp.index) == {"cell1", "cell4"}


def test_paired_test_zero_variance_baseline():
    """A deterministic baseline paired by fold can give a CONSTANT non-zero
    difference — must be handled without dividing by zero."""
    a = np.array([1.5, 1.5, 1.5, 1.5])          # target, constant
    b = np.array([1.0, 1.0, 1.0, 1.0])          # deterministic baseline
    res = paired_test(a, b)
    assert res["test"] == "constant_diff"
    assert res["pvalue"] == 0.0
    assert res["mean_diff"] == pytest.approx(0.5)
    assert np.isfinite(res["mean_diff"])         # no NaN from a 0/0


def test_compare_cells_pair_by_fold_deterministic_baseline():
    """A deterministic baseline (identical across seeds) is paired by FOLD;
    the comparison runs and holm-corrects without crashing."""
    rows = []
    for seed in range(4):
        for fold in range(5):
            rng = np.random.default_rng(seed * 10 + fold)
            rows.append({"cell": "cell4", "seed": seed, "fold": fold,
                         "split": "test", "sharpe": 1.0 + 0.1 * fold + rng.normal(0, 0.02),
                         "sortino": 1.0, "cum_log_return": 0.2, "max_drawdown": 0.2})
            # deterministic baseline: value depends only on fold, not seed
            rows.append({"cell": "equal_weight", "seed": seed, "fold": fold,
                         "split": "test", "sharpe": 0.5 + 0.1 * fold,
                         "sortino": 0.5, "cum_log_return": 0.1, "max_drawdown": 0.3})
    df = pd.DataFrame(rows)
    out = compare_cells(df, target="cell4", baselines=("equal_weight",),
                        pair_on=("fold",))
    sharpe_row = out[out["metric"] == "sharpe"].iloc[0]
    assert sharpe_row["n"] == 5                    # paired across 5 folds
    assert sharpe_row["pair_on"] == "fold"
    assert np.isfinite(sharpe_row["pvalue"])


# -------------------------------------------------------------- diagnostics

def test_exposure_vix_regression_recovers_negative_beta():
    vix = _vix()
    exposure = pd.Series(
        np.clip(1.1 - 0.02 * vix.values + RNG.normal(0, 0.03, len(vix)), 0, 1),
        index=DATES)
    res = exposure_vix_regression(exposure, vix)
    assert res["beta"] == pytest.approx(-0.02, abs=0.004)
    assert res["pvalue"] < 1e-6


def test_vix_spikes_causal_and_found():
    vix = _vix()
    spikes = find_vix_spikes(vix)
    assert DATES[900] in spikes                # crisis onset is a spike
    # future data cannot create/remove past spikes
    vix2 = vix.copy()
    vix2.iloc[1200:] += 50
    s1 = find_vix_spikes(vix)
    s2 = find_vix_spikes(vix2)
    assert s1[s1 < DATES[1200]].equals(s2[s2 < DATES[1200]])


def test_impulse_response_detects_derisking():
    # smooth VIX with ONE unambiguous spike at index 900
    vix = pd.Series(16 + RNG.normal(0, 0.5, len(DATES)).cumsum() * 0.01,
                    index=DATES)
    vix.iloc[900:960] += 40
    exposure = pd.Series(1.0, index=DATES)
    exposure.iloc[900:915] = 0.4               # agent de-risks after the spike
    res = vix_spike_impulse_response(exposure, vix)
    assert res["n_spikes"] == 1
    assert res["mean_response"][0] == pytest.approx(-0.6)
    assert res["mean_response"][10] == pytest.approx(-0.6)  # still de-risked
    # an exposure-blind agent shows no response
    flat = pd.Series(0.8, index=DATES)
    res_flat = vix_spike_impulse_response(flat, vix)
    assert max(abs(x) for x in res_flat["mean_response"]) == pytest.approx(0.0)
