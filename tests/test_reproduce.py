"""M9 acceptance (light): stages are wired, figures render to files from
fake results, saved-policy loading degrades gracefully."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from exo_portfolio.eval import figures as F
from exo_portfolio.reproduce import ALL_STAGES, BUDGETS, _load_policy
from exo_portfolio.config import Config


def test_budgets_and_stages():
    assert set(BUDGETS) == {"smoke", "full"}
    assert BUDGETS["full"]["seeds"] == list(range(10))
    assert BUDGETS["full"]["folds"] == list(range(5))
    assert ALL_STAGES == ("data", "grid", "classical", "synthetic",
                          "report", "figures")


def _fake_results(root: Path):
    for cell in ("cell1", "cell4"):
        d = root / f"{cell}_seed0_fold0"
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(yaml.safe_dump(
            {"cell": cell, "seed": 0, "fold": 0}))
        rng = np.random.default_rng(0)
        pd.DataFrame({
            "update": range(10), "env_steps": np.arange(10) * 100,
            "mean_reward": np.linspace(-0.5, 0.2, 10) + rng.normal(0, 0.01, 10),
        }).to_csv(d / "train_log.csv", index=False)
        dates = pd.bdate_range("2020-01-01", periods=60)
        pd.DataFrame({
            "date": dates, "log_return": rng.normal(0, 0.01, 60),
            "turnover": np.abs(rng.normal(0, 0.02, 60)),
            "exposure": np.clip(rng.normal(0.7, 0.1, 60), 0, 1),
        }).to_csv(d / "series_test.csv", index=False)


def test_figures_from_fake_results(tmp_path):
    _fake_results(tmp_path)
    out = F.fig_learning_curves(tmp_path, tmp_path / "figs" / "lc.png")
    assert out and out.exists()

    pd.DataFrame({
        "cell": ["cell1"] * 4 + ["cell4"] * 4,
        "seed": [0, 1] * 4,
        "shift": [0.0, 0.0, 1.0, 1.0] * 2,
        "mean_reward": np.linspace(0.1, -0.1, 8),
        "oracle_mean_reward": 0.12,
        "regret": np.linspace(0.02, 0.22, 8),
    }).to_csv(tmp_path / "syn.csv", index=False)
    out = F.fig_synthetic_shift(tmp_path / "syn.csv", tmp_path / "figs" / "ss.png")
    assert out and out.exists()

    out = F.fig_equity_curves(tmp_path, tmp_path / "figs" / "eq.png")
    assert out and out.exists()

    betas = pd.DataFrame({"cell": ["cell1", "cell4"],
                          "beta": [-0.001, -0.004], "se": [0.0005, 0.0005]})
    out = F.fig_exposure_vix_betas(betas, tmp_path / "figs" / "bv.png")
    assert out and out.exists()

    responses = {"cell4": {"n_spikes": 3,
                           "mean_response": [-0.1, -0.2, -0.15],
                           "se_response": [0.02, 0.03, 0.02],
                           "cumulative_10d": -0.15}}
    out = F.fig_impulse_response(responses, tmp_path / "figs" / "ir.png")
    assert out and out.exists()


def test_figures_graceful_when_empty(tmp_path):
    assert F.fig_learning_curves(tmp_path, tmp_path / "x.png") is None
    assert F.fig_synthetic_shift(tmp_path / "nope.csv", tmp_path / "x.png") is None
    assert F.fig_equity_curves(tmp_path, tmp_path / "x.png") is None
    assert F.fig_impulse_response({}, tmp_path / "x.png") is None


def test_load_policy_none_when_no_model(tmp_path):
    assert _load_policy(tmp_path, None, Config()) is None
