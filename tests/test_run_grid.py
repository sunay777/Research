"""M6 acceptance: grid expansion, preset consistency, job emission,
skip-completed resumability, and aggregation."""

import json
from pathlib import Path

import pytest
import yaml

from exo_portfolio.run_grid import (aggregate, expand, is_complete,
                                    job_command, load_grid, summary_table)

GRID_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_grid.yaml"


def test_grid_expands_to_full_matrix():
    grid = load_grid(GRID_PATH)
    combos = expand(grid)
    assert len(combos) == 5 * 10 * 5                    # cells × seeds × folds
    assert len(set(combos)) == len(combos)              # unique
    assert combos[0] == ("cell0", 0, 0)                 # deterministic order


def test_yaml_matches_agent_presets():
    # load_grid raises if experiment_grid.yaml diverges from CELL_PRESETS
    load_grid(GRID_PATH)
    bad = {"cells": {"cell0": {"encoder": "dual", "critic": "symmetric",
                               "exo_mode": "none"}},
           "seeds": [0], "folds": [0]}
    with pytest.raises(AssertionError):
        from exo_portfolio.run_grid import _validate_cells
        _validate_cells(bad["cells"])


def test_job_command():
    cmd = job_command(["a.yaml", "b.yaml"], "cell3", 7, 2, steps=1000)
    assert "--config a.yaml b.yaml" in cmd
    assert "--cell cell3" in cmd and "--seed 7" in cmd and "--fold 2" in cmd
    assert "--steps 1000" in cmd


def _fake_run(root: Path, cell: str, seed: int, fold: int, sharpe: float):
    run_id = f"{cell}_seed{seed}_fold{fold}"
    d = root / run_id
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(
        {"cell": cell, "seed": seed, "fold": fold}))
    row = {"cum_log_return": 0.1, "sharpe": sharpe, "sortino": 1.0,
           "max_drawdown": 0.2, "mean_turnover": 0.01, "n_steps": 100}
    (d / "metrics.json").write_text(json.dumps(
        {"train": row, "val": row, "test": {**row, "sharpe": sharpe}}))


def test_skip_completed(tmp_path):
    _fake_run(tmp_path, "cell0", 0, 0, 1.0)
    assert is_complete(tmp_path / "cell0_seed0_fold0")
    assert not is_complete(tmp_path / "cell1_seed0_fold0")


def test_aggregate_and_summary(tmp_path):
    _fake_run(tmp_path, "cell0", 0, 0, 0.5)
    _fake_run(tmp_path, "cell0", 1, 0, 1.5)
    _fake_run(tmp_path, "cell4", 0, 0, 2.0)
    df = aggregate(tmp_path)
    assert len(df) == 9                                  # 3 runs × 3 splits
    assert (tmp_path / "summary.csv").exists()
    table = summary_table(df, "test")
    assert table.loc["cell0", "sharpe_mean"] == pytest.approx(1.0)
    assert table.loc["cell0", "sharpe_std"] == pytest.approx(0.7071, abs=1e-3)
    assert table.loc["cell4", "sharpe_mean"] == pytest.approx(2.0)
