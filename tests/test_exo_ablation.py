"""M10 acceptance: the exo feature-ablation axis. Masking is dimension-
preserving, permute-in-time is a causal-safe reordering, and masking a group
changes exactly that group's columns and nothing else.
"""

import numpy as np
import pytest

from exo_portfolio.baselines.sb3_baselines import ExoMaskWrapper
from exo_portfolio.config import Config
from exo_portfolio.data.align import (EXO_GROUPS, apply_exo_ablation,
                                      build_features, exo_group_slices,
                                      mask_exo_array)
from exo_portfolio.envs.portfolio_env import PortfolioEnv
from exo_portfolio.exo_ablation import expand, load_spec

from tests.conftest import make_bundle


@pytest.fixture(scope="module")
def features():
    cfg = Config()
    cfg.data.price_window = 10
    fs = build_features(**make_bundle(seed=3, start="2014-01-01",
                                      end="2016-12-31", n_assets=4), cfg=cfg)
    return cfg, fs


def test_group_slices_tile_exactly(features):
    cfg, fs = features
    n_assets = fs.prices.shape[1]
    n_cols = fs.exo_actor.shape[1]
    sl = exo_group_slices(n_assets, cfg.data.price_window, n_cols)
    assert set(sl) == set(EXO_GROUPS)
    # the four groups partition the exo_actor columns with no gap/overlap
    covered = np.zeros(n_cols, dtype=int)
    for s in sl.values():
        covered[s] += 1
    assert (covered == 1).all()
    assert sl["asset_returns"] == slice(0, n_assets * cfg.data.price_window)
    assert sl["macro"].stop == n_cols


@pytest.mark.parametrize("mode", ["zero", "permute", "noise"])
def test_mask_preserves_dimensionality(features, mode):
    cfg, fs = features
    exo = fs.exo_actor.values.astype(np.float32)
    sl = exo_group_slices(fs.prices.shape[1], cfg.data.price_window,
                          exo.shape[1])["vix"]
    out = mask_exo_array(exo, sl, mode, np.random.default_rng(0))
    assert out.shape == exo.shape                      # (i) dims preserved


def test_mask_changes_only_that_group(features):
    cfg, fs = features
    exo = fs.exo_actor.values.astype(np.float32)
    slices = exo_group_slices(fs.prices.shape[1], cfg.data.price_window,
                              exo.shape[1])
    sl = slices["index"]
    out = mask_exo_array(exo, sl, "zero", np.random.default_rng(0))
    # (iii) the masked column actually changed ...
    assert not np.array_equal(out[:, sl], exo[:, sl])
    assert (out[:, sl] == 0).all()
    # ... and every OTHER column is byte-identical
    others = np.ones(exo.shape[1], dtype=bool)
    others[sl] = False
    assert np.array_equal(out[:, others], exo[:, others])


def test_permute_in_time_is_causal_safe(features):
    cfg, fs = features
    exo = fs.exo_actor.values.astype(np.float32)
    sl = exo_group_slices(fs.prices.shape[1], cfg.data.price_window,
                          exo.shape[1])["macro"]
    a = mask_exo_array(exo, sl, "permute", np.random.default_rng(7))
    b = mask_exo_array(exo, sl, "permute", np.random.default_rng(7))
    # reproducible from a fixed seed (drawn independently of the price path,
    # so it injects no lookahead) ...
    assert np.array_equal(a, b)
    # ... and it is a genuine REORDERING: the per-column value multiset is
    # preserved (no external/future values introduced), which is what makes
    # it a safe ablation rather than a data leak.
    for c in range(sl.start, sl.stop):
        assert np.array_equal(np.sort(a[:, c]), np.sort(exo[:, c]))
    # a different seed gives a different permutation
    c = mask_exo_array(exo, sl, "permute", np.random.default_rng(8))
    assert not np.array_equal(a, c)


def test_apply_exo_ablation_is_noop_when_keep(features):
    cfg, fs = features
    cfg2 = Config()
    cfg2.data.price_window = cfg.data.price_window
    cfg2.exo_ablation.group = "none"
    cfg2.exo_ablation.mode = "keep"
    out = apply_exo_ablation(fs, cfg2)
    assert out is fs                                   # identity, no copy


def test_apply_exo_ablation_masks_group(features):
    cfg, fs = features
    cfg2 = Config()
    cfg2.data.price_window = cfg.data.price_window
    cfg2.exo_ablation.group = "vix"
    cfg2.exo_ablation.mode = "zero"
    out = apply_exo_ablation(fs, cfg2)
    # dims + columns + index preserved
    assert out.exo_actor.shape == fs.exo_actor.shape
    assert list(out.exo_actor.columns) == list(fs.exo_actor.columns)
    sl = exo_group_slices(fs.prices.shape[1], cfg.data.price_window,
                          fs.exo_actor.shape[1])["vix"]
    assert (out.exo_actor.values[:, sl] == 0).all()
    # untouched channels: prices + privileged critic extra pass through
    assert out.prices is fs.prices
    assert out.exo_critic_extra is fs.exo_critic_extra


def test_wrapper_preserves_dims_and_serves_masked(features):
    cfg, fs = features
    base = PortfolioEnv(fs, cfg)
    d_exo = base.observation_space["exo_actor"].shape[0]
    w = ExoMaskWrapper(PortfolioEnv(fs, cfg), group="vix", mode="zero", seed=0)
    obs, _ = w.reset(seed=0)
    assert obs["exo_actor"].shape == (d_exo,)           # (i) dims preserved
    sl = exo_group_slices(fs.prices.shape[1], cfg.data.price_window,
                          d_exo)["vix"]
    assert (obs["exo_actor"][sl] == 0).all()            # masked group served
    # a few steps keep serving masked rows
    for _ in range(5):
        obs, _, term, trunc, _ = w.step(base.action_space.sample())
        assert (obs["exo_actor"][sl] == 0).all()
        if term or trunc:
            break


def test_expand_control_and_cross_product(tmp_path):
    spec = load_spec(_write_spec(tmp_path))
    runs = expand(spec)
    controls = [r for r in runs if r["mode"] == "keep"]
    ablations = [r for r in runs if r["mode"] != "keep"]
    # one keep/none control per (cell, seed, fold)
    assert len(controls) == len(spec["cells"]) * len(spec["seeds"]) * len(spec["folds"])
    assert all(r["group"] == "none" for r in controls)
    # cross product of (group x mode) ablations per (cell, seed, fold)
    assert len(ablations) == (len(spec["groups"]) * len(spec["modes"])
                              * len(spec["cells"]) * len(spec["seeds"])
                              * len(spec["folds"]))


def _write_spec(tmp_path):
    import yaml
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump({
        "cells": ["cell4", "cell1"], "groups": ["vix", "macro"],
        "modes": ["zero", "permute"], "seeds": [0], "folds": [0],
        "mask_seed": 0}))
    return p
