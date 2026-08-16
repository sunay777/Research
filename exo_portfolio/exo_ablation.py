"""Exo feature-ablation sweep (M10) — a SEPARATE experimental axis.

Like run_grid.py drives the architecture grid, this drives the exo-feature
ablation axis (configs/exo_ablation.yaml): for a couple of reference cells it
masks one NAMED group of the actor's exogenous features at a time and re-runs
training. It NEVER touches configs/experiment_grid.yaml — the architecture
switches are held fixed; only cfg.exo_ablation changes (exactly like the J.5
cost sweep is a separate axis, not a new grid cell).

Each ablation run lands under results_root/exo_ablation/{group}__{mode}/{run_id}
so standard {cell}_seed{seed}_fold{fold} run-dirs never collide across the
(group, mode) axis. `aggregate` reads cfg.exo_ablation back out of each run's
config.yaml.

Usage:
    python -m exo_portfolio.exo_ablation --config configs/base.yaml \
        configs/universe_djia30.yaml --spec configs/exo_ablation.yaml \
        --run [--steps 4096]
    python -m exo_portfolio.exo_ablation --config ... --aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from exo_portfolio.config import Config
from exo_portfolio.data.align import EXO_GROUPS, EXO_MASK_MODES


def load_spec(path: str | Path) -> dict:
    with open(path) as fh:
        spec = yaml.safe_load(fh)
    for g in spec["groups"]:
        assert g in EXO_GROUPS, f"unknown exo group in spec: {g}"
    for m in spec["modes"]:
        assert m in EXO_MASK_MODES and m != "keep", f"bad ablation mode: {m}"
    return spec


def expand(spec: dict) -> list[dict]:
    """-> ordered run specs. One keep/none CONTROL per (cell, seed, fold), then
    the cross product of (group x mode) ablations. The control is the reference
    the ablations are read against."""
    runs: list[dict] = []
    mask_seed = int(spec.get("mask_seed", 0))
    for cell in spec["cells"]:
        for seed in spec["seeds"]:
            for fold in spec["folds"]:
                runs.append(dict(cell=cell, group="none", mode="keep",
                                 seed=seed, fold=fold, mask_seed=mask_seed))
                for group in spec["groups"]:
                    for mode in spec["modes"]:
                        runs.append(dict(cell=cell, group=group, mode=mode,
                                         seed=seed, fold=fold,
                                         mask_seed=mask_seed))
    return runs


def run_subdir(results_root: str | Path, group: str, mode: str) -> Path:
    return Path(results_root) / "exo_ablation" / f"{group}__{mode}"


def run_one(spec_run: dict, cfg_paths: list[str], results_root: str | Path,
            total_steps: int | None = None) -> dict:
    from exo_portfolio.train import run as train_run

    cfg = Config.from_yaml(*cfg_paths)
    cfg.cell = spec_run["cell"]
    cfg.seed = spec_run["seed"]
    cfg.fold = spec_run["fold"]
    cfg.exo_ablation.group = spec_run["group"]
    cfg.exo_ablation.mode = spec_run["mode"]
    cfg.exo_ablation.seed = spec_run["mask_seed"]
    sub = run_subdir(results_root, spec_run["group"], spec_run["mode"])
    return train_run(cfg, total_steps=total_steps, results_root=sub)


def aggregate(results_root: str | Path) -> pd.DataFrame:
    """Collect every exo-ablation run into one long frame keyed by
    (cell, group, mode, seed, fold) with the test metrics."""
    root = Path(results_root) / "exo_ablation"
    rows = []
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        run_dir = metrics_path.parent
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        ab = cfg.get("exo_ablation", {})
        test = json.loads(metrics_path.read_text()).get("test", {})
        rows.append({"cell": cfg["cell"], "group": ab.get("group", "none"),
                     "mode": ab.get("mode", "keep"), "seed": cfg["seed"],
                     "fold": cfg["fold"], **test})
    df = pd.DataFrame(rows)
    if len(df):
        out = Path(results_root) / "tables" / "exo_ablation.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--spec", default="configs/exo_ablation.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    runs = expand(spec)

    if args.run:
        from exo_portfolio.run_grid import is_complete

        for i, r in enumerate(runs):
            sub = run_subdir(args.results, r["group"], r["mode"])
            run_id = f"{r['cell']}_seed{r['seed']}_fold{r['fold']}"
            if is_complete(sub / run_id):
                print(f"[{i + 1}/{len(runs)}] {r['group']}/{r['mode']} "
                      f"{run_id}: complete, skipping")
                continue
            print(f"[{i + 1}/{len(runs)}] {r['group']}/{r['mode']} {run_id}")
            run_one(r, args.config, args.results, total_steps=args.steps)

    if args.aggregate or args.run:
        df = aggregate(args.results)
        print(f"exo_ablation: {len(df)} runs aggregated")
        if len(df):
            print(df.groupby(["cell", "group", "mode"])["sharpe"]
                  .mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
