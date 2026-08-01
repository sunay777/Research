"""Ablation-cell presets (Manual Part H) — torch-free.

Lives apart from agent.py so grid tooling (run_grid.py) and CI can validate
the experiment grid without importing torch.
"""

from __future__ import annotations

from exo_portfolio.config import ModelCfg

# Part H — the ablation grid as presets of the three switches.
CELL_PRESETS: dict[str, dict] = {
    "cell0": dict(encoder="single", critic="symmetric", exo_mode="none"),
    "cell1": dict(encoder="single", critic="symmetric", exo_mode="concat"),
    "cell2": dict(encoder="dual", critic="symmetric", exo_mode="split"),
    "cell3": dict(encoder="single", critic="asymmetric", exo_mode="concat"),
    "cell4": dict(encoder="dual", critic="asymmetric", exo_mode="split"),
}

VALID_COMBOS = {("single", "none"), ("single", "concat"), ("dual", "split")}


def model_cfg_for_cell(cell: str, base: ModelCfg | None = None) -> ModelCfg:
    base = base or ModelCfg()
    preset = CELL_PRESETS[cell]
    return ModelCfg(encoder=preset["encoder"], critic=preset["critic"],
                    exo_mode=preset["exo_mode"],
                    d_endo=base.d_endo, d_exo=base.d_exo)
