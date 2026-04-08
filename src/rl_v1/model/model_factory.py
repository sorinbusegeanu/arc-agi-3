from __future__ import annotations

from rl_v1.model.network import RLV1Model, RecurrentBaselineModel


def build_model(cfg):
    if cfg.model.variant == "recurrent_baseline":
        return RecurrentBaselineModel(cfg.model, cfg.planner)
    return RLV1Model(cfg.model, cfg.planner)
