from __future__ import annotations

from rl_v1.planner.beam_planner import LatentBeamPlanner


class BeamSearchPlanner(LatentBeamPlanner):
    def __init__(self, cfg) -> None:
        super().__init__(
            beam_width=int(getattr(cfg, "beam_width")),
            search_depth=int(getattr(cfg, "search_depth", getattr(cfg, "depth", 1))),
            action_topk=int(getattr(cfg, "action_topk", getattr(cfg, "expansion_width", 1))),
            discount=float(getattr(cfg, "discount", 0.99)),
        )

    def plan(self, latent, action_mask, policy_logits, value_head=None, dynamics=None, model_interfaces=None):
        interfaces = model_interfaces or {"dynamics": dynamics, "value_head": value_head}
        output = super().plan(latent, action_mask, policy_logits, interfaces)
        return output.selected_first_action, {
            "branches": [output.selected_branch_action_sequence],
            **output.diagnostics,
        }


__all__ = ["LatentBeamPlanner", "BeamSearchPlanner"]
