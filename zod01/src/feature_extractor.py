from __future__ import annotations

from .types import ActionProposal, ControllerContext


def candidate_features(
    proposal: ActionProposal,
    ctx: ControllerContext,
    heuristic_score: float,
    mech_bias: float,
    safety_penalty: float,
) -> list[float]:
    action = proposal.action.name
    return [
        1.0 if action == "ACTION6" else 0.0,
        1.0 if action == "RESET" else 0.0,
        1.0 if "cycle-risk" in proposal.tags else 0.0,
        min(1.0, safety_penalty),
        mech_bias,
        heuristic_score,
    ]
