from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from v4.planning.planContracts import CandidatePlanV4
from v4.state.parsedState import ParsedStateV4


@dataclass(frozen=True)
class InformationGainScoreV4:
    information_value: float = 0.0
    safety_cost: float = 0.0
    reversibility_score: float = 0.0
    repeat_penalty: float = 0.0
    total_information_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InformationGainScorerV4:
    def score(self, parsed_state: ParsedStateV4, candidate: CandidatePlanV4) -> InformationGainScoreV4:
        first_action = candidate.action_prefix[0]
        belief_reference = parsed_state.belief_reference
        frontier_unknown_count = belief_reference.frontier_cell_count if belief_reference is not None else 0
        unknown_cell_count = (
            belief_reference.unknown_cell_count
            if belief_reference is not None
            else parsed_state.derived_control.unknown_cell_count
        )
        retry_count = float(parsed_state.derived_control.retry_counts.get(f"{first_action.action_id}:{first_action.action_name}", 0))
        cooldown_active = (
            1.0
            if f"{first_action.action_id}:{first_action.action_name}" in parsed_state.derived_control.cooldown_action_keys
            else 0.0
        )
        information_value = 1.0 if frontier_unknown_count > 0 and candidate.goal_kind == "reveal_information" else 0.0
        safety_cost = retry_count + cooldown_active
        reversibility_score = 1.0 if first_action.action_name in {"up", "down", "left", "right", "inspect", "inspect_local"} else 0.0
        repeat_penalty = retry_count
        total_information_score = information_value + reversibility_score - safety_cost - repeat_penalty
        if unknown_cell_count <= 0:
            information_value = 0.0
            total_information_score = reversibility_score - safety_cost - repeat_penalty
        return InformationGainScoreV4(
            information_value=information_value,
            safety_cost=safety_cost,
            reversibility_score=reversibility_score,
            repeat_penalty=repeat_penalty,
            total_information_score=total_information_score,
        )
