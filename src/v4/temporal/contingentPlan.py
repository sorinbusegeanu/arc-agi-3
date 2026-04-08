from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from v4.planning.planContracts import CandidatePlanV4
from v4.state.parsedState import ParsedStateV4


@dataclass(frozen=True)
class ContingentPlanNoteV4:
    trigger_code: str
    fallback_action_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.trigger_code:
            raise ValueError("trigger_code must be non-empty")
        if not self.fallback_action_names:
            raise ValueError("fallback_action_names must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContingentPlanAnnotatorV4:
    def build_note(self, parsed_state: ParsedStateV4, candidate: CandidatePlanV4) -> ContingentPlanNoteV4 | None:
        del candidate
        if parsed_state.current_observation.game_id != "sv01":
            return None
        if parsed_state.temporal_reference is None:
            return None
        if parsed_state.temporal_reference.safe_horizon_steps <= 1:
            return ContingentPlanNoteV4(
                trigger_code="low_safe_horizon",
                fallback_action_names=("inspect", "inspect_local", "up", "down", "left", "right"),
            )
        return None
