from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from v4.planning.planContracts import CandidatePlanV4
from v4.state.parsedState import ParsedStateV4


@dataclass(frozen=True)
class ExpectedEvidenceV4:
    hypothesis_id: str
    expected_outcome_code: str
    supports_hypothesis: bool

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must be non-empty")
        if not self.expected_outcome_code:
            raise ValueError("expected_outcome_code must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExpectedEvidenceModelV4:
    def predict(self, parsed_state: ParsedStateV4, candidate: CandidatePlanV4) -> tuple[ExpectedEvidenceV4, ...]:
        del parsed_state
        if candidate.goal_kind != "disambiguate_hypothesis":
            return ()
        target_hypothesis_ids = tuple(str(item) for item in candidate.expected_effect.get("target_hypothesis_ids", ()))
        template_id = str(candidate.expected_effect.get("experiment_template_id", ""))
        if not template_id or not target_hypothesis_ids:
            return ()
        predicted: list[ExpectedEvidenceV4] = []
        for hypothesis_id in target_hypothesis_ids:
            if template_id == "experiment:rs01:test_safe_color":
                predicted.append(ExpectedEvidenceV4(hypothesis_id, f"safe_color_probe:{hypothesis_id}", True))
            elif template_id == "experiment:pt01:test_board_phase":
                predicted.append(ExpectedEvidenceV4(hypothesis_id, f"board_phase_probe:{hypothesis_id}", True))
        return tuple(predicted)
