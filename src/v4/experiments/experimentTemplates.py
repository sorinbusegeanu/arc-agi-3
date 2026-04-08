from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from v4.state.parsedState import ParsedStateV4


@dataclass(frozen=True)
class ExperimentTemplateV4:
    template_id: str
    hypothesis_kind: str
    description: str
    goal_kind: str
    allowed_action_names: tuple[str, ...]
    requires_hypotheses: bool = True
    max_risk_score: float = 1.0

    def __post_init__(self) -> None:
        for field_name in ("template_id", "hypothesis_kind", "description", "goal_kind"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        if not self.allowed_action_names:
            raise ValueError("allowed_action_names must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_experiment_templates(parsed_state: ParsedStateV4) -> tuple[ExperimentTemplateV4, ...]:
    if parsed_state.hypothesis_reference is None or parsed_state.hypothesis_reference.hypothesis_count <= 0:
        return ()
    raw_game_id = str(parsed_state.current_observation.game_id)
    game_id = raw_game_id.split("-", 1)[0]
    if game_id == "rs01":
        return (
            ExperimentTemplateV4(
                template_id="experiment:rs01:test_safe_color",
                hypothesis_kind="safe_color_rule",
                description="Take a low-risk action that can discriminate between competing safe-color rule hypotheses.",
                goal_kind="disambiguate_hypothesis",
                allowed_action_names=("up", "down", "left", "right", "inspect", "inspect_local"),
                requires_hypotheses=True,
                max_risk_score=1.0,
            ),
        )
    if game_id == "pt01":
        return (
            ExperimentTemplateV4(
                template_id="experiment:pt01:test_board_phase",
                hypothesis_kind="board_phase_rule",
                description="Take a low-risk action that can discriminate between competing board-phase hypotheses.",
                goal_kind="disambiguate_hypothesis",
                allowed_action_names=("click_at", "inspect", "inspect_local"),
                requires_hypotheses=True,
                max_risk_score=1.0,
            ),
        )
    return ()
