from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from v4.state.parsedState import ParsedStateV4


_MOVEMENT_GAMES = {"ul01", "fs01", "fs02", "fs03", "tp01", "ic01", "va01", "pb01", "pb02", "pb03"}
_CLICK_GAMES = {"pt01", "sy01", "ff01", "sq01", "wm01", "mm01"}


@dataclass(frozen=True)
class ProbeTemplateV4:
    probe_id: str
    family: str
    goal_kind: str
    description: str
    allowed_action_names: tuple[str, ...]
    requires_frontier: bool = False

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("probe_id must be non-empty")
        if not self.family:
            raise ValueError("family must be non-empty")
        if not self.goal_kind:
            raise ValueError("goal_kind must be non-empty")
        if not self.description:
            raise ValueError("description must be non-empty")
        if not isinstance(self.allowed_action_names, tuple) or not self.allowed_action_names:
            raise ValueError("allowed_action_names must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_probe_templates(parsed_state: ParsedStateV4) -> tuple[ProbeTemplateV4, ...]:
    game_id = str(parsed_state.current_observation.game_id).split("-", 1)[0]
    unknown_fallback = ProbeTemplateV4(
        probe_id="probe:unknown:inspect",
        family="unknown",
        goal_kind="reveal_information",
        description="Use a low-risk local action to reveal hidden information.",
        allowed_action_names=("inspect", "inspect_local", "up", "down", "left", "right"),
        requires_frontier=True,
    )
    if game_id in _MOVEMENT_GAMES:
        return (
            ProbeTemplateV4(
                probe_id="probe:movement:frontier_reveal",
                family="movement",
                goal_kind="reveal_information",
                description="Approach or inspect a frontier-adjacent area to reveal hidden cells.",
                allowed_action_names=("inspect", "inspect_local", "up", "down", "left", "right"),
                requires_frontier=True,
            ),
            unknown_fallback,
        )
    if game_id in _CLICK_GAMES:
        return (
            ProbeTemplateV4(
                probe_id="probe:click:frontier_reveal",
                family="click",
                goal_kind="reveal_information",
                description="Use a low-risk local click or inspection action to reveal hidden cells.",
                allowed_action_names=("inspect", "inspect_local", "click_at"),
                requires_frontier=True,
            ),
            unknown_fallback,
        )
    return (unknown_fallback,)
