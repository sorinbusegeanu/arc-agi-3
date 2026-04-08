from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from v4.policy.policyBase import _ACTION_NAME_BY_ID
from v4.state.parsedState import ParsedStateV4


_MOVEMENT_FAMILIES = {"ul01", "fs01", "fs02", "fs03", "tp01", "ic01", "va01", "pb01", "pb02", "pb03"}
_CLICK_FAMILIES = {"pt01", "sy01", "ff01", "sq01", "wm01", "mm01"}


@dataclass(frozen=True)
class AffordanceSetV4:
    family: str
    available_action_ids: tuple[int, ...]
    available_action_names: tuple[str, ...]
    retry_counts: dict[str, int] = field(default_factory=dict)
    cooldown_action_keys: tuple[str, ...] = ()
    visited_before: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_common_affordances(parsed_state: ParsedStateV4) -> AffordanceSetV4:
    game_id = str(parsed_state.current_observation.game_id)
    game_stem = game_id.split("-", 1)[0]
    if game_stem in _MOVEMENT_FAMILIES:
        family = "movement"
    elif game_stem in _CLICK_FAMILIES:
        family = "click"
    else:
        family = "unknown"
    action_ids = tuple(sorted(int(action_id) for action_id in parsed_state.available_actions))
    action_names = tuple(_ACTION_NAME_BY_ID[int(action_id)] for action_id in action_ids)
    memory_reference = parsed_state.memory_reference
    return AffordanceSetV4(
        family=family,
        available_action_ids=action_ids,
        available_action_names=action_names,
        retry_counts=dict(parsed_state.derived_control.retry_counts),
        cooldown_action_keys=tuple(parsed_state.derived_control.cooldown_action_keys),
        visited_before=bool(memory_reference is not None and memory_reference.visited_before),
    )
