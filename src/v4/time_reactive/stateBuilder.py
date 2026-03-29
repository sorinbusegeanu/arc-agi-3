from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_sv01_time_reactive_state
from .typedState import TimeReactiveTypedStateV4


class TimeReactiveStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str | None = None) -> TimeReactiveTypedStateV4:
        chosen_family = family or parsed_state.current_observation.game_id.split("-", 1)[0]
        if chosen_family != "sv01":
            raise ValueError(f"unsupported time_reactive family: {chosen_family}")
        return build_sv01_time_reactive_state(parsed_state)
