from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_ms01_memory_hidden_state
from .typedState import MemoryHiddenTypedStateV4


class MemoryHiddenStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str | None = None) -> MemoryHiddenTypedStateV4:
        chosen_family = family or parsed_state.current_observation.game_id.split("-", 1)[0]
        if chosen_family != "ms01":
            raise ValueError(f"unsupported memory_hidden family: {chosen_family}")
        state = build_ms01_memory_hidden_state(parsed_state)
        if state.common.game_family != chosen_family:
            raise ValueError("memory_hidden state builder produced mismatched family state")
        return state
