from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_tb01_hybrid_construction_state
from .typedState import HybridConstructionTypedStateV4


class HybridConstructionStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str | None = None) -> HybridConstructionTypedStateV4:
        chosen_family = family or parsed_state.current_observation.game_id.split("-", 1)[0]
        if chosen_family != "tb01":
            raise ValueError(f"unsupported hybrid_construction family: {chosen_family}")
        return build_tb01_hybrid_construction_state(parsed_state)
