from __future__ import annotations

from v4.agentContract.environmentMetadata import V4EnvironmentMetadata
from v4.agentContract.types import V4Observation
from v4.memory.localMemory import LocalMemoryStateV4
from v4.state.stateParser import StateParserV4
from v4.time_reactive.stateBuilder import TimeReactiveStateBuilderV4


class HazardForecastV4:
    def forecast(self, current_observation: V4Observation, environment_metadata: V4EnvironmentMetadata | None) -> int | None:
        if not isinstance(current_observation, V4Observation):
            return None
        raw_game_id = str(current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        if game_id != "sv01":
            return None
        try:
            parsed_state = StateParserV4().build_parsed_state(
                current_observation=current_observation,
                previous_observation=None,
                environment_metadata=environment_metadata,
                local_memory_snapshot=LocalMemoryStateV4(),
                step_index=0,
            )
            typed_state = TimeReactiveStateBuilderV4().build(parsed_state)
        except Exception:
            typed_state = None
        if typed_state is not None:
            return int(typed_state.family.survival_timer_remaining)
        state = current_observation.state
        if isinstance(state, str):
            value = self._extract_int_after_prefix(state, "hazard_window=")
            if value is not None:
                return value
        if environment_metadata is None:
            return None
        additional_properties = environment_metadata.raw_payload.get("additional_properties", {})
        if isinstance(additional_properties, dict):
            fallback = additional_properties.get("sv01_default_hazard_window")
            if isinstance(fallback, int):
                return fallback
        return None

    def _extract_int_after_prefix(self, text: str, prefix: str) -> int | None:
        marker = text.find(prefix)
        if marker < 0:
            return None
        start = marker + len(prefix)
        end = start
        if end < len(text) and text[end] in "+-":
            end += 1
        while end < len(text) and text[end].isdigit():
            end += 1
        if end == start or (end == start + 1 and text[start] in "+-"):
            return None
        try:
            return int(text[start:end])
        except ValueError:
            return None
