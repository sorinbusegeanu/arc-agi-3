from __future__ import annotations

from v4.agentContract.environmentMetadata import V4EnvironmentMetadata
from v4.agentContract.types import V4Action, V4Observation
from v4.memory.localMemory import LocalMemoryStateV4
from v4.state.stateParser import StateParserV4
from v4.time_reactive.stateBuilder import TimeReactiveStateBuilderV4

from .hazardForecast import HazardForecastV4
from .resourceState import ResourceValueV4, TemporalResourceStateV4
from .timeCostModel import TimeCostModelV4


class TemporalUpdaterV4:
    def __init__(self) -> None:
        self.hazard_forecast = HazardForecastV4()
        self.time_cost_model = TimeCostModelV4()

    def initialize_from_observation(
        self,
        current_observation: V4Observation,
        environment_metadata: V4EnvironmentMetadata | None,
        step_index: int,
    ) -> TemporalResourceStateV4:
        del step_index
        if not isinstance(current_observation, V4Observation):
            return TemporalResourceStateV4(revision=0)
        raw_game_id = str(current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        if game_id != "sv01":
            return TemporalResourceStateV4(revision=0)
        resources = self._resources_from_observation(current_observation, environment_metadata)
        return TemporalResourceStateV4(
            revision=0,
            resources=resources,
            time_cost_per_action=1.0,
            hazard_window_remaining=self.hazard_forecast.forecast(current_observation, environment_metadata),
            safe_horizon_steps=self._safe_horizon(resources),
        )

    def update_after_step(
        self,
        previous_temporal: TemporalResourceStateV4 | None,
        post_observation: V4Observation,
        environment_metadata: V4EnvironmentMetadata | None,
        executed_action: V4Action,
        step_index: int,
    ) -> TemporalResourceStateV4:
        del step_index
        if not isinstance(post_observation, V4Observation):
            return TemporalResourceStateV4(revision=0 if previous_temporal is None else previous_temporal.revision + 1)
        raw_game_id = str(post_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        if game_id != "sv01":
            return TemporalResourceStateV4(revision=0 if previous_temporal is None else previous_temporal.revision + 1)
        resources = self._resources_from_observation(post_observation, environment_metadata)
        return TemporalResourceStateV4(
            revision=0 if previous_temporal is None else previous_temporal.revision + 1,
            resources=resources,
            time_cost_per_action=self.time_cost_model.cost_for_action(executed_action),
            hazard_window_remaining=self.hazard_forecast.forecast(post_observation, environment_metadata),
            safe_horizon_steps=self._safe_horizon(resources),
        )

    def _resources_from_observation(
        self,
        observation: V4Observation,
        environment_metadata: V4EnvironmentMetadata | None,
    ) -> tuple[ResourceValueV4, ...]:
        raw_game_id = str(observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        if game_id == "sv01":
            try:
                parsed_state = StateParserV4().build_parsed_state(
                    current_observation=observation,
                    previous_observation=None,
                    environment_metadata=environment_metadata,
                    local_memory_snapshot=LocalMemoryStateV4(),
                    step_index=0,
                )
                typed_state = TimeReactiveStateBuilderV4().build(parsed_state)
            except Exception:
                typed_state = None
            if typed_state is not None:
                return (
                    ResourceValueV4(name="hunger", current_value=float(typed_state.family.hunger_value), min_safe_value=1.0),
                    ResourceValueV4(name="warmth", current_value=float(typed_state.family.warmth_value), min_safe_value=1.0),
                    ResourceValueV4(name="timer", current_value=float(typed_state.family.survival_timer_remaining), min_safe_value=1.0),
                )
        resources: list[ResourceValueV4] = []
        for name, prefix in (("hunger", "hunger="), ("warmth", "warmth="), ("timer", "timer=")):
            current_value = self._extract_resource_value(observation.state, prefix)
            if current_value is None:
                raw_value = observation.raw_payload.get(name)
                if isinstance(raw_value, (int, float)):
                    current_value = float(raw_value)
            if current_value is None and isinstance(observation.action_input, dict):
                data = observation.action_input.get("data", {})
                if isinstance(data, dict):
                    action_value = data.get(name)
                    if isinstance(action_value, (int, float)):
                        current_value = float(action_value)
            if current_value is not None:
                resources.append(
                    ResourceValueV4(
                        name=name,
                        current_value=current_value,
                        min_safe_value=1.0,
                    )
                )
        return tuple(resources)

    def _safe_horizon(self, resources: tuple[ResourceValueV4, ...]) -> int:
        if not resources:
            return 0
        margins = [
            resource.current_value - resource.min_safe_value
            for resource in resources
            if resource.current_value >= resource.min_safe_value
        ]
        if not margins:
            return 0
        return max(0, int(min(margins)))

    def _extract_resource_value(self, state_text: object, prefix: str) -> float | None:
        if not isinstance(state_text, str):
            return None
        marker = state_text.find(prefix)
        if marker < 0:
            return None
        start = marker + len(prefix)
        end = start
        if end < len(state_text) and state_text[end] in "+-":
            end += 1
        while end < len(state_text) and state_text[end].isdigit():
            end += 1
        if end == start or (end == start + 1 and state_text[start] in "+-"):
            return None
        try:
            return float(int(state_text[start:end]))
        except ValueError:
            return None
