from __future__ import annotations

from v4.agentContract.environmentMetadata import V4EnvironmentMetadata
from v4_5.adapters.actionAdapter import ActionAdapter
from v4_5.contracts import GameControlProfile


class ControlProfileAdapter:
    def __init__(self, action_adapter: ActionAdapter | None = None) -> None:
        self.action_adapter = action_adapter or ActionAdapter()

    def build_profile(
        self,
        *,
        environment_metadata: V4EnvironmentMetadata,
        enabled_action_ids: tuple[int, ...],
    ) -> GameControlProfile:
        max_action_set = self.action_adapter.available_primitive_actions(tuple(environment_metadata.action_ids))
        enabled_actions = self.action_adapter.available_primitive_actions(tuple(enabled_action_ids))
        movement_actions = tuple(action for action in enabled_actions if action in {"UP", "DOWN", "LEFT", "RIGHT"})
        click_actions = tuple(action for action in enabled_actions if action == "CLICK")
        if movement_actions and click_actions:
            category = "move_and_click"
        elif click_actions:
            category = "click_only"
        else:
            category = "movement_only"
        return GameControlProfile(
            control_category=category,
            max_action_set=max_action_set,
            enabled_actions=enabled_actions,
            movement_actions=movement_actions,
            click_actions=click_actions,
            supports_wait=any(action in {"WAIT", "NOOP"} for action in enabled_actions),
        )
