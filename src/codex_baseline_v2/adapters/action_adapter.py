from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from codex_baseline_v2.shared.schemas import ActionDescriptorV2, SCHEMA_VERSION


def adapt_action(step: Dict[str, Any]) -> ActionDescriptorV2:
    action_key = step.get("action_key")
    action_id = step.get("action_id") if "action_id" in step else step.get("action_index")
    if isinstance(action_key, dict):
        action_type = str(action_key.get("type", "discrete"))
        if action_type == "coord":
            coord = (int(action_key.get("x", 0)), int(action_key.get("y", 0)))
            return ActionDescriptorV2(
                schema_version=SCHEMA_VERSION,
                action_type="coord",
                action_id=None,
                coord=coord,
                raw=action_key,
            )
        return ActionDescriptorV2(
            schema_version=SCHEMA_VERSION,
            action_type="discrete",
            action_id=int(action_id) if action_id is not None else None,
            coord=None,
            raw=action_key,
        )
    return ActionDescriptorV2(
        schema_version=SCHEMA_VERSION,
        action_type="discrete",
        action_id=int(action_id) if action_id is not None else None,
        coord=None,
        raw=None,
    )


def adapt_action_descriptor(action: Any) -> ActionDescriptorV2:
    if isinstance(action, ActionDescriptorV2):
        return action
    if isinstance(action, dict):
        return ActionDescriptorV2(
            schema_version=str(action.get("schema_version", SCHEMA_VERSION)),
            action_type=str(action.get("action_type", "discrete")),
            action_id=action.get("action_id"),
            coord=tuple(action.get("coord")) if action.get("coord") is not None else None,
            raw=action.get("raw"),
        )
    return ActionDescriptorV2(schema_version=SCHEMA_VERSION, action_type="discrete", action_id=None, coord=None, raw=None)
