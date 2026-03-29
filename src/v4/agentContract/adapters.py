from __future__ import annotations

from typing import Any

from .types import (
    V4Action,
    V4AuthoritativeState,
    V4Observation,
    V4StepResult,
    V4TransitionRecord,
)
from .validators import (
    derive_terminal_signal,
    validate_authoritative_state,
    validate_v4_action,
    validate_v4_observation,
    validate_v4_step_result,
    validate_v4_transition_record,
)


def _freeze_frame_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return tuple(_freeze_frame_value(item) for item in value)
    return value


def _frame_to_tuple(frame: Any) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    if frame is None:
        raise ValueError("frame: missing mandatory field from raw observation")
    frozen = _freeze_frame_value(frame)
    if not isinstance(frozen, tuple):
        raise ValueError("frame: could not normalize raw frame to tuple form")
    return frozen


def _action_input_to_dict(action_input: Any) -> dict[str, Any]:
    if action_input is None:
        raise ValueError("action_input: missing mandatory field from raw observation")
    if hasattr(action_input, "model_dump"):
        payload = action_input.model_dump(mode="json")
    elif isinstance(action_input, dict):
        payload = dict(action_input)
    else:
        raise ValueError("action_input: unsupported raw type")
    if not isinstance(payload, dict):
        raise ValueError("action_input: expected dict payload")
    return payload


def observation_from_frame(raw_frame: Any) -> V4Observation:
    required = (
        "game_id",
        "state",
        "levels_completed",
        "win_levels",
        "action_input",
        "full_reset",
        "available_actions",
    )
    for field_name in required:
        if not hasattr(raw_frame, field_name):
            raise ValueError(f"{field_name}: missing mandatory field from raw frame object")
    if not hasattr(raw_frame, "frame"):
        raise ValueError("frame: missing mandatory field from raw frame object")
    state_value = getattr(raw_frame, "state")
    state_name = getattr(state_value, "value", state_value)
    raw_payload = {
        "game_id": getattr(raw_frame, "game_id"),
        "frame": _freeze_frame_value(getattr(raw_frame, "frame")),
        "state": state_name,
        "levels_completed": getattr(raw_frame, "levels_completed"),
        "win_levels": getattr(raw_frame, "win_levels"),
        "action_input": _action_input_to_dict(getattr(raw_frame, "action_input")),
        "guid": getattr(raw_frame, "guid", None),
        "full_reset": getattr(raw_frame, "full_reset"),
        "available_actions": list(getattr(raw_frame, "available_actions")),
    }
    observation = V4Observation(
        raw_object_name=type(raw_frame).__name__,
        raw_payload=raw_payload,
        game_id=raw_payload["game_id"],
        frame=_frame_to_tuple(getattr(raw_frame, "frame")),
        state=str(raw_payload["state"]),
        levels_completed=int(raw_payload["levels_completed"]),
        win_levels=int(raw_payload["win_levels"]),
        action_input=raw_payload["action_input"],
        guid=raw_payload["guid"],
        full_reset=bool(raw_payload["full_reset"]),
        available_actions=tuple(int(action_id) for action_id in raw_payload["available_actions"]),
    )
    validate_v4_observation(observation)
    return observation


def static_authoritative_state_fragment_from_info(env_info: Any) -> dict[str, Any]:
    if env_info is None:
        return {}
    fragment = {}
    for field_name in ("game_id", "title", "description", "action_space"):
        if hasattr(env_info, field_name):
            fragment[field_name] = getattr(env_info, field_name)
    return fragment


def authoritative_state_from_sources(
    observation: V4Observation,
    static_fragment: dict[str, Any] | None = None,
) -> V4AuthoritativeState:
    validate_v4_observation(observation)
    fragment = dict(static_fragment or {})
    state = V4AuthoritativeState(
        game_id=observation.game_id,
        state=observation.state,
        levels_completed=observation.levels_completed,
        win_levels=observation.win_levels,
        full_reset=observation.full_reset,
        available_actions=observation.available_actions,
        guid=observation.guid,
        title=fragment.get("title"),
        description=fragment.get("description"),
        action_space=fragment.get("action_space"),
        metadata={k: v for k, v in fragment.items() if k not in {"game_id", "title", "description", "action_space"}},
    )
    validate_authoritative_state(state)
    return state


def action_from_enum(action_enum: Any, payload: dict[str, Any] | None = None, reasoning: Any | None = None) -> V4Action:
    if not hasattr(action_enum, "value") or not hasattr(action_enum, "name"):
        raise ValueError("action_enum: must expose 'value' and 'name'")
    action = V4Action(
        action_id=int(getattr(action_enum, "value")),
        action_name=str(getattr(action_enum, "name")),
        payload=dict(payload) if payload is not None else None,
        reasoning=reasoning,
    )
    validate_v4_action(action)
    return action


def transition_record_from_step(
    pre_observation: V4Observation,
    action: V4Action,
    post_observation: V4Observation,
    *,
    execution_status: str = "executed",
    step_index: int | None = None,
    timestamp_ms: int | None = None,
) -> V4TransitionRecord:
    validate_v4_observation(pre_observation)
    validate_v4_observation(post_observation)
    validate_v4_action(action)
    action_legal = action.action_id in pre_observation.available_actions
    record = V4TransitionRecord(
        pre_observation=pre_observation,
        action=action,
        post_observation=post_observation,
        action_legal=action_legal,
        execution_status=execution_status if action_legal else "rejected",
        terminal_signal=derive_terminal_signal(post_observation),
        step_index=step_index,
        timestamp_ms=timestamp_ms,
    )
    validate_v4_transition_record(record)
    return record


def step_result_from_transition(record: V4TransitionRecord) -> V4StepResult:
    validate_v4_transition_record(record)
    coordinate_payload = None
    if record.action.action_id == 6 and record.action.payload is not None:
        coordinate_payload = {
            "x": int(record.action.payload["x"]),
            "y": int(record.action.payload["y"]),
        }
    result = V4StepResult(
        action=record.action,
        action_legal=record.action_legal,
        terminal_signal=record.terminal_signal,
        raw_state_before=record.pre_observation.state,
        raw_state_after=record.post_observation.state,
        levels_completed_delta=record.post_observation.levels_completed - record.pre_observation.levels_completed,
        win_levels_delta=record.post_observation.win_levels - record.pre_observation.win_levels,
        reset_required=record.terminal_signal.reset_required,
        coordinate_payload=coordinate_payload,
    )
    validate_v4_step_result(result)
    return result
