from __future__ import annotations

import json
from typing import Any

from .types import (
    V4Action,
    V4AuthoritativeState,
    V4Observation,
    V4StepResult,
    V4TerminalSignal,
    V4TransitionRecord,
)

_FRAME_DATA_RAW_FIELDS = {
    "game_id",
    "frame",
    "state",
    "levels_completed",
    "win_levels",
    "action_input",
    "guid",
    "full_reset",
    "available_actions",
}
_GAME_ACTIONS = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}
_GAME_STATES = {"NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"}
_TERMINAL_STATUSES = {"not_played", "non_terminal", "success", "failure"}


def _fail(source_field: str, message: str) -> None:
    raise ValueError(f"{source_field}: {message}")


def _validate_json_serialisable(source_field: str, value: Any) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        _fail(source_field, f"must be JSON-serialisable: {exc}")


def _validate_frame(source_field: str, frame: Any) -> None:
    if not isinstance(frame, tuple):
        _fail(source_field, "must be a tuple of frame planes")
    for plane_index, plane in enumerate(frame):
        if not isinstance(plane, tuple):
            _fail(f"{source_field}[{plane_index}]", "must be a tuple of rows")
        for row_index, row in enumerate(plane):
            if not isinstance(row, tuple):
                _fail(f"{source_field}[{plane_index}][{row_index}]", "must be a tuple of cells")


def validate_observation_payload(raw_payload: dict[str, Any]) -> None:
    if not isinstance(raw_payload, dict):
        _fail("raw_payload", "must be a dict")
    keys = set(raw_payload.keys())
    extra = keys - _FRAME_DATA_RAW_FIELDS
    missing = _FRAME_DATA_RAW_FIELDS - keys
    if extra:
        _fail("raw_payload", f"contains unsupported authoritative fields: {sorted(extra)}")
    if missing:
        _fail("raw_payload", f"missing required authoritative fields: {sorted(missing)}")


def validate_v4_observation(observation: V4Observation) -> None:
    if observation.raw_object_name != "FrameDataRaw":
        _fail("raw_object_name", "must be 'FrameDataRaw'")
    validate_observation_payload(observation.raw_payload)
    if not isinstance(observation.game_id, str):
        _fail("game_id", "must be a string")
    if observation.state not in _GAME_STATES:
        _fail("state", f"must be one of {sorted(_GAME_STATES)}")
    if not isinstance(observation.levels_completed, int):
        _fail("levels_completed", "must be an int")
    if not isinstance(observation.win_levels, int):
        _fail("win_levels", "must be an int")
    if not isinstance(observation.action_input, dict):
        _fail("action_input", "must be a dict")
    action_input_extra = set(observation.action_input.keys()) - {"id", "data", "reasoning"}
    if action_input_extra:
        _fail("action_input", f"contains unsupported fields: {sorted(action_input_extra)}")
    if observation.guid is not None and not isinstance(observation.guid, str):
        _fail("guid", "must be a string or null")
    if not isinstance(observation.full_reset, bool):
        _fail("full_reset", "must be a bool")
    if not isinstance(observation.available_actions, tuple):
        _fail("available_actions", "must be a tuple of ints")
    for index, action_id in enumerate(observation.available_actions):
        if not isinstance(action_id, int):
            _fail(f"available_actions[{index}]", "must be an int")
        if action_id not in _GAME_ACTIONS:
            _fail(f"available_actions[{index}]", f"unknown GameAction id {action_id}")
    _validate_frame("frame", observation.frame)
    _validate_json_serialisable("raw_payload", observation.raw_payload)


def validate_v4_action(action: V4Action, observation: V4Observation | None = None) -> None:
    if action.action_id not in _GAME_ACTIONS:
        _fail("action_id", f"unknown GameAction id {action.action_id}")
    expected_name = _GAME_ACTIONS[action.action_id]
    if action.action_name != expected_name:
        _fail("action_name", f"must be '{expected_name}' for action_id {action.action_id}")
    if observation is not None and action.action_id not in observation.available_actions:
        _fail("action_id", "is not legal for the provided observation.available_actions")
    if action.action_id == 6:
        if not isinstance(action.payload, dict):
            _fail("payload", "ACTION6 requires a payload object")
        payload_keys = set(action.payload.keys())
        extra = payload_keys - {"x", "y", "game_id"}
        if extra:
            _fail("payload", f"contains unsupported fields for ACTION6: {sorted(extra)}")
        for coord in ("x", "y"):
            if coord not in action.payload:
                _fail("payload", f"ACTION6 payload missing '{coord}'")
            value = action.payload[coord]
            if not isinstance(value, int):
                _fail(f"payload.{coord}", "must be an int")
            if value < 0 or value > 63:
                _fail(f"payload.{coord}", "must be in the inclusive range 0..63")
        if "game_id" in action.payload and not isinstance(action.payload["game_id"], str):
            _fail("payload.game_id", "must be a string when present")
    else:
        if action.payload is not None:
            payload_keys = set(action.payload.keys())
            extra = payload_keys - {"game_id"}
            if extra:
                _fail("payload", f"contains unsupported fields for simple action: {sorted(extra)}")
            if "game_id" in action.payload and not isinstance(action.payload["game_id"], str):
                _fail("payload.game_id", "must be a string when present")
    if action.reasoning is not None:
        _validate_json_serialisable("reasoning", action.reasoning)


def validate_authoritative_state(state: V4AuthoritativeState) -> None:
    if not isinstance(state.game_id, str):
        _fail("game_id", "must be a string")
    if state.state not in _GAME_STATES:
        _fail("state", f"must be one of {sorted(_GAME_STATES)}")
    if not isinstance(state.levels_completed, int):
        _fail("levels_completed", "must be an int")
    if not isinstance(state.win_levels, int):
        _fail("win_levels", "must be an int")
    if not isinstance(state.full_reset, bool):
        _fail("full_reset", "must be a bool")
    if not isinstance(state.available_actions, tuple):
        _fail("available_actions", "must be a tuple of ints")
    for index, action_id in enumerate(state.available_actions):
        if action_id not in _GAME_ACTIONS:
            _fail(f"available_actions[{index}]", f"unknown GameAction id {action_id}")
    if state.guid is not None and not isinstance(state.guid, str):
        _fail("guid", "must be a string or null")
    if state.title is not None and not isinstance(state.title, str):
        _fail("title", "must be a string or null")
    if state.description is not None and not isinstance(state.description, str):
        _fail("description", "must be a string or null")
    if state.action_space is not None and not isinstance(state.action_space, int):
        _fail("action_space", "must be an int or null")
    if not isinstance(state.metadata, dict):
        _fail("metadata", "must be a dict")
    _validate_json_serialisable("metadata", state.metadata)


def derive_terminal_signal(observation: V4Observation) -> V4TerminalSignal:
    validate_v4_observation(observation)
    if observation.state == "WIN":
        return V4TerminalSignal(
            status="success",
            raw_state=observation.state,
            is_terminal=True,
            reset_required=True,
            full_reset=observation.full_reset,
        )
    if observation.state == "GAME_OVER":
        return V4TerminalSignal(
            status="failure",
            raw_state=observation.state,
            is_terminal=True,
            reset_required=True,
            full_reset=observation.full_reset,
        )
    if observation.state == "NOT_FINISHED":
        return V4TerminalSignal(
            status="non_terminal",
            raw_state=observation.state,
            is_terminal=False,
            reset_required=False,
            full_reset=observation.full_reset,
        )
    return V4TerminalSignal(
        status="not_played",
        raw_state=observation.state,
        is_terminal=False,
        reset_required=False,
        full_reset=observation.full_reset,
    )


def validate_v4_terminal_signal(signal: V4TerminalSignal) -> None:
    if signal.status not in _TERMINAL_STATUSES:
        _fail("status", f"must be one of {sorted(_TERMINAL_STATUSES)}")
    if signal.raw_state not in _GAME_STATES:
        _fail("raw_state", f"must be one of {sorted(_GAME_STATES)}")
    if not isinstance(signal.is_terminal, bool):
        _fail("is_terminal", "must be a bool")
    if not isinstance(signal.reset_required, bool):
        _fail("reset_required", "must be a bool")
    if signal.full_reset is not None and not isinstance(signal.full_reset, bool):
        _fail("full_reset", "must be a bool or null")


def validate_v4_transition_record(record: V4TransitionRecord) -> None:
    validate_v4_observation(record.pre_observation)
    validate_v4_observation(record.post_observation)
    validate_v4_action(record.action)
    validate_v4_terminal_signal(record.terminal_signal)
    expected_signal = derive_terminal_signal(record.post_observation)
    if record.terminal_signal != expected_signal:
        _fail("terminal_signal", "must be derived from post_observation.state")
    if not isinstance(record.action_legal, bool):
        _fail("action_legal", "must be a bool")
    if record.execution_status not in {"executed", "rejected"}:
        _fail("execution_status", "must be 'executed' or 'rejected'")
    if record.step_index is not None and (not isinstance(record.step_index, int) or record.step_index < 0):
        _fail("step_index", "must be a non-negative int or null")
    if record.timestamp_ms is not None and (not isinstance(record.timestamp_ms, int) or record.timestamp_ms < 0):
        _fail("timestamp_ms", "must be a non-negative int or null")
    expected_legal = record.action.action_id in record.pre_observation.available_actions
    if record.action_legal != expected_legal:
        _fail("action_legal", "must match legality implied by pre_observation.available_actions")
    if not expected_legal and record.execution_status != "rejected":
        _fail("execution_status", "must be 'rejected' when the action is not legal")


def validate_v4_step_result(result: V4StepResult) -> None:
    validate_v4_action(result.action)
    validate_v4_terminal_signal(result.terminal_signal)
    if not isinstance(result.action_legal, bool):
        _fail("action_legal", "must be a bool")
    if result.raw_state_before not in _GAME_STATES:
        _fail("raw_state_before", f"must be one of {sorted(_GAME_STATES)}")
    if result.raw_state_after not in _GAME_STATES:
        _fail("raw_state_after", f"must be one of {sorted(_GAME_STATES)}")
    if result.levels_completed_delta is not None and not isinstance(result.levels_completed_delta, int):
        _fail("levels_completed_delta", "must be an int or null")
    if result.win_levels_delta is not None and not isinstance(result.win_levels_delta, int):
        _fail("win_levels_delta", "must be an int or null")
    if not isinstance(result.reset_required, bool):
        _fail("reset_required", "must be a bool")
    if result.reset_required != result.terminal_signal.reset_required:
        _fail("reset_required", "must match terminal_signal.reset_required")
    if result.coordinate_payload is not None:
        if result.action.action_id != 6:
            _fail("coordinate_payload", "is only valid for ACTION6")
        payload_keys = set(result.coordinate_payload.keys())
        if payload_keys != {"x", "y"}:
            _fail("coordinate_payload", "must contain exactly 'x' and 'y'")
        for coord in ("x", "y"):
            value = result.coordinate_payload[coord]
            if not isinstance(value, int):
                _fail(f"coordinate_payload.{coord}", "must be an int")
            if value < 0 or value > 63:
                _fail(f"coordinate_payload.{coord}", "must be in the inclusive range 0..63")
