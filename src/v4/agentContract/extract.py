from __future__ import annotations

from typing import Any

from .adapters import (
    authoritative_state_from_sources,
    observation_from_frame,
    static_authoritative_state_fragment_from_info,
    step_result_from_transition,
    transition_record_from_step,
)
from .environmentMetadata import COORDINATE_BOUNDS_ACTION_NAME, V4EnvironmentMetadata
from .errors import (
    V4AdapterError,
    V4IllegalActionError,
    V4InvalidActionError,
    V4InvalidPayloadError,
    V4InvalidTerminalSignalError,
    V4InvalidTransitionError,
    V4MetadataMismatchError,
    V4MissingFieldError,
    V4UnknownFieldError,
    V4ValidationError,
)
from .types import V4Action, V4AuthoritativeState, V4Observation, V4StepResult, V4TransitionRecord
from .validators import (
    derive_terminal_signal,
    validate_authoritative_state,
    validate_v4_action,
    validate_v4_observation,
    validate_v4_step_result,
    validate_v4_transition_record,
)


def _raise_validation_error(exc: Exception, *, default_source_field: str) -> None:
    message = str(exc)
    source_field = default_source_field
    if ":" in message:
        head, tail = message.split(":", 1)
        if head and " " not in head:
            source_field = head
            message = tail.strip()
    lowered = message.lower()
    if "missing" in lowered:
        raise V4MissingFieldError(message, source_field=source_field)
    if "unsupported authoritative fields" in lowered or "unsupported fields" in lowered:
        raise V4UnknownFieldError(message, source_field=source_field)
    if "not legal" in lowered:
        raise V4IllegalActionError(message, source_field=source_field)
    if "unknown gameaction id" in lowered or "action_name" in source_field:
        raise V4InvalidActionError(message, source_field=source_field)
    if "payload" in source_field or "coordinate_payload" in source_field:
        raise V4InvalidPayloadError(message, source_field=source_field)
    if "terminal_signal" in source_field:
        raise V4InvalidTerminalSignalError(message, source_field=source_field)
    if "transition" in lowered or source_field in {"execution_status", "action_legal"}:
        raise V4InvalidTransitionError(message, source_field=source_field)
    raise V4ValidationError(message, source_field=source_field)


def _safe_attr(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    return value


def _coordinate_bounds_from_action(action: Any) -> tuple[int, int, int, int] | None:
    action_type = getattr(action, "action_type", None)
    if action_type is None or not hasattr(action_type, "model_json_schema"):
        return None
    schema = action_type.model_json_schema()
    properties = schema.get("properties", {})
    x_field = properties.get("x")
    y_field = properties.get("y")
    if not isinstance(x_field, dict) or not isinstance(y_field, dict):
        return None
    if {"minimum", "maximum"} - set(x_field.keys()):
        return None
    if {"minimum", "maximum"} - set(y_field.keys()):
        return None
    return (
        int(x_field["minimum"]),
        int(y_field["minimum"]),
        int(x_field["maximum"]),
        int(y_field["maximum"]),
    )


def extract_v4_observation_from_env_output(raw_env_output: Any) -> V4Observation:
    try:
        return observation_from_frame(raw_env_output)
    except ValueError as exc:
        _raise_validation_error(exc, default_source_field="raw_env_output")
    except Exception as exc:
        raise V4AdapterError(str(exc), source_field="raw_env_output", context={"source_type": type(raw_env_output).__name__}) from exc


def extract_v4_environment_metadata(source: Any) -> V4EnvironmentMetadata:
    env_obj = source if hasattr(source, "action_space") or hasattr(source, "info") else None
    info_obj = getattr(source, "info", None) if env_obj is not None else source
    raw_payload: dict[str, Any] = {}
    if info_obj is not None:
        for field_name in ("game_id", "title", "description", "local_dir", "date_downloaded"):
            if hasattr(info_obj, field_name):
                raw_payload[field_name] = _safe_attr(getattr(info_obj, field_name))
    action_space = list(getattr(env_obj, "action_space", []) or [])
    if action_space:
        raw_payload["action_space"] = [
            {
                "name": str(getattr(action, "name", "")),
                "value": int(getattr(action, "value", -1)),
                "is_complex": bool(getattr(action, "is_complex", lambda: False)()),
            }
            for action in action_space
        ]
    game_id = raw_payload.get("game_id")
    if not isinstance(game_id, str) or not game_id:
        raise V4MissingFieldError("missing required environment metadata field", source_field="env.info.game_id")
    action_ids = tuple(int(getattr(action, "value")) for action in action_space if hasattr(action, "value"))
    action_names = tuple(str(getattr(action, "name")) for action in action_space if hasattr(action, "name"))
    coordinate_action_id = None
    coordinate_bounds = None
    for action in action_space:
        name = str(getattr(action, "name", ""))
        is_complex = bool(getattr(action, "is_complex", lambda: False)())
        if is_complex and name == COORDINATE_BOUNDS_ACTION_NAME:
            coordinate_action_id = int(getattr(action, "value"))
            coordinate_bounds = _coordinate_bounds_from_action(action)
            break
    metadata = V4EnvironmentMetadata(
        source_object_name=type(info_obj).__name__ if info_obj is not None else type(source).__name__,
        raw_payload=raw_payload,
        game_id=game_id,
        title=raw_payload.get("title"),
        description=raw_payload.get("description"),
        local_dir=raw_payload.get("local_dir"),
        date_downloaded=raw_payload.get("date_downloaded"),
        action_ids=action_ids,
        action_names=action_names,
        coordinate_action_id=coordinate_action_id,
        coordinate_bounds=coordinate_bounds,
    )
    if metadata.action_ids and metadata.game_id and raw_payload.get("game_id") != metadata.game_id:
        raise V4MetadataMismatchError("normalized game_id does not match raw metadata", source_field="game_id")
    return metadata


def extract_v4_authoritative_state(
    observation: V4Observation,
    environment_metadata: V4EnvironmentMetadata | None = None,
) -> V4AuthoritativeState:
    try:
        validate_v4_observation(observation)
        static_fragment = {}
        if environment_metadata is not None:
            static_fragment = {
                "game_id": environment_metadata.game_id,
                "title": environment_metadata.title,
                "description": environment_metadata.description,
            }
            if environment_metadata.game_id != observation.game_id:
                raise V4MetadataMismatchError(
                    "environment metadata game_id does not match observation.game_id",
                    source_field="game_id",
                    context={"metadata_game_id": environment_metadata.game_id, "observation_game_id": observation.game_id},
                )
        state = authoritative_state_from_sources(observation, static_fragment)
        validate_authoritative_state(state)
        return state
    except V4MetadataMismatchError:
        raise
    except ValueError as exc:
        _raise_validation_error(exc, default_source_field="authoritative_state")


def build_v4_transition_record(
    pre_observation: V4Observation,
    action: V4Action,
    post_observation: V4Observation,
    *,
    execution_status: str = "executed",
    step_index: int | None = None,
    timestamp_ms: int | None = None,
) -> V4TransitionRecord:
    try:
        validate_v4_observation(pre_observation)
        validate_v4_action(action, observation=pre_observation)
        validate_v4_observation(post_observation)
        record = transition_record_from_step(
            pre_observation,
            action,
            post_observation,
            execution_status=execution_status,
            step_index=step_index,
            timestamp_ms=timestamp_ms,
        )
        validate_v4_transition_record(record)
        return record
    except ValueError as exc:
        _raise_validation_error(exc, default_source_field="transition_record")


def build_v4_step_result(record: V4TransitionRecord) -> V4StepResult:
    try:
        validate_v4_transition_record(record)
        terminal_signal = derive_terminal_signal(record.post_observation)
        if terminal_signal != record.terminal_signal:
            raise V4InvalidTerminalSignalError(
                "transition terminal signal does not match post observation",
                source_field="terminal_signal",
            )
        result = step_result_from_transition(record)
        validate_v4_step_result(result)
        return result
    except V4InvalidTerminalSignalError:
        raise
    except ValueError as exc:
        _raise_validation_error(exc, default_source_field="step_result")

