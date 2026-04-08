from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_ms01_memory_hidden_state
from .typedState import MemoryHiddenTypedStateV4


def _current_visible_fields(parsed_state: ParsedStateV4) -> str:
    fields: list[str] = []
    for field_name in (
        "current_observation",
        "previous_observation",
        "environment_metadata",
        "available_actions",
        "terminal_signal",
        "memory_reference",
        "derived_control",
    ):
        if getattr(parsed_state, field_name, None) is not None:
            fields.append(field_name)
    return ",".join(fields)


def _annotate_builder_exception(exc: Exception, *, parsed_state: ParsedStateV4, missing_field: str, required_fields: str) -> Exception:
    setattr(exc, "abort_site", "state_builder.build")
    setattr(exc, "missing_field", missing_field)
    setattr(exc, "required_fields", required_fields)
    setattr(exc, "current_visible_fields", _current_visible_fields(parsed_state))
    setattr(exc, "previous_state_available", parsed_state.previous_observation is not None)
    setattr(exc, "reconstruction_attempted", False)
    return exc


def _fallback_missing_field(exc: Exception, default: str) -> str:
    explicit = str(getattr(exc, "missing_field", "") or "").strip()
    if explicit:
        return explicit
    message = str(exc or "").strip()
    if ": " in message:
        return message.split(": ", 1)[0].split()[-1]
    if " requires " in message:
        return message.split(" requires ", 1)[1].split()[0]
    return default


def _raise_builder_error(message: str, *, parsed_state: ParsedStateV4, missing_field: str, required_fields: str) -> None:
    raise _annotate_builder_exception(
        ValueError(message),
        parsed_state=parsed_state,
        missing_field=missing_field,
        required_fields=required_fields,
    )


class MemoryHiddenStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str | None = None) -> MemoryHiddenTypedStateV4:
        chosen_family = family or parsed_state.current_observation.game_id.split("-", 1)[0]
        required_fields = "game_family,avatar_position,traversable_safe_cells,current_legal_actions,visible_number_cells"
        if chosen_family != "ms01":
            _raise_builder_error(f"unsupported memory_hidden family: {chosen_family}", parsed_state=parsed_state, missing_field="game_family", required_fields=required_fields)
        try:
            state = build_ms01_memory_hidden_state(parsed_state)
        except Exception as exc:
            raise _annotate_builder_exception(
                exc,
                parsed_state=parsed_state,
                missing_field=_fallback_missing_field(exc, "game_family"),
                required_fields=str(getattr(exc, "required_fields", "") or required_fields),
            )
        if state.common.game_family != chosen_family:
            _raise_builder_error("memory_hidden state builder produced mismatched family state", parsed_state=parsed_state, missing_field="game_family", required_fields=required_fields)
        return state
