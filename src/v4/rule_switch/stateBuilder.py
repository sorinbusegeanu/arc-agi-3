from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_rs01_rule_switch_state
from .typedState import RuleSwitchTypedStateV4


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


class RuleSwitchStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str | None = None) -> RuleSwitchTypedStateV4:
        chosen_family = family or parsed_state.current_observation.game_id.split("-", 1)[0]
        required_fields = "game_family,avatar_position,legal_action_ids,active_safe_color,safe_color_cycle,layout_evidence_source"
        if chosen_family != "rs01":
            _raise_builder_error(f"unsupported rule_switch family: {chosen_family}", parsed_state=parsed_state, missing_field="game_family", required_fields=required_fields)
        try:
            state = build_rs01_rule_switch_state(parsed_state)
        except Exception as exc:
            raise _annotate_builder_exception(
                exc,
                parsed_state=parsed_state,
                missing_field=_fallback_missing_field(exc, "active_safe_color"),
                required_fields=str(getattr(exc, "required_fields", "") or required_fields),
            )
        if state.common.game_family != chosen_family:
            _raise_builder_error("rule_switch state builder produced mismatched family state", parsed_state=parsed_state, missing_field="game_family", required_fields=required_fields)
        if not state.common.legal_action_ids:
            _raise_builder_error("rule_switch state builder requires at least one legal action", parsed_state=parsed_state, missing_field="legal_action_ids", required_fields=required_fields)
        if state.family.active_safe_color is None:
            _raise_builder_error("rs01 rule_switch state requires active_safe_color", parsed_state=parsed_state, missing_field="active_safe_color", required_fields=required_fields)
        if not state.family.safe_color_cycle:
            _raise_builder_error("rs01 rule_switch state requires safe_color_cycle", parsed_state=parsed_state, missing_field="safe_color_cycle", required_fields=required_fields)
        if state.layout_evidence_source is None:
            _raise_builder_error("rule_switch state builder requires layout_evidence_source", parsed_state=parsed_state, missing_field="layout_evidence_source", required_fields=required_fields)
        return state
