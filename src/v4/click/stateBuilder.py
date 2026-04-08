from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_click_state_for_family, detect_pt01_phase
from .typedState import ClickTypedStateV4


def _required_fields_for_family(family: str) -> str:
    common = "legal_action_ids,clickable_cells"
    family_fields = {
        "pt01": f"{common},rotation_tiles",
        "sy01": f"{common},reflection_axis_x,reflection_pairs,mirror_target_cells",
        "ff01": f"{common},fill_regions",
        "sq01": f"{common},sequence_order",
        "wm01": f"{common},mole_click_radius",
        "mm01": f"{common},memory_slot_colors",
    }
    return family_fields.get(family, common)


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
    lowered = message.lower()
    if lowered.startswith("pt01 transition stall"):
        return "pt01_transition_stall"
    if lowered.startswith("pt01 transition frame"):
        return "pt01_transition_frame"
    if lowered.startswith("pt01 state unavailable"):
        return "pt01_click_state"
    if lowered.startswith("wm01 state unavailable"):
        return "wm01_click_state"
    if lowered.startswith("wm01 config unavailable") and "current_level_index" in lowered:
        return "wm01_level_index"
    if lowered.startswith("wm01 config unavailable"):
        return "wm01_level_config"
    if "wm01" in lowered and "click radius" in lowered:
        return "wm01_click_state"
    if "wm01" in lowered and "clickable" in lowered:
        return "wm01_clickable_cells"
    if "wm01" in lowered and "mole" in lowered:
        return "wm01_visible_pattern_state"
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


class ClickStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str) -> ClickTypedStateV4:
        required_fields = _required_fields_for_family(family)
        if family == "pt01":
            phase = str(detect_pt01_phase(parsed_state).get("phase", ""))
            if phase == "pt01_transition_frame":
                _raise_builder_error(
                    "pt01 transition frame: awaiting stable new-level board",
                    parsed_state=parsed_state,
                    missing_field="pt01_transition_frame",
                    required_fields=required_fields,
                )
        try:
            state = build_click_state_for_family(parsed_state, family)
        except Exception as exc:
            raise _annotate_builder_exception(
                exc,
                parsed_state=parsed_state,
                missing_field=_fallback_missing_field(exc, "clickable_cells"),
                required_fields=str(getattr(exc, "required_fields", "") or required_fields),
            )
        if 6 not in state.common.legal_action_ids:
            _raise_builder_error(f"{family} state unavailable: ACTION6 not legal in current observation", parsed_state=parsed_state, missing_field="ACTION6", required_fields=required_fields)
        if not state.common.clickable_cells:
            _raise_builder_error(f"{family} state unavailable: no clickable cells extracted", parsed_state=parsed_state, missing_field="clickable_cells", required_fields=required_fields)
        if family == "pt01" and not state.family.rotation_tiles:
            _raise_builder_error("pt01 state unavailable: rotation tiles missing", parsed_state=parsed_state, missing_field="rotation_tiles", required_fields=required_fields)
        if family == "sy01":
            if state.family.reflection_axis_x is None:
                _raise_builder_error("sy01 state unavailable: reflection axis missing", parsed_state=parsed_state, missing_field="reflection_axis_x", required_fields=required_fields)
            if not state.family.reflection_pairs:
                _raise_builder_error("sy01 state unavailable: reflection pairs missing", parsed_state=parsed_state, missing_field="reflection_pairs", required_fields=required_fields)
            if not state.family.mirror_target_cells:
                _raise_builder_error("sy01 state unavailable: mirror targets missing", parsed_state=parsed_state, missing_field="mirror_target_cells", required_fields=required_fields)
        if family == "ff01" and not state.family.fill_regions:
            _raise_builder_error("ff01 state unavailable: fill regions missing", parsed_state=parsed_state, missing_field="fill_regions", required_fields=required_fields)
        if family == "sq01" and not state.family.sequence_order:
            _raise_builder_error("sq01 state unavailable: sequence order missing", parsed_state=parsed_state, missing_field="sequence_order", required_fields=required_fields)
        if family == "wm01" and state.family.mole_click_radius is None:
            _raise_builder_error("wm01 state unavailable: click radius missing", parsed_state=parsed_state, missing_field="mole_click_radius", required_fields=required_fields)
        if family == "mm01" and not state.family.memory_slot_colors:
            _raise_builder_error("mm01 state unavailable: slot colors missing", parsed_state=parsed_state, missing_field="memory_slot_colors", required_fields=required_fields)
        return state
