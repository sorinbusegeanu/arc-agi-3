from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import (
    build_fs01_movement_state,
    build_fs02_movement_state,
    build_fs03_movement_state,
    build_ic01_movement_state,
    build_pb01_movement_state,
    build_pb02_movement_state,
    build_pb03_movement_state,
    build_tp01_movement_state,
    build_ul01_movement_state,
    build_va01_movement_state,
)
from .typedState import MovementTypedStateV4


def _required_fields_for_family(family: str) -> str:
    common = "game_family,avatar_position,current_legal_actions,layout_evidence_source"
    family_fields = {
        "ul01": common,
        "fs01": f"{common},switch_positions,door_open,door_state_bits,switch_logic_mode,activated_switch_bits,door_positions",
        "fs02": f"{common},switch_positions,door_open,door_state_bits,switch_logic_mode,occupied_switch_bits,switch_group_threshold,door_positions",
        "fs03": f"{common},switch_positions,door_open,door_state_bits,switch_logic_mode,activated_switch_bits,switch_group_threshold,door_positions",
        "tp01": f"{common},teleporter_endpoint_positions,teleporter_pairs,teleporter_pair_map,target_cells",
        "ic01": f"{common},slide_mode,ice_cell_positions",
        "va01": f"{common},coverage_eligible_cells,coverage_mask",
        "pb01": f"{common},pushable_block_positions,target_cells,push_target_cells,push_variant,step_limit",
        "pb02": f"{common},pushable_block_positions,target_cells,push_target_cells,push_variant,step_limit",
        "pb03": f"{common},pushable_block_positions,target_cells,push_target_cells,push_variant,step_limit,push_decoy_lose_cells",
    }
    return family_fields.get(family, "game_family")


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


def _annotate_builder_exception(
    exc: Exception,
    *,
    parsed_state: ParsedStateV4,
    missing_field: str,
    required_fields: str,
    reconstruction_attempted: bool,
) -> Exception:
    setattr(exc, "abort_site", "state_builder.build")
    setattr(exc, "missing_field", missing_field)
    setattr(exc, "required_fields", required_fields)
    setattr(exc, "current_visible_fields", _current_visible_fields(parsed_state))
    setattr(exc, "previous_state_available", parsed_state.previous_observation is not None)
    setattr(exc, "reconstruction_attempted", reconstruction_attempted)
    return exc


def _fallback_missing_field(exc: Exception, default: str) -> str:
    explicit = str(getattr(exc, "missing_field", "") or "").strip()
    if explicit:
        return explicit
    message = str(exc or "").strip()
    lowered = message.lower()
    if lowered.startswith("fs01 switch state unavailable"):
        return "fs01_switch_state"
    if lowered.startswith("fs01 door state unavailable"):
        return "fs01_switch_state"
    if lowered.startswith("fs02 switch state unavailable"):
        return "fs02_switch_state"
    if lowered.startswith("fs03 switch state unavailable"):
        return "fs03_switch_state"
    if ": " in message:
        return message.split(": ", 1)[0].split()[-1]
    if " requires " in message:
        return message.split(" requires ", 1)[1].split()[0]
    return default


def _raise_builder_error(
    message: str,
    *,
    parsed_state: ParsedStateV4,
    missing_field: str,
    required_fields: str,
    reconstruction_attempted: bool,
) -> None:
    raise _annotate_builder_exception(
        ValueError(message),
        parsed_state=parsed_state,
        missing_field=missing_field,
        required_fields=required_fields,
        reconstruction_attempted=reconstruction_attempted,
    )


class MovementStateBuilderV4:
    def build(
        self,
        parsed_state: ParsedStateV4,
        *,
        family: str | None = None,
        carry_state: MovementTypedStateV4 | None = None,
    ) -> MovementTypedStateV4:
        chosen_family = family or parsed_state.current_observation.game_id.split("-", 1)[0]
        required_fields = _required_fields_for_family(chosen_family)
        reconstruction_attempted = chosen_family in {"fs01", "fs02", "fs03", "pb01", "pb02", "pb03"} and carry_state is not None
        try:
            if chosen_family == "ul01":
                state = build_ul01_movement_state(parsed_state)
            elif chosen_family == "fs01":
                state = build_fs01_movement_state(parsed_state, carry_state=carry_state)
            elif chosen_family == "fs02":
                state = build_fs02_movement_state(parsed_state, carry_state=carry_state)
            elif chosen_family == "fs03":
                state = build_fs03_movement_state(parsed_state, carry_state=carry_state)
            elif chosen_family == "tp01":
                state = build_tp01_movement_state(parsed_state)
            elif chosen_family == "ic01":
                state = build_ic01_movement_state(parsed_state)
            elif chosen_family == "va01":
                state = build_va01_movement_state(parsed_state)
            elif chosen_family == "pb01":
                state = build_pb01_movement_state(parsed_state, carry_state=carry_state)
            elif chosen_family == "pb02":
                state = build_pb02_movement_state(parsed_state, carry_state=carry_state)
            elif chosen_family == "pb03":
                state = build_pb03_movement_state(parsed_state, carry_state=carry_state)
            else:
                _raise_builder_error(
                    f"unsupported movement family: {chosen_family}",
                    parsed_state=parsed_state,
                    missing_field="game_family",
                    required_fields=required_fields,
                    reconstruction_attempted=reconstruction_attempted,
                )
        except Exception as exc:
            raise _annotate_builder_exception(
                exc,
                parsed_state=parsed_state,
                missing_field=_fallback_missing_field(exc, "game_family"),
                required_fields=str(getattr(exc, "required_fields", "") or required_fields),
                reconstruction_attempted=bool(getattr(exc, "reconstruction_attempted", reconstruction_attempted)),
            )
        if state.common.game_family != chosen_family:
            _raise_builder_error(
                "movement state builder produced mismatched family state",
                parsed_state=parsed_state,
                missing_field="game_family",
                required_fields=required_fields,
                reconstruction_attempted=reconstruction_attempted,
            )
        if not state.common.current_legal_actions:
            _raise_builder_error(
                "movement state builder requires at least one legal action",
                parsed_state=parsed_state,
                missing_field="current_legal_actions",
                required_fields=required_fields,
                reconstruction_attempted=reconstruction_attempted,
            )
        if state.layout_evidence_source is None:
            _raise_builder_error(
                "movement state builder requires layout_evidence_source",
                parsed_state=parsed_state,
                missing_field="layout_evidence_source",
                required_fields=required_fields,
                reconstruction_attempted=reconstruction_attempted,
            )
        if state.layout_evidence_source not in {"direct_observation", "environment_metadata", "local_memory"}:
            _raise_builder_error(
                "movement state builder received unsupported layout_evidence_source",
                parsed_state=parsed_state,
                missing_field="layout_evidence_source",
                required_fields=required_fields,
                reconstruction_attempted=reconstruction_attempted,
            )
        if chosen_family in {"fs01", "fs02", "fs03"}:
            if not state.family.switch_positions:
                _raise_builder_error(f"{chosen_family} movement state requires switch_positions", parsed_state=parsed_state, missing_field="switch_positions", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if state.family.door_open is None:
                _raise_builder_error(f"{chosen_family} movement state requires door_open", parsed_state=parsed_state, missing_field="door_open", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if state.family.door_state_bits is None:
                _raise_builder_error(f"{chosen_family} movement state requires door_state_bits", parsed_state=parsed_state, missing_field="door_state_bits", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if state.family.switch_logic_mode is None:
                _raise_builder_error(f"{chosen_family} movement state requires switch_logic_mode", parsed_state=parsed_state, missing_field="switch_logic_mode", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if chosen_family in {"fs01", "fs03"} and state.family.activated_switch_bits is None:
                _raise_builder_error(f"{chosen_family} movement state requires activated_switch_bits", parsed_state=parsed_state, missing_field="activated_switch_bits", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if chosen_family == "fs02" and state.family.occupied_switch_bits is None:
                _raise_builder_error("fs02 movement state requires occupied_switch_bits", parsed_state=parsed_state, missing_field="fs02_switch_state", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if chosen_family in {"fs02", "fs03"} and state.family.switch_group_threshold is None:
                _raise_builder_error(f"{chosen_family} movement state requires switch_group_threshold", parsed_state=parsed_state, missing_field=f"{chosen_family}_switch_state", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if state.family.door_open and state.family.door_positions:
                _raise_builder_error(f"{chosen_family} movement state cannot keep closed door positions when the door is open", parsed_state=parsed_state, missing_field="door_positions", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not state.family.door_open and not state.family.door_positions:
                _raise_builder_error(f"{chosen_family} movement state requires closed door positions while the door is shut", parsed_state=parsed_state, missing_field="door_positions", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
        if chosen_family == "tp01":
            if not state.family.teleporter_endpoint_positions:
                _raise_builder_error("tp01 movement state requires teleporter_endpoint_positions", parsed_state=parsed_state, missing_field="teleporter_endpoint_positions", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not state.family.teleporter_pairs:
                _raise_builder_error("tp01 movement state requires teleporter_pairs", parsed_state=parsed_state, missing_field="teleporter_pairs", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not state.family.teleporter_pair_map:
                _raise_builder_error("tp01 movement state requires teleporter_pair_map", parsed_state=parsed_state, missing_field="teleporter_pair_map", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not state.common.target_cells:
                _raise_builder_error("tp01 movement state requires target_cells", parsed_state=parsed_state, missing_field="target_cells", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
        if chosen_family == "ic01":
            if state.family.slide_mode != "ice":
                _raise_builder_error("ic01 movement state requires explicit ice slide_mode", parsed_state=parsed_state, missing_field="slide_mode", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not state.family.ice_cell_positions:
                _raise_builder_error("ic01 movement state requires explicit ice_cell_positions", parsed_state=parsed_state, missing_field="ice_cell_positions", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if tuple(sorted(state.family.ice_cell_positions)) != tuple(sorted(state.common.traversable_cells)):
                _raise_builder_error("ic01 movement state requires ice_cell_positions to match traversable slide cells", parsed_state=parsed_state, missing_field="ice_cell_positions", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
        if chosen_family == "va01":
            if not state.family.coverage_eligible_cells:
                _raise_builder_error("va01 movement state requires coverage_eligible_cells", parsed_state=parsed_state, missing_field="coverage_eligible_cells", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not state.family.coverage_mask:
                _raise_builder_error("va01 movement state requires coverage_mask", parsed_state=parsed_state, missing_field="coverage_mask", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if tuple(sorted(state.family.coverage_eligible_cells)) != tuple(sorted(state.common.traversable_cells)):
                _raise_builder_error("va01 movement state requires coverage_eligible_cells to match traversable cells", parsed_state=parsed_state, missing_field="coverage_eligible_cells", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not set(state.family.coverage_mask).issubset(set(state.family.coverage_eligible_cells)):
                _raise_builder_error("va01 movement state requires coverage_mask to remain inside coverage_eligible_cells", parsed_state=parsed_state, missing_field="coverage_mask", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
        if chosen_family in {"pb01", "pb02", "pb03"}:
            required_blocks = {"pb01": 1, "pb02": 2, "pb03": 1}[chosen_family]
            if len(state.family.pushable_block_positions) != required_blocks:
                _raise_builder_error(f"{chosen_family} movement state requires exactly {required_blocks} pushable_block_positions entries", parsed_state=parsed_state, missing_field="pushable_block_positions", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not state.common.target_cells:
                _raise_builder_error(f"{chosen_family} movement state requires target_cells", parsed_state=parsed_state, missing_field="target_cells", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if not state.family.push_target_cells:
                _raise_builder_error(f"{chosen_family} movement state requires push_target_cells", parsed_state=parsed_state, missing_field="push_target_cells", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if tuple(sorted(state.family.push_target_cells)) != tuple(sorted(state.common.target_cells)):
                _raise_builder_error(f"{chosen_family} movement state requires push_target_cells to match target_cells", parsed_state=parsed_state, missing_field="push_target_cells", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if state.family.push_variant is None:
                _raise_builder_error(f"{chosen_family} movement state requires push_variant", parsed_state=parsed_state, missing_field="push_variant", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if state.family.step_limit is None:
                _raise_builder_error(f"{chosen_family} movement state requires step_limit", parsed_state=parsed_state, missing_field="step_limit", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
            if chosen_family == "pb03" and not state.family.push_decoy_lose_cells:
                _raise_builder_error("pb03 movement state requires push_decoy_lose_cells", parsed_state=parsed_state, missing_field="push_decoy_lose_cells", required_fields=required_fields, reconstruction_attempted=reconstruction_attempted)
        return state
