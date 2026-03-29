from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import (
    build_fs01_movement_state,
    build_ic01_movement_state,
    build_pb01_movement_state,
    build_tp01_movement_state,
    build_ul01_movement_state,
    build_va01_movement_state,
)
from .typedState import MovementTypedStateV4


class MovementStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str | None = None) -> MovementTypedStateV4:
        chosen_family = family or parsed_state.current_observation.game_id.split("-", 1)[0]
        if chosen_family == "ul01":
            state = build_ul01_movement_state(parsed_state)
        elif chosen_family == "fs01":
            state = build_fs01_movement_state(parsed_state)
        elif chosen_family == "tp01":
            state = build_tp01_movement_state(parsed_state)
        elif chosen_family == "ic01":
            state = build_ic01_movement_state(parsed_state)
        elif chosen_family == "va01":
            state = build_va01_movement_state(parsed_state)
        elif chosen_family == "pb01":
            state = build_pb01_movement_state(parsed_state)
        else:
            raise ValueError(f"unsupported movement family: {chosen_family}")
        if state.common.game_family != chosen_family:
            raise ValueError("movement state builder produced mismatched family state")
        if not state.common.current_legal_actions:
            raise ValueError("movement state builder requires at least one legal action")
        if state.layout_evidence_source is None:
            raise ValueError("movement state builder requires layout_evidence_source")
        if state.layout_evidence_source not in {"direct_observation", "environment_metadata", "local_memory"}:
            raise ValueError("movement state builder received unsupported layout_evidence_source")
        if chosen_family == "fs01":
            if not state.family.switch_positions:
                raise ValueError("fs01 movement state requires switch_positions")
            if state.family.activated_switch_bits is None:
                raise ValueError("fs01 movement state requires activated_switch_bits")
            if state.family.door_open is None:
                raise ValueError("fs01 movement state requires door_open")
            if state.family.door_state_bits is None:
                raise ValueError("fs01 movement state requires door_state_bits")
            if state.family.door_open and state.family.door_positions:
                raise ValueError("fs01 movement state cannot keep closed door positions when the door is open")
            if not state.family.door_open and not state.family.door_positions:
                raise ValueError("fs01 movement state requires closed door positions while the door is shut")
        if chosen_family == "tp01":
            if not state.family.teleporter_endpoint_positions:
                raise ValueError("tp01 movement state requires teleporter_endpoint_positions")
            if not state.family.teleporter_pairs:
                raise ValueError("tp01 movement state requires teleporter_pairs")
            if not state.family.teleporter_pair_map:
                raise ValueError("tp01 movement state requires teleporter_pair_map")
            if not state.common.target_cells:
                raise ValueError("tp01 movement state requires target_cells")
        if chosen_family == "ic01":
            if state.family.slide_mode != "ice":
                raise ValueError("ic01 movement state requires explicit ice slide_mode")
            if not state.family.ice_cell_positions:
                raise ValueError("ic01 movement state requires explicit ice_cell_positions")
            if tuple(sorted(state.family.ice_cell_positions)) != tuple(sorted(state.common.traversable_cells)):
                raise ValueError("ic01 movement state requires ice_cell_positions to match traversable slide cells")
        if chosen_family == "va01":
            if not state.family.coverage_eligible_cells:
                raise ValueError("va01 movement state requires coverage_eligible_cells")
            if not state.family.coverage_mask:
                raise ValueError("va01 movement state requires coverage_mask")
            if tuple(sorted(state.family.coverage_eligible_cells)) != tuple(sorted(state.common.traversable_cells)):
                raise ValueError("va01 movement state requires coverage_eligible_cells to match traversable cells")
            if not set(state.family.coverage_mask).issubset(set(state.family.coverage_eligible_cells)):
                raise ValueError("va01 movement state requires coverage_mask to remain inside coverage_eligible_cells")
        if chosen_family == "pb01":
            if len(state.family.pushable_block_positions) != 1:
                raise ValueError("pb01 movement state requires exactly one pushable_block_positions entry")
            if not state.common.target_cells:
                raise ValueError("pb01 movement state requires target_cells")
            if not state.family.push_target_cells:
                raise ValueError("pb01 movement state requires push_target_cells")
            if tuple(sorted(state.family.push_target_cells)) != tuple(sorted(state.common.target_cells)):
                raise ValueError("pb01 movement state requires push_target_cells to match target_cells")
            if state.family.step_limit is None:
                raise ValueError("pb01 movement state requires step_limit")
        return state
