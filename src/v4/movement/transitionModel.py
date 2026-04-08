from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .typedState import GridPos, MovementCommonFieldsV4, MovementFamilyFieldsV4, MovementTypedStateV4


_DIRECTIONS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


@dataclass(frozen=True)
class MovementTransitionAnnotationV4:
    action_id: int
    moved: bool
    blocked: bool
    event: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _in_bounds(pos: GridPos, bounds: GridPos) -> bool:
    return 0 <= pos[0] < bounds[0] and 0 <= pos[1] < bounds[1]


def _add(pos: GridPos, delta: GridPos) -> GridPos:
    return pos[0] + delta[0], pos[1] + delta[1]


def _terminal_for(state: MovementTypedStateV4) -> str:
    family = state.common.game_family
    if family == "ul01":
        if not state.family.key_positions and not state.family.door_positions:
            return "success"
        return "non_terminal"
    if family in {"fs01", "fs02", "fs03"}:
        if state.family.door_open and state.common.avatar_position in state.common.target_cells:
            return "success"
        return "non_terminal"
    if family in {"tp01", "ic01"}:
        return "success" if state.common.avatar_position in state.common.target_cells else "non_terminal"
    if family == "va01":
        return "success" if set(state.family.coverage_mask) >= set(state.family.coverage_eligible_cells) else "non_terminal"
    if family in {"pb01", "pb02", "pb03"}:
        target_cells = state.family.push_target_cells or state.common.target_cells
        if family == "pb02" and set(state.family.pushable_block_positions) >= set(target_cells):
            return "success"
        if family in {"pb01", "pb03"} and any(pos in target_cells for pos in state.family.pushable_block_positions):
            return "success"
        if family == "pb03" and any(pos in state.family.push_decoy_lose_cells for pos in state.family.pushable_block_positions):
            return "failure"
        if state.family.step_limit is not None and state.common.step_depth >= state.family.step_limit:
            return "failure"
        return "non_terminal"
    return state.common.terminal_status


class MovementTransitionModelV4:
    def apply(self, state: MovementTypedStateV4, action_id: int) -> tuple[MovementTypedStateV4, MovementTransitionAnnotationV4]:
        if action_id not in _DIRECTIONS:
            raise ValueError(f"unsupported primitive movement action: {action_id}")
        if action_id not in state.common.current_legal_actions:
            raise ValueError(f"action {action_id} is not legal for the current movement state")
        if state.common.terminal_status in {"success", "failure"}:
            return state, MovementTransitionAnnotationV4(action_id=action_id, moved=False, blocked=True, event="terminal")
        family = state.common.game_family
        if family == "ul01":
            successor, annotation = self._apply_ul01(state, action_id)
        elif family in {"fs01", "fs02", "fs03"}:
            successor, annotation = self._apply_floor_switch(state, action_id)
        elif family == "tp01":
            successor, annotation = self._apply_tp01(state, action_id)
        elif family == "ic01":
            successor, annotation = self._apply_ic01(state, action_id)
        elif family == "va01":
            successor, annotation = self._apply_va01(state, action_id)
        elif family in {"pb01", "pb02", "pb03"}:
            successor, annotation = self._apply_push_block(state, action_id)
        else:
            raise ValueError(f"unsupported movement family: {family}")
        successor = replace(successor, common=replace(successor.common, terminal_status=_terminal_for(successor)))
        return successor, annotation

    def _advance_common(self, state: MovementTypedStateV4, *, avatar_position: GridPos | None = None) -> MovementCommonFieldsV4:
        return replace(
            state.common,
            avatar_position=state.common.avatar_position if avatar_position is None else avatar_position,
            step_depth=state.common.step_depth + 1,
        )

    def _apply_ul01(self, state: MovementTypedStateV4, action_id: int) -> tuple[MovementTypedStateV4, MovementTransitionAnnotationV4]:
        delta = _DIRECTIONS[action_id]
        next_pos = _add(state.common.avatar_position, delta)
        if not _in_bounds(next_pos, state.common.static_bounds):
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "out_of_bounds")
        if next_pos in state.family.door_positions and state.family.key_inventory_bits != 1:
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "locked_door")
        key_positions = state.family.key_positions
        door_positions = state.family.door_positions
        key_inventory_bits = state.family.key_inventory_bits
        event = "move"
        if next_pos in key_positions:
            key_positions = tuple(pos for pos in key_positions if pos != next_pos)
            key_inventory_bits = 1
            event = "pickup_key"
        if next_pos in door_positions and key_inventory_bits == 1:
            door_positions = tuple(pos for pos in door_positions if pos != next_pos)
            event = "unlock_door"
        traversable = tuple(sorted(set(state.common.traversable_cells) | set(door_positions == () and state.common.target_cells or ()) | set(pos for pos in state.family.door_positions if pos not in door_positions) | {next_pos}))
        successor = MovementTypedStateV4(
            common=replace(self._advance_common(state, avatar_position=next_pos), traversable_cells=traversable, blocked_cells=door_positions),
            family=replace(state.family, key_inventory_bits=key_inventory_bits, key_positions=key_positions, door_positions=door_positions, door_open=len(door_positions) == 0),
            layout_evidence_source=state.layout_evidence_source,
        )
        return successor, MovementTransitionAnnotationV4(action_id, True, False, event)

    def _apply_floor_switch(self, state: MovementTypedStateV4, action_id: int) -> tuple[MovementTypedStateV4, MovementTransitionAnnotationV4]:
        delta = _DIRECTIONS[action_id]
        next_pos = _add(state.common.avatar_position, delta)
        if not _in_bounds(next_pos, state.common.static_bounds):
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "out_of_bounds")
        closed_doors = set(state.family.door_positions)
        static_blockers = set(pos for pos in state.common.blocked_cells if pos not in closed_doors)
        if next_pos in static_blockers:
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "wall")
        if next_pos in closed_doors:
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "closed_door")
        occupied_mask, activated_mask, door_state_bits, door_positions, event = self.apply_floor_switch_update(state, action_id, next_pos)
        door_open = bool(door_state_bits)
        blocked_cells = tuple(sorted(tuple(static_blockers) + door_positions))
        traversable = tuple((x, y) for y in range(state.common.static_bounds[1]) for x in range(state.common.static_bounds[0]) if (x, y) not in blocked_cells)
        successor = MovementTypedStateV4(
            common=replace(self._advance_common(state, avatar_position=next_pos), blocked_cells=blocked_cells, traversable_cells=traversable),
            family=replace(
                state.family,
                occupied_switch_bits=occupied_mask,
                activated_switch_bits=activated_mask,
                door_positions=door_positions,
                door_open=door_open,
                door_state_bits=door_state_bits,
            ),
            layout_evidence_source=state.layout_evidence_source,
        )
        return successor, MovementTransitionAnnotationV4(action_id, True, False, event)

    def apply_floor_switch_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[int, int | None, int, tuple[GridPos, ...], str]:
        if state.common.game_family == "fs01":
            activated_mask, door_state_bits, door_positions, event = self.apply_fs01_switch_and_door_update(state, action_id, next_pos)
            return 0, activated_mask, door_state_bits, door_positions, event
        if state.common.game_family == "fs02":
            return self.apply_fs02_switch_and_door_update(state, action_id, next_pos)
        if state.common.game_family == "fs03":
            return self.apply_fs03_switch_and_door_update(state, action_id, next_pos)
        raise ValueError(f"unsupported floor-switch family: {state.common.game_family}")

    def apply_fs01_switch_and_door_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[int, int, tuple[GridPos, ...], str]:
        del action_id
        if state.family.activated_switch_bits is None:
            raise ValueError("fs01 transition requires explicit activated_switch_bits")
        if state.family.door_state_bits is None:
            raise ValueError("fs01 transition requires explicit door_state_bits")
        if not state.family.switch_positions:
            raise ValueError("fs01 transition requires explicit switch_positions")
        activated_mask = int(state.family.activated_switch_bits)
        door_state_bits = int(state.family.door_state_bits)
        event = "move"
        for index, pos in enumerate(state.family.switch_positions):
            if pos == next_pos:
                already_active = bool(activated_mask & (1 << index))
                activated_mask |= 1 << index
                event = "move" if already_active else "activate_switch"
                break
        if door_state_bits not in {0, 1}:
            raise ValueError("fs01 transition requires explicit one-bit door_state_bits")
        if door_state_bits == 0:
            required_mask = (1 << len(state.family.switch_positions)) - 1
            if activated_mask == required_mask:
                door_state_bits = 1
        door_positions = () if door_state_bits == 1 else tuple(state.family.door_positions)
        return activated_mask, door_state_bits, door_positions, event

    def apply_fs02_switch_and_door_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[int, int, int, tuple[GridPos, ...], str]:
        del action_id
        if not state.family.switch_positions:
            raise ValueError("fs02 transition requires explicit switch_positions")
        if state.family.occupied_switch_bits is None:
            raise ValueError("fs02 transition requires explicit occupied_switch_bits")
        occupied_mask = 0
        event = "move"
        for index, pos in enumerate(state.family.switch_positions):
            if pos == next_pos:
                occupied_mask |= 1 << index
                event = "activate_switch"
                break
        door_state_bits = 1 if (state.family.door_state_bits == 1 or occupied_mask != 0) else 0
        activated_mask = 1 if door_state_bits == 1 else 0
        door_positions = () if door_state_bits == 1 else tuple(state.family.door_positions)
        return occupied_mask, activated_mask, door_state_bits, door_positions, event

    def apply_fs03_switch_and_door_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[int, int, int, tuple[GridPos, ...], str]:
        del action_id
        if not state.family.switch_positions:
            raise ValueError("fs03 transition requires explicit switch_positions")
        if state.family.activated_switch_bits is None:
            raise ValueError("fs03 transition requires explicit activated_switch_bits")
        if state.family.switch_group_threshold is None:
            raise ValueError("fs03 transition requires explicit switch_group_threshold")
        occupied_mask = 0
        activated_mask = int(state.family.activated_switch_bits)
        event = "move"
        for index, pos in enumerate(state.family.switch_positions):
            if pos == next_pos:
                occupied_mask |= 1 << index
                already_active = bool(activated_mask & (1 << index))
                activated_mask |= 1 << index
                event = "move" if already_active else "activate_switch"
                break
        required = int(state.family.switch_group_threshold)
        door_state_bits = 1 if state.family.door_state_bits == 1 or activated_mask.bit_count() >= required else 0
        door_positions = () if door_state_bits == 1 else tuple(state.family.door_positions)
        return occupied_mask, activated_mask, door_state_bits, door_positions, event

    def _apply_tp01(self, state: MovementTypedStateV4, action_id: int) -> tuple[MovementTypedStateV4, MovementTransitionAnnotationV4]:
        delta = _DIRECTIONS[action_id]
        next_pos = _add(state.common.avatar_position, delta)
        if not _in_bounds(next_pos, state.common.static_bounds):
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "out_of_bounds")
        if next_pos in state.common.blocked_cells:
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "wall")
        dest, event = self.apply_tp01_teleport_update(state, next_pos)
        successor = MovementTypedStateV4(common=self._advance_common(state, avatar_position=dest), family=state.family, layout_evidence_source=state.layout_evidence_source)
        return successor, MovementTransitionAnnotationV4(action_id, True, False, event)

    def apply_tp01_teleport_update(
        self,
        state: MovementTypedStateV4,
        tentative_position: GridPos,
    ) -> tuple[GridPos, str]:
        if not state.family.teleporter_endpoint_positions:
            raise ValueError("tp01 transition requires explicit teleporter_endpoint_positions")
        if not state.family.teleporter_pair_map:
            raise ValueError("tp01 transition requires explicit teleporter_pair_map")
        portal_map = dict(state.family.teleporter_pair_map)
        if tentative_position not in state.family.teleporter_endpoint_positions:
            return tentative_position, "move"
        if tentative_position not in portal_map:
            raise ValueError("tp01 transition requires explicit teleporter mapping for each endpoint")
        return portal_map[tentative_position], "teleport"

    def _apply_ic01(self, state: MovementTypedStateV4, action_id: int) -> tuple[MovementTypedStateV4, MovementTransitionAnnotationV4]:
        current = state.common.avatar_position
        end, blocked = self.apply_ic01_slide_update(state, action_id)
        successor = MovementTypedStateV4(common=self._advance_common(state, avatar_position=end), family=state.family, layout_evidence_source=state.layout_evidence_source)
        return successor, MovementTransitionAnnotationV4(action_id, end != current, blocked, "slide")

    def apply_ic01_slide_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
    ) -> tuple[GridPos, bool]:
        if state.family.slide_mode != "ice":
            raise ValueError("ic01 transition requires explicit ice slide_mode")
        if not state.family.ice_cell_positions:
            raise ValueError("ic01 transition requires explicit ice_cell_positions")
        delta = _DIRECTIONS[action_id]
        current = state.common.avatar_position
        traversable = set(state.family.ice_cell_positions)
        blocked = set(state.common.blocked_cells)
        end = current
        while True:
            nxt = _add(end, delta)
            if not _in_bounds(nxt, state.common.static_bounds):
                return end, end == current
            if nxt in blocked:
                return end, end == current
            if nxt not in traversable:
                raise ValueError(f"ic01 transition requires traversable slide surface at {nxt}")
            end = nxt

    def _apply_va01(self, state: MovementTypedStateV4, action_id: int) -> tuple[MovementTypedStateV4, MovementTransitionAnnotationV4]:
        delta = _DIRECTIONS[action_id]
        next_pos = _add(state.common.avatar_position, delta)
        if not _in_bounds(next_pos, state.common.static_bounds):
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "out_of_bounds")
        if next_pos in state.common.blocked_cells:
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "wall")
        coverage, event = self.apply_va01_coverage_update(state, action_id, next_pos)
        successor = MovementTypedStateV4(
            common=self._advance_common(state, avatar_position=next_pos),
            family=replace(state.family, coverage_mask=coverage),
            layout_evidence_source=state.layout_evidence_source,
        )
        return successor, MovementTransitionAnnotationV4(action_id, True, False, event)

    def apply_va01_coverage_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[tuple[GridPos, ...], str]:
        del action_id
        if not state.family.coverage_eligible_cells:
            raise ValueError("va01 transition requires explicit coverage_eligible_cells")
        eligible = set(state.family.coverage_eligible_cells)
        covered = set(state.family.coverage_mask)
        if next_pos not in eligible:
            raise ValueError(f"va01 transition requires moved cell {next_pos} to be coverage-eligible")
        event = "coverage_revisit" if next_pos in covered else "coverage_new"
        covered.add(next_pos)
        return tuple(sorted(covered)), event

    def _apply_push_block(self, state: MovementTypedStateV4, action_id: int) -> tuple[MovementTypedStateV4, MovementTransitionAnnotationV4]:
        delta = _DIRECTIONS[action_id]
        next_pos = _add(state.common.avatar_position, delta)
        if not _in_bounds(next_pos, state.common.static_bounds):
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "out_of_bounds")
        if next_pos in state.common.blocked_cells:
            return replace(state, common=self._advance_common(state)), MovementTransitionAnnotationV4(action_id, False, True, "wall")
        avatar_position, push_positions, moved, blocked, event, terminal_status = self.apply_push_update(state, action_id, next_pos)
        successor = MovementTypedStateV4(
            common=self._advance_common(state, avatar_position=avatar_position),
            family=replace(
                state.family,
                pushable_block_positions=tuple(sorted(push_positions)),
                push_target_cells=state.family.push_target_cells,
                push_solved_goal_cells=tuple(sorted(pos for pos in push_positions if pos in set(state.family.push_target_cells))),
                push_decoy_lose_cells=state.family.push_decoy_lose_cells,
                step_limit=state.family.step_limit,
            ),
            layout_evidence_source=state.layout_evidence_source,
        )
        if terminal_status is not None:
            successor = replace(successor, common=replace(successor.common, terminal_status=terminal_status))
        return successor, MovementTransitionAnnotationV4(action_id, moved, blocked, event)

    def apply_push_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[GridPos, tuple[GridPos, ...], bool, bool, str, str | None]:
        if state.common.game_family == "pb01":
            avatar_position, push_positions, moved, blocked, event = self.apply_pb01_push_update(state, action_id, next_pos)
            return avatar_position, push_positions, moved, blocked, event, None
        if state.common.game_family == "pb02":
            return self.apply_pb02_push_update(state, action_id, next_pos)
        if state.common.game_family == "pb03":
            return self.apply_pb03_push_update(state, action_id, next_pos)
        raise ValueError(f"unsupported push family: {state.common.game_family}")

    def apply_pb01_push_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[GridPos, tuple[GridPos, ...], bool, bool, str]:
        del action_id
        if len(state.family.pushable_block_positions) != 1:
            raise ValueError("pb01 transition requires exactly one pushable block position")
        if not state.family.push_target_cells:
            raise ValueError("pb01 transition requires explicit push_target_cells")
        block_pos = state.family.pushable_block_positions[0]
        if next_pos != block_pos:
            return next_pos, state.family.pushable_block_positions, True, False, "move"
        delta = (next_pos[0] - state.common.avatar_position[0], next_pos[1] - state.common.avatar_position[1])
        push_dest = _add(block_pos, delta)
        if not _in_bounds(push_dest, state.common.static_bounds):
            return state.common.avatar_position, state.family.pushable_block_positions, False, True, "push_out_of_bounds"
        if push_dest in state.common.blocked_cells or push_dest in state.family.pushable_block_positions:
            return state.common.avatar_position, state.family.pushable_block_positions, False, True, "push_blocked"
        return next_pos, (push_dest,), True, False, "push"

    def apply_pb02_push_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[GridPos, tuple[GridPos, ...], bool, bool, str, str | None]:
        del action_id
        if len(state.family.pushable_block_positions) != 2:
            raise ValueError("pb02 transition requires exactly two pushable block positions")
        block_positions = set(state.family.pushable_block_positions)
        if next_pos not in block_positions:
            return next_pos, tuple(sorted(block_positions)), True, False, "move", None
        delta = (next_pos[0] - state.common.avatar_position[0], next_pos[1] - state.common.avatar_position[1])
        push_dest = _add(next_pos, delta)
        if not _in_bounds(push_dest, state.common.static_bounds):
            return state.common.avatar_position, tuple(sorted(block_positions)), False, True, "push_out_of_bounds", None
        if push_dest in state.common.blocked_cells:
            return state.common.avatar_position, tuple(sorted(block_positions)), False, True, "push_blocked", None
        if push_dest in block_positions:
            return state.common.avatar_position, tuple(sorted(block_positions)), False, True, "push_blocked", None
        next_blocks = set(block_positions)
        next_blocks.remove(next_pos)
        next_blocks.add(push_dest)
        return next_pos, tuple(sorted(next_blocks)), True, False, "push", None

    def apply_pb03_push_update(
        self,
        state: MovementTypedStateV4,
        action_id: int,
        next_pos: GridPos,
    ) -> tuple[GridPos, tuple[GridPos, ...], bool, bool, str, str | None]:
        del action_id
        if len(state.family.pushable_block_positions) != 1:
            raise ValueError("pb03 transition requires exactly one pushable block position")
        block_pos = state.family.pushable_block_positions[0]
        if next_pos != block_pos:
            return next_pos, (block_pos,), True, False, "move", None
        delta = (next_pos[0] - state.common.avatar_position[0], next_pos[1] - state.common.avatar_position[1])
        push_dest = _add(block_pos, delta)
        if not _in_bounds(push_dest, state.common.static_bounds):
            return state.common.avatar_position, (block_pos,), False, True, "push_out_of_bounds", None
        if push_dest in state.common.blocked_cells or push_dest in state.family.pushable_block_positions:
            return state.common.avatar_position, (block_pos,), False, True, "push_blocked", None
        if push_dest in state.family.push_decoy_lose_cells:
            return state.common.avatar_position, (push_dest,), True, False, "push_decoy_loss", "failure"
        return next_pos, (push_dest,), True, False, "push", None
