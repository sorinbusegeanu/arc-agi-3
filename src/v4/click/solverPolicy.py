from __future__ import annotations

from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .heuristics import pt01_remaining_rotation_heuristic
from .search import ClickSearchV4
from .stateBuilder import ClickStateBuilderV4
from .transitionModel import ClickTransitionModelV4


class ClickSolverPolicyV4(PolicyBaseV4):
    def __init__(
        self,
        *,
        state_builder: ClickStateBuilderV4 | None = None,
        transition_model: ClickTransitionModelV4 | None = None,
        search: ClickSearchV4 | None = None,
        max_plan_prefix: int = 3,
        search_bound: int | None = 10,
    ) -> None:
        self.state_builder = state_builder if state_builder is not None else ClickStateBuilderV4()
        self.transition_model = transition_model if transition_model is not None else ClickTransitionModelV4()
        self.search = search if search is not None else ClickSearchV4(self.transition_model)
        self.max_plan_prefix = int(max_plan_prefix)
        self.search_bound = search_bound
        self._pt01_cached_key: tuple[str, int] | None = None
        self._pt01_cached_plan: list = []

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        family = parsed_state.current_observation.game_id.split("-", 1)[0]
        if family == "pt01":
            cache_key = (parsed_state.current_observation.game_id, parsed_state.current_observation.levels_completed)
            if self._pt01_cached_key != cache_key:
                self._pt01_cached_key = cache_key
                self._pt01_cached_plan = []
            if self._pt01_cached_plan:
                action = self._pt01_cached_plan.pop(0)
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(6, parsed_state=parsed_state, payload=action.payload),
                    annotations={"policy": "click_solver", "family": family, "search_status": "cached_plan"},
                )
        typed_state = self.state_builder.build(parsed_state, family=family)
        if family == "pt01":
            outcome = self.search.search(
                typed_state,
                goal_predicate=lambda state: state.common.terminal_status == "success",
                algorithm="astar",
                max_depth=self.search_bound,
                heuristic=pt01_remaining_rotation_heuristic,
            )
            if outcome.status == "found" and outcome.plan:
                self._pt01_cached_plan = list(outcome.plan[1:])
                action = outcome.plan[0]
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(6, parsed_state=parsed_state, payload=action.payload),
                    annotations={"policy": "click_solver", "family": family, "search_status": outcome.status, "plan_length": len(outcome.plan)},
                )
        action = self._select_family_action(parsed_state, typed_state, family)
        return PolicyDecisionV4(
            primitive_action=action,
            annotations={"policy": "click_solver", "family": family, "search_status": "greedy"},
        )

    def _select_family_action(self, parsed_state: ParsedStateV4, typed_state, family: str):
        if family == "sy01":
            outcome = self.search.search(
                typed_state,
                goal_predicate=lambda state: state.common.terminal_status == "success",
                max_depth=self.search_bound,
            )
            if outcome.status == "found" and outcome.plan:
                action = outcome.plan[0]
                return legal_action_from_id(6, parsed_state=parsed_state, payload=action.payload)
            remaining = [cell for cell in typed_state.family.mirror_target_cells if cell not in typed_state.family.placed_mirror_cells]
            target = remaining[0] if remaining else typed_state.common.clickable_cells[0]
            return legal_action_from_id(6, parsed_state=parsed_state, payload={"x": target[0], "y": target[1], "game_id": typed_state.common.game_id})
        if family == "ff01":
            for index, region in enumerate(typed_state.family.fill_regions):
                if index not in typed_state.family.filled_region_indexes:
                    cell = region[len(region) // 2]
                    return legal_action_from_id(6, parsed_state=parsed_state, payload={"x": cell[0], "y": cell[1], "game_id": typed_state.common.game_id})
        if family == "sq01":
            progress = int(typed_state.family.sequence_progress or 0)
            if progress < len(typed_state.family.sequence_order):
                target_color = typed_state.family.sequence_order[progress]
                for color_name, cell in typed_state.family.clickable_color_cells:
                    if color_name == target_color:
                        return legal_action_from_id(6, parsed_state=parsed_state, payload={"x": cell[0], "y": cell[1], "game_id": typed_state.common.game_id})
        if family == "wm01" and typed_state.family.active_mole_cells:
            mole = typed_state.family.active_mole_cells[0]
            payload = {"x": min(63, mole[0] * 2), "y": min(63, mole[1] * 2), "game_id": typed_state.common.game_id}
            return legal_action_from_id(6, parsed_state=parsed_state, payload=payload)
        if family == "mm01":
            matched = set(typed_state.family.matched_slots)
            unmatched_revealed = [slot for slot in typed_state.family.revealed_slots if slot[0] not in matched]
            if unmatched_revealed:
                target_color = unmatched_revealed[-1][1]
                for slot_index, color in typed_state.family.hidden_slots:
                    if color == target_color:
                        cell = self._mm01_payload_for_slot(typed_state, slot_index)
                        return legal_action_from_id(6, parsed_state=parsed_state, payload={"x": cell[0], "y": cell[1], "game_id": typed_state.common.game_id})
            if typed_state.family.hidden_slots:
                slot_index, _ = typed_state.family.hidden_slots[0]
                cell = self._mm01_payload_for_slot(typed_state, slot_index)
                return legal_action_from_id(6, parsed_state=parsed_state, payload={"x": cell[0], "y": cell[1], "game_id": typed_state.common.game_id})
        cell = typed_state.common.clickable_cells[0]
        return legal_action_from_id(6, parsed_state=parsed_state, payload={"x": cell[0], "y": cell[1], "game_id": typed_state.common.game_id})

    @staticmethod
    def _mm01_payload_for_slot(typed_state, slot_index: int) -> tuple[int, int]:
        if typed_state.family.slot_geometry is None:
            raise ValueError("mm01 typed state missing slot geometry")
        rows, cols, tile_size, offset_x = typed_state.family.slot_geometry
        row = slot_index // cols
        col = slot_index % cols
        offset_y = int((64 - (rows * tile_size)) // 2)
        gx = offset_x + col * tile_size + tile_size // 2
        gy = offset_y + row * tile_size + tile_size // 2
        return gx, gy
