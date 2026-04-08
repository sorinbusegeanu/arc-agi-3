from __future__ import annotations

from dataclasses import replace

from v4.agentContract.types import V4Action
from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4, legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_pt01_transition_payload, detect_pt01_phase
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
        self._pt01_cached_level_index: int | None = None
        self._pt01_cached_plan: list = []
        self._pt01_last_state = None
        self._pt01_transition_replans: int = 0
        self._pt01_cache_invalidated: bool = False
        self._wm01_last_state_key: str | None = None
        self._wm01_last_payload: tuple[int, int] | None = None
        self._wm01_repeat_streak: int = 0
        self._wm01_recent_payloads: list[tuple[int, int]] = []

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        family = parsed_state.current_observation.game_id.split("-", 1)[0]
        if family == "pt01":
            phase_info = detect_pt01_phase(parsed_state, cached_level_index=self._pt01_cached_level_index)
            current_level_index = int(phase_info["current_level_index"])
            cache_invalidated = self._invalidate_pt01_cache_if_needed(parsed_state, current_level_index=current_level_index)
            phase = str(phase_info["phase"])
            if phase == "pt01_transition_frame":
                self._pt01_transition_replans += 1
                if self._pt01_transition_replans > 3:
                    raise self._pt01_transition_error(
                        parsed_state,
                        missing_field="pt01_transition_stall",
                        message="pt01 transition stall: stable new-level board not reached after 3 replans",
                        current_level_index=current_level_index,
                        cache_invalidated=cache_invalidated,
                    )
                payload = build_pt01_transition_payload(parsed_state)
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(6, parsed_state=parsed_state, payload=payload),
                    annotations=self._pt01_annotations(
                        phase=phase,
                        search_status="pt01_transition_frame",
                        current_level_index=current_level_index,
                        cache_invalidated=cache_invalidated,
                    ),
                )
            self._pt01_transition_replans = 0
            cache_key = (parsed_state.current_observation.game_id, current_level_index)
            if self._pt01_cached_key != cache_key:
                self._pt01_cached_key = cache_key
                self._pt01_cached_plan = []
                self._pt01_last_state = None
            if self._pt01_cached_plan and self._pt01_last_state is not None:
                action = self._pt01_cached_plan.pop(0)
                self._pt01_last_state = self._apply_pt01_click_to_state(
                    self._pt01_last_state,
                    payload=action.payload,
                    parsed_state=parsed_state,
                )
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(6, parsed_state=parsed_state, payload=action.payload),
                    annotations=self._pt01_annotations(
                        phase=phase,
                        search_status="cached_plan",
                        current_level_index=current_level_index,
                        cache_invalidated=cache_invalidated,
                    ),
                )
        try:
            typed_state = self.state_builder.build(parsed_state, family=family)
        except Exception:
            if family != "pt01":
                raise
            typed_state = self._recover_pt01_state_from_cache(parsed_state)
            self._clear_recovered_builder_abort(parsed_state, typed_state)
        if family == "pt01":
            self._pt01_last_state = typed_state
        if family == "pt01":
            phase = str(getattr(typed_state.family, "pt01_phase", "") or "pt01_active_board")
            cache_invalidated = bool(self._pt01_cache_invalidated or False)
            direct_plan = self._plan_pt01_rotations(typed_state)
            if direct_plan:
                self._pt01_cached_plan = list(direct_plan[1:])
                action = direct_plan[0]
                return PolicyDecisionV4(
                    primitive_action=legal_action_from_id(6, parsed_state=parsed_state, payload=action.payload),
                    annotations=self._pt01_annotations(
                        phase=phase,
                        search_status="pt01_direct_plan",
                        current_level_index=int(typed_state.common.level_index),
                        cache_invalidated=cache_invalidated,
                        plan_length=len(direct_plan),
                    ),
                )
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
                    annotations=self._pt01_annotations(
                        phase=phase,
                        search_status=outcome.status,
                        current_level_index=int(typed_state.common.level_index),
                        cache_invalidated=cache_invalidated,
                        plan_length=len(outcome.plan),
                    ),
                )
        action = self._select_family_action(parsed_state, typed_state, family)
        return PolicyDecisionV4(
            primitive_action=action,
            annotations={"policy": "click_solver", "family": family, "search_status": "greedy"},
        )

    def _invalidate_pt01_cache_if_needed(self, parsed_state: ParsedStateV4, *, current_level_index: int) -> bool:
        cache_key = (parsed_state.current_observation.game_id, current_level_index)
        changed = self._pt01_cached_level_index is not None and self._pt01_cached_level_index != current_level_index
        if changed or self._pt01_cached_key != cache_key:
            self._pt01_cached_key = cache_key
            self._pt01_cached_level_index = current_level_index
            self._pt01_cached_plan = []
            self._pt01_last_state = None
            self._pt01_cache_invalidated = True
            return True
        if self._pt01_cached_level_index is None:
            self._pt01_cached_level_index = current_level_index
        self._pt01_cache_invalidated = False
        return False

    def _pt01_transition_error(self, parsed_state: ParsedStateV4, *, missing_field: str, message: str, current_level_index: int, cache_invalidated: bool) -> ValueError:
        exc = ValueError(message)
        setattr(exc, "abort_site", "policy.decide")
        setattr(exc, "missing_field", missing_field)
        setattr(exc, "required_fields", "legal_action_ids,clickable_cells,rotation_tiles")
        setattr(exc, "current_visible_fields", "current_observation,previous_observation,environment_metadata,available_actions,terminal_signal,memory_reference,derived_control")
        setattr(exc, "previous_state_available", parsed_state.previous_observation is not None)
        setattr(exc, "reconstruction_attempted", False)
        setattr(exc, "pt01_current_level_index", current_level_index)
        setattr(exc, "pt01_cached_level_index", self._pt01_cached_level_index if self._pt01_cached_level_index is not None else "")
        setattr(exc, "pt01_transition_frame_detected", True)
        setattr(exc, "pt01_cache_invalidated", cache_invalidated)
        return exc

    def _pt01_annotations(
        self,
        *,
        phase: str,
        search_status: str,
        current_level_index: int,
        cache_invalidated: bool,
        plan_length: int | None = None,
    ) -> dict[str, object]:
        annotations: dict[str, object] = {
            "policy": "click_solver",
            "family": "pt01",
            "search_status": search_status,
            "pt01_phase": phase,
            "pt01_current_level_index": current_level_index,
            "pt01_cached_level_index": self._pt01_cached_level_index if self._pt01_cached_level_index is not None else "",
            "pt01_transition_frame_detected": phase == "pt01_transition_frame",
            "pt01_cache_invalidated": cache_invalidated,
        }
        if plan_length is not None:
            annotations["plan_length"] = int(plan_length)
        return annotations

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
            state_key = typed_state.to_key()
            if state_key == self._wm01_last_state_key:
                self._wm01_repeat_streak += 1
            else:
                self._wm01_repeat_streak = 0
                self._wm01_last_state_key = state_key
            moles = list(typed_state.family.active_mole_cells)
            if self._wm01_repeat_streak > 0:
                recent_targets = set(self._wm01_recent_payloads)
                ranked_moles = sorted(
                    moles,
                    key=lambda mole: (
                        1 if (self._wm01_payload_for_mole(typed_state, mole)["x"], self._wm01_payload_for_mole(typed_state, mole)["y"]) in recent_targets else 0,
                        abs(mole[0] - typed_state.common.static_bounds[0] // 2) + abs(mole[1] - typed_state.common.static_bounds[1] // 2),
                        mole,
                    ),
                )
                if len(ranked_moles) > 1:
                    offset = self._wm01_repeat_streak % len(ranked_moles)
                    ranked_moles = ranked_moles[offset:] + ranked_moles[:offset]
                if ranked_moles:
                    payload = self._wm01_payload_for_mole(typed_state, ranked_moles[0])
                else:
                    clickable = list(typed_state.common.clickable_cells)
                    if self._wm01_last_payload in clickable and len(clickable) > 1:
                        clickable = [cell for cell in clickable if cell != self._wm01_last_payload] + [self._wm01_last_payload]
                    cell = clickable[0]
                    payload = {"x": cell[0], "y": cell[1], "game_id": typed_state.common.game_id}
            else:
                mole = moles[0]
                payload = self._wm01_payload_for_mole(typed_state, mole)
            if self._wm01_repeat_streak > 0 and len(typed_state.common.clickable_cells) > 1:
                clickable = [cell for cell in typed_state.common.clickable_cells if (cell[0], cell[1]) not in self._wm01_recent_payloads]
                if clickable:
                    cell = clickable[0]
                    payload = {"x": cell[0], "y": cell[1], "game_id": typed_state.common.game_id}
            self._wm01_last_payload = (payload["x"], payload["y"])
            self._wm01_recent_payloads.append(self._wm01_last_payload)
            self._wm01_recent_payloads = self._wm01_recent_payloads[-6:]
            return legal_action_from_id(6, parsed_state=parsed_state, payload=payload)
        if family == "wm01" and typed_state.common.clickable_cells:
            clickable = [cell for cell in typed_state.common.clickable_cells if cell not in self._wm01_recent_payloads]
            if not clickable:
                clickable = list(typed_state.common.clickable_cells)
            cell = clickable[0]
            self._wm01_last_payload = (cell[0], cell[1])
            self._wm01_recent_payloads.append(self._wm01_last_payload)
            self._wm01_recent_payloads = self._wm01_recent_payloads[-6:]
            return legal_action_from_id(6, parsed_state=parsed_state, payload={"x": cell[0], "y": cell[1], "game_id": typed_state.common.game_id})
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

    def _recover_pt01_state_from_cache(self, parsed_state: ParsedStateV4):
        cached = self._pt01_last_state
        if cached is None or cached.common.game_family != "pt01":
            raise ValueError("pt01 cached state unavailable")
        last_action = parsed_state.current_observation.action_input
        if not isinstance(last_action, dict) or int(last_action.get("id", -1)) != 6:
            raise ValueError("pt01 cached state recovery requires previous click action")
        data = last_action.get("data") or {}
        if not isinstance(data, dict) or "x" not in data or "y" not in data:
            raise ValueError("pt01 cached state recovery requires click payload")
        return self._apply_pt01_click_to_state(cached, payload=data, parsed_state=parsed_state)

    @staticmethod
    def _apply_pt01_click_to_state(cached, *, payload, parsed_state: ParsedStateV4):
        if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
            raise ValueError("pt01 cached state recovery requires click payload")
        click_x = int(payload["x"])
        click_y = int(payload["y"])
        updated_tiles = []
        matched = False
        for position, sprite_type, rotation in cached.family.rotation_tiles:
            tile_x, tile_y = position
            if tile_x <= click_x < tile_x + 3 and tile_y <= click_y < tile_y + 3:
                updated_tiles.append((position, sprite_type, (int(rotation) + 90) % 360))
                matched = True
            else:
                updated_tiles.append((position, sprite_type, int(rotation)))
        if not matched:
            raise ValueError("pt01 cached state recovery requires clicked tile match")
        common = replace(
            cached.common,
            game_id=parsed_state.current_observation.game_id,
            level_index=ClickSolverPolicyV4._pt01_level_index(parsed_state),
            legal_action_ids=tuple(int(action_id) for action_id in parsed_state.available_actions),
            terminal_status=parsed_state.terminal_signal.status,
            step_depth=int(parsed_state.step_index),
        )
        family = replace(cached.family, rotation_tiles=tuple(updated_tiles))
        return replace(cached, common=common, family=family)

    @staticmethod
    def _pt01_payload_for_tile(typed_state, tile_position: tuple[int, int]) -> tuple[int, int]:
        width, height = typed_state.common.static_bounds
        scale = min(int(64 / width), int(64 / height))
        x_pad = int((64 - (width * scale)) / 2)
        y_pad = int((64 - (height * scale)) / 2)
        gx = int(tile_position[0]) + 1
        gy = int(tile_position[1]) + 1
        return gx * scale + scale // 2 + x_pad, gy * scale + scale // 2 + y_pad

    def _plan_pt01_rotations(self, typed_state) -> tuple:
        target_by_type = dict(typed_state.family.target_rotations_by_type)
        actions = []
        for position, sprite_type, rotation in sorted(typed_state.family.rotation_tiles, key=lambda item: (item[0][1], item[0][0], item[1])):
            target = target_by_type.get(sprite_type)
            if target is None:
                continue
            click_count = ((int(target) - int(rotation)) % 360) // 90
            if click_count <= 0:
                continue
            payload = self._pt01_payload_for_tile(typed_state, position)
            for _ in range(click_count):
                actions.append(V4Action(action_id=6, action_name="ACTION6", payload={"x": payload[0], "y": payload[1], "game_id": typed_state.common.game_id}))
        return tuple(actions)

    @staticmethod
    def _wm01_payload_for_mole(typed_state, mole_position: tuple[int, int]) -> dict[str, object]:
        width, height = typed_state.common.static_bounds
        scale = min(int(64 / width), int(64 / height))
        x_pad = int((64 - (width * scale)) / 2)
        y_pad = int((64 - (height * scale)) / 2)
        gx = int(mole_position[0]) + 2
        gy = int(mole_position[1]) + 2
        return {
            "x": gx * scale + scale // 2 + x_pad,
            "y": gy * scale + scale // 2 + y_pad,
            "game_id": typed_state.common.game_id,
        }

    def _clear_recovered_builder_abort(self, parsed_state: ParsedStateV4, typed_state) -> None:
        diagnostics = getattr(self.state_builder, "_diagnostics", None)
        if diagnostics is None:
            return
        if int(getattr(diagnostics, "abort_step", -1)) != int(parsed_state.step_index):
            return
        if not str(getattr(diagnostics, "abort_site", "")).startswith("state_builder."):
            return
        diagnostics.abort_step = None
        diagnostics.abort_site = None
        diagnostics.abort_message = None
        diagnostics.missing_field = None
        diagnostics.required_fields = ""
        diagnostics.current_visible_fields = ""
        diagnostics.previous_state_available = parsed_state.previous_observation is not None
        diagnostics.reconstruction_attempted = True
        diagnostics.typed_state_summary = typed_state.to_dict()
        trace = diagnostics.trace_for_step(int(parsed_state.step_index))
        trace.typed_state_built = True
        if diagnostics.first_divergence_type == "typed_state_build_failed":
            diagnostics.first_divergence_type = None

    @staticmethod
    def _pt01_level_index(parsed_state: ParsedStateV4) -> int:
        level_index = int(parsed_state.current_observation.levels_completed)
        authoritative_state = getattr(parsed_state, "authoritative_state", None)
        authoritative_level = getattr(authoritative_state, "levels_completed", None)
        if isinstance(authoritative_level, int) and authoritative_level >= level_index:
            return authoritative_level
        if isinstance(authoritative_state, dict):
            raw_level = authoritative_state.get("levels_completed")
            if isinstance(raw_level, int) and raw_level >= level_index:
                return raw_level
        return level_index

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
