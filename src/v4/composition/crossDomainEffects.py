from __future__ import annotations

from v4.hybrid_construction.stateBuilder import HybridConstructionStateBuilderV4
from v4.state.parsedState import ParsedStateV4


class CrossDomainEffectsV4:
    def derive(self, parsed_state: ParsedStateV4) -> tuple[str, ...]:
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        effects: list[str] = []

        def _append(code: str) -> None:
            if code not in effects:
                effects.append(code)

        if game_id == "tb01":
            try:
                typed_state = HybridConstructionStateBuilderV4().build(parsed_state)
            except Exception:
                return ()
            family = typed_state.family
            bridge_budget_remaining = family.bridge_budget_remaining
            step_limit_remaining = family.step_limit_remaining
            bridge_built_cells = tuple(family.bridge_built_cells)
            water_cells = tuple(family.water_cells)
            land_cells = tuple(family.land_cells)
            goal_cell = family.goal_cell
            if bridge_budget_remaining is not None:
                _append("construction_budget_active")
            if step_limit_remaining is not None:
                _append("construction_under_temporal_constraints")
            traversable = set(land_cells) | set(bridge_built_cells)
            if goal_cell is not None and goal_cell not in traversable:
                _append("construction_path_not_yet_complete")
            if any(action_id in {1, 2, 3, 4} for action_id in parsed_state.available_actions) and 6 in set(parsed_state.available_actions):
                _append("movement_and_construction_actions_coexist")
            return tuple(effects)

        if parsed_state.belief_reference is not None and parsed_state.hypothesis_reference is not None and parsed_state.hypothesis_reference.hypothesis_count > 0:
            _append("belief_and_hypothesis_active")
        if parsed_state.temporal_reference is not None and game_id == "sv01":
            _append("temporal_constraints_active")
        return tuple(effects)
