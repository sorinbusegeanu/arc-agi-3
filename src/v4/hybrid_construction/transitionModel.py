from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from v4.agentContract.types import V4Action

from .typedState import GridPos, HybridConstructionTypedStateV4

_MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


@dataclass(frozen=True)
class HybridConstructionTransitionAnnotationV4:
    action_id: int
    moved: bool
    blocked: bool
    event: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _grid_payload(bounds: GridPos, gx: int, gy: int) -> tuple[int, int]:
    width, height = bounds
    scale = min(int(64 / width), int(64 / height))
    x_pad = int((64 - (width * scale)) / 2)
    y_pad = int((64 - (height * scale)) / 2)
    return gx * scale + scale // 2 + x_pad, gy * scale + scale // 2 + y_pad


def _payload_to_grid(action: V4Action, bounds: GridPos) -> GridPos:
    if action.payload is None:
        raise ValueError("tb01 click action requires payload")
    px = int(action.payload["x"])
    py = int(action.payload["y"])
    width, height = bounds
    scale = min(int(64 / width), int(64 / height))
    x_pad = int((64 - (width * scale)) / 2)
    y_pad = int((64 - (height * scale)) / 2)
    return max(0, min(width - 1, (px - x_pad) // max(scale, 1))), max(0, min(height - 1, (py - y_pad) // max(scale, 1)))


class HybridConstructionTransitionModelV4:
    def apply(self, state: HybridConstructionTypedStateV4, action: int | V4Action) -> tuple[HybridConstructionTypedStateV4, HybridConstructionTransitionAnnotationV4]:
        if isinstance(action, int):
            action_id = action
            payload_action = None
        else:
            action_id = action.action_id
            payload_action = action
        if action_id in _MOVE_DELTAS:
            next_pos = (state.common.avatar_position[0] + _MOVE_DELTAS[action_id][0], state.common.avatar_position[1] + _MOVE_DELTAS[action_id][1])
            traversable = set(state.family.land_cells) | set(state.family.bridge_built_cells)
            if next_pos not in traversable:
                successor = replace(state, common=replace(state.common, step_depth=state.common.step_depth + 1))
                return successor, HybridConstructionTransitionAnnotationV4(action_id, False, True, "blocked")
            step_remaining = None if state.family.step_limit_remaining is None else max(0, state.family.step_limit_remaining - 1)
            terminal_status = "success" if next_pos == state.family.goal_cell else "non_terminal"
            successor = HybridConstructionTypedStateV4(
                common=replace(state.common, avatar_position=next_pos, step_depth=state.common.step_depth + 1, terminal_status=terminal_status),
                family=replace(state.family, step_limit_remaining=step_remaining),
                layout_evidence_source=state.layout_evidence_source,
            )
            return successor, HybridConstructionTransitionAnnotationV4(action_id, True, False, "move")
        if action_id != 6 or payload_action is None:
            raise ValueError("tb01 transition requires primitive move id or ACTION6 payload action")
        cell = _payload_to_grid(payload_action, state.common.static_bounds)
        built = set(state.family.bridge_built_cells)
        water = set(state.family.water_cells)
        if cell in built:
            built.remove(cell)
            budget = None if state.family.bridge_budget_remaining is None else state.family.bridge_budget_remaining + 1
            event = "remove_bridge"
        elif cell in water and cell not in state.family.reef_cells:
            if state.family.bridge_budget_remaining is not None and state.family.bridge_budget_remaining <= 0:
                successor = replace(state, common=replace(state.common, step_depth=state.common.step_depth + 1))
                return successor, HybridConstructionTransitionAnnotationV4(6, False, True, "bridge_budget_exhausted")
            built.add(cell)
            budget = None if state.family.bridge_budget_remaining is None else state.family.bridge_budget_remaining - 1
            water.remove(cell)
            event = "build_bridge"
        else:
            successor = replace(state, common=replace(state.common, step_depth=state.common.step_depth + 1))
            return successor, HybridConstructionTransitionAnnotationV4(6, False, True, "illegal_bridge_cell")
        step_remaining = None if state.family.step_limit_remaining is None else max(0, state.family.step_limit_remaining - 1)
        successor = HybridConstructionTypedStateV4(
            common=replace(state.common, step_depth=state.common.step_depth + 1),
            family=replace(
                state.family,
                bridge_built_cells=tuple(sorted(built)),
                water_cells=tuple(sorted(water)),
                bridge_budget_remaining=budget,
                step_limit_remaining=step_remaining,
                legal_click_cells=tuple(sorted(set(water) | set(built))),
            ),
            layout_evidence_source=state.layout_evidence_source,
        )
        return successor, HybridConstructionTransitionAnnotationV4(6, True, False, event)
