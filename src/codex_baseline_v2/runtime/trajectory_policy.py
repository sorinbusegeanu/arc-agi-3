from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.schemas import ActionContextStatsV2, ActionDescriptorV2, ActionSemanticsStatsV2, ControllerInstructionV2, SCHEMA_VERSION


@dataclass
class PolicyStateV2:
    last_action: Optional[ActionDescriptorV2] = None
    blocked_steps: int = 0
    step_idx: int = 0
    round_id: int = 0
    current_position: Optional[Tuple[int, int]] = None
    target_centroid: Optional[Tuple[int, int]] = None
    action_motion_map: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    recent_positions: List[Tuple[int, int]] = field(default_factory=list)
    recent_actions: List[int] = field(default_factory=list)
    action_semantics_table: List[ActionSemanticsStatsV2] = field(default_factory=list)
    action_context_table: List[ActionContextStatsV2] = field(default_factory=list)
    action_context_key: Optional[str] = None


class TrajectoryPolicyV2:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = int(seed or 0)
        self.rng = random.Random(seed)

    def random_action(self, available_actions: Optional[List[int]] = None) -> ActionDescriptorV2:
        if available_actions:
            action_id = self.rng.choice(list(available_actions))
        else:
            action_id = self.rng.randint(0, 5)
        return ActionDescriptorV2(schema_version=SCHEMA_VERSION, action_type="discrete", action_id=action_id, coord=None, raw=None)

    def unguided_probe(self, available_actions: Optional[List[int]] = None) -> ActionDescriptorV2:
        return self.random_action(available_actions)

    def observe_transition(
        self,
        state: PolicyStateV2,
        prev_position: Optional[Tuple[int, int]],
        new_position: Optional[Tuple[int, int]],
        action: Optional[ActionDescriptorV2],
    ) -> None:
        if prev_position is None or new_position is None or action is None or action.action_id is None:
            return
        dx = int(new_position[0] - prev_position[0])
        dy = int(new_position[1] - prev_position[1])
        state.action_motion_map[action.action_id] = (dx, dy)

    def instructed_action(
        self,
        instruction: ControllerInstructionV2,
        target_coord: Optional[Tuple[int, int]],
        available_actions: Optional[List[int]],
        state: PolicyStateV2,
    ) -> ActionDescriptorV2:
        if instruction.target_poi_id is None or target_coord is None:
            return self.unguided_probe(available_actions)
        state.target_centroid = (int(target_coord[0]), int(target_coord[1]))
        if available_actions is None:
            return ActionDescriptorV2(
                schema_version=SCHEMA_VERSION,
                action_type="coord",
                action_id=None,
                coord=(int(target_coord[0]), int(target_coord[1])),
                raw=None,
            )
        if not available_actions:
            return self.fallback_action(available_actions, state)

        cur = state.current_position
        next_goal = state.target_centroid
        desired_dx = 0 if cur is None else max(-1, min(1, next_goal[0] - cur[0]))
        desired_dy = 0 if cur is None else max(-1, min(1, next_goal[1] - cur[1]))
        semantic_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }
        scored: List[Tuple[float, int]] = []
        semantics_by_action = {row.action_id: row for row in state.action_semantics_table}
        context_by_action = {(row.action_id, row.context_key): row for row in state.action_context_table}
        for action_id in available_actions:
            motion = state.action_motion_map.get(action_id, semantic_map.get(action_id, (0, 0)))
            next_pos = cur if cur is not None else (0, 0)
            if cur is not None:
                next_pos = (cur[0] + motion[0], cur[1] + motion[1])
            routing_gain = 0.0
            if cur is not None:
                cur_dist = abs(next_goal[0] - cur[0]) + abs(next_goal[1] - cur[1])
                next_dist = abs(next_goal[0] - next_pos[0]) + abs(next_goal[1] - next_pos[1])
                routing_gain = float(cur_dist - next_dist)
            score = routing_gain * 10.0
            if motion == (desired_dx, desired_dy):
                score += 5.0
            elif desired_dx != 0 and motion[0] == desired_dx:
                score += 2.5
            elif desired_dy != 0 and motion[1] == desired_dy:
                score += 2.5
            semantics = semantics_by_action.get(int(action_id))
            if semantics is not None:
                score += semantics.confidence * 2.0
                score -= (semantics.blocked_count / float(max(1, semantics.sample_count))) * 2.0
                score -= (semantics.noop_count / float(max(1, semantics.sample_count))) * 1.5
            context_stats = context_by_action.get((int(action_id), state.action_context_key or ""))
            if context_stats is not None:
                score += context_stats.success_rate * 3.0
                score -= context_stats.blocked_rate * 2.0
                score -= context_stats.transition_rate * 0.5
            if state.last_action and state.last_action.action_id == action_id:
                score -= 0.5
            if motion == (0, 0):
                score -= 1.0
            scored.append((score, int(action_id)))

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        if not scored or scored[0][0] <= 0.0:
            return self.fallback_action(available_actions, state)
        action_id = scored[0][1]
        action = ActionDescriptorV2(schema_version=SCHEMA_VERSION, action_type="discrete", action_id=action_id, coord=None, raw=None)
        state.last_action = action
        state.step_idx += 1
        if action.action_id is not None:
            state.recent_actions.append(action.action_id)
            state.recent_actions = state.recent_actions[-8:]
        return action

    def fallback_action(self, available_actions: Optional[List[int]] = None, state: Optional[PolicyStateV2] = None) -> ActionDescriptorV2:
        if available_actions:
            idx_seed = self.seed
            if state is not None:
                idx_seed += int(state.round_id) * 997 + int(state.step_idx) * 37
            ordered = sorted(int(a) for a in available_actions)
            action_id = ordered[idx_seed % len(ordered)]
            return ActionDescriptorV2(schema_version=SCHEMA_VERSION, action_type="discrete", action_id=action_id, coord=None, raw=None)
        return self.random_action(available_actions)
