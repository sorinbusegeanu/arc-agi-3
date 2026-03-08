from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from codex_baseline_v2.shared.schemas import ActionDescriptorV2, ControllerInstructionV2, SCHEMA_VERSION


@dataclass
class PolicyStateV2:
    last_action: Optional[ActionDescriptorV2] = None
    blocked_steps: int = 0


class TrajectoryPolicyV2:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def random_action(self, available_actions: Optional[List[int]] = None) -> ActionDescriptorV2:
        if available_actions:
            action_id = self.rng.choice(list(available_actions))
        else:
            action_id = self.rng.randint(0, 5)
        return ActionDescriptorV2(schema_version=SCHEMA_VERSION, action_type="discrete", action_id=action_id, coord=None, raw=None)

    def unguided_probe(self, available_actions: Optional[List[int]] = None) -> ActionDescriptorV2:
        return self.random_action(available_actions)

    def instructed_action(
        self,
        instruction: ControllerInstructionV2,
        target_coord: Optional[Tuple[int, int]],
        available_actions: Optional[List[int]],
        state: PolicyStateV2,
    ) -> ActionDescriptorV2:
        if instruction.target_poi_id is None or target_coord is None:
            return self.unguided_probe(available_actions)
        # Minimal heuristic: attempt coordinate action if supported.
        if available_actions is None:
            return ActionDescriptorV2(
                schema_version=SCHEMA_VERSION,
                action_type="coord",
                action_id=None,
                coord=(int(target_coord[0]), int(target_coord[1])),
                raw=None,
            )
        # If only discrete actions available, pick a random valid action.
        return self.random_action(available_actions)

    def fallback_action(self, available_actions: Optional[List[int]] = None) -> ActionDescriptorV2:
        return self.random_action(available_actions)
