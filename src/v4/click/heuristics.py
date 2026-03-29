from __future__ import annotations

from .typedState import ClickTypedStateV4


def zero_heuristic(_: ClickTypedStateV4) -> int:
    return 0


def pt01_remaining_rotation_heuristic(state: ClickTypedStateV4) -> int:
    targets = dict(state.family.target_rotations_by_type)
    mismatched = 0
    for _, sprite_type, rotation in state.family.rotation_tiles:
        if targets.get(sprite_type) != rotation:
            mismatched += 1
    return mismatched
