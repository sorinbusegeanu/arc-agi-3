from __future__ import annotations

from v9.memory.m4_concept import M4Concept


def validate_concept(candidate: M4Concept, *, successful_held_out_targets: tuple[int, ...], required_targets: int = 1) -> M4Concept:
    result = candidate.with_validation(successful_held_out_targets)
    if len(result.held_out_targets) < required_targets:
        return result
    return result

