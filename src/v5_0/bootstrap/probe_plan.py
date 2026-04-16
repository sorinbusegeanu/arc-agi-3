from __future__ import annotations

from v5_0.contracts.avatar_types import ProbePlan

DEFAULT_PROBE_SEQUENCE: tuple[str, ...] = (
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
)


def build_probe_plan(*, game_id: str, level_id: str) -> ProbePlan:
    return ProbePlan(game_id=game_id, level_id=level_id, action_sequence=DEFAULT_PROBE_SEQUENCE)
