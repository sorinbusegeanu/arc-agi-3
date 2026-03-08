from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RoundSchedule:
    unguided_probe: int
    discriminating_probe: int
    poi_approach: int
    poi_interaction_probe: int
    exploit_route: int


def schedule_round(budget: int, fractions: Dict[str, float]) -> RoundSchedule:
    def _alloc(key: str) -> int:
        return max(1, int(budget * float(fractions.get(key, 0.0))))

    unguided = _alloc("unguided")
    discr = _alloc("discriminating")
    approach = _alloc("approach")
    interact = _alloc("interaction")
    exploit = max(0, budget - (unguided + discr + approach + interact))
    return RoundSchedule(
        unguided_probe=unguided,
        discriminating_probe=discr,
        poi_approach=approach,
        poi_interaction_probe=interact,
        exploit_route=exploit,
    )
