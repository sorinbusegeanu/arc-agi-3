from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MechanicClassifierConfig:
    initial_T: int = 2
    max_families_emitted: int = 8
    evidence_per_family: int = 6
    unknown_floor: float = 0.05
    score_threshold: float = 0.10
