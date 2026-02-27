from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MechanicSynthesizerConfig:
    window_N: int = 12
    L_min: float = 0.35
    R_max: float = 0.50
    ambiguity_delta: float = 0.05
    ambiguity_M: int = 4
    beam_per_action_family: int = 3
    max_candidates_total: int = 12
    max_mode_states: int = 3
    complexity_penalty: float = 0.02
