from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    mechanic_conf_threshold: float = 0.55
    hypothesis_conf_threshold: float = 0.55
    goal_conf_threshold: float = 0.70

    loop_window_N: int = 25
    loop_repeat_R: int = 6

    w_novelty: float = 0.40
    w_disambiguation: float = 0.30
    w_effect: float = 0.20
    w_loop: float = 0.30
    w_cost: float = 0.05

    w_progress: float = 0.45
    w_hypothesis_align: float = 0.25

    max_candidates: int = 64
    max_tests_considered: int = 16
    max_frontier_considered: int = 32

    coord_action_cost: float = 0.1
    recent_noop_window: int = 6
    loop_avoid_recent_K: int = 3
