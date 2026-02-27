from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoalDetectorConfig:
    min_window_steps: int = 2
    max_window_steps: int = 20

    w_target_depletion: float = 0.35
    w_filled_area: float = 0.20
    w_stability: float = 0.15
    w_uniformity: float = 0.15
    w_symmetry: float = 0.10
    w_component_consolidation: float = 0.05

    uniformity_goal_threshold: float = 0.98
    stability_goal_threshold: float = 0.95
    min_target_color_rarity: float = 0.10

    confidence_low: float = 0.30
    confidence_high: float = 0.70
