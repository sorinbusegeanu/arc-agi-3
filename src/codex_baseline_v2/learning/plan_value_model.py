from __future__ import annotations

from typing import Dict


def estimate_plan_value(features: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(float(weights.get(key, 0.0)) * float(value) for key, value in features.items())
