from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

from v7.environment.parallel_sampling import SamplingJob


class CognitionAblation(IntFlag):
    NONE = 0
    PERSISTENT_PLANNING = 1 << 0
    STRATEGY_EXECUTION = 1 << 1
    FUNCTIONAL_ROLES = 1 << 2
    RELATIONAL_WORLD_MODELS = 1 << 3
    DEVELOPMENTAL_POLICY = 1 << 4
    FUTURE_OPTION = 1 << 5


@dataclass(frozen=True, slots=True)
class MatureSamplingJob(SamplingJob):
    ablation_mask: int = 0


_NAME_TO_FLAG = {
    "persistent_planning": CognitionAblation.PERSISTENT_PLANNING,
    "planning": CognitionAblation.PERSISTENT_PLANNING,
    "strategy_execution": CognitionAblation.STRATEGY_EXECUTION,
    "strategy": CognitionAblation.STRATEGY_EXECUTION,
    "functional_roles": CognitionAblation.FUNCTIONAL_ROLES,
    "roles": CognitionAblation.FUNCTIONAL_ROLES,
    "relational_world_models": CognitionAblation.RELATIONAL_WORLD_MODELS,
    "relational_m5": CognitionAblation.RELATIONAL_WORLD_MODELS,
    "developmental_policy": CognitionAblation.DEVELOPMENTAL_POLICY,
    "development": CognitionAblation.DEVELOPMENTAL_POLICY,
    "future_option": CognitionAblation.FUTURE_OPTION,
    "future_options": CognitionAblation.FUTURE_OPTION,
}


def ablated(mask: int, flag: CognitionAblation) -> bool:
    return bool(CognitionAblation(int(mask)) & flag)


def parse_ablation_spec(spec: str | None) -> int:
    if spec is None or not str(spec).strip() or str(spec).strip().lower() == "none":
        return 0
    mask = CognitionAblation.NONE
    for raw in str(spec).split(","):
        name = raw.strip().lower().replace("-", "_")
        if not name:
            continue
        try:
            mask |= _NAME_TO_FLAG[name]
        except KeyError as exc:
            allowed = ", ".join(sorted(set(_NAME_TO_FLAG)))
            raise ValueError(f"unknown ablation '{raw}'; allowed: {allowed}") from exc
    return int(mask)


def ablation_names(mask: int) -> tuple[str, ...]:
    value = CognitionAblation(int(mask))
    canonical = (
        ("persistent_planning", CognitionAblation.PERSISTENT_PLANNING),
        ("strategy_execution", CognitionAblation.STRATEGY_EXECUTION),
        ("functional_roles", CognitionAblation.FUNCTIONAL_ROLES),
        ("relational_world_models", CognitionAblation.RELATIONAL_WORLD_MODELS),
        ("developmental_policy", CognitionAblation.DEVELOPMENTAL_POLICY),
        ("future_option", CognitionAblation.FUTURE_OPTION),
    )
    return tuple(name for name, flag in canonical if value & flag)


def standard_ablation_masks() -> dict[str, int]:
    return {
        "baseline": 0,
        "no_persistent_planning": int(CognitionAblation.PERSISTENT_PLANNING),
        "no_strategy_execution": int(CognitionAblation.STRATEGY_EXECUTION),
        "no_functional_roles": int(CognitionAblation.FUNCTIONAL_ROLES),
        "no_relational_world_models": int(
            CognitionAblation.RELATIONAL_WORLD_MODELS
        ),
        "no_developmental_policy": int(CognitionAblation.DEVELOPMENTAL_POLICY),
        "no_future_option": int(CognitionAblation.FUTURE_OPTION),
    }
