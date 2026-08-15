from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable
from weakref import ReferenceType, ref

from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.planning import TYPE_EXECUTABLE_PROCEDURE
from v7.memory.read_view import MemoryReadView
from v7.memory.status import memory_is_active


class DevelopmentStage(IntEnum):
    CONTROL = 0
    CONTINGENCY = 1
    ABSTRACTION = 2
    TRANSFER = 3
    PLANNING = 4
    STRATEGY = 5


@dataclass(frozen=True, slots=True)
class DevelopmentProfile:
    stage: DevelopmentStage
    exploration_multiplier: float
    planning_depth: int
    exact_specialization_min_support: int
    contradiction_specialization_threshold: float
    abstraction_budget: int
    replay_levels: tuple[MemoryLevel, ...]


_PROFILES = {
    DevelopmentStage.CONTROL: DevelopmentProfile(
        DevelopmentStage.CONTROL,
        exploration_multiplier=1.50,
        planning_depth=1,
        exact_specialization_min_support=2,
        contradiction_specialization_threshold=0.20,
        abstraction_budget=64,
        replay_levels=(MemoryLevel.M1,),
    ),
    DevelopmentStage.CONTINGENCY: DevelopmentProfile(
        DevelopmentStage.CONTINGENCY,
        exploration_multiplier=1.25,
        planning_depth=2,
        exact_specialization_min_support=2,
        contradiction_specialization_threshold=0.25,
        abstraction_budget=128,
        replay_levels=(MemoryLevel.M1, MemoryLevel.M2),
    ),
    DevelopmentStage.ABSTRACTION: DevelopmentProfile(
        DevelopmentStage.ABSTRACTION,
        exploration_multiplier=1.00,
        planning_depth=2,
        exact_specialization_min_support=3,
        contradiction_specialization_threshold=0.30,
        abstraction_budget=192,
        replay_levels=(MemoryLevel.M2, MemoryLevel.M3),
    ),
    DevelopmentStage.TRANSFER: DevelopmentProfile(
        DevelopmentStage.TRANSFER,
        exploration_multiplier=0.80,
        planning_depth=3,
        exact_specialization_min_support=3,
        contradiction_specialization_threshold=0.35,
        abstraction_budget=256,
        replay_levels=(MemoryLevel.M3, MemoryLevel.M4),
    ),
    DevelopmentStage.PLANNING: DevelopmentProfile(
        DevelopmentStage.PLANNING,
        exploration_multiplier=0.55,
        planning_depth=4,
        exact_specialization_min_support=4,
        contradiction_specialization_threshold=0.40,
        abstraction_budget=384,
        replay_levels=(MemoryLevel.M4, MemoryLevel.M5),
    ),
    DevelopmentStage.STRATEGY: DevelopmentProfile(
        DevelopmentStage.STRATEGY,
        exploration_multiplier=0.25,
        planning_depth=4,
        exact_specialization_min_support=4,
        contradiction_specialization_threshold=0.45,
        abstraction_budget=512,
        replay_levels=(MemoryLevel.M5, MemoryLevel.M6),
    ),
}

_PROFILE_CACHE_VIEW: ReferenceType[MemoryReadView] | None = None
_PROFILE_CACHE_VALUE: DevelopmentProfile | None = None


def development_profile(stage: DevelopmentStage) -> DevelopmentProfile:
    return _PROFILES[DevelopmentStage(stage)]


def infer_development_stage(view: MemoryReadView) -> DevelopmentStage:
    active_nodes = tuple(
        node for node in view.nodes.values() if memory_is_active(node)
    )
    stable_m1 = any(
        node.level == MemoryLevel.M1 and int(node.support_count) >= 2
        for node in active_nodes
    )
    if not stable_m1:
        return DevelopmentStage.CONTROL

    has_m2 = any(node.level == MemoryLevel.M2 for node in active_nodes)
    if not has_m2:
        return DevelopmentStage.CONTINGENCY

    # Functional roles use type 300. Other M3 subtypes deliberately do not
    # advance developmental stage, and demoted roles never count as maturity.
    has_functional_role = any(
        node.level == MemoryLevel.M3 and int(node.type_id) == 300
        for node in active_nodes
    )
    if not has_functional_role:
        return DevelopmentStage.ABSTRACTION

    validated_mask = int(
        ConceptValidationStatus.TRANSFER_VALIDATED
        | ConceptValidationStatus.TRUSTED
    )
    has_validated_concept = any(
        node.level == MemoryLevel.M4
        and bool(int(node.status_flags) & validated_mask)
        for node in active_nodes
    )
    if not has_validated_concept:
        return DevelopmentStage.TRANSFER

    has_procedure = any(
        node.level == MemoryLevel.M6
        and int(node.type_id) == int(TYPE_EXECUTABLE_PROCEDURE)
        and int(node.support_count) > 0
        for node in active_nodes
    )
    if not has_procedure:
        return DevelopmentStage.PLANNING
    return DevelopmentStage.STRATEGY


def profile_for_view(view: MemoryReadView) -> DevelopmentProfile:
    """Return the immutable generation profile without rescanning it per action."""
    global _PROFILE_CACHE_VIEW, _PROFILE_CACHE_VALUE
    cached_view = None if _PROFILE_CACHE_VIEW is None else _PROFILE_CACHE_VIEW()
    if cached_view is view and _PROFILE_CACHE_VALUE is not None:
        return _PROFILE_CACHE_VALUE
    profile = development_profile(infer_development_stage(view))
    _PROFILE_CACHE_VIEW = ref(view)
    _PROFILE_CACHE_VALUE = profile
    return profile


def focused_replay_ids(
    view: MemoryReadView,
    *,
    profile: DevelopmentProfile | None = None,
    limit: int = 64,
) -> tuple[MemoryId, ...]:
    """Return deterministic stage-relevant active memories for replay."""
    if limit <= 0:
        return ()
    profile = profile or profile_for_view(view)
    allowed = set(profile.replay_levels)
    rows: list[tuple[float, int, MemoryId]] = []
    for memory_id, node in view.nodes.items():
        if node.level not in allowed or not memory_is_active(node):
            continue
        score = view.scores.get(memory_id)
        support = min(
            1.0,
            math.log1p(max(0, int(node.support_count))) / math.log1p(8.0),
        )
        semantic = 0.0
        if score is not None:
            semantic = max(
                0.0,
                float(score.prediction_error),
                float(score.learning_value),
                float(score.transfer_prior),
                float(score.explanatory_potential),
                abs(float(score.future_option_delta)),
            )
        priority = 0.60 * semantic + 0.40 * support
        rows.append((priority, -int(node.level), memory_id))
    rows.sort(key=lambda item: (-item[0], item[1], int(item[2])))
    return tuple(item[2] for item in rows[: int(limit)])


def stage_counts(view: MemoryReadView) -> dict[str, int | str]:
    counts = {f"M{level}": 0 for level in range(7)}
    for node in view.nodes.values():
        if memory_is_active(node):
            counts[f"M{int(node.level)}"] += 1
    stage = infer_development_stage(view)
    return {"stage": stage.name, **counts}
