from __future__ import annotations

"""Environment-neutral cognition contracts for v8.

The cognitive core consumes structural transitions, opaque executable actions,
primary valence and boundary scope.  Environment-specific labels (ARC WIN,
GAME_OVER, levels, grids, etc.) are deliberately outside this module.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from v8.model import stable_u64


class BoundaryScope(str, Enum):
    NONE = "NONE"
    SUBEPISODE = "SUBEPISODE"
    EPISODE = "EPISODE"


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    scope: BoundaryScope = BoundaryScope.NONE
    primary_valence: int = 0
    continuation: bool = True

    def __post_init__(self) -> None:
        value = int(self.primary_valence)
        if value not in (-1, 0, 1):
            raise ValueError("primary_valence must be -1, 0, or +1")
        object.__setattr__(self, "primary_valence", value)
        if not isinstance(self.scope, BoundaryScope):
            object.__setattr__(self, "scope", BoundaryScope(str(self.scope)))

    @property
    def crossed(self) -> bool:
        return self.scope is not BoundaryScope.NONE

    @property
    def positive(self) -> bool:
        return self.crossed and int(self.primary_valence) > 0

    @property
    def negative(self) -> bool:
        return self.crossed and int(self.primary_valence) < 0


@dataclass(frozen=True, slots=True)
class TransitionSemantics:
    boundary: BoundaryEvent = BoundaryEvent()
    structural_changed: bool = False
    context_changed: bool = False

    @property
    def productive(self) -> bool:
        return bool(self.structural_changed or self.context_changed or self.boundary.positive)

    @property
    def successful_boundary(self) -> bool:
        return bool(self.boundary.positive)

    @property
    def terminal_failure(self) -> bool:
        return bool(
            (self.boundary.scope is BoundaryScope.EPISODE and self.boundary.negative)
            or not self.boundary.continuation
        )


@dataclass(frozen=True, slots=True)
class EnvironmentTransition:
    before_observation: Any
    after_observation: Any
    action_token: int
    available_actions_before: tuple[int, ...]
    available_actions_after: tuple[int, ...]
    structural_delta: Any
    boundary: BoundaryEvent = BoundaryEvent()


class EnvironmentCognitionAdapter(Protocol):
    """Minimal runtime surface required by environment-neutral cognition."""

    def observe(self): ...

    def reset(self): ...

    def step(self, action: int): ...

    def available_actions(self) -> list[int] | tuple[int, ...]: ...

    def cognitive_boundary_event(self) -> BoundaryEvent: ...


class OptimizationScopeKind(str, Enum):
    BOUNDARY = "BOUNDARY"
    OUTCOME = "OUTCOME"
    LOCAL = "LOCAL"


@dataclass(frozen=True, slots=True)
class OptimizationScope:
    kind: OptimizationScopeKind
    environment_scope: str
    boundary_scope: BoundaryScope = BoundaryScope.NONE
    primary_valence: int = 0
    outcome_hi: int = 0
    outcome_lo: int = 0
    local_scope: int = 0

    def label(self) -> str:
        if self.kind is OptimizationScopeKind.OUTCOME:
            return f"M6:{int(self.outcome_hi):016x}:{int(self.outcome_lo):016x}"
        if self.kind is OptimizationScopeKind.BOUNDARY:
            sign = f"{int(self.primary_valence):+d}"
            return f"{self.boundary_scope.value}:{sign}"
        return f"LOCAL:{int(self.local_scope)}"

    def legacy_budget_key(self) -> int:
        """Stable scope key stored beside environment id; it is not a level."""
        value = stable_u64(
            self.kind.value,
            self.boundary_scope.value,
            int(self.primary_valence),
            int(self.outcome_hi),
            int(self.outcome_lo),
            int(self.local_scope),
            person=b"v8.37-scope",
        )
        return 1_000_000_000 + int(value % 1_000_000_000)


def _uid_parts(uid) -> tuple[int, int]:
    if uid is None or bool(getattr(uid, "is_zero", True)):
        return 0, 0
    return int(getattr(uid, "hi", 0)), int(getattr(uid, "lo", 0))


def target_boundary(target) -> BoundaryEvent:
    raw_scope = str(getattr(target, "boundary_scope", BoundaryScope.NONE.value))
    try:
        scope = BoundaryScope(raw_scope)
    except ValueError:
        scope = BoundaryScope.NONE
    return BoundaryEvent(
        scope,
        int(getattr(target, "primary_valence", 0)),
        bool(getattr(target, "continuation", scope is not BoundaryScope.EPISODE)),
    )


def optimization_scope_for(source) -> OptimizationScope:
    environment_scope = str(getattr(getattr(source, "anchor", None), "source_id", ""))
    hi, lo = _uid_parts(getattr(source, "target_outcome_uid", None))
    if hi or lo:
        return OptimizationScope(
            OptimizationScopeKind.OUTCOME,
            environment_scope,
            outcome_hi=hi,
            outcome_lo=lo,
        )

    boundary = target_boundary(getattr(source, "target", None))
    if boundary.crossed:
        return OptimizationScope(
            OptimizationScopeKind.BOUNDARY,
            environment_scope,
            boundary.scope,
            int(boundary.primary_valence),
        )

    return OptimizationScope(
        OptimizationScopeKind.LOCAL,
        environment_scope,
        local_scope=max(0, int(getattr(getattr(source, "target", None), "levels_completed", 0))),
    )


def is_complete_positive_episode(source) -> bool:
    scope = optimization_scope_for(source)
    return bool(
        scope.kind is OptimizationScopeKind.BOUNDARY
        and scope.boundary_scope is BoundaryScope.EPISODE
        and int(scope.primary_valence) > 0
        and not tuple(getattr(getattr(source, "anchor", None), "prefix_actions", ()))
    )
