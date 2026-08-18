from __future__ import annotations

"""Environment-neutral cognition contracts for v8.

The cognitive core consumes structural transitions, opaque executable actions,
primary valence and boundary scope. Environment-specific labels (ARC WIN,
GAME_OVER, levels, grids, etc.) are deliberately outside this module.

v8.8 additionally represents observations emitted between two agent decision
points.  Those within-action frames are passive environment evolution: they
belong to one externally initiated interaction and never imply extra actions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from v8.model import stable_u64


class BoundaryScope(str, Enum):
    NONE = "NONE"
    SUBEPISODE = "SUBEPISODE"
    EPISODE = "EPISODE"


class TransitionOrigin(str, Enum):
    ACTION_TRIGGERED = "ACTION_TRIGGERED"
    INTERNAL_EVOLUTION = "INTERNAL_EVOLUTION"


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
class WithinActionFrame:
    observation: Any
    ordinal: int

    def __post_init__(self) -> None:
        ordinal = int(self.ordinal)
        if ordinal < 0:
            raise ValueError("within-action frame ordinal cannot be negative")
        object.__setattr__(self, "ordinal", ordinal)


@dataclass(frozen=True, slots=True)
class WithinActionTrace:
    """Ordered observations produced by one external action before control returns."""

    initial_observation: Any
    frames: tuple[WithinActionFrame, ...]
    settled_observation: Any

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        for expected, frame in enumerate(frames):
            if int(frame.ordinal) != expected:
                raise ValueError("within-action frame ordinals must be contiguous")
        object.__setattr__(self, "frames", frames)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def animation_frames(self) -> tuple[Any, ...]:
        return tuple(frame.observation for frame in self.frames[:-1])


@dataclass(frozen=True, slots=True)
class EnvironmentStepResult:
    """One agent action plus all observable environment evolution it triggered."""

    settled_observation: Any
    within_action_trace: WithinActionTrace
    available_actions: tuple[int, ...]
    primary_valence: int = 0
    boundary_scope: BoundaryScope = BoundaryScope.NONE
    continuation: bool = True

    def __post_init__(self) -> None:
        actions = tuple(int(value) for value in self.available_actions)
        object.__setattr__(self, "available_actions", actions)
        valence = int(self.primary_valence)
        if valence not in (-1, 0, 1):
            raise ValueError("primary_valence must be -1, 0, or +1")
        object.__setattr__(self, "primary_valence", valence)
        if not isinstance(self.boundary_scope, BoundaryScope):
            object.__setattr__(self, "boundary_scope", BoundaryScope(str(self.boundary_scope)))

    @property
    def boundary(self) -> BoundaryEvent:
        return BoundaryEvent(self.boundary_scope, self.primary_valence, self.continuation)


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
            or (not self.boundary.continuation and int(self.boundary.primary_valence) <= 0)
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
    before_context: int = 0
    after_context: int = 0
    structural_changed: bool = False
    within_action_trace: WithinActionTrace | None = None

    @property
    def semantics(self) -> TransitionSemantics:
        return TransitionSemantics(
            self.boundary,
            structural_changed=bool(self.structural_changed),
            context_changed=int(self.before_context) != int(self.after_context),
        )


class EnvironmentCognitionAdapter(Protocol):
    """Runtime surface required by environment-neutral cognition.

    Adapters are responsible for converting native environment state into the
    declared structural/boundary contract. Cognitive code must not infer native
    task labels when these methods are available.
    """

    def observe(self): ...

    def reset(self): ...

    def step(self, action: int): ...

    def available_actions(self) -> list[int] | tuple[int, ...]: ...

    def cognitive_boundary_event(self) -> BoundaryEvent: ...

    def cognitive_context_signature(self) -> int: ...

    def cognitive_transition_signature(self, before, after) -> int: ...

    def cognitive_subepisode_index(self) -> int: ...

    def cognitive_within_action_trace(self) -> WithinActionTrace | None: ...

    def cognitive_step_result(self) -> EnvironmentStepResult | None: ...

    def cognitive_transition(
        self,
        *,
        before_observation,
        after_observation,
        action_token: int,
        available_actions_before,
        available_actions_after,
    ) -> EnvironmentTransition: ...

    def cognitive_target_reached(self, target, outcome_uid=None) -> bool: ...


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
            base = f"M6:{int(self.outcome_hi):016x}:{int(self.outcome_lo):016x}"
            if self.boundary_scope is not BoundaryScope.NONE:
                return f"{base}@{self.boundary_scope.value}:{int(self.primary_valence):+d}"
            return base
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
    if not (hi or lo):
        target = getattr(source, "target", None)
        hi = int(getattr(target, "outcome_hi", 0))
        lo = int(getattr(target, "outcome_lo", 0))

    boundary = target_boundary(getattr(source, "target", None))
    if hi or lo:
        return OptimizationScope(
            OptimizationScopeKind.OUTCOME,
            environment_scope,
            boundary.scope,
            int(boundary.primary_valence),
            outcome_hi=hi,
            outcome_lo=lo,
        )

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
    boundary = target_boundary(getattr(source, "target", None))
    return bool(
        boundary.scope is BoundaryScope.EPISODE
        and int(boundary.primary_valence) > 0
        and not tuple(getattr(getattr(source, "anchor", None), "prefix_actions", ()))
        and scope.kind in {OptimizationScopeKind.BOUNDARY, OptimizationScopeKind.OUTCOME}
    )
