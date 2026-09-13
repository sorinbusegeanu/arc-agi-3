from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True, slots=True)
class DeepTransferMeasurement:
    depth: int
    maximum_radius: int
    correspondence_found: bool
    false_correspondence: bool
    expansions: int
    compute_cost: int
    causal_transfer_success: bool
    prediction_delta: float


@dataclass(frozen=True, slots=True)
class StructuralPriorMeasurement:
    condition: str
    contingency_steps: int | None
    role_count: int
    concept_count: int
    outcome_count: int
    transfer_score: float
    memory_count: int


def validate_prior_condition(condition: str) -> str:
    if condition not in {"P0", "P1", "P2", "P3"}:
        raise ValueError("structural-prior condition must be P0-P3")
    return condition


class SnapshotCapableEnvironment(Protocol):
    def capture_state(self) -> Any: ...
    def restore_state(self, state: Any) -> None: ...
    def available_actions(self) -> tuple[int, ...]: ...
    def step(self, action: int) -> Any: ...


@dataclass(frozen=True, slots=True)
class MatchedTransferResult:
    enabled_metric: float
    ablated_metric: float
    horizon: int
    initial_actions: tuple[int, ...]
    matched: bool

    @property
    def effect(self) -> float:
        return self.enabled_metric - self.ablated_metric


def run_matched_transfer_trial(
    environment: SnapshotCapableEnvironment,
    *,
    enabled_policy: Callable[[Any, tuple[int, ...], int], int],
    ablated_policy: Callable[[Any, tuple[int, ...], int], int],
    metric: Callable[[SnapshotCapableEnvironment], float],
    horizon: int,
) -> MatchedTransferResult:
    """Run memory-on/off from one exactly captured target-local state."""
    if horizon <= 0:
        raise ValueError("matched transfer horizon must be positive")
    captured = environment.capture_state()
    initial_actions = tuple(environment.available_actions())

    def run(policy: Callable[[Any, tuple[int, ...], int], int]) -> float:
        environment.restore_state(captured)
        if tuple(environment.available_actions()) != initial_actions:
            raise RuntimeError("target snapshot did not restore the same available actions")
        observation: Any = None
        for step in range(int(horizon)):
            actions = tuple(environment.available_actions())
            if not actions:
                break
            action = int(policy(observation, actions, step))
            if action not in actions:
                raise ValueError("transfer policy selected a non-local action")
            observation = environment.step(action)
        return float(metric(environment))

    enabled = run(enabled_policy)
    ablated = run(ablated_policy)
    environment.restore_state(captured)
    return MatchedTransferResult(enabled, ablated, int(horizon), initial_actions, True)
