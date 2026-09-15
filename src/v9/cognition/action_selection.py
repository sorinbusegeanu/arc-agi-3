from __future__ import annotations

from random import Random
import math
from dataclasses import dataclass
from typing import Mapping

from v9.cognition.grounding import GroundingMaturity
from v9.memory.model import MemoryLevel
from v9.runtime.read_view import ReadView
from v9.memory.identity import stable_u64


@dataclass(frozen=True, slots=True)
class GroundedActionSignal:
    native_action: int
    score: float
    grounding_maturity: GroundingMaturity
    source_level: MemoryLevel
    validated: bool
    source_environment_id: int
    target_environment_id: int

    @property
    def authoritative(self) -> bool:
        required = GroundingMaturity.G4 if self.source_environment_id == self.target_environment_id else GroundingMaturity.G5
        return self.validated and self.grounding_maturity >= required and self.source_level >= MemoryLevel.M4


def scoped_action_key(action: int, *, action_schema_id: int | None = None, environment_type: str | None = None) -> int:
    if action_schema_id is None or environment_type is None:
        return int(action)
    return int(stable_u64(int(action_schema_id), str(environment_type), int(action), person=b"v9-action-scope"))


def action_scores(view: ReadView, actions: tuple[int, ...], *, grounded_signals: tuple[GroundedActionSignal, ...] = (), learned_scores: dict[int, float] | None = None, target_environment_id: int | None = None, action_schema_id: int | None = None, environment_type: str | None = None) -> dict[int, float]:
    raw_support = {
        int(action): float(
            view.normalized_action_supports.get(
                scoped_action_key(int(action), action_schema_id=action_schema_id, environment_type=environment_type),
                0.0,
            )
        )
        for action in actions
    }
    maximum = max(raw_support.values(), default=0.0)
    scores = {
        action: (0.05 * support / maximum if maximum > 0.0 else 0.0)
        for action, support in raw_support.items()
    }
    if learned_scores:
        for action in actions:
            scores[int(action)] += float(learned_scores.get(int(action), 0.0))
    for signal in grounded_signals:
        if signal.authoritative and target_environment_id is not None and signal.target_environment_id == int(target_environment_id) and signal.native_action in scores:
            scores[signal.native_action] += float(signal.score)
    return scores


def adaptive_epsilon(
    base_epsilon: float,
    branching_factor: int,
    *,
    coverage: float = 1.0,
    uncertainty: float = 0.0,
    stagnation: float = 0.0,
) -> float:
    base = max(0.0, min(1.0, float(base_epsilon)))
    branches = max(1, int(branching_factor))
    branch_scale = 1.0 if branches <= 2 else 1.0 + math.log2(branches / 2.0)
    pressure = branch_scale + max(0.0, 1.0 - float(coverage)) + max(0.0, float(uncertainty)) + max(0.0, float(stagnation))
    return min(0.50, base * pressure)


def branching_aware_epsilon(base_epsilon: float, branching_factor: int) -> float:
    return adaptive_epsilon(base_epsilon, branching_factor)


def policy_uncertainty(actions: tuple[int, ...], scores: Mapping[int, float]) -> float:
    if len(actions) <= 1:
        return 0.0
    ranked = sorted((float(scores.get(int(action), 0.0)) for action in actions), reverse=True)
    margin = max(0.0, min(1.0, ranked[0] - ranked[1]))
    return 1.0 - margin


def choose_action(view: ReadView, actions: tuple[int, ...], *, rng: Random, epsilon: float, grounded_signals: tuple[GroundedActionSignal, ...] = (), learned_scores: dict[int, float] | None = None, target_environment_id: int | None = None, action_schema_id: int | None = None, environment_type: str | None = None, context_action_counts: Mapping[int, int] | None = None, stagnation: float = 0.0) -> int:
    if not actions:
        raise ValueError("cannot choose from an empty action set")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    scores = action_scores(
        view,
        actions,
        grounded_signals=grounded_signals,
        learned_scores=learned_scores,
        target_environment_id=target_environment_id,
        action_schema_id=action_schema_id,
        environment_type=environment_type,
    )
    unseen = tuple(
        action
        for action in actions
        if view.normalized_action_supports.get(
            scoped_action_key(int(action), action_schema_id=action_schema_id, environment_type=environment_type),
            0.0,
        ) == 0.0
        and (not learned_scores or float(learned_scores.get(int(action), 0.0)) == 0.0)
    )
    counts = context_action_counts or {}
    tried = sum(1 for action in actions if int(counts.get(int(action), 0)) > 0)
    coverage = tried / max(1, len(actions))
    effective_epsilon = adaptive_epsilon(
        epsilon,
        len(actions),
        coverage=coverage,
        uncertainty=policy_uncertainty(actions, scores),
        stagnation=stagnation,
    )
    if rng.random() < effective_epsilon:
        candidates = unseen or actions
        minimum_count = min((int(counts.get(int(action), 0)) for action in candidates), default=0)
        informative = tuple(action for action in candidates if int(counts.get(int(action), 0)) == minimum_count)
        candidates = informative or candidates
        return int(candidates[rng.randrange(len(candidates))])
    best_score = max(float(scores[action]) for action in actions)
    tied = tuple(action for action in actions if abs(float(scores[action]) - best_score) <= 1e-12)
    if len(tied) > 1:
        minimum_count = min(int(counts.get(int(action), 0)) for action in tied)
        least_used = tuple(action for action in tied if int(counts.get(int(action), 0)) == minimum_count)
        return int(least_used[rng.randrange(len(least_used))])
    return int(tied[0])
