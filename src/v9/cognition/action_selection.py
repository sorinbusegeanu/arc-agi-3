from __future__ import annotations

from random import Random
from dataclasses import dataclass

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


def choose_action(view: ReadView, actions: tuple[int, ...], *, rng: Random, epsilon: float, grounded_signals: tuple[GroundedActionSignal, ...] = (), learned_scores: dict[int, float] | None = None, target_environment_id: int | None = None, action_schema_id: int | None = None, environment_type: str | None = None) -> int:
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
    if rng.random() < epsilon:
        candidates = unseen or actions
        return int(candidates[rng.randrange(len(candidates))])
    return min(actions, key=lambda action: (-scores[action], action))
