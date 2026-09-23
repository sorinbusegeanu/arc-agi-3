from __future__ import annotations

from random import Random
from dataclasses import dataclass

from v9.cognition.grounding import GroundingMaturity
from v9.memory.model import MemoryLevel
from v9.memory.model import MemoryType
from v9.runtime.read_view import ReadView


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


def action_scores(view: ReadView, actions: tuple[int, ...], *, grounded_signals: tuple[GroundedActionSignal, ...] = (), target_environment_id: int | None = None) -> dict[int, float]:
    scores = {int(action): 0.0 for action in actions}
    for uid, node in view.nodes.items():
        if node.memory_type is not MemoryType.NORMALIZED_RELATION:
            continue
        payload = view.payloads[uid]
        observable = str(payload.get("observable_relation", ""))
        for action in scores:
            if observable.startswith(f"ACTION:{action}:"):
                scores[action] += float(payload.get("support", 0))
    for signal in grounded_signals:
        if signal.authoritative and target_environment_id is not None and signal.target_environment_id == int(target_environment_id) and signal.native_action in scores:
            scores[signal.native_action] += float(signal.score)
    return scores


def choose_action(view: ReadView, actions: tuple[int, ...], *, rng: Random, epsilon: float, grounded_signals: tuple[GroundedActionSignal, ...] = (), target_environment_id: int | None = None) -> int:
    if not actions:
        raise ValueError("cannot choose from an empty action set")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    scores = action_scores(view, actions, grounded_signals=grounded_signals, target_environment_id=target_environment_id)
    unseen = tuple(action for action in actions if scores[action] == 0)
    if rng.random() < epsilon or unseen:
        candidates = unseen or actions
        return int(candidates[rng.randrange(len(candidates))])
    return min(actions, key=lambda action: (-scores[action], action))
