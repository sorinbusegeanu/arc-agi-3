from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.model import MemoryUid
from v8.strategies import StrategyEvidence


@dataclass(frozen=True, slots=True)
class PreferenceEvidence:
    preferred: MemoryUid
    other: MemoryUid
    context_bucket: int
    strength: float
    state: str


class PreferenceEstimator:
    """Infer target-like preference only from repeated comparable strategy evidence."""

    def __init__(self, *, support_threshold: int = 6, stable_margin: float = 0.30) -> None:
        self.support_threshold = int(support_threshold)
        self.stable_margin = float(stable_margin)

    def evaluate(self, strategies: tuple[StrategyEvidence, ...]) -> tuple[PreferenceEvidence, ...]:
        by_context: dict[int, dict[MemoryUid, int]] = defaultdict(lambda: defaultdict(int))
        for strategy in strategies:
            by_context[int(strategy.context_bucket)][strategy.outcome_uid] += int(strategy.attempts)
        result = []
        for context, outcome_counts in by_context.items():
            if len(outcome_counts) < 2:
                continue
            ranked = sorted(outcome_counts.items(), key=lambda item: (-item[1], item[0]))
            preferred, preferred_count = ranked[0]
            total = sum(outcome_counts.values())
            if total < self.support_threshold:
                state = "CANDIDATE"
            else:
                second_count = ranked[1][1]
                margin = (preferred_count - second_count) / max(1, total)
                state = "STABLE" if margin >= self.stable_margin else "PROBE_ONLY"
            for other, other_count in ranked[1:]:
                strength = (preferred_count - other_count) / max(1, total)
                result.append(PreferenceEvidence(preferred, other, context, strength, state))
        return tuple(result)
