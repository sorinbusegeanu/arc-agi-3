from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedOutcome:
    context_signature: int
    action_id: int
    outcome_signature: int
    probability: float
    support: int
    expectation_watermark: int


class ExpectationTracker:
    """Maintain pre-observation expectations so prediction error is causally ordered."""

    def __init__(self, *, min_support: int = 3, stability_threshold: float = 0.60) -> None:
        self.min_support = int(min_support)
        self.stability_threshold = float(stability_threshold)
        self._counts: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._watermark: dict[tuple[int, int], int] = {}

    def expectation(self, context_signature: int, action_id: int) -> ExpectedOutcome | None:
        key = (int(context_signature), int(action_id))
        variants = self._counts.get(key)
        if not variants:
            return None
        total = sum(variants.values())
        if total < self.min_support:
            return None
        outcome, support = max(variants.items(), key=lambda item: (item[1], -item[0]))
        probability = support / total
        if probability < self.stability_threshold:
            return None
        return ExpectedOutcome(key[0], key[1], int(outcome), float(probability), int(total), int(self._watermark.get(key, 0)))

    def observe(self, *, context_signature: int, action_id: int, outcome_signature: int, watermark: int) -> float | None:
        expectation = self.expectation(context_signature, action_id)
        key = (int(context_signature), int(action_id))
        self._counts[key][int(outcome_signature)] += 1
        self._watermark[key] = max(int(watermark), self._watermark.get(key, 0))
        if expectation is None:
            return None
        return 0.0 if int(outcome_signature) == expectation.outcome_signature else 1.0
