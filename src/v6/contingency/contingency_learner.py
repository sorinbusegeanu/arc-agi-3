from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Contingency:
    id: int
    context_level: int
    context_signature: tuple
    action: int
    transformation_family: int
    support_count: int
    confidence: float


class ContingencyLearner:
    def __init__(self, support_threshold: int = 20, confidence_threshold: float = 0.8) -> None:
        self.support_threshold = int(support_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.counts: Counter[tuple[int, tuple, int, int]] = Counter()
        self.context_action_totals: Counter[tuple[int, tuple, int]] = Counter()
        self.contingencies: dict[tuple[int, tuple, int, int], Contingency] = {}
        self._next_contingency_id = 1

    def update(self, context_signature: tuple, action: int, transformation_family: int) -> Contingency | None:
        return self.update_level(0, context_signature, action, transformation_family)

    def update_level(self, context_level: int, context_signature: tuple, action: int, transformation_family: int) -> Contingency | None:
        context = tuple(context_signature)
        level = int(context_level)
        key = (level, context, int(action), int(transformation_family))
        total_key = (level, context, int(action))
        self.counts[key] += 1
        self.context_action_totals[total_key] += 1
        return self._contingency_if_stable(level, context, int(action), int(transformation_family))

    def update_multi_scale(self, context_signatures: dict[int, tuple], action: int, transformation_family: int) -> Contingency | None:
        for level, context_signature in context_signatures.items():
            context = tuple(context_signature)
            key = (int(level), context, int(action), int(transformation_family))
            total_key = (int(level), context, int(action))
            self.counts[key] += 1
            self.context_action_totals[total_key] += 1
            self._contingency_if_stable(int(level), context, int(action), int(transformation_family))
        return self.best_stable_for_action(context_signatures, action)

    def _contingency_if_stable(self, context_level: int, context: tuple, action: int, transformation_family: int) -> Contingency | None:
        key = (int(context_level), context, int(action), int(transformation_family))
        support_count = int(self.counts[key])
        confidence = self.confidence_at_level(context_level, context, action, transformation_family)
        if support_count < self.support_threshold or confidence < self.confidence_threshold:
            return None
        existing = self.contingencies.get(key)
        if existing is not None:
            contingency = Contingency(
                id=existing.id,
                context_level=int(context_level),
                context_signature=context,
                action=int(action),
                transformation_family=int(transformation_family),
                support_count=support_count,
                confidence=confidence,
            )
        else:
            contingency = Contingency(
                id=self._next_contingency_id,
                context_level=int(context_level),
                context_signature=context,
                action=int(action),
                transformation_family=int(transformation_family),
                support_count=support_count,
                confidence=confidence,
            )
            self._next_contingency_id += 1
        self.contingencies[key] = contingency
        return contingency

    def confidence(self, context_signature: tuple, action: int, transformation_family: int) -> float:
        return self.confidence_at_level(0, context_signature, action, transformation_family)

    def confidence_at_level(self, context_level: int, context_signature: tuple, action: int, transformation_family: int) -> float:
        context = tuple(context_signature)
        total = self.context_action_totals[(int(context_level), context, int(action))]
        if total <= 0:
            return 0.0
        return float(self.counts[(int(context_level), context, int(action), int(transformation_family))] / total)

    def distribution(self, context_signature: tuple, action: int) -> dict[int, float]:
        return self.distribution_at_level(0, context_signature, action)

    def distribution_at_level(self, context_level: int, context_signature: tuple, action: int) -> dict[int, float]:
        context = tuple(context_signature)
        level = int(context_level)
        total = self.context_action_totals[(level, context, int(action))]
        if total <= 0:
            return {}
        by_family: defaultdict[int, int] = defaultdict(int)
        for (known_level, known_context, known_action, family), count in self.counts.items():
            if known_level == level and known_context == context and known_action == int(action):
                by_family[int(family)] += int(count)
        return {family: count / total for family, count in by_family.items()}

    def predict(self, context_signatures: dict[int, tuple], action: int) -> int | None:
        contingency = self.best_stable_for_action(context_signatures, action)
        if contingency is not None:
            return int(contingency.transformation_family)
        for level in sorted(context_signatures, reverse=True):
            distribution = self.distribution_at_level(level, context_signatures[level], action)
            if distribution:
                return max(distribution.items(), key=lambda item: (item[1], -item[0]))[0]
        return None

    def best_stable_for_action(self, context_signatures: dict[int, tuple], action: int) -> Contingency | None:
        for level in sorted(context_signatures, reverse=True):
            context = tuple(context_signatures[level])
            candidates = [
                contingency
                for (known_level, known_context, known_action, _family), contingency in self.contingencies.items()
                if known_level == int(level) and known_context == context and known_action == int(action)
            ]
            if candidates:
                return max(candidates, key=lambda item: (item.confidence, item.support_count, -item.transformation_family))
        return None

    def stable_contingencies(self) -> list[Contingency]:
        return sorted(self.contingencies.values(), key=lambda item: (item.context_level, item.action, item.id))
