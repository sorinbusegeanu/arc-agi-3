from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.model import MemoryUid


@dataclass(frozen=True, slots=True)
class PreferenceProbe:
    outcome_a: MemoryUid
    outcome_b: MemoryUid
    context_bucket: int
    chosen_outcome: MemoryUid
    both_reachable: bool
    preference_influenced: bool


@dataclass(frozen=True, slots=True)
class PreferenceEvidence:
    preferred: MemoryUid
    other: MemoryUid
    context_bucket: int
    strength: float
    state: str
    clean_probe_count: int


class PreferenceEstimator:
    """Infer target-like preference only from causally clean comparable probes.

    A probe contributes only when both outcomes were represented as reachable before
    the choice and an already-active preference did not influence that choice. This
    prevents preference -> choice -> preference self-validation.
    """

    def __init__(self, *, support_threshold: int = 6, stable_margin: float = 0.30) -> None:
        self.support_threshold = int(support_threshold)
        self.stable_margin = float(stable_margin)
        self._probes: list[PreferenceProbe] = []

    def record_probe(
        self,
        *,
        outcome_a: MemoryUid,
        outcome_b: MemoryUid,
        context_bucket: int,
        chosen_outcome: MemoryUid,
        both_reachable: bool,
        preference_influenced: bool,
    ) -> bool:
        if outcome_a == outcome_b:
            return False
        if chosen_outcome not in {outcome_a, outcome_b}:
            return False
        if not both_reachable or preference_influenced:
            return False
        self._probes.append(
            PreferenceProbe(
                outcome_a,
                outcome_b,
                int(context_bucket),
                chosen_outcome,
                True,
                False,
            )
        )
        return True

    def evaluate(self) -> tuple[PreferenceEvidence, ...]:
        by_pair: dict[tuple[int, MemoryUid, MemoryUid], dict[MemoryUid, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for probe in self._probes:
            a, b = sorted((probe.outcome_a, probe.outcome_b))
            by_pair[(probe.context_bucket, a, b)][probe.chosen_outcome] += 1

        result: list[PreferenceEvidence] = []
        for (context, a, b), counts in by_pair.items():
            count_a = counts.get(a, 0)
            count_b = counts.get(b, 0)
            total = count_a + count_b
            if total == 0:
                continue
            if count_a >= count_b:
                preferred, other = a, b
                preferred_count, other_count = count_a, count_b
            else:
                preferred, other = b, a
                preferred_count, other_count = count_b, count_a
            strength = (preferred_count - other_count) / total
            if total < self.support_threshold:
                state = "CANDIDATE"
            else:
                state = "STABLE" if strength >= self.stable_margin else "PROBE_ONLY"
            result.append(
                PreferenceEvidence(
                    preferred,
                    other,
                    context,
                    float(strength),
                    state,
                    total,
                )
            )
        return tuple(result)
