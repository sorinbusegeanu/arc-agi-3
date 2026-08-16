from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid


@dataclass(frozen=True, slots=True)
class OutcomeClass:
    uid: MemoryUid
    members: tuple[MemoryUid, ...]
    descriptor: tuple[int, int]
    version: int
    support: int
    stability: float = 0.0
    context_consistency: float = 0.0
    within_class_diameter: float = 1.0
    predictive_interchangeability: float = 0.0
    persistent: bool = False


@dataclass(frozen=True, slots=True)
class OutcomeRevision:
    kind: str
    sources: tuple[MemoryUid, ...]
    target: MemoryUid
    descriptor: tuple[int, ...]
    version: int


class OutcomeEquivalenceEstimator:
    """Versioned M6 partition with explicit support/stability/diameter gates."""

    def __init__(
        self,
        *,
        min_support: int = 4,
        stability_threshold: float = 0.50,
        context_consistency_threshold: float = 0.50,
        max_diameter: float = 0.75,
        interchangeability_threshold: float = 0.25,
    ) -> None:
        self.min_support = int(min_support)
        self.stability_threshold = float(stability_threshold)
        self.context_consistency_threshold = float(context_consistency_threshold)
        self.max_diameter = float(max_diameter)
        self.interchangeability_threshold = float(interchangeability_threshold)
        self._version = 0
        self._classes: dict[tuple[int, int], OutcomeClass] = {}
        self._signature: tuple[tuple[int, int, tuple[MemoryUid, ...]], ...] = ()

    @staticmethod
    def _variant(row: NodeRecord) -> int:
        return int(row.key_parts[2]) if len(row.key_parts) >= 3 else 0

    def rebuild(self, rows: tuple[NodeRecord, ...]) -> tuple[OutcomeClass, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        admissible = {
            int(CognitiveState.CANDIDATE),
            int(CognitiveState.PROBATION),
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        for row in rows:
            if int(row.level) != int(MemoryLevel.M6) or int(row.memory_type) != int(MemoryType.OUTCOME):
                continue
            if len(row.key_parts) < 2 or int(row.cognitive_state) not in admissible:
                continue
            # The first two dimensions are the declared coarse consequence profile.
            # The third dimension, when present, is a fine contextual/consequence
            # variant used to test whether a stable persistent partition is justified.
            grouped[(int(row.key_parts[0]), int(row.key_parts[1]))].append(row)

        signature = tuple(
            sorted(
                (descriptor[0], descriptor[1], tuple(sorted(row.uid for row in members)))
                for descriptor, members in grouped.items()
            )
        )
        if signature != self._signature:
            self._version += 1
            self._signature = signature

        classes: dict[tuple[int, int], OutcomeClass] = {}
        for descriptor, members in grouped.items():
            uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, descriptor)
            support = sum(max(0, int(row.support_count)) for row in members)
            variants = [self._variant(row) for row in members]
            distinct_variants = max(1, len(set(variants)))
            stability = min(1.0, support / max(1.0, 2.0 * len(members)))
            dominant_support = max((max(0, int(row.support_count)) for row in members), default=0)
            context_consistency = dominant_support / max(1, support)
            if len(variants) <= 1:
                diameter = 0.0
            else:
                span = max(variants) - min(variants)
                normalizer = max(1, max(abs(value) for value in variants))
                diameter = min(1.0, abs(span) / normalizer)
            interchangeability = max(0.0, 1.0 - diameter)
            persistent = bool(
                support >= self.min_support
                and stability >= self.stability_threshold
                and context_consistency >= self.context_consistency_threshold
                and diameter <= self.max_diameter
                and interchangeability >= self.interchangeability_threshold
            )
            classes[descriptor] = OutcomeClass(
                uid,
                tuple(sorted(row.uid for row in members)),
                descriptor,
                self._version,
                support,
                float(stability),
                float(context_consistency),
                float(diameter),
                float(interchangeability),
                persistent,
            )
        self._classes = classes
        return tuple(classes.values())

    def classes(self) -> tuple[OutcomeClass, ...]:
        return tuple(self._classes.values())

    def merge_revision(self, outcome: OutcomeClass) -> OutcomeRevision | None:
        sources = tuple(uid for uid in outcome.members if uid != outcome.uid)
        if len(sources) < 2 or not outcome.persistent:
            return None
        self._version += 1
        return OutcomeRevision(
            "MERGE",
            sources,
            outcome.uid,
            outcome.descriptor,
            self._version,
        )

    def split_revision(self, source: OutcomeClass) -> tuple[OutcomeRevision, ...]:
        """Restore persistent fine members when a coarse class is invalidated."""
        if source.uid not in source.members:
            return ()
        members = tuple(uid for uid in source.members if uid != source.uid)
        result = []
        for member in members:
            self._version += 1
            result.append(
                OutcomeRevision("SPLIT", (source.uid,), member, source.descriptor, self._version)
            )
        return tuple(result)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self._version,
            "criterion": {
                "min_support": self.min_support,
                "stability_threshold": self.stability_threshold,
                "context_consistency_threshold": self.context_consistency_threshold,
                "max_diameter": self.max_diameter,
                "interchangeability_threshold": self.interchangeability_threshold,
            },
        }

    def load_state(self, state: dict[str, object] | None) -> None:
        if state:
            self._version = max(self._version, int(state.get("version", 0)))
