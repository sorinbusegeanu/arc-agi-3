from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from v8.arena import EdgeRecord, NodeRecord
from v8.model import MemoryUid, RelationType


@dataclass(frozen=True, slots=True)
class StructuralCorrespondence:
    source_uid: MemoryUid
    target_uid: MemoryUid
    similarity_score: float
    mapping_size: int
    preserved_edges: int
    mismatched_edges: int
    epsilon_struct: float
    theta_struct: float
    evidence_watermark: int

    @property
    def admissible(self) -> bool:
        return self.mapping_size > 0 and self.epsilon_struct < self.theta_struct


class StructuralCorrespondenceEstimator:
    """Bounded radius-1 typed mapping between similarity candidates.

    This is deliberately not graph isomorphism.  It compares deterministic typed
    incident-edge partitions and supplies the explicit epsilon_struct gate required
    before held-out transfer testing.
    """

    _STRUCTURAL_RELATIONS = {
        int(RelationType.PROVENANCE),
        int(RelationType.EXPLAINS),
        int(RelationType.LEADS_TO),
        int(RelationType.CONTEXT_REFINES),
        int(RelationType.DEPENDS_ON),
        int(RelationType.ENABLES),
        int(RelationType.BLOCKS),
        int(RelationType.OUTCOME_EQUIVALENT),
    }

    def __init__(self, *, theta_struct: float = 0.50, min_similarity: float = 0.50) -> None:
        self.theta_struct = float(theta_struct)
        self.min_similarity = float(min_similarity)

    @classmethod
    def _descriptor(
        cls,
        uid: MemoryUid,
        edges: tuple[EdgeRecord, ...],
        by_uid: dict[MemoryUid, NodeRecord],
    ) -> Counter[tuple[int, int, int, int]]:
        descriptor: Counter[tuple[int, int, int, int]] = Counter()
        for edge in edges:
            relation = int(edge.relation_type)
            if relation not in cls._STRUCTURAL_RELATIONS:
                continue
            if edge.source_uid == uid:
                neighbor = by_uid.get(edge.target_uid)
                if neighbor is None:
                    continue
                descriptor[(1, relation, int(neighbor.level), int(neighbor.memory_type))] += max(
                    1, int(edge.support_count)
                )
            elif edge.target_uid == uid:
                neighbor = by_uid.get(edge.source_uid)
                if neighbor is None:
                    continue
                descriptor[(-1, relation, int(neighbor.level), int(neighbor.memory_type))] += max(
                    1, int(edge.support_count)
                )
        return descriptor

    @classmethod
    def _descriptors(
        cls,
        uids: set[MemoryUid],
        edges: tuple[EdgeRecord, ...],
        by_uid: dict[MemoryUid, NodeRecord],
    ) -> dict[MemoryUid, Counter[tuple[int, int, int, int]]]:
        descriptors = {uid: Counter() for uid in uids}
        for edge in edges:
            relation = int(edge.relation_type)
            if relation not in cls._STRUCTURAL_RELATIONS:
                continue
            weight = max(1, int(edge.support_count))
            source_descriptor = descriptors.get(edge.source_uid)
            if source_descriptor is not None:
                neighbor = by_uid.get(edge.target_uid)
                if neighbor is not None:
                    source_descriptor[
                        (1, relation, int(neighbor.level), int(neighbor.memory_type))
                    ] += weight
            target_descriptor = descriptors.get(edge.target_uid)
            if target_descriptor is not None:
                neighbor = by_uid.get(edge.source_uid)
                if neighbor is not None:
                    target_descriptor[
                        (-1, relation, int(neighbor.level), int(neighbor.memory_type))
                    ] += weight
        return descriptors

    @staticmethod
    def _error(
        left: Counter[tuple[int, int, int, int]],
        right: Counter[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, float]:
        source_total = sum(left.values())
        target_total = sum(right.values())
        if source_total <= 0 or target_total <= 0:
            return 0, max(source_total, target_total), 0, 1.0
        preserved = sum(min(count, right.get(key, 0)) for key, count in left.items())
        mismatched = max(0, source_total - preserved)
        mapping_size = sum(1 for key in left if right.get(key, 0) > 0)
        epsilon = mismatched / max(1, source_total)
        return preserved, mismatched, mapping_size, float(epsilon)

    def evaluate(
        self,
        nodes: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...],
        *,
        budget: int = 256,
    ) -> tuple[StructuralCorrespondence, ...]:
        by_uid = {row.uid: row for row in nodes}
        similarities = [
            edge
            for edge in edges
            if int(edge.relation_type) == int(RelationType.SIMILAR_TO)
            and float(edge.score) >= self.min_similarity
            and edge.source_uid in by_uid
            and edge.target_uid in by_uid
        ]
        similarities.sort(key=lambda edge: (-float(edge.score), edge.source_uid, edge.target_uid))
        selected = similarities[: max(0, int(budget))]
        descriptor_uids = {
            uid
            for edge in selected
            for uid in (edge.source_uid, edge.target_uid)
        }
        descriptor_cache = self._descriptors(descriptor_uids, edges, by_uid)
        result: list[StructuralCorrespondence] = []
        for edge in selected:
            left = descriptor_cache[edge.source_uid]
            right = descriptor_cache[edge.target_uid]
            preserved_lr, mismatched_lr, mapping_lr, epsilon_lr = self._error(left, right)
            preserved_rl, mismatched_rl, mapping_rl, epsilon_rl = self._error(right, left)
            preserved = min(preserved_lr, preserved_rl)
            mismatched = max(mismatched_lr, mismatched_rl)
            mapping_size = min(mapping_lr, mapping_rl)
            epsilon = max(epsilon_lr, epsilon_rl)
            result.append(
                StructuralCorrespondence(
                    edge.source_uid,
                    edge.target_uid,
                    float(edge.score),
                    int(mapping_size),
                    int(preserved),
                    int(mismatched),
                    float(epsilon),
                    self.theta_struct,
                    max(
                        int(edge.updated_watermark),
                        int(by_uid[edge.source_uid].updated_watermark),
                        int(by_uid[edge.target_uid].updated_watermark),
                    ),
                )
            )
        return tuple(result)
