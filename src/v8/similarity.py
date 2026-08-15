from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from v8.arena import EdgeRecord, NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64


@dataclass(frozen=True, slots=True)
class NeighborhoodDescriptor:
    uid: MemoryUid
    level: int
    memory_type: int
    incoming_relations: tuple[tuple[int, int], ...]
    outgoing_relations: tuple[tuple[int, int], ...]
    neighbor_levels: tuple[tuple[int, int], ...]
    neighbor_types: tuple[tuple[int, int], ...]
    dependency_signature: int
    enable_block_signature: int
    future_option_bucket: int
    consequence_bucket: int
    context_bucket: int
    descriptor_version: int


@dataclass(frozen=True, slots=True)
class SimilarityEvidence:
    source_uid: MemoryUid
    target_uid: MemoryUid
    score: float
    relation_score: float
    level_type_score: float
    dependency_score: float | None
    enable_block_score: float | None
    future_option_score: float
    consequence_score: float | None
    context_score: float | None
    evidence_watermark: int


def _hist_similarity(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> float:
    left, right = dict(a), dict(b)
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    intersection = sum(min(left.get(k, 0), right.get(k, 0)) for k in keys)
    union = sum(max(left.get(k, 0), right.get(k, 0)) for k in keys)
    return 1.0 if union == 0 else intersection / union


def _bucket(value: float, width: float = 0.25) -> int:
    return int(round(float(value) / float(width)))


def _signature(parts: Iterable[int], *, person: bytes) -> int:
    values = tuple(int(v) for v in parts)
    return 0 if not values else stable_u64(*values, person=person)


class BoundedNeighborhoodSimilarity:
    """Incremental radius-1 typed graph similarity over M3 roles and M4 concepts.

    Canonical identity is never changed. The service only returns bounded
    structural correspondence candidates for later SIMILAR_TO reduction and
    held-out transfer testing.
    """

    def __init__(
        self,
        *,
        max_candidates: int = 32,
        top_results: int = 4,
        threshold: float = 0.65,
    ) -> None:
        if max_candidates <= 0 or top_results <= 0:
            raise ValueError("similarity candidate/result budgets must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("similarity threshold must be in [0,1]")
        self.max_candidates = int(max_candidates)
        self.top_results = int(top_results)
        self.threshold = float(threshold)
        self.candidate_comparisons = 0
        self.processed_descriptors = 0
        self._processed_versions: dict[MemoryUid, int] = {}

    def set_budget(self, max_candidates: int) -> None:
        self.max_candidates = max(1, int(max_candidates))

    @staticmethod
    def descriptors(
        nodes: Iterable[NodeRecord],
        edges: Iterable[EdgeRecord],
    ) -> dict[MemoryUid, NeighborhoodDescriptor]:
        nodes = tuple(nodes)
        edges = tuple(edges)
        by_uid = {row.uid: row for row in nodes}
        eligible = {
            row.uid
            for row in nodes
            if (
                int(row.level) == int(MemoryLevel.M4)
                or (
                    int(row.level) == int(MemoryLevel.M3)
                    and int(row.memory_type)
                    in {int(MemoryType.ROLE), int(MemoryType.CONTEXTUAL_ROLE)}
                )
            )
        }
        incoming: dict[MemoryUid, Counter[int]] = defaultdict(Counter)
        outgoing: dict[MemoryUid, Counter[int]] = defaultdict(Counter)
        neighbor_levels: dict[MemoryUid, Counter[int]] = defaultdict(Counter)
        neighbor_types: dict[MemoryUid, Counter[int]] = defaultdict(Counter)
        dependencies: dict[MemoryUid, list[int]] = defaultdict(list)
        enable_block: dict[MemoryUid, list[int]] = defaultdict(list)
        consequence_neighbors: dict[MemoryUid, list[int]] = defaultdict(list)
        versions: dict[MemoryUid, int] = {
            row.uid: int(row.updated_watermark) for row in nodes if row.uid in eligible
        }

        for edge in edges:
            src = by_uid.get(edge.source_uid)
            dst = by_uid.get(edge.target_uid)
            if src is None or dst is None:
                continue
            relation = int(edge.relation_type)
            if src.uid in eligible:
                outgoing[src.uid][relation] += 1
                neighbor_levels[src.uid][int(dst.level)] += 1
                neighbor_types[src.uid][int(dst.memory_type)] += 1
                versions[src.uid] = max(
                    versions[src.uid], int(edge.updated_watermark), int(dst.updated_watermark)
                )
                if relation == int(RelationType.DEPENDS_ON):
                    dependencies[src.uid].extend((relation, int(dst.level), int(dst.memory_type)))
                if relation in {int(RelationType.ENABLES), int(RelationType.BLOCKS)}:
                    enable_block[src.uid].extend((relation, int(dst.level), int(dst.memory_type)))
                if int(dst.level) >= int(MemoryLevel.M5):
                    consequence_neighbors[src.uid].append(int(dst.fingerprint))
            if dst.uid in eligible:
                incoming[dst.uid][relation] += 1
                neighbor_levels[dst.uid][int(src.level)] += 1
                neighbor_types[dst.uid][int(src.memory_type)] += 1
                versions[dst.uid] = max(
                    versions[dst.uid], int(edge.updated_watermark), int(src.updated_watermark)
                )
                if relation == int(RelationType.DEPENDS_ON):
                    dependencies[dst.uid].extend((relation, int(src.level), int(src.memory_type)))
                if relation in {int(RelationType.ENABLES), int(RelationType.BLOCKS)}:
                    enable_block[dst.uid].extend((relation, int(src.level), int(src.memory_type)))
                if int(src.level) >= int(MemoryLevel.M5):
                    consequence_neighbors[dst.uid].append(int(src.fingerprint))

        result: dict[MemoryUid, NeighborhoodDescriptor] = {}
        for row in nodes:
            if row.uid not in eligible:
                continue
            context = 0
            if int(row.memory_type) == int(MemoryType.CONTEXTUAL_ROLE) and row.key_parts:
                context = int(row.key_parts[0])
            consequence = _signature(
                sorted(consequence_neighbors[row.uid]), person=b"v8-sim-conseq"
            )
            result[row.uid] = NeighborhoodDescriptor(
                uid=row.uid,
                level=int(row.level),
                memory_type=int(row.memory_type),
                incoming_relations=tuple(sorted(incoming[row.uid].items())),
                outgoing_relations=tuple(sorted(outgoing[row.uid].items())),
                neighbor_levels=tuple(sorted(neighbor_levels[row.uid].items())),
                neighbor_types=tuple(sorted(neighbor_types[row.uid].items())),
                dependency_signature=_signature(
                    sorted(dependencies[row.uid]), person=b"v8-sim-dep"
                ),
                enable_block_signature=_signature(
                    sorted(enable_block[row.uid]), person=b"v8-sim-enblk"
                ),
                future_option_bucket=_bucket(row.future_option_delta),
                consequence_bucket=int(consequence),
                context_bucket=int(context),
                descriptor_version=int(versions[row.uid]),
            )
        return result

    @staticmethod
    def score(a: NeighborhoodDescriptor, b: NeighborhoodDescriptor) -> SimilarityEvidence:
        relation = 0.5 * _hist_similarity(
            a.incoming_relations, b.incoming_relations
        ) + 0.5 * _hist_similarity(a.outgoing_relations, b.outgoing_relations)
        level_type = 0.5 * _hist_similarity(
            a.neighbor_levels, b.neighbor_levels
        ) + 0.5 * _hist_similarity(a.neighbor_types, b.neighbor_types)
        dependency = None
        if a.dependency_signature and b.dependency_signature:
            dependency = 1.0 if a.dependency_signature == b.dependency_signature else 0.0
        enable = None
        if a.enable_block_signature and b.enable_block_signature:
            enable = 1.0 if a.enable_block_signature == b.enable_block_signature else 0.0
        future = 1.0 / (1.0 + abs(a.future_option_bucket - b.future_option_bucket))
        consequence = None
        if a.consequence_bucket and b.consequence_bucket:
            consequence = 1.0 if a.consequence_bucket == b.consequence_bucket else 0.0
        context = None
        if a.context_bucket and b.context_bucket:
            context = 1.0 if a.context_bucket == b.context_bucket else 0.0

        components: list[tuple[float, float]] = [
            (0.30, relation),
            (0.20, level_type),
            (0.15, future),
        ]
        if context is not None:
            components.append((0.10, context))
        if dependency is not None:
            components.append((0.10, dependency))
        if enable is not None:
            components.append((0.05, enable))
        if consequence is not None:
            components.append((0.10, consequence))
        total = sum(weight * value for weight, value in components) / sum(
            weight for weight, _ in components
        )
        source, target = sorted((a.uid, b.uid))
        return SimilarityEvidence(
            source_uid=source,
            target_uid=target,
            score=float(total),
            relation_score=float(relation),
            level_type_score=float(level_type),
            dependency_score=dependency,
            enable_block_score=enable,
            future_option_score=float(future),
            consequence_score=consequence,
            context_score=context,
            evidence_watermark=max(
                int(a.descriptor_version), int(b.descriptor_version)
            ),
        )

    @staticmethod
    def _relation_bucket(descriptor: NeighborhoodDescriptor) -> int:
        flat: list[int] = []
        for relation, count in (
            descriptor.incoming_relations + descriptor.outgoing_relations
        ):
            flat.extend((int(relation), min(7, int(count))))
        return _signature(flat, person=b"v8-sim-rel") & 0xF

    def evaluate(
        self,
        nodes: Iterable[NodeRecord],
        edges: Iterable[EdgeRecord],
    ) -> tuple[SimilarityEvidence, ...]:
        descriptors = self.descriptors(nodes, edges)
        index: dict[
            tuple[int, int, int, int], list[NeighborhoodDescriptor]
        ] = defaultdict(list)
        fallback: dict[tuple[int, int], list[NeighborhoodDescriptor]] = defaultdict(list)
        for descriptor in sorted(descriptors.values(), key=lambda item: item.uid):
            index[
                (
                    descriptor.level,
                    descriptor.memory_type,
                    self._relation_bucket(descriptor),
                    descriptor.future_option_bucket,
                )
            ].append(descriptor)
            fallback[(descriptor.level, descriptor.memory_type)].append(descriptor)

        dirty = [
            descriptor
            for descriptor in sorted(descriptors.values(), key=lambda item: item.uid)
            if descriptor.descriptor_version
            > self._processed_versions.get(descriptor.uid, -1)
        ]
        results: dict[tuple[MemoryUid, MemoryUid], SimilarityEvidence] = {}
        for descriptor in dirty:
            candidates: list[NeighborhoodDescriptor] = []
            relation_bucket = self._relation_bucket(descriptor)
            for future_delta in (0, -1, 1):
                candidates.extend(
                    index.get(
                        (
                            descriptor.level,
                            descriptor.memory_type,
                            relation_bucket,
                            descriptor.future_option_bucket + future_delta,
                        ),
                        (),
                    )
                )
            if len(candidates) < self.max_candidates:
                candidates.extend(
                    fallback.get((descriptor.level, descriptor.memory_type), ())
                )
            unique_candidates = {
                candidate.uid: candidate
                for candidate in candidates
                if candidate.uid != descriptor.uid
            }
            scored: list[SimilarityEvidence] = []
            for candidate in sorted(
                unique_candidates.values(), key=lambda item: item.uid
            )[: self.max_candidates]:
                self.candidate_comparisons += 1
                evidence = self.score(descriptor, candidate)
                if evidence.score >= self.threshold:
                    scored.append(evidence)
            for evidence in sorted(
                scored,
                key=lambda item: (-item.score, item.source_uid, item.target_uid),
            )[: self.top_results]:
                results[(evidence.source_uid, evidence.target_uid)] = evidence
            self._processed_versions[descriptor.uid] = descriptor.descriptor_version
            self.processed_descriptors += 1
        return tuple(
            results[key]
            for key in sorted(results, key=lambda pair: (pair[0], pair[1]))
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "processed_versions": [
                {
                    "hi": uid.hi,
                    "lo": uid.lo,
                    "descriptor_version": version,
                }
                for uid, version in sorted(self._processed_versions.items())
            ],
            "candidate_comparisons": self.candidate_comparisons,
            "processed_descriptors": self.processed_descriptors,
        }

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        for raw in state.get("processed_versions", []):
            if not isinstance(raw, dict):
                continue
            uid = MemoryUid(int(raw.get("hi", 0)), int(raw.get("lo", 0)))
            self._processed_versions[uid] = max(
                self._processed_versions.get(uid, -1),
                int(raw.get("descriptor_version", 0)),
            )
        self.candidate_comparisons = max(
            self.candidate_comparisons,
            int(state.get("candidate_comparisons", 0)),
        )
        self.processed_descriptors = max(
            self.processed_descriptors,
            int(state.get("processed_descriptors", 0)),
        )
