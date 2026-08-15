from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from v8.arena import EdgeRecord, NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType


@dataclass(frozen=True, slots=True)
class NeighborhoodDescriptor:
    uid: MemoryUid
    level: int
    memory_type: int
    incoming_relations: tuple[tuple[int, int], ...]
    outgoing_relations: tuple[tuple[int, int], ...]
    neighbor_levels: tuple[tuple[int, int], ...]
    neighbor_types: tuple[tuple[int, int], ...]
    future_option_bucket: int
    consequence_bucket: int
    context_bucket: int
    version: int


@dataclass(frozen=True, slots=True)
class SimilarityEvidence:
    source_uid: MemoryUid
    target_uid: MemoryUid
    score: float
    relation_score: float
    level_type_score: float
    future_option_score: float
    consequence_score: float
    context_score: float
    watermark: int


def _hist_similarity(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> float:
    left, right = dict(a), dict(b)
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    intersection = sum(min(left.get(k, 0), right.get(k, 0)) for k in keys)
    union = sum(max(left.get(k, 0), right.get(k, 0)) for k in keys)
    return 1.0 if union == 0 else intersection / union


def _bucket(value: float, width: float = 0.25) -> int:
    return int(round(float(value) / width))


class BoundedNeighborhoodSimilarity:
    """Radius-1 typed graph similarity with deterministic bounded candidate search."""

    def __init__(self, *, max_candidates: int = 32, top_results: int = 4, threshold: float = 0.65) -> None:
        self.max_candidates = int(max_candidates)
        self.top_results = int(top_results)
        self.threshold = float(threshold)
        self.candidate_comparisons = 0

    @staticmethod
    def descriptors(nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> dict[MemoryUid, NeighborhoodDescriptor]:
        nodes = tuple(nodes)
        edges = tuple(edges)
        by_uid = {row.uid: row for row in nodes}
        incoming: dict[MemoryUid, Counter[int]] = defaultdict(Counter)
        outgoing: dict[MemoryUid, Counter[int]] = defaultdict(Counter)
        neighbor_levels: dict[MemoryUid, Counter[int]] = defaultdict(Counter)
        neighbor_types: dict[MemoryUid, Counter[int]] = defaultdict(Counter)
        for edge in edges:
            src = by_uid.get(edge.source_uid)
            dst = by_uid.get(edge.target_uid)
            if src is None or dst is None:
                continue
            outgoing[src.uid][int(edge.relation_type)] += 1
            incoming[dst.uid][int(edge.relation_type)] += 1
            neighbor_levels[src.uid][int(dst.level)] += 1
            neighbor_levels[dst.uid][int(src.level)] += 1
            neighbor_types[src.uid][int(dst.memory_type)] += 1
            neighbor_types[dst.uid][int(src.memory_type)] += 1
        result: dict[MemoryUid, NeighborhoodDescriptor] = {}
        for row in nodes:
            if int(row.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
                continue
            context = 0
            if int(row.level) == int(MemoryLevel.M3) and row.key_parts:
                context = int(row.key_parts[0])
            consequence = int(row.key_parts[0]) if int(row.level) == int(MemoryLevel.M4) and row.key_parts else 0
            result[row.uid] = NeighborhoodDescriptor(
                row.uid,
                int(row.level),
                int(row.memory_type),
                tuple(sorted(incoming[row.uid].items())),
                tuple(sorted(outgoing[row.uid].items())),
                tuple(sorted(neighbor_levels[row.uid].items())),
                tuple(sorted(neighbor_types[row.uid].items())),
                _bucket(row.future_option_delta),
                int(consequence),
                int(context),
                int(row.updated_watermark),
            )
        return result

    @staticmethod
    def score(a: NeighborhoodDescriptor, b: NeighborhoodDescriptor) -> SimilarityEvidence:
        relation = 0.5 * _hist_similarity(a.incoming_relations, b.incoming_relations) + 0.5 * _hist_similarity(a.outgoing_relations, b.outgoing_relations)
        level_type = 0.5 * _hist_similarity(a.neighbor_levels, b.neighbor_levels) + 0.5 * _hist_similarity(a.neighbor_types, b.neighbor_types)
        future = 1.0 / (1.0 + abs(a.future_option_bucket - b.future_option_bucket))
        consequence = 1.0 if a.consequence_bucket == b.consequence_bucket else 0.0
        context = 1.0 if a.context_bucket == b.context_bucket else 0.5
        active = [(0.35, relation), (0.25, level_type), (0.15, future), (0.15, consequence), (0.10, context)]
        total = sum(w * s for w, s in active) / sum(w for w, _ in active)
        source, target = sorted((a.uid, b.uid))
        return SimilarityEvidence(source, target, total, relation, level_type, future, consequence, context, min(a.version, b.version))

    def evaluate(self, nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> tuple[SimilarityEvidence, ...]:
        descriptors = self.descriptors(nodes, edges)
        buckets: dict[tuple[int, int, int], list[NeighborhoodDescriptor]] = defaultdict(list)
        for descriptor in descriptors.values():
            buckets[(descriptor.level, descriptor.memory_type, descriptor.future_option_bucket)].append(descriptor)
        results: list[SimilarityEvidence] = []
        seen: set[tuple[MemoryUid, MemoryUid]] = set()
        for descriptor in sorted(descriptors.values(), key=lambda d: d.uid):
            candidates: list[NeighborhoodDescriptor] = []
            for delta in (0, -1, 1):
                candidates.extend(buckets.get((descriptor.level, descriptor.memory_type, descriptor.future_option_bucket + delta), ()))
            candidates = [row for row in sorted(candidates, key=lambda d: d.uid) if row.uid != descriptor.uid][: self.max_candidates]
            scored: list[SimilarityEvidence] = []
            for candidate in candidates:
                pair = tuple(sorted((descriptor.uid, candidate.uid)))
                if pair in seen:
                    continue
                seen.add(pair)
                self.candidate_comparisons += 1
                evidence = self.score(descriptor, candidate)
                if evidence.score >= self.threshold:
                    scored.append(evidence)
            results.extend(sorted(scored, key=lambda row: (-row.score, row.target_uid))[: self.top_results])
        return tuple(results)
