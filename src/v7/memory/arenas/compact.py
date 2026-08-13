from __future__ import annotations

from array import array
from bisect import bisect_left
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.models import MemoryNode, MemoryScore


def _readonly(typecode: str, values: Iterable[int | float]) -> memoryview:
    return memoryview(array(typecode, values)).toreadonly()


@dataclass(frozen=True, slots=True)
class NodeColumns:
    memory_ids: memoryview
    levels: memoryview
    type_ids: memoryview
    created_generations: memoryview
    updated_generations: memoryview
    status_flags: memoryview
    support_counts: memoryview
    row_by_id: Mapping[MemoryId, int]

    @classmethod
    def build(cls, nodes: Mapping[MemoryId, MemoryNode]) -> "NodeColumns":
        ordered = sorted(nodes.values(), key=lambda item: int(item.memory_id))
        row_by_id = MappingProxyType({node.memory_id: index for index, node in enumerate(ordered)})
        return cls(
            memory_ids=_readonly("Q", (int(node.memory_id) for node in ordered)),
            levels=_readonly("B", (int(node.level) for node in ordered)),
            type_ids=_readonly("I", (int(node.type_id) for node in ordered)),
            created_generations=_readonly("Q", (int(node.created_generation) for node in ordered)),
            updated_generations=_readonly("Q", (int(node.updated_generation) for node in ordered)),
            status_flags=_readonly("Q", (int(node.status_flags) for node in ordered)),
            support_counts=_readonly("q", (int(node.support_count) for node in ordered)),
            row_by_id=row_by_id,
        )

    def get(self, memory_id: MemoryId) -> MemoryNode | None:
        row = self.row_by_id.get(memory_id)
        if row is None:
            return None
        return MemoryNode(MemoryId(self.memory_ids[row]), MemoryLevel(self.levels[row]), int(self.type_ids[row]), GenerationId(self.created_generations[row]), GenerationId(self.updated_generations[row]), int(self.status_flags[row]), int(self.support_counts[row]))

    @property
    def count(self) -> int:
        return len(self.memory_ids)

    @property
    def payload_bytes(self) -> int:
        return sum(values.nbytes for values in (self.memory_ids, self.levels, self.type_ids, self.created_generations, self.updated_generations, self.status_flags, self.support_counts))


@dataclass(frozen=True, slots=True)
class ScoreColumns:
    memory_ids: memoryview
    significance: memoryview
    prediction_error: memoryview
    learning_value: memoryview
    transfer_prior: memoryview
    explanatory_potential: memoryview
    future_option_delta: memoryview
    row_by_id: Mapping[MemoryId, int]

    @classmethod
    def build(cls, scores: Mapping[MemoryId, MemoryScore]) -> "ScoreColumns":
        ordered = sorted(scores.values(), key=lambda item: int(item.memory_id))
        row_by_id = MappingProxyType({score.memory_id: index for index, score in enumerate(ordered)})
        return cls(
            memory_ids=_readonly("Q", (int(score.memory_id) for score in ordered)),
            significance=_readonly("d", (float(score.significance) for score in ordered)),
            prediction_error=_readonly("d", (float(score.prediction_error) for score in ordered)),
            learning_value=_readonly("d", (float(score.learning_value) for score in ordered)),
            transfer_prior=_readonly("d", (float(score.transfer_prior) for score in ordered)),
            explanatory_potential=_readonly("d", (float(score.explanatory_potential) for score in ordered)),
            future_option_delta=_readonly("d", (float(score.future_option_delta) for score in ordered)),
            row_by_id=row_by_id,
        )

    def get(self, memory_id: MemoryId) -> MemoryScore | None:
        row = self.row_by_id.get(memory_id)
        if row is None:
            return None
        return MemoryScore(memory_id=MemoryId(self.memory_ids[row]), significance=float(self.significance[row]), prediction_error=float(self.prediction_error[row]), learning_value=float(self.learning_value[row]), transfer_prior=float(self.transfer_prior[row]), explanatory_potential=float(self.explanatory_potential[row]), future_option_delta=float(self.future_option_delta[row]))

    @property
    def count(self) -> int:
        return len(self.memory_ids)

    @property
    def payload_bytes(self) -> int:
        return sum(values.nbytes for values in (self.memory_ids, self.significance, self.prediction_error, self.learning_value, self.transfer_prior, self.explanatory_potential, self.future_option_delta))


@dataclass(frozen=True, slots=True)
class PackedAdjacency:
    source_ids: memoryview
    relation_types: memoryview
    offsets: memoryview
    lengths: memoryview
    targets: memoryview

    @classmethod
    def build(cls, adjacency: Mapping[tuple[MemoryId, int], tuple[MemoryId, ...]]) -> "PackedAdjacency":
        keys = sorted(adjacency, key=lambda key: (int(key[0]), int(key[1])))
        source_ids, relation_types, offsets, lengths, targets = array("Q"), array("I"), array("Q"), array("Q"), array("Q")
        for source_id, relation_type in keys:
            values = tuple(sorted(set(adjacency[(source_id, relation_type)]), key=int))
            source_ids.append(int(source_id)); relation_types.append(int(relation_type)); offsets.append(len(targets)); lengths.append(len(values)); targets.extend(int(value) for value in values)
        return cls(memoryview(source_ids).toreadonly(), memoryview(relation_types).toreadonly(), memoryview(offsets).toreadonly(), memoryview(lengths).toreadonly(), memoryview(targets).toreadonly())

    def neighbors(self, source_id: MemoryId, relation_type: int) -> tuple[MemoryId, ...]:
        target_source, target_relation = int(source_id), int(relation_type)
        lo = bisect_left(self.source_ids, target_source)
        while lo < len(self.source_ids) and self.source_ids[lo] == target_source:
            relation = int(self.relation_types[lo])
            if relation == target_relation:
                start = int(self.offsets[lo]); stop = start + int(self.lengths[lo])
                return tuple(MemoryId(value) for value in self.targets[start:stop])
            if relation > target_relation:
                break
            lo += 1
        return ()

    @property
    def edge_group_count(self) -> int:
        return len(self.source_ids)

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def payload_bytes(self) -> int:
        return sum(values.nbytes for values in (self.source_ids, self.relation_types, self.offsets, self.lengths, self.targets))


@dataclass(frozen=True, slots=True)
class CompactMemoryArena:
    generation_id: GenerationId
    nodes: NodeColumns
    scores: ScoreColumns
    adjacency: PackedAdjacency

    @classmethod
    def build(cls, *, generation_id: GenerationId, nodes: Mapping[MemoryId, MemoryNode], scores: Mapping[MemoryId, MemoryScore], adjacency: Mapping[tuple[MemoryId, int], tuple[MemoryId, ...]]) -> "CompactMemoryArena":
        return cls(generation_id, NodeColumns.build(nodes), ScoreColumns.build(scores), PackedAdjacency.build(adjacency))

    @classmethod
    def build_incremental(
        cls,
        *,
        generation_id: GenerationId,
        nodes: Mapping[MemoryId, MemoryNode],
        scores: Mapping[MemoryId, MemoryScore],
        adjacency: Mapping[tuple[MemoryId, int], tuple[MemoryId, ...]],
        previous: "CompactMemoryArena | None" = None,
        nodes_dirty: bool = True,
        scores_dirty: bool = True,
        adjacency_dirty: bool = True,
    ) -> "CompactMemoryArena":
        if previous is None:
            return cls.build(generation_id=generation_id, nodes=nodes, scores=scores, adjacency=adjacency)
        return cls(
            generation_id=generation_id,
            nodes=NodeColumns.build(nodes) if nodes_dirty else previous.nodes,
            scores=ScoreColumns.build(scores) if scores_dirty else previous.scores,
            adjacency=PackedAdjacency.build(adjacency) if adjacency_dirty else previous.adjacency,
        )

    @property
    def payload_bytes(self) -> int:
        return self.nodes.payload_bytes + self.scores.payload_bytes + self.adjacency.payload_bytes
