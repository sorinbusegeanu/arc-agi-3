from __future__ import annotations

from array import array
from bisect import bisect_left
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.models import MemoryNode, MemoryScore


@dataclass(frozen=True, slots=True)
class NodeColumns:
    memory_ids: array
    levels: array
    type_ids: array
    created_generations: array
    updated_generations: array
    status_flags: array
    support_counts: array
    row_by_id: Mapping[MemoryId, int]

    @classmethod
    def build(cls, nodes: Mapping[MemoryId, MemoryNode]) -> "NodeColumns":
        ordered = sorted(nodes.values(), key=lambda item: int(item.memory_id))
        row_by_id = MappingProxyType(
            {node.memory_id: index for index, node in enumerate(ordered)}
        )
        return cls(
            memory_ids=array("Q", (int(node.memory_id) for node in ordered)),
            levels=array("B", (int(node.level) for node in ordered)),
            type_ids=array("I", (int(node.type_id) for node in ordered)),
            created_generations=array("Q", (int(node.created_generation) for node in ordered)),
            updated_generations=array("Q", (int(node.updated_generation) for node in ordered)),
            status_flags=array("Q", (int(node.status_flags) for node in ordered)),
            support_counts=array("q", (int(node.support_count) for node in ordered)),
            row_by_id=row_by_id,
        )

    def get(self, memory_id: MemoryId) -> MemoryNode | None:
        row = self.row_by_id.get(memory_id)
        if row is None:
            return None
        return MemoryNode(
            memory_id=MemoryId(self.memory_ids[row]),
            level=MemoryLevel(self.levels[row]),
            type_id=int(self.type_ids[row]),
            created_generation=GenerationId(self.created_generations[row]),
            updated_generation=GenerationId(self.updated_generations[row]),
            status_flags=int(self.status_flags[row]),
            support_count=int(self.support_counts[row]),
        )

    @property
    def count(self) -> int:
        return len(self.memory_ids)

    @property
    def payload_bytes(self) -> int:
        return sum(
            values.buffer_info()[1] * values.itemsize
            for values in (
                self.memory_ids,
                self.levels,
                self.type_ids,
                self.created_generations,
                self.updated_generations,
                self.status_flags,
                self.support_counts,
            )
        )


@dataclass(frozen=True, slots=True)
class ScoreColumns:
    memory_ids: array
    significance: array
    prediction_error: array
    learning_value: array
    transfer_prior: array
    explanatory_potential: array
    future_option_delta: array
    row_by_id: Mapping[MemoryId, int]

    @classmethod
    def build(cls, scores: Mapping[MemoryId, MemoryScore]) -> "ScoreColumns":
        ordered = sorted(scores.values(), key=lambda item: int(item.memory_id))
        row_by_id = MappingProxyType(
            {score.memory_id: index for index, score in enumerate(ordered)}
        )
        return cls(
            memory_ids=array("Q", (int(score.memory_id) for score in ordered)),
            significance=array("d", (float(score.significance) for score in ordered)),
            prediction_error=array("d", (float(score.prediction_error) for score in ordered)),
            learning_value=array("d", (float(score.learning_value) for score in ordered)),
            transfer_prior=array("d", (float(score.transfer_prior) for score in ordered)),
            explanatory_potential=array("d", (float(score.explanatory_potential) for score in ordered)),
            future_option_delta=array("d", (float(score.future_option_delta) for score in ordered)),
            row_by_id=row_by_id,
        )

    def get(self, memory_id: MemoryId) -> MemoryScore | None:
        row = self.row_by_id.get(memory_id)
        if row is None:
            return None
        return MemoryScore(
            memory_id=MemoryId(self.memory_ids[row]),
            significance=float(self.significance[row]),
            prediction_error=float(self.prediction_error[row]),
            learning_value=float(self.learning_value[row]),
            transfer_prior=float(self.transfer_prior[row]),
            explanatory_potential=float(self.explanatory_potential[row]),
            future_option_delta=float(self.future_option_delta[row]),
        )

    @property
    def count(self) -> int:
        return len(self.memory_ids)

    @property
    def payload_bytes(self) -> int:
        return sum(
            values.buffer_info()[1] * values.itemsize
            for values in (
                self.memory_ids,
                self.significance,
                self.prediction_error,
                self.learning_value,
                self.transfer_prior,
                self.explanatory_potential,
                self.future_option_delta,
            )
        )


@dataclass(frozen=True, slots=True)
class PackedAdjacency:
    source_ids: array
    relation_types: array
    offsets: array
    lengths: array
    targets: array

    @classmethod
    def build(
        cls,
        adjacency: Mapping[tuple[MemoryId, int], tuple[MemoryId, ...]],
    ) -> "PackedAdjacency":
        keys = sorted(adjacency, key=lambda key: (int(key[0]), int(key[1])))
        source_ids = array("Q")
        relation_types = array("I")
        offsets = array("Q")
        lengths = array("Q")
        targets = array("Q")
        for source_id, relation_type in keys:
            values = tuple(sorted(set(adjacency[(source_id, relation_type)]), key=int))
            source_ids.append(int(source_id))
            relation_types.append(int(relation_type))
            offsets.append(len(targets))
            lengths.append(len(values))
            targets.extend(int(value) for value in values)
        return cls(
            source_ids=source_ids,
            relation_types=relation_types,
            offsets=offsets,
            lengths=lengths,
            targets=targets,
        )

    def neighbors(self, source_id: MemoryId, relation_type: int) -> tuple[MemoryId, ...]:
        target_source = int(source_id)
        target_relation = int(relation_type)
        lo = bisect_left(self.source_ids, target_source)
        while lo < len(self.source_ids) and self.source_ids[lo] == target_source:
            relation = int(self.relation_types[lo])
            if relation == target_relation:
                start = int(self.offsets[lo])
                stop = start + int(self.lengths[lo])
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
        return sum(
            values.buffer_info()[1] * values.itemsize
            for values in (
                self.source_ids,
                self.relation_types,
                self.offsets,
                self.lengths,
                self.targets,
            )
        )


@dataclass(frozen=True, slots=True)
class CompactMemoryArena:
    generation_id: GenerationId
    nodes: NodeColumns
    scores: ScoreColumns
    adjacency: PackedAdjacency

    @classmethod
    def build(
        cls,
        *,
        generation_id: GenerationId,
        nodes: Mapping[MemoryId, MemoryNode],
        scores: Mapping[MemoryId, MemoryScore],
        adjacency: Mapping[tuple[MemoryId, int], tuple[MemoryId, ...]],
    ) -> "CompactMemoryArena":
        return cls(
            generation_id=generation_id,
            nodes=NodeColumns.build(nodes),
            scores=ScoreColumns.build(scores),
            adjacency=PackedAdjacency.build(adjacency),
        )

    @property
    def payload_bytes(self) -> int:
        return self.nodes.payload_bytes + self.scores.payload_bytes + self.adjacency.payload_bytes
