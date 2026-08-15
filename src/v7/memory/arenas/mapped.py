from __future__ import annotations

from dataclasses import dataclass

from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.models import MemoryNode, MemoryScore


def find_row(values: memoryview, target: int) -> int:
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        current = int(values[mid])
        if current < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(values) and int(values[lo]) == target:
        return lo
    return -1


@dataclass(frozen=True, slots=True)
class MappedNodeColumns:
    memory_ids: memoryview
    levels: memoryview
    type_ids: memoryview
    created_generations: memoryview
    updated_generations: memoryview
    status_flags: memoryview
    support_counts: memoryview
    cognitive_states: memoryview
    validation_states: memoryview
    gate_ids: memoryview

    def get(self, memory_id: MemoryId) -> MemoryNode | None:
        row = find_row(self.memory_ids, int(memory_id))
        if row < 0:
            return None
        return MemoryNode(
            MemoryId(int(self.memory_ids[row])),
            MemoryLevel(int(self.levels[row])),
            int(self.type_ids[row]),
            GenerationId(int(self.created_generations[row])),
            GenerationId(int(self.updated_generations[row])),
            int(self.status_flags[row]),
            int(self.support_counts[row]),
            int(self.cognitive_states[row]),
            int(self.validation_states[row]),
            int(self.gate_ids[row]),
        )

    @property
    def count(self) -> int:
        return len(self.memory_ids)


@dataclass(frozen=True, slots=True)
class MappedScoreColumns:
    memory_ids: memoryview
    significance: memoryview
    prediction_error: memoryview
    learning_value: memoryview
    transfer_prior: memoryview
    explanatory_potential: memoryview
    future_option_delta: memoryview

    def get(self, memory_id: MemoryId) -> MemoryScore | None:
        row = find_row(self.memory_ids, int(memory_id))
        if row < 0:
            return None
        return MemoryScore(
            MemoryId(int(self.memory_ids[row])),
            float(self.significance[row]),
            float(self.prediction_error[row]),
            float(self.learning_value[row]),
            float(self.transfer_prior[row]),
            float(self.explanatory_potential[row]),
            float(self.future_option_delta[row]),
        )

    @property
    def count(self) -> int:
        return len(self.memory_ids)


@dataclass(frozen=True, slots=True)
class MappedPackedAdjacency:
    source_ids: memoryview
    relation_types: memoryview
    offsets: memoryview
    lengths: memoryview
    targets: memoryview

    def neighbors(
        self, source_id: MemoryId, relation_type: int
    ) -> tuple[MemoryId, ...]:
        row = find_row(self.source_ids, int(source_id))
        if row < 0:
            return ()
        while row > 0 and int(self.source_ids[row - 1]) == int(source_id):
            row -= 1
        while row < len(self.source_ids) and int(self.source_ids[row]) == int(source_id):
            relation = int(self.relation_types[row])
            if relation == int(relation_type):
                start = int(self.offsets[row])
                stop = start + int(self.lengths[row])
                return tuple(MemoryId(int(value)) for value in self.targets[start:stop])
            if relation > int(relation_type):
                break
            row += 1
        return ()


@dataclass(frozen=True, slots=True)
class MappedCompactMemoryArena:
    generation_id: GenerationId
    nodes: MappedNodeColumns
    scores: MappedScoreColumns
    adjacency: MappedPackedAdjacency
    owners: tuple[object, ...] = ()
