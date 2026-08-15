from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import EdgeRecord, NodeRecord
from v8.model import CognitiveState, MemoryUid, RelationType


@dataclass(frozen=True, slots=True)
class PruneCandidate:
    uid: MemoryUid
    protected_by_dependencies: bool


class PruningPlanner:
    """Select retirement candidates without deleting provenance or active dependencies."""

    def candidates(
        self,
        nodes: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...],
    ) -> tuple[PruneCandidate, ...]:
        incoming_active: dict[MemoryUid, int] = defaultdict(int)
        active = {row.uid for row in nodes if int(row.cognitive_state) in {int(CognitiveState.ACTIVE), int(CognitiveState.VALIDATED), int(CognitiveState.REACTIVATED)}}
        for edge in edges:
            if edge.source_uid in active and int(edge.relation_type) in {
                int(RelationType.EXPLAINS),
                int(RelationType.LEADS_TO),
                int(RelationType.CONTEXT_REFINES),
            }:
                incoming_active[edge.target_uid] += 1
        result = []
        for row in nodes:
            if int(row.cognitive_state) != int(CognitiveState.RETIRE_PENDING):
                continue
            result.append(PruneCandidate(row.uid, incoming_active.get(row.uid, 0) > 0))
        return tuple(result)
