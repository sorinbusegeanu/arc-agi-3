from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import EdgeRecord, NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryUid, RelationType


@dataclass(frozen=True, slots=True)
class PruneCandidate:
    uid: MemoryUid
    protected_by_dependencies: bool
    protected_by_evidence: bool = False
    has_semantic_replacement: bool = False
    safe_to_retire: bool = False


class PruningPlanner:
    """Apply the v8.2 semantic safe-forgetting criterion before retirement."""

    def candidates(
        self,
        nodes: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...],
        *,
        protected_evidence_uids: frozenset[MemoryUid] | set[MemoryUid] = frozenset(),
    ) -> tuple[PruneCandidate, ...]:
        active_states = {
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        active = {row.uid for row in nodes if int(row.cognitive_state) in active_states}
        required_by_active: dict[MemoryUid, int] = defaultdict(int)
        superseded_by_active: dict[MemoryUid, int] = defaultdict(int)
        for edge in edges:
            if edge.source_uid not in active:
                continue
            relation = int(edge.relation_type)
            if relation == int(RelationType.SUPERSEDES):
                superseded_by_active[edge.target_uid] += 1
            elif relation in {
                int(RelationType.EXPLAINS),
                int(RelationType.LEADS_TO),
                int(RelationType.CONTEXT_REFINES),
                int(RelationType.DEPENDS_ON),
            }:
                required_by_active[edge.target_uid] += 1

        evidence_protected = set(protected_evidence_uids)
        result = []
        for row in nodes:
            if int(row.cognitive_state) != int(CognitiveState.RETIRE_PENDING):
                continue
            dependency_protected = required_by_active.get(row.uid, 0) > 0
            evidence_required = row.uid in evidence_protected
            replacement = superseded_by_active.get(row.uid, 0) > 0
            # Detailed low/mid-level memories may be compressed only when an active
            # replacement explicitly preserves their supported structure.  Higher
            # level obsolete hypotheses can retire without a replacement if no
            # active dependency/evidence contract requires them.
            replacement_required = int(row.level) <= int(MemoryLevel.M4)
            semantic_safe = replacement or not replacement_required
            safe = not dependency_protected and not evidence_required and semantic_safe
            result.append(
                PruneCandidate(
                    row.uid,
                    bool(dependency_protected or evidence_required or not semantic_safe),
                    bool(evidence_required),
                    bool(replacement),
                    bool(safe),
                )
            )
        return tuple(result)
