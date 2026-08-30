from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import EdgeRecord, NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType


@dataclass(frozen=True, slots=True)
class PruneCandidate:
    uid: MemoryUid
    protected_by_dependencies: bool
    protected_by_evidence: bool = False
    has_semantic_replacement: bool = False
    safe_to_retire: bool = False


_ACTIVE_STATES = {
    int(CognitiveState.ACTIVE),
    int(CognitiveState.VALIDATED),
    int(CognitiveState.REACTIVATED),
}


def _is_grounded_m1(row: NodeRecord) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) >= 4
    )


def _is_normalized_m1(row: NodeRecord) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) == 1
    )


def _valence_preserved(row: NodeRecord, replacements: tuple[NodeRecord, ...]) -> bool:
    """Require a replacement to preserve any learned directional primary valence."""
    source_weight = max(0.0, float(getattr(row, "primary_valence_weight", 0.0)))
    source_sum = float(getattr(row, "primary_valence_sum", 0.0))
    if source_weight <= 0.0 or abs(source_sum) <= 1e-12:
        return True
    for replacement in replacements:
        replacement_weight = max(
            0.0, float(getattr(replacement, "primary_valence_weight", 0.0))
        )
        replacement_sum = float(getattr(replacement, "primary_valence_sum", 0.0))
        if replacement_weight > 0.0 and source_sum * replacement_sum > 0.0:
            return True
    return False


class PruningPlanner:
    """Apply semantic safe-forgetting criteria before physical retirement.

    M0 may retire only behind an active M1+ semantic replacement.  Normalized M1
    may retire only behind an active M2+ replacement.  Grounded M1 remains a live
    control/causal anchor in v8.10: ActionArena and M7 causal admission still depend
    on it, so it can age to RETIRE_PENDING but is not physically retired yet.
    """

    def candidates(
        self,
        nodes: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...],
        *,
        protected_evidence_uids: frozenset[MemoryUid] | set[MemoryUid] = frozenset(),
        cancel_event=None,
    ) -> tuple[PruneCandidate, ...]:
        def cancelled() -> bool:
            return bool(cancel_event is not None and cancel_event.is_set())

        if cancelled():
            return ()
        by_uid = {row.uid: row for row in nodes}
        if cancelled():
            return ()
        active = {
            row.uid
            for row in nodes
            if int(row.cognitive_state) in _ACTIVE_STATES
        }
        required_by_active: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
        superseded_by_active: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
        for edge_index, edge in enumerate(edges):
            if edge_index % 256 == 0 and cancelled():
                return ()
            if edge.source_uid not in active:
                continue
            relation = int(edge.relation_type)
            if relation == int(RelationType.SUPERSEDES):
                superseded_by_active[edge.target_uid].add(edge.source_uid)
            elif relation in {
                int(RelationType.EXPLAINS),
                int(RelationType.LEADS_TO),
                int(RelationType.CONTEXT_REFINES),
                int(RelationType.DEPENDS_ON),
            }:
                required_by_active[edge.target_uid].add(edge.source_uid)

        evidence_protected = set(protected_evidence_uids)
        result = []
        for row_index, row in enumerate(nodes):
            if row_index % 256 == 0 and cancelled():
                return ()
            if int(row.cognitive_state) != int(CognitiveState.RETIRE_PENDING):
                continue

            superseder_uids = superseded_by_active.get(row.uid, set())
            replacements = tuple(
                by_uid[uid]
                for uid in sorted(superseder_uids)
                if uid in by_uid
            )

            # If the same active abstraction both EXPLAINS and explicitly SUPERSEDES
            # a lower memory, EXPLAINS is lineage/provenance, not a retirement veto.
            blocking_dependencies = required_by_active.get(row.uid, set()) - superseder_uids
            dependency_protected = bool(blocking_dependencies)
            evidence_required = row.uid in evidence_protected

            level = int(row.level)
            replacement = bool(replacements)
            if level == int(MemoryLevel.M0):
                eligible_replacements = tuple(
                    replacement_row
                    for replacement_row in replacements
                    if int(replacement_row.level) >= int(MemoryLevel.M1)
                )
                semantic_safe = bool(eligible_replacements) and _valence_preserved(
                    row, eligible_replacements
                )
            elif _is_grounded_m1(row):
                # Grounded M1 is still the canonical fallback-control record and a
                # causal dependency of M7.  Removing it would leave ActionArena and
                # strategy-causality state inconsistent.  It therefore cannot be
                # physically retired until control is represented independently.
                semantic_safe = False
            elif _is_normalized_m1(row):
                eligible_replacements = tuple(
                    replacement_row
                    for replacement_row in replacements
                    if int(replacement_row.level) >= int(MemoryLevel.M2)
                )
                semantic_safe = bool(eligible_replacements) and _valence_preserved(
                    row, eligible_replacements
                )
            elif level == int(MemoryLevel.M1):
                # Unknown M1 schema: fail closed.
                semantic_safe = False
            elif level <= int(MemoryLevel.M4):
                semantic_safe = replacement
            else:
                semantic_safe = True

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
        return () if cancelled() else tuple(result)
