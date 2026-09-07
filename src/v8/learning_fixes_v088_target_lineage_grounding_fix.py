from __future__ import annotations

from collections import Counter, deque

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.similarity import NeighborhoodDescriptor


_INSTALLED = False
_MIN_ROLE_CARRIERS = 2
_ROLE_TYPES = {int(MemoryType.ROLE), int(MemoryType.CONTEXTUAL_ROLE)}


def _role_lineage_rows(
    root_uid: MemoryUid,
    by_uid: dict[MemoryUid, object],
    edges: tuple[object, ...],
    *,
    max_depth: int = 3,
) -> tuple[object, ...]:
    """Find real M3 role nodes below a transfer candidate through EXPLAINS lineage."""
    outgoing: dict[MemoryUid, set[MemoryUid]] = {}
    for edge in edges:
        if int(edge.relation_type) != int(RelationType.EXPLAINS):
            continue
        outgoing.setdefault(edge.source_uid, set()).add(edge.target_uid)

    found: dict[MemoryUid, object] = {}
    queue = deque([(root_uid, 0)])
    visited = {root_uid}
    while queue:
        uid, depth = queue.popleft()
        row = by_uid.get(uid)
        if (
            row is not None
            and int(row.level) == int(MemoryLevel.M3)
            and int(row.memory_type) in _ROLE_TYPES
        ):
            found[uid] = row
        if depth >= max(0, int(max_depth)):
            continue
        for child in sorted(outgoing.get(uid, ())):
            if child in visited:
                continue
            visited.add(child)
            queue.append((child, depth + 1))
    return tuple(found[uid] for uid in sorted(found))


def _normalized_role_reference(row) -> dict[str, object]:
    """Represent a mature role by the minimum structural motif that forms a role."""
    count = _MIN_ROLE_CARRIERS
    descriptor = NeighborhoodDescriptor(
        uid=row.uid,
        level=int(MemoryLevel.M3),
        memory_type=int(row.memory_type),
        incoming_relations=(),
        outgoing_relations=((int(RelationType.EXPLAINS), count),),
        neighbor_levels=((int(MemoryLevel.M3), count),),
        neighbor_types=((int(MemoryType.CARRIER), count),),
        dependency_signature=0,
        enable_block_signature=0,
        future_option_bucket=0,
        consequence_bucket=0,
        context_bucket=0,
        descriptor_version=max(1, int(getattr(row, "updated_watermark", 0))),
    )
    structural_counter: Counter[tuple[int, int, int, int]] = Counter(
        {
            (
                1,
                int(RelationType.EXPLAINS),
                int(MemoryLevel.M3),
                int(MemoryType.CARRIER),
            ): count
        }
    )
    return {
        "uid": row.uid.hex(),
        "descriptor": descriptor,
        "structural_counter": structural_counter,
        "lineage_role_uid": row.uid.hex(),
        "grounding_reference_kind": "normalized_m3_role_lineage",
    }


def _install_target_lineage_grounding() -> None:
    from v8 import learning_fixes_v088 as learning
    from v8 import learning_fixes_v088_target_grounding_fix as grounding

    current_transfer_execution = learning._transfer_execution_evidence

    def transfer_execution_evidence(
        read_view,
        nodes,
        edges,
        candidate,
        source_trajectories,
        *,
        target_game_hash: int,
    ):
        result = current_transfer_execution(
            read_view,
            nodes,
            edges,
            candidate,
            source_trajectories,
            target_game_hash=target_game_hash,
        )
        if result.get("correspondence_conditioned_mapping") is True:
            return result

        target_hash = int(target_game_hash)
        template = grounding._PENDING_TARGET_TEMPLATES.get(target_hash)
        if not isinstance(template, dict):
            return result

        by_uid = getattr(read_view, "_node_by_uid", None)
        if not isinstance(by_uid, dict):
            by_uid = {row.uid: row for row in nodes}
        graph_edges = tuple(edges or ())

        role_rows: dict[MemoryUid, object] = {}
        for root_uid in (candidate.uid, candidate.correspondence_uid):
            for role in _role_lineage_rows(root_uid, by_uid, graph_edges):
                role_rows[role.uid] = role

        if not role_rows:
            grounding._PENDING_TARGET_STRUCTURES.pop(target_hash, None)
            grounding._PENDING_TARGET_TEMPLATES.pop(target_hash, None)
            rejected = dict(result)
            reason = "no_m3_role_lineage_for_transfer_candidate"
            rejected["resolution_status"] = reason
            rejected["failure_reason"] = reason
            rejected["lower_level_resolution_failures"] = [reason]
            return rejected

        references = tuple(
            _normalized_role_reference(role_rows[uid]) for uid in sorted(role_rows)
        )
        aligned_template = dict(template)
        aligned_template.update(
            {
                "m4_attribution_uid": candidate.uid.hex(),
                "grounding_reference_level": int(MemoryLevel.M3),
                "grounding_reference_kind": "normalized_m3_role_lineage",
                "grounding_reference_uids": [
                    str(reference["uid"]) for reference in references
                ],
                "grounding_rule": "production_m3_role_lineage_minimum_carrier_motif",
            }
        )
        grounding._PENDING_TARGET_STRUCTURES[target_hash] = references
        grounding._PENDING_TARGET_TEMPLATES[target_hash] = aligned_template

        pending = dict(result)
        pending["target_grounding_template"] = aligned_template
        pending["resolution_status"] = "pending_target_local_m3_role_grounding"
        pending["failure_reason"] = "pending_target_local_m3_role_grounding"
        pending["lower_level_resolution_failures"] = [
            "pending_target_local_m3_role_grounding"
        ]
        return pending

    learning._transfer_execution_evidence = transfer_execution_evidence


def install_target_lineage_grounding_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_target_lineage_grounding()
    _INSTALLED = True
