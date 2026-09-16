from __future__ import annotations

from typing import Any, Iterable

from v9.memory.identity import MemoryUid
from v9.memory.model import CanonicalNode, MemoryLevel
from v9.memory.relations import RelationEdge, RelationType
from v9.mutation.proposals import MutationKind, MutationProposal, MutationWrite, ProposalClass
from v9.mutation.read_sets import ReadDependency, ReadSet
from v9.runtime.publication import edge_ref, node_ref


PublicationRow = tuple[CanonicalNode, dict[str, Any], tuple[Any, ...]]


def _unique_parents(payload: dict[str, Any]) -> tuple[MemoryUid, ...]:
    seen: set[tuple[int, int]] = set()
    parents: list[MemoryUid] = []
    for raw_parent in payload.get("parents", []):
        if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
            continue
        key = (int(raw_parent[0]), int(raw_parent[1]))
        if key in seen:
            continue
        seen.add(key)
        parents.append(MemoryUid(*key))
    return tuple(parents)


def publication_row_dependency_cost(row: PublicationRow) -> int:
    _node, payload, _evidence = row
    return 1 + len(_unique_parents(payload))


def _publish_oversized_derivation_row(runtime: Any, row: PublicationRow) -> bool:
    node, raw_payload, evidence = row
    maximum_dependencies = int(runtime.config.scientific.maximum_read_set_size)
    parents = _unique_parents(raw_payload)
    if not parents:
        return runtime._publish_group((row,))
    if maximum_dependencies < 2:
        raise ValueError("maximum_read_set_size must be at least 2 for derivation provenance")

    payload = dict(raw_payload)
    payload["parents"] = [[uid.hi, uid.lo] for uid in parents]
    evidence_refs = tuple(sorted(set(evidence)))
    payload.setdefault("evidence_refs", [[uid.hi, uid.lo] for uid in evidence_refs])

    first_capacity = maximum_dependencies - 1
    chunks: list[tuple[MemoryUid, ...]] = [parents[:first_capacity]]
    remaining = parents[first_capacity:]
    for offset in range(0, len(remaining), maximum_dependencies):
        chunks.append(remaining[offset : offset + maximum_dependencies])

    proposals: list[MutationProposal] = []
    for index, parent_chunk in enumerate(chunks):
        writes: list[MutationWrite] = []
        dependencies: list[ReadDependency] = []
        target_partitions = {runtime.partitions.owner(node.uid)}
        if index == 0:
            writes.append(MutationWrite(node=node, payload=payload))
            node_reference = node_ref(node.uid)
            dependencies.append(ReadDependency(node_reference, runtime.graph.versions.get(node_reference)))

        for parent in parent_chunk:
            edge = RelationEdge(node.uid, RelationType.PROVENANCE, parent, evidence_refs)
            writes.append(MutationWrite(edge=edge))
            target_partitions.add(runtime.partitions.owner(parent))
            edge_reference = edge_ref(edge)
            dependencies.append(ReadDependency(edge_reference, runtime.graph.versions.get(edge_reference)))

        proposals.append(
            MutationProposal.build(
                MutationKind.UPSERT_NODE if index == 0 else MutationKind.UPSERT_EDGE,
                target_partitions=tuple(sorted(target_partitions)),
                read_set=ReadSet.build(tuple(dependencies), maximum_size=maximum_dependencies),
                evidence_refs=evidence_refs,
                causal_watermark=runtime._watermark,
                writes=tuple(writes),
                proposal_class=ProposalClass.ADDITIVE,
            )
        )

    runtime.telemetry["proposals"] += len(proposals)
    runtime.telemetry["cross_partition_transactions"] += sum(
        int(len(proposal.target_partitions) > 1) for proposal in proposals
    )
    previous_generation = runtime.graph.generation
    results = runtime.graph.publish_batch(tuple(proposals))
    first_accepted = False
    all_accepted = True
    for index, result in enumerate(results):
        outcome = result.outcome.value
        if outcome == "ACCEPTED":
            runtime.telemetry["accepted"] += 1
            if result.graph_generation == previous_generation:
                runtime.telemetry["canonical_reuse"] += 1
            previous_generation = result.graph_generation
            first_accepted = first_accepted or index == 0
        elif outcome == "STALE_READ_SET":
            runtime.telemetry["stale"] += 1
            runtime.telemetry["read_set_conflicts"] += 1
            all_accepted = False
        else:
            runtime.telemetry["rejected"] += 1
            all_accepted = False

    if first_accepted:
        if node.level >= MemoryLevel.M2:
            runtime.structural_index.add(node)
        if runtime.config.enable_lifecycle:
            runtime.lifecycle.observe(
                node.uid,
                support_delta=1,
                relevant_opportunity=True,
                watermark=runtime._watermark,
            )
    return all_accepted


def install_bounded_derivation_publication(runtime_cls: type) -> None:
    def _publish_derivation_rows(runtime: Any, publication_rows: Iterable[PublicationRow]) -> None:
        maximum_dependencies = int(runtime.config.scientific.maximum_read_set_size)
        maximum_nodes = 384
        batch: list[PublicationRow] = []
        dependency_budget = 0

        for row in publication_rows:
            row_cost = publication_row_dependency_cost(row)
            if row_cost > maximum_dependencies:
                if batch:
                    runtime._publish_group(tuple(batch))
                    batch = []
                    dependency_budget = 0
                _publish_oversized_derivation_row(runtime, row)
                continue
            if batch and (
                len(batch) >= maximum_nodes
                or dependency_budget + row_cost > maximum_dependencies
            ):
                runtime._publish_group(tuple(batch))
                batch = []
                dependency_budget = 0
            batch.append(row)
            dependency_budget += row_cost

        if batch:
            runtime._publish_group(tuple(batch))

    runtime_cls._publish_derivation_rows = _publish_derivation_rows
