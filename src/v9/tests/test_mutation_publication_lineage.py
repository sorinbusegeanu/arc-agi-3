from __future__ import annotations

from v9.memory import CanonicalNode, MemoryLevel, MemoryType, MemoryUid
from v9.memory.relations import EdgeAuthority
from v9.mutation import (
    ContextScope, LineageAwareDependencyEdge, LineageContextOverlay, LineageStore,
    LineageUid, MutationKind, MutationOutcome, MutationProposal, MutationWrite,
    ObjectRef, ReadDependency, ReadSet,
)
from v9.runtime.publication import CanonicalGraph, node_ref
from v9.runtime.reducers import PartitionReducer
from v9.memory.relations import RelationEdge, RelationType


def _proposal(graph: CanonicalGraph, node: CanonicalNode, version: int, *, watermark: int = 1) -> MutationProposal:
    ref = node_ref(node.uid)
    return MutationProposal.build(MutationKind.UPSERT_NODE, target_partitions=(node.uid.shard(graph.partition_count),), read_set=ReadSet.build((ReadDependency(ref, version),), maximum_size=8), evidence_refs=(), causal_watermark=watermark, writes=(MutationWrite(node=node, payload={"value": version}),))


def test_read_set_versions_not_global_generation_control_staleness() -> None:
    graph = CanonicalGraph(2)
    left = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (1,), 1)
    right = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (2,), 1)
    assert graph.publish(_proposal(graph, left, 0)).outcome is MutationOutcome.ACCEPTED
    assert graph.publish(_proposal(graph, right, 0)).outcome is MutationOutcome.ACCEPTED
    assert graph.publish(_proposal(graph, left, 0, watermark=2)).outcome is MutationOutcome.STALE_READ_SET


def test_read_views_are_immutable_cuts() -> None:
    graph = CanonicalGraph(1)
    node = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (1,), 1)
    graph.publish(_proposal(graph, node, 0))
    cut = graph.read_view()
    try:
        cut.nodes[node.uid] = node
    except TypeError:
        pass
    else:
        raise AssertionError("read view mapping must be immutable")


def test_dependency_scoped_suspension_preserves_independently_supported_child() -> None:
    store = LineageStore()
    parent, supported, unsupported = (MemoryUid.derive("node", value) for value in range(3))
    lineage, context = LineageUid(1), ContextScope(2)
    store.dependencies[(parent, supported, lineage, context)] = LineageAwareDependencyEdge(parent, supported, lineage, context, independent_support=1)
    store.dependencies[(parent, unsupported, lineage, context)] = LineageAwareDependencyEdge(parent, unsupported, lineage, context)
    store.suspend_parent(parent, lineage, context)
    assert store.dependencies[(parent, supported, lineage, context)].authority is EdgeAuthority.SUSPENDED
    assert (supported, lineage, context) not in store.overlays
    assert store.overlays[(unsupported, lineage, context)].lifecycle.name == "PROBATION"


def test_canonical_fork_occurs_only_on_structural_divergence() -> None:
    assert not LineageStore.canonical_fork_required((1, 2), (1, 2))
    assert LineageStore.canonical_fork_required((1, 2), (1, 3))


def test_reducer_order_is_deterministic_across_submission_schedules() -> None:
    nodes = tuple(CanonicalNode.build(MemoryLevel.M2, MemoryType.FAMILY, (value,), 1) for value in range(6))

    def run(order: tuple[int, ...]):
        graph = CanonicalGraph(1)
        reducer = PartitionReducer(0, graph, capacity=8)
        proposals = tuple(_proposal(graph, node, 0, watermark=index % 2 + 1) for index, node in enumerate(nodes))
        for index in order:
            assert reducer.submit(proposals[index])
        reducer.drain()
        return graph.state_dict()

    left = run((0, 1, 2, 3, 4, 5))
    right = run((5, 3, 1, 4, 2, 0))
    assert left == right


def test_cross_partition_invalid_proposal_is_all_or_nothing() -> None:
    graph = CanonicalGraph(2)
    source = next(MemoryUid.derive("source", value) for value in range(100) if MemoryUid.derive("source", value).shard(2) == 0)
    target = next(MemoryUid.derive("target", value) for value in range(100) if MemoryUid.derive("target", value).shard(2) == 1)
    edge = RelationEdge(source, RelationType.DEPENDS_ON, target)
    proposal = MutationProposal.build(MutationKind.UPSERT_EDGE, target_partitions=(0,), read_set=ReadSet.build((), maximum_size=1), evidence_refs=(), causal_watermark=1, writes=(MutationWrite(edge=edge),))
    assert graph.publish(proposal).outcome is MutationOutcome.INVALID
    assert not graph.edges


def test_lineage_and_object_versions_round_trip() -> None:
    graph = CanonicalGraph(1)
    node = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (8,), 1)
    graph.publish(_proposal(graph, node, 0))
    restored_graph = CanonicalGraph.from_state_dict(graph.state_dict())
    assert restored_graph.versions.get(node_ref(node.uid)) == graph.versions.get(node_ref(node.uid))
    store = LineageStore()
    lineage, context = LineageUid(3), ContextScope(4)
    store.put_overlay(LineageContextOverlay(node.uid, lineage, context, support=2))
    restored_store = LineageStore.from_state_dict(store.state_dict())
    assert restored_store.overlays == store.overlays
