from __future__ import annotations

from types import SimpleNamespace

from v9.memory import CanonicalNode, MemoryLevel, MemoryType
from v9.memory.model import CognitiveState
from v9.memory.relations import RelationEdge, RelationType
from v9.mutation import MutationKind, MutationOutcome, MutationProposal, MutationWrite, ReadDependency, ReadSet
from v9.runtime.lifecycle import LifecycleRegistry, run_lifecycle_maintenance
from v9.runtime.publication import CanonicalGraph, node_ref


def _publish_node(graph: CanonicalGraph, node: CanonicalNode, payload: dict | None = None, *, watermark: int = 1) -> None:
    ref = node_ref(node.uid)
    proposal = MutationProposal.build(
        MutationKind.UPSERT_NODE,
        target_partitions=(node.uid.shard(graph.partition_count),),
        read_set=ReadSet.build((ReadDependency(ref, graph.versions.get(ref)),), maximum_size=8),
        evidence_refs=(),
        causal_watermark=watermark,
        writes=(MutationWrite(node=node, payload=payload or {}),),
    )
    assert graph.publish(proposal).outcome is MutationOutcome.ACCEPTED


def _publish_edge(graph: CanonicalGraph, edge: RelationEdge, *, watermark: int = 1) -> None:
    partitions = tuple(sorted({edge.source.shard(graph.partition_count), edge.target.shard(graph.partition_count)}))
    proposal = MutationProposal.build(
        MutationKind.UPSERT_EDGE,
        target_partitions=partitions,
        read_set=ReadSet.build((), maximum_size=8),
        evidence_refs=(),
        causal_watermark=watermark,
        writes=(MutationWrite(edge=edge),),
    )
    assert graph.publish(proposal).outcome is MutationOutcome.ACCEPTED


class _Runtime:
    def __init__(self, graph: CanonicalGraph, lifecycle: LifecycleRegistry) -> None:
        self.graph = graph
        self.lifecycle = lifecycle
        self.watermark = 1
        self.config = SimpleNamespace(enable_lifecycle=True)
        self._replay_pool = {}
        self._deferred_base_nodes = {}
        self._latest_interaction_grounding = {}
        self.unified_telemetry = SimpleNamespace(gauges={})
        self.consolidation = []

    def record_hgt_consolidation(self, sample) -> None:
        self.consolidation.append(sample)


def test_capacity_values_are_pressure_signals_not_learning_stops() -> None:
    graph = CanonicalGraph(1, node_capacity_per_partition=1, edge_capacity_per_partition=1)
    first = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (1,), 1)
    second = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (2,), 2)
    _publish_node(graph, first)
    _publish_node(graph, second, watermark=2)
    assert graph.memory_count() == 2
    assert graph.pressure_ratio() >= 2.0


def test_dependency_safe_retirement_is_two_phase_and_persists_tombstone() -> None:
    graph = CanonicalGraph(1)
    episode = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (10,), 1)
    abstraction = CanonicalNode.build(MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (20,), 2)
    _publish_node(graph, episode, {"primary_valence": 0})
    _publish_node(graph, abstraction, {"parents": [[episode.uid.hi, episode.uid.lo]]}, watermark=2)
    _publish_edge(graph, RelationEdge(abstraction.uid, RelationType.PROVENANCE, episode.uid), watermark=2)

    lifecycle = LifecycleRegistry()
    lifecycle.observe(episode.uid, support_delta=1, relevant_opportunity=True, watermark=1)
    runtime = _Runtime(graph, lifecycle)

    runtime.watermark = 5000
    first = run_lifecycle_maintenance(runtime, dormancy_grace_watermarks=1000, retirement_grace_watermarks=1000)
    assert first["dormant"] == 1
    assert lifecycle.records[episode.uid].state is CognitiveState.DORMANT
    assert episode.uid in graph.nodes
    assert episode.uid not in graph.read_view().nodes

    runtime.watermark = 7000
    second = run_lifecycle_maintenance(runtime, dormancy_grace_watermarks=1000, retirement_grace_watermarks=1000)
    assert second["pending"] == 1
    assert lifecycle.records[episode.uid].state is CognitiveState.RETIRE_PENDING
    assert episode.uid in graph.nodes
    assert episode.uid not in graph.read_view().nodes

    runtime.watermark = 9000
    third = run_lifecycle_maintenance(runtime, dormancy_grace_watermarks=1000, retirement_grace_watermarks=1000)
    assert third["retired"] == 1
    assert lifecycle.records[episode.uid].state is CognitiveState.RETIRED
    assert episode.uid not in graph.nodes
    assert episode.uid in graph.retired_tombstones
    assert graph.retired_tombstones[episode.uid].replacement_uid == abstraction.uid
    assert not any(edge.source == episode.uid or edge.target == episode.uid for edge in graph.edges.values())

    restored = CanonicalGraph.from_state_dict(graph.state_dict())
    assert restored.retired_tombstones[episode.uid].replacement_uid == abstraction.uid


def test_higher_memory_can_go_dormant_but_is_not_physically_retired() -> None:
    graph = CanonicalGraph(1)
    concept = CanonicalNode.build(MemoryLevel.M4, MemoryType.CONCEPT, (40,), 1)
    _publish_node(graph, concept)
    lifecycle = LifecycleRegistry()
    lifecycle.observe(concept.uid, support_delta=1, relevant_opportunity=True, watermark=1)
    runtime = _Runtime(graph, lifecycle)

    runtime.watermark = 5000
    run_lifecycle_maintenance(runtime, dormancy_grace_watermarks=1000, retirement_grace_watermarks=1000)
    assert lifecycle.records[concept.uid].state is CognitiveState.DORMANT
    assert concept.uid in graph.nodes
    assert concept.uid not in graph.read_view().nodes

    runtime.watermark = 9000
    result = run_lifecycle_maintenance(runtime, dormancy_grace_watermarks=1000, retirement_grace_watermarks=1000)
    assert result["retired"] == 0
    assert concept.uid in graph.nodes


def test_positive_new_support_reactivates_dormant_memory() -> None:
    lifecycle = LifecycleRegistry()
    from v9.memory import MemoryUid

    uid = MemoryUid.derive("reactivate", 1)
    lifecycle.observe(uid, support_delta=1, relevant_opportunity=True, watermark=1)
    lifecycle.transition(uid, CognitiveState.DORMANT, watermark=100)
    row = lifecycle.observe(uid, support_delta=1, relevant_opportunity=True, watermark=101)
    assert row.state is CognitiveState.REACTIVATED
    assert row.replacement_uid is None


def test_legacy_lifecycle_state_restores_without_new_fields() -> None:
    from v9.memory import MemoryUid

    uid = MemoryUid.derive("legacy-lifecycle", 1)
    restored = LifecycleRegistry.from_state_dict({
        "transitions": 1,
        "records": [{
            "uid": [uid.hi, uid.lo],
            "state": "ACTIVE",
            "support": 2,
            "relevant_opportunities": 3,
            "last_transition_watermark": 17,
        }],
    })
    row = restored.records[uid]
    assert row.last_observed_watermark == 17
    assert row.retention_score == 1.0
