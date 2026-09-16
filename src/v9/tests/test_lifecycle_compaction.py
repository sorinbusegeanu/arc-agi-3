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


def _covered_episode_graph(*, pressure: bool = False) -> tuple[CanonicalGraph, CanonicalNode, CanonicalNode, CanonicalNode]:
    graph = CanonicalGraph(
        1,
        node_capacity_per_partition=1 if pressure else 250_000,
        edge_capacity_per_partition=1 if pressure else 500_000,
    )
    episode = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (10,), 1)
    witness = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (11,), 1)
    abstraction = CanonicalNode.build(MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (20,), 2)
    _publish_node(graph, episode, {"primary_valence": 0})
    _publish_node(graph, witness, {"primary_valence": 0})
    _publish_node(
        graph,
        abstraction,
        {"parents": [[episode.uid.hi, episode.uid.lo], [witness.uid.hi, witness.uid.lo]]},
        watermark=2,
    )
    _publish_edge(graph, RelationEdge(abstraction.uid, RelationType.PROVENANCE, episode.uid), watermark=2)
    _publish_edge(graph, RelationEdge(abstraction.uid, RelationType.PROVENANCE, witness.uid), watermark=2)
    return graph, episode, witness, abstraction


class _Runtime:
    def __init__(self, graph: CanonicalGraph, lifecycle: LifecycleRegistry) -> None:
        self.graph = graph
        self.lifecycle = lifecycle
        self.watermark = 1
        self.config = SimpleNamespace(enable_lifecycle=True)
        self._replay_pool = {}
        self._m2 = {}
        self._m3 = {}
        self._m4 = {}
        self._m5 = {}
        self._m6 = {}
        self._m7 = {}
        self._transfer_trials = {}
        self._deferred_base_nodes = {}
        self._latest_interaction_grounding = {}
        self.unified_telemetry = SimpleNamespace(gauges={})
        self.transfer_trust = SimpleNamespace(records={})
        self.grounding = SimpleNamespace(states={})
        self.telemetry = {"symbol_conditioned_prediction_observations": 0}
        self._symbol_prediction_delta_sum = 0.0
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


def test_large_event_watermark_does_not_age_memory_within_one_epoch() -> None:
    graph, episode, _, _ = _covered_episode_graph()
    lifecycle = LifecycleRegistry()
    lifecycle.observe(episode.uid, support_delta=1, relevant_opportunity=True, watermark=1)
    runtime = _Runtime(graph, lifecycle)

    runtime.watermark = 50_000
    first = run_lifecycle_maintenance(runtime)
    assert first["cycle"] == 1
    assert first["dormant"] == 0
    assert lifecycle.records[episode.uid].state is CognitiveState.ACTIVE
    assert episode.uid in graph.read_view().nodes

    runtime.watermark = 100_000
    second = run_lifecycle_maintenance(runtime)
    assert second["cycle"] == 2
    assert second["dormant"] == 0
    assert lifecycle.records[episode.uid].state is CognitiveState.ACTIVE
    assert episode.uid in graph.read_view().nodes


def test_low_pressure_dormancy_does_not_physically_compact() -> None:
    graph, episode, _, _ = _covered_episode_graph(pressure=False)
    lifecycle = LifecycleRegistry()
    lifecycle.observe(episode.uid, support_delta=1, relevant_opportunity=True, watermark=1)
    runtime = _Runtime(graph, lifecycle)

    runtime.watermark = 5000
    run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1)
    assert lifecycle.records[episode.uid].state is CognitiveState.DORMANT
    runtime.watermark = 7000
    result = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1)
    assert result["pending"] == 0
    assert result["retired"] == 0
    assert episode.uid in graph.nodes


def test_dependency_safe_retirement_physically_deletes_redundant_episode() -> None:
    graph, episode, witness, abstraction = _covered_episode_graph(pressure=True)
    lifecycle = LifecycleRegistry()
    lifecycle.observe(episode.uid, support_delta=1, relevant_opportunity=True, watermark=1)
    runtime = _Runtime(graph, lifecycle)

    runtime.watermark = 5000
    first = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1, m0_representative_floor=1)
    assert first["dormant"] == 1
    assert lifecycle.records[episode.uid].state is CognitiveState.DORMANT
    assert episode.uid in graph.nodes
    assert episode.uid not in graph.read_view().nodes

    runtime.watermark = 7000
    second = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1, m0_representative_floor=1)
    assert second["pending"] == 1
    assert lifecycle.records[episode.uid].state is CognitiveState.RETIRE_PENDING

    runtime.watermark = 9000
    third = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1, m0_representative_floor=1)
    assert third["retired"] == 1
    assert episode.uid not in graph.nodes
    assert episode.uid not in lifecycle.records
    assert witness.uid in graph.nodes
    assert episode.uid not in graph.retired_tombstones
    assert not any(edge.source == episode.uid or edge.target == episode.uid for edge in graph.edges.values())
    assert graph.provenance_replacement(witness.uid) == abstraction.uid

    restored = CanonicalGraph.from_state_dict(graph.state_dict())
    assert episode.uid not in restored.nodes
    assert episode.uid not in restored.retired_tombstones
    assert restored.provenance_replacement(witness.uid) == abstraction.uid


def test_retirement_waits_for_post_dormancy_retention_recovery() -> None:
    graph, episode, _, _ = _covered_episode_graph(pressure=True)
    lifecycle = LifecycleRegistry()
    lifecycle.observe(episode.uid, support_delta=1, relevant_opportunity=True, watermark=1)
    runtime = _Runtime(graph, lifecycle)
    runtime.unified_telemetry.gauges["historical_retention"] = 0.90
    runtime.unified_telemetry.gauges["behavioral_success_rate"] = 0.60

    runtime.watermark = 5000
    run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1, m0_representative_floor=1)
    row = lifecycle.records[episode.uid]
    assert row.state is CognitiveState.DORMANT
    assert row.baseline_hgt_retention == 0.90
    assert row.baseline_behavioral_success == 0.60

    runtime.unified_telemetry.gauges["historical_retention"] = 0.70
    runtime.watermark = 7000
    blocked = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1, m0_representative_floor=1)
    assert blocked["blocked"] == 1
    assert lifecycle.records[episode.uid].state is CognitiveState.DORMANT

    runtime.unified_telemetry.gauges["historical_retention"] = 0.90
    runtime.watermark = 9000
    pending = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1, m0_representative_floor=1)
    assert pending["pending"] == 1
    assert lifecycle.records[episode.uid].state is CognitiveState.RETIRE_PENDING

    runtime._replay_pool[episode.uid] = 1.0
    runtime.watermark = 11000
    replay_blocked = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1, m0_representative_floor=1)
    assert replay_blocked["blocked"] == 1
    assert episode.uid in graph.nodes

    runtime._replay_pool.clear()
    runtime.watermark = 13000
    retired = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1, m0_representative_floor=1)
    assert retired["retired"] == 1
    assert episode.uid not in graph.nodes
    assert episode.uid not in lifecycle.records


def test_normalized_m1_remains_active_reusable_substrate() -> None:
    graph = CanonicalGraph(1, node_capacity_per_partition=1)
    normalized = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (30,), 1)
    _publish_node(graph, normalized, {"observable_relation": "ACTION:1:9", "support": 1})
    lifecycle = LifecycleRegistry()
    lifecycle.observe(normalized.uid, support_delta=1, relevant_opportunity=True, watermark=1)
    runtime = _Runtime(graph, lifecycle)

    for watermark in (5000, 7000, 9000, 11000):
        runtime.watermark = watermark
        result = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1)
        assert result["retired"] == 0
    assert normalized.uid in graph.nodes
    assert normalized.uid in graph.read_view().nodes
    assert lifecycle.records[normalized.uid].state is CognitiveState.ACTIVE


def test_higher_memory_is_not_hidden_by_age_alone() -> None:
    graph = CanonicalGraph(1)
    concept = CanonicalNode.build(MemoryLevel.M4, MemoryType.CONCEPT, (40,), 1)
    _publish_node(graph, concept)
    lifecycle = LifecycleRegistry()
    lifecycle.observe(concept.uid, support_delta=1, relevant_opportunity=True, watermark=1)
    runtime = _Runtime(graph, lifecycle)

    for watermark in (5000, 9000, 13000):
        runtime.watermark = watermark
        result = run_lifecycle_maintenance(runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1)
        assert result["retired"] == 0
    assert lifecycle.records[concept.uid].state is CognitiveState.ACTIVE
    assert concept.uid in graph.nodes
    assert concept.uid in graph.read_view().nodes


def test_lifecycle_scan_budget_is_bounded() -> None:
    graph = CanonicalGraph(1)
    lifecycle = LifecycleRegistry()
    runtime = _Runtime(graph, lifecycle)
    for index in range(50):
        node = CanonicalNode.build(MemoryLevel.M4, MemoryType.CONCEPT, (1000 + index,), index + 1)
        _publish_node(graph, node)
        lifecycle.observe(node.uid, support_delta=1, relevant_opportunity=True, watermark=index + 1)
    runtime.watermark = 100_000
    result = run_lifecycle_maintenance(runtime, scan_limit=7)
    assert result["scanned"] == 7
    assert len(lifecycle.records) == 50


def test_positive_new_support_reactivates_dormant_memory() -> None:
    lifecycle = LifecycleRegistry()
    from v9.memory import MemoryUid

    uid = MemoryUid.derive("reactivate", 1)
    lifecycle.observe(uid, support_delta=1, relevant_opportunity=True, watermark=1)
    lifecycle.transition(uid, CognitiveState.DORMANT, watermark=100)
    row = lifecycle.observe(uid, support_delta=1, relevant_opportunity=True, watermark=101)
    assert row.state is CognitiveState.REACTIVATED
    assert row.replacement_uid is None
    assert row.baseline_hgt_retention is None


def test_legacy_lifecycle_state_reactivates_bad_watermark_aging() -> None:
    from v9.memory import MemoryUid

    active_uid = MemoryUid.derive("legacy-active", 1)
    dormant_uid = MemoryUid.derive("legacy-dormant", 1)
    restored = LifecycleRegistry.from_state_dict({
        "transitions": 2,
        "records": [
            {
                "uid": [active_uid.hi, active_uid.lo],
                "state": "ACTIVE",
                "support": 2,
                "relevant_opportunities": 3,
                "last_transition_watermark": 17,
            },
            {
                "uid": [dormant_uid.hi, dormant_uid.lo],
                "state": "DORMANT",
                "support": 1,
                "relevant_opportunities": 1,
                "last_transition_watermark": 9000,
                "last_observed_watermark": 1,
                "baseline_hgt_retention": 0.9,
            },
        ],
    })
    assert restored.maintenance_cycle == 0
    assert restored.records[active_uid].last_observed_watermark == 17
    assert restored.records[active_uid].retention_score == 1.0
    assert restored.records[dormant_uid].state is CognitiveState.ACTIVE
    assert restored.records[dormant_uid].replacement_uid is None
    assert restored.records[dormant_uid].baseline_hgt_retention is None


def test_legacy_dormant_payload_flag_is_visible_until_cycle_lifecycle_reclassifies_it() -> None:
    graph = CanonicalGraph(1)
    episode = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (99,), 1)
    _publish_node(graph, episode, {"cognitive_state": "DORMANT"})
    assert episode.uid in graph.read_view().nodes
