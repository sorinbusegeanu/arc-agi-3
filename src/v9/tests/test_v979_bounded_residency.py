from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from v9 import ContinuousMemoryRuntime
from v9.memory import CanonicalNode, MemoryLevel, MemoryType, MemoryUid
from v9.memory.relations import RelationEdge, RelationType
from v9.mutation import MutationKind, MutationOutcome, MutationProposal, MutationWrite, ReadDependency, ReadSet
from v9.runtime.config import RuntimeConfig, ScientificConfig
from v9.runtime.memory_governor import MemoryGovernorState, RuntimeMemoryGovernor
from v9.runtime.publication import CanonicalGraph, edge_ref, node_ref
from v9.hgt.epoch_dataset import iter_training_chunks, transition_training_rows


def _publish_node(graph: CanonicalGraph, node: CanonicalNode, payload: dict, *, watermark: int = 1) -> None:
    writes = [MutationWrite(node=node, payload=payload)]
    partitions = {node.uid.shard(graph.partition_count)}
    dependencies = [ReadDependency(node_ref(node.uid), graph.versions.get(node_ref(node.uid)))]
    for raw_parent in payload.get("parents", []):
        parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
        edge = RelationEdge(node.uid, RelationType.PROVENANCE, parent)
        writes.append(MutationWrite(edge=edge))
        partitions.add(parent.shard(graph.partition_count))
        dependencies.append(ReadDependency(edge_ref(edge), graph.versions.get(edge_ref(edge))))
    proposal = MutationProposal.build(
        MutationKind.UPSERT_NODE,
        target_partitions=tuple(sorted(partitions)),
        read_set=ReadSet.build(tuple(dependencies), maximum_size=64),
        evidence_refs=(),
        causal_watermark=watermark,
        writes=tuple(writes),
    )
    assert graph.publish(proposal).outcome is MutationOutcome.ACCEPTED


def test_physical_low_level_delete_removes_graph_versions_and_snapshot_tombstones() -> None:
    graph = CanonicalGraph(2)
    m0 = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (1,), 1)
    m1g = CanonicalNode.build(MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (2,), 2)
    m1n = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (3,), 3)
    _publish_node(graph, m0, {"action_id": 1, "context_signature": 10})
    _publish_node(graph, m1g, {"parents": [[m0.uid.hi, m0.uid.lo]]}, watermark=2)
    _publish_node(graph, m1n, {"parents": [[m1g.uid.hi, m1g.uid.lo]], "evidence_refs": [[m0.uid.hi, m0.uid.lo]]}, watermark=3)

    deleted = graph.delete_low_level_nodes_batch((
        (m0.uid, m1n.uid, "test"),
        (m1g.uid, m1n.uid, "test"),
    ))
    assert set(deleted) == {m0.uid, m1g.uid}
    assert m0.uid not in graph.nodes
    assert m1g.uid not in graph.nodes
    assert m1n.uid in graph.nodes
    assert graph.versions.get(node_ref(m0.uid)) == 0
    assert graph.versions.get(node_ref(m1g.uid)) == 0
    assert not any(edge.source in deleted or edge.target in deleted for edge in graph.edges.values())
    assert graph.payloads[m1n.uid].get("parents") == []
    assert graph.payloads[m1n.uid].get("evidence_refs") == []
    state = graph.state_dict()
    assert state["retired_tombstones"] == []
    assert state["low_level_nodes_deleted_total"] == 2
    restored = CanonicalGraph.from_state_dict(state)
    assert m0.uid not in restored.nodes
    assert m1g.uid not in restored.nodes
    assert restored.retired_tombstones == {}


def test_runtime_compacts_shared_normalized_group_to_configured_target(tmp_path: Path) -> None:
    scientific = ScientificConfig(
        resident_m0_limit=10,
        resident_m1_grounded_limit=10,
        resident_low_level_target_ratio=0.5,
        resident_compaction_check_interval=1,
        resident_m0_representative_floor=1,
        resident_m1_grounded_representative_floor=1,
        resident_max_delete_batch=64,
        resident_max_scan_batch=128,
        memory_rss_high_watermark_bytes=1 << 40,
        memory_rss_hard_watermark_bytes=2 << 40,
        memory_swap_high_watermark_bytes=1 << 40,
    )
    runtime = ContinuousMemoryRuntime(RuntimeConfig(tmp_path / "run", enable_snapshots=False, restore=False, scientific=scientific))
    try:
        rows = []
        grounded = []
        for index in range(12):
            m0 = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (1000 + index,), index + 1)
            m1g = CanonicalNode.build(MemoryLevel.M1, MemoryType.GROUNDED_CONTINGENCY, (2000 + index,), index + 1)
            rows.append((m0, {"action_id": index % 3, "context_signature": index % 2, "primary_valence": 0}, ()))
            rows.append((m1g, {"parents": [[m0.uid.hi, m0.uid.lo]], "executable_action_token": index % 3}, (m0.uid,)))
            grounded.append(m1g)
        m1n = CanonicalNode.build(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (9999,), 20)
        rows.append((m1n, {"parents": [[node.uid.hi, node.uid.lo] for node in grounded], "structural_signature": 9999, "observable_relation": "ACTION:test", "channel": "WORLD"}, tuple(node.uid for node in grounded)))
        assert runtime._publish_group(tuple(rows))
        runtime._resident_memory._recount_graph()
        assert runtime._resident_memory.counts() == (12, 12)
        deleted = runtime._resident_memory.maybe_compact(force=True)
        assert deleted > 0
        m0_count, m1g_count = runtime._resident_memory.counts()
        assert m0_count <= 5
        assert m1g_count <= 5
        assert m0_count >= 1
        assert m1g_count >= 1
        assert runtime._m1n_supports.get(9999, 0) == 0 or m1n.uid in runtime.graph.nodes
        assert runtime.graph.retired_tombstones == {}
    finally:
        runtime.close()


def test_hgt_transition_preprocessing_is_bounded_and_streaming(tmp_path: Path) -> None:
    path = tmp_path / "epoch.jsonl"
    rows = []
    for index in range(30):
        rows.append({
            "environment_identity": [1, "test", 1],
            "game_scenario": "g",
            "actor_id": index % 3,
            "episode_id": index // 6,
            "global_step": index,
            "before_signature": index % 4,
            "after_signature": (index + 1) % 4,
            "action_id": index % 3,
            "primary_valence": 1 if index % 11 == 0 else 0,
            "task_success": index % 6 == 5,
            "task_failure": False,
            "task_truncated": False,
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    sampled = transition_training_rows(path, max_rows=7, active_episode_limit=4)
    assert len(sampled) == 7
    chunks = list(iter_training_chunks(path, chunk_rows=5, active_episode_limit=4))
    assert chunks
    assert all(len(chunk) <= 5 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) == len(rows)


def test_memory_governor_transitions_on_rss_pressure() -> None:
    scientific = ScientificConfig(
        memory_rss_high_watermark_bytes=10,
        memory_rss_hard_watermark_bytes=20,
        memory_swap_high_watermark_bytes=10,
    )
    governor = RuntimeMemoryGovernor(scientific)
    with patch("v9.runtime.memory_governor._proc_status_bytes", side_effect=lambda name: 15 if name == "VmRSS" else 0):
        assert governor.sample().state is MemoryGovernorState.COMPACTING
    with patch("v9.runtime.memory_governor._proc_status_bytes", side_effect=lambda name: 25 if name == "VmRSS" else 0):
        assert governor.sample().state is MemoryGovernorState.HARD_PRESSURE_DRAIN
    with patch("v9.runtime.memory_governor._proc_status_bytes", return_value=0):
        assert governor.sample().state is MemoryGovernorState.NORMAL


def test_residency_scoring_uses_stable_training_reservoir_snapshot(tmp_path: Path, monkeypatch) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig(tmp_path / "run-reservoir-snapshot", enable_snapshots=False, restore=False)
    )
    graph = runtime.graph
    original_reservoir = graph._training_m0_reservoir
    try:
        m0 = CanonicalNode.build(MemoryLevel.M0, MemoryType.EPISODE, (7001,), 1)
        _publish_node(
            graph,
            m0,
            {
                "action_id": 1,
                "context_signature": 10,
                "environment_instance_id": 7,
                "episode_id": 1,
            },
        )
        stable = graph.training_m0_reservoir_snapshot()

        class ExplodingReservoir:
            def __iter__(self):
                raise RuntimeError("live training reservoir iterated")

        graph._training_m0_reservoir = ExplodingReservoir()
        monkeypatch.setattr(graph, "training_m0_reservoir_snapshot", lambda: stable)

        scored = runtime._resident_memory._score_candidates(
            [(m0.uid, m0, graph.payloads[m0.uid], m0.uid)]
        )
        assert scored
    finally:
        graph._training_m0_reservoir = original_reservoir
        runtime.close()
