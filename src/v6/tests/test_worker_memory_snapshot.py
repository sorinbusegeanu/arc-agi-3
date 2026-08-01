from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from hashlib import sha1
from pathlib import Path

from v6.contingency.contingency_learner import Contingency
from v6.evaluation.interaction_sampling import _initialize_worker_memory_snapshot, _worker_memory_snapshot_probe
from v6.main import V6Config, V6System
from v6.memory.compact_memory import ensure_memory_layout
from v6.memory.live_memory_queue import LiveMemoryReadCache
from v6.memory.query_engine import MemoryQueryEngine
from v6.memory.substrate import MemoryEdge, MemoryNode, MemoryScore, action_node_id
from v6.memory.worker_snapshot import SnapshotMemoryQueryEngine, WorkerMemoryOverlay, WorkerMemorySnapshot


class _Environment:
    pass


def _system_with_memory() -> V6System:
    system = V6System(_Environment(), V6Config(database_path=":memory:"))
    system.contingency_learner.import_contingency(Contingency(1, 0, ("ctx",), 1, 7, 20, 0.9))
    system.memory.upsert_node(
        MemoryNode(
            "M1:contingency:exact",
            "M1",
            "ContingencyMemory",
            attrs={
                "context_signature": '["ctx"]',
                "action": 1,
                "transformation_family": 7,
                "confidence": 0.9,
            },
        )
    )
    system.memory.upsert_node(MemoryNode("M0:source", "M0", "Source"))
    system.memory.upsert_node(MemoryNode(action_node_id(1), "M0", "Action"))
    system.memory.upsert_edge(MemoryEdge("M0:source", action_node_id(1), "takes_action"))
    system.memory.upsert_score(MemoryScore("M0:source", future_option_delta=0.5))
    context_id = "M0:context:" + sha1('["ctx"]'.encode("utf-8")).hexdigest()[:20]
    carrier_id = "M2:carrier:test"
    role_id = "M3:role:test"
    concept_id = "M4:concept:test"
    interaction_id = "M0:interaction:test"
    system.memory.upsert_node(MemoryNode(context_id, "M0", "Context"))
    system.memory.upsert_node(MemoryNode(carrier_id, "M2", "Carrier"))
    system.memory.upsert_node(MemoryNode(role_id, "M3", "FunctionalRoleMemory", attrs={"transfer_score": 0.4}))
    system.memory.upsert_node(MemoryNode(concept_id, "M4", "ConceptMemory", attrs={"transfer_success_count": 3}))
    system.memory.upsert_node(MemoryNode(interaction_id, "M0", "Interaction"))
    system.memory.upsert_edge(MemoryEdge(carrier_id, role_id, "plays_role"))
    system.memory.upsert_edge(MemoryEdge(carrier_id, "M2:family:7", "associated_with_family"))
    system.memory.upsert_edge(MemoryEdge(carrier_id, context_id, "appears_in_context"))
    system.memory.upsert_edge(MemoryEdge(carrier_id, interaction_id, "carried_by"))
    system.memory.upsert_edge(MemoryEdge(interaction_id, action_node_id(1), "takes_action"))
    system.memory.upsert_edge(MemoryEdge(role_id, concept_id, "transfers_to"))
    system.memory.upsert_edge(MemoryEdge(role_id, carrier_id, "abstracts_from"))
    return system


def test_worker_snapshot_matches_query_engine_and_uses_no_sql_hot_path() -> None:
    system = _system_with_memory()
    try:
        snapshot = WorkerMemorySnapshot.from_system(system)
        snapshot_engine = SnapshotMemoryQueryEngine(snapshot)
        sqlite_engine = MemoryQueryEngine(system.memory, contingency_learner=system.contingency_learner, graph=system.graph)
        contexts = {1: {0: ("ctx",)}}

        expected = sqlite_engine.rank_actions(contexts, [1])[0]
        actual = snapshot_engine.rank_actions(contexts, [1])[0]

        assert actual.action == expected.action
        assert actual.predicted_family == expected.predicted_family
        assert actual.score == expected.score
        assert snapshot_engine.metrics()["sqlite_queries_during_action_selection"] == 0
    finally:
        system.close()


def test_worker_snapshot_reuses_same_instance_for_multiple_queries() -> None:
    system = _system_with_memory()
    try:
        snapshot = WorkerMemorySnapshot.from_system(system)
        engine = SnapshotMemoryQueryEngine(snapshot)
        engine.rank_actions({1: {0: ("ctx",)}}, [1])
        engine.rank_actions({1: {0: ("ctx",)}}, [1])
        assert engine.memory_action_rank_count == 2
        assert engine.memory_query_count == 2
        assert engine.metrics()["sqlite_queries_during_action_selection"] == 0
    finally:
        system.close()


def test_live_overlay_takes_precedence_and_duplicate_sequences_are_idempotent() -> None:
    system = _system_with_memory()
    try:
        engine = SnapshotMemoryQueryEngine(WorkerMemorySnapshot.from_system(system))
        overlay = WorkerMemoryOverlay()
        event = {
            "event_type": "stable_contingency",
            "event_id": "stable-1",
            "payload": {
                "context_signature": '["ctx"]',
                "action": 1,
                "transformation_family": 99,
                "confidence": 1.0,
                "support_count": 100,
            },
        }
        overlay.apply_rows([event], 4)
        overlay.apply_rows([event], 4)
        engine.apply_live_overlay(overlay)
        result = engine.predict_family({0: ("ctx",)}, 1)
        assert result.predicted_family == 99
        assert overlay.last_sequence == 4
        assert len(overlay.contingencies) == 1
    finally:
        system.close()


def test_snapshot_role_and_concept_indexes_preserve_query_engine_scores() -> None:
    system = _system_with_memory()
    try:
        snapshot = WorkerMemorySnapshot.from_system(system)
        expected = MemoryQueryEngine(system.memory, contingency_learner=system.contingency_learner, graph=system.graph)
        actual = SnapshotMemoryQueryEngine(snapshot)
        context = {0: ("ctx",)}
        expected_score = expected.score_action(context, 1, [1])
        actual_score = actual.score_action(context, 1, [1])
        assert actual_score == expected_score
        assert snapshot.role_ids_by_action[1]
        assert snapshot.role_ids_by_context_node
        assert snapshot.concept_ids_by_role
    finally:
        system.close()


def test_snapshot_readwrite_delta_cache_never_opens_sqlite(monkeypatch, tmp_path: Path) -> None:
    delta_log: list[dict] = []
    cache = LiveMemoryReadCache(memory_dir=tmp_path, refresh_steps=1, delta_log=delta_log)
    monkeypatch.setattr("v6.memory.live_memory_queue.sqlite3.connect", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sqlite read")))
    delta_log.append(
        {
            "sequence": 1,
            "event_id": "stable-1",
            "event_type": "stable_contingency",
            "global_step": 1,
            "payload": {"context_signature": '["ctx"]', "action": 1, "transformation_family": 9, "confidence": 0.8},
        }
    )
    assert cache.refresh_if_due(1)
    assert cache.overlay.last_sequence == 1
    assert cache.refresh_rows == 1
    assert not cache.refresh_if_due(1)
    delta_log.append(dict(delta_log[0]))
    assert not cache.refresh_if_due(2)
    assert cache.overlay.last_sequence == 1


def test_snapshot_initializer_runs_once_and_reuses_snapshot_in_real_processes(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with ProcessPoolExecutor(
        max_workers=2,
        initializer=_initialize_worker_memory_snapshot,
        initargs=(str(memory_dir), False, False, False, False),
    ) as pool:
        results = list(pool.map(_worker_memory_snapshot_probe, [0.03] * 8))
    by_pid: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for result in results:
        by_pid[result[0]].append(result)
    assert len(by_pid) == 2
    for rows in by_pid.values():
        assert {row[1] for row in rows}
        assert len({row[1] for row in rows}) == 1
        assert {row[2] for row in rows} == {1}
