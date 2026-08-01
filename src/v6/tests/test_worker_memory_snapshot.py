from __future__ import annotations

from v6.contingency.contingency_learner import Contingency
from v6.main import V6Config, V6System
from v6.memory.query_engine import MemoryQueryEngine
from v6.memory.substrate import MemoryEdge, MemoryNode, MemoryScore, action_node_id
from v6.memory.worker_snapshot import SnapshotMemoryQueryEngine, WorkerMemoryOverlay, WorkerMemorySnapshot


class _Environment:
    pass


def _system_with_memory() -> V6System:
    system = V6System(_Environment(), V6Config(database_path=":memory:"))
    system.contingency_learner.import_contingency(Contingency(1, 0, ("ctx",), 1, 7, 20, 0.9))
    system.memory.upsert_node(MemoryNode("M0:source", "M0", "Source"))
    system.memory.upsert_node(MemoryNode(action_node_id(1), "M0", "Action"))
    system.memory.upsert_edge(MemoryEdge("M0:source", action_node_id(1), "takes_action"))
    system.memory.upsert_score(MemoryScore("M0:source", future_option_delta=0.5))
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
