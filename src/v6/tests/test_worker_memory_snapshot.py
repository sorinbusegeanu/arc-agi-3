from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from hashlib import sha1
from pathlib import Path

import pytest

from v6.contingency.contingency_learner import Contingency
from v6.evaluation.interaction_sampling import _initialize_worker_memory_snapshot, _worker_memory_snapshot_probe
from v6.main import V6Config, V6System
from v6.memory.compact_memory import ensure_memory_layout
from v6.memory.live_memory_queue import LiveMemoryDeltaStore, LiveMemoryReadCache
from v6.memory.query_engine import MemoryQueryEngine
from v6.memory.substrate import MemoryEdge, MemoryNode, MemoryScore, action_node_id
from v6.memory.worker_snapshot import (
    SnapshotMemoryQueryEngine,
    WorkerMemoryOverlay,
    WorkerMemorySnapshot,
    build_worker_memory_snapshot_from_directory,
    write_worker_memory_snapshot_artifact,
)


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


def test_snapshot_and_sqlite_engines_are_semantically_equivalent_across_actions() -> None:
    system = _system_with_memory()
    try:
        # Stable fallback, negative future evidence and a contradiction exercise
        # all action-ranking evidence branches; actions 3/4 deliberately tie.
        system.contingency_learner.import_contingency(Contingency(2, 0, ("ctx",), 2, 8, 30, 0.75))
        for action, delta in ((2, -0.5), (3, None), (4, None)):
            source = f"M0:source:{action}"
            system.memory.upsert_node(MemoryNode(source, "M0", "Source"))
            system.memory.upsert_node(MemoryNode(action_node_id(action), "M0", "Action"))
            system.memory.upsert_edge(MemoryEdge(source, action_node_id(action), "takes_action"))
            system.memory.upsert_score(MemoryScore(source, future_option_delta=delta, replay_priority=0.9))
            if action == 2:
                system.memory.upsert_edge(MemoryEdge(source, "M0:prediction:bad", "violates_prediction"))
        sqlite_engine = MemoryQueryEngine(system.memory, contingency_learner=system.contingency_learner, graph=system.graph)
        snapshot_engine = SnapshotMemoryQueryEngine(WorkerMemorySnapshot.from_system(system))
        contexts = {action: {0: ("ctx",)} for action in (1, 2, 3, 4)}
        sqlite_scores = sqlite_engine.rank_actions(contexts, [4, 2, 1, 3])
        snapshot_scores = snapshot_engine.rank_actions(contexts, [4, 2, 1, 3])
        assert [score.action for score in snapshot_scores] == [score.action for score in sqlite_scores]
        for expected, actual in zip(sqlite_scores, snapshot_scores, strict=True):
            assert actual.predicted_family == expected.predicted_family
            assert actual.evidence_sources == expected.evidence_sources
            assert actual.score == pytest.approx(expected.score, abs=1e-12)
            assert actual.failure_risk == pytest.approx(expected.failure_risk, abs=1e-12)
        ordered_actions = [score.action for score in snapshot_scores]
        assert ordered_actions.index(3) < ordered_actions.index(4)
        for action in (1, 2, 3, 4):
            expected = sqlite_engine.predict_family({0: ("other",)}, action)
            actual = snapshot_engine.predict_family({0: ("other",)}, action)
            assert actual == expected
    finally:
        system.close()


def test_snapshot_readwrite_delta_cache_never_opens_sqlite(monkeypatch, tmp_path: Path) -> None:
    delta_store = LiveMemoryDeltaStore()
    cache = LiveMemoryReadCache(memory_dir=tmp_path, refresh_steps=1, delta_store=delta_store)
    monkeypatch.setattr("v6.memory.live_memory_queue.sqlite3.connect", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sqlite read")))
    event = {
            "sequence": 1,
            "event_id": "stable-1",
            "event_type": "stable_contingency",
            "global_step": 1,
            "payload": {"context_signature": '["ctx"]', "action": 1, "transformation_family": 9, "confidence": 0.8},
    }
    delta_store.append_batch([event])
    assert cache.refresh_if_due(1)
    assert cache.overlay.last_sequence == 1
    assert cache.refresh_rows == 1
    assert not cache.refresh_if_due(1)
    delta_store.append_batch([dict(event)])
    assert not cache.refresh_if_due(2)
    assert cache.overlay.last_sequence == 1
    assert delta_store.stats()["get_after_calls"] == 2


def test_live_delta_refresh_changes_snapshot_ranking_without_sqlite(monkeypatch, tmp_path: Path) -> None:
    system = _system_with_memory()
    try:
        engine = SnapshotMemoryQueryEngine(WorkerMemorySnapshot.from_system(system))
        contexts = {1: {0: ("ctx",)}, 2: {0: ("ctx",)}}
        before = engine.rank_actions(contexts, [1, 2])
        assert before[0].action == 1
        store = LiveMemoryDeltaStore()
        cache = LiveMemoryReadCache(memory_dir=tmp_path, refresh_steps=1, delta_store=store)
        monkeypatch.setattr(
            "v6.memory.live_memory_queue.sqlite3.connect",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("snapshot refresh read SQLite")),
        )
        store.append_batch([
            {
                "sequence": 1,
                "event_id": "live-action-2",
                "event_type": "stable_contingency",
                "payload": {
                    "context_signature": '["ctx"]', "action": 2,
                    "transformation_family": 8, "confidence": 1.0, "support_count": 100,
                },
            }
            ,
            {
                "sequence": 2,
                "event_id": "live-future-2",
                "event_type": "future_option",
                "payload": {"action": 2, "option_delta": 1.0},
            }
        ])
        assert cache.refresh_if_due(1)
        engine.apply_live_overlay(cache.overlay)
        after = engine.rank_actions(contexts, [1, 2])
        assert after[0].action == 2
        assert engine.predict_family({0: ("ctx",)}, 2).predicted_family == 8
        assert engine.metrics()["sqlite_queries_during_action_selection"] == 0
    finally:
        system.close()


def test_delta_store_retention_and_overlay_ordering_are_explicit() -> None:
    store = LiveMemoryDeltaStore(max_events=2)
    store.append_batch([
        {"sequence": 2, "event_id": "two", "event_type": "future_option", "payload": {"option_delta": 0.1}},
        {"sequence": 1, "event_id": "one", "event_type": "future_option", "payload": {"option_delta": 0.2}},
        {"sequence": 3, "event_id": "three", "event_type": "future_option", "payload": {"option_delta": 0.3}},
    ])
    rows, high_water = store.get_after(1, 10)
    assert [row["sequence"] for row in rows] == [2, 3]
    assert high_water == 3
    cache = LiveMemoryReadCache(memory_dir=".", refresh_steps=1, delta_store=store)
    cache.last_applied_live_sequence = 0
    assert not cache.refresh_if_due(1)
    assert cache.delta_refresh_state == "worker_lagged_beyond_retention"


def test_snapshot_initializer_runs_once_and_reuses_snapshot_in_real_processes(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    snapshot = build_worker_memory_snapshot_from_directory(memory_dir, include_graph=False, include_substrate=False)
    artifact, _ = write_worker_memory_snapshot_artifact(snapshot, tmp_path / "snapshot.pkl")
    with ProcessPoolExecutor(
        max_workers=2,
        initializer=_initialize_worker_memory_snapshot,
        initargs=(str(artifact), False, False, False, False),
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
