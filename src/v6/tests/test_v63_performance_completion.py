from __future__ import annotations

from types import SimpleNamespace

from v6.contingency.contingency_learner import Contingency, ContingencyLearner
from v6.memory.query_engine import MemoryPrediction
from v6.memory.v63_performance_completion import (
    _emit_live_memory_event_batched,
    _flush_live_memory_batch,
    _build_relational_world_models_optimized,
    install_v63_performance_completion,
)
from v6.memory.v63_prediction_reuse_completion import (
    _cache_prediction_by_action,
    _controller_predict_reuse,
    install_v63_prediction_reuse_completion,
)
from v6.memory.compact_memory import ensure_memory_layout
from v6.v63_higher_order_semantics import install_v63_higher_order_semantics
from v6.v63_higher_order_compat import install_v63_higher_order_compat


class _Queue:
    def __init__(self) -> None:
        self.items = []
        self.qsize_calls = 0

    def put_nowait(self, item) -> None:
        self.items.append(item)

    def qsize(self) -> int:
        self.qsize_calls += 1
        return len(self.items)


def _install() -> None:
    install_v63_higher_order_semantics()
    install_v63_higher_order_compat()
    install_v63_performance_completion()
    install_v63_prediction_reuse_completion()


def test_incremental_contingency_persistence_exposes_only_changed_rows_once() -> None:
    _install()
    learner = ContingencyLearner(support_threshold=1, confidence_threshold=0.0)
    learner.import_contingency(
        Contingency(
            id=99,
            context_level=0,
            context_signature=(9,),
            action=9,
            transformation_family=9,
            support_count=5,
            confidence=1.0,
        )
    )

    learner.update_multi_scale({0: (1,), 1: (1, 2)}, action=1, transformation_family=2)
    dirty = learner.stable_contingencies()
    assert len(dirty) == 2
    assert all(item.action == 1 for item in dirty)

    full = learner.stable_contingencies()
    assert len(full) == 3
    assert {item.id for item in full} >= {99}


def test_selected_prediction_cache_reuses_exact_action_context_prediction() -> None:
    _install()
    prediction = MemoryPrediction(
        predicted_family=7,
        confidence=0.8,
        source="memory_contingency",
        evidence_node_ids=["M1:test"],
    )
    query_engine = SimpleNamespace()
    contexts = {0: (1,), 1: (2, 1)}
    _cache_prediction_by_action(query_engine, contexts, 2, prediction)
    controller = SimpleNamespace(query_engine=query_engine)

    reused = _controller_predict_reuse(
        controller,
        contexts,
        2,
        record_query=False,
    )
    assert reused is prediction


def test_live_memory_events_are_batched_before_manager_queue_rpc() -> None:
    _install()
    queue = _Queue()
    system = SimpleNamespace(
        live_memory_queue=queue,
        config=SimpleNamespace(
            shared_live_memory_mode="readwrite",
            live_memory_worker_id="worker",
        ),
        live_memory_events_emitted=0,
        live_memory_events_dropped_queue_full=0,
        live_memory_events_dropped_error=0,
        live_memory_queue_block_seconds=0.0,
        live_memory_queue_peak_size=0,
    )

    for index in range(64):
        _emit_live_memory_event_batched(
            system,
            "future_option_event",
            f"event:{index}",
            index + 1,
            0.5,
            {"index": index},
        )

    assert len(queue.items) == 1
    assert isinstance(queue.items[0], list)
    assert len(queue.items[0]) == 64
    assert system.live_memory_events_emitted == 64


def test_live_memory_batch_flush_preserves_deduplication() -> None:
    _install()
    queue = _Queue()
    system = SimpleNamespace(
        live_memory_queue=queue,
        config=SimpleNamespace(
            shared_live_memory_mode="readwrite",
            live_memory_worker_id="worker",
        ),
        live_memory_events_emitted=0,
        live_memory_events_dropped_queue_full=0,
        live_memory_events_dropped_error=0,
        live_memory_queue_block_seconds=0.0,
        live_memory_queue_peak_size=0,
    )
    args = (
        "stable_contingency",
        "stable:1",
        10,
        0.8,
        {"key": "stable:1", "support_count": 20},
    )
    _emit_live_memory_event_batched(system, *args)
    _emit_live_memory_event_batched(system, *args)
    _flush_live_memory_batch(system)

    assert len(queue.items) == 1
    assert len(queue.items[0]) == 1
    assert system.live_memory_events_emitted == 1
    assert system._v63_live_events_deduplicated == 1


def _seed_relational_world_model_inputs(connection) -> None:
    connection.execute(
        """
        INSERT INTO concept_candidates (
            concept_signature, concept_type, support_count,
            linked_role_count, linked_carrier_count, linked_family_count,
            transfer_success_count, strong_transfer_success_count,
            cross_game_count, cross_context_count, compression_gain,
            explanatory_reach, promotion_score, first_seen_global_step,
            last_seen_global_step, is_promoted, promotion_status
        ) VALUES
            ('concept:a','relational',10,1,2,2,4,3,2,3,2.0,8.0,0.9,3,10,1,'promoted'),
            ('concept:b','relational',10,1,2,2,4,3,2,3,2.0,8.0,0.85,4,10,1,'promoted')
        """
    )
    for concept in ("concept:a", "concept:b"):
        for kind, values in {
            "role": ("r1",),
            "carrier": ("ca", "cb"),
            "family": ("f1", "f2"),
            "context": ("c1", "c2", "c3"),
            "game": ("g1", "g2"),
        }.items():
            for value in values:
                connection.execute(
                    """
                    INSERT INTO concept_links (
                        concept_signature, linked_type, linked_key,
                        support_count, first_seen_global_step,
                        last_seen_global_step
                    ) VALUES (?, ?, ?, 1, 3, 10)
                    """,
                    (concept, kind, value),
                )
    connection.execute(
        """
        INSERT INTO role_links (
            role_signature, linked_type, linked_key, support_count,
            first_seen_global_step, last_seen_global_step
        ) VALUES
            ('r1','family','f1',1,2,10),
            ('r1','family','f2',1,2,10)
        """
    )
    connection.execute(
        "INSERT INTO family_members (family_signature, contingency_key, support_count) VALUES ('f1','x1',5)"
    )
    connection.execute(
        "INSERT INTO family_members (family_signature, contingency_key, support_count) VALUES ('f2','x2',5)"
    )
    connection.execute(
        """
        INSERT INTO future_option_events (
            event_id, owner_type, owner_key, game, context_key,
            option_delta, first_seen_global_step, last_seen_global_step
        ) VALUES
            ('e1','family','f1','g1','c1',1.0,7,7),
            ('e2','family','f1','g1','c1',1.0,8,8),
            ('e3','family','f2','g1','c1',1.0,9,9)
        """
    )


def test_relational_world_model_preloads_family_and_role_evidence(tmp_path) -> None:
    _install()
    paths = ensure_memory_layout(tmp_path / "memory")
    import sqlite3

    with sqlite3.connect(paths.current_state) as connection:
        connection.row_factory = sqlite3.Row
        _seed_relational_world_model_inputs(connection)
        statements = []
        connection.set_trace_callback(statements.append)
        summary = _build_relational_world_models_optimized(connection)
        connection.set_trace_callback(None)

        row = connection.execute(
            "SELECT linked_concept_count FROM world_model_components"
        ).fetchone()
        assert summary["world_model_component_count"] == 1
        assert summary["world_model_derivation_mode"] == "single_pass_preloaded_v1"
        assert int(row[0]) == 2
        assert not any(
            "FROM family_members WHERE family_signature=" in statement
            for statement in statements
        )


def test_installed_world_model_derivation_does_not_call_legacy_builder(tmp_path) -> None:
    _install()
    import sqlite3
    from v6 import higher_order_substrate as substrate
    from v6 import v63_higher_order_semantics as semantics

    paths = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(paths.current_state) as state_conn, sqlite3.connect(paths.graph) as graph_conn:
        state_conn.row_factory = sqlite3.Row
        graph_conn.row_factory = sqlite3.Row
        _seed_relational_world_model_inputs(state_conn)
        original = semantics._ORIGINAL_DERIVE_WORLD_MODELS

        def fail_legacy(*_args, **_kwargs):
            raise AssertionError("legacy world-model derivation must not run")

        semantics._ORIGINAL_DERIVE_WORLD_MODELS = fail_legacy
        try:
            summary = substrate.derive_world_model_components(
                state_conn,
                graph_conn,
                max_world_model_family_links=50,
            )
        finally:
            semantics._ORIGINAL_DERIVE_WORLD_MODELS = original

        assert summary["world_model_component_count"] >= 1
