from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.continuous_research import _format_epoch_status
from v6.higher_order_substrate import derive_higher_order_memory
from v6.hypothesis_h05_report import evaluate_h05_role_emergence
from v6.hypothesis_h06_report import evaluate_h06_role_transfer
from v6.hypothesis_h07_report import evaluate_h07_concept_emergence
from v6.hypothesis_h08_report import evaluate_h08_world_model_coherence
from v6.hypothesis_suite_report import run_hypothesis_suite_report
from v6.memory.compact_memory import ensure_memory_layout


def test_h05_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_higher_order_memory(memory_dir, carriers_per_role=3)
    derive_higher_order_memory(memory_dir=memory_dir)
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05")
    assert result["role_candidate_count"] >= 1
    assert result["emergent_role_count"] >= 1
    assert result["decision"] == "VALID"


def test_h05_singleton_role_is_partially_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_higher_order_memory(memory_dir, carriers_per_role=1, role_specs=[("enable", "open", "positive", "g1", ("ctx1", "ctx2"))])
    derive_higher_order_memory(memory_dir=memory_dir)
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05")
    assert result["role_candidate_count"] >= 1
    assert result["singleton_role_ratio"] == 1.0
    assert result["decision"] == "PARTIALLY_VALID"


def test_h06_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_higher_order_memory(memory_dir, carriers_per_role=3)
    derive_higher_order_memory(memory_dir=memory_dir)
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06")
    assert result["transfer_attempt_count"] >= 20
    assert result["transfer_success_rate"] >= 0.60
    assert result["decision"] == "VALID"


def test_h07_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_higher_order_memory(memory_dir, carriers_per_role=3)
    derive_higher_order_memory(memory_dir=memory_dir)
    result = evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07")
    assert result["concept_candidate_count"] >= 1
    assert result["promoted_concept_count"] >= 1
    assert result["decision"] == "VALID"


def test_h08_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_higher_order_memory(memory_dir, carriers_per_role=3)
    derive_higher_order_memory(memory_dir=memory_dir)
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08")
    assert result["world_model_component_count"] >= 1
    assert result["coherent_world_model_component_count"] >= 1
    assert result["decision"] == "VALID"


def test_suite_includes_h01_h08(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _seed_higher_order_memory(memory_dir, carriers_per_role=3)
    summary = run_hypothesis_suite_report(
        run_dir=run_dir,
        memory_dir=memory_dir,
        output_dir=tmp_path / "reports",
        scan_all_dbs=True,
        max_db_files=10,
        max_rows=1000,
    )
    for key in (
        "H01 decision",
        "H02 decision",
        "H03 decision",
        "H04 decision",
        "H05 decision",
        "H06 decision",
        "H07 decision",
        "H08 decision",
    ):
        assert key in summary


def test_epoch_status_format_includes_h05_h08() -> None:
    status = {
        "epoch_id": "epoch_0001",
        "global_step_start": 1,
        "global_step_end": 100,
        "workers_requested": 4,
        "workers_initial": 2,
        "workers_max_epoch": 4,
        "worker_execution": {"peak_workers": 3},
        "ram_snapshot_at_epoch_start": {"ram_used_percent": 12.5},
        "games": 2,
        "interactions_this_epoch": 100,
        "disk_used_percent": 10.0,
        "H01": "VALID",
        "stable_contingencies": 4,
        "games_with_stable_contingencies": "2/2",
        "H02": "VALID",
        "H02A": "VALID",
        "H02B": "INCONCLUSIVE",
        "replay_lift": 2.0,
        "direct_replay_evidence": "available",
        "h02_timing_note": "note",
        "H03": "VALID",
        "compression_ratio": 1.6,
        "singleton_ratio": 0.3,
        "cross_context_families": 2,
        "H04": "VALID",
        "carrier_candidates": 5,
        "stable_carriers": 3,
        "H05": "VALID",
        "role_candidates": 3,
        "emergent_roles": 2,
        "H06": "VALID",
        "role_transfer_attempts": 24,
        "role_transfer_success_rate": 0.8,
        "H07": "VALID",
        "concept_candidates": 2,
        "promoted_concepts": 1,
        "H08": "VALID",
        "world_model_components": 1,
        "coherent_world_model_components": 1,
        "cleanup": {"disk_before_cleanup_bytes": 0, "disk_after_cleanup_bytes": 0, "raw_files_deleted_count": 0, "disk_freed_bytes": 0},
        "deltas": {"stable_contingency_count_delta": 1},
        "next_action": "continue epoch_0002",
    }
    text = _format_epoch_status(status)
    assert "H05" in text
    assert "H06" in text
    assert "H07" in text
    assert "H08" in text
    assert "role candidates" in text
    assert "promoted concepts" in text
    assert "coherent components" in text


def _seed_higher_order_memory(memory_dir: Path, carriers_per_role: int, role_specs: list[tuple[str, str, str, str, tuple[str, ...]]] | None = None) -> None:
    paths = ensure_memory_layout(memory_dir)
    role_specs = role_specs or [
        ("enable", "open", "positive", "g1", ("ctx_e1", "ctx_e2")),
        ("enable", "open", "positive", "g2", ("ctx_e3", "ctx_e4")),
        ("block", "close", "negative", "g1", ("ctx_b1", "ctx_b2")),
        ("block", "close", "negative", "g2", ("ctx_b3", "ctx_b4")),
        ("transform", "shift", "neutral", "g1", ("ctx_t1", "ctx_t2")),
        ("transform", "shift", "neutral", "g2", ("ctx_t3", "ctx_t4")),
    ]
    family_counter = 0
    with sqlite3.connect(paths.current_state) as state_conn, sqlite3.connect(paths.graph) as graph_conn:
        for index, (effect_type, action_group, polarity, game, contexts) in enumerate(role_specs):
            family_signature = f"family:{effect_type}:{action_group}:{polarity}:{index}"
            state_conn.execute(
                """
                INSERT INTO transformation_families (
                    family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                    support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (index + 1, family_signature, family_signature, effect_type, action_group, polarity, 5, 3, 10, 20, 1.0),
            )
            family_counter += 1
            for cindex in range(carriers_per_role):
                carrier_signature = f"carrier_{index}_{cindex}"
                context_keys = list(contexts[:2])
                state_conn.execute(
                    """
                    INSERT INTO carrier_candidates (
                        carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                        first_seen_global_step, last_seen_global_step, stability_score, is_emergent
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (carrier_signature, carrier_signature, "object", 4, 1, 10, 20, 0.9, 1),
                )
                for context_key in context_keys:
                    state_conn.execute(
                        """
                        INSERT INTO carrier_links (
                            carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (carrier_signature, "context", context_key, 1, 10, 20),
                    )
                    graph_conn.execute(
                        """
                        INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(node_id) DO NOTHING
                        """,
                        (f"context:{context_key}", "context", context_key, 10, 20, 1),
                    )
                    graph_conn.execute(
                        """
                        INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(node_id) DO NOTHING
                        """,
                        (f"game:{game}", "game", game, 10, 20, 1),
                    )
                    graph_conn.execute(
                        """
                        INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(edge_id) DO NOTHING
                        """,
                        (f"game:{game}->observed_in->context:{context_key}", f"game:{game}", f"context:{context_key}", "observed_in", 10, 20, 1, 1.0),
                    )
                state_conn.execute(
                    """
                    INSERT INTO carrier_links (
                        carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (carrier_signature, "family", family_signature, 1, 10, 20),
                )
                state_conn.execute(
                    """
                    INSERT INTO carrier_links (
                        carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (carrier_signature, "contingency", f"cont:{carrier_signature}", 1, 10, 20),
                )
                graph_conn.execute(
                    """
                    INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO NOTHING
                    """,
                    (f"carrier:{carrier_signature}", "carrier", carrier_signature, 10, 20, 1),
                )
                graph_conn.execute(
                    """
                    INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO NOTHING
                    """,
                    (f"family:{family_signature}", "family", family_signature, 10, 20, 1),
                )
                graph_conn.execute(
                    """
                    INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(edge_id) DO NOTHING
                    """,
                    (f"carrier:{carrier_signature}->explains->family:{family_signature}", f"carrier:{carrier_signature}", f"family:{family_signature}", "explains", 10, 20, 1, 1.0),
                )
                for context_key in context_keys:
                    graph_conn.execute(
                        """
                        INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(edge_id) DO NOTHING
                        """,
                        (f"carrier:{carrier_signature}->appears_in->context:{context_key}", f"carrier:{carrier_signature}", f"context:{context_key}", "appears_in", 10, 20, 1, 1.0),
                    )
                graph_conn.execute(
                    """
                    INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO NOTHING
                    """,
                    (f"contradiction:{family_signature}", "contradiction", family_signature, 10, 20, 1),
                )
                graph_conn.execute(
                    """
                    INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(edge_id) DO NOTHING
                    """,
                    (f"family:{family_signature}->contradicted_by->contradiction:{family_signature}", f"family:{family_signature}", f"contradiction:{family_signature}", "contradicted_by", 10, 20, 1, 1.0),
                )
        state_conn.execute(
            """
            INSERT INTO temporal_milestones (
                game, sampler, seed, first_interaction_step, first_contingency_candidate_step, first_stable_contingency_step,
                first_prediction_violation_step, first_high_replay_priority_step, first_transformation_family_step,
                first_stable_transformation_family_step, first_carrier_candidate_step, first_emergent_carrier_step
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("g1", "mixed", 0, 1, 2, 3, 4, 4, 5, 6, 7, 8),
        )
        state_conn.execute(
            """
            INSERT INTO stable_contingencies (
                contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error, mean_replay_priority,
                representative_example_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "ctx|a1|ef1", "g1", "mixed", 1, 1, "family:test", 20, 1, 20, 1.0, 0.1, 0.2, 0),
        )
        state_conn.execute(
            """
            INSERT INTO contradiction_clusters (
                cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step, max_prediction_error, mean_replay_priority
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("cc1", "cc1", 3, 5, 20, 1.0, 0.8),
        )
        with sqlite3.connect(paths.replay_queue) as replay_conn:
            replay_conn.execute(
                """
                INSERT INTO replay_queue (
                    replay_id, owner_type, owner_id, priority_score, reason, first_seen_global_step, last_seen_global_step, compact_payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("r1", "interaction", "r1", 0.9, "priority", 5, 20, json.dumps({"prediction_error": 1})),
            )
