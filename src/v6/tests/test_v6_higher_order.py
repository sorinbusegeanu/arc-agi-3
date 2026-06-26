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


def test_h06_predicts_correct_role_across_held_out_game(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "a_g1_1", "game": "g1", "contexts": ["ctx_a1", "ctx_a2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "a_g1_2", "game": "g1", "contexts": ["ctx_a3", "ctx_a4"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "a_g2_1", "game": "g2", "contexts": ["ctx_a5", "ctx_a6"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "a_g2_2", "game": "g2", "contexts": ["ctx_a7", "ctx_a8"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "b_g1_1", "game": "g1", "contexts": ["ctx_b1", "ctx_b2"], "effect": "block", "action": "close", "polarity": "negative"},
        {"carrier": "b_g2_1", "game": "g2", "contexts": ["ctx_b3", "ctx_b4"], "effect": "block", "action": "close", "polarity": "negative"},
    ])
    derive_higher_order_memory(memory_dir=memory_dir)
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06")
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM role_transfer_attempts
            WHERE transfer_kind = 'cross_game' AND target_scope_key = 'g2' AND target_carrier_signature = 'a_g2_1'
            """
        ).fetchall()
    assert rows
    assert int(rows[0]["reuse_success"]) == 1
    assert rows[0]["predicted_role_signature"] == rows[0]["observed_role_signature"]
    assert result["successful_transfer_count"] > 0


def test_h06_detects_role_mismatch(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "a_g1_1", "game": "g1", "contexts": ["ctx_a1", "ctx_a2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "a_g1_2", "game": "g1", "contexts": ["ctx_a3", "ctx_a4"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "b_g2_1", "game": "g2", "contexts": ["ctx_b1", "ctx_b2"], "effect": "terminate", "action": "consume", "polarity": "negative"},
        {"carrier": "b_g2_2", "game": "g2", "contexts": ["ctx_b3", "ctx_b4"], "effect": "terminate", "action": "consume", "polarity": "negative"},
    ])
    derive_higher_order_memory(memory_dir=memory_dir)
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06")
    assert result["successful_transfer_count"] < result["transfer_attempt_count"]
    assert result["role_mismatch_count"] + result["low_similarity_count"] >= 1
    assert result["transfer_success_rate"] < 1.0


def test_same_family_hash_not_required_for_role_match(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "x1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive", "family_suffix": "fam_a"},
        {"carrier": "x2", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "enable", "action": "open", "polarity": "positive", "family_suffix": "fam_b"},
        {"carrier": "x3", "game": "g1", "contexts": ["ctx5", "ctx6"], "effect": "enable", "action": "open", "polarity": "positive", "family_suffix": "fam_c"},
    ])
    derive_higher_order_memory(memory_dir=memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        signatures = [row[0] for row in conn.execute("SELECT DISTINCT role_signature FROM role_neighborhood_signatures ORDER BY role_signature")]
    assert len(signatures) == 1


def test_different_function_does_not_collapse(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "e1", "game": "g1", "contexts": ["ctx_e1", "ctx_e2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "e2", "game": "g2", "contexts": ["ctx_e3", "ctx_e4"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "t1", "game": "g1", "contexts": ["ctx_t1", "ctx_t2"], "effect": "terminate", "action": "consume", "polarity": "negative"},
        {"carrier": "t2", "game": "g2", "contexts": ["ctx_t3", "ctx_t4"], "effect": "terminate", "action": "consume", "polarity": "negative"},
    ])
    derive_higher_order_memory(memory_dir=memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        signatures = [row[0] for row in conn.execute("SELECT DISTINCT role_signature FROM role_neighborhood_signatures ORDER BY role_signature")]
        failures = [row[0] for row in conn.execute("SELECT failure_reason FROM role_transfer_attempts WHERE failure_reason != 'success'")]
    assert len(signatures) >= 2
    assert failures


def test_h05_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "a1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "a2", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "a3", "game": "g3", "contexts": ["ctx5", "ctx6"], "effect": "enable", "action": "open", "polarity": "positive"},
    ])
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05")
    assert result["role_candidate_count"] >= 1
    assert result["emergent_role_count"] >= 1
    assert result["decision"] == "VALID"


def test_h05_singleton_role_is_partially_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "solo", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
    ])
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05")
    assert result["role_candidate_count"] >= 1
    assert result["singleton_role_ratio"] == 1.0
    assert result["decision"] == "PARTIALLY_VALID"


def test_h06_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, _transfer_rich_specs())
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06")
    assert result["transfer_attempt_count"] >= 20
    assert result["transfer_success_rate"] >= 0.60
    assert result["mean_best_margin"] >= 0.10
    assert result["decision"] == "VALID"


def test_h07_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, _transfer_rich_specs())
    result = evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07")
    assert result["concept_candidate_count"] >= 1
    assert result["promoted_concept_count"] >= 1
    assert result["concept_strong_transfer_success_count"] >= 2
    assert result["decision"] == "VALID"


def test_h07_does_not_promote_from_weak_transfer(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "w1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "w2", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "enable", "action": "open", "polarity": "positive"},
    ])
    derive_higher_order_memory(memory_dir=memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("DELETE FROM role_transfer_attempts")
        conn.execute(
            """
            INSERT INTO role_transfer_attempts (
                attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
                target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature,
                observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason,
                best_margin, source_carrier_count, candidate_role_count, first_seen_global_step, last_seen_global_step
            )
            SELECT 'weak1', role_signature, 'cross_game', 'not_game', 'g2', 'game', 'g2', carrier_signature, role_signature,
                   role_signature, 0.95, 0.95, 1, 'success', 0.0, 1, 1, 10, 20
            FROM role_neighborhood_signatures
            LIMIT 1
            """
        )
        conn.commit()
    result = evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07")
    assert result["concept_strong_transfer_success_count"] == 0
    assert result["promoted_concept_count"] == 0


def test_h08_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, _transfer_rich_specs())
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08")
    assert result["world_model_component_count"] >= 1
    assert result["coherent_world_model_component_count"] >= 1
    assert result["decision"] == "VALID"


def test_h08_cannot_be_valid_without_promoted_concepts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "a1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "a2", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "enable", "action": "open", "polarity": "positive"},
    ])
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08")
    assert result["promoted_concept_count"] == 0
    assert result["decision"] == "PARTIALLY_VALID"


def test_max_transfer_attempts_respected(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    specs = []
    for index in range(30):
        specs.append(
            {
                "carrier": f"cap_{index}",
                "game": f"g{index % 5}",
                "contexts": [f"ctx_{index}_1", f"ctx_{index}_2"],
                "effect": "enable" if index % 2 == 0 else "block",
                "action": "open" if index % 2 == 0 else "close",
                "polarity": "positive" if index % 2 == 0 else "negative",
            }
        )
    derive_higher_order_memory(memory_dir=_seed_memory(memory_dir, specs), max_transfer_attempts=10)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        count = conn.execute("SELECT COUNT(*) FROM role_transfer_attempts").fetchone()[0]
    assert count <= 10


def test_standalone_h05_h08_derive_safely(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, _transfer_rich_specs())
    h05 = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05")
    h06 = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06")
    h07 = evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07")
    h08 = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08")
    assert h05["role_candidate_count"] >= 1
    assert h06["transfer_attempt_count"] >= 1
    assert h07["concept_candidate_count"] >= 1
    assert h08["world_model_component_count"] >= 1


def test_suite_includes_h01_h08(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _seed_memory(memory_dir, _transfer_rich_specs())
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
        "h06_role_mismatch_count": 3,
        "h06_mean_best_margin": 0.2,
        "H07": "VALID",
        "concept_candidates": 2,
        "promoted_concepts": 1,
        "h07_strong_transfer_successes": 4,
        "H08": "VALID",
        "world_model_components": 1,
        "coherent_world_model_components": 1,
        "candidate_only_world_model_components": 0,
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
    assert "role mismatch count" in text
    assert "strong transfer successes" in text
    assert "candidate-only components" in text


def _transfer_rich_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    families = [
        ("enable", "open", "positive", "a"),
        ("block", "close", "negative", "b"),
        ("transform", "shift", "neutral", "c"),
    ]
    for effect, action, polarity, prefix in families:
        for game_index, game in enumerate(("g1", "g2", "g3")):
            for carrier_index in range(2):
                idx = f"{prefix}_{game_index}_{carrier_index}"
                specs.append(
                    {
                        "carrier": idx,
                        "game": game,
                        "contexts": [f"ctx_{idx}_1", f"ctx_{idx}_2"],
                        "effect": effect,
                        "action": action,
                        "polarity": polarity,
                    }
                )
    return specs


def _seed_memory(memory_dir: Path, specs: list[dict[str, object]]) -> Path:
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as state_conn, sqlite3.connect(paths.graph) as graph_conn, sqlite3.connect(paths.replay_queue) as replay_conn:
        family_seen: set[str] = set()
        role_counter = 0
        for spec in specs:
            carrier = str(spec["carrier"])
            game = str(spec["game"])
            contexts = [str(value) for value in spec["contexts"]]
            effect = str(spec["effect"])
            action = str(spec["action"])
            polarity = str(spec["polarity"])
            family_suffix = str(spec.get("family_suffix") or carrier)
            family_signature = f"family:{effect}:{action}:{polarity}:{family_suffix}"
            if family_signature not in family_seen:
                role_counter += 1
                family_seen.add(family_signature)
                state_conn.execute(
                    """
                    INSERT INTO transformation_families (
                        family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                        support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (role_counter, family_signature, family_signature, effect, action, polarity, 5, 3, 10, 20, 1.0),
                )
            state_conn.execute(
                """
                INSERT INTO carrier_candidates (
                    carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                    first_seen_global_step, last_seen_global_step, stability_score, is_emergent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (carrier, carrier, "object", 4, 1, 10, 20, 0.9, 1),
            )
            state_conn.execute(
                """
                INSERT INTO stable_contingencies (
                    contingency_id, canonical_key, game, sampler, action, effect_signature, support_count,
                    first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                    mean_replay_priority, representative_example_count, context_level
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"cont_{carrier}",
                    f"ctx|a{action}|f{family_signature}",
                    game,
                    "synthetic",
                    action,
                    family_signature,
                    3,
                    10,
                    20,
                    0.8,
                    0.2,
                    0.7,
                    1,
                    1,
                ),
            )
            for linked_type, linked_key in [("family", family_signature), ("contingency", f"cont_{carrier}")]:
                state_conn.execute(
                    """
                    INSERT INTO carrier_links (
                        carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (carrier, linked_type, linked_key, 1, 10, 20),
                )
            for context_key in contexts:
                state_conn.execute(
                    """
                    INSERT INTO carrier_links (
                        carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (carrier, "context", context_key, 1, 10, 20),
                )
                _insert_node(graph_conn, f"context:{context_key}", "context", context_key)
                _insert_node(graph_conn, f"game:{game}", "game", game)
                _insert_edge(graph_conn, f"game:{game}", f"context:{context_key}", "observed_in")
                _insert_edge(graph_conn, f"carrier:{carrier}", f"context:{context_key}", "appears_in")
            _insert_node(graph_conn, f"carrier:{carrier}", "carrier", carrier)
            _insert_node(graph_conn, f"family:{family_signature}", "family", family_signature)
            _insert_node(graph_conn, f"contradiction:{carrier}", "contradiction", carrier)
            _insert_edge(graph_conn, f"carrier:{carrier}", f"family:{family_signature}", "explains")
            _insert_edge(graph_conn, f"family:{family_signature}", f"contradiction:{carrier}", "contradicted_by")
            replay_conn.execute(
                """
                INSERT INTO replay_queue (
                    replay_id, owner_type, owner_id, priority_score, reason,
                    first_seen_global_step, last_seen_global_step, compact_payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (f"replay_{carrier}", "carrier", carrier, 0.8, "synthetic", 10, 20, json.dumps({"carrier": carrier})),
            )
            state_conn.execute(
                """
                INSERT INTO contradiction_clusters (
                    cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step,
                    max_prediction_error, mean_replay_priority
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (f"cluster_{carrier}", carrier, 1, 10, 20, 1.0, 0.8),
            )
        state_conn.execute(
            """
            INSERT OR REPLACE INTO temporal_milestones (
                game, sampler, seed, first_interaction_step, first_contingency_candidate_step,
                first_stable_contingency_step, first_prediction_violation_step, first_high_replay_priority_step,
                first_transformation_family_step, first_stable_transformation_family_step,
                first_carrier_candidate_step, first_emergent_carrier_step
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("g1", "synthetic", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
    return memory_dir


def _insert_node(conn: sqlite3.Connection, node_id: str, node_type: str, canonical_key: str) -> None:
    conn.execute(
        """
        INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO NOTHING
        """,
        (node_id, node_type, canonical_key, 10, 20, 1),
    )


def _insert_edge(conn: sqlite3.Connection, source: str, target: str, edge_type: str) -> None:
    edge_id = f"{source}->{edge_type}->{target}"
    conn.execute(
        """
        INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(edge_id) DO NOTHING
        """,
        (edge_id, source, target, edge_type, 10, 20, 1, 1.0),
    )
