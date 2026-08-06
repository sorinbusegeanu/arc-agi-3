from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.future_options import (
    _insert_future_link,
    _upsert_future_provenance_link,
    _resolve_motif_transfer_provenance,
    derive_future_option_events,
    derive_future_option_motifs,
    derive_future_option_attention_links,
    derive_future_option_transfer_links,
)
from v6.memory.compact_memory import ensure_memory_layout


def _open_state(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_base_state(memory_dir: Path) -> None:
    """Seed the minimum tables required for all four derivation functions."""
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        conn.execute(
            """INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score)
               VALUES ('ctx|a1', 'game-base', 'sampler-base', 1, 1, 'change', 5, 1, 2, 0.8)"""
        )
        conn.execute(
            "INSERT INTO transformation_families (canonical_signature, support_count, stability_score) VALUES ('family-1', 5, 0.8)"
        )
        conn.execute(
            """INSERT INTO carrier_candidates (carrier_signature, is_emergent, support_count)
               VALUES ('carrier-1', 1, 5)"""
        )
        conn.execute(
            "INSERT INTO role_candidates (role_signature, is_emergent, support_count) VALUES ('role-1', 1, 5)"
        )
        conn.execute(
            """INSERT INTO concept_candidates (
                   concept_signature, support_count, is_promoted,
                   promotion_status, validation_status, promotion_score
               ) VALUES ('concept-1', 5, 1, 'retained', 'passed', 0.7)"""
        )
        conn.execute(
            """INSERT INTO concept_candidates (
                   concept_signature, support_count, is_promoted,
                   promotion_status, validation_status, promotion_score
               ) VALUES ('concept-2', 3, 0, 'candidate', NULL, 0.4)"""
        )
        conn.execute(
            """INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count)
               VALUES ('carrier-1', 'family', 'family-1', 5)"""
        )
        conn.execute(
            """INSERT INTO role_links (role_signature, linked_type, linked_key, support_count)
               VALUES ('role-1', 'carrier', 'carrier-1', 5)"""
        )
        conn.execute(
            """INSERT INTO role_links (role_signature, linked_type, linked_key, support_count)
               VALUES ('role-1', 'family', 'family-1', 3)"""
        )
        conn.execute(
            """INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count)
               VALUES ('concept-1', 'carrier', 'carrier-1', 4)"""
        )
        conn.execute(
            """INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count)
               VALUES ('concept-1', 'family', 'family-1', 3)"""
        )
        conn.execute(
            """INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count)
               VALUES ('concept-1', 'role', 'role-1', 5)"""
        )
        conn.execute(
            """INSERT INTO concept_promotion_state (
                   concept_signature, historically_promoted, currently_promoted,
                   promotion_status, validation_status, first_promoted_global_step
               ) VALUES ('concept-1', 1, 1, 'retained', 'passed', 3)"""
        )
        conn.execute(
            """INSERT INTO role_transfer_attempts (
                   attempt_id, role_signature, transfer_kind, reuse_success,
                   similarity_score, transfer_score, best_margin,
                   source_carrier_count, candidate_role_count,
                   source_game_key, target_game_key,
                   source_context_key, target_context_key,
                   provenance_mode, provenance_status
               ) VALUES (
                   'transfer-1', 'role-1', 'cross_game', 1,
                   0.9, 0.85, 0.2, 2, 3,
                   'g1', 'g2', 'ctx1', 'ctx2',
                   'single_source', 'verified'
               )"""
        )
        conn.execute(
            """INSERT INTO future_option_motifs (
                   motif_signature, motif_type, support_count,
                   motif_stability_score, is_emergent
               ) VALUES ('motif-verified', 'enable', 5, 0.85, 1)"""
        )
        conn.execute(
            """INSERT INTO future_option_links (
                   motif_signature, linked_type, linked_key, support_count
               ) VALUES (
                   'motif-verified', 'motif_associated_with_role', 'role-1', 3
               )"""
        )


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    owner_type: str = "interaction",
    game: str = "g1",
) -> None:
    conn.execute(
        """INSERT INTO future_option_events (
               event_id, owner_type, owner_key, game,
               sampler, context_key, action_key, source_kind
           ) VALUES (?, ?, ?, ?, 'sampler', '?', '?', 'stable_contingency')""",
        (event_id, owner_type, f"{owner_type}-{event_id}", game),
    )


def test_insert_future_link_records_and_returns_provenance(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    state_conn = _open_state(memory_dir / "current_state.sqlite")
    try:
        _insert_future_link(
            state_conn,
            "motif-new",
            "interaction",
            "i1",
            1,
            None,
            None,
        )
        row = state_conn.execute(
            """
            SELECT motif_signature, linked_type, linked_key, support_count
            FROM future_option_links
            WHERE motif_signature = ?
              AND linked_type = ?
              AND linked_key = ?
            """,
            ("motif-new", "interaction", "i1"),
        ).fetchone()
        assert row is not None
        assert tuple(row) == ("motif-new", "interaction", "i1", 1)
    finally:
        state_conn.close()


def test_insert_future_link_detects_early_failure_and_retries(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    state_conn = _open_state(memory_dir / "current_state.sqlite")
    try:
        _insert_future_link(
            state_conn,
            "motif-no-role",
            "interaction",
            "i2",
            1,
            None,
            None,
        )
        _insert_future_link(
            state_conn,
            "motif-no-role",
            "interaction",
            "i2",
            1,
            None,
            None,
        )
        count = state_conn.execute(
            """
            SELECT COUNT(*)
            FROM future_option_links
            WHERE motif_signature = ?
              AND linked_type = ?
              AND linked_key = ?
            """,
            ("motif-no-role", "interaction", "i2"),
        ).fetchone()[0]
        assert count == 1
    finally:
        state_conn.close()


def test_upsert_future_provenance_link_is_idempotent(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    state_conn = _open_state(memory_dir / "current_state.sqlite")
    try:
        for _ in range(2):
            _upsert_future_provenance_link(
                state_conn,
                "motif-verified",
                "motif_associated_with_role",
                "role-1",
                None,
                None,
            )
        count = state_conn.execute(
            """
            SELECT COUNT(*)
            FROM future_option_links
            WHERE motif_signature = ?
              AND linked_type = ?
              AND linked_key = ?
            """,
            (
                "motif-verified",
                "motif_associated_with_role",
                "role-1",
            ),
        ).fetchone()[0]
        assert count == 1
    finally:
        state_conn.close()


def test_upsert_future_provenance_link_records_new_owner(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    state_conn = _open_state(memory_dir / "current_state.sqlite")
    try:
        _upsert_future_provenance_link(
            state_conn,
            "motif-verified",
            "motif_associated_with_role",
            "role-2",
            None,
            None,
        )
        row = state_conn.execute(
            """
            SELECT linked_key
            FROM future_option_links
            WHERE motif_signature = ?
              AND linked_type = ?
              AND linked_key = ?
            """,
            (
                "motif-verified",
                "motif_associated_with_role",
                "role-2",
            ),
        ).fetchone()
        assert row is not None
        assert row["linked_key"] == "role-2"
    finally:
        state_conn.close()


def test_derive_future_option_events_persists_provenance(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn, sqlite3.connect(
        memory_dir / "graph.sqlite"
    ) as graph_conn:
        derive_future_option_events(state_conn, graph_conn, max_events=10)
        row = state_conn.execute(
            """SELECT classification_source, source_kind
               FROM future_option_events
               ORDER BY first_seen_global_step ASC
               LIMIT 1"""
        ).fetchone()
        assert str(row[0]).startswith(("structural", "unverified_fallback"))


def test_derive_future_option_motifs_resolves_transfer_provenance(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn, sqlite3.connect(
        memory_dir / "graph.sqlite"
    ) as graph_conn:
        derive_future_option_events(state_conn, graph_conn, max_events=10)
        summary = derive_future_option_motifs(state_conn, graph_conn, max_motifs=10)
        assert summary["future_option_motif_count"] >= 1
        assert summary["motifs_with_role_provenance"] >= 1


def test_derive_future_option_attention_links_resolves_all_three_types(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn, sqlite3.connect(
        memory_dir / "graph.sqlite"
    ) as graph_conn:
        summary = derive_future_option_attention_links(state_conn)
        assert "future_option_attention_link_count" in summary
        assert summary["future_option_attention_link_count"] >= 0


def test_derive_future_option_transfer_links_verifies_concrete_pairs(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn:
        summary = derive_future_option_transfer_links(state_conn)
        assert summary["verified_concrete_transfer_link_count"] >= 1


def test_full_pipeline_preserves_row_counts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn, sqlite3.connect(
        memory_dir / "graph.sqlite"
    ) as graph_conn:
        row_counts_before = {
            table: state_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "future_option_events",
                "future_option_motifs",
                "future_option_links",
                "future_option_transfer_links",
            )
        }
        derive_future_option_events(state_conn, graph_conn, max_events=10)
        derive_future_option_motifs(state_conn, graph_conn, max_motifs=20)
        derive_future_option_attention_links(state_conn)
        derive_future_option_transfer_links(state_conn)
        row_counts_after = {
            table: state_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "future_option_events",
                "future_option_motifs",
                "future_option_links",
                "future_option_transfer_links",
            )
        }
        for table, before_count in row_counts_before.items():
            assert row_counts_after[table] >= before_count
        assert row_counts_after["future_option_events"] > 0
        assert row_counts_after["future_option_motifs"] > 0
        assert row_counts_after["future_option_links"] > 0


def test_h11_retained_concept_validation_remains_verified(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn:
        derive_future_option_transfer_links(state_conn)
        row = state_conn.execute(
            "SELECT concept_validation_status FROM future_option_transfer_links"
        ).fetchone()
        assert tuple(row)[0] == "verified"


def test_h11_provenance_resolution_paths_are_deterministic(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn:
        direct = _resolve_motif_transfer_provenance(
            state_conn,
            motif_signature="direct",
            motif_links={"event": {"interaction-1"}},
            direct_interaction_ids={"interaction-1"},
        )
        family = _resolve_motif_transfer_provenance(
            state_conn,
            motif_signature="family",
            motif_links={"motif_derived_from_family": {"family-1"}},
            direct_interaction_ids=set(),
        )
        carrier = _resolve_motif_transfer_provenance(
            state_conn,
            motif_signature="carrier",
            motif_links={"motif_expressed_by_carrier": {"carrier-1"}},
            direct_interaction_ids=set(),
        )
        role = _resolve_motif_transfer_provenance(
            state_conn,
            motif_signature="role",
            motif_links={"motif_associated_with_role": {"role-1"}},
            direct_interaction_ids=set(),
        )
        assert (direct["status"], direct["resolution_path"]) == (
            "verified",
            "motif_to_interaction",
        )
        assert family["status"] == "verified"
        assert carrier["status"] == "verified"
        assert role["status"] == "verified"
        assert direct["resolution_path"] == "motif_to_interaction"


def test_h11_transfer_metrics_aggregate_across_motifs(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn:
        for motif_signature in ("motif-a", "motif-b"):
            state_conn.execute(
                """INSERT INTO future_option_motifs (
                       motif_signature, motif_type, support_count,
                       motif_stability_score, is_emergent
                   ) VALUES (?, 'enable', 5, 0.8, 1)""",
                (motif_signature,),
            )
            state_conn.execute(
                """INSERT INTO future_option_links (
                       motif_signature, linked_type, linked_key, support_count
                   ) VALUES (?, 'motif_associated_with_role', 'role-1', 3)""",
                (motif_signature,),
            )
        state_conn.commit()
        summary = derive_future_option_transfer_links(state_conn)
        assert summary["all_motifs_with_transfer_count"] == 2


def test_h11_concept_promotion_status_transitions(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn:
        derive_future_option_transfer_links(state_conn)
        row = state_conn.execute(
            "SELECT concept_validation_status FROM future_option_transfer_links"
        ).fetchone()
        assert tuple(row)[0] == "verified"

        state_conn.execute(
            """UPDATE concept_promotion_state
               SET currently_promoted = 0,
                   promotion_status = 'demoted',
                   validation_status = 'failed'
               WHERE concept_signature = 'concept-1'"""
        )
        state_conn.commit()
        derive_future_option_transfer_links(state_conn)
        row = state_conn.execute(
            "SELECT concept_validation_status FROM future_option_transfer_links"
        ).fetchone()
        assert row is None or tuple(row)[0] != "verified"


def test_h11_multiple_game_pairs_produce_distinct_verified_pairs(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn:
        state_conn.execute(
            """INSERT INTO role_transfer_attempts (
                   attempt_id, role_signature, transfer_kind, reuse_success,
                   similarity_score, transfer_score, best_margin,
                   source_carrier_count, candidate_role_count,
                   source_game_key, target_game_key,
                   source_context_key, target_context_key,
                   provenance_mode, provenance_status
               ) VALUES (
                   'transfer-2', 'role-1', 'cross_game', 1,
                   0.9, 0.85, 0.2, 2, 3,
                   'g3', 'g4', 'ctx3', 'ctx4',
                   'single_source', 'verified'
               )"""
        )
        state_conn.commit()
        summary = derive_future_option_transfer_links(state_conn)
        assert summary["verified_concrete_transfer_link_count"] >= 2


def test_h11_emergent_motif_with_role_association_is_tracked(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_base_state(memory_dir)
    with _open_state(memory_dir / "current_state.sqlite") as state_conn:
        state_conn.execute(
            """INSERT INTO future_option_motifs (
                   motif_signature, motif_type, support_count,
                   motif_stability_score, is_emergent
               ) VALUES ('motif-emergent', 'enable', 5, 0.8, 1)"""
        )
        state_conn.execute(
            """INSERT INTO future_option_links (
                   motif_signature, linked_type, linked_key, support_count
               ) VALUES (
                   'motif-emergent', 'motif_associated_with_role', 'role-1', 3
               )"""
        )
        state_conn.commit()
        summary = derive_future_option_transfer_links(state_conn)
        assert summary["emergent_motifs_with_strong_transfer_count"] >= 1
