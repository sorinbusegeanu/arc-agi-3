from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.future_options import derive_future_option_events, derive_future_option_motifs, derive_future_option_transfer_links
from v6.memory.compact_memory import ensure_memory_layout


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    owner_type: str,
    owner_key: str,
    family: str | None = None,
    carrier: str | None = None,
    context: str = "ctx1",
    game: str = "g1",
) -> None:
    conn.execute(
        """
        INSERT INTO future_option_events (
            event_id, owner_type, owner_key, game, sampler, context_key, action_key, source_kind,
            motif_type, option_delta, option_delta_bucket, novelty_score, reversibility_score,
            branching_score, termination_score, contradiction_score, replay_priority_score,
            memory_priority_score, first_seen_global_step, last_seen_global_step,
            source_family_id, source_carrier_id, source_context_signature, source_action,
            source_game_id, source_sampler, evidence_json
        ) VALUES (?, ?, ?, ?, 'sampler', ?, 'a1', 'stable_contingency', 'enable', 1.0,
                  'large_positive', 0.5, 0.0, 1.0, 0.0, 0.0, 0.8, 0.7, 1, 2,
                  ?, ?, ?, 'a1', ?, 'sampler', ?)
        """,
        (
            event_id,
            owner_type,
            owner_key,
            game,
            context,
            family,
            carrier,
            context,
            game,
            json.dumps(
                {
                    "support_count": 5,
                    "source_family_ids": [] if family is None else [family],
                    "source_carrier_ids": [] if carrier is None else [carrier],
                }
            ),
        ),
    )


def _provenance_memory(tmp_path: Path) -> tuple[Path, sqlite3.Connection, sqlite3.Connection]:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    state_conn = sqlite3.connect(memory_dir / "current_state.sqlite")
    graph_conn = sqlite3.connect(memory_dir / "graph.sqlite")
    state_conn.row_factory = sqlite3.Row
    graph_conn.row_factory = sqlite3.Row
    state_conn.execute("INSERT INTO carrier_candidates (carrier_signature, support_count) VALUES ('carrier-1', 5)")
    state_conn.execute("INSERT INTO role_candidates (role_signature, support_count) VALUES ('role-1', 5)")
    state_conn.execute("INSERT INTO concept_candidates (concept_signature, support_count, is_promoted) VALUES ('concept-1', 5, 1)")
    state_conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count) VALUES ('carrier-1', 'family', 'family-1', 5)")
    state_conn.execute("INSERT INTO role_links (role_signature, linked_type, linked_key, support_count) VALUES ('role-1', 'carrier', 'carrier-1', 5)")
    state_conn.execute("INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count) VALUES ('concept-1', 'role', 'role-1', 5)")
    state_conn.execute(
        """INSERT INTO role_transfer_attempts (
            attempt_id, role_signature, transfer_kind, reuse_success, similarity_score,
            transfer_score, best_margin, source_carrier_count, candidate_role_count,
            source_role_signature, predicted_target_role_signature, observed_target_role_signature,
            source_game_key, target_game_key, source_carrier_signature, provenance_mode, provenance_status
        ) VALUES ('attempt-1', 'role-1', 'cross_game', 1, 0.9, 0.9, 0.2, 2, 2,
                  'role-1', 'role-1', 'role-1', 'g1', 'g2', 'carrier-1', 'single_source', 'verified')"""
    )
    state_conn.commit()
    return memory_dir, state_conn, graph_conn


def test_family_and_carrier_provenance_resolve_through_role_and_transfer(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        for index, (context, game) in enumerate((("ctx1", "g1"), ("ctx2", "g1"), ("ctx3", "g2")), start=1):
            _insert_event(
                state_conn,
                event_id=f"family-event-{index}",
                owner_type="family",
                owner_key="family-1",
                family="family-1",
                context=context,
                game=game,
            )
        _insert_event(
            state_conn,
            event_id="carrier-event",
            owner_type="carrier",
            owner_key="carrier-1",
            carrier="carrier-1",
        )
        state_conn.commit()
        summary = derive_future_option_motifs(state_conn, graph_conn, max_motifs=20)
        transfer = derive_future_option_transfer_links(state_conn)
        links = {
            (str(row["linked_type"]), str(row["linked_key"]))
            for row in state_conn.execute("SELECT linked_type, linked_key FROM future_option_links").fetchall()
        }
        carrier_motifs = [
            str(row[0])
            for row in state_conn.execute(
                "SELECT motif_signature FROM future_option_links WHERE linked_type = 'event' AND linked_key = 'carrier-event'"
            ).fetchall()
        ]
        family_motifs = [
            str(row[0])
            for row in state_conn.execute(
                "SELECT motif_signature FROM future_option_links WHERE linked_type = 'event' AND linked_key = 'family-event-1'"
            ).fetchall()
        ]
        assert ("motif_derived_from_family", "family-1") in links
        assert ("motif_expressed_by_carrier", "carrier-1") in links
        assert ("motif_associated_with_role", "role-1") in links
        assert ("motif_supports_concept", "concept-1") in links
        assert summary["motifs_with_family_provenance"] >= 1
        assert summary["motifs_with_carrier_provenance"] >= 1
        assert summary["motifs_with_role_provenance"] >= 1
        assert summary["motifs_with_concept_provenance"] >= 1
        assert summary["emergent_motifs_with_role_links"] >= 1
        assert transfer["emergent_motif_transfer_link_count"] >= 1
        assert carrier_motifs
        assert family_motifs
        assert all(
            state_conn.execute(
                "SELECT COUNT(*) FROM future_option_links WHERE motif_signature = ? AND linked_type = 'motif_associated_with_role' AND linked_key = 'role-1'",
                (motif_signature,),
            ).fetchone()[0]
            == 1
            for motif_signature in carrier_motifs + family_motifs
        )
        row_counts_before = {
            table: state_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("future_option_motifs", "future_option_links", "future_option_transfer_links")
        }
        derive_future_option_motifs(state_conn, graph_conn, max_motifs=20)
        derive_future_option_transfer_links(state_conn)
        row_counts_after = {
            table: state_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("future_option_motifs", "future_option_links", "future_option_transfer_links")
        }
        assert row_counts_after == row_counts_before
        assert state_conn.execute(
            "SELECT support_count FROM future_option_links WHERE motif_signature = ? AND linked_type = 'motif_associated_with_role' AND linked_key = 'role-1'",
            (family_motifs[0],),
        ).fetchone()[0] == 1
    finally:
        state_conn.close()
        graph_conn.close()


def test_transfer_metrics_distinguish_unique_roles_from_motif_role_associations(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        for motif_signature in ("motif-1", "motif-2"):
            state_conn.execute(
                """INSERT INTO future_option_motifs (
                    motif_signature, motif_type, support_count, motif_stability_score, is_emergent
                ) VALUES (?, 'enable', 3, 0.8, 1)""",
                (motif_signature,),
            )
            state_conn.execute(
                """INSERT INTO future_option_links (
                    motif_signature, linked_type, linked_key, support_count
                ) VALUES (?, 'motif_associated_with_role', 'role-1', 1)""",
                (motif_signature,),
            )
        state_conn.commit()
        summary = derive_future_option_transfer_links(state_conn)
        assert summary["unique_roles_seen_from_motif_links"] == 1
        assert summary["unique_roles_with_transfer_attempts"] == 1
        assert summary["unique_roles_with_concepts"] == 1
        assert summary["motif_role_link_count"] == 2
        assert summary["motif_role_transfer_attempt_link_count"] == 2
        assert summary["motif_role_concept_link_count"] == 2
    finally:
        state_conn.close()
        graph_conn.close()


def test_event_derivation_persists_normalized_contingency_provenance(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        state_conn.execute(
            """INSERT INTO stable_contingencies (
                canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score
            ) VALUES ('ctx-provenance|a7', 'game-provenance', 'sampler-provenance', 1, 7,
                      'enable', 5, 10, 12, 0.8)"""
        )
        state_conn.execute(
            "INSERT INTO transformation_families (canonical_signature, support_count, stability_score) VALUES ('family-provenance', 5, 0.8)"
        )
        state_conn.execute(
            "INSERT INTO family_members (family_signature, contingency_key, support_count) VALUES ('family-provenance', 'ctx-provenance|a7', 5)"
        )
        state_conn.commit()
        derive_future_option_events(state_conn, graph_conn, max_events=20)
        row = state_conn.execute(
            """SELECT source_family_id, source_context_signature, source_action, source_game_id, source_sampler
               FROM future_option_events WHERE owner_type = 'contingency'"""
        ).fetchone()
        assert tuple(row) == ("family-provenance", "ctx-provenance", "7", "game-provenance", "sampler-provenance")
    finally:
        state_conn.close()
        graph_conn.close()


def test_missing_provenance_never_invents_links_and_propagation_is_idempotent(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        _insert_event(
            state_conn,
            event_id="unlinked-family-event",
            owner_type="family",
            owner_key="family-without-links",
            family="family-without-links",
        )
        state_conn.commit()
        first = derive_future_option_motifs(state_conn, graph_conn, max_motifs=20)
        explicit_before = state_conn.execute(
            "SELECT COUNT(*) FROM future_option_links WHERE linked_type = 'motif_associated_with_role'"
        ).fetchone()[0]
        derive_future_option_motifs(state_conn, graph_conn, max_motifs=20)
        explicit_after = state_conn.execute(
            "SELECT COUNT(*) FROM future_option_links WHERE linked_type = 'motif_associated_with_role'"
        ).fetchone()[0]
        invented = state_conn.execute(
            "SELECT COUNT(*) FROM future_option_links WHERE linked_key = 'role-1'"
        ).fetchone()[0]
        assert first["provenance_resolution_failures"] >= 1
        assert first["unresolved_family_to_carrier_count"] >= 1
        assert first["unresolved_carrier_to_role_count"] == 0
        assert first["unresolved_role_to_concept_count"] == 0
        assert first["failure_occurrence_count"] >= first["unique_unresolved_id_count"]
        assert explicit_before == explicit_after == 0
        assert invented == 0
    finally:
        state_conn.close()
        graph_conn.close()
