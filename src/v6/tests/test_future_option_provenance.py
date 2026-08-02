from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.future_options import (
    _resolve_motif_transfer_provenance,
    derive_future_option_events,
    derive_future_option_motifs,
    derive_future_option_transfer_links,
)
from v6.hypothesis_h11_report import evaluate_h11_future_option_transfer_concepts
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
    conn.execute(
        """UPDATE future_option_events
           SET source_interaction_id = ?, classification_source = 'structured_effect',
               classification_rule = 'structural_effect', classification_evidence_id = ?
           WHERE event_id = ?""",
        (event_id, event_id, event_id),
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
            source_game_key, target_game_key, source_context_key, target_context_key,
            source_interaction_id, target_interaction_id,
            source_game_is_surrogate, target_game_is_surrogate,
            source_context_is_surrogate, target_context_is_surrogate,
            source_game_resolution_source, target_game_resolution_source,
            source_context_resolution_source, target_context_resolution_source,
            source_carrier_signature, provenance_mode, provenance_status
        ) VALUES ('attempt-1', 'role-1', 'cross_game', 1, 0.9, 0.9, 0.2, 2, 2,
                  'role-1', 'role-1', 'role-1', 'g1', 'g2', 'ctx1', 'ctx2',
                  'source-interaction-1', 'target-interaction-1', 0, 0, 0, 0,
                  'interaction', 'interaction', 'interaction', 'interaction',
                  'carrier-1', 'single_source', 'verified')"""
    )
    state_conn.commit()
    return memory_dir, state_conn, graph_conn


def _seed_verified_transfer_motif(state_conn: sqlite3.Connection, *, motif_signature: str = "motif-verified") -> None:
    state_conn.execute(
        """UPDATE concept_candidates
           SET is_promoted = 1, promotion_status = 'retained', validation_status = 'passed'
           WHERE concept_signature = 'concept-1'"""
    )
    state_conn.execute(
        """INSERT INTO future_option_motifs (
            motif_signature, motif_type, support_count, motif_stability_score, is_emergent,
            source_interaction_ids_json
        ) VALUES (?, 'enable', 3, 0.8, 1, '[\"interaction-1\"]')""",
        (motif_signature,),
    )
    state_conn.execute(
        """INSERT INTO future_option_links (motif_signature, linked_type, linked_key, support_count)
           VALUES (?, 'motif_associated_with_role', 'role-1', 1)""",
        (motif_signature,),
    )
    state_conn.commit()


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


def test_h11_retained_persistent_concept_state_remains_verified(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        state_conn.execute(
            """UPDATE concept_candidates
               SET is_promoted = 0, promotion_status = 'candidate', validation_status = NULL
               WHERE concept_signature = 'concept-1'"""
        )
        state_conn.execute(
            """INSERT INTO concept_promotion_state (
                concept_signature, historically_promoted, currently_promoted,
                promotion_status, validation_status, first_promoted_global_step
            ) VALUES ('concept-1', 1, 1, 'retained', 'passed', 3)"""
        )
        _seed_verified_transfer_motif(state_conn)
        derive_future_option_transfer_links(state_conn)
        row = state_conn.execute(
            """SELECT concept_validation_status, concept_resolution_mode
               FROM future_option_transfer_links"""
        ).fetchone()
        assert tuple(row) == ("verified", "direct_role")
    finally:
        state_conn.close()
        graph_conn.close()


def test_h11_indirect_role_concept_resolution_uses_shared_carrier_only_without_direct_link(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        state_conn.execute(
            "DELETE FROM concept_links WHERE concept_signature = 'concept-1' AND linked_type = 'role'"
        )
        state_conn.execute(
            """INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count)
               VALUES ('concept-1', 'carrier', 'carrier-1', 1)"""
        )
        _seed_verified_transfer_motif(state_conn)
        summary = derive_future_option_transfer_links(state_conn)
        row = state_conn.execute(
            """SELECT concept_signature, concept_resolution_mode, shared_carrier_count,
                      shared_family_count, concept_validation_status
               FROM future_option_transfer_links"""
        ).fetchone()
        assert tuple(row) == ("concept-1", "shared_carrier", 1, 0, "verified")
        assert summary["roles_resolved_via_shared_carrier"] == 1
        assert summary["h11_links_using_indirect_concept_resolution"] == 1
        assert summary["indirect_verified_chain_count"] == 1
    finally:
        state_conn.close()
        graph_conn.close()


def test_h11_indirect_role_concept_resolution_uses_shared_family_deterministically(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        state_conn.execute(
            "DELETE FROM concept_links WHERE concept_signature = 'concept-1' AND linked_type = 'role'"
        )
        state_conn.execute(
            """INSERT INTO role_links (role_signature, linked_type, linked_key, support_count)
               VALUES ('role-1', 'family', 'family-1', 1)"""
        )
        state_conn.execute(
            """INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count)
               VALUES ('concept-1', 'family', 'family-1', 1)"""
        )
        state_conn.execute("UPDATE concept_candidates SET promotion_score = 1.0 WHERE concept_signature = 'concept-1'")
        state_conn.execute(
            """INSERT INTO concept_candidates (concept_signature, support_count, is_promoted,
                                                  promotion_status, validation_status, promotion_score)
               VALUES ('concept-z', 1, 1, 'promoted', 'passed', 1.0)"""
        )
        state_conn.execute(
            """INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count)
               VALUES ('concept-z', 'family', 'family-1', 1)"""
        )
        _seed_verified_transfer_motif(state_conn)
        derive_future_option_transfer_links(state_conn)
        row = state_conn.execute(
            """SELECT concept_signature, concept_resolution_mode, shared_family_count
               FROM future_option_transfer_links"""
        ).fetchone()
        # Identical structural evidence is resolved by lexical concept signature.
        assert tuple(row) == ("concept-1", "shared_family", 1)
    finally:
        state_conn.close()
        graph_conn.close()


def test_h11_does_not_resolve_unrelated_indirect_concept(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        state_conn.execute(
            "DELETE FROM concept_links WHERE concept_signature = 'concept-1' AND linked_type = 'role'"
        )
        state_conn.execute(
            """INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count)
               VALUES ('concept-1', 'family', 'unrelated-family', 1)"""
        )
        _seed_verified_transfer_motif(state_conn)
        summary = derive_future_option_transfer_links(state_conn)
        row = state_conn.execute(
            """SELECT concept_signature, concept_resolution_mode
               FROM future_option_transfer_links"""
        ).fetchone()
        assert tuple(row) == ("__none__", "missing")
        assert summary["roles_still_without_concept"] == 1
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


def test_h11_retained_concept_and_concrete_pairs_remain_verified(tmp_path: Path) -> None:
    memory_dir, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        _seed_verified_transfer_motif(state_conn)
        state_conn.execute(
            """INSERT INTO role_transfer_attempts (
                attempt_id, role_signature, transfer_kind, reuse_success, similarity_score,
                transfer_score, best_margin, source_carrier_count, source_evidence_support_count,
                candidate_role_count, source_role_signature, predicted_target_role_signature,
                observed_target_role_signature, source_game_key, target_game_key,
                source_context_key, target_context_key, source_carrier_signature,
                provenance_mode, provenance_status
            ) VALUES ('attempt-2', 'role-1', 'cross_game', 1, 0.9, 0.9, 0.2, 1, 2, 2,
                      'role-1', 'role-1', 'role-1', 'g3', 'g4', 'ctx3', 'ctx4',
                      'carrier-1', 'single_source', 'verified')"""
        )
        state_conn.commit()
        summary = derive_future_option_transfer_links(state_conn)
        rows = state_conn.execute(
            """SELECT source_game_key, target_game_key, provenance_mode,
                      transfer_provenance_status, concept_validation_status
               FROM future_option_transfer_links
               WHERE motif_signature = 'motif-verified'
               ORDER BY source_game_key"""
        ).fetchall()
        assert len(rows) == 2
        assert {(row[0], row[1]) for row in rows} == {("g1", "g2"), ("g3", "g4")}
        assert all(tuple(row[2:]) == ("single_source", "verified", "verified") for row in rows)
        assert summary["verified_concrete_transfer_link_count"] == 2
        assert summary["verified_transfer_pair_count"] == 2
        state_conn.commit()
        report = evaluate_h11_future_option_transfer_concepts(
            memory_dir=memory_dir,
            run_dir=None,
            output_dir=tmp_path / "h11",
            already_derived=True,
        )
        assert report["verified_future_option_transfer_count"] == 2
        assert report["verified_cross_game_pair_count"] == 2
        assert report["decision"] != "INSUFFICIENT_EVIDENCE"
    finally:
        state_conn.close()
        graph_conn.close()


def test_h11_concept_validation_statuses_are_explicit(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        _seed_verified_transfer_motif(state_conn)
        derive_future_option_transfer_links(state_conn)
        assert state_conn.execute(
            "SELECT concept_validation_status FROM future_option_transfer_links"
        ).fetchone()[0] == "verified"

        state_conn.execute(
            """UPDATE concept_candidates
               SET is_promoted = 0, promotion_status = 'candidate', validation_status = 'passed'
               WHERE concept_signature = 'concept-1'"""
        )
        state_conn.commit()
        derive_future_option_transfer_links(state_conn)
        assert state_conn.execute(
            "SELECT concept_validation_status FROM future_option_transfer_links"
        ).fetchone()[0] == "proxy"

        state_conn.execute(
            """UPDATE concept_candidates
               SET is_promoted = 0, promotion_status = 'demoted', validation_status = 'demoted'
               WHERE concept_signature = 'concept-1'"""
        )
        state_conn.commit()
        derive_future_option_transfer_links(state_conn)
        assert state_conn.execute(
            "SELECT concept_validation_status FROM future_option_transfer_links"
        ).fetchone()[0] == "demoted"
    finally:
        state_conn.close()
        graph_conn.close()


def test_motif_provenance_resolves_direct_and_structural_paths(tmp_path: Path) -> None:
    _, state_conn, graph_conn = _provenance_memory(tmp_path)
    try:
        _insert_event(
            state_conn,
            event_id="family-provenance-event",
            owner_type="family",
            owner_key="family-1",
            family="family-1",
        )
        state_conn.commit()
        direct = _resolve_motif_transfer_provenance(
            state_conn,
            motif_signature="direct",
            motif_links={"event": {"family-provenance-event"}},
            direct_interaction_ids=set(),
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
        missing = _resolve_motif_transfer_provenance(
            state_conn,
            motif_signature="missing",
            motif_links={},
            direct_interaction_ids=set(),
        )
        assert (direct["status"], direct["resolution_path"]) == ("verified", "motif_to_interaction")
        assert (family["status"], family["resolution_path"]) == ("verified", "motif_to_family_to_interaction")
        assert (carrier["status"], carrier["resolution_path"]) == (
            "verified", "motif_to_carrier_to_family_to_interaction",
        )
        assert (role["status"], role["resolution_path"]) == (
            "verified", "motif_to_role_to_carrier_to_family_to_interaction",
        )
        assert missing["status"] == "missing"
    finally:
        state_conn.close()
        graph_conn.close()


def _seed_many_h11_links(memory_dir: Path, *, count: int = 205) -> None:
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        for index in range(count):
            motif = f"motif-{index:04d}"
            conn.execute(
                """INSERT INTO future_option_motifs (
                    motif_signature, motif_type, support_count, motif_stability_score, is_emergent
                ) VALUES (?, 'enable', 3, 0.8, 1)""",
                (motif,),
            )
            conn.execute(
                """INSERT INTO future_option_transfer_links (
                    motif_signature, role_signature, concept_signature,
                    transfer_attempt_count, successful_transfer_count,
                    strong_transfer_success_count, promoted_concept_count,
                    mean_transfer_score, mean_best_margin, source_role_signature,
                    source_game_key, target_game_key, source_context_key, target_context_key,
                    provenance_mode, motif_provenance_status,
                    transfer_provenance_status, concept_validation_status,
                    motif_provenance_resolution_path, first_seen_global_step, last_seen_global_step
                ) VALUES (?, 'role-1', 'concept-1', 1, 1, 1, 1, 0.9, 0.2, 'role-1',
                          'source-game', 'target-game', 'a very long source context key',
                          'a very long target context key', 'single_source', 'verified',
                          'verified', 'verified', 'motif_to_interaction', 1, 2)""",
                (motif,),
            )


def test_h11_streams_full_provenance_and_bounds_main_report(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_many_h11_links(memory_dir)
    output_dir = tmp_path / "h11"
    result = evaluate_h11_future_option_transfer_concepts(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=output_dir,
        already_derived=True,
        max_main_report_bytes=1_000_000,
    )
    main = json.loads((output_dir / "h11_future_option_transfer_concepts_report.json").read_text(encoding="utf-8"))
    provenance_rows = [json.loads(line) for line in (output_dir / "h11_transfer_chain_provenance.jsonl").read_text(encoding="utf-8").splitlines()]
    context_rows = [json.loads(line) for line in (output_dir / "h11_context_lookup.jsonl").read_text(encoding="utf-8").splitlines()]
    game_pairs = [json.loads(line) for line in (output_dir / "h11_transfer_by_game_pair.jsonl").read_text(encoding="utf-8").splitlines()]
    context_pairs = [json.loads(line) for line in (output_dir / "h11_transfer_by_context_pair.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["future_option_transfer_link_count"] == 205
    assert main["motif_transfer_chain_provenance_total_count"] == 205
    assert main["motif_transfer_chain_provenance_sample_count"] == 200
    assert main["motif_transfer_chain_provenance_truncated"] is True
    assert main["motif_transfer_chain_provenance_is_sample"] is True
    assert len(main["motif_transfer_chain_provenance"]) == 200
    assert [row["motif_signature"] for row in main["motif_transfer_chain_provenance_sample"][:3]] == [
        "motif-0000", "motif-0001", "motif-0002"
    ]
    assert len(provenance_rows) == 205
    assert len(context_rows) == 2
    assert len({row["transfer_pair_id"] for row in provenance_rows}) == 1
    assert len(game_pairs) == len(context_pairs) == 1
    assert game_pairs[0]["link_count"] == context_pairs[0]["link_count"] == 205
    assert all(
        row["source_context_id"] in {item["context_id"] for item in context_rows}
        and row["target_context_id"] in {item["context_id"] for item in context_rows}
        for row in provenance_rows
    )
    assert (output_dir / "h11_future_option_transfer_concepts_report.json").stat().st_size <= 1_000_000


def test_h11_detail_controls_preserve_decision_and_counters(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_many_h11_links(memory_dir, count=3)
    detailed = evaluate_h11_future_option_transfer_concepts(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=tmp_path / "detailed",
        already_derived=True,
    )
    sampled = evaluate_h11_future_option_transfer_concepts(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=tmp_path / "sampled",
        already_derived=True,
        provenance_sample_limit=0,
        write_full_provenance_jsonl=False,
    )
    for key in (
        "decision",
        "future_option_transfer_link_count",
        "verified_future_option_transfer_count",
        "verified_cross_game_pair_count",
        "fully_verified_emergent_chain_count",
    ):
        assert sampled[key] == detailed[key]
    assert sampled["motif_transfer_chain_provenance_sample"] == []
    assert sampled["motif_transfer_chain_provenance_sample_count"] == 0
    assert not (tmp_path / "sampled" / "h11_transfer_chain_provenance.jsonl").exists()
    assert (tmp_path / "sampled" / "h11_transfer_by_game_pair.jsonl").exists()
    assert (tmp_path / "sampled" / "h11_transfer_by_context_pair.jsonl").exists()
