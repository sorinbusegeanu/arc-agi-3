from __future__ import annotations

import json
import sqlite3

from v6 import future_options as fo
from v6.suite_runtime_repairs import (
    _derive_future_option_transfer_links_bounded,
    _match_world_model_predictions_scope_aware,
)


def _h08_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE future_option_events (
            event_id TEXT PRIMARY KEY, owner_type TEXT, owner_key TEXT,
            source_family_id TEXT, first_seen_global_step INTEGER,
            last_seen_global_step INTEGER, context_key TEXT, game TEXT,
            motif_type TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE world_model_prediction_events (
            prediction_event_id TEXT PRIMARY KEY, component_signature TEXT,
            prediction_global_step INTEGER, predicted_family TEXT,
            predicted_effect TEXT, predicted_outcome TEXT, game_key TEXT,
            context_key TEXT, action_key TEXT, baseline_prediction_score REAL,
            component_prediction_score REAL, observed_event_id TEXT,
            observed_global_step INTEGER, observed_family TEXT,
            observed_effect TEXT, prediction_correct INTEGER,
            provenance_status TEXT
        )
        """
    )
    return conn


def test_component_scope_prediction_survives_unrelated_progress_then_matches() -> None:
    conn = _h08_connection()
    payload = json.dumps(
        {
            "kind": "next_family",
            "family": "fam-a",
            "contexts": ["ctx-a", "ctx-b"],
            "games": ["game-a"],
            "scope_version": "component_scope_v2",
        }
    )
    conn.execute(
        """
        INSERT INTO world_model_prediction_events (
            prediction_event_id, component_signature, prediction_global_step,
            predicted_family, predicted_outcome, provenance_status
        ) VALUES ('pred', 'wm:test', 100, 'fam-a', ?, 'prospective')
        """,
        (payload,),
    )
    conn.execute(
        "INSERT INTO future_option_events VALUES "
        "('outside','family','fam-z','fam-z',101,200,'ctx-z','game-z','branch')"
    )
    _match_world_model_predictions_scope_aware(conn, "wm:test")
    assert conn.execute(
        "SELECT provenance_status FROM world_model_prediction_events WHERE prediction_event_id='pred'"
    ).fetchone()[0] == "prospective"

    conn.execute(
        "INSERT INTO future_option_events VALUES "
        "('inside','family','fam-a','fam-a',90,250,'ctx-b','game-a','branch')"
    )
    _match_world_model_predictions_scope_aware(conn, "wm:test")
    row = conn.execute(
        "SELECT observed_event_id, observed_global_step, prediction_correct, provenance_status "
        "FROM world_model_prediction_events WHERE prediction_event_id='pred'"
    ).fetchone()
    assert row == ("inside", 250, 1, "verified")


def _transfer_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE future_option_motifs (
            motif_signature TEXT PRIMARY KEY, is_emergent INTEGER,
            support_count INTEGER, motif_stability_score REAL,
            source_interaction_ids_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE role_transfer_attempts (
            attempt_id TEXT PRIMARY KEY, source_role_signature TEXT,
            transfer_score REAL, best_margin REAL, reuse_success INTEGER,
            similarity_score REAL, source_evidence_support_count INTEGER,
            candidate_role_count INTEGER, source_game_key TEXT,
            target_game_key TEXT, source_context_key TEXT,
            target_context_key TEXT, source_interaction_id TEXT,
            target_interaction_id TEXT, source_game_is_surrogate INTEGER,
            target_game_is_surrogate INTEGER, source_context_is_surrogate INTEGER,
            target_context_is_surrogate INTEGER, source_game_resolution_source TEXT,
            target_game_resolution_source TEXT, source_context_resolution_source TEXT,
            target_context_resolution_source TEXT, provenance_mode TEXT,
            provenance_status TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE future_option_transfer_links (
            motif_signature TEXT, role_signature TEXT, concept_signature TEXT,
            transfer_attempt_count INTEGER, successful_transfer_count INTEGER,
            strong_transfer_success_count INTEGER, promoted_concept_count INTEGER,
            mean_transfer_score REAL, mean_best_margin REAL,
            source_role_signature TEXT, source_game_key TEXT, target_game_key TEXT,
            source_context_key TEXT, target_context_key TEXT,
            source_interaction_id TEXT, target_interaction_id TEXT,
            source_game_is_surrogate INTEGER, target_game_is_surrogate INTEGER,
            source_context_is_surrogate INTEGER, target_context_is_surrogate INTEGER,
            source_game_resolution_source TEXT, target_game_resolution_source TEXT,
            source_context_resolution_source TEXT, target_context_resolution_source TEXT,
            transfer_scope TEXT, provenance_mode TEXT, motif_provenance_status TEXT,
            transfer_provenance_status TEXT, concept_validation_status TEXT,
            motif_provenance_resolution_path TEXT, motif_resolved_interaction_count INTEGER,
            motif_resolved_family_count INTEGER, motif_resolved_carrier_count INTEGER,
            motif_resolved_role_count INTEGER, motif_resolved_concept_count INTEGER,
            concept_resolution_mode TEXT, concept_resolution_path TEXT,
            shared_carrier_count INTEGER, shared_family_count INTEGER,
            first_seen_global_step INTEGER, last_seen_global_step INTEGER
        )
        """
    )
    for index in range(10):
        conn.execute(
            "INSERT INTO future_option_motifs VALUES (?,1,10,0.9,'[]')",
            (f"m{index}",),
        )
    rows = [
        ("a1", "role-a", "g1", "g2", "c1", "c2"),
        ("a2", "role-a", "g1", "g2", "c3", "c4"),
    ]
    for attempt_id, role, sg, tg, sc, tc in rows:
        conn.execute(
            """
            INSERT INTO role_transfer_attempts VALUES (
                ?, ?, 0.8, 0.2, 1, 0.8, 5, 3, ?, ?, ?, ?,
                'i1','i2',0,0,0,0,'direct','direct','direct','direct',
                'single_source','verified'
            )
            """,
            (attempt_id, role, sg, tg, sc, tc),
        )
    return conn


def test_large_transfer_derivation_aggregates_exact_pairs_by_scope(monkeypatch) -> None:
    conn = _transfer_connection()
    monkeypatch.setenv("ARC_H11_COMPACT_TRANSFER_THRESHOLD", "1")

    def links(_conn, table, _signature):
        if table == "future_option_links":
            return {"m0": {"role": {"role-a"}}}
        if table == "concept_links":
            return {}
        return {}

    monkeypatch.setattr(fo, "_links_by_signature", links)
    monkeypatch.setattr(
        fo,
        "_concept_validation_records",
        lambda _conn: {"concept-a": {"status": "verified"}},
    )
    monkeypatch.setattr(
        fo,
        "_resolve_concepts_for_roles",
        lambda _conn, **_kwargs: {
            "role-a": [{
                "concept_signature": "concept-a",
                "mode": "direct_role",
                "path": "role_to_concept",
                "shared_carrier_count": 0,
                "shared_family_count": 0,
                "status": "verified",
            }]
        },
    )
    monkeypatch.setattr(
        fo,
        "_resolve_motif_transfer_provenance",
        lambda _conn, **_kwargs: {
            "status": "verified",
            "resolution_path": "motif_to_interaction",
            "interaction_ids": {"i1"},
            "family_ids": {"f1"},
            "carrier_ids": {"k1"},
            "role_ids": {"role-a"},
            "concept_ids": {"concept-a"},
        },
    )
    monkeypatch.setattr(fo, "_safe_min", lambda *_args: 1)

    result = _derive_future_option_transfer_links_bounded(conn)
    assert result["future_option_transfer_compaction_applied"] is True
    row = conn.execute(
        "SELECT transfer_scope, transfer_attempt_count, successful_transfer_count "
        "FROM future_option_transfer_links"
    ).fetchone()
    assert tuple(row) == ("cross_game_and_context", 2, 2)
    assert result["future_option_transfer_link_count"] == 1
