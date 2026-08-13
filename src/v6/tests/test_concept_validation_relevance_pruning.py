from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6 import concept_validation_fastpath as fast
from v6 import higher_order_substrate as substrate
from v6.memory.compact_memory import ensure_memory_layout


def _seed_candidate(memory_dir: Path) -> None:
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        for role in ("role-a", "role-b"):
            conn.execute(
                """
                INSERT INTO role_candidates (
                    role_signature, linked_carrier_count, linked_family_count,
                    linked_context_count, cross_game_count, support_count
                ) VALUES (?, 1, 1, 1, 1, 2)
                """,
                (role,),
            )
            for linked_type, linked_key in (
                ("carrier", f"carrier-{role}"),
                ("family", f"family-{role}"),
                ("context", "ctx-shared"),
                ("game", f"game-{role}"),
            ):
                conn.execute(
                    "INSERT INTO role_links (role_signature, linked_type, linked_key, support_count) VALUES (?, ?, ?, 1)",
                    (role, linked_type, linked_key),
                )
            conn.execute(
                "INSERT INTO transformation_families (canonical_signature, prediction_lift, last_seen_global_step) VALUES (?, 0.2, 5)",
                (f"family-{role}",),
            )
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (
                    attempt_id, role_signature, source_role_signature,
                    predicted_target_role_signature, observed_target_role_signature,
                    source_game_key, target_game_key, source_context_key, target_context_key,
                    source_carrier_signature, target_carrier_signature,
                    provenance_mode, provenance_status, reuse_success, last_seen_global_step
                ) VALUES (?, ?, ?, ?, ?, ?, 'target-game', 'ctx-shared', 'ctx-target', ?, ?,
                          'single_source', 'verified', 1, 5)
                """,
                (
                    f"history-{role}", role, role, role, role, f"game-{role}",
                    f"carrier-{role}", f"target-carrier-{role}",
                ),
            )
        conn.execute(
            """
            INSERT INTO concept_candidates (
                concept_signature, compression_gain, explanatory_reach, promotion_score,
                cross_context_count, cross_game_count, first_seen_global_step, is_promoted
            ) VALUES ('concept-prune', 1.0, 4.0, 0.9, 2, 2, 10, 0)
            """
        )
        for role in ("role-a", "role-b"):
            conn.execute(
                "INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count) VALUES ('concept-prune', 'role', ?, 1)",
                (role,),
            )
        conn.commit()


def _diagnostic(memory_dir: Path) -> dict:
    substrate.validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=substrate.IncrementalPromotionValidationConfig(
            enabled=True,
            min_relevant_heldout_event_count=1,
        ),
        validate_roles_and_concepts=True,
        validate_world_models=False,
        diagnostic_epoch_id="epoch-prune",
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        payload = conn.execute(
            "SELECT payload_json FROM concept_promotion_validation_diagnostics WHERE concept_signature='concept-prune'"
        ).fetchone()[0]
    return json.loads(payload)


def test_unrelated_transfer_rows_are_counted_but_not_scored(monkeypatch, tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_candidate(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO role_transfer_attempts (
                attempt_id, role_signature, source_role_signature,
                predicted_target_role_signature, observed_target_role_signature,
                source_game_key, target_game_key, source_context_key, target_context_key,
                source_carrier_signature, target_carrier_signature,
                provenance_mode, provenance_status, reuse_success, last_seen_global_step
            ) VALUES ('relevant', 'role-a', 'role-a', 'role-target', 'role-target',
                      'game-role-a', 'target-game', 'ctx-shared', 'ctx-target',
                      'carrier-role-a', 'target-carrier', 'single_source', 'verified', 1, 20)
            """
        )
        for index in range(200):
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (
                    attempt_id, role_signature, source_role_signature,
                    predicted_target_role_signature, observed_target_role_signature,
                    source_game_key, target_game_key, source_context_key, target_context_key,
                    source_carrier_signature, target_carrier_signature,
                    provenance_mode, provenance_status, reuse_success, last_seen_global_step
                ) VALUES (?, 'role-unrelated', 'role-unrelated', 'target-unrelated', 'target-unrelated',
                          'other-game', 'other-target', 'other-context', 'other-target-context',
                          'other-carrier', 'other-target-carrier', 'single_source', 'verified', 0, ?)
                """,
                (f"unrelated-{index}", 21 + index),
            )
        conn.commit()

    original = fast._role_score_bundle
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fast, "_role_score_bundle", counted)
    diagnostic = _diagnostic(memory_dir)

    assert diagnostic["relevant_heldout_event_count"] == 1
    assert diagnostic["total_later_event_count"] == 201
    assert diagnostic["unrelated_event_count"] == 200
    assert diagnostic["prefiltered_unrelated_event_count"] == 200
    assert diagnostic["prefiltered_unrelated_event_type_counts"] == {"transfer": 200}
    assert diagnostic["materialized_later_event_count"] == 1
    assert calls == 1


def test_unrelated_prediction_rows_do_not_trigger_role_scoring(monkeypatch, tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_candidate(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            CREATE TABLE prediction_results (
                id INTEGER PRIMARY KEY, global_step INTEGER, context_signature TEXT,
                predicted_family TEXT, actual_family TEXT, context_contradiction INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO prediction_results VALUES (1, 20, 'ctx-shared', 'family-role-a', 'other-family', 0)"
        )
        for index in range(200):
            conn.execute(
                "INSERT INTO prediction_results VALUES (?, ?, 'ctx-shared', 'unrelated-family', 'other-unrelated', 0)",
                (index + 2, index + 21),
            )
        conn.commit()

    original = fast._role_score_bundle
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fast, "_role_score_bundle", counted)
    diagnostic = _diagnostic(memory_dir)

    assert diagnostic["relevant_prediction_event_count"] == 1
    assert diagnostic["total_later_event_count"] == 201
    assert diagnostic["unrelated_event_count"] == 200
    assert diagnostic["prefiltered_unrelated_event_type_counts"] == {"prediction": 200}
    assert diagnostic["context_only_overlap_count"] == 200
    assert diagnostic["materialized_later_event_count"] == 1
    assert calls == 1
