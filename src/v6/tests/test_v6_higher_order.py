from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from v6.continuous_research import _format_epoch_status
from v6 import higher_order_substrate
from v6.higher_order_substrate import (
    IncrementalPromotionValidationConfig,
    _derive_role_transfer_attempts_chunk,
    _predict_transfer_attempt,
    derive_higher_order_memory,
    derive_role_candidates_only,
    derive_role_transfer_attempts_only,
    validate_incremental_promotions_only,
)
from v6.hypothesis_h05_report import evaluate_h05_role_emergence
from v6.hypothesis_h06_report import evaluate_h06_role_transfer
from v6.hypothesis_h07_report import evaluate_h07_concept_emergence
from v6.hypothesis_h08_report import evaluate_h08_world_model_coherence
from v6.hypothesis_h09_report import evaluate_h09_future_option_motifs
from v6.hypothesis_h10_report import evaluate_h10_future_option_attention
from v6.hypothesis_h11_report import evaluate_h11_future_option_transfer_concepts
from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention
from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence
from v6.hypothesis_suite_report import (
    _apply_epoch_maturity_gates,
    _apply_higher_order_dependency_gates,
    build_hypothesis_suite_summary,
    run_hypothesis_suite_report,
)
from v6.memory.compact_memory import ensure_memory_layout


class _ThreadPoolCompat(ThreadPoolExecutor):
    def __init__(self, max_workers=None):
        super().__init__(max_workers=max_workers)


def _seed_set_coverage_candidate(
    memory_dir: Path,
    *,
    concept_signature: str = "concept-set",
    extra_family_count: int = 0,
    legacy_promoted: int = 0,
    cross_scope_count: int = 2,
    promotion_score: float = 0.9,
    include_heldout: bool = True,
) -> None:
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        conn.execute(
            """
            INSERT INTO role_candidates (
                role_signature, linked_carrier_count, linked_family_count, linked_context_count,
                cross_game_count, support_count
            ) VALUES ('role-source', 1, 8, 0, 0, 1)
            """
        )
        for index in range(8):
            family = f"family-source-{index:02d}"
            conn.execute(
                "INSERT INTO role_links (role_signature, linked_type, linked_key, support_count) VALUES ('role-source', 'family', ?, 1)",
                (family,),
            )
            conn.execute(
                "INSERT INTO transformation_families (canonical_signature, prediction_lift, last_seen_global_step) VALUES (?, 0.4, 20)",
                (family,),
            )
        conn.execute(
            """
            INSERT INTO concept_candidates (
                concept_signature, compression_gain, explanatory_reach, promotion_score,
                cross_context_count, cross_game_count, first_seen_global_step, is_promoted
            ) VALUES (?, 2.0, 8.0, ?, ?, ?, 10, ?)
            """,
            (concept_signature, promotion_score, cross_scope_count, cross_scope_count, legacy_promoted),
        )
        conn.execute(
            "INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count) VALUES (?, 'role', 'role-source', 1)",
            (concept_signature,),
        )
        for index in range(extra_family_count):
            conn.execute(
                "INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count) VALUES (?, 'family', ?, 1)",
                (concept_signature, f"family-concept-{index:02d}"),
            )
        conn.execute(
            "INSERT INTO role_transfer_attempts (attempt_id, role_signature, reuse_success, last_seen_global_step) VALUES ('derive', 'role-source', 0, 5)"
        )
        if include_heldout:
            conn.execute(
                "INSERT INTO role_transfer_attempts (attempt_id, role_signature, reuse_success, last_seen_global_step) VALUES ('heldout', 'role-source', 1, 20)"
            )
        conn.commit()


def _coverage_diagnostic(memory_dir: Path, *, epoch_id: str) -> dict:
    validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=IncrementalPromotionValidationConfig(enabled=True),
        validate_roles_and_concepts=True,
        validate_world_models=False,
        diagnostic_epoch_id=epoch_id,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        payload_json = conn.execute(
            "SELECT payload_json FROM concept_promotion_validation_diagnostics ORDER BY concept_signature ASC LIMIT 1"
        ).fetchone()[0]
    return json.loads(payload_json)


def _seed_functional_candidate(
    memory_dir: Path,
    *,
    concept_signature: str = "concept-functional",
    legacy_promoted: int = 0,
    historical_rates: tuple[tuple[int, int], tuple[int, int]] = ((1, 1), (1, 1)),
    heldout_success: int | None = 1,
    include_prediction: bool = False,
) -> None:
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        for index, role in enumerate(("role-a", "role-b")):
            conn.execute(
                """
                INSERT INTO role_candidates (
                    role_signature, linked_carrier_count, linked_family_count, linked_context_count,
                    cross_game_count, support_count
                ) VALUES (?, 1, 1, 1, 1, 2)
                """,
                (role,),
            )
            for linked_type, linked_key in (
                ("carrier", f"carrier-{role}"),
                ("family", f"family-{role}"),
                ("context", "ctx-functional"),
                ("game", f"game-{role}"),
            ):
                conn.execute(
                    "INSERT INTO role_links (role_signature, linked_type, linked_key, support_count) VALUES (?, ?, ?, 1)",
                    (role, linked_type, linked_key),
                )
            successes, failures = historical_rates[index]
            for attempt_index in range(successes):
                conn.execute(
                    """
                    INSERT INTO role_transfer_attempts (
                        attempt_id, role_signature, target_scope_type, target_scope_key,
                        reuse_success, last_seen_global_step
                    ) VALUES (?, ?, 'game', 'target-game', 1, 5)
                    """,
                    (f"{role}-success-{attempt_index}", role),
                )
            for attempt_index in range(failures):
                conn.execute(
                    """
                    INSERT INTO role_transfer_attempts (
                        attempt_id, role_signature, target_scope_type, target_scope_key,
                        reuse_success, last_seen_global_step
                    ) VALUES (?, ?, 'game', 'target-game', 0, 6)
                    """,
                    (f"{role}-failure-{attempt_index}", role),
                )
        conn.execute(
            """
            INSERT INTO concept_candidates (
                concept_signature, compression_gain, explanatory_reach, promotion_score,
                cross_context_count, cross_game_count, first_seen_global_step, is_promoted
            ) VALUES (?, 0.0, 0.0, 0.9, 2, 2, 10, ?)
            """,
            (concept_signature, legacy_promoted),
        )
        for role in ("role-a", "role-b"):
            conn.execute(
                "INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count) VALUES (?, 'role', ?, 1)",
                (concept_signature, role),
            )
        if heldout_success is not None:
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (
                    attempt_id, role_signature, observed_role_signature, target_scope_type, target_scope_key,
                    reuse_success, last_seen_global_step
                ) VALUES ('heldout-functional', 'role-a', 'role-target', 'game', 'target-game', ?, 20)
                """,
                (heldout_success,),
            )
        if include_prediction:
            conn.execute(
                """
                CREATE TABLE prediction_results (
                    id INTEGER PRIMARY KEY, global_step INTEGER, context_signature TEXT,
                    predicted_family TEXT, actual_family TEXT, context_contradiction INTEGER
                )
                """
            )
            conn.execute(
                """
                INSERT INTO prediction_results VALUES (1, 21, 'ctx-functional', 'family-ok', 'family-ok', 1)
                """
            )
        conn.commit()


def test_functional_coverage_promotes_without_legacy_promotion_and_keeps_structural_overlap(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir, legacy_promoted=0)

    diagnostic = _coverage_diagnostic(memory_dir, epoch_id="epoch_1")

    assert diagnostic["legacy_promoted"] is False
    assert diagnostic["incremental_validation_promoted"] is True
    assert diagnostic["promoted"] is True
    assert diagnostic["structural_overlap_ratio"] == 1.0
    assert diagnostic["eligible_explanation_event_count"] == 1
    assert diagnostic["explained_event_count"] == 1
    assert diagnostic["incremental_explanatory_coverage"] == 1.0
    assert diagnostic["explained_event_type_counts"] == {"transfer": 1}


def test_identical_provenance_without_functional_gain_has_zero_coverage(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir, historical_rates=((2, 0), (1, 1)))

    diagnostic = _coverage_diagnostic(memory_dir, epoch_id="epoch_1")

    assert diagnostic["incremental_explanatory_coverage"] == 0.0
    assert diagnostic["incremental_explanatory_coverage"] >= 0.0
    assert diagnostic["structural_overlap_ratio"] == 1.0
    assert "no_incremental_explanatory_gain" in diagnostic["rejection_reasons"]


def test_functional_coverage_ignores_provenance_support_counts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir)
    first = _coverage_diagnostic(memory_dir, epoch_id="epoch_1")
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("UPDATE role_candidates SET support_count = 999")
        conn.commit()

    second = _coverage_diagnostic(memory_dir, epoch_id="epoch_2")

    assert second["incremental_explanatory_coverage"] == first["incremental_explanatory_coverage"]
    assert second["coverage_longitudinal_change"]["incremental_coverage_delta"] == 0.0


def test_prediction_and_contradiction_events_are_held_out_and_deterministic(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir, include_prediction=True)
    diagnostic = _coverage_diagnostic(memory_dir, epoch_id="epoch_1")

    assert {"transfer", "prediction", "contradiction_resolution"} <= set(diagnostic["explained_event_type_counts"])
    assert diagnostic["explained_event_ids_sample"] == sorted(diagnostic["explained_event_ids_sample"])
    assert all(
        event_id.startswith(("transfer:", "prediction:", "contradiction_resolution:"))
        for event_id in diagnostic["explained_event_ids_sample"]
    )


def test_no_eligible_explanation_events_is_reported_separately(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir, heldout_success=None)
    diagnostic = _coverage_diagnostic(memory_dir, epoch_id="epoch_1")
    assert diagnostic["eligible_explanation_event_count"] == 0
    assert "no_eligible_explanation_events" in diagnostic["rejection_reasons"]


def test_failed_heldout_transfer_does_not_receive_prediction_credit(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir, heldout_success=0)

    diagnostic = _coverage_diagnostic(memory_dir, epoch_id="epoch_1")

    assert diagnostic["eligible_explanation_event_count"] == 1
    assert diagnostic["explained_event_count"] == 0
    assert diagnostic["mean_prediction_gain"] < 0.0
    assert "no_incremental_explanatory_gain" in diagnostic["rejection_reasons"]


def test_one_source_role_already_explains_event_gets_no_concept_credit(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir, historical_rates=((2, 0), (1, 1)))

    diagnostic = _coverage_diagnostic(memory_dir, epoch_id="epoch_1")

    assert diagnostic["eligible_explanation_event_count"] == 1
    assert diagnostic["explained_event_count"] == 0
    assert diagnostic["mean_compression_gain"] < 0.0


def test_event_diagnostic_artifact_has_functional_scores(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir)
    artifact = tmp_path / "reports" / "h07" / "h07_concept_explanation_events.jsonl"

    validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=IncrementalPromotionValidationConfig(enabled=True),
        validate_roles_and_concepts=True,
        validate_world_models=False,
        diagnostic_epoch_id="epoch_1",
        explanation_events_path=artifact,
    )

    rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event_id"].startswith("transfer:")
    assert {
        "best_single_role_score", "lower_level_baseline_score", "concept_enabled_score",
        "prediction_gain", "behavioral_gain", "compression_gain", "concept_incremental_gain",
        "explained", "rejection_reason",
    } <= set(rows[0])


def test_incremental_validation_disabled_preserves_legacy_promotion(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_set_coverage_candidate(memory_dir, extra_family_count=0, legacy_promoted=1)

    result = validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=IncrementalPromotionValidationConfig(enabled=False),
        validate_roles_and_concepts=True,
        validate_world_models=False,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        is_promoted = conn.execute(
            "SELECT is_promoted FROM concept_candidates WHERE concept_signature = 'concept-set'"
        ).fetchone()[0]
    assert result == {"incremental_promotion_validation_enabled": False}
    assert is_promoted == 1


def test_incremental_validation_failure_count_is_epoch_idempotent_and_resets_on_success(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_functional_candidate(memory_dir, heldout_success=None)
    config = IncrementalPromotionValidationConfig(enabled=True)
    for _ in range(2):
        validate_incremental_promotions_only(
            memory_dir=memory_dir,
            config=config,
            validate_roles_and_concepts=True,
            validate_world_models=False,
            diagnostic_epoch_id="epoch_1",
        )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        first_failures = conn.execute(
            "SELECT failure_count FROM promotion_validation_state WHERE candidate_type = 'concept' AND candidate_signature = 'concept-functional'"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO role_transfer_attempts (
                attempt_id, role_signature, observed_role_signature, target_scope_type, target_scope_key,
                reuse_success, last_seen_global_step
            ) VALUES ('heldout-after-failure', 'role-a', 'role-target', 'game', 'target-game', 1, 20)
            """
        )
        conn.commit()
    assert first_failures == 1
    validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=config,
        validate_roles_and_concepts=True,
        validate_world_models=False,
        diagnostic_epoch_id="epoch_2",
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        status = conn.execute(
            "SELECT failure_count, promotion_status, last_validation_epoch, last_validation_result FROM promotion_validation_state WHERE candidate_type = 'concept' AND candidate_signature = 'concept-functional'"
        ).fetchone()
    assert status == (0, "promoted", "epoch_2", "passed")


def test_incremental_promotion_validation_uses_later_evidence_and_demotes_without_deleting(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        for role, family, context, game in (("role-a", "family-a", "ctx-a", "game-a"), ("role-b", "family-b", "ctx-b", "game-b")):
            conn.execute(
                """
                INSERT INTO role_candidates (
                    role_signature, linked_carrier_count, linked_family_count, linked_context_count,
                    cross_game_count, support_count
                ) VALUES (?, 2, 1, 1, 1, 8)
                """,
                (role,),
            )
            for linked_type, linked_key in (("carrier", f"carrier-{role}"), ("family", family), ("context", context), ("game", game)):
                conn.execute(
                    "INSERT INTO role_links (role_signature, linked_type, linked_key, support_count) VALUES (?, ?, ?, 1)",
                    (role, linked_type, linked_key),
                )
            conn.execute(
                "INSERT INTO transformation_families (canonical_signature, prediction_lift, last_seen_global_step) VALUES (?, 0.4, 20)",
                (family,),
            )
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (attempt_id, role_signature, reuse_success, last_seen_global_step)
                VALUES (?, ?, ?, ?)
                """,
                (f"{role}-derivation", role, 0, 5),
            )
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (attempt_id, role_signature, reuse_success, last_seen_global_step)
                VALUES (?, ?, ?, ?)
                """,
                (f"{role}-positive-history", role, 1, 6),
            )
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (attempt_id, role_signature, reuse_success, last_seen_global_step)
                VALUES (?, ?, ?, ?)
                """,
                (f"{role}-heldout", role, 1, 20),
            )
        conn.execute(
            """
            INSERT INTO concept_candidates (
                concept_signature, compression_gain, explanatory_reach, promotion_score,
                cross_context_count, cross_game_count, first_seen_global_step, is_promoted
            ) VALUES ('concept-a', 2.0, 8.0, 0.9, 2, 2, 10, 1)
            """
        )
        for linked_type, linked_key in (
            ("role", "role-a"), ("role", "role-b"), ("family", "family-a"), ("family", "family-b"),
            ("family", "family-concept-only"),
            ("context", "ctx-a"), ("context", "ctx-b"), ("game", "game-a"), ("game", "game-b"),
        ):
            conn.execute(
                "INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count) VALUES ('concept-a', ?, ?, 1)",
                (linked_type, linked_key),
            )
        conn.execute(
            """
            INSERT INTO world_model_components (component_signature, linked_concept_count, first_seen_global_step, is_coherent)
            VALUES ('wm-a', 1, 10, 1)
            """
        )
        conn.execute(
            "INSERT INTO world_model_links (component_signature, linked_type, linked_key, support_count) VALUES ('wm-a', 'concept', 'concept-a', 1)"
        )
        conn.commit()

    config = IncrementalPromotionValidationConfig(enabled=True, demotion_failure_limit=1)
    first = validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=config,
        validate_roles_and_concepts=True,
        validate_world_models=False,
    )
    assert first["concepts_promoted_with_behavioral_lift"] == 1
    h07_after_first_validation = evaluate_h07_concept_emergence(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=tmp_path / "h07_after_first_validation",
        already_derived=True,
        incremental_promotion_validation=config,
    )
    validation_report = h07_after_first_validation["incremental_promotion_validation"]
    assert validation_report["enabled"] is True
    assert validation_report["thresholds"] == {
        "min_incremental_coverage": 0.05,
        "min_incremental_explanatory_coverage": 0.05,
        "min_event_prediction_gain": 0.01,
        "min_event_behavioral_gain": 0.01,
        "min_event_compression_gain": 0.0,
        "min_explanation_event_count": 1,
        "min_cross_context_or_game_evidence": 2,
        "min_behavioral_or_predictive_lift": 0.01,
        "demotion_failure_limit": 1,
        "promotion_score_threshold": 0.55,
    }
    assert validation_report["summary"]["concept_candidates_evaluated"] == 1
    candidate = validation_report["candidates"][0]
    assert candidate["concept_id"] == "concept-a"
    assert candidate["promoted"] is True
    assert candidate["validation_pass"] is True
    assert candidate["rejection_reasons"] == []
    assert candidate["validation_evidence_count"] == 2
    assert candidate["heldout_transfer_lift"] > 0.0
    assert "source_role_ids" not in candidate
    assert candidate["source_role_count"] == 2
    assert candidate["source_role_ids_sample"] == ["role-a", "role-b"]
    assert (tmp_path / "h07_after_first_validation" / "h07_concept_emergence_report.txt").read_text(
        encoding="utf-8"
    ).count("concept=concept-a") == 1
    with sqlite3.connect(paths.current_state) as conn:
        promoted = conn.execute(
            "SELECT is_promoted, validation_scope, validation_transfer_lift FROM concept_candidates WHERE concept_signature='concept-a'"
        ).fetchone()
    assert promoted[0:2] == (1, "later_global_step")
    assert promoted[2] > 0.0
    validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=config,
        validate_roles_and_concepts=False,
        validate_world_models=True,
    )

    with sqlite3.connect(paths.current_state) as conn:
        conn.execute("DELETE FROM role_transfer_attempts WHERE last_seen_global_step > 10")
        for role in ("role-a", "role-b"):
            conn.execute(
                "INSERT INTO role_transfer_attempts (attempt_id, role_signature, reuse_success, last_seen_global_step) VALUES (?, ?, 0, 20)",
                (f"{role}-failed-heldout", role),
            )
        conn.commit()
    second = validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=config,
        validate_roles_and_concepts=True,
        validate_world_models=False,
    )
    world = validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=config,
        validate_roles_and_concepts=False,
        validate_world_models=True,
    )
    assert second["concepts_rejected_no_heldout_lift"] == 1
    assert second["concepts_demoted"] == 1
    assert world["world_model_components_demoted"] == 1
    with sqlite3.connect(paths.current_state) as conn:
        concept = conn.execute(
            "SELECT is_promoted, promotion_status FROM concept_candidates WHERE concept_signature='concept-a'"
        ).fetchone()
        component = conn.execute(
            "SELECT is_coherent, candidate_only FROM world_model_components WHERE component_signature='wm-a'"
        ).fetchone()
    assert concept == (0, "demoted")
    assert component == (0, 1)
    h07_after_demotion = evaluate_h07_concept_emergence(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=tmp_path / "h07_after_demotion",
        already_derived=True,
        incremental_promotion_validation=config,
    )
    demoted_candidate = h07_after_demotion["incremental_promotion_validation"]["candidates"][0]
    assert demoted_candidate["demoted"] is True
    assert demoted_candidate["demotion_reason"] == "demoted_after_repeated_failure"
    assert "heldout_validation_failed" in demoted_candidate["rejection_reasons"]


def test_h07_incremental_validation_reports_explicit_rejection_reasons(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        for role in ("role-coverage", "role-scope", "role-heldout"):
            conn.execute(
                """
                INSERT INTO role_candidates (
                    role_signature, linked_carrier_count, linked_family_count, linked_context_count,
                    cross_game_count, support_count
                ) VALUES (?, 2, 1, 1, 1, 8)
                """,
                (role,),
            )
            for linked_type, linked_key in (
                ("carrier", f"carrier-{role}"),
                ("family", f"family-{role}"),
                ("context", f"context-{role}"),
                ("game", f"game-{role}"),
            ):
                conn.execute(
                    "INSERT INTO role_links (role_signature, linked_type, linked_key, support_count) VALUES (?, ?, ?, 1)",
                    (role, linked_type, linked_key),
                )
            conn.execute(
                "INSERT INTO transformation_families (canonical_signature, prediction_lift, last_seen_global_step) VALUES (?, 0.4, 5)",
                (f"family-{role}",),
            )
        for signature, role, explanatory_reach, contexts, games in (
            ("concept-coverage", "role-coverage", 0.0, 2, 2),
            ("concept-scope", "role-scope", 8.0, 1, 1),
            ("concept-heldout", "role-heldout", 8.0, 2, 2),
        ):
            conn.execute(
                """
                INSERT INTO concept_candidates (
                    concept_signature, compression_gain, explanatory_reach, promotion_score,
                    cross_context_count, cross_game_count, first_seen_global_step, is_promoted
                ) VALUES (?, 2.0, ?, 0.9, ?, ?, 10, 1)
                """,
                (signature, explanatory_reach, contexts, games),
            )
            concept_links = [
                ("role", role),
                ("carrier", f"carrier-{role}"),
                ("family", f"family-{role}"),
            ]
            if signature != "concept-coverage":
                concept_links.append(("family", f"family-{signature}-only"))
            for linked_type, linked_key in concept_links:
                conn.execute(
                    "INSERT INTO concept_links (concept_signature, linked_type, linked_key, support_count) VALUES (?, ?, ?, 1)",
                    (signature, linked_type, linked_key),
                )
        conn.commit()

    config = IncrementalPromotionValidationConfig(enabled=True)
    validate_incremental_promotions_only(
        memory_dir=memory_dir,
        config=config,
        validate_roles_and_concepts=True,
        validate_world_models=False,
    )
    result = evaluate_h07_concept_emergence(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=tmp_path / "h07",
        already_derived=True,
        incremental_promotion_validation=config,
    )
    validation = result["incremental_promotion_validation"]
    candidates = {item["concept_id"]: item for item in validation["candidates"]}
    assert "no_eligible_explanation_events" in candidates["concept-coverage"]["rejection_reasons"]
    assert "insufficient_cross_context_or_game_evidence" in candidates["concept-scope"]["rejection_reasons"]
    assert "no_heldout_samples" in candidates["concept-heldout"]["rejection_reasons"]
    assert validation["summary"]["concept_candidates_evaluated"] == len(candidates)
    assert validation["summary"]["concepts_promoted"] == 0
    assert validation["summary"]["concepts_rejected_no_incremental_coverage"] == 3
    assert validation["summary"]["concepts_rejected_insufficient_cross_scope"] == 1
    assert validation["summary"]["concepts_rejected_no_heldout_samples"] == 3


def test_h07_incremental_validation_disabled_keeps_candidates_empty(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    result = evaluate_h07_concept_emergence(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=tmp_path / "h07",
        already_derived=True,
    )
    assert result["incremental_promotion_validation"] == {
        "enabled": False,
        "summary": {},
        "candidates": [],
    }


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
    assert result["decision"] == "PARTIALLY_VALID"
    assert "Role timing is not fully grounded in real carrier evidence timing." in result["missing_evidence"]


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
        assert conn.execute("SELECT COUNT(*) FROM role_transfer_attempts WHERE attempt_id='weak1'").fetchone()[0] == 1
    result = evaluate_h07_concept_emergence(
        memory_dir=memory_dir,
        run_dir=None,
        output_dir=tmp_path / "h07",
        already_derived=True,
    )
    assert result["concept_strong_transfer_success_count"] == 0
    assert result["promoted_concept_count"] == 0
    assert result["decision"] != "VALID"


def test_h08_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, _transfer_rich_specs())
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08")
    assert result["world_model_component_count"] >= 1
    assert result["coherent_world_model_component_count"] >= 1
    assert result["decision"] == "PARTIALLY_VALID"


def test_h08_cannot_be_valid_without_promoted_concepts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "a1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "a2", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "enable", "action": "open", "polarity": "positive"},
    ])
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08")
    assert result["promoted_concept_count"] == 0
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"


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


def test_h06_profile_cache_bounds_prediction_work(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    specs = [
        {
            "carrier": "perf_000_unique",
            "game": "g_unique",
            "contexts": ["perf_ctx_unique_1", "perf_ctx_unique_2"],
            "effect": "reversible",
            "action": "toggle",
            "polarity": "neutral",
        }
    ]
    families = [
        ("enable", "open", "positive"),
        ("block", "close", "negative"),
        ("transform", "shift", "neutral"),
        ("terminate", "consume", "negative"),
    ]
    for index in range(80):
        effect, action, polarity = families[index % len(families)]
        specs.append(
            {
                "carrier": f"perf_{index}",
                "game": f"g{index % 8}",
                "contexts": [f"perf_ctx_{index}_1", f"perf_ctx_{index}_2"],
                "effect": effect,
                "action": action,
                "polarity": polarity,
            }
        )
    derive_higher_order_memory(memory_dir=_seed_memory(memory_dir, specs), max_transfer_attempts=25)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT attempt_id, reuse_success, failure_reason FROM role_transfer_attempts ORDER BY attempt_id ASC").fetchall()
    assert len(rows) <= 25
    assert len({str(row["attempt_id"]) for row in rows}) == len(rows)
    assert any(int(row["reuse_success"] or 0) == 1 for row in rows)
    assert any(str(row["failure_reason"]) != "success" for row in rows)


def test_h06_records_no_source_profile_attempt(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "solo", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
    ])
    derive_higher_order_memory(memory_dir=memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT reuse_success, failure_reason
            FROM role_transfer_attempts
            WHERE failure_reason = 'no_source_profile'
            ORDER BY attempt_id ASC
            """
        ).fetchall()
    assert rows
    assert all(int(row["reuse_success"] or 0) == 0 for row in rows)
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06")
    assert result["transfer_attempt_count"] >= 1
    assert result["no_source_profile_count"] >= 1
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["decision"] != "VALID"


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


def test_h09_detects_motifs(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "e1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "b1", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "block", "action": "close", "polarity": "negative"},
        {"carrier": "t1", "game": "g3", "contexts": ["ctx5", "ctx6"], "effect": "terminate", "action": "consume", "polarity": "negative"},
    ])
    result = evaluate_h09_future_option_motifs(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h09")
    assert result["future_option_event_count"] > 0
    assert result["future_option_motif_count"] > 0
    assert "enable" in result["motif_type_counts"]
    assert "block" in result["motif_type_counts"]
    assert result["decision"] in {"PARTIALLY_VALID", "VALID"}


def test_h09_emergent_motif_across_contexts_games(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "e1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "e2", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "e3", "game": "g3", "contexts": ["ctx5", "ctx6"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "b1", "game": "g1", "contexts": ["ctx7", "ctx8"], "effect": "block", "action": "close", "polarity": "negative"},
    ])
    result = evaluate_h09_future_option_motifs(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h09")
    assert result["emergent_future_option_motif_count"] >= 1
    assert result["decision"] == "VALID"


def test_h10_lift(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [])
    _seed_future_option_attention_rows(
        memory_dir,
        [
            {"event_id": "e1", "motif_signature": "m1", "option_delta_abs": 2.0, "replay_priority_score": 0.9, "memory_priority_score": 0.8, "contradiction_score": 0.7, "high_option_change": 1, "high_attention": 1},
            {"event_id": "e2", "motif_signature": "m1", "option_delta_abs": 2.0, "replay_priority_score": 0.8, "memory_priority_score": 0.7, "contradiction_score": 0.6, "high_option_change": 1, "high_attention": 1},
            {"event_id": "e3", "motif_signature": "m2", "option_delta_abs": 1.2, "replay_priority_score": 0.7, "memory_priority_score": 0.6, "contradiction_score": 0.5, "high_option_change": 1, "high_attention": 1},
                {"event_id": "e4", "motif_signature": "m2", "option_delta_abs": 0.1, "replay_priority_score": 0.1, "memory_priority_score": 0.1, "contradiction_score": 0.0, "high_option_change": 0, "high_attention": 0},
                {"event_id": "e5", "motif_signature": "m3", "option_delta_abs": 0.1, "replay_priority_score": 0.6, "memory_priority_score": 0.1, "contradiction_score": 0.0, "high_option_change": 0, "high_attention": 1},
                {"event_id": "e6", "motif_signature": "m3", "option_delta_abs": 0.0, "replay_priority_score": 0.1, "memory_priority_score": 0.1, "contradiction_score": 0.0, "high_option_change": 0, "high_attention": 0},
                {"event_id": "e7", "motif_signature": "m4", "option_delta_abs": 2.0, "replay_priority_score": 0.9, "memory_priority_score": 0.8, "contradiction_score": 0.8, "high_option_change": 1, "high_attention": 1},
                {"event_id": "e8", "motif_signature": "m4", "option_delta_abs": 1.7, "replay_priority_score": 0.85, "memory_priority_score": 0.7, "contradiction_score": 0.5, "high_option_change": 1, "high_attention": 1},
            ],
        )
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10", already_derived=True)
    assert (result["option_attention_lift"] or 0.0) >= 1.25
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["h10_blocked_by_h09"] is True


def test_h10_no_false_invalid_on_missing_low_group(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [])
    _seed_future_option_attention_rows(
        memory_dir,
        [
            {"event_id": "e1", "motif_signature": "m1", "option_delta_abs": 2.0, "replay_priority_score": 0.9, "memory_priority_score": 0.8, "contradiction_score": 0.7, "high_option_change": 1, "high_attention": 1},
            {"event_id": "e2", "motif_signature": "m1", "option_delta_abs": 2.0, "replay_priority_score": 0.8, "memory_priority_score": 0.7, "contradiction_score": 0.6, "high_option_change": 1, "high_attention": 1},
            {"event_id": "e3", "motif_signature": "m2", "option_delta_abs": 1.5, "replay_priority_score": 0.9, "memory_priority_score": 0.8, "contradiction_score": 0.5, "high_option_change": 1, "high_attention": 1},
        ],
    )
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10", already_derived=True)
    assert result["option_attention_lift"] is None
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["decision"] != "INVALID"


def test_h10_memory_priority_diagnostic_does_not_create_attention(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [])
    _seed_future_option_attention_rows(
        memory_dir,
        [
            {"event_id": f"e{i}", "motif_signature": "m1", "option_delta_abs": 2.0, "replay_priority_score": 0.0, "memory_priority_score": 1.0, "contradiction_score": 0.0, "high_option_change": 1}
            for i in range(5)
        ],
    )
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10", already_derived=True)
    assert result["high_option_change_count"] == 5
    assert result["high_attention_count"] == 0
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["decision"] not in {"VALID", "INVALID"}


def test_h10_attention_target_is_not_memory_priority(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [])
    _seed_future_option_attention_rows(
        memory_dir,
        [
            {"event_id": f"h{i}", "motif_signature": "mh", "option_delta_abs": 2.0, "replay_priority_score": 0.0, "memory_priority_score": 1.0, "contradiction_score": 0.0, "high_option_change": 1}
            for i in range(5)
        ] + [
            {"event_id": f"l{i}", "motif_signature": "ml", "option_delta_abs": 0.0, "replay_priority_score": 0.8 if i == 0 else 0.0, "memory_priority_score": 0.0, "contradiction_score": 0.0, "high_option_change": 0}
            for i in range(5)
        ],
    )
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10", already_derived=True)
    assert result["option_attention_lift"] is None or (result["option_attention_lift"] or 0.0) <= 1.0
    assert result["decision"] != "VALID"


def test_h10_validates_from_replay_contradiction_attention(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [])
    _seed_future_option_attention_rows(
        memory_dir,
        [
            {"event_id": f"h{i}", "motif_signature": "mh", "option_delta_abs": 2.0, "replay_priority_score": 0.8, "memory_priority_score": 1.0, "contradiction_score": 0.0, "high_option_change": 1}
            for i in range(5)
        ] + [
            {"event_id": f"l{i}", "motif_signature": "ml", "option_delta_abs": 0.0, "replay_priority_score": 0.8 if i == 0 else 0.0, "memory_priority_score": 0.0, "contradiction_score": 0.0, "high_option_change": 0}
            for i in range(5)
        ],
    )
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10", already_derived=True)
    assert result["high_option_change_attention_rate"] == 1.0
    assert result["low_option_change_attention_rate"] == 0.2
    assert (result["option_attention_lift"] or 0.0) >= 1.25
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["h10_blocked_by_h09"] is True


def test_h11_links_motifs_to_transfer_concepts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [])
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO future_option_motifs (motif_signature, motif_type, support_count, linked_event_count, linked_family_count, linked_carrier_count, linked_role_count, linked_concept_count, cross_context_count, cross_game_count, mean_option_delta, mean_abs_option_delta, mean_novelty_score, mean_reversibility_score, mean_branching_score, mean_termination_score, mean_replay_priority_score, first_seen_global_step, last_seen_global_step, motif_stability_score, is_emergent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("m_em", "enable", 6, 6, 2, 2, 2, 1, 3, 2, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.8, 10, 20, 0.8, 1))
        conn.execute("INSERT INTO concept_candidates (concept_signature, concept_type, support_count, linked_role_count, linked_carrier_count, linked_family_count, transfer_success_count, strong_transfer_success_count, cross_game_count, cross_context_count, compression_gain, explanatory_reach, promotion_score, first_seen_global_step, last_seen_global_step, is_promoted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("concept:em", "x", 5, 1, 2, 2, 5, 5, 2, 2, 2.0, 4.0, 0.9, 10, 20, 1))
        for idx in range(5):
            conn.execute("INSERT INTO future_option_transfer_links (motif_signature, role_signature, concept_signature, transfer_attempt_count, successful_transfer_count, strong_transfer_success_count, promoted_concept_count, mean_transfer_score, mean_best_margin, first_seen_global_step, last_seen_global_step) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("m_em", f"role:e{idx}", "concept:em", 5, 5, 5, 1, 0.9, 0.2, 10, 20))
        conn.commit()
    result = evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h11", already_derived=True)
    assert result["future_option_transfer_link_count"] > 0
    assert result["motifs_with_strong_transfer_count"] >= 1
    assert result["motifs_with_promoted_concept_count"] >= 1
    assert result["emergent_motif_transfer_link_count"] >= 5
    assert result["decision"] == "VALID"


def test_h11_no_valid_without_h09(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "s1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "unknown_effect", "action": "unknown_action", "polarity": "neutral"},
        {"carrier": "s2", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "unknown_effect", "action": "unknown_action", "polarity": "neutral"},
    ])
    result = evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h11")
    assert result["decision"] in {"INCONCLUSIVE", "PARTIALLY_VALID"}
    assert result["decision"] != "VALID"


def test_h11_does_not_validate_from_non_emergent_motif_links(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [])
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO future_option_motifs (motif_signature, motif_type, support_count, linked_event_count, linked_family_count, linked_carrier_count, linked_role_count, linked_concept_count, cross_context_count, cross_game_count, mean_option_delta, mean_abs_option_delta, mean_novelty_score, mean_reversibility_score, mean_branching_score, mean_termination_score, mean_replay_priority_score, first_seen_global_step, last_seen_global_step, motif_stability_score, is_emergent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("m_non", "enable", 5, 5, 1, 1, 1, 1, 2, 2, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.8, 10, 20, 0.8, 0))
        for idx in range(5):
            conn.execute("INSERT INTO future_option_transfer_links (motif_signature, role_signature, concept_signature, transfer_attempt_count, successful_transfer_count, strong_transfer_success_count, promoted_concept_count, mean_transfer_score, mean_best_margin, first_seen_global_step, last_seen_global_step) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("m_non", f"role:{idx}", f"concept:{idx}", 5, 5, 5, 1, 0.9, 0.2, 10, 20))
        conn.execute("INSERT INTO concept_candidates (concept_signature, concept_type, support_count, linked_role_count, linked_carrier_count, linked_family_count, transfer_success_count, strong_transfer_success_count, cross_game_count, cross_context_count, compression_gain, explanatory_reach, promotion_score, first_seen_global_step, last_seen_global_step, is_promoted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("concept:0", "x", 5, 1, 2, 2, 5, 5, 2, 2, 2.0, 4.0, 0.9, 10, 20, 1))
        conn.commit()
    result = evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h11", already_derived=True)
    assert result["future_option_transfer_link_count"] > 0
    assert result["non_emergent_motif_transfer_link_count"] > 0
    assert result["emergent_motif_transfer_link_count"] == 0
    assert result["decision"] != "VALID"
    assert any("emergent future-option motifs" in item for item in result["missing_evidence"])


def test_h11_validates_only_when_emergent_motif_has_transfer_concept_evidence(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [])
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO future_option_motifs (motif_signature, motif_type, support_count, linked_event_count, linked_family_count, linked_carrier_count, linked_role_count, linked_concept_count, cross_context_count, cross_game_count, mean_option_delta, mean_abs_option_delta, mean_novelty_score, mean_reversibility_score, mean_branching_score, mean_termination_score, mean_replay_priority_score, first_seen_global_step, last_seen_global_step, motif_stability_score, is_emergent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("m_non", "enable", 5, 5, 1, 1, 1, 1, 2, 2, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.8, 10, 20, 0.8, 0))
        conn.execute("INSERT INTO future_option_motifs (motif_signature, motif_type, support_count, linked_event_count, linked_family_count, linked_carrier_count, linked_role_count, linked_concept_count, cross_context_count, cross_game_count, mean_option_delta, mean_abs_option_delta, mean_novelty_score, mean_reversibility_score, mean_branching_score, mean_termination_score, mean_replay_priority_score, first_seen_global_step, last_seen_global_step, motif_stability_score, is_emergent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("m_em", "enable", 5, 5, 1, 1, 1, 1, 2, 2, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.8, 10, 20, 0.8, 1))
        conn.execute("INSERT INTO concept_candidates (concept_signature, concept_type, support_count, linked_role_count, linked_carrier_count, linked_family_count, transfer_success_count, strong_transfer_success_count, cross_game_count, cross_context_count, compression_gain, explanatory_reach, promotion_score, first_seen_global_step, last_seen_global_step, is_promoted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("concept:em", "x", 5, 1, 2, 2, 5, 5, 2, 2, 2.0, 4.0, 0.9, 10, 20, 1))
        conn.execute("INSERT INTO future_option_transfer_links (motif_signature, role_signature, concept_signature, transfer_attempt_count, successful_transfer_count, strong_transfer_success_count, promoted_concept_count, mean_transfer_score, mean_best_margin, first_seen_global_step, last_seen_global_step) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("m_non", "role:n", "concept:n", 5, 5, 5, 1, 0.9, 0.2, 10, 20))
        for idx in range(5):
            conn.execute("INSERT INTO future_option_transfer_links (motif_signature, role_signature, concept_signature, transfer_attempt_count, successful_transfer_count, strong_transfer_success_count, promoted_concept_count, mean_transfer_score, mean_best_margin, first_seen_global_step, last_seen_global_step) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("m_em", f"role:e{idx}", "concept:em", 5, 5, 5, 1, 0.9, 0.2, 10, 20))
        conn.commit()
    result = evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h11", already_derived=True)
    assert result["emergent_motif_transfer_link_count"] >= 5
    assert result["emergent_motifs_with_strong_transfer_count"] >= 1
    assert result["emergent_motifs_with_promoted_concept_count"] >= 1
    assert result["decision"] == "VALID"


def test_future_option_transfer_links_uses_sentinel_concept(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "s1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
        {"carrier": "s2", "game": "g2", "contexts": ["ctx3", "ctx4"], "effect": "enable", "action": "open", "polarity": "positive"},
    ])
    evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h11")
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        null_count = conn.execute("SELECT COUNT(*) FROM future_option_transfer_links WHERE concept_signature IS NULL").fetchone()[0]
        sentinel_count = conn.execute("SELECT COUNT(*) FROM future_option_transfer_links WHERE concept_signature='__none__'").fetchone()[0]
    assert null_count == 0
    assert sentinel_count >= 0


def test_suite_includes_h01_h11(tmp_path: Path) -> None:
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
        "H09 decision",
        "H10 decision",
        "H11 decision",
    ):
        assert key in summary
    assert "H09 core metrics" in summary
    assert "H10 core metrics" in summary
    assert "H11 core metrics" in summary


def test_fast_suite_does_not_call_derivations(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _seed_memory(memory_dir, _transfer_rich_specs())
    calls = {"future": 0}

    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_role_candidates_only",
        lambda **kwargs: {"role_candidate_count": 0},
    )
    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_role_transfer_attempts_only",
        lambda **kwargs: {"transfer_attempt_count": 0},
    )
    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_concept_candidates_only",
        lambda **kwargs: {"concept_candidate_count": 0},
    )
    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_world_model_components_only",
        lambda **kwargs: {"world_model_component_count": 0},
    )
    monkeypatch.setattr("v6.hypothesis_suite_report.derive_future_option_memory", lambda **kwargs: calls.__setitem__("future", calls["future"] + 1) or {})

    summary = run_hypothesis_suite_report(
        run_dir=run_dir,
        memory_dir=memory_dir,
        output_dir=tmp_path / "reports",
        scan_all_dbs=True,
        max_db_files=10,
        max_rows=1000,
        suite_mode="fast",
    )
    assert calls["future"] == 0
    assert summary["suite_mode"] == "fast"
    assert "derive_role_candidates_seconds" in summary
    assert "suite_total_seconds" in summary


def test_h05_written_after_role_candidates_only_without_later_derivations(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _seed_memory(memory_dir, _transfer_rich_specs())
    state = {"role_done": False, "h05_seen": False}

    def _derive_role(**kwargs):
        state["role_done"] = True
        return derive_role_candidates_only(**kwargs)

    def _h05(**kwargs):
        assert state["role_done"] is True
        assert state["h05_seen"] is False
        state["h05_seen"] = True
        return evaluate_h05_role_emergence(**kwargs)

    monkeypatch.setattr("v6.hypothesis_suite_report.derive_role_candidates_only", _derive_role)
    monkeypatch.setattr("v6.hypothesis_suite_report.evaluate_h05_role_emergence", _h05)
    monkeypatch.setattr("v6.hypothesis_suite_report.derive_role_transfer_attempts_only", lambda **kwargs: (_ for _ in ()).throw(AssertionError("transfer derivation should not run before H05")))
    monkeypatch.setattr("v6.hypothesis_suite_report.derive_concept_candidates_only", lambda **kwargs: (_ for _ in ()).throw(AssertionError("concept derivation should not run before H05")))
    monkeypatch.setattr("v6.hypothesis_suite_report.derive_world_model_components_only", lambda **kwargs: (_ for _ in ()).throw(AssertionError("world derivation should not run before H05")))

    try:
        run_hypothesis_suite_report(
            run_dir=run_dir,
            memory_dir=memory_dir,
            output_dir=tmp_path / "reports",
            scan_all_dbs=True,
            max_db_files=10,
            max_rows=1000,
            suite_mode="fast",
        )
    except AssertionError as exc:
        assert str(exc) == "transfer derivation should not run before H05"

    assert state["h05_seen"] is True
    assert (tmp_path / "reports" / "h05" / "h05_functional_role_emergence_report.json").exists()
    phase_rows = [
        json.loads(line)
        for line in (tmp_path / "reports" / "hypothesis_phase_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phase_names = [row["phase"] for row in phase_rows]
    derive_index = phase_names.index("derive_role_candidates")
    h05_index = phase_names.index("H05")
    assert derive_index < h05_index


def test_full_suite_calls_derivations_with_limits(tmp_path: Path, monkeypatch) -> None:
    memory_dir = tmp_path / "memory"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _seed_memory(memory_dir, _transfer_rich_specs())
    captured: dict[str, Any] = {"order": []}

    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_role_candidates_only",
        lambda **kwargs: captured["order"].append("role") or {},
    )
    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_role_transfer_attempts_only",
        lambda **kwargs: captured.update({"higher": int(kwargs["max_transfer_attempts"])}) or captured["order"].append("transfer") or {},
    )
    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_concept_candidates_only",
        lambda **kwargs: captured["order"].append("concept") or {},
    )
    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_world_model_components_only",
        lambda **kwargs: captured["order"].append("world") or {},
    )
    monkeypatch.setattr(
        "v6.hypothesis_suite_report.derive_future_option_memory",
        lambda **kwargs: captured.update(
            {"events": int(kwargs["max_events"]), "motifs": int(kwargs["max_motifs"])}
        ) or captured["order"].append("future") or {},
    )

    summary = run_hypothesis_suite_report(
        run_dir=run_dir,
        memory_dir=memory_dir,
        output_dir=tmp_path / "reports",
        scan_all_dbs=True,
        max_db_files=10,
        max_rows=1000,
        suite_mode="full",
        max_role_transfer_attempts=123,
        max_future_option_events=456,
        max_future_option_motifs=789,
    )
    assert captured["higher"] == 123
    assert captured["events"] == 456
    assert captured["motifs"] == 789
    assert captured["order"] == ["role", "transfer", "concept", "world", "future"]
    assert summary["suite_mode"] == "full"


def test_derive_role_transfer_attempts_only_respects_max_attempts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, _transfer_rich_specs())
    derive_role_candidates_only(memory_dir=memory_dir)
    summary = derive_role_transfer_attempts_only(memory_dir=memory_dir, max_transfer_attempts=3, workers=1, chunk_size=2)
    assert summary["transfer_attempt_count"] == 3
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM role_transfer_attempts").fetchone()[0] == 3


def test_transfer_attempt_workers_one_and_four_match_counts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(higher_order_substrate, "ProcessPoolExecutor", _ThreadPoolCompat)
    memory_dir_a = tmp_path / "memory_a"
    memory_dir_b = tmp_path / "memory_b"
    specs = _transfer_rich_specs()
    _seed_memory(memory_dir_a, specs)
    _seed_memory(memory_dir_b, specs)
    derive_role_candidates_only(memory_dir=memory_dir_a)
    derive_role_candidates_only(memory_dir=memory_dir_b)
    summary_a = derive_role_transfer_attempts_only(memory_dir=memory_dir_a, max_transfer_attempts=25, workers=1, chunk_size=5)
    summary_b = derive_role_transfer_attempts_only(memory_dir=memory_dir_b, max_transfer_attempts=25, workers=4, chunk_size=5)
    assert summary_a["transfer_attempt_count"] == summary_b["transfer_attempt_count"]
    assert summary_a["successful_transfer_count"] == summary_b["successful_transfer_count"]
    assert summary_a["successful_role_count"] == summary_b["successful_role_count"]


def test_transfer_worker_chunk_does_not_open_sqlite(monkeypatch) -> None:
    monkeypatch.setattr(higher_order_substrate.sqlite3, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sqlite connect should not be called in worker chunk")))
    rows = _derive_role_transfer_attempts_chunk(
        chunk=[("carrier1", "cross_game", "g2")],
        role_rows={
            "carrier1": {
                "carrier_signature": "carrier1",
                "role_signature": "roleA",
                "tokens": ("a", "b"),
                "first_seen_global_step": 1,
                "last_seen_global_step": 2,
            }
        },
        profile_cache={
            ("cross_game", "g2"): [
                {
                    "role_signature": "roleB",
                    "profile_tokens": ["a", "b"],
                    "profile_token_set": {"a", "b"},
                    "source_carrier_count": 2,
                    "source_context_count": 2,
                    "source_game_count": 1,
                }
            ]
        },
    )
    assert len(rows) == 1


def test_predict_transfer_attempt_does_not_sort_candidates(monkeypatch) -> None:
    import builtins

    monkeypatch.setattr(builtins, "sorted", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sorted should not be called")))
    attempt = _predict_transfer_attempt(
        profile_cache={
            ("cross_game", "g2"): [
                {"role_signature": "roleB", "profile_tokens": ["a", "b"], "profile_token_set": {"a", "b"}, "source_carrier_count": 2},
                {"role_signature": "roleC", "profile_tokens": ["a"], "profile_token_set": {"a"}, "source_carrier_count": 3},
            ]
        },
        role_rows={
            "carrier1": {
                "carrier_signature": "carrier1",
                "role_signature": "roleA",
                "tokens": ("a", "b"),
                "first_seen_global_step": 1,
                "last_seen_global_step": 2,
            }
        },
        target_carrier_signature="carrier1",
        transfer_kind="cross_game",
        target_scope_key="g2",
    )
    assert attempt["predicted_role_signature"] == "roleB"


def test_suite_total_interactions_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"games": ["g1"], "samplers": ["s1"], "seeds": [0], "runs": []}),
        encoding="utf-8",
    )
    summary = build_hypothesis_suite_summary(
        run_dir=run_dir,
        h01={"decision": "INCONCLUSIVE", "total_interaction_count": None},
        h02={"decision": "INCONCLUSIVE"},
        h03={"decision": "INCONCLUSIVE"},
        interactions_this_epoch=500,
        total_interactions_seen=None,
    )
    assert summary["total_interactions"] == 500
    assert summary["raw_report_total_interactions"] == 0
    assert summary["total_interactions_source"] == "continuous_epoch_argument"


def test_h02_incomplete_raw_evidence_does_not_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"games": ["g1"], "samplers": ["s1"], "seeds": [0], "runs": []}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, [
        {"carrier": "a1", "game": "g1", "contexts": ["ctx1", "ctx2"], "effect": "enable", "action": "open", "polarity": "positive"},
    ])
    result = evaluate_h02_prediction_violation_attention(run_dir=run_dir, output_dir=tmp_path / "h02", memory_dir=memory_dir)
    assert result["decision"] in {"INCONCLUSIVE", "PARTIALLY_VALID"}
    assert result["decision"] != "INVALID"
    assert result["raw_h02_evidence_incomplete"] is True


def test_h04_missing_timing_cannot_be_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, _transfer_rich_specs())
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("DELETE FROM temporal_milestones")
        conn.commit()
    result = evaluate_h04_carrier_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h04")
    assert result["decision"] == "PARTIALLY_VALID"
    assert "H04 timing is not fully grounded in real carrier evidence timing." in result["missing_evidence"]


def test_h05_missing_timing_cannot_be_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    _seed_memory(memory_dir, _transfer_rich_specs())
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("DELETE FROM temporal_milestones")
        conn.commit()
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05")
    assert result["decision"] == "PARTIALLY_VALID"
    assert "Role timing is not fully grounded in real carrier evidence timing." in result["missing_evidence"]


def test_dependency_gate_demotes_h07_h08() -> None:
    h05 = {"decision": "VALID", "missing_evidence": []}
    h06 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h07 = {"decision": "VALID", "missing_evidence": []}
    h08 = {"decision": "VALID", "missing_evidence": []}
    h09 = {"decision": "VALID", "missing_evidence": []}
    h10 = {"decision": "VALID", "missing_evidence": []}
    h11 = {"decision": "VALID", "missing_evidence": []}
    h04 = {"decision": "VALID"}
    _h05, _h06, h07_out, h08_out, _h09, _h10, _h11, notes = _apply_higher_order_dependency_gates(h04, h05, h06, h07, h08, h09, h10, h11)
    assert h07_out["decision"] == "PARTIALLY_VALID"
    assert h08_out["decision"] == "PARTIALLY_VALID"
    assert notes


def test_maturity_gate_demotes_early_h04_h08() -> None:
    base = {"decision": "VALID", "missing_evidence": []}
    h04, h05, h06, h07, h08, h09, h10, h11, notes = _apply_epoch_maturity_gates(
        h04=dict(base),
        h05=dict(base),
        h06=dict(base),
        h07=dict(base),
        h08=dict(base),
        h09=dict(base),
        h10=dict(base),
        h11=dict(base),
        total_interactions=500,
        interactions_this_epoch=500,
        game_count=251,
        sampler_count=7,
    )
    for item in (h04, h05, h06, h07, h08, h09, h10, h11):
        assert item["decision"] == "PARTIALLY_VALID"
        assert item["epoch_maturity_demoted"] is True
    assert notes


def test_no_maturity_demotion_after_enough_interactions() -> None:
    base = {"decision": "VALID", "missing_evidence": []}
    h04, h05, h06, h07, h08, h09, h10, h11, notes = _apply_epoch_maturity_gates(
        h04=dict(base),
        h05=dict(base),
        h06=dict(base),
        h07=dict(base),
        h08=dict(base),
        h09=dict(base),
        h10=dict(base),
        h11=dict(base),
        total_interactions=5000,
        interactions_this_epoch=5000,
        game_count=251,
        sampler_count=7,
    )
    for item in (h04, h05, h06, h07, h08, h09, h10, h11):
        assert item["decision"] == "VALID"
        assert "epoch_maturity_demoted" not in item
    assert notes == []


def test_dependency_gate_demotes_h10_if_h09_not_valid() -> None:
    h05 = {"decision": "VALID", "missing_evidence": []}
    h06 = {"decision": "VALID", "missing_evidence": []}
    h07 = {"decision": "VALID", "missing_evidence": []}
    h08 = {"decision": "VALID", "missing_evidence": []}
    h09 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h10 = {"decision": "VALID", "missing_evidence": []}
    h11 = {"decision": "VALID", "missing_evidence": []}
    h04 = {"decision": "VALID"}
    _h05, _h06, _h07, _h08, _h09, h10_out, h11_out, notes = _apply_higher_order_dependency_gates(h04, h05, h06, h07, h08, h09, h10, h11)
    assert h10_out["decision"] == "PARTIALLY_VALID"
    assert h11_out["decision"] == "PARTIALLY_VALID"
    assert notes


def test_dependency_gate_demotes_h11_if_h09_absent() -> None:
    h05 = {"decision": "VALID", "missing_evidence": []}
    h06 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h07 = {"decision": "VALID", "missing_evidence": []}
    h08 = {"decision": "VALID", "missing_evidence": []}
    h09 = {"decision": "INCONCLUSIVE", "missing_evidence": []}
    h10 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h11 = {"decision": "VALID", "missing_evidence": []}
    h04 = {"decision": "VALID"}
    _h05, _h06, _h07, _h08, _h09, _h10, h11_out, notes = _apply_higher_order_dependency_gates(h04, h05, h06, h07, h08, h09, h10, h11)
    assert h11_out["decision"] == "PARTIALLY_VALID"
    assert notes


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
        "H09": "VALID",
        "future_option_events": 9,
        "future_option_motifs": 3,
        "emergent_future_option_motifs": 1,
        "H10": "VALID",
        "option_attention_lift": 2.0,
        "high_option_change_attention_rate": 0.8,
        "h10_replay_attention_count": 5,
        "h10_contradiction_attention_count": 2,
        "H11": "VALID",
        "future_option_transfer_links": 4,
        "motifs_with_strong_transfer": 2,
        "motifs_with_promoted_concepts": 1,
        "h11_emergent_motif_transfer_links": 3,
        "h11_emergent_motifs_with_strong_transfer": 1,
        "h11_emergent_motifs_with_promoted_concepts": 1,
        "h11_non_emergent_motif_transfer_links": 1,
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
    assert "H09" in text
    assert "H10" in text
    assert "H11" in text
    assert "future-option events" in text
    assert "option-attention lift" in text
    assert "future-option transfer links" in text
    assert "replay attention count" in text
    assert "contradiction attention count" in text
    assert "emergent motif transfer links" in text
    assert "non-emergent motif transfer links" in text


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
            _insert_node(graph_conn, f"contingency:cont_{carrier}", "contingency", f"cont_{carrier}")
            _insert_node(graph_conn, f"contradiction:{carrier}", "contradiction", carrier)
            _insert_edge(graph_conn, f"carrier:{carrier}", f"family:{family_signature}", "explains")
            _insert_edge(graph_conn, f"carrier:{carrier}", f"contingency:cont_{carrier}", "anchors")
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


def _seed_future_option_attention_rows(memory_dir: Path, rows: list[dict[str, object]]) -> None:
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        for row in rows:
            replay_priority = float(row["replay_priority_score"])
            contradiction_score = float(row["contradiction_score"])
            high_attention = 1 if (replay_priority >= 0.50 or contradiction_score >= 0.50) else 0
            if replay_priority >= 0.50 and contradiction_score >= 0.50:
                attention_signal_source = "replay_priority+contradiction"
            elif replay_priority >= 0.50:
                attention_signal_source = "replay_priority"
            elif contradiction_score >= 0.50:
                attention_signal_source = "contradiction"
            else:
                attention_signal_source = "none"
            conn.execute(
                """
                INSERT INTO future_option_attention_links (
                    event_id, motif_signature, owner_type, owner_key, option_delta_abs, replay_priority_score,
                    memory_priority_score, contradiction_score, high_option_change, high_attention,
                    attention_signal_source, first_seen_global_step, last_seen_global_step
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["event_id"]),
                    str(row["motif_signature"]),
                    "synthetic",
                    str(row["event_id"]),
                    float(row["option_delta_abs"]),
                    replay_priority,
                    float(row["memory_priority_score"]),
                    contradiction_score,
                    int(row["high_option_change"]),
                    high_attention,
                    attention_signal_source,
                    10,
                    20,
                ),
            )
        conn.commit()


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
