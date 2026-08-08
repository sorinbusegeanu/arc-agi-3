from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.higher_order_substrate import (
    IncrementalPromotionValidationConfig,
    derive_higher_order_memory,
)


_REJECTED_PROMOTION_STATUSES = {"failed", "invalid", "demoted", "rejected"}


def _row_value(row: sqlite3.Row | dict[str, Any], key: str) -> Any:
    if isinstance(row, sqlite3.Row):
        return row[key] if key in row.keys() else None
    return row.get(key)


def _effective_concept_is_promoted(row: sqlite3.Row | dict[str, Any]) -> bool:
    persistent_promoted = _row_value(row, "persistent_currently_promoted")
    if persistent_promoted is not None:
        promoted = int(persistent_promoted) == 1
    else:
        candidate_promoted = _row_value(row, "candidate_is_promoted")
        if candidate_promoted is None:
            candidate_promoted = _row_value(row, "is_promoted")
        promoted = int(candidate_promoted or 0) == 1

    promotion_status = str(
        _row_value(row, "persistent_promotion_status") or ""
    ).strip().lower()
    validation_status = str(
        _row_value(row, "persistent_validation_status") or ""
    ).strip().lower()

    if promotion_status in _REJECTED_PROMOTION_STATUSES:
        return False
    if validation_status in _REJECTED_PROMOTION_STATUSES:
        return False
    return promoted


def evaluate_h07_concept_emergence(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
    incremental_promotion_validation: IncrementalPromotionValidationConfig | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"

    if not already_derived and current_state.exists():
        derive_higher_order_memory(memory_dir=memory_dir, run_dir=run_dir)

    if not current_state.exists():
        result = _base_result(
            "INSUFFICIENT_EVIDENCE",
            [f"Missing expected compact-memory file: {current_state}"],
        )
        result["incremental_promotion_validation"] = (
            _empty_incremental_promotion_validation_report(
                incremental_promotion_validation
            )
        )
        _write_outputs(output_dir, result)
        return result

    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row

        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        required_tables = {
            "role_transfer_attempts",
            "role_candidates",
            "role_links",
            "concept_candidates",
            "higher_order_milestones",
        }
        missing_tables = sorted(required_tables - tables)
        if missing_tables:
            result = _base_result(
                "INSUFFICIENT_EVIDENCE",
                [
                    "Missing expected compact-memory table(s): "
                    + ", ".join(missing_tables)
                ],
            )
            result["incremental_promotion_validation"] = (
                _empty_incremental_promotion_validation_report(
                    incremental_promotion_validation
                )
            )
            _write_outputs(output_dir, result)
            return result

        transfer_columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(role_transfer_attempts)"
            ).fetchall()
        }

        transfer_attempt_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM role_transfer_attempts"
            ).fetchone()[0]
        )
        successful_transfers = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM role_transfer_attempts
                WHERE COALESCE(reuse_success, 0) = 1
                """
            ).fetchone()[0]
        )

        if "attempt_id" in transfer_columns:
            deduplicated_transfer_attempt_count = int(
                conn.execute(
                    """
                    SELECT COUNT(DISTINCT attempt_id)
                    FROM role_transfer_attempts
                    WHERE attempt_id IS NOT NULL
                    """
                ).fetchone()[0]
            )
            deduplicated_transfer_success_count = int(
                conn.execute(
                    """
                    SELECT COUNT(
                        DISTINCT CASE
                            WHEN COALESCE(reuse_success, 0) = 1
                            THEN attempt_id
                        END
                    )
                    FROM role_transfer_attempts
                    WHERE attempt_id IS NOT NULL
                    """
                ).fetchone()[0]
            )

            null_attempt_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM role_transfer_attempts
                    WHERE attempt_id IS NULL
                    """
                ).fetchone()[0]
            )
            null_attempt_success_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM role_transfer_attempts
                    WHERE attempt_id IS NULL
                      AND COALESCE(reuse_success, 0) = 1
                    """
                ).fetchone()[0]
            )
            deduplicated_transfer_attempt_count += null_attempt_count
            deduplicated_transfer_success_count += null_attempt_success_count
        else:
            deduplicated_transfer_attempt_count = transfer_attempt_count
            deduplicated_transfer_success_count = successful_transfers

        roles_seen_for_concept_derivation = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM role_candidates
                WHERE COALESCE(is_emergent, 0) = 1
                   OR COALESCE(role_stability_score, 0.0) >= 0.50
                """
            ).fetchone()[0]
        )

        role_link_rows = conn.execute(
            """
            SELECT role_signature, linked_type
            FROM role_links
            ORDER BY role_signature ASC, linked_type ASC
            """
        ).fetchall()
        linked_types_by_role: dict[str, set[str]] = {}
        for row in role_link_rows:
            linked_types_by_role.setdefault(
                str(row["role_signature"]), set()
            ).add(str(row["linked_type"]))

        successful_roles = {
            str(row["canonical_source_role"])
            for row in conn.execute(
                """
                SELECT DISTINCT
                    COALESCE(source_role_signature, role_signature)
                        AS canonical_source_role
                FROM role_transfer_attempts
                WHERE COALESCE(reuse_success, 0) = 1
                  AND COALESCE(source_role_signature, role_signature)
                      IS NOT NULL
                """
            ).fetchall()
        }

        roles_with_transfer_attempts = int(
            conn.execute(
                """
                SELECT COUNT(
                    DISTINCT COALESCE(
                        source_role_signature,
                        role_signature
                    )
                )
                FROM role_transfer_attempts
                """
            ).fetchone()[0]
        )
        roles_with_successful_transfers = int(
            conn.execute(
                """
                SELECT COUNT(
                    DISTINCT COALESCE(
                        source_role_signature,
                        role_signature
                    )
                )
                FROM role_transfer_attempts
                WHERE COALESCE(reuse_success, 0) = 1
                """
            ).fetchone()[0]
        )

        if "concept_promotion_state" in tables:
            concept_rows = conn.execute(
                """
                SELECT
                    candidate.concept_signature,
                    candidate.compression_gain,
                    candidate.promotion_score,
                    candidate.transfer_success_count,
                    candidate.strong_transfer_success_count,
                    candidate.linked_role_count,
                    candidate.linked_carrier_count,
                    candidate.linked_family_count,
                    candidate.cross_context_count,
                    candidate.cross_game_count,
                    candidate.is_promoted AS candidate_is_promoted,
                    candidate.transfer_success_concentration,
                    candidate.is_overconcentrated,
                    persistent.currently_promoted
                        AS persistent_currently_promoted,
                    persistent.promotion_status
                        AS persistent_promotion_status,
                    persistent.validation_status
                        AS persistent_validation_status
                FROM concept_candidates AS candidate
                LEFT JOIN concept_promotion_state AS persistent
                    ON persistent.concept_signature =
                       candidate.concept_signature
                ORDER BY candidate.concept_signature ASC
                """
            ).fetchall()
        else:
            concept_rows = conn.execute(
                """
                SELECT
                    concept_signature,
                    compression_gain,
                    promotion_score,
                    transfer_success_count,
                    strong_transfer_success_count,
                    linked_role_count,
                    linked_carrier_count,
                    linked_family_count,
                    cross_context_count,
                    cross_game_count,
                    is_promoted AS candidate_is_promoted,
                    transfer_success_concentration,
                    is_overconcentrated,
                    NULL AS persistent_currently_promoted,
                    NULL AS persistent_promotion_status,
                    NULL AS persistent_validation_status
                FROM concept_candidates
                ORDER BY concept_signature ASC
                """
            ).fetchall()

        incremental_validation = (
            _load_incremental_promotion_validation_report(
                conn,
                config=incremental_promotion_validation,
                expected_candidate_count=len(concept_rows),
            )
        )

        milestone_map = dict(
            conn.execute(
                """
                SELECT milestone_name, first_global_step
                FROM higher_order_milestones
                """
            ).fetchall()
        )

        if "higher_order_milestone_history" in tables:
            historical_milestone_map = dict(
                conn.execute(
                    """
                    SELECT milestone_name, first_global_step
                    FROM higher_order_milestone_history
                    """
                ).fetchall()
            )
        else:
            historical_milestone_map = {}

    roles_skipped_missing_carrier_links = 0
    roles_skipped_missing_family_links = 0
    roles_skipped_missing_transfer_success = 0
    roles_used_for_concepts = 0

    for role_signature, linked_types in linked_types_by_role.items():
        if "carrier" not in linked_types:
            roles_skipped_missing_carrier_links += 1
            continue
        if "family" not in linked_types:
            roles_skipped_missing_family_links += 1
            continue
        if role_signature not in successful_roles:
            roles_skipped_missing_transfer_success += 1
            continue
        roles_used_for_concepts += 1

    concept_candidate_count = len(concept_rows)
    promoted_rows = [
        row for row in concept_rows if _effective_concept_is_promoted(row)
    ]
    promoted_concept_count = len(promoted_rows)

    mean_compression_gain = (
        sum(float(row["compression_gain"] or 0.0) for row in concept_rows)
        / max(1, concept_candidate_count)
        if concept_rows
        else None
    )
    max_compression_gain = max(
        (float(row["compression_gain"] or 0.0) for row in concept_rows),
        default=None,
    )
    mean_promotion_score = (
        sum(float(row["promotion_score"] or 0.0) for row in concept_rows)
        / max(1, concept_candidate_count)
        if concept_rows
        else None
    )
    max_promotion_score = max(
        (float(row["promotion_score"] or 0.0) for row in concept_rows),
        default=None,
    )

    candidate_cross_context_count = sum(
        1
        for row in concept_rows
        if int(row["cross_context_count"] or 0) >= 1
    )
    candidate_cross_game_count = sum(
        1
        for row in concept_rows
        if int(row["cross_game_count"] or 0) >= 1
    )
    promoted_cross_context_count = sum(
        1
        for row in promoted_rows
        if int(row["cross_context_count"] or 0) >= 1
    )
    promoted_cross_game_count = sum(
        1
        for row in promoted_rows
        if int(row["cross_game_count"] or 0) >= 1
    )

    concept_transfer_success_count = (
        deduplicated_transfer_success_count
    )
    concept_strong_transfer_success_count = sum(
        int(row["strong_transfer_success_count"] or 0)
        for row in concept_rows
    )

    source_role_count_mean = (
        sum(float(row["linked_role_count"] or 0.0) for row in promoted_rows)
        / max(1, promoted_concept_count)
        if promoted_rows
        else None
    )
    source_carrier_count_mean = (
        sum(
            float(row["linked_carrier_count"] or 0.0)
            for row in promoted_rows
        )
        / max(1, promoted_concept_count)
        if promoted_rows
        else None
    )

    candidate_cross_game_count_max = max(
        (int(row["cross_game_count"] or 0) for row in concept_rows),
        default=0,
    )
    candidate_cross_context_count_max = max(
        (int(row["cross_context_count"] or 0) for row in concept_rows),
        default=0,
    )
    promoted_cross_game_count_max = max(
        (int(row["cross_game_count"] or 0) for row in promoted_rows),
        default=0,
    )
    promoted_cross_context_count_max = max(
        (int(row["cross_context_count"] or 0) for row in promoted_rows),
        default=0,
    )

    strong_transfer_counts = [
        int(row["strong_transfer_success_count"] or 0)
        for row in promoted_rows
    ]
    concept_transfer_success_concentration = (
        max(strong_transfer_counts) / sum(strong_transfer_counts)
        if strong_transfer_counts and sum(strong_transfer_counts) > 0
        else None
    )

    overconcentrated_concept_count = sum(
        1
        for row in concept_rows
        if int(row["is_overconcentrated"] or 0) == 1
    )
    promoted_overconcentrated_concept_count = sum(
        1
        for row in promoted_rows
        if int(row["is_overconcentrated"] or 0) == 1
    )

    transfer_success_rate = (
        float(deduplicated_transfer_success_count)
        / float(deduplicated_transfer_attempt_count)
        if deduplicated_transfer_attempt_count > 0
        else None
    )

    max_source_role_count = max(
        (int(row["linked_role_count"] or 0) for row in promoted_rows),
        default=0,
    )
    max_source_family_count = max(
        (int(row["linked_family_count"] or 0) for row in promoted_rows),
        default=0,
    )

    metrics = {
        "concept_candidate_count": concept_candidate_count,
        "promoted_concept_count": promoted_concept_count,
        "mean_compression_gain": mean_compression_gain,
        "max_compression_gain": max_compression_gain,
        "mean_promotion_score": mean_promotion_score,
        "max_promotion_score": max_promotion_score,
        "transfer_attempt_count": transfer_attempt_count,
        "deduplicated_transfer_attempt_count":
            deduplicated_transfer_attempt_count,
        "concept_transfer_success_count":
            concept_transfer_success_count,
        "concept_strong_transfer_success_count":
            concept_strong_transfer_success_count,
        "transfer_success_rate": transfer_success_rate,
        "candidate_cross_context_count":
            candidate_cross_context_count,
        "candidate_cross_game_count": candidate_cross_game_count,
        "promoted_cross_context_count":
            promoted_cross_context_count,
        "promoted_cross_game_count": promoted_cross_game_count,
        "candidate_cross_game_count_max":
            candidate_cross_game_count_max,
        "candidate_cross_context_count_max":
            candidate_cross_context_count_max,
        "promoted_cross_game_count_max":
            promoted_cross_game_count_max,
        "promoted_cross_context_count_max":
            promoted_cross_context_count_max,
        "cross_context_concept_count":
            candidate_cross_context_count,
        "cross_game_concept_count": candidate_cross_game_count,
        "concept_cross_game_count_max":
            candidate_cross_game_count_max,
        "concept_cross_context_count_max":
            candidate_cross_context_count_max,
        "source_role_count_mean": source_role_count_mean,
        "source_carrier_count_mean": source_carrier_count_mean,
        "max_source_role_count": max_source_role_count,
        "max_source_family_count": max_source_family_count,
        "concept_transfer_success_concentration":
            concept_transfer_success_concentration,
        "overconcentrated_concept_count":
            overconcentrated_concept_count,
        "promoted_overconcentrated_concept_count":
            promoted_overconcentrated_concept_count,
        "first_concept_candidate_step":
            milestone_map.get("first_concept_candidate_step"),
        "first_promoted_concept_step":
            milestone_map.get("first_promoted_concept_step"),
        "historical_first_promoted_concept_step":
            historical_milestone_map.get(
                "first_promoted_concept_step"
            ),
        "first_role_transfer_success_step":
            milestone_map.get("first_role_transfer_success_step"),
        "roles_seen_for_concept_derivation":
            roles_seen_for_concept_derivation,
        "roles_with_transfer_attempts":
            roles_with_transfer_attempts,
        "roles_with_successful_transfers":
            roles_with_successful_transfers,
        "roles_eligible_for_concept_derivation":
            roles_used_for_concepts,
        "roles_skipped_missing_carrier_links":
            roles_skipped_missing_carrier_links,
        "roles_skipped_missing_family_links":
            roles_skipped_missing_family_links,
        "roles_skipped_missing_transfer_success":
            roles_skipped_missing_transfer_success,
        "roles_used_for_concepts": roles_used_for_concepts,
        "evidence_stage": None,
    }
    metrics.update(incremental_validation["summary"])

    if (
        deduplicated_transfer_success_count == 0
        and concept_candidate_count == 0
    ):
        decision = "INSUFFICIENT_EVIDENCE"
        missing = [
            "no successful role transfers and no concept candidates available"
        ]
    elif deduplicated_transfer_attempt_count <= 0:
        decision = "INCONCLUSIVE"
        missing = ["no role transfer attempts available"]
    elif (
        deduplicated_transfer_success_count > 0
        and concept_candidate_count == 0
    ):
        if (
            roles_skipped_missing_family_links > 0
            or roles_skipped_missing_carrier_links > 0
        ):
            decision = "INSUFFICIENT_EVIDENCE"
            missing = []
            if roles_skipped_missing_family_links > 0:
                missing.append(
                    "concept derivation blocked by missing role-family links"
                )
            if roles_skipped_missing_carrier_links > 0:
                missing.append(
                    "concept derivation blocked by missing role-carrier links"
                )
        else:
            decision = "INVALID"
            missing = []
    elif (
        promoted_concept_count >= 1
        and concept_strong_transfer_success_count >= 2
        and (max_compression_gain or 0.0) >= 1.50
        and (max_promotion_score or 0.0) >= 0.55
        and (
            promoted_cross_context_count_max >= 3
            or promoted_cross_game_count_max >= 2
        )
        and max_source_role_count >= 1
        and max_source_family_count >= 2
        and roles_used_for_concepts >= 3
        and (transfer_success_rate or 0.0) > 0.0
        and (
            (concept_transfer_success_concentration or 0.0)
            <= 0.80
        )
        and promoted_overconcentrated_concept_count == 0
    ):
        decision = "VALID"
        missing = []
    elif concept_candidate_count > 0 and promoted_concept_count == 0:
        decision = "INSUFFICIENT_EVIDENCE"
        metrics["evidence_stage"] = "concept_precursor_only"
        missing = ["No promoted concept available."]
        if overconcentrated_concept_count == concept_candidate_count:
            missing.append("All concept candidates are overconcentrated.")
    elif promoted_concept_count > 0:
        if (
            promoted_overconcentrated_concept_count
            == promoted_concept_count
        ):
            decision = "INSUFFICIENT_EVIDENCE"
            metrics["evidence_stage"] = "precursor_only"
            missing = ["Promoted concepts are overconcentrated."]
        else:
            decision = "PARTIALLY_VALID"
            missing = []
    else:
        decision = "PARTIALLY_VALID"
        missing = []

    if (
        bool(incremental_validation.get("enabled", False))
        and not bool(
            incremental_validation.get("diagnostics_complete", True)
        )
    ):
        decision = "INSUFFICIENT_EVIDENCE"
        missing = list(
            dict.fromkeys(
                [
                    *missing,
                    *[
                        str(item)
                        for item in incremental_validation.get(
                            "consistency_warnings", []
                        )
                    ],
                ]
            )
        )

    result = _base_result(decision, missing)
    result.update(metrics)
    result["core_metrics"] = dict(metrics)
    result["incremental_promotion_validation"] = incremental_validation

    cross_scope_warnings: list[str] = []
    if (
        metrics["cross_context_concept_count"]
        != metrics["candidate_cross_context_count"]
    ):
        cross_scope_warnings.append(
            "cross-context candidate metric aliases disagree"
        )
    if (
        metrics["cross_game_concept_count"]
        != metrics["candidate_cross_game_count"]
    ):
        cross_scope_warnings.append(
            "cross-game candidate metric aliases disagree"
        )
    if (
        metrics["concept_cross_context_count_max"]
        != metrics["candidate_cross_context_count_max"]
    ):
        cross_scope_warnings.append(
            "cross-context maximum metric aliases disagree"
        )
    if (
        metrics["concept_cross_game_count_max"]
        != metrics["candidate_cross_game_count_max"]
    ):
        cross_scope_warnings.append(
            "cross-game maximum metric aliases disagree"
        )
    result["consistency_warnings"] = cross_scope_warnings

    if (
        isinstance(incremental_validation, dict)
        and "incremental_coverage_aggregate"
        in incremental_validation
    ):
        result["incremental_coverage_aggregate"] = (
            incremental_validation["incremental_coverage_aggregate"]
        )

    _write_outputs(output_dir, result)
    return result


def _base_result(
    decision: str,
    missing_evidence: list[str],
) -> dict[str, Any]:
    return {
        "hypothesis_id": "H07",
        "decision": decision,
        "missing_evidence": list(missing_evidence),
        "evidence_source": "compact_memory",
    }


def _empty_incremental_promotion_validation_report(
    config: IncrementalPromotionValidationConfig | None,
) -> dict[str, Any]:
    if config is None or not config.enabled:
        return {"enabled": False, "summary": {}, "candidates": []}
    return {
        "enabled": True,
        "thresholds": _incremental_promotion_thresholds(config),
        "summary": {
            "concept_candidates_evaluated": 0,
            "concepts_promoted": 0,
            "concepts_rejected_no_incremental_coverage": 0,
            "concepts_rejected_insufficient_relevant_samples": 0,
            "concepts_rejected_insufficient_cross_scope": 0,
            "concepts_rejected_no_predictive_or_behavioral_lift": 0,
            "concepts_rejected_no_heldout_samples": 0,
            "concepts_rejected_heldout_validation_failed": 0,
            "concepts_rejected_below_threshold": 0,
            "concepts_demoted": 0,
        },
        "candidates": [],
    }


def _incremental_promotion_thresholds(
    config: IncrementalPromotionValidationConfig,
) -> dict[str, int | float]:
    return {
        "min_incremental_coverage":
            float(config.min_incremental_coverage),
        "min_incremental_explanatory_coverage":
            float(config.min_incremental_explanatory_coverage),
        "min_event_prediction_gain":
            float(config.min_event_prediction_gain),
        "min_event_behavioral_gain":
            float(config.min_event_behavioral_gain),
        "min_event_compression_gain":
            float(config.min_event_compression_gain),
        "min_explanation_event_count":
            int(config.min_explanation_event_count),
        "min_relevant_heldout_event_count":
            int(config.min_relevant_heldout_event_count),
        "promotion_population_comparability_threshold":
            float(
                config.promotion_population_comparability_threshold
            ),
        "min_cross_context_or_game_evidence":
            int(config.min_cross_context_or_game_evidence),
        "min_behavioral_or_predictive_lift":
            float(config.min_behavioral_or_predictive_lift),
        "demotion_failure_limit":
            int(config.demotion_failure_limit),
        "promotion_score_threshold":
            float(config.promotion_score_threshold),
    }


def _load_incremental_promotion_validation_report(
    conn: sqlite3.Connection,
    *,
    config: IncrementalPromotionValidationConfig | None,
    expected_candidate_count: int,
) -> dict[str, Any]:
    report = _empty_incremental_promotion_validation_report(config)
    if not report["enabled"]:
        return report

    table_exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'concept_promotion_validation_diagnostics'
        """
    ).fetchone()
    if table_exists is None:
        report.update(
            {
                "diagnostics_complete": False,
                "consistency_warnings": [
                    "incremental promotion validation diagnostics "
                    "table is missing"
                ],
            }
        )
        return report

    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    rows = conn.execute(
        """
        SELECT diagnostic.payload_json
        FROM concept_promotion_validation_diagnostics AS diagnostic
        WHERE diagnostic.rowid IN (
            SELECT MAX(rowid)
            FROM concept_promotion_validation_diagnostics
            GROUP BY concept_signature
        )
        ORDER BY diagnostic.concept_signature ASC
        """
    ).fetchall()

    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            warnings.append(
                "invalid incremental promotion validation "
                "diagnostic payload"
            )
            continue
        if not isinstance(payload, dict):
            warnings.append(
                "invalid incremental promotion validation "
                "diagnostic payload"
            )
            continue
        candidates.append(payload)

    summary = {
        "concept_candidates_evaluated": len(candidates),
        "concepts_promoted": sum(
            1 for item in candidates if bool(item.get("promoted"))
        ),
        "concepts_rejected_no_incremental_coverage": sum(
            1
            for item in candidates
            if any(
                reason
                in {
                    "relevant_coverage_below_threshold",
                    "insufficient_relevant_samples",
                }
                for reason in item.get("rejection_reasons", [])
            )
        ),
        "concepts_rejected_insufficient_relevant_samples": sum(
            1
            for item in candidates
            if "insufficient_relevant_samples"
            in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_insufficient_cross_scope": sum(
            1
            for item in candidates
            if "insufficient_cross_context_or_game_evidence"
            in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_no_predictive_or_behavioral_lift": sum(
            1
            for item in candidates
            if "no_predictive_or_behavioral_lift"
            in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_no_heldout_samples": sum(
            1
            for item in candidates
            if "no_heldout_samples"
            in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_heldout_validation_failed": sum(
            1
            for item in candidates
            if "heldout_validation_failed"
            in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_below_threshold": sum(
            1
            for item in candidates
            if "below_promotion_score_threshold"
            in item.get("rejection_reasons", [])
        ),
        "concepts_demoted": sum(
            1 for item in candidates if bool(item.get("demoted"))
        ),
    }

    if summary["concept_candidates_evaluated"] != expected_candidate_count:
        warnings.append(
            "candidate count does not match incremental "
            "validation diagnostics"
        )

    for candidate in candidates:
        promoted = bool(candidate.get("promoted"))
        reasons = list(candidate.get("rejection_reasons", []))
        if (promoted and reasons) or (not promoted and not reasons):
            warnings.append(
                "inconsistent rejection reasons for "
                f"{candidate.get('concept_id', 'unknown')}"
            )
        for error in candidate.get("diagnostics_errors", []):
            warnings.append(
                "coverage diagnostics error for "
                f"{candidate.get('concept_id', 'unknown')}: {error}"
            )

    report["summary"] = summary
    report["candidates"] = candidates
    report["incremental_coverage_aggregate"] = (
        _incremental_coverage_aggregate(candidates)
    )
    report["diagnostics_complete"] = not warnings
    report["consistency_warnings"] = warnings
    return report


def _incremental_coverage_aggregate(
    candidates: list[dict[str, Any]],
) -> dict[str, int | float]:
    changes = [
        item.get("coverage_longitudinal_change", {})
        for item in candidates
        if isinstance(
            item.get("coverage_longitudinal_change"), dict
        )
        and item["coverage_longitudinal_change"].get(
            "incremental_coverage_delta"
        )
        is not None
    ]

    classifications = [
        {
            "classification":
                item.get(
                    "functional_coverage_longitudinal_change", {}
                ).get("classification"),
            "numerator_growth_rate": (
                float(
                    item.get(
                        "functional_coverage_longitudinal_change",
                        {},
                    ).get("explained_event_count_delta")
                    or 0.0
                )
                / max(
                    1.0,
                    float(
                        item.get(
                            "functional_coverage_longitudinal_change",
                            {},
                        ).get("previous_explained_event_count")
                        or 0.0
                    ),
                )
            ),
            "denominator_growth_rate": (
                float(
                    item.get(
                        "functional_coverage_longitudinal_change",
                        {},
                    ).get("eligible_event_count_delta")
                    or 0.0
                )
                / max(
                    1.0,
                    float(
                        item.get(
                            "functional_coverage_longitudinal_change",
                            {},
                        ).get("previous_eligible_event_count")
                        or 0.0
                    ),
                )
            ),
            "overlap_growth_rate": 0.0,
        }
        for item in candidates
        if isinstance(
            item.get("functional_coverage_longitudinal_change"),
            dict,
        )
    ]

    return {
        "candidates_with_declining_coverage": sum(
            1
            for change in changes
            if float(
                change.get("incremental_coverage_delta", 0.0)
                or 0.0
            )
            < 0.0
        ),
        "candidates_with_denominator_growth": sum(
            1
            for item in classifications
            if item.get("classification")
            in {"eligible_event_growth", "mixed"}
        ),
        "candidates_with_numerator_decline": sum(
            1
            for item in classifications
            if item.get("classification")
            in {"concept_gain_decline", "mixed"}
        ),
        "candidates_with_overlap_growth": sum(
            1
            for item in classifications
            if item.get("classification")
            in {"overlap_growth", "mixed"}
        ),
        "mean_candidate_explained_growth_rate": _mean_float(
            item.get("numerator_growth_rate")
            for item in classifications
        ),
        "mean_denominator_growth_rate": _mean_float(
            item.get("denominator_growth_rate")
            for item in classifications
        ),
        "mean_incremental_coverage_delta": _mean_float(
            item.get("incremental_coverage_delta")
            for item in changes
        ),
    }


def _mean_float(values: Any) -> float:
    numeric = [
        float(value) for value in values if value is not None
    ]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _write_outputs(
    output_dir: Path,
    result: dict[str, Any],
) -> None:
    (
        output_dir / "h07_concept_emergence_report.json"
    ).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    validation = result.get(
        "incremental_promotion_validation", {}
    )
    validation_summary = (
        validation.get("summary", {})
        if isinstance(validation, dict)
        else {}
    )

    text = (
        f"H07 decision: {result.get('decision')}\n"
        "concept candidates: "
        f"{result.get('concept_candidate_count')}\n"
        "promoted concepts: "
        f"{result.get('promoted_concept_count')}\n"
        "strong transfer successes: "
        f"{result.get('concept_strong_transfer_success_count')}\n"
        "roles seen for concept derivation: "
        f"{result.get('roles_seen_for_concept_derivation')}\n"
        "roles skipped missing carrier links: "
        f"{result.get('roles_skipped_missing_carrier_links')}\n"
        "roles skipped missing family links: "
        f"{result.get('roles_skipped_missing_family_links')}\n"
        "roles skipped missing transfer success: "
        f"{result.get('roles_skipped_missing_transfer_success')}\n"
        "roles used for concepts: "
        f"{result.get('roles_used_for_concepts')}\n"
        "source role count mean: "
        f"{result.get('source_role_count_mean')}\n"
        "source carrier count mean: "
        f"{result.get('source_carrier_count_mean')}\n"
        "cross-game max: "
        f"{result.get('concept_cross_game_count_max')}\n"
        "cross-context max: "
        f"{result.get('concept_cross_context_count_max')}\n"
        "transfer success concentration: "
        f"{result.get('concept_transfer_success_concentration')}\n"
        "max compression gain: "
        f"{result.get('max_compression_gain')}\n"
        "max promotion score: "
        f"{result.get('max_promotion_score')}\n"
        "incremental promotion validation enabled: "
        f"{bool(validation.get('enabled', False)) if isinstance(validation, dict) else False}\n"
        "concept candidates evaluated: "
        f"{validation_summary.get('concept_candidates_evaluated')}\n"
        "concepts promoted: "
        f"{validation_summary.get('concepts_promoted')}\n"
        "rejected no incremental coverage: "
        f"{validation_summary.get('concepts_rejected_no_incremental_coverage')}\n"
        "rejected insufficient relevant samples: "
        f"{validation_summary.get('concepts_rejected_insufficient_relevant_samples')}\n"
        "rejected insufficient cross scope: "
        f"{validation_summary.get('concepts_rejected_insufficient_cross_scope')}\n"
        "rejected no predictive or behavioral lift: "
        f"{validation_summary.get('concepts_rejected_no_predictive_or_behavioral_lift')}\n"
        "rejected no heldout samples: "
        f"{validation_summary.get('concepts_rejected_no_heldout_samples')}\n"
        "rejected heldout validation failed: "
        f"{validation_summary.get('concepts_rejected_heldout_validation_failed')}\n"
        "rejected below threshold: "
        f"{validation_summary.get('concepts_rejected_below_threshold')}\n"
        "concepts demoted: "
        f"{validation_summary.get('concepts_demoted')}\n"
    )

    if (
        isinstance(validation, dict)
        and bool(validation.get("enabled", False))
    ):
        for candidate in validation.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            reasons = (
                ",".join(
                    str(item)
                    for item in candidate.get(
                        "rejection_reasons", []
                    )
                )
                or "none"
            )
            text += (
                "\n"
                f"concept={candidate.get('concept_id')}\n"
                "promoted="
                f"{str(bool(candidate.get('promoted'))).lower()}\n"
                "score="
                f"{float(candidate.get('promotion_score', 0.0) or 0.0):.2f} "
                "threshold="
                f"{float(candidate.get('promotion_threshold', 0.0) or 0.0):.2f}\n"
                "incremental_coverage="
                f"{float(candidate.get('incremental_explanatory_coverage', 0.0) or 0.0):.2f}\n"
                "prediction_lift="
                f"{float(candidate.get('prediction_lift', 0.0) or 0.0):.2f}\n"
                "behavioral_lift="
                f"{float(candidate.get('heldout_action_selection_lift', 0.0) or 0.0):.2f}\n"
                f"rejection_reasons={reasons}\n"
            )

    (
        output_dir / "h07_concept_emergence_report.txt"
    ).write_text(text, encoding="utf-8")
    (
        output_dir / "h07_concept_emergence.md"
    ).write_text(
        "```\n" + text + "```\n",
        encoding="utf-8",
    )

# v6.3 canonical semantics
_evaluate_h07_concept_emergence_base = evaluate_h07_concept_emergence

def evaluate_h07_concept_emergence(*args: Any, **kwargs: Any) -> dict:
    from v6.v63_semantics import _rewrite_json, normalize_h07_result
    result = _evaluate_h07_concept_emergence_base(*args, **kwargs)
    normalize_h07_result(result)
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        _rewrite_json(output_dir, "h07_concept_emergence_report.json", result)
    return result
