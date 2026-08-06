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
