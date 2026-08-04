"""H07 hypothesis report — concept emergence from roles."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    """1.1 — Required-table validation using sqlite_master."""
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return [name for name in required if name not in tables]


def _context_id(context_key: object) -> str | None:
    """Validate context keys explicitly — reject null/empty/partial contexts."""
    if context_key in (None, ""):
        return None
    text = str(context_key).strip().lower()
    if not text or "null" in text or "none" in text:
        return None
    return "ctx:" + sha1(text.encode("utf-8")).hexdigest()[:20]


def _transfer_pair_id(
    source_game_key: object,
    target_game_key: object,
    source_context_id: str | None,
    target_context_id: str | None,
) -> str:
    payload = {
        "source_game_key": source_game_key,
        "target_game_key": target_game_key,
        "source_context_id": source_context_id,
        "target_context_id": target_context_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "h07pair:" + sha1(encoded.encode("utf-8")).hexdigest()[:20]


def _is_fully_verified(row: dict[str, object]) -> bool:
    """Explicit predicate — NOT assertion-only. Returns False for any malformed row."""
    motif_status = str(row.get("motif_provenance_status") or "missing")
    transfer_status = str(row.get("transfer_provenance_status") or "missing")
    concept_status = str(row.get("concept_validation_status") or "missing")
    return (motif_status == "verified" and transfer_status == "verified" and concept_status == "verified")


def _is_missing_chain(row: dict[str, object]) -> bool:
    motif_status = str(row.get("motif_provenance_status") or "missing")
    transfer_status = str(row.get("transfer_provenance_status") or "missing")
    concept_status = str(row.get("concept_validation_status") or "missing")
    return any(status == "missing" for status in (motif_status, transfer_status, concept_status))


def _int(value: object) -> int:
    return int(value or 0)


def _float(value: object) -> float | None:
    return None if value is None else float(value)


from hashlib import sha1


DEFAULT_PROVENANCE_SAMPLE_LIMIT = 200
DEFAULT_MAX_MAIN_REPORT_BYTES = 5_000_000


def evaluate_h07_concept_emergence(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, Any]:
    from v6.higher_order_substrate import (
        IncrementalPromotionValidationConfig,
        derive_higher_order_memory,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not already_derived and current_state.exists():
        derive_higher_order_memory(memory_dir=memory_dir, run_dir=run_dir)
    if not current_state.exists():
        result = _base_result("INSUFFICIENT_EVIDENCE", [f"Missing expected compact-memory file: {current_state}"])
        result["incremental_promotion_validation"] = _empty_incremental_promotion_validation_report(None)
        _write_outputs(output_dir, result)
        return result

    with sqlite3.connect(current_state) as conn:
        # 1.1 — Required-table validation using sqlite_master.
        required_tables = ("role_transfer_attempts", "role_candidates", "role_links", "concept_candidates", "higher_order_milestones")
        optional_tables = ("concept_promotion_state", "higher_order_milestone_history")
        missing = _missing_tables(conn, required_tables)
        if missing:
            result = _base_result("INSUFFICIENT_EVIDENCE", [f"Missing expected compact-memory table(s): {', '.join(missing)}"])
            result["incremental_promotion_validation"] = _empty_incremental_promotion_validation_report(None)
            _write_outputs(output_dir, result)
            return result

        # 1.2 — Durable concept promotion state via LEFT JOIN with COALESCE pattern.
        concepts_rows = conn.execute(
            """
            SELECT candidate.concept_signature,
                   candidate.is_promoted AS candidate_is_promoted,
                   persistent.currently_promoted AS persistent_currently_promoted,
                   persistent.promotion_status AS persistent_promotion_status,
                   persistent.validation_status AS persistent_validation_status,
                   candidate.compression_gain,
                   candidate.promotion_score,
                   candidate.transfer_success_count,
                   candidate.strong_transfer_success_count,
                   candidate.linked_role_count,
                   candidate.linked_carrier_count,
                   candidate.linked_family_count,
                   candidate.cross_context_count,
                   candidate.cross_game_count,
                   candidate.is_overconcentrated
                FROM concept_candidates AS candidate
                LEFT JOIN concept_promotion_state AS persistent
                  ON persistent.concept_signature = candidate.concept_signature
            ORDER BY candidate.concept_signature ASC
            """
        ).fetchall()

        # 1.3 — Canonical source-role identity across all role queries: COALESCE(source_role_signature, role_signature).
        roles_with_transfer_attempts = int(
            conn.execute(
                "SELECT COUNT(DISTINCT COALESCE(source_role_signature, role_signature)) FROM role_transfer_attempts"
            ).fetchone()[0]
        )
        roles_with_successful_transfers = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT COALESCE(source_role_signature, role_signature))
                FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1
                """
            ).fetchone()[0]
        )

        successful_roles = {
            str(row["role_signature"])
            for row in conn.execute(
                """
                SELECT DISTINCT COALESCE(source_role_signature, role_signature) AS canonical_source_role
                FROM role_transfer_attempts
                WHERE COALESCE(reuse_success, 0) = 1
                  AND (source_role_signature IS NOT NULL OR role_signature IS NOT NULL)
                """
            ).fetchall()
        }

        # 1.4 — Stop mixing populations: deduplicate attempt IDs from role_transfer_attempts and count each once when linked to concepts.
        successful_transfers_deduped = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT attempt_id)
                FROM role_transfer_attempts
                WHERE COALESCE(reuse_success, 0) = 1
                """
            ).fetchone()[0]
        )

        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())

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
    # Apply durable validation state: use persistent.currently_promoted when available, else fall back to candidate.is_promoted.
    promoted_rows = [row for row in concept_rows if int(row["persistent_currently_promoted"] or row["candidate_is_promoted"] or 0) == 1]
    promoted_concept_count = len(promoted_rows)

    # Only count strong_transfer_success from the deduplicated attempt population.
    concept_strong_transfer_success_count = sum(int(row["strong_transfer_success_count"] or 0) for row in concept_rows)
    mean_compression_gain = (
        sum(float(row["compression_gain"] or 0.0) for row in concept_rows) / max(1, concept_candidate_count)
        if concept_rows
        else None
    )
    max_compression_gain = max((float(row["compression_gain"] or 0.0) for row in concept_rows), default=None)
    mean_promotion_score = (
        sum(float(row["promotion_score"] or 0.0) for row in concept_rows) / max(1, concept_candidate_count)
        if concept_rows
        else None
    )
    max_promotion_score = max((float(row["promotion_score"] or 0.0) for row in concept_rows), default=None)
    candidate_cross_context_count = sum(1 for row in concept_rows if int(row["cross_context_count"] or 0) >= 1)
    candidate_cross_game_count = sum(1 for row in concept_rows if int(row["cross_game_count"] or 0) >= 1)
    promoted_cross_context_count = sum(1 for row in promoted_rows if int(row["cross_context_count"] or 0) >= 1)
    promoted_cross_game_count = sum(1 for row in promoted_rows if int(row["cross_game_count"] or 0) >= 1)

    transfer_success_rate = (
        float(successful_transfers_deduped) / float(roles_with_transfer_attempts)
        if roles_with_transfer_attempts > 0
        else None
    )
    max_source_role_count = max((int(row["linked_role_count"] or 0) for row in promoted_rows), default=0)
    max_source_family_count = max((int(row["linked_family_count"] or 0) for row in promoted_rows), default=0)

    metrics = {
        "concept_candidate_count": concept_candidate_count,
        "promoted_concept_count": promoted_concept_count,
        "mean_compression_gain": mean_compression_gain,
        "max_compression_gain": max_compression_gain,
        "mean_promotion_score": mean_promotion_score,
        "max_promotion_score": max_promotion_score,
        "concept_transfer_success_count": successful_transfers_deduped,
        "concept_strong_transfer_success_count": concept_strong_transfer_success_count,
        "transfer_success_rate": transfer_success_rate,
        "candidate_cross_context_count": candidate_cross_context_count,
        "candidate_cross_game_count": candidate_cross_game_count,
        "promoted_cross_context_count": promoted_cross_context_count,
        "promoted_cross_game_count": promoted_cross_game_count,
        "candidate_cross_game_count_max": max((int(row["cross_game_count"] or 0) for row in concept_rows), default=0),
        "candidate_cross_context_count_max": max((int(row["cross_context_count"] or 0) for row in concept_rows), default=0),
        "promoted_cross_game_count_max": max((int(row["cross_game_count"] or 0) for row in promoted_rows), default=0),
        "promoted_cross_context_count_max": max((int(row["cross_context_count"] or 0) for row in promoted_rows), default=0),
        "source_role_count_mean": (
            sum(float(row["linked_role_count"] or 0.0) for row in promoted_rows) / max(1, promoted_concept_count)
            if promoted_rows
            else None
        ),
        "source_carrier_count_mean": (
            sum(float(row["linked_carrier_count"] or 0.0) for row in promoted_rows) / max(1, promoted_concept_count)
            if promoted_rows
            else None
        ),
        "roles_seen_for_concept_derivation": roles_with_transfer_attempts,
        "roles_with_transfer_attempts": roles_with_transfer_attempts,
        "roles_with_successful_transfers": roles_with_successful_transfers,
        "roles_eligible_for_concept_derivation": roles_used_for_concepts,
        "roles_skipped_missing_carrier_links": roles_skipped_missing_carrier_links,
        "roles_skipped_missing_family_links": roles_skipped_missing_family_links,
        "roles_skipped_missing_transfer_success": roles_skipped_missing_transfer_success,
        "roles_used_for_concepts": roles_used_for_concepts,
    }

    if successful_transfers_deduped == 0 and concept_candidate_count == 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["no successful role transfers and no concept candidates available"]
    elif concepts_rows and not any(int(row["persistent_currently_promoted"] or row["candidate_is_promoted"] or 0) == 1 for row in concepts_rows):
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["No effectively promoted concept available."]
    elif (
        promoted_concept_count >= 1
        and concept_strong_transfer_success_count >= 2
        and (max_compression_gain or 0.0) >= 1.50
        and (max_promotion_score or 0.0) >= 0.55
        and (promoted_cross_context_count_max >= 3 or promoted_cross_game_count_max >= 2)
        and max_source_role_count >= 1
        and max_source_family_count >= 2
        and roles_used_for_concepts >= 3
        and transfer_success_rate is not None and transfer_success_rate > 0.0
        and ((concept_strong_transfer_success_count / max(1, successful_transfers_deduped)) <= 0.80) if successful_transfers_deduped else False
    ):
        decision = "VALID"
        missing = []
    elif concept_candidate_count > 0 and promoted_concept_count == 0:
        decision = "INSUFFICIENT_EVIDENCE"
        metrics["evidence_stage"] = "concept_precursor_only"
        missing = ["No promoted concept available."]
    elif promoted_concept_count > 0:
        if any(int(row["is_overconcentrated"] or 0) == 1 for row in promoted_rows):
            decision = "INSUFFICIENT_EVIDENCE"
            metrics["evidence_stage"] = "precursor_only"
            missing = ["Promoted concepts are overconcentrated."]
        else:
            decision = "PARTIALLY_VALID"
            missing = []
    else:
        decision = "PARTIALLY_VALID"
        missing = []

    result = _base_result(decision, missing)
    result.update(metrics)
    result["core_metrics"] = dict(metrics)
    incremental_validation = _load_incremental_promotion_validation_report(conn, None, expected_candidate_count=len(concept_rows))
    result["incremental_promotion_validation"] = incremental_validation
    _write_outputs(output_dir, result)
    return result


def _base_result(decision: str, missing_evidence: list[str]) -> dict[str, Any]:
    return {
        "hypothesis_id": "H07",
        "decision": decision,
        "missing_evidence": list(missing_evidence),
        "evidence_source": "compact_memory",
    }


def _empty_incremental_promotion_validation_report(config: IncrementalPromotionValidationConfig | None) -> dict[str, Any]:
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


def _incremental_promotion_thresholds(config: IncrementalPromotionValidationConfig) -> dict[str, int | float]:
    return {
        "min_incremental_coverage": float(config.min_incremental_coverage),
        "min_incremental_explanatory_coverage": float(config.min_incremental_explanatory_coverage),
        "min_event_prediction_gain": float(config.min_event_prediction_gain),
        "min_event_behavioral_gain": float(config.min_event_behavioral_gain),
        "min_event_compression_gain": float(config.min_event_compression_gain),
        "min_explanation_event_count": int(config.min_explanation_event_count),
        "min_relevant_heldout_event_count": int(config.min_relevant_heldout_event_count),
        "promotion_population_comparability_threshold": float(config.promotion_population_comparability_threshold),
        "min_cross_context_or_game_evidence": int(config.min_cross_context_or_game_evidence),
        "min_behavioral_or_predictive_lift": float(config.min_behavioral_or_predictive_lift),
        "demotion_failure_limit": int(config.demotion_failure_limit),
        "promotion_score_threshold": float(config.promotion_score_threshold),
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
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'concept_promotion_validation_diagnostics'"
    ).fetchone()
    if table_exists is None:
        report.update({
            "diagnostics_complete": False,
            "consistency_warnings": ["incremental promotion validation diagnostics table is missing"],
        })
        return report
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in conn.execute(
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
    ).fetchall():
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            warnings.append("invalid incremental promotion validation diagnostic payload")
            continue
        if not isinstance(payload, dict):
            warnings.append("invalid incremental promotion validation diagnostic payload")
            continue
        candidates.append(payload)
    summary = {
        "concept_candidates_evaluated": len(candidates),
        "concepts_promoted": sum(1 for item in candidates if bool(item.get("promoted"))),
        "concepts_rejected_no_incremental_coverage": sum(
            1
            for item in candidates
            if any(
                reason in {"relevant_coverage_below_threshold", "insufficient_relevant_samples"}
                for reason in item.get("rejection_reasons", [])
            )
        ),
        "concepts_rejected_insufficient_relevant_samples": sum(
            1 for item in candidates if "insufficient_relevant_samples" in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_insufficient_cross_scope": sum(
            1 for item in candidates if "insufficient_cross_context_or_game_evidence" in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_no_predictive_or_behavioral_lift": sum(
            1 for item in candidates if "no_predictive_or_behavioral_lift" in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_no_heldout_samples": sum(
            1 for item in candidates if "no_heldout_samples" in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_heldout_validation_failed": sum(
            1 for item in candidates if "heldout_validation_failed" in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_below_threshold": sum(
            1 for item in candidates if "below_promotion_score_threshold" in item.get("rejection_reasons", [])
        ),
        "concepts_demoted": sum(1 for item in candidates if bool(item.get("demoted"))),
    }
    if summary["concept_candidates_evaluated"] != expected_candidate_count:
        warnings.append("candidate count does not match incremental validation diagnostics")
    for candidate in candidates:
        promoted = bool(candidate.get("promoted"))
        reasons = list(candidate.get("rejection_reasons", []))
        if (promoted and reasons) or (not promoted and not reasons):
            warnings.append(f"inconsistent rejection reasons for {candidate.get('concept_id', 'unknown')}")
        for error in candidate.get("diagnostics_errors", []):
            warnings.append(f"coverage diagnostics error for {candidate.get('concept_id', 'unknown')}: {error}")
    report["summary"] = summary
    report["candidates"] = candidates
    report["incremental_coverage_aggregate"] = _incremental_coverage_aggregate(candidates)
    report["diagnostics_complete"] = not warnings
    report["consistency_warnings"] = warnings
    return report


def _incremental_coverage_aggregate(candidates: list[dict[str, Any]]) -> dict[str, int | float]:
    changes = [
        item.get("coverage_longitudinal_change", {})
        for item in candidates
        if isinstance(item.get("coverage_longitudinal_change"), dict)
        and item["coverage_longitudinal_change"].get("incremental_coverage_delta") is not None
    ]
    classifications = [
        {
            "classification": item.get("functional_coverage_longitudinal_change", {}).get("classification"),
            "numerator_growth_rate": (
                float(item.get("functional_coverage_longitudinal_change", {}).get("explained_event_count_delta") or 0.0)
                / max(1.0, float(item.get("functional_coverage_longitudinal_change", {}).get("previous_explained_event_count") or 0.0))
            ),
            "denominator_growth_rate": (
                float(item.get("functional_coverage_longitudinal_change", {}).get("eligible_event_count_delta") or 0.0)
                / max(1.0, float(item.get("functional_coverage_longitudinal_change", {}).get("previous_eligible_event_count") or 0.0))
            ),
            "overlap_growth_rate": 0.0,
        }
        for item in candidates
        if isinstance(item.get("functional_coverage_longitudinal_change"), dict)
    ]
    return {
        "candidates_with_declining_coverage": sum(
            1 for change in changes if float(change.get("incremental_coverage_delta", 0.0) or 0.0) < 0.0
        ),
        "candidates_with_denominator_growth": sum(
            1 for item in classifications if item.get("classification") in {"eligible_event_growth", "mixed"}
        ),
        "candidates_with_numerator_decline": sum(
            1 for item in classifications if item.get("classification") in {"concept_gain_decline", "mixed"}
        ),
        "candidates_with_overlap_growth": sum(
            1 for item in classifications if item.get("classification") in {"overlap_growth", "mixed"}
        ),
        "mean_candidate_explained_growth_rate": _mean_float(
            item.get("numerator_growth_rate") for item in classifications
        ),
        "mean_denominator_growth_rate": _mean_float(
            item.get("denominator_growth_rate") for item in classifications
        ),
        "mean_incremental_coverage_delta": _mean_float(
            item.get("incremental_coverage_delta") for item in changes
        ),
    }


def _mean_float(values: Any) -> float:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h07_concept_emergence_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    validation = result.get("incremental_promotion_validation", {})
    validation_summary = validation.get("summary", {}) if isinstance(validation, dict) else {}
    text = (
        f"H07 decision: {result.get('decision')}\n"
        f"concept candidates: {result.get('concept_candidate_count')}\n"
        f"promoted concepts: {result.get('promoted_concept_count')}\n"
        f"strong transfer successes: {result.get('concept_strong_transfer_success_count')}\n"
        f"roles seen for concept derivation: {result.get('roles_seen_for_concept_derivation')}\n"
        f"roles skipped missing carrier links: {result.get('roles_skipped_missing_carrier_links')}\n"
        f"roles skipped missing family links: {result.get('roles_skipped_missing_family_links')}\n"
        f"roles skipped missing transfer success: {result.get('roles_skipped_missing_transfer_success')}\n"
        f"roles used for concepts: {result.get('roles_used_for_concepts')}\n"
        f"source role count mean: {result.get('source_role_count_mean')}\n"
        f"source carrier count mean: {result.get('source_carrier_count_mean')}\n"
        f"cross-game max: {result.get('concept_cross_game_count_max')}\n"
        f"cross-context max: {result.get('concept_cross_context_count_max')}\n"
        f"transfer success concentration: {result.get('concept_transfer_success_concentration')}\n"
        f"max compression gain: {result.get('max_compression_gain')}\n"
        f"max promotion score: {result.get('max_promotion_score')}\n"
        f"incremental promotion validation enabled: {bool(validation.get('enabled', False)) if isinstance(validation, dict) else False}\n"
        f"concept candidates evaluated: {validation_summary.get('concept_candidates_evaluated')}\n"
        f"concepts promoted: {validation_summary.get('concepts_promoted')}\n"
        f"rejected no incremental coverage: {validation_summary.get('concepts_rejected_no_incremental_coverage')}\n"
        f"rejected insufficient relevant samples: {validation_summary.get('concepts_rejected_insufficient_relevant_samples')}\n"
        f"rejected insufficient cross scope: {validation_summary.get('concepts_rejected_insufficient_cross_scope')}\n"
        f"rejected no predictive or behavioral lift: {validation_summary.get('concepts_rejected_no_predictive_or_behavioral_lift')}\n"
        f"rejected no heldout samples: {validation_summary.get('concepts_rejected_no_heldout_samples')}\n"
        f"rejected heldout validation failed: {validation_summary.get('concepts_rejected_heldout_validation_failed')}\n"
        f"rejected below threshold: {validation_summary.get('concepts_rejected_below_threshold')}\n"
        f"concepts demoted: {validation_summary.get('concepts_demoted')}\n"
    )
    if isinstance(validation, dict) and bool(validation.get("enabled", False)):
        for candidate in validation.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            reasons = ",".join(str(item) for item in candidate.get("rejection_reasons", [])) or "none"
            longitudinal = candidate.get("functional_coverage_longitudinal_change", {})
            previous_coverage = longitudinal.get("previous_incremental_explanatory_coverage") if isinstance(longitudinal, dict) else None
            current_coverage = longitudinal.get("current_incremental_explanatory_coverage") if isinstance(longitudinal, dict) else None
            if previous_coverage is None:
                coverage_section = "Functional coverage baseline unavailable.\n"
            elif current_coverage is not None and float(current_coverage) < float(previous_coverage):
                coverage_section = (
                    f"Functional coverage declined from {float(previous_coverage):.5f} to {float(current_coverage):.5f}.\n"
                    f"Explained events: {longitudinal.get('previous_explained_event_count')} → {longitudinal.get('current_explained_event_count')}.\n"
                    f"Eligible events: {longitudinal.get('previous_eligible_event_count')} → {longitudinal.get('current_eligible_event_count')}.\n"
                    f"Primary cause: {longitudinal.get('classification', 'not_determined')}.\n"
                )
            else:
                coverage_section = "Functional coverage did not decline.\n"
            text += (
                "\n"
                f"concept={candidate.get('concept_id')}\n"
                f"promoted={str(bool(candidate.get('promoted'))).lower()}\n"
                f"score={float(candidate.get('promotion_score', 0.0) or 0.0):.2f} "
                f"threshold={float(candidate.get('promotion_threshold', 0.0) or 0.0):.2f}\n"
                f"incremental_coverage={float(candidate.get('incremental_explanatory_coverage', 0.0) or 0.0):.2f}\n"
                f"relevant_heldout_samples={int(candidate.get('relevant_heldout_event_count', 0) or 0)} "
                f"global_reach={int(candidate.get('global_explanatory_reach', 0) or 0)}\n"
                f"baseline={candidate.get('baseline_type', 'N/A')} "
                f"candidate={candidate.get('candidate_type', 'N/A')} "
                f"population={int(candidate.get('common_event_count', 0) or 0)}/"
                f"{int(candidate.get('baseline_event_count', 0) or 0)}/"
                f"{int(candidate.get('candidate_event_count', 0) or 0)}\n"
                f"retained_events={int(candidate.get('retained_event_count', 0) or 0)} "
                f"current_coverage={float(candidate.get('current_population_coverage', 0.0) or 0.0):.2f} "
                f"retained_coverage={float(candidate.get('retained_population_coverage', 0.0) or 0.0):.2f}\n"
                f"prediction_lift={float(candidate.get('prediction_lift', 0.0) or 0.0):.2f}\n"
                f"behavioral_lift={float(candidate.get('heldout_action_selection_lift', 0.0) or 0.0):.2f}\n"
                f"rejection_reasons={reasons}\n"
                f"eligible_events={int(candidate.get('eligible_explanation_event_count', 0) or 0)} "
                f"explained_events={int(candidate.get('explained_event_count', 0) or 0)} "
                f"structural_overlap={float(candidate.get('structural_overlap_ratio', 0.0) or 0.0):.2f}\n"
                f"{coverage_section}"
            )
    (output_dir / "h07_concept_emergence_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h07_concept_emergence.md").write_text("```\n" + text + "```\n", encoding="utf-8")
