from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.higher_order_substrate import (
    IncrementalPromotionValidationConfig,
    derive_higher_order_memory,
)


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
        result = _base_result("INSUFFICIENT_EVIDENCE", [f"Missing expected compact-memory file: {current_state}"])
        result["incremental_promotion_validation"] = _empty_incremental_promotion_validation_report(
            incremental_promotion_validation
        )
        _write_outputs(output_dir, result)
        return result
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        transfer_attempt_count = int(conn.execute("SELECT COUNT(*) FROM role_transfer_attempts").fetchone()[0])
        successful_transfers = int(conn.execute("SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1").fetchone()[0])
        roles_seen_for_concept_derivation = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM role_candidates
                WHERE COALESCE(is_emergent, 0) = 1 OR COALESCE(role_stability_score, 0.0) >= 0.50
                """
            ).fetchone()[0]
        )
        role_link_rows = conn.execute(
            "SELECT role_signature, linked_type FROM role_links ORDER BY role_signature ASC, linked_type ASC"
        ).fetchall()
        linked_types_by_role: dict[str, set[str]] = {}
        for row in role_link_rows:
            linked_types_by_role.setdefault(str(row["role_signature"]), set()).add(str(row["linked_type"]))
        successful_roles = {
            str(row["role_signature"])
            for row in conn.execute(
                """
                SELECT DISTINCT role_signature
                FROM role_transfer_attempts
                WHERE COALESCE(reuse_success, 0) = 1
                """
            ).fetchall()
        }
        concept_rows = conn.execute(
            """
            SELECT concept_signature, compression_gain, promotion_score, transfer_success_count,
                   strong_transfer_success_count, linked_role_count, linked_carrier_count, linked_family_count,
                   cross_context_count, cross_game_count, is_promoted,
                   transfer_success_concentration, is_overconcentrated
            FROM concept_candidates
            ORDER BY concept_signature ASC
            """
        ).fetchall()
        incremental_validation = _load_incremental_promotion_validation_report(
            conn,
            config=incremental_promotion_validation,
            expected_candidate_count=len(concept_rows),
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
    promoted_rows = [row for row in concept_rows if int(row["is_promoted"] or 0) == 1]
    promoted_concept_count = len(promoted_rows)
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
    cross_context_concept_count = sum(1 for row in concept_rows if int(row["cross_context_count"] or 0) >= 1)
    cross_game_concept_count = sum(1 for row in concept_rows if int(row["cross_game_count"] or 0) >= 1)
    concept_transfer_success_count = successful_transfers
    concept_strong_transfer_success_count = sum(int(row["strong_transfer_success_count"] or 0) for row in concept_rows)
    source_role_count_mean = (
        sum(float(row["linked_role_count"] or 0.0) for row in promoted_rows) / max(1, promoted_concept_count)
        if promoted_rows
        else None
    )
    source_carrier_count_mean = (
        sum(float(row["linked_carrier_count"] or 0.0) for row in promoted_rows) / max(1, promoted_concept_count)
        if promoted_rows
        else None
    )
    concept_cross_game_count_max = max((int(row["cross_game_count"] or 0) for row in promoted_rows), default=0)
    concept_cross_context_count_max = max((int(row["cross_context_count"] or 0) for row in promoted_rows), default=0)
    strong_transfer_counts = [int(row["strong_transfer_success_count"] or 0) for row in promoted_rows]
    concept_transfer_success_concentration = (
        max(strong_transfer_counts) / max(1, sum(strong_transfer_counts))
        if strong_transfer_counts and sum(strong_transfer_counts) > 0
        else None
    )
    overconcentrated_concept_count = sum(1 for row in concept_rows if int(row["is_overconcentrated"] or 0) == 1)
    promoted_overconcentrated_concept_count = sum(
        1 for row in concept_rows
        if int(row["is_overconcentrated"] or 0) == 1 and int(row["is_promoted"] or 0) == 1
    )
    transfer_success_rate = (
        float(concept_strong_transfer_success_count) / float(transfer_attempt_count)
        if transfer_attempt_count > 0
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
        "concept_transfer_success_count": concept_transfer_success_count,
        "concept_strong_transfer_success_count": concept_strong_transfer_success_count,
        "transfer_success_rate": transfer_success_rate,
        "cross_context_concept_count": cross_context_concept_count,
        "cross_game_concept_count": cross_game_concept_count,
        "concept_cross_game_count_max": concept_cross_game_count_max,
        "concept_cross_context_count_max": concept_cross_context_count_max,
        "source_role_count_mean": source_role_count_mean,
        "source_carrier_count_mean": source_carrier_count_mean,
        "max_source_role_count": max_source_role_count,
        "max_source_family_count": max_source_family_count,
        "concept_transfer_success_concentration": concept_transfer_success_concentration,
        "overconcentrated_concept_count": overconcentrated_concept_count,
        "promoted_overconcentrated_concept_count": promoted_overconcentrated_concept_count,
        "first_concept_candidate_step": milestone_map.get("first_concept_candidate_step"),
        "first_promoted_concept_step": milestone_map.get("first_promoted_concept_step"),
        "first_role_transfer_success_step": milestone_map.get("first_role_transfer_success_step"),
        "roles_seen_for_concept_derivation": roles_seen_for_concept_derivation,
        "roles_skipped_missing_carrier_links": roles_skipped_missing_carrier_links,
        "roles_skipped_missing_family_links": roles_skipped_missing_family_links,
        "roles_skipped_missing_transfer_success": roles_skipped_missing_transfer_success,
        "roles_used_for_concepts": roles_used_for_concepts,
        "evidence_stage": None,
    }
    metrics.update(incremental_validation["summary"])
    if successful_transfers == 0 and concept_candidate_count == 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["no successful role transfers and no concept candidates available"]
    elif transfer_attempt_count <= 0:
        decision = "INCONCLUSIVE"
        missing = ["no role transfer attempts available"]
    elif successful_transfers > 0 and concept_candidate_count == 0:
        if roles_skipped_missing_family_links > 0 or roles_skipped_missing_carrier_links > 0:
            decision = "INSUFFICIENT_EVIDENCE"
            missing = []
            if roles_skipped_missing_family_links > 0:
                missing.append("concept derivation blocked by missing role-family links")
            if roles_skipped_missing_carrier_links > 0:
                missing.append("concept derivation blocked by missing role-carrier links")
        else:
            decision = "INVALID"
            missing = []
    elif (
        promoted_concept_count >= 1
        and concept_strong_transfer_success_count >= 2
        and (max_compression_gain or 0.0) >= 1.50
        and (max_promotion_score or 0.0) >= 0.55
        and (concept_cross_context_count_max >= 3 or concept_cross_game_count_max >= 2)
        and max_source_role_count >= 1
        and max_source_family_count >= 2
        and roles_used_for_concepts >= 3
        and (transfer_success_rate or 0.0) > 0.0
        and ((concept_transfer_success_concentration or 0.0) <= 0.80)
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
        if promoted_overconcentrated_concept_count == promoted_concept_count and promoted_concept_count > 0:
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
        "min_cross_context_or_game_evidence": int(config.min_cross_context_or_game_evidence),
        "min_behavioral_or_predictive_lift": float(config.min_behavioral_or_predictive_lift),
        "demotion_failure_limit": int(config.demotion_failure_limit),
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
        raise AssertionError("incremental promotion validation diagnostics table is missing")
    candidates: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT payload_json FROM concept_promotion_validation_diagnostics ORDER BY concept_signature ASC"
    ).fetchall():
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise AssertionError("invalid incremental promotion validation diagnostic payload")
        candidates.append(payload)
    summary = {
        "concept_candidates_evaluated": len(candidates),
        "concepts_promoted": sum(1 for item in candidates if bool(item.get("promoted"))),
        "concepts_rejected_no_incremental_coverage": sum(
            1 for item in candidates if "no_incremental_coverage" in item.get("rejection_reasons", [])
        ),
        "concepts_rejected_insufficient_cross_scope": sum(
            1
            for item in candidates
            if "insufficient_cross_context_or_game_evidence" in item.get("rejection_reasons", [])
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
            1 for item in candidates if "below_promotion_threshold" in item.get("rejection_reasons", [])
        ),
        "concepts_demoted": sum(1 for item in candidates if bool(item.get("demoted"))),
    }
    assert summary["concept_candidates_evaluated"] == expected_candidate_count
    assert summary["concepts_promoted"] == sum(1 for item in candidates if bool(item.get("promoted")))
    assert summary["concepts_demoted"] == sum(1 for item in candidates if bool(item.get("demoted")))
    for candidate in candidates:
        promoted = bool(candidate.get("promoted"))
        reasons = list(candidate.get("rejection_reasons", []))
        assert not reasons if promoted else bool(reasons)
    report["summary"] = summary
    report["candidates"] = candidates
    return report


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
            text += (
                "\n"
                f"concept={candidate.get('concept_id')}\n"
                f"promoted={str(bool(candidate.get('promoted'))).lower()}\n"
                f"score={float(candidate.get('promotion_score', 0.0) or 0.0):.2f} "
                f"threshold={float(candidate.get('promotion_threshold', 0.0) or 0.0):.2f}\n"
                f"incremental_coverage={float(candidate.get('incremental_explanatory_coverage', 0.0) or 0.0):.2f}\n"
                f"heldout_samples={int(candidate.get('validation_evidence_count', 0) or 0)}\n"
                f"prediction_lift={float(candidate.get('prediction_lift', 0.0) or 0.0):.2f}\n"
                f"behavioral_lift={float(candidate.get('heldout_action_selection_lift', 0.0) or 0.0):.2f}\n"
                f"rejection_reasons={reasons}\n"
            )
    (output_dir / "h07_concept_emergence_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h07_concept_emergence.md").write_text("```\n" + text + "```\n", encoding="utf-8")
