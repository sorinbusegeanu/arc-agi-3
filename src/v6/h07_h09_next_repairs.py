from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PATCHED = False


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def _candidate_signature(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("candidate_signature")
        or candidate.get("concept_id")
        or ""
    )


def _current_concept_signatures(connection: sqlite3.Connection) -> set[str]:
    if not _table_exists(connection, "concept_candidates"):
        return set()
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT concept_signature FROM concept_candidates "
            "ORDER BY concept_signature ASC"
        ).fetchall()
        if row[0] not in (None, "")
    }


def _float(candidate: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = candidate.get(key)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _int(candidate: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = candidate.get(key)
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _current_rejection_reasons(
    candidate: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []

    relevant_samples = _int(candidate, "relevant_heldout_event_count")
    min_samples = int(thresholds.get("min_relevant_heldout_event_count", 0) or 0)
    if relevant_samples <= 0:
        reasons.append("no_heldout_samples")
    elif relevant_samples < min_samples:
        reasons.append("insufficient_relevant_samples")

    coverage = max(
        _float(candidate, "relevant_incremental_coverage"),
        _float(candidate, "incremental_explanatory_coverage"),
    )
    min_coverage = max(
        float(thresholds.get("min_incremental_coverage", 0.0) or 0.0),
        float(
            thresholds.get(
                "min_incremental_explanatory_coverage",
                0.0,
            )
            or 0.0
        ),
    )
    if coverage < min_coverage:
        reasons.append("relevant_coverage_below_threshold")

    cross_scope = max(
        _int(candidate, "cross_context_evidence_count"),
        _int(candidate, "cross_game_evidence_count"),
    )
    min_cross_scope = int(
        thresholds.get("min_cross_context_or_game_evidence", 0) or 0
    )
    if cross_scope < min_cross_scope:
        reasons.append("insufficient_cross_context_or_game_evidence")

    predictive_or_behavioral_lift = max(
        _float(candidate, "heldout_prediction_lift"),
        _float(candidate, "heldout_action_selection_lift"),
        _float(candidate, "heldout_transfer_lift"),
        _float(candidate, "mean_prediction_gain"),
        _float(candidate, "mean_behavioral_gain"),
    )
    min_lift = float(
        thresholds.get("min_behavioral_or_predictive_lift", 0.0) or 0.0
    )
    if predictive_or_behavioral_lift < min_lift:
        reasons.append("no_predictive_or_behavioral_lift")

    score = _float(
        candidate,
        "promotion_gate_score_used",
        _float(candidate, "promotion_score"),
    )
    threshold = float(
        candidate.get("promotion_threshold")
        or thresholds.get("promotion_score_threshold", 0.0)
        or 0.0
    )
    score_pass = bool(
        candidate.get("promotion_score_gate_passed", score >= threshold)
    )
    if not score_pass or score < threshold:
        reasons.append("below_promotion_score_threshold")

    if not bool(candidate.get("population_comparable", True)):
        reasons.append("heldout_population_not_comparable")

    if (
        not bool(candidate.get("current_validation_passed"))
        and not reasons
    ):
        reasons.append("heldout_validation_failed")

    return list(dict.fromkeys(reasons))


def _repair_h07_validation_report(
    *,
    connection: sqlite3.Connection,
    report: dict[str, Any],
    expected_candidate_count: int,
) -> dict[str, Any]:
    if not bool(report.get("enabled")):
        return report

    current_signatures = _current_concept_signatures(connection)
    all_candidates = [
        dict(candidate)
        for candidate in report.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    current_candidates = [
        candidate
        for candidate in all_candidates
        if _candidate_signature(candidate) in current_signatures
    ]
    stale_candidates = [
        candidate
        for candidate in all_candidates
        if _candidate_signature(candidate) not in current_signatures
    ]
    thresholds = dict(report.get("thresholds", {}) or {})

    for candidate in current_candidates:
        persistent_promoted = bool(
            candidate.get("currently_promoted")
            or candidate.get("promoted")
            or candidate.get("historically_promoted")
        )
        current_promoted = bool(
            candidate.get("current_validation_passed")
            and candidate.get(
                "promotion_score_gate_passed",
                _float(candidate, "promotion_score")
                >= _float(candidate, "promotion_threshold"),
            )
        )
        candidate["persistent_or_historical_promoted"] = persistent_promoted
        candidate["promotion_retained_without_current_validation"] = bool(
            persistent_promoted and not current_promoted
        )
        candidate["report_currently_validated_promoted"] = current_promoted
        candidate["promoted"] = current_promoted
        candidate["rejection_reasons"] = (
            [] if current_promoted
            else _current_rejection_reasons(candidate, thresholds)
        )

    summary = {
        "concept_candidates_evaluated": len(current_candidates),
        "concepts_promoted": sum(
            bool(item.get("report_currently_validated_promoted"))
            for item in current_candidates
        ),
        "concepts_persistently_retained": sum(
            bool(item.get("persistent_or_historical_promoted"))
            for item in current_candidates
        ),
        "concepts_retained_without_current_validation": sum(
            bool(item.get("promotion_retained_without_current_validation"))
            for item in current_candidates
        ),
        "stale_historical_diagnostics_excluded": len(stale_candidates),
        "concepts_rejected_no_incremental_coverage": sum(
            "relevant_coverage_below_threshold"
            in item.get("rejection_reasons", [])
            for item in current_candidates
        ),
        "concepts_rejected_insufficient_relevant_samples": sum(
            "insufficient_relevant_samples"
            in item.get("rejection_reasons", [])
            for item in current_candidates
        ),
        "concepts_rejected_insufficient_cross_scope": sum(
            "insufficient_cross_context_or_game_evidence"
            in item.get("rejection_reasons", [])
            for item in current_candidates
        ),
        "concepts_rejected_no_predictive_or_behavioral_lift": sum(
            "no_predictive_or_behavioral_lift"
            in item.get("rejection_reasons", [])
            for item in current_candidates
        ),
        "concepts_rejected_no_heldout_samples": sum(
            "no_heldout_samples"
            in item.get("rejection_reasons", [])
            for item in current_candidates
        ),
        "concepts_rejected_heldout_validation_failed": sum(
            (
                "heldout_validation_failed"
                in item.get("rejection_reasons", [])
                or "heldout_population_not_comparable"
                in item.get("rejection_reasons", [])
            )
            for item in current_candidates
        ),
        "concepts_rejected_below_threshold": sum(
            "below_promotion_score_threshold"
            in item.get("rejection_reasons", [])
            for item in current_candidates
        ),
        "concepts_demoted": sum(
            bool(item.get("demoted")) for item in current_candidates
        ),
    }

    warnings: list[str] = []
    if len(current_candidates) != expected_candidate_count:
        warnings.append(
            "current candidate count does not match current incremental "
            "validation diagnostics"
        )
    for candidate in current_candidates:
        for error in candidate.get("diagnostics_errors", []) or []:
            warnings.append(
                "coverage diagnostics error for "
                f"{_candidate_signature(candidate) or 'unknown'}: {error}"
            )

    report["summary"] = summary
    report["candidates"] = current_candidates
    report["current_candidate_signatures"] = sorted(current_signatures)
    report["stale_historical_diagnostic_count"] = len(stale_candidates)
    report["stale_historical_diagnostic_signatures_sample"] = [
        _candidate_signature(candidate)
        for candidate in stale_candidates[:20]
    ]
    report["diagnostics_complete"] = not warnings
    report["consistency_warnings"] = warnings
    return report


def _update_h07_result(
    h07_module: Any,
    *,
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validation = result.get("incremental_promotion_validation")
    if not isinstance(validation, dict) or not validation.get("enabled"):
        return result

    summary = dict(validation.get("summary", {}) or {})
    persistent_count = int(result.get("promoted_concept_count") or 0)
    current_count = int(summary.get("concepts_promoted") or 0)

    result["persistent_retained_concept_count"] = persistent_count
    result["current_validated_promoted_concept_count"] = current_count
    result["promoted_concept_count"] = current_count
    result["stale_historical_promotion_diagnostic_count"] = int(
        validation.get("stale_historical_diagnostic_count") or 0
    )

    for key, value in summary.items():
        result[key] = value

    if current_count <= 0:
        result["promoted_cross_context_count"] = 0
        result["promoted_cross_game_count"] = 0
        result["promoted_cross_context_count_max"] = 0
        result["promoted_cross_game_count_max"] = 0
        result["source_role_count_mean"] = None
        result["source_carrier_count_mean"] = None
        result["max_source_role_count"] = 0
        result["max_source_family_count"] = 0
        result["promoted_overconcentrated_concept_count"] = 0
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["evidence_stage"] = "current_validation_failed"
        result["missing_evidence"] = list(
            dict.fromkeys(
                [
                    *list(result.get("missing_evidence", []) or []),
                    "No current concept candidate passed incremental validation.",
                ]
            )
        )
    elif not bool(validation.get("diagnostics_complete", True)):
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"] = list(
            dict.fromkeys(
                [
                    *list(result.get("missing_evidence", []) or []),
                    *list(validation.get("consistency_warnings", []) or []),
                ]
            )
        )
    else:
        valid = all(
            (
                current_count >= 1,
                int(result.get("concept_strong_transfer_success_count") or 0) >= 2,
                float(result.get("max_compression_gain") or 0.0) >= 1.50,
                float(result.get("max_promotion_score") or 0.0) >= 0.55,
                (
                    int(result.get("promoted_cross_context_count_max") or 0) >= 3
                    or int(result.get("promoted_cross_game_count_max") or 0) >= 2
                ),
                int(result.get("max_source_role_count") or 0) >= 1,
                int(result.get("max_source_family_count") or 0) >= 2,
                int(result.get("roles_used_for_concepts") or 0) >= 3,
                float(result.get("transfer_success_rate") or 0.0) > 0.0,
                float(
                    result.get("concept_transfer_success_concentration")
                    if result.get("concept_transfer_success_concentration")
                    is not None
                    else 1.0
                )
                <= 0.80,
                int(
                    result.get("promoted_overconcentrated_concept_count")
                    or 0
                )
                == 0,
            )
        )
        result["decision"] = "VALID" if valid else "PARTIALLY_VALID"
        result["missing_evidence"] = []

    core = dict(result.get("core_metrics", {}) or {})
    for key in (
        "promoted_concept_count",
        "persistent_retained_concept_count",
        "current_validated_promoted_concept_count",
        "stale_historical_promotion_diagnostic_count",
        *summary.keys(),
    ):
        core[key] = result.get(key)
    result["core_metrics"] = core
    h07_module._write_outputs(Path(output_dir), result)
    return result


def _complete_scope(value: Any) -> bool:
    if value in (None, ""):
        return False
    text = str(value).strip().lower()
    return bool(text) and "null" not in text and "none" not in text and text not in {
        "[]",
        "{}",
    }


def _real_scope_values(
    observations: list[dict[str, Any]],
    *,
    kind: str,
) -> set[str]:
    values: set[str] = set()
    for observation in observations:
        for side in ("source", "target"):
            key = f"{side}_{kind}_key"
            surrogate_key = f"{side}_{kind}_is_surrogate"
            value = observation.get(key)
            if (
                _complete_scope(value)
                and int(observation.get(surrogate_key) or 0) == 0
            ):
                values.add(str(value))
    return values


def _repair_h09_result(
    h09_module: Any,
    *,
    memory_dir: Path,
    output_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return result

    with sqlite3.connect(current_state) as connection:
        connection.row_factory = sqlite3.Row
        if not all(
            _table_exists(connection, table)
            for table in (
                "future_option_motifs",
                "future_option_events",
                "future_option_motif_observations",
            )
        ):
            return result

        motifs = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM future_option_motifs "
                "ORDER BY motif_signature ASC"
            ).fetchall()
        ]
        events = {
            str(row["event_id"]): dict(row)
            for row in connection.execute(
                "SELECT * FROM future_option_events ORDER BY event_id ASC"
            ).fetchall()
        }
        observations = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM future_option_motif_observations "
                "ORDER BY motif_signature ASC, event_id ASC"
            ).fetchall()
        ]

    verified_by_motif: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        if str(observation.get("provenance_status") or "missing") != "verified":
            continue
        verified_by_motif[str(observation.get("motif_signature") or "")].append(
            observation
        )

    records: list[dict[str, Any]] = []
    qualifying: list[dict[str, Any]] = []
    cross_game_motifs: set[str] = set()
    cross_context_motifs: set[str] = set()
    cross_game_observation_count = 0
    cross_context_observation_count = 0

    for motif in motifs:
        signature = str(motif.get("motif_signature") or "")
        verified = verified_by_motif.get(signature, [])
        games = _real_scope_values(verified, kind="game")
        contexts = _real_scope_values(verified, kind="context")
        has_cross_game_recurrence = len(games) >= 2
        has_cross_context_recurrence = len(contexts) >= 2

        if has_cross_game_recurrence:
            cross_game_motifs.add(signature)
            cross_game_observation_count += len(verified)
        if has_cross_context_recurrence:
            cross_context_motifs.add(signature)
            cross_context_observation_count += len(verified)

        event_ids = {
            str(observation.get("event_id"))
            for observation in verified
            if observation.get("event_id") not in (None, "")
        }
        verified_events = [
            events[event_id]
            for event_id in sorted(event_ids)
            if event_id in events
        ]
        nonzero_count = sum(
            abs(float(event.get("option_delta") or 0.0)) > 0.0
            for event in verified_events
        )
        unknown_count = sum(
            str(event.get("motif_type") or "unknown") == "unknown"
            or str(event.get("classification_source") or "unknown")
            .lower()
            .startswith("unknown")
            for event in verified_events
        )

        record = {
            "motif_signature": signature,
            "motif_type": str(motif.get("motif_type") or "unknown"),
            "is_emergent": int(motif.get("is_emergent") or 0),
            "provenance_status": str(
                motif.get("provenance_status") or "missing"
            ),
            "has_verified_observation": bool(verified),
            "verified_distinct_game_count": len(games),
            "verified_distinct_context_count": len(contexts),
            "verified_game_keys_sample": sorted(games)[:20],
            "verified_context_count": len(contexts),
            "has_verified_cross_game_recurrence": has_cross_game_recurrence,
            "has_verified_cross_context_recurrence": (
                has_cross_context_recurrence
            ),
            # Compatibility aliases now represent recurrence across verified
            # observations, rather than requiring a source-target pair in one
            # observation.
            "has_verified_cross_game_observation": has_cross_game_recurrence,
            "has_verified_cross_context_observation": (
                has_cross_context_recurrence
            ),
            "verified_event_count": len(verified_events),
            "verified_nonzero_option_delta_event_count": nonzero_count,
            "unknown_verified_event_count": unknown_count,
            "classification_sources": sorted(
                {
                    str(event.get("classification_source") or "unknown")
                    for event in verified_events
                }
            ),
        }
        records.append(record)

        if (
            record["is_emergent"] == 1
            and record["provenance_status"] == "verified"
            and record["motif_type"] != "unknown"
            and record["has_verified_observation"]
            and (
                has_cross_game_recurrence
                or has_cross_context_recurrence
            )
            and nonzero_count >= 1
        ):
            qualifying.append(record)

    qualifying_types = Counter(
        str(record["motif_type"]) for record in qualifying
    )
    verified_unknown_ratio = result.get("verified_unknown_event_ratio")

    result["pairwise_verified_cross_game_observation_count"] = result.get(
        "verified_cross_game_observation_count"
    )
    result["pairwise_verified_cross_context_observation_count"] = result.get(
        "verified_cross_context_observation_count"
    )
    result["verified_cross_game_recurrence_motif_count"] = len(
        cross_game_motifs
    )
    result["verified_cross_context_recurrence_motif_count"] = len(
        cross_context_motifs
    )
    result["verified_cross_game_recurrence_observation_count"] = (
        cross_game_observation_count
    )
    result["verified_cross_context_recurrence_observation_count"] = (
        cross_context_observation_count
    )
    result["cross_game_motif_count"] = len(cross_game_motifs)
    result["cross_context_motif_count"] = len(cross_context_motifs)
    result["qualifying_emergent_motif_count"] = len(qualifying)
    result["qualifying_emergent_motif_signatures"] = [
        str(record["motif_signature"]) for record in qualifying
    ]
    result["qualifying_motif_type_count"] = len(qualifying_types)
    result["qualifying_motif_type_counts"] = dict(
        sorted(qualifying_types.items())
    )
    result["motif_scientific_evidence_records"] = records[:200]
    result["motif_scope_evidence_method"] = (
        "distinct_real_scopes_across_verified_observations_per_motif"
    )
    result["motif_scope_summary"] = {
        "observation_count": len(observations),
        "pairwise_verified_cross_game_observation_count": result.get(
            "pairwise_verified_cross_game_observation_count"
        ),
        "pairwise_verified_cross_context_observation_count": result.get(
            "pairwise_verified_cross_context_observation_count"
        ),
        "verified_cross_game_recurrence_motif_count": len(
            cross_game_motifs
        ),
        "verified_cross_context_recurrence_motif_count": len(
            cross_context_motifs
        ),
        "verified_cross_game_recurrence_observation_count": (
            cross_game_observation_count
        ),
        "verified_cross_context_recurrence_observation_count": (
            cross_context_observation_count
        ),
    }

    if (
        len(qualifying) >= 1
        and len(qualifying_types) >= 2
        and (
            verified_unknown_ratio is None
            or float(verified_unknown_ratio) <= 0.20
        )
    ):
        result["decision"] = "VALID"
        result["missing_evidence"] = []
    elif int(result.get("emergent_future_option_motif_count") or 0) <= 0:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"] = [
            "No emergent future-option motif available."
        ]
    else:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"] = [
            "No emergent motif satisfies verified recurrence across at "
            "least two real games or contexts with nonzero option delta."
        ]

    result["core_metrics"] = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "core_metrics",
            "motif_scope_sample",
            "motif_scientific_evidence_records",
        }
    }
    h09_module._write(Path(output_dir), result)
    return result


def apply_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("ARC_AGI3_DISABLE_H07_H09_NEXT_REPAIRS") == "1":
        return False

    import v6.hypothesis_h07_report as h07
    import v6.hypothesis_h09_report as h09

    if not getattr(h07, "_ARC_AGI3_CURRENT_CANDIDATE_REPORT_FIX", False):
        original_loader = h07._load_incremental_promotion_validation_report
        original_evaluate_h07 = h07.evaluate_h07_concept_emergence

        def repaired_loader(
            connection: sqlite3.Connection,
            *,
            config: Any,
            expected_candidate_count: int,
        ) -> dict[str, Any]:
            report = original_loader(
                connection,
                config=config,
                expected_candidate_count=expected_candidate_count,
            )
            return _repair_h07_validation_report(
                connection=connection,
                report=report,
                expected_candidate_count=expected_candidate_count,
            )

        def repaired_evaluate_h07(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original_evaluate_h07(*args, **kwargs)
            output_dir = kwargs.get("output_dir")
            if output_dir is None and len(args) >= 3:
                output_dir = args[2]
            if output_dir is None:
                return result
            return _update_h07_result(
                h07,
                result=result,
                output_dir=Path(output_dir),
            )

        h07._load_incremental_promotion_validation_report = repaired_loader
        h07.evaluate_h07_concept_emergence = repaired_evaluate_h07
        h07._ARC_AGI3_CURRENT_CANDIDATE_REPORT_FIX = True

    if not getattr(h09, "_ARC_AGI3_CROSS_OBSERVATION_SCOPE_FIX", False):
        original_evaluate_h09 = h09.evaluate_h09_future_option_motifs

        def repaired_evaluate_h09(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original_evaluate_h09(*args, **kwargs)
            memory_dir = kwargs.get("memory_dir")
            output_dir = kwargs.get("output_dir")
            if memory_dir is None and args:
                memory_dir = args[0]
            if output_dir is None and len(args) >= 3:
                output_dir = args[2]
            if memory_dir is None or output_dir is None:
                return result
            return _repair_h09_result(
                h09,
                memory_dir=Path(memory_dir),
                output_dir=Path(output_dir),
                result=result,
            )

        h09.evaluate_h09_future_option_motifs = repaired_evaluate_h09
        h09._ARC_AGI3_CROSS_OBSERVATION_SCOPE_FIX = True

    # hypothesis_suite_report imports both evaluators by value.
    import v6.hypothesis_suite_report as suite

    suite.evaluate_h07_concept_emergence = (
        h07.evaluate_h07_concept_emergence
    )
    suite.evaluate_h09_future_option_motifs = (
        h09.evaluate_h09_future_option_motifs
    )

    _PATCHED = True
    return True
