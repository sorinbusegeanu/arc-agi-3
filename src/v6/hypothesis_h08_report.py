from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

from v6.higher_order_substrate import derive_higher_order_memory


_REJECTED_STATES = {"failed", "invalid", "demoted", "rejected"}


def _evidence_diagnostics(
    memory_dir: Path,
    run_dir: Path | None,
    *,
    missing_target: str,
) -> dict[str, Any]:
    current_state = Path(memory_dir) / "current_state.sqlite"
    raw_db_exists = bool(
        run_dir is not None and any(Path(run_dir).rglob("*.sqlite"))
    )
    return {
        "expected_current_state_path": str(current_state),
        "compact_memory_exists": bool(current_state.exists()),
        "raw_db_evidence_exists": bool(raw_db_exists),
        "direct_streamed_manifest_exists": bool(
            (
                Path(memory_dir)
                / "direct_streaming_fold_manifest.sqlite"
            ).exists()
        ),
        "missing_target": str(missing_target),
    }


def _effective_concept_is_promoted(row: dict[str, Any]) -> bool:
    persistent_promoted = row.get("persistent_currently_promoted")
    if persistent_promoted is not None:
        promoted = int(persistent_promoted) == 1
    else:
        promoted = int(row.get("candidate_is_promoted") or 0) == 1

    promotion_status = str(
        row.get("persistent_promotion_status") or ""
    ).strip().lower()
    validation_status = str(
        row.get("persistent_validation_status") or ""
    ).strip().lower()

    if promotion_status in _REJECTED_STATES:
        return False
    if validation_status in _REJECTED_STATES:
        return False
    return promoted


def _has_positive_heldout_gain(row: dict[str, Any]) -> bool:
    return any(
        value is not None and float(value) > 0.0
        for value in (
            row.get("heldout_prediction_gain"),
            row.get("validation_action_selection_lift"),
            row.get("validation_transfer_lift"),
            row.get("validation_contradiction_resolution"),
            row.get("validation_explanatory_gain"),
        )
    )


def _component_passes_h08_validity(
    record: dict[str, Any],
) -> bool:
    validation_status = str(
        record.get("effective_validation_status") or ""
    ).strip().lower()

    if not bool(record.get("effective_currently_coherent")):
        return False
    if validation_status in _REJECTED_STATES:
        return False
    if not bool(record.get("has_positive_heldout_gain")):
        return False
    if (
        int(record.get("cross_context_count") or 0) < 3
        and int(record.get("cross_game_count") or 0) < 2
    ):
        return False
    if int(record.get("role_link_count") or 0) < 1:
        return False
    if int(record.get("family_link_count") or 0) < 2:
        return False
    if int(record.get("supported_context_count") or 0) < 2:
        return False
    if (
        int(record.get("verified_predicted_outcome_count") or 0)
        < 1
    ):
        return False
    if float(record.get("coherence_score") or 0.0) < 0.45:
        return False
    if float(record.get("explanatory_coverage") or 0.0) <= 0.0:
        return False
    if bool(record.get("candidate_only")):
        return False
    return True


def evaluate_h08_world_model_coherence(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"

    if not already_derived and current_state.exists():
        derive_higher_order_memory(
            memory_dir=memory_dir,
            run_dir=run_dir,
        )

    if not current_state.exists():
        result = _base_result(
            "INSUFFICIENT_EVIDENCE",
            [
                "Missing expected compact-memory file: "
                f"{current_state}"
            ],
        )
        result["evidence_diagnostics"] = _evidence_diagnostics(
            memory_dir,
            run_dir,
            missing_target="current_state.sqlite",
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
            "concept_candidates",
            "world_model_components",
            "world_model_links",
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
            result["evidence_diagnostics"] = _evidence_diagnostics(
                memory_dir,
                run_dir,
                missing_target=",".join(missing_tables),
            )
            result["evidence_diagnostics"]["tables_seen"] = sorted(
                tables
            )
            _write_outputs(output_dir, result)
            return result

        concept_candidate_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM concept_candidates"
            ).fetchone()[0]
        )

        if "concept_promotion_state" in tables:
            concept_rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        candidate.concept_signature,
                        candidate.is_promoted
                            AS candidate_is_promoted,
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
            ]
        else:
            concept_rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        concept_signature,
                        is_promoted AS candidate_is_promoted,
                        NULL AS persistent_currently_promoted,
                        NULL AS persistent_promotion_status,
                        NULL AS persistent_validation_status
                    FROM concept_candidates
                    ORDER BY concept_signature ASC
                    """
                ).fetchall()
            ]

        promoted_concept_count = sum(
            1
            for row in concept_rows
            if _effective_concept_is_promoted(row)
        )

        role_candidate_count = (
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM role_candidates"
                ).fetchone()[0]
            )
            if "role_candidates" in tables
            else 0
        )

        role_transfer_success_count = (
            int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM role_transfer_attempts
                    WHERE COALESCE(reuse_success, 0) = 1
                    """
                ).fetchone()[0]
            )
            if "role_transfer_attempts" in tables
            else 0
        )

        component_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    component_signature,
                    coherence_score,
                    explanatory_coverage,
                    cross_context_count,
                    cross_game_count,
                    linked_concept_count,
                    linked_role_count,
                    linked_family_count,
                    prediction_support_count,
                    contradiction_coverage_count,
                    predicted_outcome_count,
                    predicted_outcome_count_is_proxy,
                    observed_outcome_count,
                    correct_prediction_count,
                    prediction_error_count,
                    prediction_evidence_status,
                    baseline_prediction_score,
                    component_prediction_score,
                    heldout_prediction_gain,
                    matched_prediction_event_count,
                    unmatched_prediction_event_count,
                    structural_coherence_score,
                    functional_coherence_score,
                    combined_coherence_score,
                    candidate_family_link_count,
                    retained_family_link_count,
                    dropped_family_link_count,
                    family_links_dropped_low_support,
                    family_links_dropped_limit,
                    is_coherent,
                    candidate_only,
                    validation_prediction_lift,
                    validation_action_selection_lift,
                    validation_transfer_lift,
                    validation_contradiction_resolution,
                    validation_explanatory_gain
                FROM world_model_components
                ORDER BY component_signature ASC
                """
            ).fetchall()
        ]

        component_links = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    component_signature,
                    linked_type,
                    linked_key
                FROM world_model_links
                ORDER BY
                    component_signature ASC,
                    linked_type ASC,
                    linked_key ASC
                """
            ).fetchall()
        ]

        component_state_rows = (
            [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        component_signature,
                        historically_coherent,
                        currently_coherent,
                        validation_status
                    FROM world_model_component_state
                    """
                ).fetchall()
            ]
            if "world_model_component_state" in tables
            else []
        )

        milestone_map = dict(
            conn.execute(
                """
                SELECT milestone_name, first_global_step
                FROM higher_order_milestones
                """
            ).fetchall()
        )

    link_map: dict[str, dict[str, set[str]]] = {}
    for row in component_links:
        component_signature = str(row["component_signature"])
        linked_type = str(row["linked_type"])
        linked_key = str(row["linked_key"])
        groups = link_map.setdefault(component_signature, {})
        groups.setdefault(linked_type, set()).add(linked_key)

    component_state = {
        str(row["component_signature"]): row
        for row in component_state_rows
    }

    component_records: list[dict[str, Any]] = []
    for row in component_rows:
        signature = str(row["component_signature"])
        persistent_state = component_state.get(signature)

        if (
            persistent_state is not None
            and persistent_state.get("currently_coherent") is not None
        ):
            effective_currently_coherent = (
                int(persistent_state["currently_coherent"]) == 1
            )
        else:
            effective_currently_coherent = (
                int(row.get("is_coherent") or 0) == 1
            )

        effective_validation_status = (
            str(
                persistent_state.get("validation_status") or ""
            ).strip().lower()
            if persistent_state is not None
            else ""
        )

        links = link_map.get(signature, {})
        verified_predicted_outcome_count = (
            int(row.get("predicted_outcome_count") or 0)
            if str(
                row.get("prediction_evidence_status") or "missing"
            ) == "verified"
            else 0
        )

        record = {
            "component_signature": signature,
            "effective_currently_coherent":
                effective_currently_coherent,
            "effective_validation_status":
                effective_validation_status,
            "has_positive_heldout_gain":
                _has_positive_heldout_gain(row),
            "cross_context_count":
                int(row.get("cross_context_count") or 0),
            "cross_game_count":
                int(row.get("cross_game_count") or 0),
            "supported_context_count":
                len(links.get("context", set())),
            "concept_link_count":
                len(links.get("concept", set())),
            "role_link_count":
                len(links.get("role", set())),
            "family_link_count":
                int(row.get("linked_family_count") or 0),
            "verified_predicted_outcome_count":
                verified_predicted_outcome_count,
            "coherence_score":
                float(row.get("coherence_score") or 0.0),
            "explanatory_coverage":
                float(row.get("explanatory_coverage") or 0.0),
            "candidate_only":
                int(row.get("candidate_only") or 0) == 1,
            "heldout_prediction_gain":
                row.get("heldout_prediction_gain"),
            "validation_action_selection_lift":
                row.get("validation_action_selection_lift"),
            "validation_transfer_lift":
                row.get("validation_transfer_lift"),
            "validation_contradiction_resolution":
                row.get("validation_contradiction_resolution"),
            "validation_explanatory_gain":
                row.get("validation_explanatory_gain"),
        }
        component_records.append(record)

    qualifying_component_records = [
        record
        for record in component_records
        if _component_passes_h08_validity(record)
    ]
    qualifying_component_count = len(
        qualifying_component_records
    )
    qualifying_component_signatures = [
        str(record["component_signature"])
        for record in qualifying_component_records
    ]

    world_model_component_count = len(component_rows)
    coherent_world_model_component_count = sum(
        1
        for record in component_records
        if bool(record["effective_currently_coherent"])
        and bool(record["has_positive_heldout_gain"])
        and str(
            record.get("effective_validation_status") or ""
        ) not in _REJECTED_STATES
    )
    structural_coherent_world_model_component_count = sum(
        1
        for row in component_rows
        if int(row.get("is_coherent") or 0) == 1
    )
    candidate_only_world_model_component_count = sum(
        1
        for row in component_rows
        if int(row.get("candidate_only") or 0) == 1
    )

    coherent_cross_context_component_count = sum(
        1
        for record in component_records
        if bool(record["effective_currently_coherent"])
        and bool(record["has_positive_heldout_gain"])
        and str(
            record.get("effective_validation_status") or ""
        ) not in _REJECTED_STATES
        and int(record["cross_context_count"]) >= 1
    )
    coherent_cross_game_component_count = sum(
        1
        for record in component_records
        if bool(record["effective_currently_coherent"])
        and bool(record["has_positive_heldout_gain"])
        and str(
            record.get("effective_validation_status") or ""
        ) not in _REJECTED_STATES
        and int(record["cross_game_count"]) >= 1
    )

    mean_coherence_score = (
        sum(
            float(row.get("coherence_score") or 0.0)
            for row in component_rows
        )
        / world_model_component_count
        if component_rows
        else None
    )
    max_coherence_score = max(
        (
            float(row.get("coherence_score") or 0.0)
            for row in component_rows
        ),
        default=None,
    )
    mean_explanatory_coverage = (
        sum(
            float(row.get("explanatory_coverage") or 0.0)
            for row in component_rows
        )
        / world_model_component_count
        if component_rows
        else None
    )
    max_explanatory_coverage = max(
        (
            float(row.get("explanatory_coverage") or 0.0)
            for row in component_rows
        ),
        default=None,
    )

    component_cross_context_count = max(
        (
            int(row.get("cross_context_count") or 0)
            for row in component_rows
        ),
        default=0,
    )
    component_cross_game_count = max(
        (
            int(row.get("cross_game_count") or 0)
            for row in component_rows
        ),
        default=0,
    )

    predicted_outcome_count = sum(
        int(row.get("predicted_outcome_count") or 0)
        for row in component_rows
    )
    verified_predicted_outcome_count = sum(
        int(row.get("predicted_outcome_count") or 0)
        for row in component_rows
        if str(
            row.get("prediction_evidence_status") or "missing"
        ) == "verified"
    )
    proxy_predicted_outcome_count = sum(
        int(row.get("predicted_outcome_count") or 0)
        for row in component_rows
        if str(
            row.get("prediction_evidence_status") or "missing"
        ) == "proxy"
    )
    missing_outcome_count = sum(
        int(row.get("unmatched_prediction_event_count") or 0)
        for row in component_rows
    )
    predicted_outcome_count_is_proxy_count = sum(
        int(row.get("predicted_outcome_count_is_proxy") or 0)
        for row in component_rows
    )

    supported_context_count = max(
        (
            int(record["supported_context_count"])
            for record in component_records
        ),
        default=0,
    )
    concept_link_count = max(
        (
            int(record["concept_link_count"])
            for record in component_records
        ),
        default=0,
    )
    role_link_count = max(
        (
            int(record["role_link_count"])
            for record in component_records
        ),
        default=0,
    )
    family_link_count = max(
        (
            int(record["family_link_count"])
            for record in component_records
        ),
        default=0,
    )

    contradiction_coverage_count = sum(
        int(row.get("contradiction_coverage_count") or 0)
        for row in component_rows
    )

    component_statistics = {
        "cross_context_count": _count_statistics(
            [
                int(record["cross_context_count"])
                for record in component_records
            ]
        ),
        "cross_game_count": _count_statistics(
            [
                int(record["cross_game_count"])
                for record in component_records
            ]
        ),
        "predicted_outcome_count": _count_statistics(
            [
                int(row.get("predicted_outcome_count") or 0)
                for row in component_rows
            ]
        ),
        "supported_context_count": _count_statistics(
            [
                int(record["supported_context_count"])
                for record in component_records
            ]
        ),
        "concept_link_count": _count_statistics(
            [
                int(record["concept_link_count"])
                for record in component_records
            ]
        ),
        "role_link_count": _count_statistics(
            [
                int(record["role_link_count"])
                for record in component_records
            ]
        ),
        "family_link_count": _count_statistics(
            [
                int(record["family_link_count"])
                for record in component_records
            ]
        ),
    }

    heldout_components = [
        {
            "component_signature":
                str(record["component_signature"]),
            "heldout_prediction_gain":
                record["heldout_prediction_gain"],
            "heldout_behavior_gain":
                record["validation_action_selection_lift"],
            "heldout_contradiction_resolution":
                record["validation_contradiction_resolution"],
            "heldout_explanatory_gain":
                record["validation_explanatory_gain"],
            "heldout_transfer_lift":
                record["validation_transfer_lift"],
            "heldout_validation_pass":
                bool(record["has_positive_heldout_gain"]),
        }
        for record in component_records
        if bool(record["effective_currently_coherent"])
    ]

    metrics = {
        "world_model_component_count":
            world_model_component_count,
        "coherent_world_model_component_count":
            coherent_world_model_component_count,
        "structural_coherent_world_model_component_count":
            structural_coherent_world_model_component_count,
        "candidate_only_world_model_component_count":
            candidate_only_world_model_component_count,
        "qualifying_component_count":
            qualifying_component_count,
        "qualifying_component_signatures":
            qualifying_component_signatures,
        "qualifying_component_records":
            qualifying_component_records[:200],
        "historically_coherent_component_count": sum(
            int(row.get("historically_coherent") or 0)
            for row in component_state.values()
        ),
        "currently_coherent_component_count": sum(
            1
            for record in component_records
            if bool(record["effective_currently_coherent"])
        ),
        "promoted_concept_count": promoted_concept_count,
        "role_candidate_count": role_candidate_count,
        "role_transfer_success_count":
            role_transfer_success_count,
        "mean_coherence_score": mean_coherence_score,
        "max_coherence_score": max_coherence_score,
        "mean_explanatory_coverage":
            mean_explanatory_coverage,
        "max_explanatory_coverage":
            max_explanatory_coverage,
        "coherent_cross_context_component_count":
            coherent_cross_context_component_count,
        "coherent_cross_game_component_count":
            coherent_cross_game_component_count,
        "component_cross_context_count":
            component_cross_context_count,
        "component_cross_game_count":
            component_cross_game_count,
        "predicted_outcome_count": predicted_outcome_count,
        "verified_predicted_outcome_count":
            verified_predicted_outcome_count,
        "proxy_predicted_outcome_count":
            proxy_predicted_outcome_count,
        "missing_outcome_count": missing_outcome_count,
        "matched_prediction_event_count": sum(
            int(row.get("matched_prediction_event_count") or 0)
            for row in component_rows
        ),
        "unmatched_prediction_event_count": sum(
            int(row.get("unmatched_prediction_event_count") or 0)
            for row in component_rows
        ),
        "predicted_outcome_count_is_proxy_count":
            predicted_outcome_count_is_proxy_count,
        "structural_coherence_score": _numeric_statistics(
            [
                float(row.get("structural_coherence_score") or 0.0)
                for row in component_rows
            ]
        ),
        "functional_coherence_score": _numeric_statistics(
            [
                float(row.get("functional_coherence_score") or 0.0)
                for row in component_rows
            ]
        ),
        "combined_coherence_score": _numeric_statistics(
            [
                float(row.get("combined_coherence_score") or 0.0)
                for row in component_rows
            ]
        ),
        "candidate_family_link_count": sum(
            int(row.get("candidate_family_link_count") or 0)
            for row in component_rows
        ),
        "retained_family_link_count": sum(
            int(row.get("retained_family_link_count") or 0)
            for row in component_rows
        ),
        "dropped_family_link_count": sum(
            int(row.get("dropped_family_link_count") or 0)
            for row in component_rows
        ),
        "family_links_dropped_low_support": sum(
            int(row.get("family_links_dropped_low_support") or 0)
            for row in component_rows
        ),
        "family_links_dropped_limit": sum(
            int(row.get("family_links_dropped_limit") or 0)
            for row in component_rows
        ),
        "family_link_verification_ratio": (
            sum(
                int(row.get("retained_family_link_count") or 0)
                for row in component_rows
            )
            / max(
                1,
                sum(
                    int(
                        row.get("candidate_family_link_count")
                        or 0
                    )
                    for row in component_rows
                ),
            )
        ),
        "supported_context_count": supported_context_count,
        "concept_link_count": concept_link_count,
        "role_link_count": role_link_count,
        "family_link_count": family_link_count,
        "proxy_world_model_component_count":
            candidate_only_world_model_component_count,
        "candidate_proxy_only": bool(
            promoted_concept_count == 0
            and coherent_world_model_component_count == 0
        ),
        "overlinked_world_model_component_count": sum(
            1
            for row in component_rows
            if int(row.get("linked_family_count") or 0) > 50
        ),
        "max_family_link_count": max(
            (
                int(row.get("linked_family_count") or 0)
                for row in component_rows
            ),
            default=0,
        ),
        "family_link_count_is_proxy": bool(
            predicted_outcome_count_is_proxy_count > 0
        ),
        "contradiction_coverage_count":
            contradiction_coverage_count,
        "component_aggregate_statistics":
            component_statistics,
        "coherent_component_validation":
            heldout_components,
        "first_world_model_component_step":
            milestone_map.get("first_world_model_component_step"),
        "first_coherent_world_model_step":
            milestone_map.get("first_coherent_world_model_step"),
        "first_promoted_concept_step":
            milestone_map.get("first_promoted_concept_step"),
        "evidence_stage": None,
    }

    h08_validity_gates = {
        "promoted_concepts": {
            "required": 1,
            "actual": promoted_concept_count,
            "passed": promoted_concept_count >= 1,
        },
        "role_candidates": {
            "required": 1,
            "actual": role_candidate_count,
            "passed": role_candidate_count >= 1,
        },
        "role_transfer_successes": {
            "required": 1,
            "actual": role_transfer_success_count,
            "passed": role_transfer_success_count >= 1,
        },
        "qualifying_components": {
            "required": 1,
            "actual": qualifying_component_count,
            "passed": qualifying_component_count >= 1,
        },
    }

    if world_model_component_count <= 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["no world-model components available"]
    elif (
        promoted_concept_count == 0
        and qualifying_component_count == 0
    ):
        decision = "INSUFFICIENT_EVIDENCE"
        metrics["evidence_stage"] = "candidate_proxy_only"
        missing = [
            "No promoted concepts or qualifying "
            "world-model components available."
        ]
    elif (
        world_model_component_count > 0
        and qualifying_component_count == 0
    ):
        decision = "PARTIALLY_VALID"
        missing = [
            "No single world-model component satisfies all "
            "H08 coherence gates."
        ]
    elif (
        promoted_concept_count >= 1
        and role_candidate_count >= 1
        and role_transfer_success_count >= 1
        and qualifying_component_count >= 1
    ):
        decision = "VALID"
        missing = []
    elif (
        promoted_concept_count > 0
        and world_model_component_count == 0
    ):
        decision = "INVALID"
        missing = []
    elif (
        promoted_concept_count > 0
        and component_rows
        and max(
            (
                float(row.get("coherence_score") or 0.0)
                for row in component_rows
            ),
            default=0.0,
        )
        < 0.20
    ):
        decision = "INVALID"
        missing = []
    else:
        decision = "PARTIALLY_VALID"
        missing = [
            (
                f"H08 gate failed: {name} "
                f"(required={gate['required']}, "
                f"actual={gate['actual']})."
            )
            for name, gate in h08_validity_gates.items()
            if not gate["passed"]
        ]

    result = _base_result(decision, missing)
    result.update(metrics)
    result["h08_validity_gates"] = h08_validity_gates
    result["core_metrics"] = dict(metrics)
    result["evidence_diagnostics"] = _evidence_diagnostics(
        memory_dir,
        run_dir,
        missing_target="none",
    )
    _write_outputs(output_dir, result)
    return result


def _base_result(
    decision: str,
    missing_evidence: list[str],
) -> dict[str, Any]:
    return {
        "hypothesis_id": "H08",
        "hypothesis_name":
            "World-model coherence from promoted concepts",
        "decision": decision,
        "missing_evidence": list(missing_evidence),
        "evidence_source": "compact_memory",
    }


def _count_statistics(values: list[int]) -> dict[str, Any]:
    cooked = [int(value) for value in values]
    distribution: dict[str, int] = {}
    for value in sorted(cooked):
        key = str(value)
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "total": sum(cooked),
        "distinct": len(set(cooked)),
        "mean": (sum(cooked) / len(cooked)) if cooked else 0.0,
        "median": float(median(cooked)) if cooked else 0.0,
        "maximum": max(cooked, default=0),
        "distribution": distribution,
    }


def _numeric_statistics(
    values: list[float],
) -> dict[str, Any]:
    cooked = [float(value) for value in values]
    distribution: dict[str, int] = {}
    for value in sorted(cooked):
        key = f"{value:.6f}"
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "total": sum(cooked),
        "distinct": len(set(cooked)),
        "mean": (sum(cooked) / len(cooked)) if cooked else 0.0,
        "median": float(median(cooked)) if cooked else 0.0,
        "maximum": max(cooked, default=0.0),
        "distribution": distribution,
    }


def _write_outputs(
    output_dir: Path,
    result: dict[str, Any],
) -> None:
    (
        output_dir / "h08_world_model_coherence_report.json"
    ).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    text = (
        "H08 world-model coherence decision: "
        f"{result.get('decision')}\n"
        f"hypothesis name: {result.get('hypothesis_name')}\n"
        "promoted concepts: "
        f"{result.get('promoted_concept_count')}\n"
        "world model components: "
        f"{result.get('world_model_component_count')}\n"
        "candidate-only components: "
        f"{result.get('candidate_only_world_model_component_count')}\n"
        "coherent world model components: "
        f"{result.get('coherent_world_model_component_count')}\n"
        "qualifying world model components: "
        f"{result.get('qualifying_component_count')}\n"
        f"max coherence score: {result.get('max_coherence_score')}\n"
        "max explanatory coverage: "
        f"{result.get('max_explanatory_coverage')}\n"
    )
    (
        output_dir / "h08_world_model_coherence_report.txt"
    ).write_text(text, encoding="utf-8")
    (
        output_dir / "h08_world_model_coherence.md"
    ).write_text(
        "```\n" + text + "```\n",
        encoding="utf-8",
    )
