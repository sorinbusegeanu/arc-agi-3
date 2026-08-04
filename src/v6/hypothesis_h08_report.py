"""H08 hypothesis report — world-model coherence from promoted concepts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _evidence_diagnostics(
    memory_dir: Path, run_dir: Path | None, *, missing_target: str
) -> dict[str, Any]:
    current_state = Path(memory_dir) / "current_state.sqlite"
    raw_db_exists = bool(run_dir is not None and any(Path(run_dir).rglob("*.sqlite")))
    return {
        "expected_current_state_path": str(current_state),
        "compact_memory_exists": bool(current_state.exists()),
        "raw_db_evidence_exists": bool(raw_db_exists),
        "direct_streamed_manifest_exists": bool(
            (Path(memory_dir) / "direct_streaming_fold_manifest.sqlite").exists()
        ),
        "missing_target": str(missing_target),
    }


def evaluate_h08_world_model_coherence(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, Any]:
    from v6.higher_order_substrate import derive_higher_order_memory

    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not already_derived and current_state.exists():
        derive_higher_order_memory(memory_dir=memory_dir, run_dir=run_dir)

    if not current_state.exists():
        result = _base_result(
            "INSUFFICIENT_EVIDENCE", [f"Missing expected compact-memory file: {current_state}"]
        )
        result["evidence_diagnostics"] = _evidence_diagnostics(memory_dir, run_dir, missing_target="current_state.sqlite")
        _write_outputs(output_dir, result)
        return result

    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        missing_tables = [name for name in ("concept_candidates", "world_model_components", "world_model_links", "higher_order_milestones") if name not in tables]

        # 2.1 — Per-component validation records: every component gets a record with all required evidence fields.
        component_rows_all = conn.execute(
            """SELECT component_signature, is_coherent AS structural_is_coherent,
                       currently_coherent AS effective_currently_coherent,
                       validation_prediction_lift, validation_action_selection_lift,
                       validation_contradiction_resolution, validation_explanatory_gain,
                       heldout_prediction_gain, matched_prediction_event_count,
                       predicted_outcome_count, prediction_evidence_status,
                       explanatory_coverage, coherence_score, candidate_only
               FROM world_model_components
               ORDER BY component_signature ASC"""
        ).fetchall()

        # 2.5 — Durable concept-promotion filter (same as H07).
        promoted_concept_count = int(
            conn.execute(
                """SELECT COUNT(*) FROM concept_candidates AS candidate
                   LEFT JOIN concept_promotion_state AS persistent
                     ON persistent.concept_signature = candidate.concept_signature
                   WHERE COALESCE(persistent.currently_promoted, candidate.is_promoted, 0) = 1"""
            ).fetchone()[0]
        )

        role_candidate_count = int(conn.execute("SELECT COUNT(*) FROM role_candidates").fetchone()[0]) if "role_candidates" in tables else 0

        role_transfer_success_count = (
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1"
                ).fetchone()[0]
            )
            if "role_transfer_attempts" in tables
            else 0
        )

        # Build per-component validation records with all required H08 evidence fields.
        component_validation_records: dict[str, dict[str, Any]] = {}
        for row in component_rows_all:
            sig = str(row["component_signature"])
            structural_coherent = int(row["structural_is_coherent"] or 0) == 1
            effective_currently_coherent = (
                int(row["effective_currently_coherent"] or 0) == 1 if row["effective_currently_coherent"] is not None else False
            )
            validation_lift = float(row.get("validation_prediction_lift") or 0.0)
            heldout_gain = float(row.get("heldout_prediction_gain") or 0.0)
            prediction_evidence_status = str(row.get("prediction_evidence_status") or "missing")

            # Use durable concept-promotion state when available, querying directly from concept_promotion_state.
            concepts_rows = conn.execute(
                """SELECT candidate.concept_signature,
                           COALESCE(persistent.currently_promoted, candidate.is_promoted, 0) AS effective_is_promoted,
                           persistent.promotion_status AS persistent_promotion_status,
                           persistent.validation_status AS persistent_validation_status
                    FROM concept_candidates AS candidate
                    LEFT JOIN concept_promotion_state AS persistent
                      ON persistent.concept_signature = candidate.concept_signature"""
            ).fetchall()

            record: dict[str, Any] = {
                "component_signature": sig,
                "effective_currently_coherent": effective_currently_coherent,
                "validation_prediction_lift": validation_lift,
                "heldout_prediction_gain": heldout_gain,
                "cross_context_count": int(row.get("cross_context_count") or 0),
                "cross_game_count": int(row.get("cross_game_count") or 0),
                "role_link_count": int(row.get("linked_role_count") or 0),
                "family_link_count": int(row.get("linked_family_count") or 0),
                "supported_context_count": int(row.get("supported_context_count") or 0),
                "verified_predicted_outcome_count": int(
                    row["matched_prediction_event_count"] if prediction_evidence_status == "verified" else 0
                ),
                "concept_link_count": 0,
                "candidate_only": bool(int(row.get("candidate_only") or 0)),
            }

            # Filter promoted concepts by durable validation state (same logic as H07).
            for cr in concepts_rows:
                if int(cr["effective_is_promoted"] or 0) == 1 and str(cr.get("persistent_validation_status") or "") not in {"failed", "invalid", "demoted", "rejected"}:
                    record["concept_link_count"] += 1

            component_validation_records[sig] = record

        # 2.2 — Single qualifying-component predicate: a component qualifies only when it satisfies ALL gates.
        def _component_passes_h08_validity(record: dict[str, Any]) -> bool:
            if not record["effective_currently_coherent"]:
                return False
            if record["heldout_prediction_gain"] <= 0.0 or record["validation_prediction_lift"] <= 0.0:
                return False
            if record["cross_context_count"] < 3 and record["cross_game_count"] < 2:
                return False
            if record["role_link_count"] < 1:
                return False
            if record["family_link_count"] < 2:
                return False
            if record["supported_context_count"] < 2:
                return False
            if record["verified_predicted_outcome_count"] < 1:
                return False
            concept_link_count = int(record.get("concept_link_count", 0))
            if concept_link_count < 1:
                return False
            if record["candidate_only"]:
                return False
            coherence_score = float(record.get("coherence_score") or 0.0)
            explanatory_coverage = float(record.get("explanatory_coverage") or 0.0)
            return coherence_score >= 0.45 and explanatory_coverage > 0.0

        qualifying_component_count = sum(1 for sig, rec in component_validation_records.items() if _component_passes_h08_validity(rec))
        qualifying_component_signatures: list[str] = [sig for sig, rec in sorted(component_validation_records.items()) if _component_passes_h08_validity(rec)]

        # 2.4 — Correct coherent cross-scope counts from components that are effectively currently coherent AND have positive held-out gain.
        structural_coherent_rows = component_rows_all
        coherent_cross_context_component_count = sum(
            int(row["cross_context_count"] or 0) for row in structural_coherent_rows if int(row["structural_is_coherent"] or 0) == 1 and float(row.get("heldout_prediction_gain") or 0.0) > 0.0
        )
        coherent_cross_game_component_count = sum(
            int(row["cross_game_count"] or 0) for row in structural_coherent_rows if int(row["structural_is_coherent"] or 0) == 1 and float(row.get("heldout_prediction_gain") or 0.0) > 0.0
        )

        # Diagnostic counters: structural-only components (coherent but no held-out gain).
        structural_cross_context_component_count = sum(
            int(row["cross_context_count"] or 0) for row in component_rows_all if int(row["structural_is_coherent"] or 0) == 1 and float(row.get("heldout_prediction_gain") or 0.0) <= 0.0
        )
        structural_cross_game_component_count = sum(
            int(row["cross_game_count"] or 0) for row in component_rows_all if int(row["structural_is_coherent"] or 0) == 1 and float(row.get("heldout_prediction_gain") or 0.0) <= 0.0
        )

    # Promoted concepts filtered by durable validation state (section 2.5 — same as H07).
    # promoted_concept_count was already computed above.

    milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall()) if "higher_order_milestones" in tables else {}

    metrics: dict[str, Any] = {
        "world_model_component_count": len(component_rows_all),
        "qualifying_component_count": qualifying_component_count,
        "qualifying_component_signatures": qualifying_component_signatures,
        "structural_coherent_world_model_component_count": sum(int(row["structural_is_coherent"] or 0) for row in component_rows_all),
        "currently_coherent_component_count": sum(int(row.get("effective_currently_coherent") or 0) for row in component_rows_all),
        "promoted_concept_count": promoted_concept_count,
        "role_candidate_count": role_candidate_count,
        "role_transfer_success_count": role_transfer_success_count,
        "coherent_cross_context_component_count": coherent_cross_context_component_count,
        "coherent_cross_game_component_count": coherent_cross_game_component_count,
        "structural_cross_context_component_count": structural_cross_context_component_count,
        "structural_cross_game_component_count": structural_cross_game_component_count,
        "mean_coherence_score": _numeric_statistics([float(row.get("coherence_score") or 0.0) for row in component_rows_all]),
        "max_coherence_score": max((float(row["coherence_score"] or 0.0) for row in component_rows_all), default=None),
        "mean_explanatory_coverage": _numeric_statistics([float(row.get("explanatory_coverage") or 0.0) for row in component_rows_all]),
        "max_explanatory_coverage": max((float(row["explanatory_coverage"] or 0.0) for row in component_rows_all), default=None),
        "predicted_outcome_count": sum(int(row.get("predicted_outcome_count") or 0) for row in component_rows_all),
        "verified_predicted_outcome_count": sum(
            int(row["matched_prediction_event_count"] or 0) for row in component_rows_all if str(row.get("prediction_evidence_status") or "missing") == "verified"
        ),
        "proxy_predicted_outcome_count": sum(int(row.get("predicted_outcome_count_is_proxy") or 0) for row in component_rows_all),
        "missing_outcome_count": sum(int(row.get("unmatched_prediction_event_count") or 0) for row in component_rows_all),
        "supported_context_count": _count_statistics([len(component_validation_records.get(str(row["component_signature"]), {}).get("context", set())) for row in component_rows_all]),
        "concept_link_count": _count_statistics([len(component_validation_records.get(str(row["component_signature"]), {}).get("concept", set())) for row in component_rows_all]),
        "role_link_count": _count_statistics([len(component_validation_records.get(str(row["component_signature"]), {}).get("role", set())) for row in component_rows_all]),
        "family_link_count": _count_statistics([int(row["linked_family_count"] or 0) for row in component_rows_all]),
    }

    h08_validity_gates: dict[str, Any] = {
        "promoted_concepts": {"required": 1, "actual": promoted_concept_count, "passed": bool(promoted_concept_count)},
        "qualifying_components": {"required": 1, "actual": qualifying_component_count, "passed": bool(qualifying_component_count >= 1)},
        "heldout_positive_gain": {
            "required": "> 0 in one held-out metric",
            "actual": sum(1 for sig, rec in component_validation_records.items() if _component_passes_h08_validity(rec) and rec["heldout_prediction_gain"] > 0.0),
            "passed": bool(component_validation_records and any(_component_passes_h08_validity(rec) for rec in component_validation_records.values())),
        },
        "cross_scope": {
            "required": "cross_context >= 3 OR cross_game >= 2",
            "actual": {"cross_context": coherent_cross_context_component_count, "cross_game": coherent_cross_game_component_count},
            "passed": bool(coherent_cross_context_component_count >= 3 or coherent_cross_game_component_count >= 2),
        },
        "prediction_evidence": {
            "required": 1,
            "actual": verified_predicted_outcome_count,
            "passed": bool(verified_predicted_outcome_count),
        },
    }

    if len(component_rows_all) <= 0:
        decision = "INSUFFICIENT_EVIDENCE"
        missing = ["no world-model components available"]
    elif promoted_concept_count == 0 and qualifying_component_count == 0:
        decision = "INSUFFICIENT_EVIDENCE"
        metrics["evidence_stage"] = "candidate_proxy_only"
        missing = ["No promoted concepts or qualifying world-model components available."]
    elif len(component_rows_all) > 0 and qualifying_component_count == 0:
        decision = "PARTIALLY_VALID"
        missing = ["Structural world-model components lack positive held-out predictive, behavioral, contradiction, or explanatory gain."]
    elif (
        promoted_concept_count >= 1
        and qualifying_component_count >= 1
        and role_candidate_count >= 1
        and role_transfer_success_count >= 1
        and len(component_rows_all) >= 1
        and metrics["max_coherence_score"] is not None and metrics["max_coherence_score"] >= 0.45
        and metrics["max_explanatory_coverage"] is not None and metrics["max_explanatory_coverage"] > 0.0
        and (coherent_cross_context_component_count >= 3 or coherent_cross_game_component_count >= 2)
        and metrics.get("role_link_count", {}).get("total", 0) >= 1
        and metrics.get("family_link_count", {}).get("total", 0) >= 2
        and bool(verified_predicted_outcome_count)
    ):
        decision = "VALID"
        missing = []
    elif promoted_concept_count > 0 and len(component_rows_all) == 0:
        decision = "INVALID"
        missing = []
    else:
        decision = "PARTIALLY_VALID"
        missing = []

    result = _base_result(decision, missing)
    result.update(metrics)
    result["h08_validity_gates"] = h08_validity_gates
    result["evidence_diagnostics"] = _evidence_diagnostics(memory_dir, run_dir, missing_target="none")
    _write_outputs(output_dir, result)
    return result


# ---------------------------------------------------------------------------
# Helpers — kept identical to the original.
# ---------------------------------------------------------------------------



def _base_result(decision: str, missing_evidence: list[str]) -> dict[str, Any]:
    return {
        "hypothesis_id": "H08",
        "hypothesis_name": "World-model coherence from promoted concepts",
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
        "median": float(__import__("statistics").median(cooked)) if cooked else 0.0,
        "maximum": max(cooked, default=0),
        "distribution": distribution,
    }


def _numeric_statistics(values: list[float]) -> dict[str, Any]:
    cooked = [float(value) for value in values]
    distribution: dict[str, int] = {}
    for value in sorted(cooked):
        key = f"{value:.6f}"
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "total": sum(cooked),
        "distinct": len(set(cooked)),
        "mean": (sum(cooked) / len(cooked)) if cooked else 0.0,
        "median": float(__import__("statistics").median(cooked)) if cooked else 0.0,
        "maximum": max(cooked, default=0.0),
        "distribution": distribution,
    }


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h08_world_model_coherence_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H08 world-model coherence decision: {result.get('decision')}\n"
        f"hypothesis name: {result.get('hypothesis_name')}\n"
        f"promoted concepts: {result.get('promoted_concept_count')}\n"
        f"world model components: {result.get('world_model_component_count')}\n"
        f"candidate-only components: {result.get('structural_coherent_world_model_component_count')}\n"
        f"qualifying world-model components: {result.get('qualifying_component_count')}\n"
        f"max coherence score: {result.get('max_coherence_score')}\n"
        f"max explanatory coverage: {result.get('max_explanatory_coverage')}"
    )
    (output_dir / "h08_world_model_coherence_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h08_world_model_coherence.md").write_text("```\n" + text + "```\n", encoding="utf-8")
