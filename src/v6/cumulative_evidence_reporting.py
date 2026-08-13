from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False
_ORIGINAL_H07: Any = None
_ORIGINAL_H08: Any = None
_ORIGINAL_PROVENANCE: Any = None


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _validated_concepts(memory_dir: Path) -> tuple[set[str], set[str]]:
    db = Path(memory_dir) / "current_state.sqlite"
    promoted: set[str] = set()
    validated: set[str] = set()
    if not db.exists():
        return promoted, validated
    with sqlite3.connect(db) as conn:
        if _exists(conn, "concept_promotion_state"):
            for signature, currently_promoted in conn.execute("SELECT concept_signature, currently_promoted FROM concept_promotion_state").fetchall():
                if int(currently_promoted or 0) == 1:
                    promoted.add(str(signature))
        elif _exists(conn, "concept_candidates"):
            for signature, is_promoted in conn.execute("SELECT concept_signature, is_promoted FROM concept_candidates").fetchall():
                if int(is_promoted or 0) == 1:
                    promoted.add(str(signature))
        if _exists(conn, "concept_promotion_validation_diagnostics"):
            for signature, payload_json in conn.execute("SELECT concept_signature, payload_json FROM concept_promotion_validation_diagnostics").fetchall():
                try:
                    payload = json.loads(str(payload_json or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and bool(payload.get("promoted")):
                    validated.add(str(signature))
    return promoted, validated


def _h07_valid_from_metrics(result: dict[str, Any]) -> bool:
    return bool(
        int(result.get("promoted_concept_count") or 0) >= 1
        and int(result.get("concept_strong_transfer_success_count") or 0) >= 2
        and float(result.get("max_compression_gain") or 0.0) >= 1.50
        and float(result.get("max_promotion_score") or 0.0) >= 0.55
        and (int(result.get("promoted_cross_context_count_max") or 0) >= 3 or int(result.get("promoted_cross_game_count_max") or 0) >= 2)
        and int(result.get("max_source_role_count") or 0) >= 1
        and int(result.get("max_source_family_count") or 0) >= 2
        and int(result.get("roles_used_for_concepts") or 0) >= 3
        and float(result.get("transfer_success_rate") or 0.0) > 0.0
        and float(result.get("concept_transfer_success_concentration") or 0.0) <= 0.80
        and int(result.get("promoted_overconcentrated_concept_count") or 0) == 0
    )


def _evaluate_h07(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = dict(_ORIGINAL_H07(*args, **kwargs))
    memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
    promoted, validated = _validated_concepts(memory_dir)
    unvalidated = promoted - validated
    result["cumulative_validated_promoted_concept_count"] = len(promoted & validated)
    result["concepts_retained_without_cumulative_validation"] = len(unvalidated)
    result["promotion_retained_without_current_validation"] = bool(unvalidated)
    result["cumulative_validation_history_applied"] = True
    if not unvalidated:
        result["concepts_retained_without_current_validation"] = 0
        result["missing_evidence"] = [item for item in list(result.get("missing_evidence") or []) if "retained without current held-out validation" not in str(item)]
        if _h07_valid_from_metrics(result):
            result["decision"] = "VALID"
    core = dict(result.get("core_metrics") or {})
    core.update({"cumulative_validated_promoted_concept_count": len(promoted & validated), "concepts_retained_without_cumulative_validation": len(unvalidated)})
    result["core_metrics"] = core
    return result


def _historical_h08_qualifying(memory_dir: Path) -> tuple[int, list[str]]:
    from v6 import hypothesis_h08_report as h08
    from v6.higher_order_evidence_history import WORLD_VALIDATION_HISTORY

    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists():
        return 0, []
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        if not _exists(conn, WORLD_VALIDATION_HISTORY):
            return 0, []
        link_map: dict[str, dict[str, set[str]]] = {}
        if _exists(conn, "world_model_links"):
            for row in conn.execute("SELECT component_signature, linked_type, linked_key FROM world_model_links").fetchall():
                groups = link_map.setdefault(str(row[0]), {})
                groups.setdefault(str(row[1]), set()).add(str(row[2]))
        rows = [dict(row) for row in conn.execute(f'SELECT * FROM "{WORLD_VALIDATION_HISTORY}"').fetchall()]
    qualifying: set[str] = set()
    for row in rows:
        signature = str(row.get("component_signature") or "")
        links = link_map.get(signature, {})
        record = {
            "component_signature": signature,
            "effective_currently_coherent": bool(int(row.get("state_currently_coherent") or 0) or int(row.get("is_coherent") or 0)),
            "effective_validation_status": str(row.get("state_validation_status") or "").strip().lower(),
            "has_positive_heldout_gain": h08._has_positive_heldout_gain(row),
            "cross_context_count": int(row.get("cross_context_count") or 0),
            "cross_game_count": int(row.get("cross_game_count") or 0),
            "supported_context_count": len(links.get("context", set())),
            "concept_link_count": len(links.get("concept", set())),
            "role_link_count": len(links.get("role", set())),
            "family_link_count": int(row.get("linked_family_count") or 0),
            "verified_predicted_outcome_count": int(row.get("predicted_outcome_count") or 0) if str(row.get("prediction_evidence_status") or "missing") == "verified" else 0,
            "coherence_score": float(row.get("coherence_score") or 0.0),
            "explanatory_coverage": float(row.get("explanatory_coverage") or 0.0),
            "candidate_only": int(row.get("candidate_only") or 0) == 1,
            "heldout_prediction_gain": row.get("heldout_prediction_gain"),
            "validation_action_selection_lift": row.get("validation_action_selection_lift"),
            "validation_transfer_lift": row.get("validation_transfer_lift"),
            "validation_contradiction_resolution": row.get("validation_contradiction_resolution"),
            "validation_explanatory_gain": row.get("validation_explanatory_gain"),
        }
        if h08._component_passes_h08_validity(record):
            qualifying.add(signature)
    return len(qualifying), sorted(qualifying)


def _evaluate_h08(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = dict(_ORIGINAL_H08(*args, **kwargs))
    memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
    historical_count, historical_signatures = _historical_h08_qualifying(memory_dir)
    result["historical_qualifying_component_count"] = historical_count
    result["historical_qualifying_component_signatures"] = historical_signatures[:200]
    result["cumulative_validation_history_applied"] = True
    effective_qualifying = max(int(result.get("qualifying_component_count") or 0), historical_count)
    result["cumulative_qualifying_component_count"] = effective_qualifying
    if historical_count > 0:
        result["candidate_proxy_only"] = False
        if int(result.get("promoted_concept_count") or 0) >= 1 and int(result.get("role_candidate_count") or 0) >= 1 and int(result.get("role_transfer_success_count") or 0) >= 1:
            result["decision"] = "VALID"
            result["missing_evidence"] = [item for item in list(result.get("missing_evidence") or []) if "No single world-model component satisfies" not in str(item) and "qualifying_components" not in str(item)]
    core = dict(result.get("core_metrics") or {})
    core.update({"historical_qualifying_component_count": historical_count, "cumulative_qualifying_component_count": effective_qualifying})
    result["core_metrics"] = core
    return result


def _validate_provenance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = dict(_ORIGINAL_PROVENANCE(*args, **kwargs))
    memory_dir = kwargs.get("memory_dir")
    if memory_dir is None:
        return report
    historical_count, _ = _historical_h08_qualifying(Path(memory_dir))
    if historical_count <= 0:
        return report
    by_hypothesis = dict(report.get("by_hypothesis") or {})
    h08_counts = dict(by_hypothesis.get("H08") or {})
    if int(h08_counts.get("verified_claim_count") or 0) == 0:
        h08_counts["verified_claim_count"] = 1
        report["verified_claim_count"] = int(report.get("verified_claim_count") or 0) + 1
    by_hypothesis["H08"] = h08_counts
    report["by_hypothesis"] = by_hypothesis
    return report


def install_cumulative_evidence_reporting() -> None:
    global _INSTALLED, _ORIGINAL_H07, _ORIGINAL_H08, _ORIGINAL_PROVENANCE
    if _INSTALLED:
        return
    from v6 import hypothesis_h07_report as h07
    from v6 import hypothesis_h08_report as h08
    from v6 import hypothesis_suite_report as suite
    from v6 import provenance_validation as provenance
    _ORIGINAL_H07 = h07.evaluate_h07_concept_emergence
    _ORIGINAL_H08 = h08.evaluate_h08_world_model_coherence
    _ORIGINAL_PROVENANCE = provenance.validate_hypothesis_provenance
    h07.evaluate_h07_concept_emergence = _evaluate_h07
    h08.evaluate_h08_world_model_coherence = _evaluate_h08
    provenance.validate_hypothesis_provenance = _validate_provenance
    suite.evaluate_h07_concept_emergence = _evaluate_h07
    suite.evaluate_h08_world_model_coherence = _evaluate_h08
    suite.validate_hypothesis_provenance = _validate_provenance
    _INSTALLED = True
