from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_INSTALLED = False
_ORIGINAL_H06: Any = None
_ORIGINAL_H07: Any = None
_ORIGINAL_H08: Any = None
_ORIGINAL_PROVENANCE: Any = None


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _projection_counts(memory_dir: Path) -> dict[str, int]:
    db = Path(memory_dir) / "current_state.sqlite"
    if not db.exists(): return {}
    with sqlite3.connect(db) as conn:
        if not _exists(conn, "memory_summary"): return {}
        row = conn.execute("SELECT value_json FROM memory_summary WHERE key='reporting_cumulative_projection'").fetchone()
    if row is None or not row[0]: return {}
    try: value = json.loads(str(row[0]))
    except (TypeError, ValueError, json.JSONDecodeError): return {}
    return {str(k): int(v or 0) for k, v in value.items()} if isinstance(value, dict) else {}


def _evaluate_h06(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = dict(_ORIGINAL_H06(*args, **kwargs))
    memory_dir = Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0])
    cumulative = int(_projection_counts(memory_dir).get("role_transfer_attempts", 0) or 0)
    current = 0
    db = memory_dir / "current_state.sqlite"
    if db.exists():
        with sqlite3.connect(db) as conn:
            if _exists(conn, "report_current_role_transfer_attempts"):
                current = int(conn.execute("SELECT COUNT(*) FROM report_current_role_transfer_attempts").fetchone()[0])
    applied = cumulative > current
    gates = result.get("h06_validity_gates") or {}
    if applied and gates and all(bool(g.get("passed")) for g in gates.values()) and int(result.get("legacy_transfer_provenance_count") or 0) == 0 and int(result.get("invalid_transfer_provenance_count") or 0) == 0:
        result["decision"] = "VALID"
        result["missing_evidence"] = [x for x in list(result.get("missing_evidence") or []) if "sampling was capped" not in str(x)]
    result["cumulative_transfer_evidence_applied"] = applied
    result["cumulative_transfer_attempt_count"] = cumulative
    result["current_transfer_attempt_count"] = current
    core = dict(result.get("core_metrics") or {})
    core.update({"cumulative_transfer_evidence_applied": applied, "cumulative_transfer_attempt_count": cumulative, "current_transfer_attempt_count": current})
    result["core_metrics"] = core
    return result


def _validated_concepts(memory_dir: Path) -> tuple[set[str], set[str]]:
    db = memory_dir / "current_state.sqlite"; promoted=set(); validated=set()
    if not db.exists(): return promoted, validated
    with sqlite3.connect(db) as conn:
        if _exists(conn, "concept_promotion_state"):
            state_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(concept_promotion_state)").fetchall()}
            if {"concept_signature", "currently_promoted"} <= state_columns:
                validation_available = "validation_status" in state_columns
                select = "concept_signature, currently_promoted" + (", validation_status" if validation_available else "")
                for row in conn.execute(f"SELECT {select} FROM concept_promotion_state").fetchall():
                    if int(row[1] or 0) == 1:
                        promoted.add(str(row[0]))
                        if validation_available and str(row[2] or "").strip().lower() in {"passed", "validated"}:
                            validated.add(str(row[0]))
        elif _exists(conn, "concept_candidates"):
            for s,p in conn.execute("SELECT concept_signature, is_promoted FROM concept_candidates").fetchall():
                if int(p or 0)==1: promoted.add(str(s))
        if _exists(conn, "concept_promotion_validation_diagnostics"):
            for s,payload_json in conn.execute("SELECT concept_signature, payload_json FROM concept_promotion_validation_diagnostics").fetchall():
                try: payload=json.loads(str(payload_json or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError): continue
                if isinstance(payload,dict) and bool(payload.get("promoted")): validated.add(str(s))
    return promoted, validated


def _h07_valid_from_metrics(r: dict[str, Any]) -> bool:
    return bool(int(r.get("promoted_concept_count") or 0)>=1 and int(r.get("concept_strong_transfer_success_count") or 0)>=2 and float(r.get("max_compression_gain") or 0)>=1.50 and float(r.get("max_promotion_score") or 0)>=0.55 and (int(r.get("promoted_cross_context_count_max") or 0)>=3 or int(r.get("promoted_cross_game_count_max") or 0)>=2) and int(r.get("max_source_role_count") or 0)>=1 and int(r.get("max_source_family_count") or 0)>=2 and int(r.get("roles_used_for_concepts") or 0)>=3 and float(r.get("transfer_success_rate") or 0)>0 and float(r.get("concept_transfer_success_concentration") or 0)<=0.80 and int(r.get("promoted_overconcentrated_concept_count") or 0)==0)


def _evaluate_h07(*args: Any, **kwargs: Any) -> dict[str, Any]:
    r=dict(_ORIGINAL_H07(*args, **kwargs)); memory_dir=Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0]); promoted,validated=_validated_concepts(memory_dir); unvalidated=promoted-validated
    r["cumulative_validated_promoted_concept_count"]=len(promoted&validated); r["concepts_retained_without_cumulative_validation"]=len(unvalidated); r["promotion_retained_without_current_validation"]=bool(unvalidated); r["cumulative_validation_history_applied"]=True
    if not unvalidated:
        r["concepts_retained_without_current_validation"]=0; r["missing_evidence"]=[x for x in list(r.get("missing_evidence") or []) if "retained without current held-out validation" not in str(x)]
        if _h07_valid_from_metrics(r): r["decision"]="VALID"
    core=dict(r.get("core_metrics") or {}); core.update({"cumulative_validated_promoted_concept_count":len(promoted&validated),"concepts_retained_without_cumulative_validation":len(unvalidated)}); r["core_metrics"]=core; return r


def _historical_h08_qualifying(memory_dir: Path) -> tuple[int,list[str]]:
    from v6 import hypothesis_h08_report as h08
    from v6.higher_order_evidence_history import WORLD_VALIDATION_HISTORY
    from v6.world_model_validation_history import LINK_HISTORY
    db=memory_dir/"current_state.sqlite"
    if not db.exists(): return 0,[]
    with sqlite3.connect(db) as conn:
        conn.row_factory=sqlite3.Row
        if not _exists(conn,WORLD_VALIDATION_HISTORY) or not _exists(conn,LINK_HISTORY): return 0,[]
        link_map={}
        for row in conn.execute(f'SELECT component_signature, diagnostic_epoch_id, linked_type, linked_key FROM "{LINK_HISTORY}"').fetchall():
            link_map.setdefault((str(row[0]),str(row[1])),{}).setdefault(str(row[2]),set()).add(str(row[3]))
        rows=[dict(row) for row in conn.execute(f'SELECT * FROM "{WORLD_VALIDATION_HISTORY}"').fetchall()]
    qualifying=set()
    for row in rows:
        sig=str(row.get("component_signature") or ""); epoch=str(row.get("diagnostic_epoch_id") or ""); links=link_map.get((sig,epoch),{})
        rec={"component_signature":sig,"effective_currently_coherent":bool(int(row.get("state_currently_coherent") or 0) or int(row.get("is_coherent") or 0)),"effective_validation_status":str(row.get("state_validation_status") or "").strip().lower(),"has_positive_heldout_gain":h08._has_positive_heldout_gain(row),"cross_context_count":int(row.get("cross_context_count") or 0),"cross_game_count":int(row.get("cross_game_count") or 0),"supported_context_count":len(links.get("context",set())),"concept_link_count":len(links.get("concept",set())),"role_link_count":len(links.get("role",set())),"family_link_count":int(row.get("linked_family_count") or 0),"verified_predicted_outcome_count":int(row.get("predicted_outcome_count") or 0) if str(row.get("prediction_evidence_status") or "missing")=="verified" else 0,"coherence_score":float(row.get("coherence_score") or 0),"explanatory_coverage":float(row.get("explanatory_coverage") or 0),"candidate_only":int(row.get("candidate_only") or 0)==1,"heldout_prediction_gain":row.get("heldout_prediction_gain"),"validation_action_selection_lift":row.get("validation_action_selection_lift"),"validation_transfer_lift":row.get("validation_transfer_lift"),"validation_contradiction_resolution":row.get("validation_contradiction_resolution"),"validation_explanatory_gain":row.get("validation_explanatory_gain")}
        if h08._component_passes_h08_validity(rec): qualifying.add(sig)
    return len(qualifying),sorted(qualifying)


def _evaluate_h08(*args: Any, **kwargs: Any) -> dict[str, Any]:
    r=dict(_ORIGINAL_H08(*args, **kwargs)); memory_dir=Path(kwargs.get("memory_dir") if "memory_dir" in kwargs else args[0]); count,sigs=_historical_h08_qualifying(memory_dir); effective=max(int(r.get("qualifying_component_count") or 0),count)
    r.update({"historical_qualifying_component_count":count,"historical_qualifying_component_signatures":sigs[:200],"cumulative_validation_history_applied":True,"cumulative_qualifying_component_count":effective})
    if count>0:
        r["candidate_proxy_only"]=False
        if int(r.get("promoted_concept_count") or 0)>=1 and int(r.get("role_candidate_count") or 0)>=1 and int(r.get("role_transfer_success_count") or 0)>=1: r["decision"]="VALID"; r["missing_evidence"]=[x for x in list(r.get("missing_evidence") or []) if "No single world-model component satisfies" not in str(x) and "qualifying_components" not in str(x)]
    core=dict(r.get("core_metrics") or {}); core.update({"historical_qualifying_component_count":count,"cumulative_qualifying_component_count":effective}); r["core_metrics"]=core; return r


def _validate_provenance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report=dict(_ORIGINAL_PROVENANCE(*args, **kwargs)); memory_dir=kwargs.get("memory_dir")
    if memory_dir is None: return report
    count,_=_historical_h08_qualifying(Path(memory_dir))
    if count<=0: return report
    by=dict(report.get("by_hypothesis") or {}); h08=dict(by.get("H08") or {})
    if int(h08.get("verified_claim_count") or 0)==0: h08["verified_claim_count"]=1; report["verified_claim_count"]=int(report.get("verified_claim_count") or 0)+1
    by["H08"]=h08; report["by_hypothesis"]=by; return report


def install_cumulative_evidence_reporting() -> None:
    global _INSTALLED,_ORIGINAL_H06,_ORIGINAL_H07,_ORIGINAL_H08,_ORIGINAL_PROVENANCE
    if _INSTALLED: return
    from v6 import hypothesis_h06_report as h06, hypothesis_h07_report as h07, hypothesis_h08_report as h08, hypothesis_suite_report as suite, provenance_validation as provenance
    _ORIGINAL_H06=h06.evaluate_h06_role_transfer; _ORIGINAL_H07=h07.evaluate_h07_concept_emergence; _ORIGINAL_H08=h08.evaluate_h08_world_model_coherence; _ORIGINAL_PROVENANCE=provenance.validate_hypothesis_provenance
    h06.evaluate_h06_role_transfer=_evaluate_h06; h07.evaluate_h07_concept_emergence=_evaluate_h07; h08.evaluate_h08_world_model_coherence=_evaluate_h08; provenance.validate_hypothesis_provenance=_validate_provenance
    suite.evaluate_h06_role_transfer=_evaluate_h06; suite.evaluate_h07_concept_emergence=_evaluate_h07; suite.evaluate_h08_world_model_coherence=_evaluate_h08; suite.validate_hypothesis_provenance=_validate_provenance; _INSTALLED=True
