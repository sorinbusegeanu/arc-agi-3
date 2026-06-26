from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.higher_order_substrate import derive_higher_order_memory
from v6.memory.compact_memory import ensure_memory_layout


def evaluate_h08_world_model_coherence(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_memory_layout(memory_dir)
    if not already_derived:
        derive_higher_order_memory(memory_dir=memory_dir, run_dir=run_dir)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        result = _base_result("INCONCLUSIVE", ["compact memory missing current_state.sqlite"])
        _write_outputs(output_dir, result)
        return result
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        concept_candidate_count = int(conn.execute("SELECT COUNT(*) FROM concept_candidates").fetchone()[0])
        promoted_concept_count = int(conn.execute("SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1").fetchone()[0])
        component_rows = conn.execute(
            """
            SELECT component_signature, coherence_score, explanatory_coverage, cross_context_count, cross_game_count,
                   linked_concept_count, linked_role_count, linked_family_count, prediction_support_count,
                   contradiction_coverage_count, is_coherent, candidate_only
            FROM world_model_components
            ORDER BY component_signature ASC
            """
        ).fetchall()
        component_links = [dict(row) for row in conn.execute("SELECT component_signature, linked_type, linked_key FROM world_model_links ORDER BY component_signature ASC, linked_type ASC, linked_key ASC").fetchall()]
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
    link_map: dict[str, dict[str, set[str]]] = {}
    for row in component_links:
        groups = link_map.setdefault(str(row["component_signature"]), {})
        groups.setdefault(str(row["linked_type"]), set()).add(str(row["linked_key"]))
    world_model_component_count = len(component_rows)
    coherent_rows = [row for row in component_rows if int(row["is_coherent"] or 0) == 1]
    coherent_world_model_component_count = len(coherent_rows)
    mean_coherence_score = (
        sum(float(row["coherence_score"] or 0.0) for row in component_rows) / max(1, world_model_component_count)
        if component_rows
        else None
    )
    max_coherence_score = max((float(row["coherence_score"] or 0.0) for row in component_rows), default=None)
    mean_explanatory_coverage = (
        sum(float(row["explanatory_coverage"] or 0.0) for row in component_rows) / max(1, world_model_component_count)
        if component_rows
        else None
    )
    max_explanatory_coverage = max((float(row["explanatory_coverage"] or 0.0) for row in component_rows), default=None)
    coherent_cross_context_component_count = sum(
        1 for row in component_rows if int(row["is_coherent"] or 0) == 1 and int(row["cross_context_count"] or 0) >= 1
    )
    coherent_cross_game_component_count = sum(
        1 for row in component_rows if int(row["is_coherent"] or 0) == 1 and int(row["cross_game_count"] or 0) >= 1
    )
    candidate_only_world_model_component_count = sum(1 for row in component_rows if int(row["candidate_only"] or 0) == 1)
    component_cross_context_count = max((int(row["cross_context_count"] or 0) for row in component_rows), default=0)
    component_cross_game_count = max((int(row["cross_game_count"] or 0) for row in component_rows), default=0)
    predicted_outcome_count = max((int(row["prediction_support_count"] or 0) for row in component_rows), default=0)
    supported_context_count = max((len(link_map.get(str(row["component_signature"]), {}).get("context", set())) for row in component_rows), default=0)
    concept_link_count = max((len(link_map.get(str(row["component_signature"]), {}).get("concept", set())) for row in component_rows), default=0)
    role_link_count = max((len(link_map.get(str(row["component_signature"]), {}).get("role", set())) for row in component_rows), default=0)
    family_link_count = max((int(row["linked_family_count"] or 0) for row in component_rows), default=0)
    contradiction_coverage_count = sum(int(row["contradiction_coverage_count"] or 0) for row in component_rows)
    metrics = {
        "world_model_component_count": world_model_component_count,
        "coherent_world_model_component_count": coherent_world_model_component_count,
        "candidate_only_world_model_component_count": candidate_only_world_model_component_count,
        "promoted_concept_count": promoted_concept_count,
        "mean_coherence_score": mean_coherence_score,
        "max_coherence_score": max_coherence_score,
        "mean_explanatory_coverage": mean_explanatory_coverage,
        "max_explanatory_coverage": max_explanatory_coverage,
        "coherent_cross_context_component_count": coherent_cross_context_component_count,
        "coherent_cross_game_component_count": coherent_cross_game_component_count,
        "component_cross_context_count": component_cross_context_count,
        "component_cross_game_count": component_cross_game_count,
        "predicted_outcome_count": predicted_outcome_count,
        "supported_context_count": supported_context_count,
        "concept_link_count": concept_link_count,
        "role_link_count": role_link_count,
        "family_link_count": family_link_count,
        "contradiction_coverage_count": contradiction_coverage_count,
        "first_world_model_component_step": milestone_map.get("first_world_model_component_step"),
        "first_coherent_world_model_step": milestone_map.get("first_coherent_world_model_step"),
        "first_promoted_concept_step": milestone_map.get("first_promoted_concept_step"),
    }
    if concept_candidate_count <= 0:
        decision = "INCONCLUSIVE"
        missing = ["no concept candidates available"]
    elif concept_candidate_count > 0 and promoted_concept_count == 0:
        decision = "PARTIALLY_VALID"
        missing = []
    elif world_model_component_count > 0 and coherent_world_model_component_count == 0:
        decision = "PARTIALLY_VALID"
        missing = []
    elif (
        promoted_concept_count > 0
        and coherent_world_model_component_count >= 5
        and (max_coherence_score or 0.0) >= 0.45
        and (max_explanatory_coverage or 0.0) > 0.0
        and (coherent_cross_context_component_count >= 1 or coherent_cross_game_component_count >= 1)
        and concept_link_count >= 2
        and family_link_count >= 2
        and supported_context_count >= 2
        and predicted_outcome_count > 0
    ):
        decision = "VALID"
        missing = []
    elif promoted_concept_count > 0 and world_model_component_count == 0:
        decision = "INVALID"
        missing = []
    elif promoted_concept_count > 0 and component_rows and max((float(row["coherence_score"] or 0.0) for row in component_rows), default=0.0) < 0.20:
        decision = "INVALID"
        missing = []
    else:
        decision = "PARTIALLY_VALID"
        missing = []
    result = _base_result(decision, missing)
    result.update(metrics)
    result["core_metrics"] = dict(metrics)
    _write_outputs(output_dir, result)
    return result


def _base_result(decision: str, missing_evidence: list[str]) -> dict[str, Any]:
    return {
        "hypothesis_id": "H08",
        "decision": decision,
        "missing_evidence": list(missing_evidence),
        "evidence_source": "compact_memory",
    }


def _write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h08_world_model_coherence_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H08 decision: {result.get('decision')}\n"
        f"promoted concepts: {result.get('promoted_concept_count')}\n"
        f"world model components: {result.get('world_model_component_count')}\n"
        f"candidate-only components: {result.get('candidate_only_world_model_component_count')}\n"
        f"coherent world model components: {result.get('coherent_world_model_component_count')}\n"
        f"max coherence score: {result.get('max_coherence_score')}\n"
        f"max explanatory coverage: {result.get('max_explanatory_coverage')}\n"
    )
    (output_dir / "h08_world_model_coherence_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h08_world_model_coherence.md").write_text("```\n" + text + "```\n", encoding="utf-8")
