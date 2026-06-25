from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def evaluate_h04_carrier_emergence(
    *,
    run_dir: Path | None,
    memory_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    del run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"
    graph_db = Path(memory_dir) / "graph.sqlite"
    if not current_state.exists():
        result = {
            "hypothesis_id": "H04",
            "decision": "INCONCLUSIVE",
            "core_metrics": {},
            "missing_evidence": ["compact memory missing current_state.sqlite"],
            "evidence_source": "compact_memory",
        }
        _write_outputs(result, output_dir)
        return result
    with sqlite3.connect(current_state) as state_conn:
        state_conn.row_factory = sqlite3.Row
        carrier_rows = state_conn.execute(
            """
            SELECT carrier_signature, carrier_source, support_count, linked_family_count, stability_score, is_emergent
            FROM carrier_candidates
            """
        ).fetchall()
        milestone = state_conn.execute(
            """
            SELECT
                MIN(first_stable_transformation_family_step),
                MIN(first_carrier_candidate_step),
                MIN(first_emergent_carrier_step)
            FROM temporal_milestones
            """
        ).fetchone()
        carrier_links = state_conn.execute(
            """
            SELECT carrier_signature, linked_type, linked_key, support_count
            FROM carrier_links
            """
        ).fetchall()
    graph_counts = {
        "carrier_explains_edge_count": 0,
        "carrier_anchors_edge_count": 0,
    }
    if graph_db.exists():
        with sqlite3.connect(graph_db) as graph_conn:
            graph_counts = {
                "carrier_explains_edge_count": int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'explains'").fetchone()[0]),
                "carrier_anchors_edge_count": int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'anchors'").fetchone()[0]),
            }
    stable = [row for row in carrier_rows if int(row["support_count"] or 0) >= 3]
    emergent = [row for row in carrier_rows if int(row["is_emergent"] or 0) == 1]
    fallback = [row for row in carrier_rows if str(row["carrier_source"] or "") == "context_action_fallback"]
    emergent_fallback = [row for row in emergent if str(row["carrier_source"] or "") == "context_action_fallback"]
    links_by_carrier: dict[str, dict[str, set[str]]] = {}
    for row in carrier_links:
        carrier = str(row["carrier_signature"])
        links_by_carrier.setdefault(carrier, {"family": set(), "context": set(), "contingency": set()})
        linked_type = str(row["linked_type"])
        if linked_type in links_by_carrier[carrier] and row["linked_key"] not in (None, ""):
            links_by_carrier[carrier][linked_type].add(str(row["linked_key"]))
    linked_family_counts = [len(values["family"]) for values in links_by_carrier.values()]
    linked_context_counts = [len(values["context"]) for values in links_by_carrier.values()]
    first_stable_family_step = None if milestone is None else milestone[0]
    first_carrier_candidate_step = None if milestone is None else milestone[1]
    first_emergent_carrier_step = None if milestone is None else milestone[2]
    h03_before_h04 = (
        None
        if first_stable_family_step is None or first_emergent_carrier_step is None
        else int(first_stable_family_step) <= int(first_emergent_carrier_step)
    )
    metrics = {
        "carrier_candidate_count": len(carrier_rows),
        "stable_carrier_count": len(stable),
        "emergent_carrier_count": len(emergent),
        "fallback_carrier_count": len(fallback),
        "emergent_context_action_fallback_count": len(emergent_fallback),
        "carrier_linked_family_count_mean": (sum(linked_family_counts) / len(linked_family_counts)) if linked_family_counts else None,
        "carrier_linked_family_count_max": max(linked_family_counts, default=0),
        "carrier_cross_context_count": max(linked_context_counts, default=0),
        "carrier_cross_family_count": max(linked_family_counts, default=0),
        "carrier_prediction_lift_available": any(float(row["stability_score"] or 0.0) > 0.0 for row in carrier_rows),
        "carrier_compression_gain_available": any(int(row["linked_family_count"] or 0) > 1 for row in carrier_rows),
        "first_stable_transformation_family_step": first_stable_family_step,
        "first_carrier_candidate_step": first_carrier_candidate_step,
        "first_emergent_carrier_step": first_emergent_carrier_step,
        "h03_before_h04_cases": 0 if h03_before_h04 is None else 1,
        **graph_counts,
    }
    if not carrier_rows:
        decision = "INVALID" if first_stable_family_step is not None else "INCONCLUSIVE"
    elif (
        emergent
        and not emergent_fallback
        and (metrics["carrier_cross_family_count"] >= 2 or metrics["carrier_cross_context_count"] >= 2)
        and graph_counts["carrier_explains_edge_count"] > 0
        and graph_counts["carrier_anchors_edge_count"] > 0
        and (h03_before_h04 is True or h03_before_h04 is None)
    ):
        decision = "VALID"
    elif emergent_fallback and len(emergent_fallback) == len(emergent):
        decision = "INVALID"
    elif carrier_rows:
        decision = "PARTIALLY_VALID"
    else:
        decision = "INCONCLUSIVE"
    result = {
        "hypothesis_id": "H04",
        "decision": decision,
        "core_metrics": metrics,
        "missing_evidence": [] if carrier_rows else ["no carrier candidates in compact memory"],
        "evidence_source": "compact_memory",
    }
    _write_outputs(result, output_dir)
    return result


def _write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "h04_carrier_emergence_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H04 decision: {result.get('decision')}\n"
        f"carrier candidates: {(result.get('core_metrics') or {}).get('carrier_candidate_count')}\n"
        f"stable carriers: {(result.get('core_metrics') or {}).get('stable_carrier_count')}\n"
        f"emergent carriers: {(result.get('core_metrics') or {}).get('emergent_carrier_count')}\n"
    )
    (output_dir / "h04_carrier_emergence_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h04_carrier_emergence.md").write_text("```\n" + text + "```\n", encoding="utf-8")
