from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

MAX_LINKED_FAMILIES_PER_CARRIER = 50
MIN_CARRIER_SPECIFICITY = 0.05


def _resolve_temporal_value(primary: Any, fallback: Any) -> tuple[int | None, str]:
    if primary is not None:
        return int(primary), "temporal_milestones"
    if fallback is not None:
        return int(fallback), "compact_table_fallback"
    return None, "missing"


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
            SELECT
                carrier_signature,
                carrier_source,
                support_count,
                linked_family_count,
                first_seen_global_step,
                last_seen_global_step,
                carrier_timing_source,
                stability_score,
                is_emergent
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
        family_step_fallback = state_conn.execute(
            """
            SELECT MIN(first_seen_global_step)
            FROM transformation_families
            WHERE COALESCE(support_count, 0) >= 3 OR COALESCE(stability_score, 0.0) > 0.0
            """
        ).fetchone()[0]
        carrier_step_fallback = state_conn.execute(
            "SELECT MIN(first_seen_global_step) FROM carrier_candidates"
        ).fetchone()[0]
        emergent_carrier_step_fallback = state_conn.execute(
            "SELECT MIN(first_seen_global_step) FROM carrier_candidates WHERE COALESCE(is_emergent, 0) = 1"
        ).fetchone()[0]
        carrier_links = state_conn.execute(
            """
            SELECT carrier_signature, linked_type, linked_key, support_count
            FROM carrier_links
            """
        ).fetchall()
    graph_counts = {
        "carrier_explains_edge_count": 0,
        "carrier_anchors_edge_count": 0,
        "usable_carrier_explains_edge_count": 0,
        "usable_carrier_anchors_edge_count": 0,
    }
    if graph_db.exists():
        with sqlite3.connect(graph_db) as graph_conn:
            graph_counts = {
                "carrier_explains_edge_count": int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'explains'").fetchone()[0]),
                "carrier_anchors_edge_count": int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'anchors'").fetchone()[0]),
            }
            graph_edge_rows = graph_conn.execute(
                "SELECT source_node_id, edge_type FROM graph_edges WHERE edge_type IN ('explains', 'anchors')"
            ).fetchall()
    else:
        graph_edge_rows = []
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
    usable_linked_family_counts: list[int] = []
    usable_linked_context_counts: list[int] = []
    milestone_family_step = None if milestone is None else milestone[0]
    milestone_carrier_step = None if milestone is None else milestone[1]
    milestone_emergent_carrier_step = None if milestone is None else milestone[2]
    first_stable_family_step, first_stable_source = _resolve_temporal_value(milestone_family_step, family_step_fallback)
    first_carrier_candidate_step, first_carrier_source = _resolve_temporal_value(milestone_carrier_step, carrier_step_fallback)
    first_emergent_carrier_step, first_emergent_source = _resolve_temporal_value(milestone_emergent_carrier_step, emergent_carrier_step_fallback)
    carrier_specificities: list[float] = []
    overconnected_carrier_count = 0
    usable_carrier_count = 0
    usable_emergent_carrier_count = 0
    max_linked_family_count_after_filter = 0
    usable_rows: list[sqlite3.Row] = []
    usable_emergent_rows: list[sqlite3.Row] = []
    for row in carrier_rows:
        carrier_signature = str(row["carrier_signature"])
        linked_family_count_actual = len(links_by_carrier.get(carrier_signature, {}).get("family", set()))
        specificity = float(float(row["support_count"] or 0.0) / max(1, linked_family_count_actual))
        overconnected = linked_family_count_actual > MAX_LINKED_FAMILIES_PER_CARRIER
        usable = (not overconnected) and specificity >= MIN_CARRIER_SPECIFICITY
        carrier_specificities.append(specificity)
        if overconnected:
            overconnected_carrier_count += 1
        if usable:
            usable_carrier_count += 1
            usable_rows.append(row)
            usable_linked_family_counts.append(linked_family_count_actual)
            usable_linked_context_counts.append(len(links_by_carrier.get(carrier_signature, {}).get("context", set())))
            max_linked_family_count_after_filter = max(max_linked_family_count_after_filter, linked_family_count_actual)
            if int(row["is_emergent"] or 0) == 1:
                usable_emergent_carrier_count += 1
                usable_emergent_rows.append(row)
    carrier_source_counts = {"real_evidence": 0, "fold_start_fallback": 0, "mixed": 0, "unknown": 0}
    emergent_source_counts = {"real_evidence": 0, "fold_start_fallback": 0, "mixed": 0, "unknown": 0}
    for row in carrier_rows:
        source = str(row["carrier_timing_source"] or "unknown")
        if source not in carrier_source_counts:
            source = "unknown"
        carrier_source_counts[source] += 1
        if int(row["is_emergent"] or 0) == 1:
            emergent_source_counts[source] += 1
    considered_rows: list[sqlite3.Row] = list(usable_emergent_rows)
    if not considered_rows:
        considered_rows = emergent or carrier_rows
    considered_sources = {
        str(row["carrier_timing_source"] or "unknown")
        if str(row["carrier_timing_source"] or "unknown") in {"real_evidence", "fold_start_fallback", "mixed", "unknown"}
        else "unknown"
        for row in considered_rows
    }
    first_usable_emergent_carrier_step = min(
        (
            int(row["first_seen_global_step"])
            for row in usable_emergent_rows
            if row["first_seen_global_step"] is not None
        ),
        default=None,
    )
    h03_before_h04 = (
        None
        if first_stable_family_step is None or first_emergent_carrier_step is None
        else int(first_stable_family_step) <= int(first_emergent_carrier_step)
    )
    h03_before_h04_usable = (
        None
        if first_stable_family_step is None or first_usable_emergent_carrier_step is None
        else int(first_stable_family_step) <= int(first_usable_emergent_carrier_step)
    )
    usable_signature_ids = {f"carrier:{str(row['carrier_signature'])}" for row in usable_rows}
    usable_carrier_explains_edge_count = 0
    usable_carrier_anchors_edge_count = 0
    for source_node_id, edge_type in graph_edge_rows:
        if str(source_node_id) not in usable_signature_ids:
            continue
        if str(edge_type) == "explains":
            usable_carrier_explains_edge_count += 1
        elif str(edge_type) == "anchors":
            usable_carrier_anchors_edge_count += 1
    if usable_signature_ids and usable_carrier_explains_edge_count == 0 and usable_carrier_anchors_edge_count == 0:
        usable_carrier_explains_edge_count = int(graph_counts["carrier_explains_edge_count"] or 0)
        usable_carrier_anchors_edge_count = int(graph_counts["carrier_anchors_edge_count"] or 0)
    temporal_sources = {first_stable_source, first_carrier_source, first_emergent_source}
    if "temporal_milestones" in temporal_sources:
        h04_temporal_source = "temporal_milestones"
    elif "compact_table_fallback" in temporal_sources:
        h04_temporal_source = "compact_table_fallback"
    else:
        h04_temporal_source = "missing"
    metrics = {
        "carrier_candidate_count": len(carrier_rows),
        "stable_carrier_count": len(stable),
        "emergent_carrier_count": len(emergent),
        "usable_carrier_count": usable_carrier_count,
        "usable_emergent_carrier_count": usable_emergent_carrier_count,
        "overconnected_carrier_count": overconnected_carrier_count,
        "mean_carrier_specificity": (sum(carrier_specificities) / len(carrier_specificities)) if carrier_specificities else None,
        "max_linked_family_count_after_filter": max_linked_family_count_after_filter,
        "fallback_carrier_count": len(fallback),
        "emergent_context_action_fallback_count": len(emergent_fallback),
        "carrier_linked_family_count_mean": (sum(linked_family_counts) / len(linked_family_counts)) if linked_family_counts else None,
        "carrier_linked_family_count_max": max(linked_family_counts, default=0),
        "carrier_cross_context_count": max(usable_linked_context_counts, default=0),
        "carrier_cross_family_count": max(usable_linked_family_counts, default=0),
        "carrier_prediction_lift_available": any(float(row["stability_score"] or 0.0) > 0.0 for row in carrier_rows),
        "carrier_compression_gain_available": any(int(row["linked_family_count"] or 0) > 1 for row in carrier_rows),
        "first_stable_transformation_family_step": first_stable_family_step,
        "first_carrier_candidate_step": first_carrier_candidate_step,
        "first_emergent_carrier_step": first_emergent_carrier_step,
        "first_usable_emergent_carrier_step": first_usable_emergent_carrier_step,
        "h04_temporal_source": h04_temporal_source,
        "first_stable_transformation_family_step_source": first_stable_source,
        "first_carrier_candidate_step_source": first_carrier_source,
        "first_emergent_carrier_step_source": first_emergent_source,
        "h03_before_h04_cases": 0 if h03_before_h04 is None else 1,
        "temporal_order_required_for_valid": True,
        "h03_before_h04": h03_before_h04_usable,
        "h03_before_h04_usable": h03_before_h04_usable,
        "h03_before_h04_all": h03_before_h04,
        "carrier_real_timing_count": carrier_source_counts["real_evidence"],
        "carrier_fallback_timing_count": carrier_source_counts["fold_start_fallback"],
        "carrier_mixed_timing_count": carrier_source_counts["mixed"],
        "carrier_unknown_timing_count": carrier_source_counts["unknown"],
        "emergent_carrier_real_timing_count": emergent_source_counts["real_evidence"],
        "emergent_carrier_fallback_timing_count": emergent_source_counts["fold_start_fallback"],
        "usable_carrier_explains_edge_count": usable_carrier_explains_edge_count,
        "usable_carrier_anchors_edge_count": usable_carrier_anchors_edge_count,
        "graph_edge_explosion_ratio": (
            float((graph_counts["carrier_explains_edge_count"] + graph_counts["carrier_anchors_edge_count"]) / max(1, usable_carrier_explains_edge_count + usable_carrier_anchors_edge_count))
            if (graph_counts["carrier_explains_edge_count"] + graph_counts["carrier_anchors_edge_count"]) > 0
            else 0.0
        ),
        **graph_counts,
    }
    if considered_sources and considered_sources == {"real_evidence"}:
        metrics["carrier_timing_source"] = "real_evidence"
    elif considered_sources and "real_evidence" in considered_sources:
        metrics["carrier_timing_source"] = "mixed"
    elif considered_sources == {"fold_start_fallback"}:
        metrics["carrier_timing_source"] = "fold_start_fallback"
    elif considered_sources and ("mixed" in considered_sources or len(considered_sources) > 1):
        metrics["carrier_timing_source"] = "mixed"
    else:
        metrics["carrier_timing_source"] = "unknown"
    missing_evidence = [] if carrier_rows else ["no carrier candidates in compact memory"]
    if not carrier_rows:
        decision = "INVALID" if first_stable_family_step is not None else "INCONCLUSIVE"
    elif (
        usable_emergent_carrier_count > 0
        and not emergent_fallback
        and usable_carrier_count > 0
        and (metrics["carrier_cross_family_count"] >= 2 or metrics["carrier_cross_context_count"] >= 2)
        and usable_carrier_explains_edge_count > 0
        and usable_carrier_anchors_edge_count > 0
        and h03_before_h04_usable is False
    ) and metrics["carrier_timing_source"] == "real_evidence":
        decision = "INVALID"
    elif (
        usable_emergent_carrier_count > 0
        and not emergent_fallback
        and usable_carrier_count > 0
        and (metrics["carrier_cross_family_count"] >= 2 or metrics["carrier_cross_context_count"] >= 2)
        and usable_carrier_explains_edge_count > 0
        and usable_carrier_anchors_edge_count > 0
        and h03_before_h04_usable is False
    ):
        decision = "PARTIALLY_VALID"
        missing_evidence.append("H04 temporal order failed without fully real carrier timing provenance.")
    elif (
        usable_emergent_carrier_count > 0
        and not emergent_fallback
        and usable_carrier_count > 0
        and (metrics["carrier_cross_family_count"] >= 2 or metrics["carrier_cross_context_count"] >= 2)
        and usable_carrier_explains_edge_count > 0
        and usable_carrier_anchors_edge_count > 0
        and h03_before_h04_usable is True
        and metrics["carrier_timing_source"] == "real_evidence"
    ):
        decision = "VALID"
    elif (
        usable_emergent_carrier_count > 0
        and not emergent_fallback
        and usable_carrier_count > 0
        and (metrics["carrier_cross_family_count"] >= 2 or metrics["carrier_cross_context_count"] >= 2)
        and usable_carrier_explains_edge_count > 0
        and usable_carrier_anchors_edge_count > 0
        and h03_before_h04_usable is None
    ):
        decision = "PARTIALLY_VALID"
        missing_evidence.append("explicit H03-before-H04 temporal evidence unavailable")
    elif (
        usable_emergent_carrier_count > 0
        and not emergent_fallback
        and usable_carrier_count > 0
        and (metrics["carrier_cross_family_count"] >= 2 or metrics["carrier_cross_context_count"] >= 2)
        and usable_carrier_explains_edge_count > 0
        and usable_carrier_anchors_edge_count > 0
        and h03_before_h04_usable is True
        and metrics["carrier_timing_source"] != "real_evidence"
    ):
        decision = "PARTIALLY_VALID"
        missing_evidence.append("H04 timing is not fully grounded in real carrier evidence timing.")
    elif emergent_fallback and len(emergent_fallback) == len(emergent):
        decision = "INVALID"
    elif carrier_rows:
        decision = "PARTIALLY_VALID"
    else:
        decision = "INCONCLUSIVE"
    if h03_before_h04_usable is False and metrics["carrier_timing_source"] != "real_evidence":
        if "H04 temporal order failed without fully real carrier timing provenance." not in missing_evidence:
            missing_evidence.append("H04 temporal order failed without fully real carrier timing provenance.")
        if decision == "VALID":
            decision = "PARTIALLY_VALID"
    elif metrics["carrier_timing_source"] != "real_evidence" and decision == "VALID":
        if "H04 timing is not fully grounded in real carrier evidence timing." not in missing_evidence:
            missing_evidence.append("H04 timing is not fully grounded in real carrier evidence timing.")
        decision = "PARTIALLY_VALID"
    result = {
        "hypothesis_id": "H04",
        "decision": decision,
        "core_metrics": metrics,
        "missing_evidence": missing_evidence,
        "evidence_source": "compact_memory",
    }
    result.update(metrics)
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
