from __future__ import annotations


def compute_planner_probe_escalation_metrics(events: list[dict], *, total_rounds: int | None = None) -> dict:
    rows = [dict(row) for row in list(events or [])]
    selected = [row for row in rows if str(row.get("event_type") or "") in {"detector poi selected", "detector poi revisited"}]
    revisited = [row for row in rows if str(row.get("event_type") or "") == "detector poi revisited"]
    escalated_verification = [row for row in rows if str(row.get("event_type") or "") == "detector poi escalated to verification"]
    escalated_chain = [row for row in rows if str(row.get("event_type") or "") == "detector poi escalated to chain"]
    stale = [row for row in rows if str(row.get("event_type") or "") == "detector poi marked stale"]
    total = max(1, int(total_rounds or len({row.get("round_id") for row in rows}) or 1))
    downstream_graph_support_after_probe = sum(
        int(dict(row.get("payload", {}) or {}).get("downstream_support_gained", {}).get("new_graph_edges", 0) or 0)
        for row in selected
    )
    return {
        "detector_poi_selection_rate": float(len(selected)) / float(total),
        "repeated_same_probe_rate": float(len(revisited)) / float(max(1, len(selected))),
        "probe_to_verification_rate": float(len(escalated_verification)) / float(max(1, len(selected))),
        "probe_to_chain_rate": float(len(escalated_chain)) / float(max(1, len(selected))),
        "probe_stale_rate": float(len(stale)) / float(max(1, len(selected))),
        "downstream_graph_support_after_probe": downstream_graph_support_after_probe,
        "exit_success_after_probe_escalation": len(escalated_chain),
    }
