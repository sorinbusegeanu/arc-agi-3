from __future__ import annotations


def compute_entity_identity_metrics(entity_rows: list[dict]) -> dict:
    rows = [dict(row or {}) for row in list(entity_rows or [])]
    stable = [row for row in rows if str(row.get("identity_status") or "") == "match_existing"]
    ambiguous = [row for row in rows if str(row.get("identity_status") or "") == "ambiguous_match"]
    forced_merge = [row for row in rows if bool(row.get("forced_merge", False))]
    new_entity = [row for row in rows if str(row.get("identity_status") or "") == "new_entity"]
    drift = [row for row in rows if bool(row.get("identity_drift", False))]
    path_breakage = [row for row in rows if bool(row.get("path_breakage_due_to_identity_instability", False))]
    total = max(1, len(rows))
    return {
        "stable_identity_rate": float(len(stable)) / float(total),
        "ambiguous_identity_rate": float(len(ambiguous)) / float(total),
        "forced_merge_rate": float(len(forced_merge)) / float(total),
        "identity_drift_rate": float(len(drift)) / float(total),
        "path_breakage_from_identity_instability": len(path_breakage),
        "new_entity_rate": float(len(new_entity)) / float(total),
    }
