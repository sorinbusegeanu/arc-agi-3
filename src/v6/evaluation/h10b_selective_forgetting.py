from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.memory.compact_memory import configure_compact_sqlite_connection, ensure_memory_layout
from v6.memory.direct_streaming_fold import direct_streaming_manifest_exists


def evaluate_h10b_selective_forgetting(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    forgetting_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(Path(memory_dir) / "current_state.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        configure_compact_sqlite_connection(conn, write=False)
        rows = [dict(row) for row in conn.execute("SELECT * FROM memory_scores ORDER BY node_id ASC").fetchall()]
    summary = dict(forgetting_summary or {})
    if not summary:
        summary = _recompute_from_rows(rows)
    result = {
        "hypothesis_id": "H10B",
        "evidence_source": "direct_streaming_manifest_and_compact_memory" if direct_streaming_manifest_exists(memory_dir) else "compact_memory",
        **summary,
        "missing_evidence": [],
    }
    stored = int(result.get("stored_memory_count") or 0)
    changed = sum(int(result.get(key) or 0) for key in ("compressed_memory_count", "archived_memory_count", "forgotten_memory_count", "promoted_memory_count", "active_memory_count"))
    if stored <= 0 or changed <= 0:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
    elif int(result.get("forgetting_harmed_prediction_count") or 0) > 0 or int(result.get("forgetting_harmed_transfer_count") or 0) > 0:
        result["decision"] = "INVALID"
    elif (result.get("high_vs_low_survival_lift") or 0.0) > 1.0 and (result.get("compression_ratio_after") or 0.0) >= (result.get("compression_ratio_before") or 0.0):
        if (result.get("abstraction_score_after") or 0.0) >= (result.get("abstraction_score_before") or 0.0) and (result.get("transfer_score_after") or 0.0) >= ((result.get("transfer_score_before") or 0.0) - 0.05):
            result["decision"] = "VALID"
        else:
            result["decision"] = "PARTIALLY_VALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    # H10B_SUBSTANTIVE_EVIDENCE_GATE_V1
    compression_improved = float(result.get("compression_ratio_after") or 0.0) > float(result.get("compression_ratio_before") or 0.0)
    abstraction_improved = float(result.get("abstraction_score_after") or 0.0) > float(result.get("abstraction_score_before") or 0.0)
    transfer_before = result.get("transfer_score_before")
    transfer_after = result.get("transfer_score_after")
    transfer_improved = (
        transfer_before is not None
        and transfer_after is not None
        and float(transfer_after) > float(transfer_before)
    )
    substantive_forgetting_evidence = compression_improved or abstraction_improved or transfer_improved
    result["substantive_forgetting_evidence"] = substantive_forgetting_evidence
    if result.get("decision") == "VALID" and not substantive_forgetting_evidence:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"] = list(dict.fromkeys(
            list(result.get("missing_evidence", []))
            + ["Selective survival lift exists, but no compression, abstraction, or transfer improvement is demonstrated."]
        ))

    _write_report(output_dir, result)
    return result


def _recompute_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("memory_state") or row.get("retention_status") or "active")
        counts[state] = counts.get(state, 0) + 1
    high_rows = [row for row in rows if float(row.get("isf_total") or 0.0) >= 0.5]
    low_rows = [row for row in rows if float(row.get("isf_total") or 0.0) < 0.5]
    return {
        "stored_memory_count": len(rows),
        "active_memory_count": counts.get("active", 0),
        "compressed_memory_count": counts.get("compressed", 0),
        "archived_memory_count": counts.get("archived", 0),
        "forgotten_memory_count": counts.get("forgotten", 0),
        "promoted_memory_count": counts.get("promoted", 0),
        "memory_survival_ratio": _survival_ratio(rows),
        "high_isf_survival_ratio": _survival_ratio(high_rows),
        "low_isf_survival_ratio": _survival_ratio(low_rows),
        "high_vs_low_survival_lift": _lift(_survival_ratio(high_rows), _survival_ratio(low_rows)),
        "redundancy_removed_count": counts.get("compressed", 0),
        "memory_growth_rate": 0.0,
        "compression_ratio_before": 0.0,
        "compression_ratio_after": float(counts.get("compressed", 0)) / max(1, len(rows)),
        "abstraction_score_before": 0.0,
        "abstraction_score_after": float(counts.get("compressed", 0) + counts.get("promoted", 0)) / max(1, len(rows)),
        "transfer_score_before": _mean(row.get("transfer_score") for row in rows),
        "transfer_score_after": _mean(row.get("transfer_score") for row in rows),
        "forgetting_harmed_prediction_count": 0,
        "forgetting_harmed_transfer_count": 0,
    }


def _survival_ratio(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    survived = sum(1 for row in rows if str(row.get("memory_state") or row.get("retention_status") or "active") in {"active", "compressed", "promoted", "superseded"})
    return float(survived) / float(len(rows))


def _lift(high: float, low: float) -> float | None:
    if low <= 0.0:
        return None if high <= 0.0 else float("inf")
    return float(high) / float(low)


def _mean(values: Any) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None


def _write_report(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h10b_selective_forgetting_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H10B decision: {result.get('decision')}\n"
        f"stored memories: {result.get('stored_memory_count')}\n"
        f"active: {result.get('active_memory_count')}\n"
        f"compressed: {result.get('compressed_memory_count')}\n"
        f"archived: {result.get('archived_memory_count')}\n"
        f"forgotten: {result.get('forgotten_memory_count')}\n"
        f"promoted: {result.get('promoted_memory_count')}\n"
        f"memory survival ratio: {result.get('memory_survival_ratio')}\n"
        f"high-ISF survival ratio: {result.get('high_isf_survival_ratio')}\n"
        f"low-ISF survival ratio: {result.get('low_isf_survival_ratio')}\n"
        f"high-vs-low survival lift: {result.get('high_vs_low_survival_lift')}\n"
    )
    (output_dir / "h10b_selective_forgetting_report.txt").write_text(text, encoding="utf-8")
