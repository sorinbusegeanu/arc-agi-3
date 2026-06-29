from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from v6.future_options import derive_future_option_memory


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    return [name for name in required if name not in tables]


def evaluate_h09_future_option_motifs(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not already_derived and current_state.exists():
        derive_future_option_memory(memory_dir=memory_dir, run_dir=run_dir)
    if not current_state.exists():
        result = {
            "hypothesis_id": "H09",
            "evidence_source": "compact_memory",
            "decision": "INSUFFICIENT_EVIDENCE",
            "missing_evidence": [f"Missing expected compact-memory file: {current_state}"],
            "core_metrics": {},
        }
        _write(output_dir, result)
        return result
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        missing_tables = _missing_tables(
            conn,
            (
                "future_option_events",
                "future_option_motifs",
                "higher_order_milestones",
                "stable_contingencies",
                "transformation_families",
            ),
        )
        if missing_tables:
            result = {
                "hypothesis_id": "H09",
                "evidence_source": "compact_memory",
                "decision": "INSUFFICIENT_EVIDENCE",
                "missing_evidence": [f"Missing expected compact-memory table(s): {', '.join(missing_tables)}"],
                "core_metrics": {},
            }
            _write(output_dir, result)
            return result
        events = [dict(row) for row in conn.execute("SELECT * FROM future_option_events ORDER BY event_id ASC").fetchall()]
        motifs = [dict(row) for row in conn.execute("SELECT * FROM future_option_motifs ORDER BY motif_signature ASC").fetchall()]
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
        summary_row = conn.execute(
            "SELECT value_json FROM memory_summary WHERE key = 'future_option_derivation_summary'"
        ).fetchone()
        derivation_summary = json.loads(str(summary_row[0])) if summary_row and summary_row[0] else {}
        stable_contingencies_count = int(conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0])
        transformation_families_count = int(conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0])
    motif_type_counts = Counter(str(row["motif_type"] or "unknown") for row in motifs)
    emergent_count = sum(1 for row in motifs if int(row["is_emergent"] or 0) == 1)
    cross_context_motif_count = sum(1 for row in motifs if int(row["cross_context_count"] or 0) >= 2)
    cross_game_motif_count = sum(1 for row in motifs if int(row["cross_game_count"] or 0) >= 2)
    mean_abs_option_delta = _mean([abs(float(row.get("option_delta") or 0.0)) for row in events])
    max_abs_option_delta = max((abs(float(row.get("option_delta") or 0.0)) for row in events), default=None)
    mean_motif_stability_score = _mean([row.get("motif_stability_score") for row in motifs])
    result = {
        "hypothesis_id": "H09",
        "evidence_source": "compact_memory",
        "future_option_event_count": len(events),
        "future_option_motif_count": len(motifs),
        "emergent_future_option_motif_count": emergent_count,
        "motif_type_counts": dict(sorted(motif_type_counts.items())),
        "motif_type_source_counts": None,
        "unknown_motif_source_count": None,
        "unknown_motif_source_ratio": None,
        "cross_context_motif_count": cross_context_motif_count,
        "cross_game_motif_count": cross_game_motif_count,
        "mean_abs_option_delta": mean_abs_option_delta,
        "max_abs_option_delta": max_abs_option_delta,
        "mean_motif_stability_score": mean_motif_stability_score,
        "unknown_motif_count": None,
        "unknown_motif_ratio": None,
        "unknown_motif_event_count": None,
        "unknown_motif_event_ratio": None,
        "live_delta_event_count": 0,
        "structured_effect_event_count": 0,
        "text_keyword_event_count": 0,
        "future_option_edge_event_count": 0,
        "first_future_option_event_step": milestone_map.get("first_future_option_event_step"),
        "first_emergent_future_option_motif_step": milestone_map.get("first_emergent_future_option_motif_step"),
        "stable_contingencies_count": stable_contingencies_count,
        "transformation_families_count": transformation_families_count,
        "stable_contingency_rows_seen": derivation_summary.get("stable_contingency_rows_seen"),
        "stable_contingency_events_inserted": derivation_summary.get("stable_contingency_events_inserted"),
        "transformation_family_rows_seen": derivation_summary.get("transformation_family_rows_seen"),
        "transformation_family_events_inserted": derivation_summary.get("transformation_family_events_inserted"),
        "carrier_rows_seen": derivation_summary.get("carrier_rows_seen"),
        "carrier_events_inserted": derivation_summary.get("carrier_events_inserted"),
        "role_rows_seen": derivation_summary.get("role_rows_seen"),
        "role_events_inserted": derivation_summary.get("role_events_inserted"),
        "future_option_events_inserted_total": derivation_summary.get("future_option_events_inserted_total"),
        "future_option_motifs_inserted_total": derivation_summary.get("future_option_motifs_inserted_total"),
        "future_option_derivation_error": derivation_summary.get("future_option_derivation_error"),
        "missing_evidence": [],
    }
    source_counts = Counter()
    unknown_event_count = 0
    for row in events:
        evidence = json.loads(str(row.get("evidence_json") or "{}"))
        source_counts[str(evidence.get("motif_type_source") or "unknown")] += 1
        if str(row.get("motif_type") or "unknown") == "unknown":
            unknown_event_count += 1
    unknown_motif_count = int(motif_type_counts.get("unknown", 0))
    result["motif_type_source_counts"] = dict(sorted(source_counts.items()))
    result["unknown_motif_count"] = unknown_motif_count
    result["unknown_motif_ratio"] = (unknown_motif_count / len(motifs)) if motifs else None
    result["unknown_motif_event_count"] = unknown_event_count
    result["unknown_motif_event_ratio"] = (unknown_event_count / len(events)) if events else None
    result["unknown_motif_source_count"] = int(source_counts.get("unknown", 0))
    result["unknown_motif_source_ratio"] = (result["unknown_motif_source_count"] / len(events)) if events else None
    result["live_delta_event_count"] = int(source_counts.get("live_delta", 0)) + int(source_counts.get("live_delta_rule", 0))
    result["structured_effect_event_count"] = int(source_counts.get("structured_effect", 0))
    result["text_keyword_event_count"] = int(source_counts.get("text_keyword", 0))
    result["future_option_edge_event_count"] = int(source_counts.get("future_option_edge", 0))
    non_unknown_types = [key for key, value in motif_type_counts.items() if key != "unknown" and value > 0]
    if not events:
        if stable_contingencies_count > 0 or transformation_families_count > 0:
            result["decision"] = "INSUFFICIENT_EVIDENCE"
            result["missing_evidence"].append("Future-option derivation produced zero events despite available substrate.")
        else:
            result["decision"] = "INCONCLUSIVE"
    elif events and not motifs:
        result["decision"] = "INVALID"
    elif motifs and emergent_count == 0:
        result["decision"] = "PARTIALLY_VALID"
    elif (
        emergent_count >= 1
        and len(non_unknown_types) >= 2
        and (cross_context_motif_count >= 1 or cross_game_motif_count >= 1)
        and (mean_abs_option_delta or 0.0) > 0.0
        and ((result.get("unknown_motif_ratio") or 0.0) <= 0.20)
        and ((result.get("unknown_motif_event_ratio") or 0.0) <= 0.20)
        and ((result.get("unknown_motif_source_ratio") or 0.0) <= 0.20)
    ):
        result["decision"] = "VALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    if (result.get("unknown_motif_event_ratio") or 0.0) > 0.20:
        result["missing_evidence"].append("Future-option event classification remains mostly unknown.")
    result["evidence_diagnostics"] = {
        "stable_contingencies_count": stable_contingencies_count,
        "transformation_families_count": transformation_families_count,
        "future_option_event_count": len(events),
        "future_option_motif_count": len(motifs),
        "future_option_derivation_summary_present": bool(derivation_summary),
    }
    result["core_metrics"] = {
        key: result.get(key)
        for key in (
            "future_option_event_count",
            "stable_contingency_rows_seen",
            "stable_contingency_events_inserted",
            "transformation_family_rows_seen",
            "transformation_family_events_inserted",
            "carrier_rows_seen",
            "carrier_events_inserted",
            "role_rows_seen",
            "role_events_inserted",
            "future_option_events_inserted_total",
            "future_option_motifs_inserted_total",
            "future_option_motif_count",
            "emergent_future_option_motif_count",
            "motif_type_counts",
            "motif_type_source_counts",
            "cross_context_motif_count",
            "cross_game_motif_count",
            "mean_abs_option_delta",
            "max_abs_option_delta",
            "mean_motif_stability_score",
            "unknown_motif_count",
            "unknown_motif_ratio",
            "unknown_motif_event_count",
            "unknown_motif_event_ratio",
            "unknown_motif_source_count",
            "unknown_motif_source_ratio",
            "live_delta_event_count",
            "structured_effect_event_count",
            "text_keyword_event_count",
            "future_option_edge_event_count",
        )
    }
    _write(output_dir, result)
    return result


def _mean(values: list[Any]) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None


def _write(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h09_future_option_motifs_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H09 decision: {result.get('decision')}\n"
        f"future-option events: {result.get('future_option_event_count')}\n"
        f"future-option motifs: {result.get('future_option_motif_count')}\n"
        f"emergent motifs: {result.get('emergent_future_option_motif_count')}\n"
        f"motif types: {result.get('motif_type_counts')}\n"
        f"motif type sources: {result.get('motif_type_source_counts')}\n"
        f"unknown motif ratio: {result.get('unknown_motif_ratio')}\n"
    )
    (output_dir / "h09_future_option_motifs_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h09_future_option_motifs.md").write_text("```\n" + text + "```\n", encoding="utf-8")
