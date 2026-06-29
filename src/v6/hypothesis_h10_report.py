from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.future_options import derive_future_option_memory


def _missing_tables(connection: sqlite3.Connection, required: tuple[str, ...]) -> list[str]:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    return [name for name in required if name not in tables]


def _lift(high_value: float | None, low_value: float | None) -> float | None:
    if high_value is None or low_value is None:
        return None
    if low_value <= 0.0:
        return None if high_value <= 0.0 else float("inf")
    return float(high_value) / float(low_value)


def _subtest_result(rows: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    covered = [row for row in rows if row.get(score_key) is not None]
    high_rows = [row for row in covered if int(row.get("high_option_change") or 0) == 1]
    low_rows = [row for row in covered if int(row.get("high_option_change") or 0) == 0]
    high_rate = (sum(1 for row in high_rows if float(row.get(score_key) or 0.0) >= threshold) / len(high_rows)) if high_rows else None
    low_rate = (sum(1 for row in low_rows if float(row.get(score_key) or 0.0) >= threshold) / len(low_rows)) if low_rows else None
    saturation = bool(covered) and (
        sum(1 for row in covered if float(row.get(score_key) or 0.0) >= threshold) in {0, len(covered)}
    )
    return {
        "coverage": len(covered),
        "high_rate": high_rate,
        "low_rate": low_rate,
        "lift": _lift(high_rate, low_rate),
        "saturation": saturation,
        "threshold": threshold,
    }


def evaluate_h10_future_option_attention(
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
            "hypothesis_id": "H10",
            "evidence_source": "compact_memory",
            "decision": "INSUFFICIENT_EVIDENCE",
            "missing_evidence": [f"Missing expected compact-memory file: {current_state}"],
            "future_option_event_count": 0,
            "future_option_attention_link_count": 0,
            "h10_blocked_by_h09": True,
            "core_metrics": {},
        }
        _write(output_dir, result)
        return result
    with sqlite3.connect(current_state) as conn:
        conn.row_factory = sqlite3.Row
        missing_tables = _missing_tables(conn, ("future_option_attention_links", "future_option_events"))
        if missing_tables:
            result = {
                "hypothesis_id": "H10",
                "evidence_source": "compact_memory",
                "decision": "INSUFFICIENT_EVIDENCE",
                "missing_evidence": [f"Missing expected compact-memory table(s): {', '.join(missing_tables)}"],
                "core_metrics": {},
            }
            _write(output_dir, result)
            return result
        rows = [dict(row) for row in conn.execute("SELECT * FROM future_option_attention_links ORDER BY event_id ASC").fetchall()]
        future_option_event_count = int(conn.execute("SELECT COUNT(*) FROM future_option_events").fetchone()[0])
    high_rows = [row for row in rows if int(row["high_option_change"] or 0) == 1]
    low_rows = [row for row in rows if int(row["high_option_change"] or 0) == 0]
    high_attention_rows = [row for row in rows if int(row["high_attention"] or 0) == 1]
    high_both = [row for row in high_rows if int(row["high_attention"] or 0) == 1]
    low_attention = [row for row in low_rows if int(row["high_attention"] or 0) == 1]
    high_rate = (len(high_both) / len(high_rows)) if high_rows else None
    low_rate = (len(low_attention) / len(low_rows)) if low_rows else None
    lift_unbounded = False
    lift = None
    if high_rows:
        if not low_rows:
            lift = None
        elif (low_rate or 0.0) <= 0.0:
            if (high_rate or 0.0) > 0.0:
                lift_unbounded = True
        else:
            lift = (high_rate or 0.0) / low_rate
    result = {
        "hypothesis_id": "H10",
        "evidence_source": "compact_memory",
        "h10_attention_target_definition": "high_attention is raw replay/contradiction attention; calibrated_high_attention is percentile-calibrated and reported separately.",
        "future_option_attention_link_count": len(rows),
        "future_option_event_count": future_option_event_count,
        "h10_blocked_by_h09": bool(future_option_event_count == 0),
        "live_future_option_delta_count": sum(1 for row in rows if str(row.get("source_label") or "") == "live"),
        "heuristic_future_option_delta_count": sum(1 for row in rows if str(row.get("source_label") or "") == "heuristic"),
        "null_future_option_delta_count": sum(1 for row in rows if float(row.get("option_delta_abs") or 0.0) <= 0.0 and int(row.get("high_option_change") or 0) == 0),
        "h10_live_rows_used": sum(1 for row in rows if str(row.get("source_label") or "") == "live"),
        "h10_heuristic_rows_used": sum(1 for row in rows if str(row.get("source_label") or "") == "heuristic"),
        "h10_fallback_reason": (
            "no_live_future_option_deltas"
            if rows and all(str(row.get("source_label") or "") != "live" for row in rows)
            else "no_live_high_option_change"
            if rows and any(str(row.get("source_label") or "") == "heuristic" for row in rows) and sum(
                1 for row in rows if str(row.get("source_label") or "") == "live" and int(row.get("high_option_change") or 0) == 1
            ) == 0
            else "all_live_option_deltas_zero"
            if rows and any(str(row.get("source_label") or "") == "heuristic" for row in rows) and sum(
                1 for row in rows if str(row.get("source_label") or "") == "live" and float(row.get("option_delta_abs") or 0.0) > 0.0
            ) == 0
            else None
        ),
        "high_option_change_count": len(high_rows),
        "high_option_change_source": (
            "none" if not rows else
            "live" if all(str(row.get("source_label") or "") == "live" for row in rows) else
            "heuristic" if all(str(row.get("source_label") or "") == "heuristic" for row in rows) else
            "mixed"
        ),
        "high_attention_count": len(high_attention_rows),
        "high_option_change_attention_count": len(high_both),
        "low_option_change_attention_count": len(low_attention),
        "high_option_change_attention_rate": high_rate,
        "low_option_change_attention_rate": low_rate,
        "option_attention_lift": lift,
        "option_attention_lift_unbounded": lift_unbounded,
        "mean_replay_priority_high_option_change": _mean([row.get("replay_priority_score") for row in high_rows]),
        "mean_replay_priority_low_option_change": _mean([row.get("replay_priority_score") for row in low_rows]),
        "mean_memory_priority_high_option_change": _mean([row.get("memory_priority_score") for row in high_rows]),
        "mean_memory_priority_low_option_change": _mean([row.get("memory_priority_score") for row in low_rows]),
        "replay_attention_count": sum(1 for row in rows if float(row.get("replay_priority_score") or 0.0) >= 0.50),
        "contradiction_attention_count": sum(1 for row in rows if float(row.get("contradiction_score") or 0.0) >= 0.50),
        "replay_or_contradiction_attention_count": len(high_attention_rows),
        "attention_threshold_method": rows[0].get("attention_threshold_method") if rows else None,
        "attention_calibration_degenerate": bool(rows) and any(int(row.get("attention_calibration_degenerate") or 0) == 1 for row in rows),
        "attention_all_high_saturation": bool(rows) and len(high_attention_rows) == len(rows),
        "attention_all_low_saturation": bool(rows) and len(high_attention_rows) == 0,
        "attention_saturation": bool(rows) and (len(high_attention_rows) == len(rows) or len(high_attention_rows) == 0),
        "replay_attention_saturation": bool(rows) and sum(1 for row in rows if float(row.get("replay_priority_score") or 0.0) >= 0.50) == len(rows),
        "contradiction_attention_saturation": bool(rows) and sum(1 for row in rows if float(row.get("contradiction_score") or 0.0) >= 0.50) == len(rows),
        "missing_evidence": [],
    }
    residual_rows: list[dict[str, Any]] = []
    residual_scores: list[float] = []
    for row in rows:
        residual_score = max(
            float(row.get("replay_priority_score") or 0.0),
            float(row.get("memory_priority_score") or 0.0),
        )
        enriched = dict(row)
        enriched["attention_residual_score"] = residual_score
        residual_rows.append(enriched)
        residual_scores.append(residual_score)
    residual_threshold = _percentile80(residual_scores)
    h10a = _subtest_result(rows, "replay_priority_score", 0.50)
    h10b = _subtest_result(rows, "memory_priority_score", 0.50)
    h10c = _subtest_result(residual_rows, "attention_residual_score", residual_threshold)
    h10d = _subtest_result(rows, "memory_priority_score", 0.70)
    result["h10_subtests"] = {
        "H10A_future_option_to_replay_priority": h10a,
        "H10B_future_option_to_memory_priority": h10b,
        "H10C_future_option_to_contradiction_independent_attention": h10c,
        "H10D_future_option_to_later_promotion_replay_survival": h10d,
    }
    result["residual_attention_percentile_threshold"] = residual_threshold
    if future_option_event_count == 0:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append("H10 blocked because H09 future-option events are absent.")
    elif not rows:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
    elif not high_rows:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
    elif result["attention_all_high_saturation"] and (high_rate or 0.0) == 1.0 and (low_rate or 0.0) == 1.0:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append(
            "Attention signal is saturated all-high across high and low future-option-change interactions; selective attention is not demonstrated."
        )
    elif result["attention_all_low_saturation"]:
        result["decision"] = "INSUFFICIENT_EVIDENCE" if low_rows else "PARTIALLY_VALID"
        result["missing_evidence"].append(
            "Attention signal is saturated all-low; selective attention is not demonstrated."
        )
    elif result["attention_calibration_degenerate"]:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"].append(
            "Attention calibration is degenerate because attention scores do not separate the evaluated interactions."
        )
    elif (
        result["option_attention_lift_unbounded"] is True
        and len(high_rows) >= 5
        and len(high_attention_rows) >= 5
        and not result["attention_saturation"]
    ):
        result["decision"] = "VALID"
    elif len(high_both) > 0 and lift is None:
        result["decision"] = "PARTIALLY_VALID"
    elif (
        len(high_rows) >= 5
        and len(high_attention_rows) >= 5
        and lift is not None
        and lift >= 1.25
        and (high_rate or 0.0) > (low_rate or 0.0)
        and not result["attention_saturation"]
    ):
        result["decision"] = "VALID"
    elif (
        len(high_rows) >= 5
        and len(high_attention_rows) >= 5
        and lift is not None
        and lift <= 1.0
    ):
        sufficient_subtests = [
            payload for payload in (h10a, h10b, h10c, h10d)
            if payload["coverage"] >= 5 and not payload["saturation"]
        ]
        if sufficient_subtests and all((item["lift"] or 0.0) <= 1.0 for item in sufficient_subtests):
            result["decision"] = "INVALID"
        else:
            result["decision"] = "INSUFFICIENT_EVIDENCE"
    else:
        result["decision"] = "PARTIALLY_VALID"
    result["core_metrics"] = {
        key: result.get(key)
        for key in (
            "future_option_attention_link_count",
            "future_option_event_count",
            "h10_blocked_by_h09",
            "h10_attention_target_definition",
            "live_future_option_delta_count",
            "heuristic_future_option_delta_count",
            "null_future_option_delta_count",
            "high_option_change_count",
            "high_option_change_source",
            "high_attention_count",
            "high_option_change_attention_count",
            "low_option_change_attention_count",
            "high_option_change_attention_rate",
            "low_option_change_attention_rate",
            "option_attention_lift",
            "option_attention_lift_unbounded",
            "replay_attention_count",
            "contradiction_attention_count",
            "replay_or_contradiction_attention_count",
            "attention_threshold_method",
            "residual_attention_percentile_threshold",
            "attention_calibration_degenerate",
            "attention_all_high_saturation",
            "attention_all_low_saturation",
            "attention_saturation",
            "replay_attention_saturation",
            "contradiction_attention_saturation",
            "mean_replay_priority_high_option_change",
            "mean_replay_priority_low_option_change",
            "mean_memory_priority_high_option_change",
            "mean_memory_priority_low_option_change",
            "h10_subtests",
        )
    }
    _write(output_dir, result)
    return result


def _mean(values: list[Any]) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None


def _percentile80(values: list[float]) -> float:
    cooked = sorted(float(value) for value in values)
    if not cooked:
        return 1.0
    index = max(0, min(len(cooked) - 1, int(round(0.80 * (len(cooked) - 1)))))
    threshold = float(cooked[index])
    return threshold if threshold > 0.0 else 1.0


def _write(output_dir: Path, result: dict[str, Any]) -> None:
    subtests = result.get("h10_subtests") or {}
    (output_dir / "h10_future_option_attention_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H10 decision: {result.get('decision')}\n"
        f"attention target: {result.get('h10_attention_target_definition')}\n"
        f"live future-option deltas: {result.get('live_future_option_delta_count')}\n"
        f"heuristic future-option deltas: {result.get('heuristic_future_option_delta_count')}\n"
        f"H10 live rows used: {result.get('h10_live_rows_used')}\n"
        f"H10 heuristic rows used: {result.get('h10_heuristic_rows_used')}\n"
        f"H10 fallback reason: {result.get('h10_fallback_reason')}\n"
        f"null future-option deltas: {result.get('null_future_option_delta_count')}\n"
        f"high-option-change source: {result.get('high_option_change_source')}\n"
        f"option-attention lift: {result.get('option_attention_lift')}\n"
        f"high-option-change attention rate: {result.get('high_option_change_attention_rate')}\n"
        f"low-option-change attention rate: {result.get('low_option_change_attention_rate')}\n"
        f"attention threshold method: {result.get('attention_threshold_method')}\n"
        f"attention calibration degenerate: {result.get('attention_calibration_degenerate')}\n"
        f"attention all-high saturation: {result.get('attention_all_high_saturation')}\n"
        f"attention all-low saturation: {result.get('attention_all_low_saturation')}\n"
        f"attention saturation: {result.get('attention_saturation')}\n"
        f"replay attention count: {result.get('replay_attention_count')}\n"
        f"replay attention saturation: {result.get('replay_attention_saturation')}\n"
        f"contradiction attention count: {result.get('contradiction_attention_count')}\n"
        f"H10A: coverage={subtests.get('H10A_future_option_to_replay_priority', {}).get('coverage')} high_rate={subtests.get('H10A_future_option_to_replay_priority', {}).get('high_rate')} low_rate={subtests.get('H10A_future_option_to_replay_priority', {}).get('low_rate')} lift={subtests.get('H10A_future_option_to_replay_priority', {}).get('lift')} saturation={subtests.get('H10A_future_option_to_replay_priority', {}).get('saturation')} threshold={subtests.get('H10A_future_option_to_replay_priority', {}).get('threshold')}\n"
        f"H10B: coverage={subtests.get('H10B_future_option_to_memory_priority', {}).get('coverage')} high_rate={subtests.get('H10B_future_option_to_memory_priority', {}).get('high_rate')} low_rate={subtests.get('H10B_future_option_to_memory_priority', {}).get('low_rate')} lift={subtests.get('H10B_future_option_to_memory_priority', {}).get('lift')} saturation={subtests.get('H10B_future_option_to_memory_priority', {}).get('saturation')} threshold={subtests.get('H10B_future_option_to_memory_priority', {}).get('threshold')}\n"
        f"H10C: coverage={subtests.get('H10C_future_option_to_contradiction_independent_attention', {}).get('coverage')} high_rate={subtests.get('H10C_future_option_to_contradiction_independent_attention', {}).get('high_rate')} low_rate={subtests.get('H10C_future_option_to_contradiction_independent_attention', {}).get('low_rate')} lift={subtests.get('H10C_future_option_to_contradiction_independent_attention', {}).get('lift')} saturation={subtests.get('H10C_future_option_to_contradiction_independent_attention', {}).get('saturation')} threshold={subtests.get('H10C_future_option_to_contradiction_independent_attention', {}).get('threshold')}\n"
        f"H10D: coverage={subtests.get('H10D_future_option_to_later_promotion_replay_survival', {}).get('coverage')} high_rate={subtests.get('H10D_future_option_to_later_promotion_replay_survival', {}).get('high_rate')} low_rate={subtests.get('H10D_future_option_to_later_promotion_replay_survival', {}).get('low_rate')} lift={subtests.get('H10D_future_option_to_later_promotion_replay_survival', {}).get('lift')} saturation={subtests.get('H10D_future_option_to_later_promotion_replay_survival', {}).get('saturation')} threshold={subtests.get('H10D_future_option_to_later_promotion_replay_survival', {}).get('threshold')}\n"
    )
    (output_dir / "h10_future_option_attention_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h10_future_option_attention.md").write_text("```\n" + text + "```\n", encoding="utf-8")
