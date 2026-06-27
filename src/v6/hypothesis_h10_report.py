from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.higher_order_substrate import derive_higher_order_memory
from v6.future_options import derive_future_option_memory
from v6.memory.compact_memory import ensure_memory_layout


def evaluate_h10_future_option_attention(
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
        derive_future_option_memory(memory_dir=memory_dir, run_dir=run_dir)
    with sqlite3.connect(Path(memory_dir) / "current_state.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM future_option_attention_links ORDER BY event_id ASC").fetchall()]
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
        "h10_attention_target_definition": "high_attention is based on replay_priority_score or contradiction_score only; derived memory_priority_score is diagnostic and is not used for validation.",
        "future_option_attention_link_count": len(rows),
        "live_future_option_delta_count": sum(1 for row in rows if str(row.get("owner_type")) == "interaction"),
        "heuristic_future_option_delta_count": sum(1 for row in rows if str(row.get("owner_type")) != "interaction"),
        "null_future_option_delta_count": sum(1 for row in rows if float(row.get("option_delta_abs") or 0.0) <= 0.0 and int(row.get("high_option_change") or 0) == 0),
        "high_option_change_count": len(high_rows),
        "high_option_change_source": (
            "none" if not rows else
            "live" if all(str(row.get("owner_type")) == "interaction" for row in rows) else
            "heuristic" if all(str(row.get("owner_type")) != "interaction" for row in rows) else
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
        "attention_saturation": bool(rows) and len(high_attention_rows) == len(rows),
        "replay_attention_saturation": bool(rows) and sum(1 for row in rows if float(row.get("replay_priority_score") or 0.0) >= 0.50) == len(rows),
        "contradiction_attention_saturation": bool(rows) and sum(1 for row in rows if float(row.get("contradiction_score") or 0.0) >= 0.50) == len(rows),
        "missing_evidence": [],
    }
    if not rows:
        result["decision"] = "INCONCLUSIVE"
    elif not high_rows:
        result["decision"] = "INCONCLUSIVE"
    elif result["attention_saturation"] and (high_rate or 0.0) == 1.0 and (low_rate or 0.0) == 1.0:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"].append(
            "Attention signal is saturated across high and low future-option-change interactions; selective attention is not demonstrated."
        )
    elif len(high_both) > 0 and lift is None:
        result["decision"] = "PARTIALLY_VALID"
    elif (
        len(high_rows) >= 5
        and len(high_attention_rows) >= 5
        and lift is not None
        and lift >= 1.25
        and (high_rate or 0.0) > (low_rate or 0.0)
    ):
        result["decision"] = "VALID"
    elif (
        len(high_rows) >= 5
        and len(high_attention_rows) >= 5
        and lift is not None
        and lift <= 1.0
    ):
        result["decision"] = "INVALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    result["core_metrics"] = {
        key: result.get(key)
        for key in (
            "future_option_attention_link_count",
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
            "attention_saturation",
            "replay_attention_saturation",
            "contradiction_attention_saturation",
            "mean_replay_priority_high_option_change",
            "mean_replay_priority_low_option_change",
            "mean_memory_priority_high_option_change",
            "mean_memory_priority_low_option_change",
        )
    }
    _write(output_dir, result)
    return result


def _mean(values: list[Any]) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None


def _write(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h10_future_option_attention_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H10 decision: {result.get('decision')}\n"
        f"attention target: {result.get('h10_attention_target_definition')}\n"
        f"live future-option deltas: {result.get('live_future_option_delta_count')}\n"
        f"heuristic future-option deltas: {result.get('heuristic_future_option_delta_count')}\n"
        f"null future-option deltas: {result.get('null_future_option_delta_count')}\n"
        f"high-option-change source: {result.get('high_option_change_source')}\n"
        f"option-attention lift: {result.get('option_attention_lift')}\n"
        f"high-option-change attention rate: {result.get('high_option_change_attention_rate')}\n"
        f"low-option-change attention rate: {result.get('low_option_change_attention_rate')}\n"
        f"attention saturation: {result.get('attention_saturation')}\n"
        f"replay attention count: {result.get('replay_attention_count')}\n"
        f"contradiction attention count: {result.get('contradiction_attention_count')}\n"
    )
    (output_dir / "h10_future_option_attention_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h10_future_option_attention.md").write_text("```\n" + text + "```\n", encoding="utf-8")
