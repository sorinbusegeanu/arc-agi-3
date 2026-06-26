from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from v6.higher_order_substrate import derive_higher_order_memory
from v6.future_options import derive_future_option_memory
from v6.memory.compact_memory import ensure_memory_layout


def evaluate_h09_future_option_motifs(
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
        events = [dict(row) for row in conn.execute("SELECT * FROM future_option_events ORDER BY event_id ASC").fetchall()]
        motifs = [dict(row) for row in conn.execute("SELECT * FROM future_option_motifs ORDER BY motif_signature ASC").fetchall()]
        milestone_map = dict(conn.execute("SELECT milestone_name, first_global_step FROM higher_order_milestones").fetchall())
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
        "cross_context_motif_count": cross_context_motif_count,
        "cross_game_motif_count": cross_game_motif_count,
        "mean_abs_option_delta": mean_abs_option_delta,
        "max_abs_option_delta": max_abs_option_delta,
        "mean_motif_stability_score": mean_motif_stability_score,
        "first_future_option_event_step": milestone_map.get("first_future_option_event_step"),
        "first_emergent_future_option_motif_step": milestone_map.get("first_emergent_future_option_motif_step"),
        "missing_evidence": [],
    }
    non_unknown_types = [key for key, value in motif_type_counts.items() if key != "unknown" and value > 0]
    if not events:
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
    ):
        result["decision"] = "VALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    result["core_metrics"] = {
        key: result.get(key)
        for key in (
            "future_option_event_count",
            "future_option_motif_count",
            "emergent_future_option_motif_count",
            "motif_type_counts",
            "cross_context_motif_count",
            "cross_game_motif_count",
            "mean_abs_option_delta",
            "max_abs_option_delta",
            "mean_motif_stability_score",
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
    )
    (output_dir / "h09_future_option_motifs_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h09_future_option_motifs.md").write_text("```\n" + text + "```\n", encoding="utf-8")
