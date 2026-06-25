from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence
from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention
from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation
from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence


SUITE_JSON_NAME = "hypothesis_suite_summary.json"
SUITE_TXT_NAME = "hypothesis_suite_summary.txt"
SUITE_MD_NAME = "hypothesis_suite_summary.md"
INPUT_REPORT_NAME = "interaction_sampling_v05c_report.json"


def run_hypothesis_suite_report(
    *,
    run_dir: Path,
    memory_dir: Path | None = None,
    output_dir: Path,
    scan_all_dbs: bool,
    max_db_files: int,
    max_rows: int,
    epoch_id: str | None = None,
    global_step_start: int | None = None,
    global_step_end: int | None = None,
    interactions_this_epoch: int | None = None,
    total_interactions_seen: int | None = None,
    memory_size_before_bytes: int | None = None,
    memory_size_after_bytes: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    memory_dir = None if memory_dir is None else Path(memory_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h01_dir = output_dir / "h01"
    h02_dir = output_dir / "h02"
    h03_dir = output_dir / "h03"
    h01 = evaluate_h01_contingency_emergence(run_dir=run_dir, output_dir=h01_dir, memory_dir=memory_dir)
    h02 = evaluate_h02_prediction_violation_attention(
        run_dir=run_dir,
        output_dir=h02_dir,
        memory_dir=memory_dir,
        max_rows=int(max_rows),
        max_db_files=int(max_db_files),
        scan_all_dbs=bool(scan_all_dbs),
    )
    h03 = evaluate_h03_transformation_family_formation(
        run_dir=run_dir,
        output_dir=h03_dir,
        memory_dir=memory_dir,
        max_db_files=int(max_db_files),
        max_rows=int(max_rows),
        scan_all_dbs=bool(scan_all_dbs),
    )
    h04 = (
        evaluate_h04_carrier_emergence(memory_dir=memory_dir, run_dir=run_dir, output_dir=output_dir / "h04")
        if memory_dir is not None
        else {"hypothesis_id": "H04", "decision": "NOT_IMPLEMENTED", "core_metrics": {}, "missing_evidence": ["memory_dir not provided"]}
    )
    summary = build_hypothesis_suite_summary(
        run_dir=run_dir,
        memory_dir=memory_dir,
        h01=h01,
        h02=h02,
        h03=h03,
        h04=h04,
        epoch_id=epoch_id,
        global_step_start=global_step_start,
        global_step_end=global_step_end,
        interactions_this_epoch=interactions_this_epoch,
        total_interactions_seen=total_interactions_seen,
        memory_size_before_bytes=memory_size_before_bytes,
        memory_size_after_bytes=memory_size_after_bytes,
    )
    _write_suite_summary(summary, output_dir)
    return summary


def build_hypothesis_suite_summary(
    *,
    run_dir: Path,
    memory_dir: Path | None = None,
    h01: dict[str, Any],
    h02: dict[str, Any],
    h03: dict[str, Any],
    h04: dict[str, Any] | None = None,
    epoch_id: str | None = None,
    global_step_start: int | None = None,
    global_step_end: int | None = None,
    interactions_this_epoch: int | None = None,
    total_interactions_seen: int | None = None,
    memory_size_before_bytes: int | None = None,
    memory_size_after_bytes: int | None = None,
) -> dict[str, Any]:
    input_report = _load_json(Path(run_dir) / INPUT_REPORT_NAME) or {}
    runs = [dict(item) for item in input_report.get("runs", []) if isinstance(item, dict)]
    temporal_rows = list(dict(item) for item in ((input_report.get("temporal_milestones") or {}).get("by_game_sampler_seed", []) or []) if isinstance(item, dict))
    games = sorted({str(row.get("game")) for row in runs if row.get("game")})
    if not games:
        games = [str(item) for item in input_report.get("games", []) if item]
    samplers = sorted({str(row.get("sampler_name")) for row in runs if row.get("sampler_name")})
    if not samplers:
        samplers = [str(item) for item in input_report.get("samplers", []) if item]
    seeds = sorted({int(item.get("seed")) for item in temporal_rows if item.get("seed") is not None})
    if not seeds:
        seeds = [int(item) for item in input_report.get("seeds", []) if item is not None]

    per_game = _per_game_diagnostics(runs)
    per_sampler = _per_sampler_diagnostics(runs)
    temporal = _temporal_order_diagnostics(temporal_rows)
    total_interactions = int(sum(int(row.get("total_interactions", 0) or 0) for row in runs))

    missing_evidence = _merge_unique(
        list(h01.get("missing_evidence", [])),
        list(h02.get("missing_evidence", [])),
        list(h03.get("missing_evidence", [])),
        list((h04 or {}).get("missing_evidence", [])),
    )
    h04 = h04 or {"decision": "NOT_IMPLEMENTED", "core_metrics": {}}
    summary = {
        "epoch_id": epoch_id,
        "global_step_start": global_step_start,
        "global_step_end": global_step_end,
        "source_run_dir": str(run_dir),
        "memory_dir": None if memory_dir is None else str(memory_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_count": len(games),
        "sampler_count": len(samplers),
        "seed_count": len(seeds),
        "total_interactions": total_interactions,
        "interactions_this_epoch": interactions_this_epoch,
        "total_interactions_seen": total_interactions_seen,
        "memory_size_before_bytes": memory_size_before_bytes,
        "memory_size_after_bytes": memory_size_after_bytes,
        "H01 decision": h01.get("decision"),
        "H02 decision": h02.get("decision"),
        "H03 decision": h03.get("decision"),
        "H04 decision": h04.get("decision", "NOT_IMPLEMENTED"),
        "H01 core metrics": {
            "stable_contingency_count": h01.get("stable_contingency_count"),
            "interaction_count": h01.get("total_interaction_count"),
            "mean_prediction_accuracy": h01.get("mean_prediction_accuracy"),
            "games_with_stable_contingencies": _count_positive(h01.get("per_game_contingency_counts")),
            "samplers_with_stable_contingencies": _count_positive(h01.get("per_sampler_contingency_counts")),
        },
        "H02 core metrics": {
            "prediction_violation_replay_lift": h02.get("prediction_violation_replay_lift"),
            "prediction_violation_base_ratio": h02.get("prediction_violation_base_ratio"),
            "high_priority_replay_prediction_violation_ratio": h02.get("high_priority_replay_prediction_violation_ratio"),
            "mean_replay_priority_for_prediction_violating_interactions": h02.get("mean_replay_priority_for_prediction_violating_interactions"),
            "mean_replay_priority_for_non_prediction_violating_interactions": h02.get("mean_replay_priority_for_non_prediction_violating_interactions"),
            "direct_replay_lift_available": h02.get("direct_replay_lift_available"),
        },
        "H03 core metrics": {
            "transformation_family_count": h03.get("transformation_family_count"),
            "compression_ratio": h03.get("compression_ratio"),
            "compression_gain": h03.get("compression_gain"),
            "singleton_family_ratio": h03.get("singleton_family_ratio"),
            "family_cross_game_count": h03.get("family_cross_game_count"),
            "family_cross_sampler_count": h03.get("family_cross_sampler_count"),
            "family_cross_context_count": h03.get("family_cross_context_count"),
            "relaxed_singleton_family_ratio": h03.get("relaxed_singleton_family_ratio"),
            "merge_safety_passed": h03.get("merge_safety_passed"),
        },
        "H04 core metrics": dict(h04.get("core_metrics", {})),
        "per_game_status_table": per_game,
        "per-game status table": per_game,
        "per_sampler_status_table": per_sampler,
        "per-sampler status table": per_sampler,
        "temporal_order_diagnostics": temporal,
        "missing_evidence": missing_evidence,
        "missing evidence": missing_evidence,
        "next_recommended_action": _next_recommended_action(h01, h02, h03, missing_evidence),
        "next recommended action": _next_recommended_action(h01, h02, h03, missing_evidence),
    }
    return summary


def _per_game_diagnostics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        game = str(row.get("game") or "")
        if not game:
            continue
        by_game.setdefault(game, []).append(row)
    output: list[dict[str, Any]] = []
    for game, items in sorted(by_game.items()):
        interaction_count = int(sum(int(row.get("total_interactions", 0) or 0) for row in items))
        stable_contingency_count = int(sum(int(row.get("stable_contingency_count", 0) or 0) for row in items))
        mean_prediction_accuracy = _mean([row.get("prediction_accuracy") for row in items])
        transformation_family_count = int(sum(int(row.get("unique_transformation_families", 0) or 0) for row in items))
        prediction_violation_replay_lift = None
        compression_ratio = None
        singleton_family_ratio = None
        h01_signal = stable_contingency_count > 0
        h02_signal = any(
            (row.get("mean_isf_prediction_error") or 0.0) > 0.0 and int(row.get("high_priority_replay_count", 0) or 0) > 0
            for row in items
        )
        h03_signal = transformation_family_count > 0
        if interaction_count <= 0:
            status = "missing"
        elif h01_signal and h02_signal and h03_signal:
            status = "supported"
        elif h01_signal:
            status = "partial"
        elif interaction_count > 0 and stable_contingency_count <= 1:
            status = "weak"
        else:
            status = "failed"
        output.append(
            {
                "game": game,
                "interaction_count": interaction_count,
                "stable_contingency_count": stable_contingency_count,
                "mean_prediction_accuracy": mean_prediction_accuracy,
                "prediction_violation_replay_lift": prediction_violation_replay_lift,
                "transformation_family_count": transformation_family_count,
                "compression_ratio": compression_ratio,
                "singleton_family_ratio": singleton_family_ratio,
                "status": status,
            }
        )
    return output


def _per_sampler_diagnostics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sampler: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        sampler = str(row.get("sampler_name") or "")
        if not sampler:
            continue
        by_sampler.setdefault(sampler, []).append(row)
    output: list[dict[str, Any]] = []
    for sampler, items in sorted(by_sampler.items()):
        stable_contingency_count = int(sum(int(row.get("stable_contingency_count", 0) or 0) for row in items))
        transformation_family_count = int(sum(int(row.get("unique_transformation_families", 0) or 0) for row in items))
        replay_pressure = int(sum(int(row.get("high_priority_replay_count", 0) or 0) for row in items))
        if stable_contingency_count > 0 and transformation_family_count > 0 and replay_pressure > 0:
            status = "supported"
        elif stable_contingency_count > 0:
            status = "partial"
        elif items:
            status = "weak"
        else:
            status = "missing"
        output.append(
            {
                "sampler": sampler,
                "interaction_count": int(sum(int(row.get("total_interactions", 0) or 0) for row in items)),
                "stable_contingency_count": stable_contingency_count,
                "transformation_family_count": transformation_family_count,
                "high_priority_replay_count": replay_pressure,
                "status": status,
            }
        )
    return output


def _temporal_order_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    h01_before_h03_values: list[bool] = []
    h02_before_h03_values: list[bool] = []
    h03_before_h04_values: list[bool] = []
    missing_count = 0
    cases_available = 0
    diagnostics_rows: list[dict[str, Any]] = []
    for row in rows:
        h01_before_h03 = _ordered_bool(row.get("first_stable_contingency_step"), row.get("first_transformation_family_step"))
        h02_before_h03 = _ordered_bool(row.get("first_prediction_violation_step"), row.get("first_transformation_family_step"))
        h03_before_h04 = _ordered_bool(row.get("first_stable_transformation_family_step"), row.get("first_emergent_carrier_step"))
        diagnostics_rows.append(
            {
                "game": row.get("game"),
                "sampler": row.get("sampler"),
                "seed": row.get("seed"),
                "h01_before_h03": h01_before_h03,
                "h02_before_h03": h02_before_h03,
                "h03_before_h04": h03_before_h04,
            }
        )
        local_values = [h01_before_h03, h02_before_h03, h03_before_h04]
        if any(value is not None for value in local_values):
            cases_available += 1
        missing_count += sum(1 for value in local_values if value is None)
        if h01_before_h03 is not None:
            h01_before_h03_values.append(h01_before_h03)
        if h02_before_h03 is not None:
            h02_before_h03_values.append(h02_before_h03)
        if h03_before_h04 is not None:
            h03_before_h04_values.append(h03_before_h04)
    return {
        "per_case": diagnostics_rows,
        "temporal_order_cases_available": cases_available,
        "h01_before_h03_ratio": _true_ratio(h01_before_h03_values),
        "h02_before_h03_ratio": _true_ratio(h02_before_h03_values),
        "h03_before_h04_ratio": _true_ratio(h03_before_h04_values),
        "temporal_order_missing_count": missing_count,
    }


def _ordered_bool(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return int(left) <= int(right)


def _true_ratio(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


def _next_recommended_action(h01: dict[str, Any], h02: dict[str, Any], h03: dict[str, Any], missing_evidence: list[str]) -> str:
    decisions = {str(h01.get("decision")), str(h02.get("decision")), str(h03.get("decision"))}
    if "INVALID" in decisions:
        return "Inspect invalidated hypothesis outputs and repair the shared sampling configuration before H04 work."
    if "INCONCLUSIVE" in decisions or missing_evidence:
        return "Keep the shared broad run, fill the missing evidence paths, and rerun the suite summary before H04 analysis."
    if "PARTIALLY_VALID" in decisions:
        return "Use this shared dataset for targeted follow-up diagnostics, then evaluate H04 carrier emergence on the same run artifacts."
    return "Proceed to H04 carrier-emergence analysis using this shared interaction-memory dataset."


def _write_suite_summary(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / SUITE_JSON_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / SUITE_TXT_NAME).write_text(_format_text(summary), encoding="utf-8")
    (output_dir / SUITE_MD_NAME).write_text(_format_md(summary), encoding="utf-8")


def _format_text(summary: dict[str, Any]) -> str:
    lines = [
        "Hypothesis Suite Summary",
        f"source_run_dir: {summary['source_run_dir']}",
        f"H01: {summary['H01 decision']}",
        f"H02: {summary['H02 decision']}",
        f"H03: {summary['H03 decision']}",
        f"games: {summary['game_count']} samplers: {summary['sampler_count']} seeds: {summary['seed_count']}",
        f"total_interactions: {summary['total_interactions']}",
        f"next_recommended_action: {summary['next_recommended_action']}",
    ]
    return "\n".join(lines) + "\n"


def _format_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Hypothesis Suite Summary",
        "",
        f"- source run: `{summary['source_run_dir']}`",
        f"- H01: `{summary['H01 decision']}`",
        f"- H02: `{summary['H02 decision']}`",
        f"- H03: `{summary['H03 decision']}`",
        f"- total interactions: `{summary['total_interactions']}`",
        "",
        "## Next Action",
        "",
        summary["next_recommended_action"],
    ]
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[Any]) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return float(sum(items) / len(items))


def _count_positive(mapping: Any) -> int | None:
    if not isinstance(mapping, dict):
        return None
    return sum(1 for value in mapping.values() if int(value or 0) > 0)


def _merge_unique(*groups: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
    return output
