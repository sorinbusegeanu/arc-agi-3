from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from v6.memory.compact_memory import configure_compact_sqlite_connection, ensure_memory_layout
from v6.memory.direct_streaming_fold import direct_streaming_manifest_exists
from v6.memory.trajectory_efficiency import save_best_known_solution_lengths


def evaluate_h12_efficiency_emergence(
    *,
    run_dir: Path,
    memory_dir: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, evidence_source, reconstruction = _load_trajectory_rows(run_dir=run_dir, memory_dir=memory_dir)
    rows = [dict(row) for row in rows]
    _write_parquet(output_dir / "h12_trajectory_metrics.parquet", rows)
    efficiency_root = ((memory_dir.parent / "efficiency") if memory_dir is not None else (run_dir / "efficiency"))
    efficiency_root.mkdir(parents=True, exist_ok=True)
    best_known_path = efficiency_root / "best_known_solution_lengths.json"
    state_path = efficiency_root / "trajectory_efficiency_state.json"
    previous_state = _load_json(state_path) or {}
    previous_epoch_mean = previous_state.get("mean_normalized_solve_efficiency")
    best_known_map: dict[str, int] = {}
    for row in rows:
        if int(row.get("success") or 0) != 1:
            continue
        game_id = str(row.get("game_id") or "unknown_game")
        level_id = str(row.get("level_id") or "__none__")
        best = row.get("best_known_solution_length")
        if best is None:
            continue
        best_known_map[f"{game_id}|{level_id}"] = int(best)
    save_best_known_solution_lengths(best_known_path, best_known_map)
    successful_rows = [row for row in rows if int(row.get("success") or 0) == 1]
    active_rows = [row for row in rows if int(row.get("efficiency_active") or 0) == 1]
    comparable_groups = len({str(row.get("comparable_outcome_group_id") or "") for row in active_rows if row.get("comparable_outcome_group_id")})
    memory_bonus_rows = [row for row in rows if float(row.get("efficiency_memory_bonus") or 0.0) > 0.0]
    replay_bonus_rows = [row for row in rows if float(row.get("efficiency_replay_bonus") or 0.0) > 0.0]
    current_mean_norm = _mean(row.get("normalized_solve_efficiency") for row in successful_rows)
    result = {
        "hypothesis_id": "H12",
        "evidence_source": evidence_source,
        "games": sorted({str(row.get("game_id")) for row in rows if row.get("game_id")}),
        "levels": sorted({str(row.get("level_id")) for row in rows if row.get("level_id")}),
        "successful_trajectories": len(successful_rows),
        "comparable_trajectory_groups": comparable_groups,
        "efficiency_active_trajectory_count": len(active_rows),
        "trajectories_with_memory_bonus": len(memory_bonus_rows),
        "trajectories_with_replay_bonus": len(replay_bonus_rows),
        "mean_efficiency_memory_bonus": _mean(row.get("efficiency_memory_bonus") for row in rows),
        "mean_efficiency_replay_bonus": _mean(row.get("efficiency_replay_bonus") for row in rows),
        "median_steps_to_success": _median(row.get("steps_to_success") for row in successful_rows),
        "best_known_solution_count": len(best_known_map),
        "best_known_solution_improvements": _count_improvements(rows),
        "mean_normalized_solve_efficiency": current_mean_norm,
        "median_normalized_solve_efficiency": _median(row.get("normalized_solve_efficiency") for row in successful_rows),
        "mean_equivalent_outcome_cost_gap": _mean(row.get("equivalent_outcome_cost_gap") for row in successful_rows),
        "mean_loop_ratio": _mean(row.get("loop_ratio") for row in rows),
        "mean_repeated_state_ratio": _mean(row.get("repeated_state_ratio") for row in rows),
        "mean_blocked_action_ratio": _mean(row.get("blocked_action_ratio") for row in rows),
        "efficiency_replay_priority_correlation": _correlation(rows, "trajectory_efficiency_score", "efficiency_replay_bonus"),
        "efficiency_memory_fitness_correlation": _correlation(rows, "trajectory_efficiency_score", "efficiency_memory_bonus"),
        "cost_gap_replay_priority_correlation": _correlation(successful_rows, "equivalent_outcome_cost_gap", "efficiency_replay_bonus"),
        "future_option_gain_per_action_correlation": _correlation(rows, "future_option_gain_per_action", "trajectory_efficiency_score"),
        "efficiency_improved_vs_previous_epoch": (
            None
            if previous_epoch_mean is None or current_mean_norm is None
            else float(current_mean_norm) > float(previous_epoch_mean)
        ),
        "trajectory_reconstruction_diagnostics": reconstruction,
        "missing_evidence": [],
    }
    if len(successful_rows) < 2 or comparable_groups <= 0:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["missing_evidence"].append("Too few successful or comparable trajectories are available for H12.")
        for key, message in (
            ("no_success_events", "No success events were recorded."),
            ("no_terminal_events", "No terminal events were recorded."),
            ("no_episode_boundaries", "No episode boundaries were recoverable."),
            ("missing_state_hashes", "State hashes are missing for trajectory reconstruction."),
            ("missing_compact_trajectory_records", "Compact trajectory-efficiency records are missing."),
        ):
            if reconstruction.get(key):
                result["missing_evidence"].append(message)
    elif result["trajectories_with_memory_bonus"] <= 0 and result["trajectories_with_replay_bonus"] <= 0:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"].append("Trajectory efficiency is computed but not yet linked strongly to memory or replay bonuses.")
    elif (result["efficiency_memory_fitness_correlation"] or 0.0) < 0.0 or (result["efficiency_replay_priority_correlation"] or 0.0) < 0.0:
        result["decision"] = "INVALID"
        result["missing_evidence"].append("Lower-cost equivalent trajectories are not receiving stronger memory or replay preference.")
    elif result["efficiency_improved_vs_previous_epoch"] and (result["efficiency_memory_fitness_correlation"] or 0.0) >= 0.0:
        result["decision"] = "VALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    state_payload = {
        "mean_normalized_solve_efficiency": result.get("mean_normalized_solve_efficiency"),
        "best_known_solution_count": result.get("best_known_solution_count"),
        "last_decision": result.get("decision"),
    }
    state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
    result["core_metrics"] = {
        key: result.get(key)
        for key in (
            "successful_trajectories",
            "comparable_trajectory_groups",
            "efficiency_active_trajectory_count",
            "trajectories_with_memory_bonus",
            "trajectories_with_replay_bonus",
            "mean_efficiency_memory_bonus",
            "mean_efficiency_replay_bonus",
            "median_steps_to_success",
            "best_known_solution_count",
            "best_known_solution_improvements",
            "mean_normalized_solve_efficiency",
            "median_normalized_solve_efficiency",
            "mean_equivalent_outcome_cost_gap",
            "mean_loop_ratio",
            "mean_repeated_state_ratio",
            "mean_blocked_action_ratio",
            "efficiency_replay_priority_correlation",
            "efficiency_memory_fitness_correlation",
            "cost_gap_replay_priority_correlation",
            "future_option_gain_per_action_correlation",
            "efficiency_improved_vs_previous_epoch",
        )
    }
    _write_report(output_dir, result)
    return result


def _load_trajectory_rows(*, run_dir: Path, memory_dir: Path | None) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    sqlite_paths = sorted(run_dir.rglob("*.sqlite"))
    rows: list[dict[str, Any]] = []
    diagnostics = {
        "no_success_events": False,
        "no_terminal_events": False,
        "no_episode_boundaries": False,
        "missing_state_hashes": False,
        "missing_compact_trajectory_records": False,
    }
    saw_interactions = False
    for db_path in sqlite_paths:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            tables = {item[0] for item in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "interactions" in tables:
                saw_interactions = True
                try:
                    row = conn.execute(
                        """
                        SELECT
                            SUM(CASE WHEN outcome_state IN ('WIN', 'GAME_OVER') OR COALESCE(level_completed_event, 0) = 1 THEN 1 ELSE 0 END),
                            SUM(CASE WHEN episode_id IS NOT NULL THEN 1 ELSE 0 END),
                            SUM(CASE WHEN state_hash_before IS NOT NULL AND state_hash_after IS NOT NULL THEN 1 ELSE 0 END)
                        FROM interactions
                        """
                    ).fetchone()
                    success_terminal = int(row[0] or 0)
                    episode_boundaries = int(row[1] or 0)
                    hashed = int(row[2] or 0)
                    diagnostics["no_terminal_events"] = diagnostics["no_terminal_events"] or success_terminal <= 0
                    diagnostics["no_episode_boundaries"] = diagnostics["no_episode_boundaries"] or episode_boundaries <= 0
                    diagnostics["missing_state_hashes"] = diagnostics["missing_state_hashes"] or hashed <= 0
                except sqlite3.DatabaseError:
                    pass
            if "trajectory_efficiency" not in tables:
                continue
            rows.extend(dict(row) for row in conn.execute("SELECT * FROM trajectory_efficiency ORDER BY trajectory_id ASC").fetchall())
    if rows:
        diagnostics["no_success_events"] = all(int(row.get("success") or 0) != 1 for row in rows)
        return rows, "raw_epoch_db", diagnostics
    if memory_dir is None:
        diagnostics["missing_compact_trajectory_records"] = True
        return [], "none", diagnostics
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(Path(memory_dir) / "current_state.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        configure_compact_sqlite_connection(conn, write=False)
        tables = {item[0] for item in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "trajectory_efficiency" in tables:
            rows = [dict(row) for row in conn.execute("SELECT * FROM trajectory_efficiency ORDER BY trajectory_id ASC").fetchall()]
            if rows:
                diagnostics["no_success_events"] = all(int(row.get("success") or 0) != 1 for row in rows)
                if direct_streaming_manifest_exists(memory_dir) and not sqlite_paths:
                    return rows, "direct_streaming_manifest_and_compact_memory", diagnostics
                return rows, "compact_memory", diagnostics
    diagnostics["missing_compact_trajectory_records"] = True
    if not saw_interactions:
        diagnostics["no_episode_boundaries"] = True
        diagnostics["missing_state_hashes"] = True
    return [], "none", diagnostics


def _count_improvements(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if int(row.get("success") or 0) != 1:
            continue
        best_known = row.get("best_known_solution_length")
        steps = row.get("steps_to_success")
        if best_known is None or steps is None:
            continue
        if int(steps) <= int(best_known):
            count += 1
    return count


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _mean(values: Any) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None


def _median(values: Any) -> float | None:
    cooked = sorted(float(value) for value in values if value is not None)
    if not cooked:
        return None
    mid = len(cooked) // 2
    if len(cooked) % 2 == 1:
        return cooked[mid]
    return (cooked[mid - 1] + cooked[mid]) / 2.0


def _correlation(rows: list[dict[str, Any]], left_key: str, right_key: str) -> float | None:
    points = [(float(row[left_key]), float(row[right_key])) for row in rows if row.get(left_key) is not None and row.get(right_key) is not None]
    if len(points) < 2:
        return None
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in points)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        path.write_text("[]", encoding="utf-8")
        return
    table = pa.Table.from_pylist(rows or [{"trajectory_id": None}])
    pq.write_table(table, path, compression="zstd")


def _write_report(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "h12_efficiency_emergence_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H12 decision: {result.get('decision')}\n"
        f"evidence source: {result.get('evidence_source')}\n"
        f"successful trajectories: {result.get('successful_trajectories')}\n"
        f"comparable trajectory groups: {result.get('comparable_trajectory_groups')}\n"
        f"efficiency-active trajectories: {result.get('efficiency_active_trajectory_count')}\n"
        f"trajectories with memory bonus: {result.get('trajectories_with_memory_bonus')}\n"
        f"trajectories with replay bonus: {result.get('trajectories_with_replay_bonus')}\n"
        f"mean efficiency memory bonus: {result.get('mean_efficiency_memory_bonus')}\n"
        f"mean efficiency replay bonus: {result.get('mean_efficiency_replay_bonus')}\n"
        f"median steps to success: {result.get('median_steps_to_success')}\n"
        f"mean normalized solve efficiency: {result.get('mean_normalized_solve_efficiency')}\n"
        f"mean loop ratio: {result.get('mean_loop_ratio')}\n"
        f"mean repeated-state ratio: {result.get('mean_repeated_state_ratio')}\n"
        f"mean blocked-action ratio: {result.get('mean_blocked_action_ratio')}\n"
    )
    (output_dir / "h12_efficiency_emergence_report.txt").write_text(text, encoding="utf-8")
