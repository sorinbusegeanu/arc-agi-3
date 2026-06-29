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
        "h12_efficiency_not_improving": (
            previous_epoch_mean is not None
            and current_mean_norm is not None
            and not (float(current_mean_norm) > float(previous_epoch_mean))
        ),
        "raw_trajectory_rows": int(reconstruction.get("raw_trajectory_rows", 0) or 0),
        "compact_trajectory_rows": int(reconstruction.get("compact_trajectory_rows", 0) or 0),
        "reconstructed_trajectory_rows": int(reconstruction.get("reconstructed_trajectory_rows", 0) or 0),
        "missing_episode_id_count": int(reconstruction.get("missing_episode_id_count", 0) or 0),
        "missing_state_hash_count": int(reconstruction.get("missing_state_hash_count", 0) or 0),
        "blocked_by_missing_trajectory_evidence": False,
        "trajectory_reconstruction_diagnostics": reconstruction,
        "missing_evidence": [],
    }
    cost_gap_replay_selection_bad = (
        result.get("cost_gap_replay_priority_correlation") is not None
        and float(result["cost_gap_replay_priority_correlation"]) > 0.0
    )
    result["cost_gap_replay_selection_bad"] = cost_gap_replay_selection_bad
    result["negative_cost_gap_replay_selection"] = False
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
        if not rows:
            result["blocked_by_missing_trajectory_evidence"] = True
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
    if result["h12_efficiency_not_improving"] and result["decision"] == "VALID":
        result["decision"] = "PARTIALLY_VALID"
    if result["h12_efficiency_not_improving"]:
        result["missing_evidence"] = list(
            dict.fromkeys(
                list(result.get("missing_evidence", []))
                + ["Trajectory efficiency evidence exists but is not improving versus the previous epoch."]
            )
        )
    if cost_gap_replay_selection_bad:
        result["missing_evidence"] = list(
            dict.fromkeys(
                list(result.get("missing_evidence", []))
                + ["Replay preference is positively correlated with equivalent-outcome cost gap; inefficient trajectories are being preferentially selected."]
            )
        )
        if result["decision"] == "VALID":
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
            "cost_gap_replay_selection_bad",
            "negative_cost_gap_replay_selection",
            "future_option_gain_per_action_correlation",
            "efficiency_improved_vs_previous_epoch",
            "h12_efficiency_not_improving",
            "raw_trajectory_rows",
            "compact_trajectory_rows",
            "reconstructed_trajectory_rows",
            "missing_episode_id_count",
            "missing_state_hash_count",
            "blocked_by_missing_trajectory_evidence",
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
        "raw_trajectory_rows": 0,
        "compact_trajectory_rows": 0,
        "reconstructed_trajectory_rows": 0,
        "missing_episode_id_count": 0,
        "missing_state_hash_count": 0,
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
                    total_rows = int(conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0])
                    diagnostics["no_terminal_events"] = diagnostics["no_terminal_events"] or success_terminal <= 0
                    diagnostics["no_episode_boundaries"] = diagnostics["no_episode_boundaries"] or episode_boundaries <= 0
                    diagnostics["missing_state_hashes"] = diagnostics["missing_state_hashes"] or hashed <= 0
                    diagnostics["missing_episode_id_count"] += max(0, total_rows - episode_boundaries)
                    diagnostics["missing_state_hash_count"] += max(0, total_rows - hashed)
                except sqlite3.DatabaseError:
                    pass
            if "trajectory_efficiency" not in tables:
                continue
            batch = [dict(row) for row in conn.execute("SELECT * FROM trajectory_efficiency ORDER BY trajectory_id ASC").fetchall()]
            rows.extend(batch)
            diagnostics["raw_trajectory_rows"] += len(batch)
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
                diagnostics["compact_trajectory_rows"] = len(rows)
                diagnostics["no_success_events"] = all(int(row.get("success") or 0) != 1 for row in rows)
                if direct_streaming_manifest_exists(memory_dir) and not sqlite_paths:
                    return rows, "direct_streaming_manifest_and_compact_memory", diagnostics
                return rows, "compact_memory", diagnostics
        if "compact_interaction_trajectory_events" in tables:
            event_rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM compact_interaction_trajectory_events ORDER BY game_id ASC, level_id ASC, sampler ASC, seed ASC, episode_id ASC, global_step ASC, event_id ASC"
                ).fetchall()
            ]
            if event_rows:
                diagnostics["compact_trajectory_rows"] = len(event_rows)
                rows = _reconstruct_trajectory_rows_from_compact_events(event_rows, diagnostics)
                if rows:
                    if direct_streaming_manifest_exists(memory_dir) and not sqlite_paths:
                        return rows, "direct_streaming_manifest_and_compact_memory", diagnostics
                    return rows, "compact_memory", diagnostics
    diagnostics["missing_compact_trajectory_records"] = True
    if not saw_interactions:
        diagnostics["no_episode_boundaries"] = True
        diagnostics["missing_state_hashes"] = True
    return [], "none", diagnostics


def _reconstruct_trajectory_rows_from_compact_events(
    event_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        game_id = str(row.get("game_id") or "unknown_game")
        level_id = None if row.get("level_id") in (None, "") else str(row.get("level_id"))
        sampler = None if row.get("sampler") in (None, "") else str(row.get("sampler"))
        seed = int(row.get("seed") or 0)
        episode_id = row.get("episode_id")
        if episode_id in (None, ""):
            diagnostics["missing_episode_id_count"] += 1
            episode_key = f"no_episode:{row.get('event_id')}"
        else:
            episode_key = str(int(episode_id))
        if row.get("state_hash_before") in (None, "") or row.get("state_hash_after") in (None, ""):
            diagnostics["missing_state_hash_count"] += 1
        trajectory_id = f"{game_id}|{level_id or '__none__'}|{sampler or 'unknown'}|{seed}|{episode_key}"
        by_trajectory.setdefault(trajectory_id, []).append(row)
    best_known_by_scope: dict[str, int] = {}
    prepared: list[dict[str, Any]] = []
    for trajectory_id, rows in sorted(by_trajectory.items()):
        rows = sorted(rows, key=lambda item: (int(item.get("global_step") or 0), str(item.get("event_id") or "")))
        last = rows[-1]
        game_id = str(last.get("game_id") or "unknown_game")
        level_id = None if last.get("level_id") in (None, "") else str(last.get("level_id"))
        sampler = None if last.get("sampler") in (None, "") else str(last.get("sampler"))
        seed = int(last.get("seed") or 0)
        epoch = None if last.get("epoch") in (None, "") else int(last.get("epoch"))
        length = len(rows)
        success = bool(int(last.get("level_completed_event") or 0)) or str(last.get("outcome_state") or "") in {"WIN", "LEVEL_COMPLETE"}
        terminal = success or str(last.get("outcome_state") or "") == "GAME_OVER"
        if success:
            outcome_class = "LEVEL_COMPLETE" if bool(int(last.get("level_completed_event") or 0)) else "WIN"
        elif str(last.get("outcome_state") or "") == "GAME_OVER":
            outcome_class = "GAME_OVER"
        else:
            outcome_class = str(last.get("outcome_state") or "NOT_FINISHED")
        future_option_gain = sum(float(item.get("future_option_gain") or 0.0) for item in rows)
        future_option_gain_per_action = (future_option_gain / float(length)) if length > 0 else None
        state_hashes = [str(item.get("state_hash_after")) for item in rows if item.get("state_hash_after") not in (None, "")]
        repeated_state_count = max(0, len(state_hashes) - len(set(state_hashes)))
        loop_count = repeated_state_count
        blocked_action_count = sum(1 for item in rows if int(item.get("no_effect_action") or 0) == 1)
        wasted_action_count = blocked_action_count
        unique_state_count = len(set(state_hashes))
        scope_key = f"{game_id}|{level_id or '__none__'}"
        steps_to_success = length if success else None
        if success:
            current_best = best_known_by_scope.get(scope_key)
            best_known_by_scope[scope_key] = length if current_best is None else min(current_best, length)
        prepared.append(
            {
                "trajectory_id": trajectory_id,
                "game_id": game_id,
                "level_id": level_id,
                "sampler": sampler,
                "seed": seed,
                "epoch": epoch,
                "outcome_class": outcome_class,
                "terminal": int(terminal),
                "success": int(success),
                "trajectory_length": length,
                "steps_to_success": steps_to_success,
                "future_option_gain": future_option_gain,
                "future_option_gain_per_action": future_option_gain_per_action,
                "loop_count": loop_count,
                "loop_ratio": (float(loop_count) / float(length)) if length > 0 else 0.0,
                "repeated_state_count": repeated_state_count,
                "repeated_state_ratio": (float(repeated_state_count) / float(length)) if length > 0 else 0.0,
                "blocked_action_count": blocked_action_count,
                "blocked_action_ratio": (float(blocked_action_count) / float(length)) if length > 0 else 0.0,
                "wasted_action_count": wasted_action_count,
                "wasted_action_ratio": (float(wasted_action_count) / float(length)) if length > 0 else 0.0,
                "unique_state_count": unique_state_count,
                "final_state_hash": state_hashes[-1] if state_hashes else None,
                "_scope_key": scope_key,
            }
        )
    group_counts: dict[str, int] = {}
    for row in prepared:
        if int(row.get("success") or 0) == 1:
            group_id = f"{row['_scope_key']}|success"
        elif row.get("final_state_hash"):
            group_id = f"{row['_scope_key']}|{row['outcome_class']}|state:{row['final_state_hash']}"
        elif row.get("future_option_gain_per_action") is not None:
            bucket = round(float(row["future_option_gain_per_action"]), 3)
            group_id = f"{row['_scope_key']}|{row['outcome_class']}|gain:{bucket}"
        else:
            group_id = ""
        row["comparable_outcome_group_id"] = group_id
        if group_id:
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
    output: list[dict[str, Any]] = []
    for row in prepared:
        group_id = str(row.get("comparable_outcome_group_id") or "")
        efficiency_active = bool(group_id) and int(group_counts.get(group_id, 0)) > 1
        best_known = best_known_by_scope.get(str(row["_scope_key"]))
        steps_to_success = row.get("steps_to_success")
        normalized = (
            float(best_known) / float(steps_to_success)
            if efficiency_active and best_known is not None and steps_to_success not in (None, 0)
            else None
        )
        equivalent_gap = (
            float(int(row["trajectory_length"]) - int(best_known))
            if best_known is not None and int(row.get("success") or 0) == 1
            else None
        )
        if int(row.get("success") or 0) == 1 and efficiency_active:
            efficiency_score = normalized
        elif efficiency_active and row.get("future_option_gain_per_action") is not None:
            efficiency_score = float(row["future_option_gain_per_action"])
        else:
            efficiency_score = None
        output.append(
            {
                "trajectory_id": row["trajectory_id"],
                "game_id": row["game_id"],
                "level_id": row["level_id"],
                "sampler": row["sampler"],
                "seed": row["seed"],
                "epoch": row["epoch"],
                "outcome_class": row["outcome_class"],
                "comparable_outcome_group_id": group_id,
                "efficiency_active": int(efficiency_active),
                "success": int(row["success"]),
                "terminal": int(row["terminal"]),
                "trajectory_length": int(row["trajectory_length"]),
                "steps_to_success": steps_to_success,
                "best_known_solution_length": best_known,
                "normalized_solve_efficiency": normalized,
                "future_option_gain": row["future_option_gain"],
                "future_option_gain_per_action": row["future_option_gain_per_action"],
                "equivalent_outcome_cost_gap": equivalent_gap,
                "loop_count": int(row["loop_count"]),
                "loop_ratio": float(row["loop_ratio"]),
                "repeated_state_count": int(row["repeated_state_count"]),
                "repeated_state_ratio": float(row["repeated_state_ratio"]),
                "blocked_action_count": int(row["blocked_action_count"]),
                "blocked_action_ratio": float(row["blocked_action_ratio"]),
                "wasted_action_count": int(row["wasted_action_count"]),
                "wasted_action_ratio": float(row["wasted_action_ratio"]),
                "unique_state_count": int(row["unique_state_count"]),
                "trajectory_efficiency_score": efficiency_score,
                "efficiency_score": efficiency_score,
                "efficiency_memory_bonus": 0.0,
                "efficiency_replay_bonus": 0.0,
                "efficiency_retention_bonus": 0.0,
                "efficiency_promotion_bonus": 0.0,
            }
        )
    diagnostics["reconstructed_trajectory_rows"] = len(output)
    diagnostics["no_success_events"] = all(int(row.get("success") or 0) != 1 for row in output)
    diagnostics["no_terminal_events"] = all(int(row.get("terminal") or 0) != 1 for row in output)
    diagnostics["no_episode_boundaries"] = all("no_episode:" in str(row.get("trajectory_id") or "") for row in output)
    diagnostics["missing_state_hashes"] = diagnostics["missing_state_hash_count"] > 0
    return output


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
