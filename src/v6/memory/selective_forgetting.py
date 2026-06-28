from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from v6.memory.compact_memory import configure_compact_sqlite_connection, ensure_memory_layout


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def run_selective_forgetting_pass(
    *,
    memory_dir: str | Path,
    epoch: int,
) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    with sqlite3.connect(paths.current_state) as conn:
        conn.row_factory = sqlite3.Row
        configure_compact_sqlite_connection(conn, write=True)
        nodes = {
            str(row["node_id"]): dict(row)
            for row in conn.execute(
                "SELECT node_id, memory_level, node_type, support_count, attrs_json FROM memory_nodes ORDER BY node_id ASC"
            ).fetchall()
        }
        promotions_by_source: dict[str, list[dict[str, Any]]] = {}
        for row in conn.execute(
            "SELECT source_node_id, target_node_id, promotion_type, status FROM memory_promotions ORDER BY promotion_id ASC"
        ).fetchall():
            promotions_by_source.setdefault(str(row["source_node_id"]), []).append(dict(row))
        scores = [dict(row) for row in conn.execute("SELECT * FROM memory_scores ORDER BY node_id ASC").fetchall()]
        before_counts = _state_counts(scores)
        promoted_targets = {str(row["target_node_id"]) for row in conn.execute("SELECT target_node_id FROM memory_promotions").fetchall()}
        harmed_prediction_count = 0
        harmed_transfer_count = 0
        redundancy_removed_count = 0
        for row in scores:
            node_id = str(row["node_id"])
            node = nodes.get(node_id, {})
            support_count = int(node.get("support_count") or 0)
            isf_total = float(row.get("isf_total") or 0.0)
            replay_priority = float(row.get("replay_priority") or 0.0)
            explanatory_reach = float(row.get("explanatory_reach") or 0.0)
            transfer_score = float(row.get("transfer_score") or 0.0)
            future_option_impact = abs(float(row.get("future_option_delta") or 0.0))
            recurrence = clamp01(float(support_count) / 5.0)
            efficiency_bonus = 0.0
            try:
                attrs = json.loads(node.get("attrs_json") or "{}")
            except Exception:
                attrs = {}
            efficiency_bonus = float(attrs.get("efficiency_memory_bonus") or attrs.get("trajectory_efficiency_score") or 0.0)
            retention_score = clamp01(
                0.25 * isf_total
                + 0.20 * replay_priority
                + 0.15 * explanatory_reach
                + 0.15 * transfer_score
                + 0.10 * recurrence
                + 0.10 * future_option_impact
                + 0.05 * clamp01(efficiency_bonus)
            )
            low_replay_usage = 1.0 if replay_priority <= 0.0 else 0.0
            age = max(0, int(epoch) - int(row.get("stored_epoch") or epoch))
            age_penalty = clamp01(float(age) / 5.0)
            forgetting_score = clamp01(
                0.40 * (1.0 - retention_score)
                + 0.20 * low_replay_usage
                + 0.15 * (1.0 - explanatory_reach)
                + 0.15 * (1.0 - transfer_score)
                + 0.10 * age_penalty
            )
            state = "active"
            forgetting_reason = None
            compressed_into_id = None
            superseded_by_id = None
            if node_id in promoted_targets:
                state = "promoted"
            source_promotions = promotions_by_source.get(node_id, [])
            if source_promotions and retention_score < 0.75:
                compressed_into_id = str(source_promotions[0]["target_node_id"])
                state = "compressed"
                forgetting_reason = "explained_by_promoted_structure"
                redundancy_removed_count += 1
            elif support_count <= 1 and isf_total < 0.20 and transfer_score < 0.20 and explanatory_reach < 0.20 and future_option_impact <= 0.0:
                state = "forgotten"
                forgetting_reason = "low_value_no_reuse"
            elif forgetting_score >= 0.70:
                state = "archived"
                forgetting_reason = "age_without_reuse"
            elif forgetting_score >= 0.55 and transfer_score < 0.15 and explanatory_reach < 0.15:
                state = "superseded"
                superseded_by_id = compressed_into_id
                forgetting_reason = "low_value_conflict_or_redundancy"
            if isf_total >= 0.70 and state in {"archived", "forgotten"}:
                state = "active"
                forgetting_reason = None
            if state in {"archived", "forgotten"} and isf_total >= 0.50:
                harmed_prediction_count += 1
            if state in {"archived", "forgotten"} and transfer_score >= 0.50:
                harmed_transfer_count += 1
            conn.execute(
                """
                UPDATE memory_scores
                SET
                    memory_state = ?,
                    stored_epoch = COALESCE(stored_epoch, ?),
                    last_promoted_epoch = CASE WHEN ? = 'promoted' THEN COALESCE(last_promoted_epoch, ?) ELSE last_promoted_epoch END,
                    retention_score = ?,
                    forgetting_score = ?,
                    compressed_into_id = ?,
                    superseded_by_id = ?,
                    forgetting_reason = ?,
                    retention_status = COALESCE(retention_status, ?)
                WHERE node_id = ?
                """,
                (
                    state,
                    int(epoch),
                    state,
                    int(epoch),
                    float(retention_score),
                    float(forgetting_score),
                    compressed_into_id,
                    superseded_by_id,
                    forgetting_reason,
                    state,
                    node_id,
                ),
            )
        conn.commit()
        scores_after = [dict(row) for row in conn.execute("SELECT * FROM memory_scores ORDER BY node_id ASC").fetchall()]
    after_counts = _state_counts(scores_after)
    stored_count = len(scores_after)
    high_isf_rows = [row for row in scores_after if float(row.get("isf_total") or 0.0) >= 0.5]
    low_isf_rows = [row for row in scores_after if float(row.get("isf_total") or 0.0) < 0.5]
    result = {
        "stored_memory_count": stored_count,
        "active_memory_count": after_counts.get("active", 0),
        "compressed_memory_count": after_counts.get("compressed", 0),
        "archived_memory_count": after_counts.get("archived", 0),
        "forgotten_memory_count": after_counts.get("forgotten", 0),
        "promoted_memory_count": after_counts.get("promoted", 0),
        "memory_survival_ratio": _survival_ratio(scores_after),
        "high_isf_survival_ratio": _survival_ratio(high_isf_rows),
        "low_isf_survival_ratio": _survival_ratio(low_isf_rows),
        "high_vs_low_survival_lift": _lift(_survival_ratio(high_isf_rows), _survival_ratio(low_isf_rows)),
        "redundancy_removed_count": redundancy_removed_count,
        "memory_growth_rate": 0.0 if not before_counts else float(stored_count - sum(before_counts.values())) / max(1, sum(before_counts.values())),
        "compression_ratio_before": float(before_counts.get("compressed", 0)) / max(1, stored_count),
        "compression_ratio_after": float(after_counts.get("compressed", 0)) / max(1, stored_count),
        "abstraction_score_before": _abstraction_score(before_counts, stored_count),
        "abstraction_score_after": _abstraction_score(after_counts, stored_count),
        "transfer_score_before": _mean(row.get("transfer_score") for row in scores),
        "transfer_score_after": _mean(row.get("transfer_score") for row in scores_after),
        "forgetting_harmed_prediction_count": harmed_prediction_count,
        "forgetting_harmed_transfer_count": harmed_transfer_count,
    }
    return result


def _state_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("memory_state") or row.get("retention_status") or "active")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _survival_ratio(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    survived = sum(1 for row in rows if str(row.get("memory_state") or row.get("retention_status") or "active") in {"active", "compressed", "promoted", "superseded"})
    return float(survived) / float(len(rows))


def _lift(high: float, low: float) -> float | None:
    if low <= 0.0:
        return None if high <= 0.0 else float("inf")
    return float(high) / float(low)


def _abstraction_score(counts: dict[str, int], total: int) -> float:
    if total <= 0:
        return 0.0
    return float(counts.get("compressed", 0) + counts.get("promoted", 0)) / float(total)


def _mean(values: Any) -> float | None:
    cooked = [float(value) for value in values if value is not None]
    return (sum(cooked) / len(cooked)) if cooked else None
