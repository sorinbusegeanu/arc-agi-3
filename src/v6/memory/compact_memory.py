from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_EXAMPLES_PER_CONTINGENCY = 4
DEFAULT_MAX_EXAMPLES_PER_FAMILY = 5
DEFAULT_MAX_EXAMPLES_PER_CARRIER = 5
DEFAULT_MAX_EXAMPLES_PER_CONTRADICTION_CLUSTER = 5


@dataclass(frozen=True)
class CompactMemoryPaths:
    root: Path
    current_state: Path
    graph: Path
    replay_queue: Path
    summary_json: Path


@dataclass(frozen=True)
class CompactMemoryFoldConfig:
    global_step_start: int
    global_step_end: int
    max_replay_queue_size: int = 50_000
    replay_retention_percent: int = 5
    max_examples_per_contingency: int = DEFAULT_MAX_EXAMPLES_PER_CONTINGENCY
    max_examples_per_family: int = DEFAULT_MAX_EXAMPLES_PER_FAMILY
    max_examples_per_carrier: int = DEFAULT_MAX_EXAMPLES_PER_CARRIER
    max_examples_per_contradiction_cluster: int = DEFAULT_MAX_EXAMPLES_PER_CONTRADICTION_CLUSTER


def ensure_memory_layout(memory_dir: str | Path) -> CompactMemoryPaths:
    root = Path(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = CompactMemoryPaths(
        root=root,
        current_state=root / "current_state.sqlite",
        graph=root / "graph.sqlite",
        replay_queue=root / "replay_queue.sqlite",
        summary_json=root / "memory_summary.json",
    )
    _ensure_current_state_schema(paths.current_state)
    _ensure_graph_schema(paths.graph)
    _ensure_replay_schema(paths.replay_queue)
    if not paths.summary_json.exists():
        paths.summary_json.write_text(json.dumps({"initialized": True}, indent=2), encoding="utf-8")
    return paths


def fold_epoch_raw_into_compact_memory(
    *,
    epoch_raw_dir: str | Path,
    memory_dir: str | Path,
    fold_config: CompactMemoryFoldConfig,
) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    raw_dir = Path(epoch_raw_dir)
    db_paths = sorted(raw_dir.rglob("*.sqlite"))
    report = _load_json(raw_dir / "interaction_sampling_v05c_report.json") or {}
    temporal_rows = list(((report.get("temporal_milestones") or {}).get("by_game_sampler_seed") or []))
    totals = {
        "stable_contingencies_added": 0,
        "transformation_families_added": 0,
        "carrier_candidates_added": 0,
        "contradiction_clusters_added": 0,
        "replay_queue_size": 0,
        "representative_examples_retained": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "total_interactions_seen": int((report.get("validation") or {}).get("memory_record_count", 0) or 0),
    }
    stable_threshold = 20
    current_summary = load_memory_summary(paths.summary_json)
    totals["total_interactions_seen"] += int(current_summary.get("total_interactions_seen", 0) or 0)

    with sqlite3.connect(paths.current_state) as state_conn, sqlite3.connect(paths.graph) as graph_conn, sqlite3.connect(paths.replay_queue) as replay_conn:
        for db_path in db_paths:
            _fold_single_db(
                db_path=db_path,
                state_conn=state_conn,
                graph_conn=graph_conn,
                replay_conn=replay_conn,
                fold_config=fold_config,
                totals=totals,
                stable_threshold=stable_threshold,
            )
        _upsert_temporal_milestones(state_conn, temporal_rows)
        _trim_representative_examples(state_conn, fold_config)
        _trim_replay_queue(replay_conn, fold_config)
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
        totals["replay_queue_size"] = _count_rows(replay_conn, "replay_queue")
        totals["representative_examples_retained"] = _count_rows(state_conn, "representative_examples")
        totals["graph_node_count"] = _count_rows(graph_conn, "graph_nodes")
        totals["graph_edge_count"] = _count_rows(graph_conn, "graph_edges")
        summary = build_memory_summary(paths)
        summary.update(totals)
        paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return totals


def build_memory_summary(paths: CompactMemoryPaths) -> dict[str, Any]:
    with sqlite3.connect(paths.current_state) as state_conn, sqlite3.connect(paths.graph) as graph_conn, sqlite3.connect(paths.replay_queue) as replay_conn:
        return {
            "stable_contingency_count": _count_rows(state_conn, "stable_contingencies"),
            "transformation_family_count": _count_rows(state_conn, "transformation_families"),
            "carrier_candidate_count": _count_rows(state_conn, "carrier_candidates"),
            "contradiction_cluster_count": _count_rows(state_conn, "contradiction_clusters"),
            "representative_example_count": _count_rows(state_conn, "representative_examples"),
            "graph_node_count": _count_rows(graph_conn, "graph_nodes"),
            "graph_edge_count": _count_rows(graph_conn, "graph_edges"),
            "replay_queue_size": _count_rows(replay_conn, "replay_queue"),
            "current_state_path": str(paths.current_state),
            "graph_path": str(paths.graph),
            "replay_queue_path": str(paths.replay_queue),
        }


def load_memory_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _fold_single_db(
    *,
    db_path: Path,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    replay_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any],
    stable_threshold: int,
) -> None:
    with sqlite3.connect(db_path) as raw_conn:
        tables = {str(row[0]) for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "contingencies" in tables:
            rows = raw_conn.execute(
                """
                SELECT id, context_signature, action, transformation_family, support_count
                FROM contingencies
                """
            ).fetchall()
            for contingency_id, context_signature, action, family_id, support_count in rows:
                if int(support_count or 0) < stable_threshold:
                    continue
                canonical_key = f"{context_signature}|a{int(action)}|f{int(family_id)}"
                _upsert_state_row(
                    state_conn,
                    """
                    INSERT INTO stable_contingencies (
                        contingency_id, canonical_key, game, sampler, action, effect_signature, support_count,
                        first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                        mean_replay_priority, representative_example_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(canonical_key) DO UPDATE SET
                        support_count = MAX(stable_contingencies.support_count, excluded.support_count),
                        last_seen_global_step = MAX(stable_contingencies.last_seen_global_step, excluded.last_seen_global_step),
                        stability_score = MAX(stable_contingencies.stability_score, excluded.stability_score)
                    """,
                    (
                        int(contingency_id),
                        canonical_key,
                        _path_segment(db_path, -4),
                        _path_segment(db_path, -3),
                        int(action),
                        str(family_id),
                        int(support_count),
                        int(fold_config.global_step_start),
                        int(fold_config.global_step_end),
                        float(support_count) / max(1.0, float(stable_threshold)),
                        0.0,
                        0.0,
                        0,
                    ),
                )
                totals["stable_contingencies_added"] += 1
                _retain_example(
                    state_conn,
                    owner_type="contingency",
                    owner_id=canonical_key,
                    limit=fold_config.max_examples_per_contingency,
                    example_kind="first",
                    game=_path_segment(db_path, -4),
                    sampler=_path_segment(db_path, -3),
                    seed=int(db_path.stem.split("_")[-1]),
                    global_step=fold_config.global_step_start,
                    priority_score=float(support_count),
                    compact_payload_json=json.dumps({"context_signature": context_signature, "family_id": family_id, "support_count": support_count}),
                )
                _upsert_graph_node(graph_conn, node_id=f"contingency:{canonical_key}", node_type="contingency", canonical_key=canonical_key, fold_config=fold_config)
                _upsert_graph_node(graph_conn, node_id=f"family:{family_id}", node_type="family", canonical_key=str(family_id), fold_config=fold_config)
                _upsert_graph_edge(graph_conn, source_node_id=f"contingency:{canonical_key}", target_node_id=f"family:{family_id}", edge_type="member_of", fold_config=fold_config)
                _upsert_state_row(
                    state_conn,
                    """
                    INSERT INTO family_members (family_id, contingency_id, support_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(family_id, contingency_id) DO UPDATE SET
                        support_count = MAX(family_members.support_count, excluded.support_count)
                    """,
                    (str(family_id), int(contingency_id), int(support_count)),
                )
        if "prediction_results" in tables:
            family_rows = raw_conn.execute(
                """
                SELECT COALESCE(actual_family, predicted_family), COUNT(*), MAX(COALESCE(global_step, interaction_id)), AVG(COALESCE(isf_prediction_error, 0.0)),
                       AVG(COALESCE(memory_replay_priority, 0.0)), MIN(COALESCE(global_step, interaction_id))
                FROM prediction_results
                WHERE COALESCE(actual_family, predicted_family) IS NOT NULL
                GROUP BY COALESCE(actual_family, predicted_family)
                """
            ).fetchall()
            for family_id, member_count, last_step, mean_error, mean_replay, first_step in family_rows:
                family_key = str(family_id)
                _upsert_state_row(
                    state_conn,
                    """
                    INSERT INTO transformation_families (
                        family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                        support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(family_id) DO UPDATE SET
                        support_count = transformation_families.support_count + excluded.support_count,
                        member_count = transformation_families.member_count + excluded.member_count,
                        last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
                        stability_score = MAX(transformation_families.stability_score, excluded.stability_score)
                    """,
                    (
                        family_key,
                        family_key,
                        family_key,
                        "unknown",
                        "unknown",
                        "unknown",
                        int(member_count),
                        int(member_count),
                        int(first_step or fold_config.global_step_start),
                        int(last_step or fold_config.global_step_end),
                        float(member_count),
                    ),
                )
                totals["transformation_families_added"] += 1
                _retain_example(
                    state_conn,
                    owner_type="family",
                    owner_id=family_key,
                    limit=fold_config.max_examples_per_family,
                    example_kind="support",
                    game=_path_segment(db_path, -4),
                    sampler=_path_segment(db_path, -3),
                    seed=int(db_path.stem.split("_")[-1]),
                    global_step=int(last_step or fold_config.global_step_end),
                    priority_score=float(mean_error or 0.0) + float(member_count),
                    compact_payload_json=json.dumps({"family_id": family_key, "member_count": member_count, "mean_prediction_error": mean_error, "mean_replay_priority": mean_replay}),
                )
            contradiction_rows = raw_conn.execute(
                """
                SELECT context_signature, COUNT(*), MAX(COALESCE(isf_prediction_error, 0.0)), AVG(COALESCE(memory_replay_priority, 0.0)),
                       MIN(COALESCE(global_step, interaction_id)), MAX(COALESCE(global_step, interaction_id))
                FROM prediction_results
                WHERE COALESCE(context_contradiction, 0) = 1 OR COALESCE(prediction_error, 0) = 1
                GROUP BY context_signature
                """
            ).fetchall()
            for cluster_key, support_count, max_error, mean_replay, first_step, last_step in contradiction_rows:
                _upsert_state_row(
                    state_conn,
                    """
                    INSERT INTO contradiction_clusters (
                        cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step,
                        max_prediction_error, mean_replay_priority
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(canonical_key) DO UPDATE SET
                        support_count = contradiction_clusters.support_count + excluded.support_count,
                        last_seen_global_step = MAX(contradiction_clusters.last_seen_global_step, excluded.last_seen_global_step),
                        max_prediction_error = MAX(contradiction_clusters.max_prediction_error, excluded.max_prediction_error),
                        mean_replay_priority = MAX(contradiction_clusters.mean_replay_priority, excluded.mean_replay_priority)
                    """,
                    (
                        str(cluster_key),
                        str(cluster_key),
                        int(support_count),
                        int(first_step or fold_config.global_step_start),
                        int(last_step or fold_config.global_step_end),
                        float(max_error or 0.0),
                        float(mean_replay or 0.0),
                    ),
                )
                totals["contradiction_clusters_added"] += 1
                _retain_example(
                    state_conn,
                    owner_type="contradiction_cluster",
                    owner_id=str(cluster_key),
                    limit=fold_config.max_examples_per_contradiction_cluster,
                    example_kind="prediction_error",
                    game=_path_segment(db_path, -4),
                    sampler=_path_segment(db_path, -3),
                    seed=int(db_path.stem.split("_")[-1]),
                    global_step=int(last_step or fold_config.global_step_end),
                    priority_score=float(max_error or 0.0) + float(mean_replay or 0.0),
                    compact_payload_json=json.dumps({"cluster_key": cluster_key, "support_count": support_count}),
                )
        if "interactions" in tables:
            replay_rows = raw_conn.execute(
                """
                SELECT id, COALESCE(global_step, id), COALESCE(memory_replay_priority, 0.0), COALESCE(carrier_signature, ''), COALESCE(context_depth_used, 0)
                FROM interactions
                WHERE COALESCE(memory_replay_candidate, 0) = 1 OR COALESCE(memory_replay_priority, 0.0) > 0.0
                ORDER BY COALESCE(memory_replay_priority, 0.0) DESC, id DESC
                """
            ).fetchall()
            for interaction_id, global_step, priority_score, carrier_signature, context_depth_used in replay_rows:
                reason = "carrier_linked" if carrier_signature else "priority"
                _upsert_state_row(
                    replay_conn,
                    """
                    INSERT INTO replay_queue (
                        replay_id, owner_type, owner_id, priority_score, reason, first_seen_global_step,
                        last_seen_global_step, compact_payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(replay_id) DO UPDATE SET
                        priority_score = MAX(replay_queue.priority_score, excluded.priority_score),
                        last_seen_global_step = MAX(replay_queue.last_seen_global_step, excluded.last_seen_global_step),
                        compact_payload_json = excluded.compact_payload_json
                    """,
                    (
                        str(interaction_id),
                        "interaction",
                        str(interaction_id),
                        float(priority_score or 0.0),
                        reason,
                        int(global_step or fold_config.global_step_start),
                        int(global_step or fold_config.global_step_end),
                        json.dumps({"carrier_signature": carrier_signature, "context_depth_used": context_depth_used}),
                    ),
                )
        carrier_path = db_path.with_name("carrier_candidates.json")
        if carrier_path.exists():
            for item in json.loads(carrier_path.read_text(encoding="utf-8")):
                carrier_signature = str(item.get("carrier_signature") or item.get("carrier_id") or "")
                if not carrier_signature:
                    continue
                _upsert_state_row(
                    state_conn,
                    """
                    INSERT INTO carrier_candidates (
                        carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                        first_seen_global_step, last_seen_global_step, stability_score, is_emergent
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(carrier_signature) DO UPDATE SET
                        support_count = MAX(carrier_candidates.support_count, excluded.support_count),
                        linked_family_count = MAX(carrier_candidates.linked_family_count, excluded.linked_family_count),
                        last_seen_global_step = MAX(carrier_candidates.last_seen_global_step, excluded.last_seen_global_step),
                        stability_score = MAX(carrier_candidates.stability_score, excluded.stability_score),
                        is_emergent = MAX(carrier_candidates.is_emergent, excluded.is_emergent)
                    """,
                    (
                        str(item.get("carrier_id") or carrier_signature),
                        carrier_signature,
                        str(item.get("carrier_source") or "unknown"),
                        int(item.get("support_count", 0) or 0),
                        int(item.get("distinct_family_count", 0) or 0),
                        int(fold_config.global_step_start),
                        int(fold_config.global_step_end),
                        float(item.get("prediction_lift", 0.0) or 0.0),
                        int(str(item.get("status") or "") == "emergent_carrier"),
                    ),
                )
                totals["carrier_candidates_added"] += 1
                _retain_example(
                    state_conn,
                    owner_type="carrier",
                    owner_id=carrier_signature,
                    limit=fold_config.max_examples_per_carrier,
                    example_kind="support",
                    game=_path_segment(db_path, -4),
                    sampler=_path_segment(db_path, -3),
                    seed=int(db_path.stem.split("_")[-1]),
                    global_step=fold_config.global_step_end,
                    priority_score=float(item.get("prediction_lift", 0.0) or 0.0) + float(item.get("support_count", 0) or 0.0),
                    compact_payload_json=json.dumps(item, sort_keys=True),
                )
                _upsert_graph_node(graph_conn, node_id=f"carrier:{carrier_signature}", node_type="carrier", canonical_key=carrier_signature, fold_config=fold_config)


def _upsert_temporal_milestones(connection: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        connection.execute(
            """
            INSERT INTO temporal_milestones (
                game, sampler, seed, first_interaction_step, first_contingency_candidate_step, first_stable_contingency_step,
                first_prediction_violation_step, first_high_replay_priority_step, first_transformation_family_step,
                first_stable_transformation_family_step, first_carrier_candidate_step, first_emergent_carrier_step
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game, sampler, seed) DO UPDATE SET
                first_interaction_step = COALESCE(temporal_milestones.first_interaction_step, excluded.first_interaction_step),
                first_contingency_candidate_step = COALESCE(temporal_milestones.first_contingency_candidate_step, excluded.first_contingency_candidate_step),
                first_stable_contingency_step = COALESCE(temporal_milestones.first_stable_contingency_step, excluded.first_stable_contingency_step),
                first_prediction_violation_step = COALESCE(temporal_milestones.first_prediction_violation_step, excluded.first_prediction_violation_step),
                first_high_replay_priority_step = COALESCE(temporal_milestones.first_high_replay_priority_step, excluded.first_high_replay_priority_step),
                first_transformation_family_step = COALESCE(temporal_milestones.first_transformation_family_step, excluded.first_transformation_family_step),
                first_stable_transformation_family_step = COALESCE(temporal_milestones.first_stable_transformation_family_step, excluded.first_stable_transformation_family_step),
                first_carrier_candidate_step = COALESCE(temporal_milestones.first_carrier_candidate_step, excluded.first_carrier_candidate_step),
                first_emergent_carrier_step = COALESCE(temporal_milestones.first_emergent_carrier_step, excluded.first_emergent_carrier_step)
            """,
            (
                row.get("game"),
                row.get("sampler"),
                int(row.get("seed", 0) or 0),
                row.get("first_interaction_step"),
                row.get("first_contingency_candidate_step"),
                row.get("first_stable_contingency_step"),
                row.get("first_prediction_violation_step"),
                row.get("first_high_replay_priority_step"),
                row.get("first_transformation_family_step"),
                row.get("first_stable_transformation_family_step"),
                row.get("first_carrier_candidate_step"),
                row.get("first_emergent_carrier_step"),
            ),
        )


def _retain_example(
    connection: sqlite3.Connection,
    *,
    owner_type: str,
    owner_id: str,
    limit: int,
    example_kind: str,
    game: str,
    sampler: str,
    seed: int,
    global_step: int,
    priority_score: float,
    compact_payload_json: str,
) -> None:
    example_id = f"{owner_type}:{owner_id}:{example_kind}:{seed}:{global_step}"
    connection.execute(
        """
        INSERT OR REPLACE INTO representative_examples (
            example_id, owner_type, owner_id, game, sampler, seed, global_step, example_kind, compact_payload_json, priority_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (example_id, owner_type, owner_id, game, sampler, int(seed), int(global_step), example_kind, compact_payload_json, float(priority_score)),
    )
    rows = connection.execute(
        """
        SELECT example_id
        FROM representative_examples
        WHERE owner_type = ? AND owner_id = ?
        ORDER BY priority_score DESC, global_step DESC, example_id ASC
        """,
        (owner_type, owner_id),
    ).fetchall()
    for row in rows[limit:]:
        connection.execute("DELETE FROM representative_examples WHERE example_id = ?", (row[0],))
    connection.execute(
        """
        UPDATE stable_contingencies
        SET representative_example_count = (
            SELECT COUNT(*)
            FROM representative_examples
            WHERE owner_type = 'contingency' AND owner_id = ?
        )
        WHERE canonical_key = ?
        """,
        (owner_id, owner_id),
    )


def _trim_representative_examples(connection: sqlite3.Connection, fold_config: CompactMemoryFoldConfig) -> None:
    limits = {
        "contingency": fold_config.max_examples_per_contingency,
        "family": fold_config.max_examples_per_family,
        "carrier": fold_config.max_examples_per_carrier,
        "contradiction_cluster": fold_config.max_examples_per_contradiction_cluster,
    }
    for owner_type, limit in limits.items():
        owners = connection.execute("SELECT DISTINCT owner_id FROM representative_examples WHERE owner_type = ?", (owner_type,)).fetchall()
        for (owner_id,) in owners:
            rows = connection.execute(
                """
                SELECT example_id
                FROM representative_examples
                WHERE owner_type = ? AND owner_id = ?
                ORDER BY priority_score DESC, global_step DESC, example_id ASC
                """,
                (owner_type, owner_id),
            ).fetchall()
            for row in rows[limit:]:
                connection.execute("DELETE FROM representative_examples WHERE example_id = ?", (row[0],))


def _trim_replay_queue(connection: sqlite3.Connection, fold_config: CompactMemoryFoldConfig) -> None:
    row_count = _count_rows(connection, "replay_queue")
    limit = max(1, int(fold_config.max_replay_queue_size))
    if row_count <= limit:
        return
    keep_rows = connection.execute(
        """
        SELECT replay_id
        FROM replay_queue
        ORDER BY
            CASE WHEN reason IN ('contradiction_linked', 'new_stable_contingency', 'new_family', 'new_carrier') THEN 0 ELSE 1 END,
            priority_score DESC,
            last_seen_global_step DESC,
            replay_id ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    keep_ids = {str(row[0]) for row in keep_rows}
    placeholders = ",".join("?" for _ in keep_ids) or "''"
    connection.execute(f"DELETE FROM replay_queue WHERE replay_id NOT IN ({placeholders})", tuple(keep_ids))


def _upsert_graph_node(graph_conn: sqlite3.Connection, *, node_id: str, node_type: str, canonical_key: str, fold_config: CompactMemoryFoldConfig) -> None:
    _upsert_state_row(
        graph_conn,
        """
        INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            last_seen_global_step = MAX(graph_nodes.last_seen_global_step, excluded.last_seen_global_step),
            support_count = graph_nodes.support_count + 1
        """,
        (node_id, node_type, canonical_key, int(fold_config.global_step_start), int(fold_config.global_step_end), 1),
    )


def _upsert_graph_edge(graph_conn: sqlite3.Connection, *, source_node_id: str, target_node_id: str, edge_type: str, fold_config: CompactMemoryFoldConfig) -> None:
    edge_id = f"{source_node_id}->{edge_type}->{target_node_id}"
    _upsert_state_row(
        graph_conn,
        """
        INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(edge_id) DO UPDATE SET
            last_seen_global_step = MAX(graph_edges.last_seen_global_step, excluded.last_seen_global_step),
            support_count = graph_edges.support_count + 1,
            weight = MAX(graph_edges.weight, excluded.weight)
        """,
        (edge_id, source_node_id, target_node_id, edge_type, int(fold_config.global_step_start), int(fold_config.global_step_end), 1, 1.0),
    )


def _upsert_state_row(connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...]) -> None:
    connection.execute(query, parameters)


def _count_rows(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _path_segment(path: Path, index_from_end: int) -> str:
    parts = path.parts
    try:
        return str(parts[index_from_end])
    except IndexError:
        return "unknown"


def _ensure_current_state_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS stable_contingencies (
                contingency_id INTEGER,
                canonical_key TEXT PRIMARY KEY,
                game TEXT,
                sampler TEXT,
                action INTEGER,
                effect_signature TEXT,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                stability_score REAL,
                mean_prediction_error REAL,
                mean_replay_priority REAL,
                representative_example_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS transformation_families (
                family_id TEXT PRIMARY KEY,
                canonical_signature TEXT,
                relaxed_signature TEXT,
                effect_type TEXT,
                action_group TEXT,
                polarity TEXT,
                support_count INTEGER,
                member_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                stability_score REAL
            );
            CREATE TABLE IF NOT EXISTS family_members (
                family_id TEXT,
                contingency_id INTEGER,
                support_count INTEGER,
                PRIMARY KEY (family_id, contingency_id)
            );
            CREATE TABLE IF NOT EXISTS carrier_candidates (
                carrier_id TEXT,
                carrier_signature TEXT PRIMARY KEY,
                carrier_source TEXT,
                support_count INTEGER,
                linked_family_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                stability_score REAL,
                is_emergent INTEGER
            );
            CREATE TABLE IF NOT EXISTS contradiction_clusters (
                cluster_id TEXT,
                canonical_key TEXT PRIMARY KEY,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                max_prediction_error REAL,
                mean_replay_priority REAL
            );
            CREATE TABLE IF NOT EXISTS representative_examples (
                example_id TEXT PRIMARY KEY,
                owner_type TEXT,
                owner_id TEXT,
                game TEXT,
                sampler TEXT,
                seed INTEGER,
                global_step INTEGER,
                example_kind TEXT,
                compact_payload_json TEXT,
                priority_score REAL
            );
            CREATE TABLE IF NOT EXISTS temporal_milestones (
                game TEXT,
                sampler TEXT,
                seed INTEGER,
                first_interaction_step INTEGER,
                first_contingency_candidate_step INTEGER,
                first_stable_contingency_step INTEGER,
                first_prediction_violation_step INTEGER,
                first_high_replay_priority_step INTEGER,
                first_transformation_family_step INTEGER,
                first_stable_transformation_family_step INTEGER,
                first_carrier_candidate_step INTEGER,
                first_emergent_carrier_step INTEGER,
                PRIMARY KEY (game, sampler, seed)
            );
            CREATE TABLE IF NOT EXISTS memory_summary (
                key TEXT PRIMARY KEY,
                value_json TEXT
            );
            """
        )
        connection.commit()


def _ensure_graph_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT,
                canonical_key TEXT,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                support_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id TEXT PRIMARY KEY,
                source_node_id TEXT,
                target_node_id TEXT,
                edge_type TEXT,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                support_count INTEGER,
                weight REAL
            );
            """
        )
        connection.commit()


def _ensure_replay_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS replay_queue (
                replay_id TEXT PRIMARY KEY,
                owner_type TEXT,
                owner_id TEXT,
                priority_score REAL,
                reason TEXT,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                compact_payload_json TEXT
            );
            """
        )
        connection.commit()
