from __future__ import annotations

import os
import json
import sqlite3
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm
from v6.memory.trajectory_efficiency import infer_epoch_from_path
from v6.memory.substrate import interaction_node_id, scoped_interaction_key


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
    fold_memory_substrate: bool = True
    fold_graph: bool = True
    max_graph_edges_per_fold: int = 1_000_000
    max_edges_per_source_node: int = 128
    max_edges_per_carrier: int = 32
    max_edges_per_family: int = 64
    enable_graph_edge_caps: bool = True
    use_set_based_merge: bool = True


@dataclass
class RawDbFoldCaches:
    family_signature_by_family_id: dict[str, str]
    family_signature_by_payload_key: dict[str, str]
    context_level_by_key: dict[tuple[str, str, str], int]
    normalized_jsonish: dict[str, str]
    normalized_contingency_identity: dict[tuple[int, str, str], str]
    transformation_family_rows: dict[str, sqlite3.Row]
    prediction_context_level_lookup: dict[tuple[str, str, str], int]


def configure_compact_sqlite_connection(
    conn: sqlite3.Connection,
    *,
    write: bool,
    synchronous: str = "NORMAL",
    temporary_shard: bool = False,
    busy_timeout_ms: int = 10000,
) -> None:
    mode = str(synchronous or "NORMAL").strip().upper()
    if mode not in {"OFF", "NORMAL", "FULL"}:
        mode = "NORMAL"
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    if write:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA synchronous={mode}")
        if temporary_shard:
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-65536")
            conn.execute("PRAGMA wal_autocheckpoint=0")
    else:
        conn.execute("PRAGMA query_only=ON")
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")


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


def _graph_edge_total_fields() -> dict[str, int]:
    return {
        "graph_edges_skipped_by_fold_cap": 0,
        "graph_edges_skipped_by_source_cap": 0,
        "graph_edges_skipped_by_carrier_cap": 0,
        "graph_edges_skipped_by_family_cap": 0,
        "graph_edges_pruned_by_post_merge_fold_cap": 0,
        "graph_edges_pruned_by_post_merge_source_cap": 0,
        "graph_edges_pruned_by_post_merge_carrier_cap": 0,
        "graph_edges_pruned_by_post_merge_family_cap": 0,
        "graph_edges_written_by_set_based_merge": 0,
        "graph_edges_before_set_based_merge": 0,
        "graph_edges_after_set_based_merge": 0,
        "graph_edges_attempted": 0,
        "graph_edges_written": 0,
    }


def checkpoint_compact_memory(memory_dir: str | Path, truncate: bool = True) -> dict[str, Any]:
    root = Path(memory_dir)
    mode = "TRUNCATE" if truncate else "PASSIVE"
    results: dict[str, Any] = {"mode": mode.lower(), "databases": {}}
    for name in ("current_state.sqlite", "graph.sqlite", "replay_queue.sqlite"):
        db_path = root / name
        wal_path = root / f"{name}-wal"
        shm_path = root / f"{name}-shm"
        before = {
            "db_bytes": int(db_path.stat().st_size) if db_path.exists() else 0,
            "wal_bytes": int(wal_path.stat().st_size) if wal_path.exists() else 0,
            "shm_bytes": int(shm_path.stat().st_size) if shm_path.exists() else 0,
        }
        if db_path.exists():
            conn = sqlite3.connect(db_path, timeout=10.0)
            try:
                conn.execute("PRAGMA busy_timeout=10000")
                checkpoint_row = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
            finally:
                conn.close()
        else:
            checkpoint_row = None
        after = {
            "db_bytes": int(db_path.stat().st_size) if db_path.exists() else 0,
            "wal_bytes": int(wal_path.stat().st_size) if wal_path.exists() else 0,
            "shm_bytes": int(shm_path.stat().st_size) if shm_path.exists() else 0,
        }
        results["databases"][name] = {
            "checkpoint_result": list(checkpoint_row) if checkpoint_row is not None else None,
            "before": before,
            "after": after,
        }
    return results


def derive_missing_transformation_families_from_stable_contingencies(memory_dir: str | Path) -> dict[str, Any]:
    memory_dir = Path(memory_dir)
    paths = ensure_memory_layout(memory_dir)
    current_state_path = memory_dir / "current_state.sqlite"
    if not current_state_path.exists():
        return {
            "compact_family_repair_used": False,
            "compact_family_repair_reason": "current_state_missing",
            "stable_contingencies_count": 0,
            "transformation_families_count": 0,
            "family_members_count": 0,
            "representative_contingency_example_count": 0,
            "contingency_memory_node_count": 0,
            "contingency_graph_edge_count": 0,
            "compact_m1_substrate_missing": False,
            "transformation_families_before": 0,
            "transformation_families_after": 0,
            "family_members_before": 0,
            "family_members_after": 0,
            "compact_family_repair_family_count": 0,
            "compact_family_repair_member_count": 0,
            "compact_family_repair_graph_node_count": 0,
            "compact_family_repair_graph_edge_count": 0,
            "compact_family_repair_error": None,
        }
    summary = {
        "compact_family_repair_used": False,
        "compact_family_repair_reason": "unknown",
        "stable_contingencies_count": 0,
        "transformation_families_count": 0,
        "family_members_count": 0,
        "representative_contingency_example_count": 0,
        "contingency_memory_node_count": 0,
        "contingency_graph_edge_count": 0,
        "compact_m1_substrate_missing": False,
        "transformation_families_before": 0,
        "transformation_families_after": 0,
        "family_members_before": 0,
        "family_members_after": 0,
        "compact_family_repair_family_count": 0,
        "compact_family_repair_member_count": 0,
        "compact_family_repair_graph_node_count": 0,
        "compact_family_repair_graph_edge_count": 0,
        "compact_family_repair_error": None,
        "compact_family_prediction_lift_backfill_used": False,
        "compact_family_prediction_lift_backfill_count": 0,
    }
    with sqlite3.connect(paths.current_state) as state_conn:
        configure_compact_sqlite_connection(state_conn, write=True)
        state_conn.row_factory = sqlite3.Row
        stable_count = int(state_conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0])
        family_count = int(state_conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0])
        family_members_before = int(state_conn.execute("SELECT COUNT(*) FROM family_members").fetchone()[0])
        representative_contingency_example_count = _safe_scalar(
            state_conn,
            "SELECT COUNT(*) FROM representative_examples WHERE owner_type = 'contingency'",
        )
        contingency_memory_node_count = _safe_scalar(
            state_conn,
            """
            SELECT COUNT(*)
            FROM memory_nodes
            WHERE node_type = 'ContingencyMemory'
               OR node_id LIKE 'M1:contingency:%'
            """,
        )
        summary["stable_contingencies_count"] = stable_count
        summary["transformation_families_count"] = family_count
        summary["family_members_count"] = family_members_before
        summary["representative_contingency_example_count"] = representative_contingency_example_count
        summary["contingency_memory_node_count"] = contingency_memory_node_count
        summary["transformation_families_before"] = family_count
        summary["family_members_before"] = family_members_before
        if stable_count <= 0:
            summary["compact_family_repair_reason"] = (
                "missing_m1_substrate_with_existing_m2"
                if family_count > 0
                else "no_stable_contingencies"
            )
            summary["transformation_families_after"] = family_count
            summary["family_members_after"] = family_members_before
            try:
                with sqlite3.connect(paths.graph) as graph_conn:
                    configure_compact_sqlite_connection(graph_conn, write=False)
                    summary["contingency_graph_edge_count"] = _safe_scalar(
                        graph_conn,
                        """
                        SELECT COUNT(*)
                        FROM graph_edges
                        WHERE (
                            source_node_id LIKE 'M1:contingency:%'
                            OR source_node_id LIKE 'contingency:%'
                        )
                          AND (
                            target_node_id LIKE 'M2:family:%'
                            OR target_node_id LIKE 'family:%'
                        )
                        """,
                    )
            except Exception as exc:
                summary["compact_family_repair_error"] = str(exc)
            summary["compact_m1_substrate_missing"] = bool(
                family_count > 0 and stable_count <= 0
            )
            _write_family_repair_summary(state_conn, summary)
            state_conn.commit()
            return summary
        missing_prediction_lift_count = int(
            state_conn.execute(
                "SELECT COUNT(*) FROM transformation_families WHERE prediction_lift IS NULL"
            ).fetchone()[0]
        )
        if family_count > 0 and family_members_before > 0 and missing_prediction_lift_count <= 0:
            summary["compact_family_repair_reason"] = "family_substrate_already_present"
            summary["transformation_families_after"] = family_count
            summary["family_members_after"] = family_members_before
            _write_family_repair_summary(state_conn, summary)
            state_conn.commit()
            return summary
        if family_count > 0 and family_members_before > 0 and missing_prediction_lift_count > 0:
            backfill_count = _backfill_transformation_family_prediction_lift(state_conn)
            summary["compact_family_prediction_lift_backfill_used"] = backfill_count > 0
            summary["compact_family_prediction_lift_backfill_count"] = backfill_count
            summary["compact_family_repair_reason"] = (
                "family_prediction_lift_backfilled" if backfill_count > 0 else "family_substrate_already_present"
            )
            summary["transformation_families_after"] = family_count
            summary["family_members_after"] = family_members_before
            _write_family_repair_summary(state_conn, summary)
            state_conn.commit()
            return summary
        rows = state_conn.execute(
            """
            SELECT
                canonical_key,
                COALESCE(effect_signature, normalized_contingency_key, canonical_key) AS family_signature,
                COALESCE(action, 0) AS action_value,
                COALESCE(effect_signature, 'unknown') AS effect_signature,
                COALESCE(game, 'unknown') AS game_name,
                COALESCE(sampler, 'unknown') AS sampler_name,
                COALESCE(support_count, 0) AS support_count,
                COALESCE(first_seen_global_step, 0) AS first_seen_global_step,
                COALESCE(last_seen_global_step, 0) AS last_seen_global_step,
                COALESCE(stability_score, 0.0) AS stability_score,
                prediction_accuracy,
                prediction_error_before,
                prediction_error_after
            FROM stable_contingencies
            ORDER BY canonical_key ASC
            """
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(str(row["family_signature"]), []).append(row)
        inserted_families = 0
        inserted_members = 0
        for family_signature in sorted(grouped):
            members = grouped[family_signature]
            support_total = sum(int(row["support_count"] or 0) for row in members)
            member_count = len(members)
            first_seen = min(int(row["first_seen_global_step"] or 0) for row in members)
            last_seen = max(int(row["last_seen_global_step"] or 0) for row in members)
            stability_values = [float(row["stability_score"] or 0.0) for row in members]
            stability_score = (
                (sum(stability_values) / len(stability_values))
                if stability_values
                else 0.0
            )
            prediction_accuracy_values = [
                float(row["prediction_accuracy"])
                for row in members
                if row["prediction_accuracy"] is not None
            ]
            prediction_error_before_values = [
                float(row["prediction_error_before"])
                for row in members
                if row["prediction_error_before"] is not None
            ]
            prediction_error_after_values = [
                float(row["prediction_error_after"])
                for row in members
                if row["prediction_error_after"] is not None
            ]
            prediction_accuracy_mean = (
                float(sum(prediction_accuracy_values) / len(prediction_accuracy_values))
                if prediction_accuracy_values
                else None
            )
            prediction_error_before_mean = (
                float(sum(prediction_error_before_values) / len(prediction_error_before_values))
                if prediction_error_before_values
                else None
            )
            prediction_error_after_mean = (
                float(sum(prediction_error_after_values) / len(prediction_error_after_values))
                if prediction_error_after_values
                else None
            )
            if prediction_error_before_mean is not None and prediction_error_after_mean is not None:
                prediction_lift = float(prediction_error_before_mean - prediction_error_after_mean)
            elif prediction_accuracy_mean is not None:
                prediction_lift = float(prediction_accuracy_mean)
            else:
                prediction_lift = None
            action_counts: dict[str, int] = {}
            for member in members:
                action_key = str(member["action_value"] if member["action_value"] is not None else "unknown")
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
            action_value = sorted(action_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0] if action_counts else "unknown"
            family_id = _stable_family_int_id(family_signature)
            effect_type = "compact_derived"
            action_group = str(action_value) if action_counts else "unknown"
            polarity = "unknown"
            state_conn.execute(
                """
                INSERT INTO transformation_families (
                    family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                    support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score,
                    prediction_lift, prediction_accuracy_mean, prediction_error_before_mean, prediction_error_after_mean
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_signature) DO UPDATE SET
                    support_count = MAX(transformation_families.support_count, excluded.support_count),
                    member_count = MAX(transformation_families.member_count, excluded.member_count),
                    first_seen_global_step = MIN(transformation_families.first_seen_global_step, excluded.first_seen_global_step),
                    last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
                    stability_score = MAX(transformation_families.stability_score, excluded.stability_score),
                    prediction_lift = COALESCE(excluded.prediction_lift, transformation_families.prediction_lift),
                    prediction_accuracy_mean = COALESCE(excluded.prediction_accuracy_mean, transformation_families.prediction_accuracy_mean),
                    prediction_error_before_mean = COALESCE(excluded.prediction_error_before_mean, transformation_families.prediction_error_before_mean),
                    prediction_error_after_mean = COALESCE(excluded.prediction_error_after_mean, transformation_families.prediction_error_after_mean)
                """,
                (
                    family_id,
                    family_signature,
                    family_signature,
                    effect_type,
                    action_group,
                    polarity,
                    support_total,
                    member_count,
                    first_seen,
                    last_seen,
                    stability_score,
                    prediction_lift,
                    prediction_accuracy_mean,
                    prediction_error_before_mean,
                    prediction_error_after_mean,
                ),
            )
            inserted_families += 1
            for member in members:
                state_conn.execute(
                    """
                    INSERT INTO family_members (
                        family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(family_signature, contingency_key) DO UPDATE SET
                        support_count = MAX(family_members.support_count, excluded.support_count),
                        first_seen_global_step = MIN(family_members.first_seen_global_step, excluded.first_seen_global_step),
                        last_seen_global_step = MAX(family_members.last_seen_global_step, excluded.last_seen_global_step)
                    """,
                    (
                        family_signature,
                        str(member["canonical_key"]),
                        int(member["support_count"] or 0),
                        int(member["first_seen_global_step"] or 0),
                        int(member["last_seen_global_step"] or 0),
                    ),
                )
                inserted_members += 1
        contingency_to_family = {
            str(row["contingency_key"]): str(row["family_signature"])
            for row in state_conn.execute(
                "SELECT family_signature, contingency_key FROM family_members ORDER BY family_signature ASC, contingency_key ASC"
            ).fetchall()
        }
        carrier_contingency_rows = state_conn.execute(
            """
            SELECT carrier_signature, linked_key, support_count, first_seen_global_step, last_seen_global_step
            FROM carrier_links
            WHERE linked_type = 'contingency'
            ORDER BY carrier_signature ASC, linked_key ASC
            """
        ).fetchall()
        for row in carrier_contingency_rows:
            family_signature = contingency_to_family.get(str(row["linked_key"]))
            if family_signature is None:
                continue
            state_conn.execute(
                """
                INSERT INTO carrier_links (
                    carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
                ) VALUES (?, 'family', ?, ?, ?, ?)
                ON CONFLICT(carrier_signature, linked_type, linked_key) DO UPDATE SET
                    support_count = MAX(carrier_links.support_count, excluded.support_count),
                    first_seen_global_step = MIN(carrier_links.first_seen_global_step, excluded.first_seen_global_step),
                    last_seen_global_step = MAX(carrier_links.last_seen_global_step, excluded.last_seen_global_step)
                """,
                (
                    str(row["carrier_signature"]),
                    family_signature,
                    int(row["support_count"] or 0),
                    int(row["first_seen_global_step"] or 0),
                    int(row["last_seen_global_step"] or 0),
                ),
            )
        transformation_families_after = int(state_conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0])
        family_members_after = int(state_conn.execute("SELECT COUNT(*) FROM family_members").fetchone()[0])
        summary.update(
            {
                "compact_family_repair_used": inserted_families > 0 or inserted_members > 0,
                "compact_family_repair_reason": (
                    "repaired_from_stable_contingencies"
                    if inserted_families > 0 or inserted_members > 0
                    else "family_substrate_already_present"
                ),
                "transformation_families_after": transformation_families_after,
                "family_members_after": family_members_after,
                "compact_family_repair_family_count": max(0, transformation_families_after - family_count),
                "compact_family_repair_member_count": max(0, family_members_after - family_members_before),
            }
        )
        _write_family_repair_summary(state_conn, summary)
        state_conn.commit()
    try:
        with sqlite3.connect(paths.graph) as graph_conn:
            configure_compact_sqlite_connection(graph_conn, write=True)
            graph_conn.row_factory = sqlite3.Row
            graph_node_count = 0
            graph_edge_count = 0
            for family_signature, members in grouped.items():
                family_node = "M2:family:" + sha1(family_signature.encode("utf-8")).hexdigest()[:20]
                graph_conn.execute(
                    """
                    INSERT INTO graph_nodes (
                        node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count
                    ) VALUES (?, 'TransformationFamily', ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        first_seen_global_step = MIN(graph_nodes.first_seen_global_step, excluded.first_seen_global_step),
                        last_seen_global_step = MAX(graph_nodes.last_seen_global_step, excluded.last_seen_global_step),
                        support_count = MAX(graph_nodes.support_count, excluded.support_count)
                    """,
                    (
                        family_node,
                        family_signature,
                        min(int(row["first_seen_global_step"] or 0) for row in members),
                        max(int(row["last_seen_global_step"] or 0) for row in members),
                        sum(int(row["support_count"] or 0) for row in members),
                    ),
                )
                graph_node_count += 1
                for member in members:
                    contingency_key = str(member["canonical_key"])
                    contingency_node = "M1:contingency:" + sha1(contingency_key.encode("utf-8")).hexdigest()[:20]
                    edge_id = sha1(f"{contingency_node}|{family_node}|supports".encode("utf-8")).hexdigest()[:24]
                    graph_conn.execute(
                        """
                        INSERT INTO graph_edges (
                            edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight
                        ) VALUES (?, ?, ?, 'supports', ?, ?, ?, 1.0)
                        ON CONFLICT(edge_id) DO UPDATE SET
                            first_seen_global_step = MIN(graph_edges.first_seen_global_step, excluded.first_seen_global_step),
                            last_seen_global_step = MAX(graph_edges.last_seen_global_step, excluded.last_seen_global_step),
                            support_count = MAX(graph_edges.support_count, excluded.support_count),
                            weight = MAX(graph_edges.weight, excluded.weight)
                        """,
                        (
                            edge_id,
                            contingency_node,
                            family_node,
                            int(member["first_seen_global_step"] or 0),
                            int(member["last_seen_global_step"] or 0),
                            int(member["support_count"] or 0),
                        ),
                    )
                    graph_edge_count += 1
            graph_conn.commit()
            summary["compact_family_repair_graph_node_count"] = graph_node_count
            summary["compact_family_repair_graph_edge_count"] = graph_edge_count
            summary["contingency_graph_edge_count"] = graph_edge_count
    except Exception as exc:
        summary["compact_family_repair_error"] = str(exc)
    return summary


def _backfill_transformation_family_prediction_lift(state_conn: sqlite3.Connection) -> int:
    rows = state_conn.execute(
        """
        SELECT
            tf.canonical_signature,
            sc.prediction_accuracy,
            sc.prediction_error_before,
            sc.prediction_error_after
        FROM transformation_families tf
        JOIN family_members fm
          ON fm.family_signature = tf.canonical_signature
        JOIN stable_contingencies sc
          ON sc.canonical_key = fm.contingency_key
        ORDER BY tf.canonical_signature ASC
        """
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["canonical_signature"]), []).append(row)
    updated = 0
    for canonical_signature, members in grouped.items():
        prediction_accuracy_values = [
            float(row["prediction_accuracy"])
            for row in members
            if row["prediction_accuracy"] is not None
        ]
        prediction_error_before_values = [
            float(row["prediction_error_before"])
            for row in members
            if row["prediction_error_before"] is not None
        ]
        prediction_error_after_values = [
            float(row["prediction_error_after"])
            for row in members
            if row["prediction_error_after"] is not None
        ]
        prediction_accuracy_mean = (
            float(sum(prediction_accuracy_values) / len(prediction_accuracy_values))
            if prediction_accuracy_values
            else None
        )
        prediction_error_before_mean = (
            float(sum(prediction_error_before_values) / len(prediction_error_before_values))
            if prediction_error_before_values
            else None
        )
        prediction_error_after_mean = (
            float(sum(prediction_error_after_values) / len(prediction_error_after_values))
            if prediction_error_after_values
            else None
        )
        if prediction_error_before_mean is not None and prediction_error_after_mean is not None:
            prediction_lift = float(prediction_error_before_mean - prediction_error_after_mean)
        elif prediction_accuracy_mean is not None:
            prediction_lift = float(prediction_accuracy_mean)
        else:
            prediction_lift = None
        if (
            prediction_lift is None
            and prediction_accuracy_mean is None
            and prediction_error_before_mean is None
            and prediction_error_after_mean is None
        ):
            continue
        state_conn.execute(
            """
            UPDATE transformation_families
            SET
                prediction_lift = ?,
                prediction_accuracy_mean = ?,
                prediction_error_before_mean = ?,
                prediction_error_after_mean = ?
            WHERE canonical_signature = ?
            """,
            (
                prediction_lift,
                prediction_accuracy_mean,
                prediction_error_before_mean,
                prediction_error_after_mean,
                canonical_signature,
            ),
        )
        updated += 1
    return updated


def fold_epoch_raw_into_compact_memory(
    *,
    epoch_raw_dir: str | Path,
    memory_dir: str | Path,
    fold_config: CompactMemoryFoldConfig,
) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    raw_dir = Path(epoch_raw_dir)
    db_paths = sorted(raw_dir.rglob("*.sqlite"))
    live_graph_paths = sorted(raw_dir.rglob("live_graph_compact.json"))
    report = _load_json(raw_dir / "interaction_sampling_v05c_report.json") or {}
    temporal_rows = list(((report.get("temporal_milestones") or {}).get("by_game_sampler_seed") or []))
    current_summary = load_memory_summary(paths.summary_json)
    totals = {
        "stable_contingencies_added": 0,
        "stable_contingencies_inserted": 0,
        "transformation_families_added": 0,
        "transformation_families_inserted": 0,
        "family_members_inserted": 0,
        "carrier_candidates_added": 0,
        "contradiction_clusters_added": 0,
        "raw_contingency_rows_seen": 0,
        "raw_dbs_without_contingency_table": 0,
        "replay_queue_size": 0,
        "representative_examples_retained": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "graph_live_exports_ingested": 0,
        "db_files_folded": len(db_paths),
        "total_interactions_seen": int(current_summary.get("total_interactions_seen", 0) or 0)
        + int((report.get("validation") or {}).get("memory_record_count", 0) or 0),
        **_graph_edge_total_fields(),
    }

    parallel_workers = _fold_parallel_worker_count(len(db_paths))
    if parallel_workers > 1:
        return _fold_epoch_raw_into_compact_memory_parallel(
            paths=paths,
            raw_dir=raw_dir,
            db_paths=db_paths,
            live_graph_paths=live_graph_paths,
            temporal_rows=temporal_rows,
            current_summary=current_summary,
            totals=totals,
            fold_config=fold_config,
            parallel_workers=parallel_workers,
        )

    with (
        sqlite3.connect(paths.current_state) as state_conn,
        sqlite3.connect(paths.graph) as graph_conn,
        sqlite3.connect(paths.replay_queue) as replay_conn,
    ):
        for db_path in db_paths:
            _fold_single_db(
                db_path=db_path,
                state_conn=state_conn,
                graph_conn=graph_conn,
                replay_conn=replay_conn,
                fold_config=fold_config,
                totals=totals,
            )
        for graph_path in live_graph_paths:
            _ingest_live_graph_export(graph_conn, graph_path, fold_config, totals=totals)
            totals["graph_live_exports_ingested"] += 1
        _upsert_temporal_milestones(state_conn, temporal_rows)
        _trim_representative_examples(state_conn, fold_config)
        _trim_replay_queue(replay_conn, fold_config)
        summary = _build_memory_summary_from_connections(
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            paths=paths,
        )
        summary.update(totals)
        summary["fold_summary"] = dict(totals)
        _write_graph_epoch_summary(state_conn, graph_conn, fold_config, totals)
        _write_memory_summary_table(state_conn, summary)
        paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
    return totals


def fold_single_sampling_db_into_main_compact_memory(
    *,
    db_path: str | Path,
    memory_dir: str | Path,
    fold_config: CompactMemoryFoldConfig,
    finalize_after_fold: bool = False,
    sqlite_synchronous: str = "NORMAL",
    temporary_shard: bool = False,
    busy_timeout_ms: int = 10000,
) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    sqlite_path = Path(db_path)
    totals = {
        "stable_contingencies_added": 0,
        "stable_contingencies_inserted": 0,
        "transformation_families_added": 0,
        "transformation_families_inserted": 0,
        "family_members_inserted": 0,
        "carrier_candidates_added": 0,
        "contradiction_clusters_added": 0,
        "raw_contingency_rows_seen": 0,
        "raw_dbs_without_contingency_table": 0,
        "replay_queue_size": 0,
        "representative_examples_retained": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "graph_live_exports_ingested": 0,
        "db_files_folded": 1,
        **_graph_edge_total_fields(),
    }
    live_graph_path = sqlite_path.with_name("live_graph_compact.json")
    with (
        sqlite3.connect(paths.current_state, timeout=max(1.0, float(busy_timeout_ms) / 1000.0)) as state_conn,
        sqlite3.connect(paths.graph, timeout=max(1.0, float(busy_timeout_ms) / 1000.0)) as graph_conn,
        sqlite3.connect(paths.replay_queue, timeout=max(1.0, float(busy_timeout_ms) / 1000.0)) as replay_conn,
    ):
        configure_compact_sqlite_connection(state_conn, write=True, synchronous=sqlite_synchronous, temporary_shard=temporary_shard, busy_timeout_ms=busy_timeout_ms)
        configure_compact_sqlite_connection(graph_conn, write=True, synchronous=sqlite_synchronous, temporary_shard=temporary_shard, busy_timeout_ms=busy_timeout_ms)
        configure_compact_sqlite_connection(replay_conn, write=True, synchronous=sqlite_synchronous, temporary_shard=temporary_shard, busy_timeout_ms=busy_timeout_ms)
        _fold_single_db(
            db_path=sqlite_path,
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            fold_config=fold_config,
            totals=totals,
            busy_timeout_ms=busy_timeout_ms,
        )
        if live_graph_path.exists():
            _ingest_live_graph_export(graph_conn, live_graph_path, fold_config, totals=totals)
            totals["graph_live_exports_ingested"] += 1
        if finalize_after_fold:
            _trim_representative_examples(state_conn, fold_config)
            _trim_replay_queue(replay_conn, fold_config)
            summary = _build_memory_summary_from_connections(
                state_conn=state_conn,
                graph_conn=graph_conn,
                replay_conn=replay_conn,
                paths=paths,
            )
            existing_fold_summary: dict[str, Any] = {}
            row = state_conn.execute("SELECT value_json FROM memory_summary WHERE key = 'fold_summary'").fetchone()
            if row is not None and row[0] is not None:
                try:
                    payload = json.loads(row[0])
                    if isinstance(payload, dict):
                        existing_fold_summary = dict(payload)
                except Exception:
                    existing_fold_summary = {}
            summary.update(totals)
            existing_fold_summary.update(dict(totals))
            summary["fold_summary"] = existing_fold_summary
            summary["direct_streaming_fold"] = {
                "db_path": str(sqlite_path),
                "global_step_start": int(fold_config.global_step_start),
                "global_step_end": int(fold_config.global_step_end),
            }
            _write_graph_epoch_summary(state_conn, graph_conn, fold_config, totals)
            _write_memory_summary_table(state_conn, summary)
            paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
    return totals


def finalize_main_compact_memory(
    *,
    memory_dir: str | Path,
    fold_config: CompactMemoryFoldConfig,
    finalize_mode: str = "full",
) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    mode = str(finalize_mode or "full").strip().lower()
    if mode not in {"none", "summary_only", "full"}:
        mode = "full"
    with (
        sqlite3.connect(paths.current_state) as state_conn,
        sqlite3.connect(paths.graph) as graph_conn,
        sqlite3.connect(paths.replay_queue) as replay_conn,
    ):
        configure_compact_sqlite_connection(state_conn, write=True)
        configure_compact_sqlite_connection(graph_conn, write=True)
        configure_compact_sqlite_connection(replay_conn, write=True)
        if mode == "full":
            _trim_representative_examples(state_conn, fold_config)
            _trim_replay_queue(replay_conn, fold_config)
        elif mode == "none":
            summary = load_memory_summary(paths.summary_json)
            summary["fold_summary"] = dict(summary.get("fold_summary", {}) or {})
            summary["fold_summary"].update(
                {
                    "global_step_start": int(fold_config.global_step_start),
                    "global_step_end": int(fold_config.global_step_end),
                    "finalize_mode": mode,
                }
            )
            _write_graph_epoch_summary(state_conn, graph_conn, fold_config, summary.get("fold_summary", {}))
            _write_memory_summary_table(state_conn, summary)
            paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            state_conn.commit()
            graph_conn.commit()
            replay_conn.commit()
            return summary
        summary = _build_memory_summary_from_connections(
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            paths=paths,
        )
        existing_fold_summary: dict[str, Any] = {}
        row = state_conn.execute("SELECT value_json FROM memory_summary WHERE key = 'fold_summary'").fetchone()
        if row is not None and row[0] is not None:
            try:
                payload = json.loads(row[0])
                if isinstance(payload, dict):
                    existing_fold_summary = dict(payload)
            except Exception:
                existing_fold_summary = {}
        existing_fold_summary.update(
            {
                "global_step_start": int(fold_config.global_step_start),
                "global_step_end": int(fold_config.global_step_end),
                "finalize_mode": mode,
            }
        )
        summary["fold_summary"] = existing_fold_summary
        _write_graph_epoch_summary(state_conn, graph_conn, fold_config, existing_fold_summary)
        _write_memory_summary_table(state_conn, summary)
        paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
    return summary


def fold_sampling_job_sidecars_into_compact_memory(
    *,
    db_path: str | Path,
    memory_dir: str | Path,
    fold_config: CompactMemoryFoldConfig,
    delete_after_merge: bool = True,
) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    sqlite_path = Path(db_path)
    live_graph_path = sqlite_path.with_name("live_graph_compact.json")
    carrier_path = sqlite_path.with_name("carrier_candidates.json")
    contradiction_path = sqlite_path.with_name("context_contradictions.json")
    memory_summary_path = sqlite_path.with_name("memory_lifecycle_summary.json")
    sidecars_present = any(path.exists() for path in (live_graph_path, carrier_path, contradiction_path, memory_summary_path))
    totals = {
        "graph_live_exports_ingested": 0,
        "carrier_candidates_added": 0,
        "deleted_sidecar_files": [],
        "sidecar_files_present": int(sidecars_present),
        **_graph_edge_total_fields(),
    }
    if not sidecars_present:
        return totals

    with (
        sqlite3.connect(paths.current_state) as state_conn,
        sqlite3.connect(paths.graph) as graph_conn,
        sqlite3.connect(paths.replay_queue) as replay_conn,
        sqlite3.connect(sqlite_path) as raw_conn,
    ):
        raw_conn.row_factory = sqlite3.Row
        if live_graph_path.exists():
            _ingest_live_graph_export(graph_conn, live_graph_path, fold_config, totals=totals)
            totals["graph_live_exports_ingested"] += 1
        if carrier_path.exists():
            _fold_carrier_candidates_sidecar(
                raw_conn=raw_conn,
                carrier_path=carrier_path,
                state_conn=state_conn,
                graph_conn=graph_conn,
                fold_config=fold_config,
                totals=totals,
            )
        summary = _build_memory_summary_from_connections(
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            paths=paths,
        )
        summary.update({key: value for key, value in totals.items() if key not in {"deleted_sidecar_files"}})
        summary["incremental_sidecar_fold"] = {
            "db_path": str(sqlite_path),
            "graph_live_exports_ingested": int(totals["graph_live_exports_ingested"]),
            "carrier_candidates_added": int(totals["carrier_candidates_added"]),
        }
        _write_memory_summary_table(state_conn, summary)
        paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()

    if delete_after_merge:
        for path in (live_graph_path, carrier_path, contradiction_path, memory_summary_path):
            if not path.exists():
                continue
            try:
                path.unlink()
                totals["deleted_sidecar_files"].append(str(path))
            except OSError:
                pass
    return totals


def fold_sampling_job_sidecars_into_compact_memory_shard(
    *,
    db_path: str | Path,
    shard_memory_dir: str | Path,
    fold_config: CompactMemoryFoldConfig,
    delete_after_merge: bool = True,
) -> dict[str, Any]:
    shard_paths = ensure_memory_layout(shard_memory_dir)
    sqlite_path = Path(db_path)
    live_graph_path = sqlite_path.with_name("live_graph_compact.json")
    carrier_path = sqlite_path.with_name("carrier_candidates.json")
    contradiction_path = sqlite_path.with_name("context_contradictions.json")
    memory_summary_path = sqlite_path.with_name("memory_lifecycle_summary.json")
    sidecars_present = any(path.exists() for path in (live_graph_path, carrier_path, contradiction_path, memory_summary_path))
    totals = {
        "graph_live_exports_ingested": 0,
        "carrier_candidates_added": 0,
        "deleted_sidecar_files": [],
        "sidecar_files_present": int(sidecars_present),
        "shard_memory_dir": str(shard_paths.root),
        **_graph_edge_total_fields(),
    }
    if not sidecars_present:
        return totals

    with (
        sqlite3.connect(shard_paths.current_state) as state_conn,
        sqlite3.connect(shard_paths.graph) as graph_conn,
        sqlite3.connect(shard_paths.replay_queue) as replay_conn,
        sqlite3.connect(sqlite_path) as raw_conn,
    ):
        raw_conn.row_factory = sqlite3.Row
        if live_graph_path.exists():
            _ingest_live_graph_export(graph_conn, live_graph_path, fold_config, totals=totals)
            totals["graph_live_exports_ingested"] += 1
        if carrier_path.exists():
            _fold_carrier_candidates_sidecar(
                raw_conn=raw_conn,
                carrier_path=carrier_path,
                state_conn=state_conn,
                graph_conn=graph_conn,
                fold_config=fold_config,
                totals=totals,
            )
        summary = _build_memory_summary_from_connections(
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            paths=shard_paths,
        )
        summary.update({key: value for key, value in totals.items() if key not in {"deleted_sidecar_files"}})
        summary["incremental_sidecar_fold"] = {
            "db_path": str(sqlite_path),
            "graph_live_exports_ingested": int(totals["graph_live_exports_ingested"]),
            "carrier_candidates_added": int(totals["carrier_candidates_added"]),
            "shard_memory_dir": str(shard_paths.root),
        }
        _write_memory_summary_table(state_conn, summary)
        shard_paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()

    if delete_after_merge:
        for path in (live_graph_path, carrier_path, contradiction_path, memory_summary_path):
            if not path.exists():
                continue
            try:
                path.unlink()
                totals["deleted_sidecar_files"].append(str(path))
            except OSError:
                pass
    return totals


def merge_compact_memory_shards_into_main(
    *,
    memory_dir: str | Path,
    shard_dirs: list[str | Path],
    fold_config: CompactMemoryFoldConfig,
    parallel_workers: int | None = None,
    progress: bool = False,
    progress_desc: str = "merge fold shards",
) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    merged_dirs = [Path(item) for item in shard_dirs if Path(item).exists()]
    requested_parallel_workers = max(1, int(parallel_workers or 1))
    effective_parallel_workers = min(requested_parallel_workers, max(1, len(merged_dirs)))
    temp_root: Path | None = None
    try:
        if len(merged_dirs) > 1 and effective_parallel_workers > 1:
            merged_dirs, temp_root = _reduce_compact_memory_shards_parallel(
                shard_dirs=merged_dirs,
                parent_memory_dir=paths.root,
                fold_config=fold_config,
                parallel_workers=effective_parallel_workers,
            )
        with (
            sqlite3.connect(paths.current_state) as state_conn,
            sqlite3.connect(paths.graph) as graph_conn,
            sqlite3.connect(paths.replay_queue) as replay_conn,
        ):
            configure_compact_sqlite_connection(state_conn, write=True)
            configure_compact_sqlite_connection(graph_conn, write=True)
            configure_compact_sqlite_connection(replay_conn, write=True)
            merge_totals = _graph_edge_total_fields()
            iterator = sorted(merged_dirs)
            if progress:
                iterator = tqdm(iterator, desc=progress_desc, unit="shard", dynamic_ncols=True, leave=True)
            for shard_dir in iterator:
                _merge_compact_memory_dir_into_main(
                    temp_dir=shard_dir,
                    state_conn=state_conn,
                    graph_conn=graph_conn,
                    replay_conn=replay_conn,
                    fold_config=fold_config,
                    totals=merge_totals,
                    enforce_graph_caps_after_merge=True,
                )
            summary = _build_memory_summary_from_connections(
                state_conn=state_conn,
                graph_conn=graph_conn,
                replay_conn=replay_conn,
                paths=paths,
            )
            summary.update(merge_totals)
            summary["parallel_sidecar_shard_merge"] = {
                "merged_shard_count": len(merged_dirs),
                "shard_dirs": [str(path) for path in merged_dirs],
                "parallel_workers": int(effective_parallel_workers),
            }
            _write_graph_epoch_summary(state_conn, graph_conn, fold_config, merge_totals)
            _write_memory_summary_table(state_conn, summary)
            paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            state_conn.commit()
            graph_conn.commit()
            replay_conn.commit()
        return summary
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)


def _reduce_compact_memory_shards_parallel(
    *,
    shard_dirs: list[Path],
    parent_memory_dir: Path,
    fold_config: CompactMemoryFoldConfig,
    parallel_workers: int,
) -> tuple[list[Path], Path]:
    round_inputs = sorted(Path(path) for path in shard_dirs if Path(path).exists())
    temp_root = Path(tempfile.mkdtemp(prefix="compact_merge_", dir=str(parent_memory_dir)))
    round_index = 0
    while len(round_inputs) > 1:
        group_size = max(2, int(np.ceil(len(round_inputs) / max(1, parallel_workers))))
        grouped_inputs = [
            round_inputs[index:index + group_size]
            for index in range(0, len(round_inputs), group_size)
        ]
        next_round: list[Path] = []
        with ProcessPoolExecutor(max_workers=min(parallel_workers, len(grouped_inputs)), max_tasks_per_child=1) as executor:
            futures = []
            for group_index, group in enumerate(grouped_inputs):
                if len(group) == 1:
                    next_round.append(group[0])
                    continue
                output_dir = temp_root / f"round_{round_index:04d}" / f"group_{group_index:04d}"
                next_round.append(output_dir)
                futures.append(
                    executor.submit(
                        _merge_compact_memory_dirs_worker,
                        [str(path) for path in group],
                        str(output_dir),
                        fold_config,
                    )
                )
            future_iterator = futures
            if futures:
                future_iterator = tqdm(
                    futures,
                    desc=f"reduce fold shards r{round_index}",
                    unit="merge",
                    dynamic_ncols=True,
                    leave=True,
                )
            for future in future_iterator:
                future.result()
        round_inputs = sorted(next_round)
        round_index += 1
    return round_inputs, temp_root


def _merge_compact_memory_dirs_worker(
    shard_dirs: list[str],
    output_dir: str,
    fold_config: CompactMemoryFoldConfig,
) -> dict[str, Any]:
    output_paths = ensure_memory_layout(output_dir)
    with (
        sqlite3.connect(output_paths.current_state) as state_conn,
        sqlite3.connect(output_paths.graph) as graph_conn,
        sqlite3.connect(output_paths.replay_queue) as replay_conn,
    ):
        configure_compact_sqlite_connection(state_conn, write=True)
        configure_compact_sqlite_connection(graph_conn, write=True)
        configure_compact_sqlite_connection(replay_conn, write=True)
        merge_totals = _graph_edge_total_fields()
        for shard_dir in sorted(Path(item) for item in shard_dirs):
            _merge_compact_memory_dir_into_main(
                temp_dir=shard_dir,
                state_conn=state_conn,
                graph_conn=graph_conn,
                replay_conn=replay_conn,
                fold_config=fold_config,
                totals=merge_totals,
                enforce_graph_caps_after_merge=False,
            )
        summary = _build_memory_summary_from_connections(
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            paths=output_paths,
        )
        summary.update(merge_totals)
        summary["parallel_intermediate_shard_merge"] = {
            "input_shard_count": len(shard_dirs),
            "input_shard_dirs": [str(item) for item in shard_dirs],
        }
        _write_graph_epoch_summary(state_conn, graph_conn, fold_config, merge_totals)
        _write_memory_summary_table(state_conn, summary)
        output_paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
    return summary


def _fold_epoch_raw_into_compact_memory_parallel(
    *,
    paths: CompactMemoryPaths,
    raw_dir: Path,
    db_paths: list[Path],
    live_graph_paths: list[Path],
    temporal_rows: list[dict[str, Any]],
    current_summary: dict[str, Any],
    totals: dict[str, Any],
    fold_config: CompactMemoryFoldConfig,
    parallel_workers: int,
) -> dict[str, Any]:
    del raw_dir
    chunk_count = min(parallel_workers, len(db_paths))
    chunks = [db_paths[index::chunk_count] for index in range(chunk_count)]
    chunks = [chunk for chunk in chunks if chunk]
    temp_root = Path(tempfile.mkdtemp(prefix="compact_fold_", dir=str(paths.root)))
    temp_dirs: list[Path] = []
    chunk_totals: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=len(chunks), max_tasks_per_child=1) as executor:
            futures = []
            for index, chunk in enumerate(chunks):
                temp_dir = temp_root / f"chunk_{index:04d}"
                temp_dirs.append(temp_dir)
                futures.append(
                    executor.submit(
                        _fold_db_chunk_worker,
                        [str(path) for path in chunk],
                        str(temp_dir),
                        fold_config,
                    )
                )
            for future in futures:
                chunk_totals.append(dict(future.result()))

        with (
            sqlite3.connect(paths.current_state) as state_conn,
            sqlite3.connect(paths.graph) as graph_conn,
            sqlite3.connect(paths.replay_queue) as replay_conn,
        ):
            for temp_dir in sorted(temp_dirs):
                _merge_compact_memory_dir_into_main(
                    temp_dir=temp_dir,
                    state_conn=state_conn,
                    graph_conn=graph_conn,
                    replay_conn=replay_conn,
                    fold_config=fold_config,
                    totals=totals,
                    enforce_graph_caps_after_merge=True,
                )
            for graph_path in live_graph_paths:
                _ingest_live_graph_export(graph_conn, graph_path, fold_config, totals=totals)
                totals["graph_live_exports_ingested"] += 1
            _upsert_temporal_milestones(state_conn, temporal_rows)
            _trim_representative_examples(state_conn, fold_config)
            _trim_replay_queue(replay_conn, fold_config)
            summary = _build_memory_summary_from_connections(
                state_conn=state_conn,
                graph_conn=graph_conn,
                replay_conn=replay_conn,
                paths=paths,
            )
            for key in (
                "stable_contingencies_added",
                "stable_contingencies_inserted",
                "transformation_families_added",
                "transformation_families_inserted",
                "family_members_inserted",
                "carrier_candidates_added",
                "contradiction_clusters_added",
                "raw_contingency_rows_seen",
                "raw_dbs_without_contingency_table",
            ):
                totals[key] = sum(int(item.get(key, 0) or 0) for item in chunk_totals)
            totals["replay_queue_size"] = int(summary.get("replay_queue_size", 0) or 0)
            totals["representative_examples_retained"] = int(summary.get("representative_example_count", 0) or 0)
            totals["graph_node_count"] = int(summary.get("graph_node_count", 0) or 0)
            totals["graph_edge_count"] = int(summary.get("graph_edge_count", 0) or 0)
            totals["db_files_folded"] = len(db_paths)
            for key in _graph_edge_total_fields().keys():
                shard_value = sum(int(item.get(key, 0) or 0) for item in chunk_totals)
                totals[key] = int(totals.get(key, 0) or 0) + shard_value
            summary.update(totals)
            summary["fold_summary"] = dict(totals)
            _write_graph_epoch_summary(state_conn, graph_conn, fold_config, totals)
            _write_memory_summary_table(state_conn, summary)
            paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            state_conn.commit()
            graph_conn.commit()
            replay_conn.commit()
        return totals
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _fold_parallel_worker_count(db_count: int) -> int:
    if db_count < 8:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(30, cpu_count, db_count))


def _fold_db_chunk_worker(db_paths: list[str], temp_memory_dir: str, fold_config: CompactMemoryFoldConfig) -> dict[str, Any]:
    temp_paths = ensure_memory_layout(temp_memory_dir)
    totals = {
        "stable_contingencies_added": 0,
        "stable_contingencies_inserted": 0,
        "transformation_families_added": 0,
        "transformation_families_inserted": 0,
        "family_members_inserted": 0,
        "carrier_candidates_added": 0,
        "contradiction_clusters_added": 0,
        "raw_contingency_rows_seen": 0,
        "raw_dbs_without_contingency_table": 0,
        "replay_queue_size": 0,
        "representative_examples_retained": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "graph_live_exports_ingested": 0,
        "db_files_folded": len(db_paths),
        "total_interactions_seen": 0,
        **_graph_edge_total_fields(),
    }
    with (
        sqlite3.connect(temp_paths.current_state) as state_conn,
        sqlite3.connect(temp_paths.graph) as graph_conn,
        sqlite3.connect(temp_paths.replay_queue) as replay_conn,
    ):
        for db_path in [Path(item) for item in db_paths]:
            _fold_single_db(
                db_path=db_path,
                state_conn=state_conn,
                graph_conn=graph_conn,
                replay_conn=replay_conn,
                fold_config=fold_config,
                totals=totals,
            )
        summary = _build_memory_summary_from_connections(
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            paths=temp_paths,
        )
        totals["replay_queue_size"] = int(summary.get("replay_queue_size", 0) or 0)
        totals["representative_examples_retained"] = int(summary.get("representative_example_count", 0) or 0)
        totals["graph_node_count"] = int(summary.get("graph_node_count", 0) or 0)
        totals["graph_edge_count"] = int(summary.get("graph_edge_count", 0) or 0)
        summary.update(totals)
        summary["fold_summary"] = dict(totals)
        _write_graph_epoch_summary(state_conn, graph_conn, fold_config, totals)
        _write_memory_summary_table(state_conn, summary)
        temp_paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
    return totals


def fold_live_system_into_compact_memory(system: Any, memory_dir: str | Path) -> dict[str, Any]:
    paths = ensure_memory_layout(memory_dir)
    global_step_end = int(getattr(system, "step_count", 0) or 0) + int(getattr(system.config, "global_step_offset", 0) or 0)
    global_step_start = max(1, global_step_end)
    graph_rows = {}
    live_connection = getattr(system, "connection", None)
    if getattr(system, "graph", None) is not None and hasattr(system.graph, "export_compact_rows"):
        graph_rows = system.graph.export_compact_rows()
    family_id_to_signature = {
        int(family.id): canonical_family_signature_from_family(family)
        for family in getattr(system.clusterer, "families", {}).values()
    }
    semantic_family_fallback_count = 0
    with (
        sqlite3.connect(paths.current_state) as state_conn,
        sqlite3.connect(paths.graph) as graph_conn,
        sqlite3.connect(paths.replay_queue) as replay_conn,
    ):
        for contingency in getattr(system.contingency_learner, "stable_contingencies", lambda: [])():
            canonical_signature = family_id_to_signature.get(int(contingency.transformation_family))
            if canonical_signature is None:
                canonical_signature = f"unknown_live_family:{int(contingency.transformation_family)}"
                semantic_family_fallback_count += 1
            canonical_key = canonicalize_context_action_effect(
                context_signature=json.dumps(list(contingency.context_signature)),
                action=contingency.action,
                effect_signature=canonical_signature,
            )
            state_conn.execute(
                """
                INSERT INTO stable_contingencies (
                    contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                    first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                    mean_replay_priority, representative_example_count, prediction_attempt_count,
                    prediction_success_count, prediction_accuracy, prediction_error_before,
                    prediction_error_after, normalized_contingency_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_key) DO UPDATE SET
                    support_count = COALESCE(stable_contingencies.support_count, 0) + COALESCE(excluded.support_count, 0),
                    context_level = MAX(stable_contingencies.context_level, excluded.context_level),
                    last_seen_global_step = MAX(stable_contingencies.last_seen_global_step, excluded.last_seen_global_step),
                    stability_score = CAST(
                        COALESCE(stable_contingencies.support_count, 0) + COALESCE(excluded.support_count, 0)
                        AS REAL
                    ) / 20.0,
                    prediction_attempt_count = COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0),
                    prediction_success_count = COALESCE(stable_contingencies.prediction_success_count, 0) + COALESCE(excluded.prediction_success_count, 0),
                    prediction_accuracy = CASE
                        WHEN (COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0)) > 0
                        THEN CAST(
                            (COALESCE(stable_contingencies.prediction_success_count, 0) + COALESCE(excluded.prediction_success_count, 0)) AS REAL
                        ) / CAST(
                            (COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0)) AS REAL
                        )
                        ELSE NULL
                    END,
                    normalized_contingency_key = COALESCE(stable_contingencies.normalized_contingency_key, excluded.normalized_contingency_key)
                """,
                (
                    int(contingency.id),
                    canonical_key,
                    None,
                    None,
                    int(contingency.context_level),
                    int(contingency.action),
                    canonical_signature,
                    int(contingency.support_count),
                    global_step_start,
                    global_step_end,
                    float(contingency.confidence),
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    normalized_contingency_identity(
                        context_level=int(contingency.context_level),
                        action=int(contingency.action),
                        effect_signature=canonical_signature,
                    ),
                ),
            )
            state_conn.execute(
                """
                INSERT INTO family_members (
                    family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(family_signature, contingency_key) DO UPDATE SET
                    support_count = family_members.support_count + excluded.support_count,
                    first_seen_global_step = MIN(family_members.first_seen_global_step, excluded.first_seen_global_step),
                    last_seen_global_step = MAX(family_members.last_seen_global_step, excluded.last_seen_global_step)
                """,
                (canonical_signature, canonical_key, int(contingency.support_count), global_step_start, global_step_end),
            )
            _upsert_observation_graph(
                graph_conn,
                fold_config=CompactMemoryFoldConfig(global_step_start=global_step_start, global_step_end=global_step_end),
                game=None,
                sampler=None,
                context_signature=json.dumps(list(contingency.context_signature)),
                action=int(contingency.action),
                effect_signature=canonical_signature,
                contingency_key=canonical_key,
                family_signature=canonical_signature,
                carrier_signature=None,
                contradiction_key=None,
                replay_id=None,
            )
        for family in getattr(system.clusterer, "families", {}).values():
            signature = canonical_family_signature_from_family(family)
            stable_id = stable_family_int_id(signature)
            _upsert_family_identity_map(state_conn, signature, stable_id)
            state_conn.execute(
                """
                INSERT INTO transformation_families (
                    family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                    support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score,
                    prediction_lift, prediction_accuracy_mean, prediction_error_before_mean, prediction_error_after_mean
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_signature) DO UPDATE SET
                    support_count = MAX(transformation_families.support_count, excluded.support_count),
                    member_count = MAX(transformation_families.member_count, excluded.member_count),
                    last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
                    stability_score = MAX(transformation_families.stability_score, excluded.stability_score),
                    prediction_lift = COALESCE(excluded.prediction_lift, transformation_families.prediction_lift),
                    prediction_accuracy_mean = COALESCE(excluded.prediction_accuracy_mean, transformation_families.prediction_accuracy_mean),
                    prediction_error_before_mean = COALESCE(excluded.prediction_error_before_mean, transformation_families.prediction_error_before_mean),
                    prediction_error_after_mean = COALESCE(excluded.prediction_error_after_mean, transformation_families.prediction_error_after_mean)
                """,
                (
                    stable_id,
                    signature,
                    signature,
                    "live_export",
                    "unknown",
                    "unknown",
                    int(family.support_count),
                    len(family.member_delta_ids),
                    global_step_start,
                    global_step_end,
                    float(family.support_count),
                    None,
                    None,
                    None,
                    None,
                ),
            )
        for replay in getattr(system.memory_lifecycle, "replay_candidates", {}).values():
            replay_payload = _build_minimal_replay_payload(replay.to_dict(), {})
            replay_conn.execute(
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
                    str(replay.interaction_id),
                    "interaction",
                    str(replay.interaction_id),
                    float(replay.replay_priority),
                    str(replay.reason),
                    global_step_start,
                    global_step_end,
                    json.dumps(replay_payload, sort_keys=True),
                ),
            )
        for candidate in getattr(system.carrier_tracker, "build_candidates", lambda: [])():
            is_emergent = int(candidate.status == "emergent_carrier" and candidate.carrier_source != "context_action_fallback")
            first_seen_step_value = (
                candidate.first_emergent_global_step
                if is_emergent and candidate.first_emergent_global_step is not None
                else candidate.first_seen_global_step
            )
            last_seen_step_value = candidate.last_seen_global_step
            carrier_timing_source = "real_evidence" if (first_seen_step_value is not None or last_seen_step_value is not None) else "fold_start_fallback"
            first_seen_step = int(first_seen_step_value) if first_seen_step_value is not None else global_step_start
            last_seen_step = int(last_seen_step_value) if last_seen_step_value is not None else global_step_end
            state_conn.execute(
                """
                INSERT INTO carrier_candidates (
                    carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                    first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(carrier_signature) DO UPDATE SET
                    support_count = MAX(carrier_candidates.support_count, excluded.support_count),
                    linked_family_count = MAX(carrier_candidates.linked_family_count, excluded.linked_family_count),
                    first_seen_global_step = MIN(carrier_candidates.first_seen_global_step, excluded.first_seen_global_step),
                    last_seen_global_step = MAX(carrier_candidates.last_seen_global_step, excluded.last_seen_global_step),
                    carrier_timing_source = CASE
                        WHEN carrier_candidates.carrier_timing_source = 'real_evidence'
                             OR excluded.carrier_timing_source = 'real_evidence'
                        THEN 'real_evidence'
                        WHEN COALESCE(carrier_candidates.carrier_timing_source, 'unknown') = COALESCE(excluded.carrier_timing_source, 'unknown')
                        THEN COALESCE(excluded.carrier_timing_source, carrier_candidates.carrier_timing_source, 'unknown')
                        WHEN carrier_candidates.carrier_timing_source = 'unknown'
                             AND excluded.carrier_timing_source = 'fold_start_fallback'
                        THEN 'fold_start_fallback'
                        WHEN carrier_candidates.carrier_timing_source = 'fold_start_fallback'
                             AND excluded.carrier_timing_source = 'unknown'
                        THEN 'fold_start_fallback'
                        ELSE 'mixed'
                    END,
                    stability_score = MAX(carrier_candidates.stability_score, excluded.stability_score),
                    is_emergent = MAX(carrier_candidates.is_emergent, excluded.is_emergent)
                """,
                (
                    candidate.carrier_id,
                    candidate.carrier_signature,
                    candidate.carrier_source,
                    int(candidate.support_count),
                    int(candidate.distinct_family_count),
                    first_seen_step,
                    last_seen_step,
                    carrier_timing_source,
                    float(candidate.prediction_lift),
                    is_emergent,
                ),
            )
            family_signature = None
            if candidate.family_id is not None:
                try:
                    family_signature = family_id_to_signature.get(int(candidate.family_id))
                except (TypeError, ValueError):
                    family_signature = None
            if family_signature is None and candidate.family_id is not None:
                family_signature = f"unknown_live_family:{candidate.family_id}"
                semantic_family_fallback_count += 1
            _upsert_carrier_link(
                state_conn,
                carrier_signature=str(candidate.carrier_signature),
                linked_type="family",
                linked_key=family_signature,
                fold_config=CompactMemoryFoldConfig(global_step_start=global_step_start, global_step_end=global_step_end),
                first_seen_global_step=first_seen_step,
                last_seen_global_step=last_seen_step,
            )
            _upsert_carrier_link(
                state_conn,
                carrier_signature=str(candidate.carrier_signature),
                linked_type="context",
                linked_key=None if candidate.context_signature is None else _normalize_jsonish(candidate.context_signature),
                fold_config=CompactMemoryFoldConfig(global_step_start=global_step_start, global_step_end=global_step_end),
                first_seen_global_step=first_seen_step,
                last_seen_global_step=last_seen_step,
            )
            if family_signature is not None:
                _upsert_observation_graph(
                    graph_conn,
                    fold_config=CompactMemoryFoldConfig(global_step_start=global_step_start, global_step_end=global_step_end),
                    game=None,
                    sampler=None,
                    context_signature=candidate.context_signature,
                    action=None,
                    effect_signature=None,
                    contingency_key=None,
                    family_signature=family_signature,
                    carrier_signature=candidate.carrier_signature,
                    contradiction_key=None,
                    replay_id=None,
                )
        if live_connection is not None:
            live_connection.row_factory = sqlite3.Row
            for row in live_connection.execute("SELECT * FROM memory_nodes ORDER BY node_id ASC").fetchall():
                state_conn.execute(
                    """
                    INSERT INTO memory_nodes (
                        node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        memory_level = excluded.memory_level,
                        node_type = excluded.node_type,
                        canonical_key = COALESCE(memory_nodes.canonical_key, excluded.canonical_key),
                        support_count = memory_nodes.support_count + excluded.support_count,
                        first_seen_step = CASE
                            WHEN memory_nodes.first_seen_step IS NULL THEN excluded.first_seen_step
                            WHEN excluded.first_seen_step IS NULL THEN memory_nodes.first_seen_step
                            ELSE MIN(memory_nodes.first_seen_step, excluded.first_seen_step)
                        END,
                        last_seen_step = CASE
                            WHEN memory_nodes.last_seen_step IS NULL THEN excluded.last_seen_step
                            WHEN excluded.last_seen_step IS NULL THEN memory_nodes.last_seen_step
                            ELSE MAX(memory_nodes.last_seen_step, excluded.last_seen_step)
                        END,
                        attrs_json = excluded.attrs_json
                    """,
                    tuple(row[column] for column in row.keys()),
                )
            for row in live_connection.execute(
                "SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json FROM memory_edges ORDER BY source_node_id ASC, target_node_id ASC, edge_type ASC"
            ).fetchall():
                state_conn.execute(
                    """
                    INSERT INTO memory_edges (
                        source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_node_id, target_node_id, edge_type) DO UPDATE SET
                        weight = MAX(memory_edges.weight, excluded.weight),
                        support_count = memory_edges.support_count + excluded.support_count,
                        evidence_json = excluded.evidence_json
                    """,
                    tuple(row[column] for column in row.keys()),
                )
            for row in live_connection.execute("SELECT * FROM memory_evidence ORDER BY evidence_id ASC").fetchall():
                state_conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_evidence (
                        evidence_id, target_node_id, source_interaction_id, evidence_type, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    tuple(row[column] for column in row.keys()),
                )
            for row in live_connection.execute("SELECT * FROM memory_scores ORDER BY node_id ASC").fetchall():
                state_conn.execute(
                    """
                    INSERT INTO memory_scores (
                        node_id, isf_total, prediction_lift, transfer_score, explanatory_reach,
                        compression_gain, future_option_delta, replay_priority, retention_status,
                        memory_state, stored_epoch, last_replayed_epoch, last_promoted_epoch,
                        retention_score, forgetting_score, compressed_into_id, superseded_by_id,
                        forgetting_reason, updated_step
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        isf_total = COALESCE(excluded.isf_total, memory_scores.isf_total),
                        prediction_lift = COALESCE(excluded.prediction_lift, memory_scores.prediction_lift),
                        transfer_score = COALESCE(excluded.transfer_score, memory_scores.transfer_score),
                        explanatory_reach = COALESCE(excluded.explanatory_reach, memory_scores.explanatory_reach),
                        compression_gain = COALESCE(excluded.compression_gain, memory_scores.compression_gain),
                        future_option_delta = COALESCE(excluded.future_option_delta, memory_scores.future_option_delta),
                        replay_priority = MAX(COALESCE(memory_scores.replay_priority, 0.0), COALESCE(excluded.replay_priority, 0.0)),
                        retention_status = COALESCE(excluded.retention_status, memory_scores.retention_status),
                        memory_state = COALESCE(excluded.memory_state, memory_scores.memory_state),
                        stored_epoch = COALESCE(excluded.stored_epoch, memory_scores.stored_epoch),
                        last_replayed_epoch = COALESCE(excluded.last_replayed_epoch, memory_scores.last_replayed_epoch),
                        last_promoted_epoch = COALESCE(excluded.last_promoted_epoch, memory_scores.last_promoted_epoch),
                        retention_score = COALESCE(excluded.retention_score, memory_scores.retention_score),
                        forgetting_score = COALESCE(excluded.forgetting_score, memory_scores.forgetting_score),
                        compressed_into_id = COALESCE(excluded.compressed_into_id, memory_scores.compressed_into_id),
                        superseded_by_id = COALESCE(excluded.superseded_by_id, memory_scores.superseded_by_id),
                        forgetting_reason = COALESCE(excluded.forgetting_reason, memory_scores.forgetting_reason),
                        updated_step = COALESCE(excluded.updated_step, memory_scores.updated_step)
                    """,
                    tuple(row[column] for column in row.keys()),
                )
            for row in live_connection.execute("SELECT * FROM memory_promotions ORDER BY promotion_id ASC").fetchall():
                state_conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_promotions (
                        promotion_id, source_node_id, target_node_id, promotion_type,
                        evidence_count, promotion_score, status, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(row[column] for column in row.keys()),
                )
        if graph_rows:
            temp_path = paths.root / "_live_graph_export_tmp.json"
            temp_path.write_text(json.dumps(graph_rows, indent=2), encoding="utf-8")
            _ingest_live_graph_export(graph_conn, temp_path, CompactMemoryFoldConfig(global_step_start=global_step_start, global_step_end=global_step_end))
            temp_path.unlink(missing_ok=True)
        summary = _build_memory_summary_from_connections(
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            paths=paths,
        )
        summary["semantic_family_fallback_count"] = semantic_family_fallback_count
        _write_memory_summary_table(state_conn, summary)
        paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
    return summary


def build_memory_summary(paths: CompactMemoryPaths) -> dict[str, Any]:
    with (
        sqlite3.connect(paths.current_state) as state_conn,
        sqlite3.connect(paths.graph) as graph_conn,
        sqlite3.connect(paths.replay_queue) as replay_conn,
    ):
        return _build_memory_summary_from_connections(
            state_conn=state_conn,
            graph_conn=graph_conn,
            replay_conn=replay_conn,
            paths=paths,
        )


def load_memory_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def canonicalize_context_action_effect(context_signature: str, action: int | str, effect_signature: str) -> str:
    normalized_context = _normalize_jsonish(context_signature)
    return f"{normalized_context}|a{action}|e{effect_signature}"


def normalized_contingency_identity(*, context_level: int | None, action: int | str | None, effect_signature: str | None) -> str:
    effect_text = str(effect_signature or "unknown").lower()
    if effect_text.startswith("effect_signature:"):
        effect_bucket = "effect_signature"
    elif effect_text.startswith("delta_signature:"):
        effect_bucket = "delta_signature"
    elif effect_text.startswith("changed_cells_signature:"):
        effect_bucket = "changed_cells_signature"
    elif effect_text.startswith("outcome_signature:"):
        effect_bucket = "outcome_signature"
    elif effect_text.startswith("centroid:"):
        effect_bucket = "centroid"
    elif effect_text.startswith("local_family:"):
        effect_bucket = "local_family"
    else:
        effect_bucket = "unknown"
    context_bucket = f"ctx{max(0, min(int(context_level or 0), 3))}"
    action_bucket = f"a{action}" if action is not None else "aunknown"
    payload = {
        "action_bucket": action_bucket,
        "effect_bucket": effect_bucket,
        "context_bucket": context_bucket,
    }
    return "ncont:" + sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


def normalized_contingency_identity_cached(
    *,
    context_level: int | None,
    action: int | str | None,
    effect_signature: str | None,
    caches: RawDbFoldCaches | None = None,
) -> str:
    cache_key = (int(context_level or 0), str(action), str(effect_signature))
    if caches is not None and cache_key in caches.normalized_contingency_identity:
        return caches.normalized_contingency_identity[cache_key]
    normalized = normalized_contingency_identity(
        context_level=context_level,
        action=action,
        effect_signature=effect_signature,
    )
    if caches is not None:
        caches.normalized_contingency_identity[cache_key] = normalized
    return normalized


def canonical_family_signature_from_family(family: Any) -> str:
    centroid = getattr(family, "centroid_vector", None)
    if centroid is not None:
        return "centroid:" + json.dumps([round(float(value), 6) for value in np.asarray(centroid, dtype=float).tolist()], separators=(",", ":"))
    return f"family:{getattr(family, 'id', 'unknown')}"


def canonical_family_signature_from_raw_db(
    raw_conn: sqlite3.Connection,
    family_id: Any,
    row_payload: dict[str, Any] | None,
    caches: RawDbFoldCaches | None = None,
) -> str:
    payload = dict(row_payload or {})
    family_key = "" if family_id is None else str(family_id)
    if caches is not None and family_key in caches.family_signature_by_family_id:
        return caches.family_signature_by_family_id[family_key]
    if caches is not None and family_id is not None:
        row = caches.transformation_family_rows.get(family_key)
        if row is not None and row["centroid_vector"]:
            centroid = _normalize_jsonish_cached(row["centroid_vector"], caches)
            signature = f"centroid:{centroid}"
            caches.family_signature_by_family_id[family_key] = signature
            return signature
    elif family_id is not None:
        transformation_tables = {row[0] for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "transformation_families" in transformation_tables:
            row = raw_conn.execute(
                "SELECT centroid_vector, support_count FROM transformation_families WHERE id = ?",
                (family_id,),
            ).fetchone()
            if row and row[0]:
                centroid = _normalize_jsonish(row[0])
                return f"centroid:{centroid}"
    for key in ("effect_signature", "delta_signature", "changed_cells_signature", "outcome_signature"):
        value = payload.get(key)
        if value not in (None, ""):
            signature = f"{key}:{_normalize_jsonish_cached(value, caches)}"
            if caches is not None and family_id is not None:
                caches.family_signature_by_family_id[family_key] = signature
            return signature
    db_identity = str(raw_conn.execute("PRAGMA database_list").fetchone()[2])
    signature = f"local_family:{db_identity}:{family_id}"
    if caches is not None and family_id is not None:
        caches.family_signature_by_family_id[family_key] = signature
    return signature


def _build_raw_db_fold_caches(raw_conn: sqlite3.Connection, tables: set[str]) -> RawDbFoldCaches:
    caches = RawDbFoldCaches(
        family_signature_by_family_id={},
        family_signature_by_payload_key={},
        context_level_by_key={},
        normalized_jsonish={},
        normalized_contingency_identity={},
        transformation_family_rows={},
        prediction_context_level_lookup={},
    )
    if "transformation_families" in tables:
        for row in raw_conn.execute("SELECT id, centroid_vector, support_count FROM transformation_families").fetchall():
            key = str(row["id"])
            caches.transformation_family_rows[key] = row
            if row["centroid_vector"]:
                caches.family_signature_by_family_id[key] = f"centroid:{_normalize_jsonish_cached(row['centroid_vector'], caches)}"
    if "prediction_results" in tables:
        prediction_columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(prediction_results)").fetchall()}
        if "context_level" in prediction_columns:
            context_expr = "COALESCE(context_signature, context_action, '')" if "context_action" in prediction_columns else "COALESCE(context_signature, '')"
            for row in raw_conn.execute(
                f"""
                SELECT
                    {context_expr} AS context_key,
                    COALESCE(action, -1) AS action_key,
                    COALESCE(actual_family, predicted_family, '') AS family_key,
                    MAX(COALESCE(context_level, 0)) AS max_context_level
                FROM prediction_results
                GROUP BY 1, 2, 3
                """
            ).fetchall():
                caches.prediction_context_level_lookup[
                    (str(row["context_key"]), str(row["action_key"]), str(row["family_key"]))
                ] = int(row["max_context_level"] or 0)
    return caches


def stable_family_int_id(canonical_signature: str) -> int:
    return int.from_bytes(sha1(str(canonical_signature).encode("utf-8")).digest()[:8], "big") % 2_147_483_647


def _attach_shard_database(conn: sqlite3.Connection, alias: str, path: Path) -> str:
    safe_alias = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in alias)
    conn.execute(f"ATTACH DATABASE ? AS {safe_alias}", (str(path),))
    return safe_alias


def _detach_shard_database(conn: sqlite3.Connection, alias: str) -> None:
    conn.commit()
    conn.execute(f"DETACH DATABASE {alias}")


def _delete_edges(graph_conn: sqlite3.Connection, edge_ids: list[str]) -> int:
    if not edge_ids:
        return 0
    graph_conn.executemany(
        "DELETE FROM graph_edges WHERE edge_id = ?",
        [(edge_id,) for edge_id in edge_ids],
    )
    return len(edge_ids)


def _prune_edges_per_source(
    graph_conn: sqlite3.Connection,
    *,
    limit: int,
    totals: dict[str, Any] | None,
    counter_name: str,
) -> None:
    if limit <= 0:
        return
    sources = graph_conn.execute(
        """
        SELECT source_node_id
        FROM graph_edges
        GROUP BY source_node_id
        HAVING COUNT(*) > ?
        """,
        (limit,),
    ).fetchall()
    pruned = 0
    for (source_node_id,) in sources:
        rows = graph_conn.execute(
            """
            SELECT edge_id
            FROM graph_edges
            WHERE source_node_id = ?
            ORDER BY
                support_count DESC,
                weight DESC,
                last_seen_global_step DESC,
                edge_id ASC
            LIMIT -1 OFFSET ?
            """,
            (source_node_id, limit),
        ).fetchall()
        pruned += _delete_edges(graph_conn, [str(row[0]) for row in rows])
    if totals is not None:
        totals[counter_name] = int(totals.get(counter_name, 0) or 0) + pruned


def _prune_edges_by_source_prefix(
    graph_conn: sqlite3.Connection,
    *,
    source_prefix: str,
    limit: int,
    edge_types: tuple[str, ...] | None,
    totals: dict[str, Any] | None,
    counter_name: str,
) -> None:
    if limit <= 0:
        return
    type_filter = ""
    params: list[Any] = [f"{source_prefix}%"]
    if edge_types:
        placeholders = ",".join("?" for _ in edge_types)
        type_filter = f"AND edge_type IN ({placeholders})"
        params.extend(edge_types)
    sources = graph_conn.execute(
        f"""
        SELECT source_node_id
        FROM graph_edges
        WHERE source_node_id LIKE ?
        {type_filter}
        GROUP BY source_node_id
        HAVING COUNT(*) > ?
        """,
        tuple(params + [limit]),
    ).fetchall()
    pruned = 0
    for (source_node_id,) in sources:
        edge_params: list[Any] = [source_node_id]
        edge_type_filter = ""
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            edge_type_filter = f"AND edge_type IN ({placeholders})"
            edge_params.extend(edge_types)
        rows = graph_conn.execute(
            f"""
            SELECT edge_id
            FROM graph_edges
            WHERE source_node_id = ?
            {edge_type_filter}
            ORDER BY
                support_count DESC,
                weight DESC,
                last_seen_global_step DESC,
                edge_id ASC
            LIMIT -1 OFFSET ?
            """,
            tuple(edge_params + [limit]),
        ).fetchall()
        pruned += _delete_edges(graph_conn, [str(row[0]) for row in rows])
    if totals is not None:
        totals[counter_name] = int(totals.get(counter_name, 0) or 0) + pruned


def _enforce_graph_caps_after_merge(
    graph_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any] | None = None,
) -> None:
    if not bool(fold_config.enable_graph_edge_caps):
        return
    max_total = int(fold_config.max_graph_edges_per_fold)
    if max_total > 0:
        rows = graph_conn.execute(
            """
            SELECT edge_id
            FROM graph_edges
            ORDER BY
                support_count DESC,
                weight DESC,
                last_seen_global_step DESC,
                edge_id ASC
            LIMIT -1 OFFSET ?
            """,
            (max_total,),
        ).fetchall()
        delete_ids = [str(row[0]) for row in rows]
        if delete_ids:
            _delete_edges(graph_conn, delete_ids)
            if totals is not None:
                totals["graph_edges_pruned_by_post_merge_fold_cap"] = (
                    int(totals.get("graph_edges_pruned_by_post_merge_fold_cap", 0) or 0)
                    + len(delete_ids)
                )
    _prune_edges_per_source(
        graph_conn,
        limit=int(fold_config.max_edges_per_source_node),
        totals=totals,
        counter_name="graph_edges_pruned_by_post_merge_source_cap",
    )
    _prune_edges_by_source_prefix(
        graph_conn,
        source_prefix="carrier:",
        limit=int(fold_config.max_edges_per_carrier),
        edge_types=("explains", "anchors", "appears_in"),
        totals=totals,
        counter_name="graph_edges_pruned_by_post_merge_carrier_cap",
    )
    _prune_edges_by_source_prefix(
        graph_conn,
        source_prefix="family:",
        limit=int(fold_config.max_edges_per_family),
        edge_types=None,
        totals=totals,
        counter_name="graph_edges_pruned_by_post_merge_family_cap",
    )


def _write_graph_epoch_summary(
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any],
) -> None:
    epoch_key = f"{int(fold_config.global_step_start)}:{int(fold_config.global_step_end)}"
    cumulative_nodes = int(graph_conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0] or 0)
    cumulative_edges = int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] or 0)
    state_conn.execute(
        """
        INSERT INTO graph_epoch_summary (
            epoch_key, global_step_start, global_step_end,
            graph_edges_attempted, graph_edges_written,
            graph_edges_skipped_by_fold_cap, graph_edges_skipped_by_source_cap,
            graph_edges_skipped_by_carrier_cap, graph_edges_skipped_by_family_cap,
            graph_edges_pruned_by_post_merge_fold_cap,
            graph_edges_pruned_by_post_merge_source_cap,
            graph_edges_pruned_by_post_merge_carrier_cap,
            graph_edges_pruned_by_post_merge_family_cap,
            graph_edges_written_by_set_based_merge,
            graph_edges_before_set_based_merge,
            graph_edges_after_set_based_merge,
            cumulative_graph_node_count, cumulative_graph_edge_count, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(epoch_key) DO UPDATE SET
            graph_edges_attempted = excluded.graph_edges_attempted,
            graph_edges_written = excluded.graph_edges_written,
            graph_edges_skipped_by_fold_cap = excluded.graph_edges_skipped_by_fold_cap,
            graph_edges_skipped_by_source_cap = excluded.graph_edges_skipped_by_source_cap,
            graph_edges_skipped_by_carrier_cap = excluded.graph_edges_skipped_by_carrier_cap,
            graph_edges_skipped_by_family_cap = excluded.graph_edges_skipped_by_family_cap,
            graph_edges_pruned_by_post_merge_fold_cap = excluded.graph_edges_pruned_by_post_merge_fold_cap,
            graph_edges_pruned_by_post_merge_source_cap = excluded.graph_edges_pruned_by_post_merge_source_cap,
            graph_edges_pruned_by_post_merge_carrier_cap = excluded.graph_edges_pruned_by_post_merge_carrier_cap,
            graph_edges_pruned_by_post_merge_family_cap = excluded.graph_edges_pruned_by_post_merge_family_cap,
            graph_edges_written_by_set_based_merge = excluded.graph_edges_written_by_set_based_merge,
            graph_edges_before_set_based_merge = excluded.graph_edges_before_set_based_merge,
            graph_edges_after_set_based_merge = excluded.graph_edges_after_set_based_merge,
            cumulative_graph_node_count = excluded.cumulative_graph_node_count,
            cumulative_graph_edge_count = excluded.cumulative_graph_edge_count,
            created_at = excluded.created_at
        """,
        (
            epoch_key,
            int(fold_config.global_step_start),
            int(fold_config.global_step_end),
            int(totals.get("graph_edges_attempted", 0) or 0),
            int(totals.get("graph_edges_written", 0) or 0),
            int(totals.get("graph_edges_skipped_by_fold_cap", 0) or 0),
            int(totals.get("graph_edges_skipped_by_source_cap", 0) or 0),
            int(totals.get("graph_edges_skipped_by_carrier_cap", 0) or 0),
            int(totals.get("graph_edges_skipped_by_family_cap", 0) or 0),
            int(totals.get("graph_edges_pruned_by_post_merge_fold_cap", 0) or 0),
            int(totals.get("graph_edges_pruned_by_post_merge_source_cap", 0) or 0),
            int(totals.get("graph_edges_pruned_by_post_merge_carrier_cap", 0) or 0),
            int(totals.get("graph_edges_pruned_by_post_merge_family_cap", 0) or 0),
            int(totals.get("graph_edges_written_by_set_based_merge", 0) or 0),
            int(totals.get("graph_edges_before_set_based_merge", 0) or 0),
            int(totals.get("graph_edges_after_set_based_merge", 0) or 0),
            cumulative_nodes,
            cumulative_edges,
        ),
    )


def _merge_compact_memory_dir_into_main(
    *,
    temp_dir: Path,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    replay_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any] | None = None,
    enforce_graph_caps_after_merge: bool = True,
) -> None:
    temp_paths = ensure_memory_layout(temp_dir)
    if bool(fold_config.use_set_based_merge):
        before_graph_edges = int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] or 0)
        _merge_state_tables_set_based(temp_paths.current_state, state_conn, fold_config)
        _merge_graph_tables_set_based(temp_paths.graph, graph_conn, fold_config)
        if enforce_graph_caps_after_merge:
            _enforce_graph_caps_after_merge(graph_conn, fold_config, totals=totals)
        after_graph_edges = int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] or 0)
        if totals is not None:
            totals["graph_edges_before_set_based_merge"] = before_graph_edges
            totals["graph_edges_after_set_based_merge"] = after_graph_edges
            totals["graph_edges_written_by_set_based_merge"] = (
                int(totals.get("graph_edges_written_by_set_based_merge", 0) or 0)
                + max(0, after_graph_edges - before_graph_edges)
            )
        _merge_replay_queue_set_based(temp_paths.replay_queue, replay_conn)
        return
    with (
        sqlite3.connect(temp_paths.current_state) as temp_state,
        sqlite3.connect(temp_paths.graph) as temp_graph,
        sqlite3.connect(temp_paths.replay_queue) as temp_replay,
    ):
        temp_state.row_factory = sqlite3.Row
        temp_graph.row_factory = sqlite3.Row
        temp_replay.row_factory = sqlite3.Row
        _merge_state_tables(temp_state, state_conn, fold_config)
        _merge_graph_tables(temp_graph, graph_conn, fold_config)
        _merge_replay_queue(temp_replay, replay_conn)


def _merge_state_tables_set_based(temp_state_path: Path, state_conn: sqlite3.Connection, fold_config: CompactMemoryFoldConfig) -> None:
    alias = _attach_shard_database(state_conn, "shard_state", temp_state_path)
    try:
        state_conn.execute(
            f"""
            INSERT INTO family_identity_map (canonical_signature, stable_family_id)
            SELECT canonical_signature, stable_family_id
            FROM {alias}.family_identity_map
            WHERE 1=1
            ON CONFLICT(canonical_signature) DO UPDATE SET
                stable_family_id = excluded.stable_family_id
            """
        )
        state_conn.execute(
            f"""
            INSERT INTO stable_contingencies (
                contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                mean_replay_priority, representative_example_count, prediction_attempt_count,
                prediction_success_count, prediction_accuracy, prediction_error_before,
                prediction_error_after, normalized_contingency_key
            )
            SELECT
                contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                mean_replay_priority, representative_example_count, prediction_attempt_count,
                prediction_success_count, prediction_accuracy, prediction_error_before,
                prediction_error_after, normalized_contingency_key
            FROM {alias}.stable_contingencies
            WHERE 1=1
            ON CONFLICT(canonical_key) DO UPDATE SET
                support_count = COALESCE(stable_contingencies.support_count, 0) + COALESCE(excluded.support_count, 0),
                context_level = MAX(stable_contingencies.context_level, excluded.context_level),
                first_seen_global_step = MIN(stable_contingencies.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(stable_contingencies.last_seen_global_step, excluded.last_seen_global_step),
                stability_score = CAST(
                    COALESCE(stable_contingencies.support_count, 0) + COALESCE(excluded.support_count, 0)
                    AS REAL
                ) / 20.0,
                representative_example_count = MAX(stable_contingencies.representative_example_count, excluded.representative_example_count),
                prediction_attempt_count = COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0),
                prediction_success_count = COALESCE(stable_contingencies.prediction_success_count, 0) + COALESCE(excluded.prediction_success_count, 0),
                prediction_accuracy = CASE
                    WHEN (COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0)) > 0
                    THEN CAST(
                        (COALESCE(stable_contingencies.prediction_success_count, 0) + COALESCE(excluded.prediction_success_count, 0)) AS REAL
                    ) / CAST(
                        (COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0)) AS REAL
                    )
                    ELSE NULL
                END,
                normalized_contingency_key = COALESCE(stable_contingencies.normalized_contingency_key, excluded.normalized_contingency_key)
            """
        )
        state_conn.execute(
            f"""
            INSERT INTO transformation_families (
                family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score,
                prediction_lift, prediction_accuracy_mean, prediction_error_before_mean, prediction_error_after_mean
            )
            SELECT
                family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score,
                prediction_lift, prediction_accuracy_mean, prediction_error_before_mean, prediction_error_after_mean
            FROM {alias}.transformation_families
            WHERE 1=1
            ON CONFLICT(canonical_signature) DO UPDATE SET
                support_count = transformation_families.support_count + excluded.support_count,
                member_count = transformation_families.member_count + excluded.member_count,
                first_seen_global_step = MIN(transformation_families.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
                stability_score = MAX(transformation_families.stability_score, excluded.stability_score),
                prediction_lift = COALESCE(excluded.prediction_lift, transformation_families.prediction_lift),
                prediction_accuracy_mean = COALESCE(excluded.prediction_accuracy_mean, transformation_families.prediction_accuracy_mean),
                prediction_error_before_mean = COALESCE(excluded.prediction_error_before_mean, transformation_families.prediction_error_before_mean),
                prediction_error_after_mean = COALESCE(excluded.prediction_error_after_mean, transformation_families.prediction_error_after_mean)
            """
        )
        state_conn.execute(
            f"""
            INSERT INTO family_members (
                family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step
            )
            SELECT
                family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step
            FROM {alias}.family_members
            WHERE 1=1
            ON CONFLICT(family_signature, contingency_key) DO UPDATE SET
                support_count = family_members.support_count + excluded.support_count,
                first_seen_global_step = MIN(family_members.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(family_members.last_seen_global_step, excluded.last_seen_global_step)
            """
        )
        state_conn.execute(
            f"""
            INSERT INTO carrier_candidates (
                carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent
            )
            SELECT
                carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent
            FROM {alias}.carrier_candidates
            WHERE 1=1
            ON CONFLICT(carrier_signature) DO UPDATE SET
                support_count = MAX(carrier_candidates.support_count, excluded.support_count),
                linked_family_count = MAX(carrier_candidates.linked_family_count, excluded.linked_family_count),
                first_seen_global_step = MIN(carrier_candidates.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(carrier_candidates.last_seen_global_step, excluded.last_seen_global_step),
                carrier_timing_source = CASE
                    WHEN carrier_candidates.carrier_timing_source = 'real_evidence'
                         OR excluded.carrier_timing_source = 'real_evidence'
                    THEN 'real_evidence'
                    WHEN COALESCE(carrier_candidates.carrier_timing_source, 'unknown') = COALESCE(excluded.carrier_timing_source, 'unknown')
                    THEN COALESCE(excluded.carrier_timing_source, carrier_candidates.carrier_timing_source, 'unknown')
                    WHEN carrier_candidates.carrier_timing_source = 'unknown'
                         AND excluded.carrier_timing_source = 'fold_start_fallback'
                    THEN 'fold_start_fallback'
                    WHEN carrier_candidates.carrier_timing_source = 'fold_start_fallback'
                         AND excluded.carrier_timing_source = 'unknown'
                    THEN 'fold_start_fallback'
                    ELSE 'mixed'
                END,
                stability_score = MAX(carrier_candidates.stability_score, excluded.stability_score),
                is_emergent = MAX(carrier_candidates.is_emergent, excluded.is_emergent)
            """
        )
        state_conn.execute(
            f"""
            INSERT INTO carrier_links (
                carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
            )
            SELECT
                carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
            FROM {alias}.carrier_links
            WHERE 1=1
            ON CONFLICT(carrier_signature, linked_type, linked_key) DO UPDATE SET
                support_count = carrier_links.support_count + excluded.support_count,
                first_seen_global_step = MIN(carrier_links.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(carrier_links.last_seen_global_step, excluded.last_seen_global_step)
            """
        )
        state_conn.execute(
            f"""
            INSERT INTO contradiction_clusters (
                cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step,
                max_prediction_error, mean_replay_priority
            )
            SELECT
                cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step,
                max_prediction_error, mean_replay_priority
            FROM {alias}.contradiction_clusters
            WHERE 1=1
            ON CONFLICT(canonical_key) DO UPDATE SET
                support_count = contradiction_clusters.support_count + excluded.support_count,
                first_seen_global_step = MIN(contradiction_clusters.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(contradiction_clusters.last_seen_global_step, excluded.last_seen_global_step),
                max_prediction_error = MAX(contradiction_clusters.max_prediction_error, excluded.max_prediction_error),
                mean_replay_priority = MAX(contradiction_clusters.mean_replay_priority, excluded.mean_replay_priority)
            """
        )
        state_conn.execute(
            f"""
            INSERT OR REPLACE INTO representative_examples (
                example_id, owner_type, owner_id, game, sampler, seed, global_step, example_kind, compact_payload_json, priority_score
            )
            SELECT
                example_id, owner_type, owner_id, game, sampler, seed, global_step, example_kind, compact_payload_json, priority_score
            FROM {alias}.representative_examples
            """
        )
        state_conn.execute(
            f"""
            INSERT INTO temporal_milestones (
                game, sampler, seed, first_interaction_step, first_contingency_candidate_step, first_stable_contingency_step,
                first_prediction_violation_step, first_high_replay_priority_step, first_transformation_family_step,
                first_stable_transformation_family_step, first_carrier_candidate_step, first_emergent_carrier_step
            )
            SELECT
                game, sampler, seed, first_interaction_step, first_contingency_candidate_step, first_stable_contingency_step,
                first_prediction_violation_step, first_high_replay_priority_step, first_transformation_family_step,
                first_stable_transformation_family_step, first_carrier_candidate_step, first_emergent_carrier_step
            FROM {alias}.temporal_milestones
            WHERE 1=1
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
            """
        )
        state_conn.execute(
            f"""
            INSERT OR REPLACE INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class,
                comparable_outcome_group_id, efficiency_active, success, terminal,
                trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, future_option_gain, future_option_gain_per_action,
                equivalent_outcome_cost_gap, loop_count, loop_ratio, repeated_state_count,
                repeated_state_ratio, blocked_action_count, blocked_action_ratio,
                wasted_action_count, wasted_action_ratio, unique_state_count, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus, efficiency_retention_bonus,
                efficiency_promotion_bonus
            )
            SELECT
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class,
                comparable_outcome_group_id, efficiency_active, success, terminal,
                trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, future_option_gain, future_option_gain_per_action,
                equivalent_outcome_cost_gap, loop_count, loop_ratio, repeated_state_count,
                repeated_state_ratio, blocked_action_count, blocked_action_ratio,
                wasted_action_count, wasted_action_ratio, unique_state_count, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus, efficiency_retention_bonus,
                efficiency_promotion_bonus
            FROM {alias}.trajectory_efficiency
            """
        )
        state_conn.execute(
            f"""
            INSERT OR REPLACE INTO compact_interaction_trajectory_events (
                event_id, interaction_id, game_id, level_id, sampler, seed, epoch, episode_id,
                global_step, outcome_state, level_completed_event, state_hash_before, state_hash_after,
                action, no_effect_action, future_option_gain
            )
            SELECT
                event_id, interaction_id, game_id, level_id, sampler, seed, epoch, episode_id,
                global_step, outcome_state, level_completed_event, state_hash_before, state_hash_after,
                action, no_effect_action, future_option_gain
            FROM {alias}.compact_interaction_trajectory_events
            """
        )
        if bool(fold_config.fold_memory_substrate):
            state_conn.execute(
                f"""
                INSERT INTO memory_nodes (
                    node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json
                )
                SELECT
                    node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json
                FROM {alias}.memory_nodes
                WHERE 1=1
                ON CONFLICT(node_id) DO UPDATE SET
                    memory_level = excluded.memory_level,
                    node_type = excluded.node_type,
                    canonical_key = COALESCE(memory_nodes.canonical_key, excluded.canonical_key),
                    support_count = memory_nodes.support_count + excluded.support_count,
                    first_seen_step = CASE
                        WHEN memory_nodes.first_seen_step IS NULL THEN excluded.first_seen_step
                        WHEN excluded.first_seen_step IS NULL THEN memory_nodes.first_seen_step
                        ELSE MIN(memory_nodes.first_seen_step, excluded.first_seen_step)
                    END,
                    last_seen_step = CASE
                        WHEN memory_nodes.last_seen_step IS NULL THEN excluded.last_seen_step
                        WHEN excluded.last_seen_step IS NULL THEN memory_nodes.last_seen_step
                        ELSE MAX(memory_nodes.last_seen_step, excluded.last_seen_step)
                    END,
                    attrs_json = excluded.attrs_json
                """
            )
            state_conn.execute(
                f"""
                INSERT INTO memory_edges (
                    source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                )
                SELECT
                    source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                FROM {alias}.memory_edges
                WHERE 1=1
                ON CONFLICT(source_node_id, target_node_id, edge_type) DO UPDATE SET
                    weight = MAX(memory_edges.weight, excluded.weight),
                    support_count = memory_edges.support_count + excluded.support_count,
                    evidence_json = excluded.evidence_json
                """
            )
            state_conn.execute(
                f"""
                INSERT OR REPLACE INTO memory_evidence (
                    evidence_id, target_node_id, source_interaction_id, evidence_type, payload_json
                )
                SELECT
                    evidence_id, target_node_id, source_interaction_id, evidence_type, payload_json
                FROM {alias}.memory_evidence
                """
            )
            state_conn.execute(
                f"""
                INSERT INTO memory_scores (
                    node_id, isf_total, prediction_lift, transfer_score, explanatory_reach,
                    compression_gain, future_option_delta, replay_priority, retention_status,
                    memory_state, stored_epoch, last_replayed_epoch, last_promoted_epoch,
                    retention_score, forgetting_score, compressed_into_id, superseded_by_id,
                    forgetting_reason, updated_step
                )
                SELECT
                    node_id, isf_total, prediction_lift, transfer_score, explanatory_reach,
                    compression_gain, future_option_delta, replay_priority, retention_status,
                    memory_state, stored_epoch, last_replayed_epoch, last_promoted_epoch,
                    retention_score, forgetting_score, compressed_into_id, superseded_by_id,
                    forgetting_reason, updated_step
                FROM {alias}.memory_scores
                WHERE 1=1
                ON CONFLICT(node_id) DO UPDATE SET
                    isf_total = COALESCE(excluded.isf_total, memory_scores.isf_total),
                    prediction_lift = COALESCE(excluded.prediction_lift, memory_scores.prediction_lift),
                    transfer_score = COALESCE(excluded.transfer_score, memory_scores.transfer_score),
                    explanatory_reach = COALESCE(excluded.explanatory_reach, memory_scores.explanatory_reach),
                    compression_gain = COALESCE(excluded.compression_gain, memory_scores.compression_gain),
                    future_option_delta = COALESCE(excluded.future_option_delta, memory_scores.future_option_delta),
                    replay_priority = MAX(COALESCE(memory_scores.replay_priority, 0.0), COALESCE(excluded.replay_priority, 0.0)),
                    retention_status = COALESCE(excluded.retention_status, memory_scores.retention_status),
                    memory_state = COALESCE(excluded.memory_state, memory_scores.memory_state),
                    stored_epoch = COALESCE(excluded.stored_epoch, memory_scores.stored_epoch),
                    last_replayed_epoch = COALESCE(excluded.last_replayed_epoch, memory_scores.last_replayed_epoch),
                    last_promoted_epoch = COALESCE(excluded.last_promoted_epoch, memory_scores.last_promoted_epoch),
                    retention_score = COALESCE(excluded.retention_score, memory_scores.retention_score),
                    forgetting_score = COALESCE(excluded.forgetting_score, memory_scores.forgetting_score),
                    compressed_into_id = COALESCE(excluded.compressed_into_id, memory_scores.compressed_into_id),
                    superseded_by_id = COALESCE(excluded.superseded_by_id, memory_scores.superseded_by_id),
                    forgetting_reason = COALESCE(excluded.forgetting_reason, memory_scores.forgetting_reason),
                    updated_step = COALESCE(excluded.updated_step, memory_scores.updated_step)
                """
            )
            state_conn.execute(
                f"""
                INSERT OR REPLACE INTO memory_promotions (
                    promotion_id, source_node_id, target_node_id, promotion_type,
                    evidence_count, promotion_score, status, payload_json
                )
                SELECT
                    promotion_id, source_node_id, target_node_id, promotion_type,
                    evidence_count, promotion_score, status, payload_json
                FROM {alias}.memory_promotions
                """
            )
    finally:
        _detach_shard_database(state_conn, alias)


def _merge_graph_tables_set_based(temp_graph_path: Path, graph_conn: sqlite3.Connection, fold_config: CompactMemoryFoldConfig) -> None:
    if not bool(fold_config.fold_graph):
        return
    alias = _attach_shard_database(graph_conn, "shard_graph", temp_graph_path)
    try:
        graph_conn.execute(
            f"""
            INSERT INTO graph_nodes (
                node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count
            )
            SELECT
                node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count
            FROM {alias}.graph_nodes
            WHERE 1=1
            ON CONFLICT(node_id) DO UPDATE SET
                node_type = COALESCE(graph_nodes.node_type, excluded.node_type),
                canonical_key = COALESCE(graph_nodes.canonical_key, excluded.canonical_key),
                first_seen_global_step = MIN(graph_nodes.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(graph_nodes.last_seen_global_step, excluded.last_seen_global_step),
                support_count = MAX(graph_nodes.support_count, excluded.support_count)
            """
        )
        # Shard-level edge caps are applied before merge; this merge preserves shard rows as-is.
        graph_conn.execute(
            f"""
            INSERT INTO graph_edges (
                edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight
            )
            SELECT
                edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight
            FROM {alias}.graph_edges
            WHERE 1=1
            ON CONFLICT(edge_id) DO UPDATE SET
                first_seen_global_step = MIN(graph_edges.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(graph_edges.last_seen_global_step, excluded.last_seen_global_step),
                support_count = MAX(graph_edges.support_count, excluded.support_count),
                weight = MAX(graph_edges.weight, excluded.weight)
            """
        )
    finally:
        _detach_shard_database(graph_conn, alias)


def _merge_replay_queue_set_based(temp_replay_path: Path, replay_conn: sqlite3.Connection) -> None:
    alias = _attach_shard_database(replay_conn, "shard_replay", temp_replay_path)
    try:
        replay_conn.execute(
            f"""
            INSERT INTO replay_queue (
                replay_id, owner_type, owner_id, priority_score, reason,
                first_seen_global_step, last_seen_global_step, compact_payload_json
            )
            SELECT
                replay_id, owner_type, owner_id, priority_score, reason,
                first_seen_global_step, last_seen_global_step, compact_payload_json
            FROM {alias}.replay_queue
            WHERE 1=1
            ON CONFLICT(replay_id) DO UPDATE SET
                priority_score = MAX(replay_queue.priority_score, excluded.priority_score),
                first_seen_global_step = MIN(replay_queue.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(replay_queue.last_seen_global_step, excluded.last_seen_global_step),
                compact_payload_json = excluded.compact_payload_json
            """
        )
    finally:
        _detach_shard_database(replay_conn, alias)


def _merge_state_tables(temp_state: sqlite3.Connection, state_conn: sqlite3.Connection, fold_config: CompactMemoryFoldConfig) -> None:
    for row in temp_state.execute("SELECT * FROM family_identity_map ORDER BY canonical_signature ASC").fetchall():
        _upsert_family_identity_map(state_conn, str(row["canonical_signature"]), int(row["stable_family_id"]))
    for row in temp_state.execute("SELECT * FROM stable_contingencies ORDER BY canonical_key ASC").fetchall():
        state_conn.execute(
            """
            INSERT INTO stable_contingencies (
                contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                mean_replay_priority, representative_example_count, prediction_attempt_count,
                prediction_success_count, prediction_accuracy, prediction_error_before,
                prediction_error_after, normalized_contingency_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_key) DO UPDATE SET
                support_count = COALESCE(stable_contingencies.support_count, 0) + COALESCE(excluded.support_count, 0),
                context_level = MAX(stable_contingencies.context_level, excluded.context_level),
                first_seen_global_step = MIN(stable_contingencies.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(stable_contingencies.last_seen_global_step, excluded.last_seen_global_step),
                stability_score = CAST(
                    COALESCE(stable_contingencies.support_count, 0) + COALESCE(excluded.support_count, 0)
                    AS REAL
                ) / 20.0,
                representative_example_count = MAX(stable_contingencies.representative_example_count, excluded.representative_example_count),
                prediction_attempt_count = COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0),
                prediction_success_count = COALESCE(stable_contingencies.prediction_success_count, 0) + COALESCE(excluded.prediction_success_count, 0),
                prediction_accuracy = CASE
                    WHEN (COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0)) > 0
                    THEN CAST(
                        (COALESCE(stable_contingencies.prediction_success_count, 0) + COALESCE(excluded.prediction_success_count, 0)) AS REAL
                    ) / CAST(
                        (COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0)) AS REAL
                    )
                    ELSE NULL
                END,
                normalized_contingency_key = COALESCE(stable_contingencies.normalized_contingency_key, excluded.normalized_contingency_key)
            """,
            tuple(row[column] for column in row.keys()),
        )
    for row in temp_state.execute("SELECT * FROM transformation_families ORDER BY canonical_signature ASC").fetchall():
        state_conn.execute(
            """
            INSERT INTO transformation_families (
                family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score,
                prediction_lift, prediction_accuracy_mean, prediction_error_before_mean, prediction_error_after_mean
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_signature) DO UPDATE SET
                support_count = transformation_families.support_count + excluded.support_count,
                member_count = transformation_families.member_count + excluded.member_count,
                first_seen_global_step = MIN(transformation_families.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
                stability_score = MAX(transformation_families.stability_score, excluded.stability_score),
                prediction_lift = COALESCE(excluded.prediction_lift, transformation_families.prediction_lift),
                prediction_accuracy_mean = COALESCE(excluded.prediction_accuracy_mean, transformation_families.prediction_accuracy_mean),
                prediction_error_before_mean = COALESCE(excluded.prediction_error_before_mean, transformation_families.prediction_error_before_mean),
                prediction_error_after_mean = COALESCE(excluded.prediction_error_after_mean, transformation_families.prediction_error_after_mean)
            """,
            tuple(row[column] for column in row.keys()),
        )
    for row in temp_state.execute("SELECT * FROM family_members ORDER BY family_signature ASC, contingency_key ASC").fetchall():
        state_conn.execute(
            """
            INSERT INTO family_members (
                family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(family_signature, contingency_key) DO UPDATE SET
                support_count = family_members.support_count + excluded.support_count,
                first_seen_global_step = MIN(family_members.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(family_members.last_seen_global_step, excluded.last_seen_global_step)
            """,
            tuple(row[column] for column in row.keys()),
        )
    for row in temp_state.execute("SELECT * FROM carrier_candidates ORDER BY carrier_signature ASC").fetchall():
        state_conn.execute(
            """
            INSERT INTO carrier_candidates (
                carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(carrier_signature) DO UPDATE SET
                support_count = MAX(carrier_candidates.support_count, excluded.support_count),
                linked_family_count = MAX(carrier_candidates.linked_family_count, excluded.linked_family_count),
                first_seen_global_step = MIN(carrier_candidates.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(carrier_candidates.last_seen_global_step, excluded.last_seen_global_step),
                carrier_timing_source = CASE
                    WHEN carrier_candidates.carrier_timing_source = 'real_evidence'
                         OR excluded.carrier_timing_source = 'real_evidence'
                    THEN 'real_evidence'
                    WHEN COALESCE(carrier_candidates.carrier_timing_source, 'unknown') = COALESCE(excluded.carrier_timing_source, 'unknown')
                    THEN COALESCE(excluded.carrier_timing_source, carrier_candidates.carrier_timing_source, 'unknown')
                    WHEN carrier_candidates.carrier_timing_source = 'unknown'
                         AND excluded.carrier_timing_source = 'fold_start_fallback'
                    THEN 'fold_start_fallback'
                    WHEN carrier_candidates.carrier_timing_source = 'fold_start_fallback'
                         AND excluded.carrier_timing_source = 'unknown'
                    THEN 'fold_start_fallback'
                    ELSE 'mixed'
                END,
                stability_score = MAX(carrier_candidates.stability_score, excluded.stability_score),
                is_emergent = MAX(carrier_candidates.is_emergent, excluded.is_emergent)
            """,
            tuple(row[column] for column in row.keys()),
        )
    for row in temp_state.execute("SELECT * FROM carrier_links ORDER BY carrier_signature ASC, linked_type ASC, linked_key ASC").fetchall():
        _upsert_carrier_link(
            state_conn,
            carrier_signature=str(row["carrier_signature"]),
            linked_type=str(row["linked_type"]),
            linked_key=None if row["linked_key"] is None else str(row["linked_key"]),
            fold_config=CompactMemoryFoldConfig(
                global_step_start=int(row["first_seen_global_step"] or fold_config.global_step_start),
                global_step_end=int(row["last_seen_global_step"] or fold_config.global_step_end),
            ),
        )
    for row in temp_state.execute("SELECT * FROM contradiction_clusters ORDER BY canonical_key ASC").fetchall():
        state_conn.execute(
            """
            INSERT INTO contradiction_clusters (
                cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step,
                max_prediction_error, mean_replay_priority
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_key) DO UPDATE SET
                support_count = contradiction_clusters.support_count + excluded.support_count,
                first_seen_global_step = MIN(contradiction_clusters.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(contradiction_clusters.last_seen_global_step, excluded.last_seen_global_step),
                max_prediction_error = MAX(contradiction_clusters.max_prediction_error, excluded.max_prediction_error),
                mean_replay_priority = MAX(contradiction_clusters.mean_replay_priority, excluded.mean_replay_priority)
            """,
            tuple(row[column] for column in row.keys()),
        )
    for row in temp_state.execute("SELECT * FROM representative_examples ORDER BY example_id ASC").fetchall():
        state_conn.execute(
            """
            INSERT OR REPLACE INTO representative_examples (
                example_id, owner_type, owner_id, game, sampler, seed, global_step, example_kind, compact_payload_json, priority_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row[column] for column in row.keys()),
        )
    _upsert_temporal_milestones(
        state_conn,
        [dict(row) for row in temp_state.execute("SELECT * FROM temporal_milestones ORDER BY game ASC, sampler ASC, seed ASC").fetchall()],
    )
    for row in temp_state.execute("SELECT * FROM trajectory_efficiency ORDER BY trajectory_id ASC").fetchall():
        state_conn.execute(
            """
            INSERT OR REPLACE INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class,
                comparable_outcome_group_id, efficiency_active, success, terminal,
                trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, future_option_gain, future_option_gain_per_action,
                equivalent_outcome_cost_gap, loop_count, loop_ratio, repeated_state_count,
                repeated_state_ratio, blocked_action_count, blocked_action_ratio,
                wasted_action_count, wasted_action_ratio, unique_state_count, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus, efficiency_retention_bonus,
                efficiency_promotion_bonus
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row[column] for column in row.keys()),
        )
    for row in temp_state.execute("SELECT * FROM compact_interaction_trajectory_events ORDER BY event_id ASC").fetchall():
        state_conn.execute(
            """
            INSERT OR REPLACE INTO compact_interaction_trajectory_events (
                event_id, interaction_id, game_id, level_id, sampler, seed, epoch, episode_id,
                global_step, outcome_state, level_completed_event, state_hash_before, state_hash_after,
                action, no_effect_action, future_option_gain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row[column] for column in row.keys()),
        )
    if bool(fold_config.fold_memory_substrate):
        for row in temp_state.execute("SELECT * FROM memory_nodes ORDER BY node_id ASC").fetchall():
            state_conn.execute(
                """
                INSERT INTO memory_nodes (
                    node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    memory_level = excluded.memory_level,
                    node_type = excluded.node_type,
                    canonical_key = COALESCE(memory_nodes.canonical_key, excluded.canonical_key),
                    support_count = memory_nodes.support_count + excluded.support_count,
                    first_seen_step = CASE
                        WHEN memory_nodes.first_seen_step IS NULL THEN excluded.first_seen_step
                        WHEN excluded.first_seen_step IS NULL THEN memory_nodes.first_seen_step
                        ELSE MIN(memory_nodes.first_seen_step, excluded.first_seen_step)
                    END,
                    last_seen_step = CASE
                        WHEN memory_nodes.last_seen_step IS NULL THEN excluded.last_seen_step
                        WHEN excluded.last_seen_step IS NULL THEN memory_nodes.last_seen_step
                        ELSE MAX(memory_nodes.last_seen_step, excluded.last_seen_step)
                    END,
                    attrs_json = excluded.attrs_json
                """,
                tuple(row[column] for column in row.keys()),
            )
        for row in temp_state.execute(
            "SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json FROM memory_edges ORDER BY source_node_id ASC, target_node_id ASC, edge_type ASC"
        ).fetchall():
            state_conn.execute(
                """
                INSERT INTO memory_edges (
                    source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_node_id, target_node_id, edge_type) DO UPDATE SET
                    weight = MAX(memory_edges.weight, excluded.weight),
                    support_count = memory_edges.support_count + excluded.support_count,
                    evidence_json = excluded.evidence_json
                """,
                tuple(row[column] for column in row.keys()),
            )
        for row in temp_state.execute("SELECT * FROM memory_evidence ORDER BY evidence_id ASC").fetchall():
            state_conn.execute(
                """
                INSERT OR REPLACE INTO memory_evidence (
                    evidence_id, target_node_id, source_interaction_id, evidence_type, payload_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                tuple(row[column] for column in row.keys()),
            )
        for row in temp_state.execute("SELECT * FROM memory_scores ORDER BY node_id ASC").fetchall():
            state_conn.execute(
                """
                INSERT INTO memory_scores (
                    node_id, isf_total, prediction_lift, transfer_score, explanatory_reach,
                    compression_gain, future_option_delta, replay_priority, retention_status,
                    memory_state, stored_epoch, last_replayed_epoch, last_promoted_epoch,
                    retention_score, forgetting_score, compressed_into_id, superseded_by_id,
                    forgetting_reason, updated_step
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    isf_total = COALESCE(excluded.isf_total, memory_scores.isf_total),
                    prediction_lift = COALESCE(excluded.prediction_lift, memory_scores.prediction_lift),
                    transfer_score = COALESCE(excluded.transfer_score, memory_scores.transfer_score),
                    explanatory_reach = COALESCE(excluded.explanatory_reach, memory_scores.explanatory_reach),
                    compression_gain = COALESCE(excluded.compression_gain, memory_scores.compression_gain),
                    future_option_delta = COALESCE(excluded.future_option_delta, memory_scores.future_option_delta),
                    replay_priority = MAX(COALESCE(memory_scores.replay_priority, 0.0), COALESCE(excluded.replay_priority, 0.0)),
                    retention_status = COALESCE(excluded.retention_status, memory_scores.retention_status),
                    memory_state = COALESCE(excluded.memory_state, memory_scores.memory_state),
                    stored_epoch = COALESCE(excluded.stored_epoch, memory_scores.stored_epoch),
                    last_replayed_epoch = COALESCE(excluded.last_replayed_epoch, memory_scores.last_replayed_epoch),
                    last_promoted_epoch = COALESCE(excluded.last_promoted_epoch, memory_scores.last_promoted_epoch),
                    retention_score = COALESCE(excluded.retention_score, memory_scores.retention_score),
                    forgetting_score = COALESCE(excluded.forgetting_score, memory_scores.forgetting_score),
                    compressed_into_id = COALESCE(excluded.compressed_into_id, memory_scores.compressed_into_id),
                    superseded_by_id = COALESCE(excluded.superseded_by_id, memory_scores.superseded_by_id),
                    forgetting_reason = COALESCE(excluded.forgetting_reason, memory_scores.forgetting_reason),
                    updated_step = COALESCE(excluded.updated_step, memory_scores.updated_step)
                """,
                tuple(row[column] for column in row.keys()),
            )
        for row in temp_state.execute("SELECT * FROM memory_promotions ORDER BY promotion_id ASC").fetchall():
            state_conn.execute(
                """
                INSERT OR REPLACE INTO memory_promotions (
                    promotion_id, source_node_id, target_node_id, promotion_type,
                    evidence_count, promotion_score, status, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row[column] for column in row.keys()),
            )


def _merge_graph_tables(temp_graph: sqlite3.Connection, graph_conn: sqlite3.Connection, fold_config: CompactMemoryFoldConfig) -> None:
    if not bool(fold_config.fold_graph):
        return
    for row in temp_graph.execute("SELECT * FROM graph_nodes ORDER BY node_id ASC").fetchall():
        _upsert_graph_node(
            graph_conn,
            node_id=str(row["node_id"]),
            node_type=str(row["node_type"] or "Unknown"),
            canonical_key=None if row["canonical_key"] is None else str(row["canonical_key"]),
            fold_config=CompactMemoryFoldConfig(
                global_step_start=int(row["first_seen_global_step"] or fold_config.global_step_start),
                global_step_end=int(row["last_seen_global_step"] or fold_config.global_step_end),
            ),
            support_count=int(row["support_count"] or 1),
        )
    for row in temp_graph.execute("SELECT * FROM graph_edges ORDER BY edge_id ASC").fetchall():
        _upsert_graph_edge(
            graph_conn,
            source_node_id=str(row["source_node_id"]),
            target_node_id=str(row["target_node_id"]),
            edge_type=str(row["edge_type"] or "related_to"),
            fold_config=CompactMemoryFoldConfig(
                global_step_start=int(row["first_seen_global_step"] or fold_config.global_step_start),
                global_step_end=int(row["last_seen_global_step"] or fold_config.global_step_end),
            ),
            support_count=int(row["support_count"] or 1),
            weight=float(row["weight"] or 1.0),
        )


def _merge_replay_queue(temp_replay: sqlite3.Connection, replay_conn: sqlite3.Connection) -> None:
    for row in temp_replay.execute("SELECT * FROM replay_queue ORDER BY replay_id ASC").fetchall():
        replay_conn.execute(
            """
            INSERT INTO replay_queue (
                replay_id, owner_type, owner_id, priority_score, reason,
                first_seen_global_step, last_seen_global_step, compact_payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(replay_id) DO UPDATE SET
                priority_score = MAX(replay_queue.priority_score, excluded.priority_score),
                first_seen_global_step = MIN(replay_queue.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(replay_queue.last_seen_global_step, excluded.last_seen_global_step),
                compact_payload_json = excluded.compact_payload_json
            """,
            tuple(row[column] for column in row.keys()),
        )


def _build_minimal_replay_payload(payload: dict[str, Any], prediction_payload: dict[str, Any]) -> dict[str, Any]:
    context_signature = prediction_payload.get("context_signature") or payload.get("context_signature")
    context_signature_text = None if context_signature in (None, "") else str(context_signature)
    family_id = prediction_payload.get("actual_family") or prediction_payload.get("predicted_family")
    contradiction_key = None
    if int(prediction_payload.get("context_contradiction") or prediction_payload.get("prediction_error") or 0):
        contradiction_key = prediction_payload.get("context_contradiction_key") or context_signature_text or f"interaction:{payload.get('id')}"
    return _json_safe(
        {
            "replay_id": payload.get("id") or payload.get("interaction_id"),
            "game": payload.get("game"),
            "sampler": payload.get("sampler"),
            "seed": payload.get("seed"),
            "global_step": payload.get("global_step") or prediction_payload.get("global_step"),
            "action": prediction_payload.get("action") if prediction_payload.get("action") is not None else payload.get("action"),
            "context_signature": context_signature_text,
            "context_signature_hash": None if context_signature_text is None else sha1(context_signature_text.encode("utf-8")).hexdigest()[:20],
            "family_signature": None if family_id in (None, "") else str(family_id),
            "carrier_signature": payload.get("carrier_signature"),
            "contradiction_key": contradiction_key,
            "prediction_error": prediction_payload.get("prediction_error"),
            "isf_prediction_error": payload.get("isf_prediction_error") if payload.get("isf_prediction_error") is not None else prediction_payload.get("isf_prediction_error"),
            "memory_replay_priority": payload.get("memory_replay_priority"),
            "future_option_delta": payload.get("future_option_delta") if payload.get("future_option_delta") is not None else prediction_payload.get("future_option_delta"),
            "terminal": payload.get("terminated") if payload.get("terminated") is not None else payload.get("terminal"),
            "success": payload.get("success"),
        }
    )


def _fold_single_db(
    *,
    db_path: Path,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    replay_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any],
    busy_timeout_ms: int = 10000,
) -> None:
    with sqlite3.connect(db_path, timeout=max(1.0, float(busy_timeout_ms) / 1000.0)) as raw_conn:
        raw_conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        raw_conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        caches = _build_raw_db_fold_caches(raw_conn, tables)
        game = _path_segment(db_path, -4)
        sampler = _path_segment(db_path, -3)
        seed = _seed_from_db_path(db_path)
        stable_threshold = 20
        prediction_payload_by_interaction = _prediction_payload_by_interaction_id(raw_conn) if "prediction_results" in tables else {}
        if "contingencies" not in tables:
            totals["raw_dbs_without_contingency_table"] = int(totals.get("raw_dbs_without_contingency_table", 0) or 0) + 1

        family_members_by_signature: dict[str, set[str]] = {}
        raw_contingency_rows_seen = 0
        if "contingencies" in tables:
            for row in raw_conn.execute("SELECT * FROM contingencies").fetchall():
                raw_contingency_rows_seen += 1
                payload = dict(row)
                family_signature = canonical_family_signature_from_raw_db(
                    raw_conn,
                    payload.get("transformation_family"),
                    payload,
                    caches=caches,
                )
                stable_family_id = stable_family_int_id(family_signature)
                canonical_key = canonicalize_context_action_effect(
                    context_signature=str(payload.get("context_signature") or payload.get("context_action") or "[]"),
                    action=int(payload.get("action") or 0),
                    effect_signature=family_signature,
                )
                support_count = int(payload.get("support_count") or payload.get("support") or 0)
                _upsert_family_identity_map(state_conn, family_signature, stable_family_id)
                _upsert_stable_contingency(
                    state_conn,
                    canonical_key=canonical_key,
                    contingency_id=int(payload.get("id") or payload.get("contingency_id") or 0),
                    game=game,
                    sampler=sampler,
                    context_level=_context_level_from_raw(
                        raw_conn,
                        payload=payload,
                        family_id=payload.get("transformation_family"),
                        caches=caches,
                    ),
                    action=int(payload.get("action") or 0),
                    effect_signature=family_signature,
                    support_count=support_count,
                    prediction_attempt_count=int(payload.get("prediction_attempt_count") or 0),
                    prediction_success_count=int(payload.get("prediction_success_count") or 0),
                    prediction_accuracy=_coerce_float(payload.get("prediction_accuracy")),
                    prediction_error_before=_coerce_float(payload.get("prediction_error_before")),
                    prediction_error_after=_coerce_float(payload.get("prediction_error_after")),
                    normalized_contingency_key=str(
                        payload.get("normalized_contingency_key")
                        or normalized_contingency_identity_cached(
                            context_level=_context_level_from_raw(
                                raw_conn,
                                payload=payload,
                                family_id=payload.get("transformation_family"),
                                caches=caches,
                            ),
                            action=int(payload.get("action") or 0),
                            effect_signature=family_signature,
                            caches=caches,
                        )
                    ),
                    fold_config=fold_config,
                    stable_threshold=stable_threshold,
                )
                totals["stable_contingencies_inserted"] = int(totals.get("stable_contingencies_inserted", 0) or 0) + 1
                if support_count >= stable_threshold:
                    totals["stable_contingencies_added"] = int(totals.get("stable_contingencies_added", 0) or 0) + 1
                family_members_by_signature.setdefault(family_signature, set()).add(canonical_key)
                _retain_example(
                    state_conn,
                    owner_type="contingency",
                    owner_id=canonical_key,
                    limit=fold_config.max_examples_per_contingency,
                    example_kind="first",
                    game=game,
                    sampler=sampler,
                    seed=seed,
                    global_step=fold_config.global_step_start,
                    priority_score=float(support_count),
                    compact_payload_json=json.dumps(payload, sort_keys=True),
                )
                if bool(fold_config.fold_graph):
                    _upsert_observation_graph(
                        graph_conn,
                        fold_config=fold_config,
                        game=game,
                        sampler=sampler,
                        context_signature=str(payload.get("context_signature") or payload.get("context_action") or "[]"),
                        action=int(payload.get("action") or 0),
                        effect_signature=family_signature,
                        contingency_key=canonical_key,
                        family_signature=family_signature,
                        carrier_signature=None,
                        contradiction_key=None,
                        replay_id=None,
                        totals=totals,
                    )
        prediction_results_m1_fallback_rows = 0
        if raw_contingency_rows_seen <= 0 and "prediction_results" in tables:
            prediction_results_m1_fallback_rows = _fold_prediction_results_into_m1_substrate(
                raw_conn=raw_conn,
                state_conn=state_conn,
                graph_conn=graph_conn,
                fold_config=fold_config,
                totals=totals,
                game=game,
                sampler=sampler,
                seed=seed,
                family_members_by_signature=family_members_by_signature,
                caches=caches,
            )
            raw_contingency_rows_seen += prediction_results_m1_fallback_rows
        totals["raw_contingency_rows_seen"] = int(totals.get("raw_contingency_rows_seen", 0) or 0) + raw_contingency_rows_seen
        if bool(fold_config.fold_memory_substrate) and "memory_nodes" in tables:
            for row in raw_conn.execute("SELECT * FROM memory_nodes ORDER BY node_id ASC").fetchall():
                scoped_node_id = _scope_memory_node_id(str(row["node_id"]), db_path)
                state_conn.execute(
                    """
                    INSERT INTO memory_nodes (
                        node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        memory_level = excluded.memory_level,
                        node_type = excluded.node_type,
                        canonical_key = COALESCE(memory_nodes.canonical_key, excluded.canonical_key),
                        support_count = memory_nodes.support_count + excluded.support_count,
                        first_seen_step = CASE
                            WHEN memory_nodes.first_seen_step IS NULL THEN excluded.first_seen_step
                            WHEN excluded.first_seen_step IS NULL THEN memory_nodes.first_seen_step
                            ELSE MIN(memory_nodes.first_seen_step, excluded.first_seen_step)
                        END,
                        last_seen_step = CASE
                            WHEN memory_nodes.last_seen_step IS NULL THEN excluded.last_seen_step
                            WHEN excluded.last_seen_step IS NULL THEN memory_nodes.last_seen_step
                            ELSE MAX(memory_nodes.last_seen_step, excluded.last_seen_step)
                        END,
                        attrs_json = excluded.attrs_json
                    """,
                    (
                        scoped_node_id,
                        row["memory_level"],
                        row["node_type"],
                        _scope_memory_canonical_key(str(row["node_id"]), row["node_type"], row["canonical_key"], db_path),
                        row["support_count"],
                        row["first_seen_step"],
                        row["last_seen_step"],
                        _json_dumps_or_none(_scope_payload_interaction_nodes(_json_loads_or_none(row["attrs_json"]), db_path)),
                    ),
                )
        if bool(fold_config.fold_memory_substrate) and "memory_edges" in tables:
            for row in raw_conn.execute(
                "SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json FROM memory_edges ORDER BY source_node_id ASC, target_node_id ASC, edge_type ASC"
            ).fetchall():
                source_node_id = _scope_memory_node_id(str(row["source_node_id"]), db_path)
                target_node_id = _scope_memory_node_id(str(row["target_node_id"]), db_path)
                state_conn.execute(
                    """
                    INSERT INTO memory_edges (
                        source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_node_id, target_node_id, edge_type) DO UPDATE SET
                        weight = MAX(memory_edges.weight, excluded.weight),
                        support_count = memory_edges.support_count + excluded.support_count,
                        evidence_json = excluded.evidence_json
                    """,
                    (
                        source_node_id,
                        target_node_id,
                        row["edge_type"],
                        row["weight"],
                        row["support_count"],
                        _json_dumps_or_none(_scope_payload_interaction_nodes(_json_loads_or_none(row["evidence_json"]), db_path)),
                    ),
                )
        if bool(fold_config.fold_memory_substrate) and "memory_evidence" in tables:
            for row in raw_conn.execute("SELECT * FROM memory_evidence ORDER BY evidence_id ASC").fetchall():
                state_conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_evidence (
                        evidence_id, target_node_id, source_interaction_id, evidence_type, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        _scope_raw_local_id("evidence", row["evidence_id"], db_path),
                        _scope_memory_node_id(str(row["target_node_id"]), db_path),
                        row["source_interaction_id"],
                        row["evidence_type"],
                        _json_dumps_or_none(_scope_payload_interaction_nodes(_json_loads_or_none(row["payload_json"]), db_path)),
                    ),
                )
        if bool(fold_config.fold_memory_substrate) and "memory_scores" in tables:
            for row in raw_conn.execute("SELECT * FROM memory_scores ORDER BY node_id ASC").fetchall():
                scoped_node_id = _scope_memory_node_id(str(row["node_id"]), db_path)
                state_conn.execute(
                    """
                    INSERT INTO memory_scores (
                        node_id, isf_total, prediction_lift, transfer_score, explanatory_reach,
                        compression_gain, future_option_delta, replay_priority, retention_status,
                        memory_state, stored_epoch, last_replayed_epoch, last_promoted_epoch,
                        retention_score, forgetting_score, compressed_into_id, superseded_by_id,
                        forgetting_reason, updated_step
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        isf_total = COALESCE(excluded.isf_total, memory_scores.isf_total),
                        prediction_lift = COALESCE(excluded.prediction_lift, memory_scores.prediction_lift),
                        transfer_score = COALESCE(excluded.transfer_score, memory_scores.transfer_score),
                        explanatory_reach = COALESCE(excluded.explanatory_reach, memory_scores.explanatory_reach),
                        compression_gain = COALESCE(excluded.compression_gain, memory_scores.compression_gain),
                        future_option_delta = COALESCE(excluded.future_option_delta, memory_scores.future_option_delta),
                        replay_priority = MAX(COALESCE(memory_scores.replay_priority, 0.0), COALESCE(excluded.replay_priority, 0.0)),
                        retention_status = COALESCE(excluded.retention_status, memory_scores.retention_status),
                        memory_state = COALESCE(excluded.memory_state, memory_scores.memory_state),
                        stored_epoch = COALESCE(excluded.stored_epoch, memory_scores.stored_epoch),
                        last_replayed_epoch = COALESCE(excluded.last_replayed_epoch, memory_scores.last_replayed_epoch),
                        last_promoted_epoch = COALESCE(excluded.last_promoted_epoch, memory_scores.last_promoted_epoch),
                        retention_score = COALESCE(excluded.retention_score, memory_scores.retention_score),
                        forgetting_score = COALESCE(excluded.forgetting_score, memory_scores.forgetting_score),
                        compressed_into_id = COALESCE(excluded.compressed_into_id, memory_scores.compressed_into_id),
                        superseded_by_id = COALESCE(excluded.superseded_by_id, memory_scores.superseded_by_id),
                        forgetting_reason = COALESCE(excluded.forgetting_reason, memory_scores.forgetting_reason),
                        updated_step = COALESCE(excluded.updated_step, memory_scores.updated_step)
                    """,
                    (
                        scoped_node_id,
                        row["isf_total"],
                        row["prediction_lift"],
                        row["transfer_score"],
                        row["explanatory_reach"],
                        row["compression_gain"],
                        row["future_option_delta"],
                        row["replay_priority"],
                        row["retention_status"],
                        row["memory_state"] if "memory_state" in row.keys() else None,
                        row["stored_epoch"] if "stored_epoch" in row.keys() else None,
                        row["last_replayed_epoch"] if "last_replayed_epoch" in row.keys() else None,
                        row["last_promoted_epoch"] if "last_promoted_epoch" in row.keys() else None,
                        row["retention_score"] if "retention_score" in row.keys() else None,
                        row["forgetting_score"] if "forgetting_score" in row.keys() else None,
                        row["compressed_into_id"] if "compressed_into_id" in row.keys() else None,
                        row["superseded_by_id"] if "superseded_by_id" in row.keys() else None,
                        row["forgetting_reason"] if "forgetting_reason" in row.keys() else None,
                        row["updated_step"],
                    ),
                )
        if bool(fold_config.fold_memory_substrate) and "memory_promotions" in tables:
            for row in raw_conn.execute("SELECT * FROM memory_promotions ORDER BY promotion_id ASC").fetchall():
                state_conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_promotions (
                        promotion_id, source_node_id, target_node_id, promotion_type,
                        evidence_count, promotion_score, status, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _scope_raw_local_id("promotion", row["promotion_id"], db_path),
                        _scope_memory_node_id(str(row["source_node_id"]), db_path),
                        _scope_memory_node_id(str(row["target_node_id"]), db_path),
                        row["promotion_type"],
                        row["evidence_count"],
                        row["promotion_score"],
                        row["status"],
                        _json_dumps_or_none(_scope_payload_interaction_nodes(_json_loads_or_none(row["payload_json"]), db_path)),
                    ),
                )
        if "prediction_results" in tables:
            family_supports: dict[str, dict[str, Any]] = {}
            contradiction_supports: dict[str, dict[str, Any]] = {}
            for row in raw_conn.execute("SELECT * FROM prediction_results").fetchall():
                payload = dict(row)
                family_signature = canonical_family_signature_from_raw_db(
                    raw_conn,
                    payload.get("actual_family") or payload.get("predicted_family"),
                    payload,
                    caches=caches,
                )
                stable_family_id = stable_family_int_id(family_signature)
                _upsert_family_identity_map(state_conn, family_signature, stable_family_id)
                info = family_supports.setdefault(
                    family_signature,
                    {
                        "member_count": 0,
                        "support_count": 0,
                        "first_step": None,
                        "last_step": None,
                        "mean_error_total": 0.0,
                        "mean_replay_total": 0.0,
                        "prediction_success_count": 0,
                        "prediction_error_before_values": [],
                        "prediction_error_after_values": [],
                    },
                )
                info["member_count"] += 1
                info["support_count"] += 1
                global_step = int(payload.get("global_step") or payload.get("interaction_id") or fold_config.global_step_end)
                info["first_step"] = global_step if info["first_step"] is None else min(int(info["first_step"]), global_step)
                info["last_step"] = global_step if info["last_step"] is None else max(int(info["last_step"]), global_step)
                info["mean_error_total"] += float(payload.get("isf_prediction_error") or payload.get("prediction_error") or 0.0)
                info["mean_replay_total"] += float(payload.get("memory_replay_priority") or payload.get("replay_priority") or 0.0)
                predicted_family = payload.get("predicted_family")
                actual_family = payload.get("actual_family")
                if predicted_family is not None and actual_family is not None and str(predicted_family) == str(actual_family):
                    info["prediction_success_count"] += 1
                if payload.get("prediction_error_before") is not None:
                    info["prediction_error_before_values"].append(float(payload.get("prediction_error_before")))
                if payload.get("prediction_error_after") is not None:
                    info["prediction_error_after_values"].append(float(payload.get("prediction_error_after")))
                contradiction_key = None
                if int(payload.get("context_contradiction") or payload.get("prediction_error") or 0):
                    contradiction_key = str(payload.get("context_contradiction_key") or payload.get("context_signature") or f"interaction:{payload.get('interaction_id')}")
                    contradiction = contradiction_supports.setdefault(
                        contradiction_key,
                        {
                            "support_count": 0,
                            "first_step": global_step,
                            "last_step": global_step,
                            "max_prediction_error": 0.0,
                            "mean_replay_total": 0.0,
                        },
                    )
                    contradiction["support_count"] += 1
                    contradiction["first_step"] = min(int(contradiction["first_step"]), global_step)
                    contradiction["last_step"] = max(int(contradiction["last_step"]), global_step)
                    contradiction["max_prediction_error"] = max(float(contradiction["max_prediction_error"]), float(payload.get("isf_prediction_error") or payload.get("prediction_error") or 0.0))
                    contradiction["mean_replay_total"] += float(payload.get("memory_replay_priority") or 0.0)
                _upsert_observation_graph(
                    graph_conn,
                    fold_config=fold_config,
                    game=game,
                    sampler=sampler,
                    context_signature=payload.get("context_signature"),
                    action=None if payload.get("action") is None else int(payload.get("action") or 0),
                    effect_signature=family_signature,
                    contingency_key=None,
                    family_signature=family_signature,
                    carrier_signature=None,
                    contradiction_key=contradiction_key,
                    replay_id=None,
                    totals=totals,
                )
            allow_prediction_family_fold = prediction_results_m1_fallback_rows <= 0 and (
                ("contingencies" not in tables) or raw_contingency_rows_seen > 0
            )
            for family_signature, info in family_supports.items():
                if not allow_prediction_family_fold:
                    continue
                member_count = int(info["member_count"])
                prediction_attempt_count = member_count
                prediction_success_count = int(info.get("prediction_success_count") or 0)
                prediction_accuracy_mean = (
                    float(prediction_success_count) / float(prediction_attempt_count)
                    if prediction_attempt_count > 0
                    else None
                )
                prediction_error_before_values = [float(value) for value in info.get("prediction_error_before_values", [])]
                prediction_error_after_values = [float(value) for value in info.get("prediction_error_after_values", [])]
                prediction_error_before_mean = (
                    float(sum(prediction_error_before_values) / len(prediction_error_before_values))
                    if prediction_error_before_values
                    else None
                )
                prediction_error_after_mean = (
                    float(sum(prediction_error_after_values) / len(prediction_error_after_values))
                    if prediction_error_after_values
                    else None
                )
                if prediction_error_before_mean is not None and prediction_error_after_mean is not None:
                    prediction_lift = float(prediction_error_before_mean - prediction_error_after_mean)
                elif prediction_accuracy_mean is not None:
                    prediction_lift = float(prediction_accuracy_mean)
                else:
                    prediction_lift = None
                _upsert_transformation_family(
                    state_conn,
                    family_signature=family_signature,
                    support_count=int(info["support_count"]),
                    member_count=member_count,
                    first_seen=int(info["first_step"] or fold_config.global_step_start),
                    last_seen=int(info["last_step"] or fold_config.global_step_end),
                    stability_score=float(member_count),
                    prediction_lift=prediction_lift,
                    prediction_accuracy_mean=prediction_accuracy_mean,
                    prediction_error_before_mean=prediction_error_before_mean,
                    prediction_error_after_mean=prediction_error_after_mean,
                )
                totals["transformation_families_added"] = int(totals.get("transformation_families_added", 0) or 0) + 1
                totals["transformation_families_inserted"] = int(totals.get("transformation_families_inserted", 0) or 0) + 1
                _retain_example(
                    state_conn,
                    owner_type="family",
                    owner_id=family_signature,
                    limit=fold_config.max_examples_per_family,
                    example_kind="support",
                    game=game,
                    sampler=sampler,
                    seed=seed,
                    global_step=int(info["last_step"] or fold_config.global_step_end),
                    priority_score=float(member_count) + float(info["mean_error_total"]),
                    compact_payload_json=json.dumps({"family_signature": family_signature, **info}, sort_keys=True),
                )
            for contradiction_key, info in contradiction_supports.items():
                state_conn.execute(
                    """
                    INSERT INTO contradiction_clusters (
                        cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step,
                        max_prediction_error, mean_replay_priority
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(canonical_key) DO UPDATE SET
                        support_count = contradiction_clusters.support_count + excluded.support_count,
                        first_seen_global_step = MIN(contradiction_clusters.first_seen_global_step, excluded.first_seen_global_step),
                        last_seen_global_step = MAX(contradiction_clusters.last_seen_global_step, excluded.last_seen_global_step),
                        max_prediction_error = MAX(contradiction_clusters.max_prediction_error, excluded.max_prediction_error),
                        mean_replay_priority = MAX(contradiction_clusters.mean_replay_priority, excluded.mean_replay_priority)
                    """,
                    (
                        contradiction_key,
                        contradiction_key,
                        int(info["support_count"]),
                        int(info["first_step"]),
                        int(info["last_step"]),
                        float(info["max_prediction_error"]),
                        float(info["mean_replay_total"]) / max(1, int(info["support_count"])),
                    ),
                )
                totals["contradiction_clusters_added"] += 1
                _retain_example(
                    state_conn,
                    owner_type="contradiction_cluster",
                    owner_id=contradiction_key,
                    limit=fold_config.max_examples_per_contradiction_cluster,
                    example_kind="prediction_error",
                    game=game,
                    sampler=sampler,
                    seed=seed,
                    global_step=int(info["last_step"]),
                    priority_score=float(info["max_prediction_error"]),
                    compact_payload_json=json.dumps({"canonical_key": contradiction_key, **info}, sort_keys=True),
                )
        for family_signature, contingency_keys in family_members_by_signature.items():
            for contingency_key in sorted(contingency_keys):
                state_conn.execute(
                    """
                    INSERT INTO family_members (
                        family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(family_signature, contingency_key) DO UPDATE SET
                        support_count = family_members.support_count + excluded.support_count,
                        first_seen_global_step = MIN(family_members.first_seen_global_step, excluded.first_seen_global_step),
                        last_seen_global_step = MAX(family_members.last_seen_global_step, excluded.last_seen_global_step)
                    """,
                    (family_signature, contingency_key, 1, fold_config.global_step_start, fold_config.global_step_end),
                )
                totals["family_members_inserted"] = int(totals.get("family_members_inserted", 0) or 0) + 1
        if "interactions" in tables:
            for row in raw_conn.execute("SELECT * FROM interactions WHERE COALESCE(memory_replay_candidate, 0) = 1 OR COALESCE(memory_replay_priority, 0.0) > 0.0").fetchall():
                payload = dict(row)
                replay_id = str(payload.get("id"))
                priority = float(payload.get("memory_replay_priority") or 0.0)
                reason = "carrier_linked" if payload.get("carrier_signature") else "priority"
                prediction_payload = dict(prediction_payload_by_interaction.get(replay_id, {}))
                family_signature = None
                family_id = prediction_payload.get("actual_family") or prediction_payload.get("predicted_family")
                if family_id not in (None, ""):
                    family_signature = canonical_family_signature_from_raw_db(raw_conn, family_id, prediction_payload, caches=caches)
                    _upsert_family_identity_map(state_conn, family_signature, stable_family_int_id(family_signature))
                contradiction_key = None
                if int(prediction_payload.get("context_contradiction") or prediction_payload.get("prediction_error") or 0):
                    contradiction_key = str(prediction_payload.get("context_contradiction_key") or prediction_payload.get("context_signature") or f"interaction:{replay_id}")
                replay_conn.execute(
                    """
                    INSERT INTO replay_queue (
                        replay_id, owner_type, owner_id, priority_score, reason, first_seen_global_step,
                        last_seen_global_step, compact_payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(replay_id) DO UPDATE SET
                        priority_score = MAX(replay_queue.priority_score, excluded.priority_score),
                        first_seen_global_step = MIN(replay_queue.first_seen_global_step, excluded.first_seen_global_step),
                        last_seen_global_step = MAX(replay_queue.last_seen_global_step, excluded.last_seen_global_step),
                        compact_payload_json = excluded.compact_payload_json
                    """,
                    (
                        replay_id,
                        "interaction",
                        replay_id,
                        priority,
                        reason,
                        int(payload.get("global_step") or fold_config.global_step_start),
                        int(payload.get("global_step") or fold_config.global_step_end),
                        json.dumps(_build_minimal_replay_payload(payload, prediction_payload), sort_keys=True),
                    ),
                )
                _upsert_observation_graph(
                    graph_conn,
                    fold_config=fold_config,
                    game=game,
                    sampler=sampler,
                    context_signature=prediction_payload.get("context_signature") or payload.get("context_signature"),
                    action=None if (prediction_payload.get("action") or payload.get("action")) is None else int(prediction_payload.get("action") or payload.get("action") or 0),
                    effect_signature=None,
                    contingency_key=None,
                    family_signature=family_signature,
                    carrier_signature=str(payload.get("carrier_signature") or "") or None,
                    contradiction_key=contradiction_key,
                    replay_id=replay_id,
                    totals=totals,
                )
        if "trajectory_efficiency" in tables:
            for row in raw_conn.execute("SELECT * FROM trajectory_efficiency ORDER BY trajectory_id ASC").fetchall():
                state_conn.execute(
                    """
                    INSERT OR REPLACE INTO trajectory_efficiency (
                        trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class,
                        comparable_outcome_group_id, efficiency_active, success, terminal,
                        trajectory_length, steps_to_success, best_known_solution_length,
                        normalized_solve_efficiency, future_option_gain, future_option_gain_per_action,
                        equivalent_outcome_cost_gap, loop_count, loop_ratio, repeated_state_count,
                        repeated_state_ratio, blocked_action_count, blocked_action_ratio,
                        wasted_action_count, wasted_action_ratio, unique_state_count, efficiency_score,
                        efficiency_memory_bonus, efficiency_replay_bonus, efficiency_retention_bonus,
                        efficiency_promotion_bonus
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _scope_raw_local_id("trajectory", row["trajectory_id"], db_path),
                        row["game_id"],
                        row["level_id"],
                        row["sampler"],
                        row["seed"],
                        row["epoch"],
                        row["outcome_class"],
                        row["comparable_outcome_group_id"],
                        row["efficiency_active"],
                        row["success"],
                        row["terminal"],
                        row["trajectory_length"],
                        row["steps_to_success"],
                        row["best_known_solution_length"],
                        row["normalized_solve_efficiency"],
                        row["future_option_gain"],
                        row["future_option_gain_per_action"],
                        row["equivalent_outcome_cost_gap"],
                        row["loop_count"],
                        row["loop_ratio"],
                        row["repeated_state_count"],
                        row["repeated_state_ratio"],
                        row["blocked_action_count"],
                        row["blocked_action_ratio"],
                        row["wasted_action_count"],
                        row["wasted_action_ratio"],
                        row["unique_state_count"],
                        row["efficiency_score"],
                        row["efficiency_memory_bonus"],
                        row["efficiency_replay_bonus"],
                        row["efficiency_retention_bonus"],
                        row["efficiency_promotion_bonus"],
                    ),
                )
        if "interactions" in tables:
            interaction_columns = {str(row[1]) for row in raw_conn.execute("PRAGMA table_info(interactions)").fetchall()}
            interaction_select_map = {
                "id": "id",
                "game_id": "game_id" if "game_id" in interaction_columns else f"'{game}' AS game_id",
                "level_id": "level_id" if "level_id" in interaction_columns else "NULL AS level_id",
                "sampler_name": "sampler_name" if "sampler_name" in interaction_columns else f"'{sampler}' AS sampler_name",
                "episode_id": "episode_id" if "episode_id" in interaction_columns else "NULL AS episode_id",
                "global_step": "global_step" if "global_step" in interaction_columns else "id AS global_step",
                "outcome_state": "outcome_state" if "outcome_state" in interaction_columns else "NULL AS outcome_state",
                "level_completed_event": (
                    "level_completed_event" if "level_completed_event" in interaction_columns else "0 AS level_completed_event"
                ),
                "state_hash_before": "state_hash_before" if "state_hash_before" in interaction_columns else "NULL AS state_hash_before",
                "state_hash_after": "state_hash_after" if "state_hash_after" in interaction_columns else "NULL AS state_hash_after",
                "action": "action" if "action" in interaction_columns else "NULL AS action",
                "efficiency_no_effect_action": (
                    "efficiency_no_effect_action"
                    if "efficiency_no_effect_action" in interaction_columns
                    else "0 AS efficiency_no_effect_action"
                ),
                "efficiency_future_option_gain_per_cost": (
                    "efficiency_future_option_gain_per_cost"
                    if "efficiency_future_option_gain_per_cost" in interaction_columns
                    else "NULL AS efficiency_future_option_gain_per_cost"
                ),
            }
            interaction_select_sql = ",\n                    ".join(interaction_select_map.values())
            interaction_rows = raw_conn.execute(
                f"""
                SELECT
                    {interaction_select_sql}
                FROM interactions
                ORDER BY COALESCE(global_step, id) ASC, id ASC
                """
            ).fetchall()
            for row in interaction_rows:
                scoped_interaction_id = _scope_raw_local_id("interaction", row["id"], db_path)
                state_conn.execute(
                    """
                    INSERT OR REPLACE INTO compact_interaction_trajectory_events (
                        event_id, interaction_id, game_id, level_id, sampler, seed, epoch, episode_id,
                        global_step, outcome_state, level_completed_event, state_hash_before, state_hash_after,
                        action, no_effect_action, future_option_gain
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"event:{scoped_interaction_id}",
                        scoped_interaction_id,
                        row["game_id"],
                        row["level_id"],
                        row["sampler_name"],
                        seed,
                        infer_epoch_from_path(db_path),
                        row["episode_id"],
                        row["global_step"],
                        row["outcome_state"],
                        int(row["level_completed_event"] or 0),
                        row["state_hash_before"],
                        row["state_hash_after"],
                        row["action"],
                        int(row["efficiency_no_effect_action"] or 0),
                        row["efficiency_future_option_gain_per_cost"],
                    ),
                )
        carrier_path = db_path.with_name("carrier_candidates.json")
        if carrier_path.exists():
            for item in json.loads(carrier_path.read_text(encoding="utf-8")):
                carrier_signature = str(item.get("carrier_signature") or item.get("carrier_id") or "")
                if not carrier_signature:
                    continue
                carrier_source = str(item.get("carrier_source") or "unknown")
                is_emergent = str(item.get("status") or "") == "emergent_carrier" and carrier_source != "context_action_fallback"
                first_seen_value = item.get("first_emergent_global_step") if is_emergent and item.get("first_emergent_global_step") is not None else item.get("first_seen_global_step")
                last_seen_value = item.get("last_seen_global_step")
                carrier_timing_source = "real_evidence" if (first_seen_value is not None or last_seen_value is not None) else "fold_start_fallback"
                if first_seen_value is None and last_seen_value is None:
                    totals["carrier_sidecar_missing_timing_count"] = int(totals.get("carrier_sidecar_missing_timing_count", 0) or 0) + 1
                else:
                    totals["carrier_sidecar_real_timing_count"] = int(totals.get("carrier_sidecar_real_timing_count", 0) or 0) + 1
                first_seen_step = int(first_seen_value) if first_seen_value is not None else int(fold_config.global_step_start)
                last_seen_step = int(last_seen_value) if last_seen_value is not None else int(fold_config.global_step_end)
                state_conn.execute(
                    """
                    INSERT INTO carrier_candidates (
                        carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                        first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(carrier_signature) DO UPDATE SET
                        support_count = MAX(carrier_candidates.support_count, excluded.support_count),
                        linked_family_count = MAX(carrier_candidates.linked_family_count, excluded.linked_family_count),
                        first_seen_global_step = MIN(carrier_candidates.first_seen_global_step, excluded.first_seen_global_step),
                        last_seen_global_step = MAX(carrier_candidates.last_seen_global_step, excluded.last_seen_global_step),
                        carrier_timing_source = CASE
                            WHEN carrier_candidates.carrier_timing_source = 'real_evidence'
                                 OR excluded.carrier_timing_source = 'real_evidence'
                            THEN 'real_evidence'
                            WHEN COALESCE(carrier_candidates.carrier_timing_source, 'unknown') = COALESCE(excluded.carrier_timing_source, 'unknown')
                            THEN COALESCE(excluded.carrier_timing_source, carrier_candidates.carrier_timing_source, 'unknown')
                            WHEN carrier_candidates.carrier_timing_source = 'unknown'
                                 AND excluded.carrier_timing_source = 'fold_start_fallback'
                            THEN 'fold_start_fallback'
                            WHEN carrier_candidates.carrier_timing_source = 'fold_start_fallback'
                                 AND excluded.carrier_timing_source = 'unknown'
                            THEN 'fold_start_fallback'
                            ELSE 'mixed'
                        END,
                        stability_score = MAX(carrier_candidates.stability_score, excluded.stability_score),
                        is_emergent = MAX(carrier_candidates.is_emergent, excluded.is_emergent)
                    """,
                    (
                        str(item.get("carrier_id") or carrier_signature),
                        carrier_signature,
                        carrier_source,
                        int(item.get("support_count", 0) or 0),
                        int(item.get("distinct_family_count", 0) or item.get("linked_family_count", 0) or 0),
                        first_seen_step,
                        last_seen_step,
                        carrier_timing_source,
                        float(item.get("prediction_lift", 0.0) or item.get("stability_score", 0.0) or 0.0),
                        int(is_emergent),
                    ),
                )
                totals["carrier_candidates_added"] += 1
                family_signature = None
                family_id = item.get("family_id")
                if family_id not in (None, ""):
                    family_signature = canonical_family_signature_from_raw_db(raw_conn, family_id, item, caches=caches)
                contingency_key = None
                if item.get("context_signature") is not None and family_signature is not None:
                    contingency_key = canonicalize_context_action_effect(
                        context_signature=str(item.get("context_signature")),
                        action=item.get("action", "carrier"),
                        effect_signature=family_signature,
                    )
                _retain_example(
                    state_conn,
                    owner_type="carrier",
                    owner_id=carrier_signature,
                    limit=fold_config.max_examples_per_carrier,
                    example_kind="support",
                    game=game,
                    sampler=sampler,
                    seed=seed,
                    global_step=last_seen_step,
                    priority_score=float(item.get("prediction_lift", 0.0) or 0.0) + float(item.get("support_count", 0) or 0),
                    compact_payload_json=json.dumps(item, sort_keys=True),
                )
                _upsert_carrier_link(
                    state_conn,
                    carrier_signature=carrier_signature,
                    linked_type="family",
                    linked_key=family_signature,
                    fold_config=fold_config,
                    first_seen_global_step=first_seen_step,
                    last_seen_global_step=last_seen_step,
                )
                _upsert_carrier_link(
                    state_conn,
                    carrier_signature=carrier_signature,
                    linked_type="context",
                    linked_key=None if item.get("context_signature") is None else _normalize_jsonish(item.get("context_signature")),
                    fold_config=fold_config,
                    first_seen_global_step=first_seen_step,
                    last_seen_global_step=last_seen_step,
                )
                _upsert_carrier_link(
                    state_conn,
                    carrier_signature=carrier_signature,
                    linked_type="contingency",
                    linked_key=contingency_key,
                    fold_config=fold_config,
                    first_seen_global_step=first_seen_step,
                    last_seen_global_step=last_seen_step,
                )
                _upsert_observation_graph(
                    graph_conn,
                    fold_config=fold_config,
                    game=game,
                    sampler=sampler,
                    context_signature=item.get("context_signature"),
                    action=None,
                    effect_signature=None,
                    contingency_key=contingency_key,
                    family_signature=family_signature,
                    carrier_signature=carrier_signature,
                    contradiction_key=None,
                    replay_id=None,
                    totals=totals,
                )


def _upsert_family_identity_map(connection: sqlite3.Connection, canonical_signature: str, stable_family_id: int) -> None:
    connection.execute(
        """
        INSERT INTO family_identity_map (canonical_signature, stable_family_id)
        VALUES (?, ?)
        ON CONFLICT(canonical_signature) DO UPDATE SET
            stable_family_id = excluded.stable_family_id
        """,
        (canonical_signature, int(stable_family_id)),
    )


def _fold_carrier_candidates_sidecar(
    *,
    raw_conn: sqlite3.Connection,
    carrier_path: Path,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any],
) -> None:
    db_parts = carrier_path.parts
    game = db_parts[-4] if len(db_parts) >= 4 else "unknown"
    sampler = db_parts[-3] if len(db_parts) >= 3 else "unknown"
    seed = 0
    for sqlite_candidate in sorted(carrier_path.parent.glob("seed_*.sqlite")):
        try:
            seed = int(sqlite_candidate.stem.split("_")[-1])
            break
        except (TypeError, ValueError):
            continue
    for item in json.loads(carrier_path.read_text(encoding="utf-8")):
        carrier_signature = str(item.get("carrier_signature") or item.get("carrier_id") or "")
        if not carrier_signature:
            continue
        carrier_source = str(item.get("carrier_source") or "unknown")
        is_emergent = str(item.get("status") or "") == "emergent_carrier" and carrier_source != "context_action_fallback"
        first_seen_value = item.get("first_emergent_global_step") if is_emergent and item.get("first_emergent_global_step") is not None else item.get("first_seen_global_step")
        last_seen_value = item.get("last_seen_global_step")
        carrier_timing_source = "real_evidence" if (first_seen_value is not None or last_seen_value is not None) else "fold_start_fallback"
        if first_seen_value is None and last_seen_value is None:
            totals["carrier_sidecar_missing_timing_count"] = int(totals.get("carrier_sidecar_missing_timing_count", 0) or 0) + 1
        else:
            totals["carrier_sidecar_real_timing_count"] = int(totals.get("carrier_sidecar_real_timing_count", 0) or 0) + 1
        first_seen_step = int(first_seen_value) if first_seen_value is not None else int(fold_config.global_step_start)
        last_seen_step = int(last_seen_value) if last_seen_value is not None else int(fold_config.global_step_end)
        state_conn.execute(
            """
            INSERT INTO carrier_candidates (
                carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(carrier_signature) DO UPDATE SET
                support_count = MAX(carrier_candidates.support_count, excluded.support_count),
                linked_family_count = MAX(carrier_candidates.linked_family_count, excluded.linked_family_count),
                first_seen_global_step = MIN(carrier_candidates.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(carrier_candidates.last_seen_global_step, excluded.last_seen_global_step),
                carrier_timing_source = CASE
                    WHEN carrier_candidates.carrier_timing_source = 'real_evidence'
                         OR excluded.carrier_timing_source = 'real_evidence'
                    THEN 'real_evidence'
                    WHEN COALESCE(carrier_candidates.carrier_timing_source, 'unknown') = COALESCE(excluded.carrier_timing_source, 'unknown')
                    THEN COALESCE(excluded.carrier_timing_source, carrier_candidates.carrier_timing_source, 'unknown')
                    WHEN carrier_candidates.carrier_timing_source = 'unknown'
                         AND excluded.carrier_timing_source = 'fold_start_fallback'
                    THEN 'fold_start_fallback'
                    WHEN carrier_candidates.carrier_timing_source = 'fold_start_fallback'
                         AND excluded.carrier_timing_source = 'unknown'
                    THEN 'fold_start_fallback'
                    ELSE 'mixed'
                END,
                stability_score = MAX(carrier_candidates.stability_score, excluded.stability_score),
                is_emergent = MAX(carrier_candidates.is_emergent, excluded.is_emergent)
            """,
            (
                str(item.get("carrier_id") or carrier_signature),
                carrier_signature,
                carrier_source,
                int(item.get("support_count", 0) or 0),
                int(item.get("distinct_family_count", 0) or item.get("linked_family_count", 0) or 0),
                first_seen_step,
                last_seen_step,
                carrier_timing_source,
                float(item.get("prediction_lift", 0.0) or item.get("stability_score", 0.0) or 0.0),
                int(is_emergent),
            ),
        )
        totals["carrier_candidates_added"] = int(totals.get("carrier_candidates_added", 0) or 0) + 1
        family_signature = None
        family_id = item.get("family_id")
        if family_id not in (None, ""):
            family_signature = canonical_family_signature_from_raw_db(raw_conn, family_id, item)
        contingency_key = None
        if item.get("context_signature") is not None and family_signature is not None:
            contingency_key = canonicalize_context_action_effect(
                context_signature=str(item.get("context_signature")),
                action=item.get("action", "carrier"),
                effect_signature=family_signature,
            )
        _retain_example(
            state_conn,
            owner_type="carrier",
            owner_id=carrier_signature,
            limit=fold_config.max_examples_per_carrier,
            example_kind="support",
            game=game,
            sampler=sampler,
            seed=seed,
            global_step=last_seen_step,
            priority_score=float(item.get("prediction_lift", 0.0) or 0.0) + float(item.get("support_count", 0) or 0),
            compact_payload_json=json.dumps(item, sort_keys=True),
        )
        _upsert_carrier_link(
            state_conn,
            carrier_signature=carrier_signature,
            linked_type="family",
            linked_key=family_signature,
            fold_config=fold_config,
            first_seen_global_step=first_seen_step,
            last_seen_global_step=last_seen_step,
        )
        _upsert_carrier_link(
            state_conn,
            carrier_signature=carrier_signature,
            linked_type="context",
            linked_key=None if item.get("context_signature") is None else _normalize_jsonish(item.get("context_signature")),
            fold_config=fold_config,
            first_seen_global_step=first_seen_step,
            last_seen_global_step=last_seen_step,
        )
        _upsert_carrier_link(
            state_conn,
            carrier_signature=carrier_signature,
            linked_type="contingency",
            linked_key=contingency_key,
            fold_config=fold_config,
            first_seen_global_step=first_seen_step,
            last_seen_global_step=last_seen_step,
        )
        _upsert_observation_graph(
            graph_conn,
            fold_config=fold_config,
            game=game,
            sampler=sampler,
            context_signature=item.get("context_signature"),
            action=None,
            effect_signature=None,
            contingency_key=contingency_key,
            family_signature=family_signature,
            carrier_signature=carrier_signature,
            contradiction_key=None,
            replay_id=None,
            totals=totals,
        )


def _upsert_stable_contingency(
    connection: sqlite3.Connection,
    *,
    canonical_key: str,
    contingency_id: int,
    game: str,
    sampler: str,
    context_level: int,
    action: int,
    effect_signature: str,
    support_count: int,
    prediction_attempt_count: int = 0,
    prediction_success_count: int = 0,
    prediction_accuracy: float | None = None,
    prediction_error_before: float | None = None,
    prediction_error_after: float | None = None,
    normalized_contingency_key: str | None = None,
    fold_config: CompactMemoryFoldConfig,
    stable_threshold: int,
) -> None:
    connection.execute(
        """
        INSERT INTO stable_contingencies (
            contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
            first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
            mean_replay_priority, representative_example_count, prediction_attempt_count,
            prediction_success_count, prediction_accuracy, prediction_error_before,
            prediction_error_after, normalized_contingency_key
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_key) DO UPDATE SET
            support_count = COALESCE(stable_contingencies.support_count, 0) + COALESCE(excluded.support_count, 0),
            context_level = MAX(stable_contingencies.context_level, excluded.context_level),
            first_seen_global_step = MIN(stable_contingencies.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX(stable_contingencies.last_seen_global_step, excluded.last_seen_global_step),
            stability_score = CAST(
                COALESCE(stable_contingencies.support_count, 0) + COALESCE(excluded.support_count, 0)
                AS REAL
            ) / 20.0,
            prediction_attempt_count = COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0),
            prediction_success_count = COALESCE(stable_contingencies.prediction_success_count, 0) + COALESCE(excluded.prediction_success_count, 0),
            prediction_accuracy = CASE
                WHEN (COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0)) > 0
                THEN CAST(
                    (COALESCE(stable_contingencies.prediction_success_count, 0) + COALESCE(excluded.prediction_success_count, 0)) AS REAL
                ) / CAST(
                    (COALESCE(stable_contingencies.prediction_attempt_count, 0) + COALESCE(excluded.prediction_attempt_count, 0)) AS REAL
                )
                ELSE NULL
            END,
            normalized_contingency_key = COALESCE(stable_contingencies.normalized_contingency_key, excluded.normalized_contingency_key)
        """,
        (
            contingency_id,
            canonical_key,
            game,
            sampler,
            int(context_level),
            action,
            effect_signature,
            support_count,
            fold_config.global_step_start,
            fold_config.global_step_end,
            float(support_count) / max(1.0, float(stable_threshold)),
            0.0,
            0.0,
            0,
            int(prediction_attempt_count),
            int(prediction_success_count),
            None if prediction_accuracy is None else float(prediction_accuracy),
            None if prediction_error_before is None else float(prediction_error_before),
            None if prediction_error_after is None else float(prediction_error_after),
            normalized_contingency_key
            or normalized_contingency_identity(
                context_level=int(context_level),
                action=action,
                effect_signature=effect_signature,
            ),
        ),
    )


def _upsert_carrier_link(
    connection: sqlite3.Connection,
    *,
    carrier_signature: str,
    linked_type: str,
    linked_key: str | None,
    fold_config: CompactMemoryFoldConfig,
    first_seen_global_step: int | None = None,
    last_seen_global_step: int | None = None,
) -> None:
    if linked_key in (None, ""):
        return
    connection.execute(
        """
        INSERT INTO carrier_links (
            carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(carrier_signature, linked_type, linked_key) DO UPDATE SET
            support_count = carrier_links.support_count + excluded.support_count,
            first_seen_global_step = MIN(carrier_links.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX(carrier_links.last_seen_global_step, excluded.last_seen_global_step)
        """,
        (
            str(carrier_signature),
            str(linked_type),
            str(linked_key),
            1,
            int(first_seen_global_step) if first_seen_global_step is not None else fold_config.global_step_start,
            int(last_seen_global_step) if last_seen_global_step is not None else fold_config.global_step_end,
        ),
    )


def _prediction_payload_by_interaction_id(raw_conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = raw_conn.execute("SELECT * FROM prediction_results").fetchall()
    by_interaction: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        interaction_id = str(payload.get("interaction_id"))
        if interaction_id in ("None", ""):
            continue
        existing = by_interaction.get(interaction_id)
        if existing is None:
            by_interaction[interaction_id] = payload
            continue
        current_error = float(payload.get("isf_prediction_error") or payload.get("prediction_error") or 0.0)
        existing_error = float(existing.get("isf_prediction_error") or existing.get("prediction_error") or 0.0)
        if current_error >= existing_error:
            by_interaction[interaction_id] = payload
    return by_interaction


def _fold_prediction_results_into_m1_substrate(
    *,
    raw_conn: sqlite3.Connection,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any],
    game: str,
    sampler: str,
    seed: int,
    family_members_by_signature: dict[str, set[str]],
    caches: RawDbFoldCaches | None = None,
) -> int:
    rows = raw_conn.execute("SELECT * FROM prediction_results").fetchall()
    if not rows:
        return 0
    grouped: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        actual_family = payload.get("actual_family")
        if actual_family in (None, ""):
            continue
        family_signature = canonical_family_signature_from_raw_db(raw_conn, actual_family, payload, caches=caches)
        context_level = int(payload.get("context_level") or 0)
        context_signature = str(payload.get("context_signature") or payload.get("context_action") or "[]")
        action = int(payload.get("action") or 0)
        key = (context_level, context_signature, action, family_signature)
        entry = grouped.setdefault(
            key,
            {
                "support_count": 0,
                "prediction_success_count": 0,
                "first_step": None,
                "last_step": None,
            },
        )
        entry["support_count"] += 1
        predicted_family = payload.get("predicted_family")
        if predicted_family not in (None, ""):
            predicted_signature = canonical_family_signature_from_raw_db(raw_conn, predicted_family, payload, caches=caches)
            if predicted_signature == family_signature:
                entry["prediction_success_count"] += 1
        global_step = int(payload.get("global_step") or payload.get("interaction_id") or fold_config.global_step_end)
        entry["first_step"] = global_step if entry["first_step"] is None else min(int(entry["first_step"]), global_step)
        entry["last_step"] = global_step if entry["last_step"] is None else max(int(entry["last_step"]), global_step)
    if not grouped:
        return 0
    contingency_rows_seen = 0
    for index, ((context_level, context_signature, action, family_signature), info) in enumerate(sorted(grouped.items()), start=1):
        support_count = int(info["support_count"] or 0)
        prediction_attempt_count = support_count
        prediction_success_count = int(info["prediction_success_count"] or 0)
        prediction_accuracy = (
            float(prediction_success_count) / float(prediction_attempt_count)
            if prediction_attempt_count > 0
            else None
        )
        canonical_key = canonicalize_context_action_effect(
            context_signature=context_signature,
            action=action,
            effect_signature=family_signature,
        )
        _upsert_family_identity_map(state_conn, family_signature, stable_family_int_id(family_signature))
        _upsert_stable_contingency(
            state_conn,
            canonical_key=canonical_key,
            contingency_id=index,
            game=game,
            sampler=sampler,
            context_level=context_level,
            action=action,
            effect_signature=family_signature,
            support_count=support_count,
            prediction_attempt_count=prediction_attempt_count,
            prediction_success_count=prediction_success_count,
            prediction_accuracy=prediction_accuracy,
            prediction_error_before=None,
            prediction_error_after=None,
            normalized_contingency_key=normalized_contingency_identity_cached(
                context_level=context_level,
                action=action,
                effect_signature=family_signature,
                caches=caches,
            ),
            fold_config=fold_config,
            stable_threshold=20,
        )
        totals["stable_contingencies_inserted"] = int(totals.get("stable_contingencies_inserted", 0) or 0) + 1
        if support_count >= 20:
            totals["stable_contingencies_added"] = int(totals.get("stable_contingencies_added", 0) or 0) + 1
        family_members_by_signature.setdefault(family_signature, set()).add(canonical_key)
        _upsert_transformation_family(
            state_conn,
            family_signature=family_signature,
            support_count=support_count,
            member_count=1,
            first_seen=int(info["first_step"] or fold_config.global_step_start),
            last_seen=int(info["last_step"] or fold_config.global_step_end),
            stability_score=float(support_count) / 20.0,
            prediction_lift=prediction_accuracy,
            prediction_accuracy_mean=prediction_accuracy,
            prediction_error_before_mean=None,
            prediction_error_after_mean=None,
        )
        totals["transformation_families_added"] = int(totals.get("transformation_families_added", 0) or 0) + 1
        totals["transformation_families_inserted"] = int(totals.get("transformation_families_inserted", 0) or 0) + 1
        _retain_example(
            state_conn,
            owner_type="contingency",
            owner_id=canonical_key,
            limit=fold_config.max_examples_per_contingency,
            example_kind="first",
            game=game,
            sampler=sampler,
            seed=seed,
            global_step=int(info["last_step"] or fold_config.global_step_end),
            priority_score=float(support_count),
            compact_payload_json=json.dumps(
                {
                    "context_level": context_level,
                    "context_signature": context_signature,
                    "action": action,
                    "family_signature": family_signature,
                    "support_count": support_count,
                    "prediction_attempt_count": prediction_attempt_count,
                    "prediction_success_count": prediction_success_count,
                    "prediction_accuracy": prediction_accuracy,
                },
                sort_keys=True,
            ),
        )
        _upsert_observation_graph(
            graph_conn,
            fold_config=fold_config,
            game=game,
            sampler=sampler,
            context_signature=context_signature,
            action=action,
            effect_signature=family_signature,
            contingency_key=canonical_key,
            family_signature=family_signature,
            carrier_signature=None,
            contradiction_key=None,
            replay_id=None,
            totals=totals,
        )
        contingency_rows_seen += 1
    return contingency_rows_seen


def _context_level_from_raw(
    raw_conn: sqlite3.Connection,
    *,
    payload: dict[str, Any],
    family_id: Any,
    caches: RawDbFoldCaches | None = None,
) -> int:
    if payload.get("context_level") not in (None, ""):
        return int(payload.get("context_level") or 0)
    lookup_key = (
        str(payload.get("context_signature") or payload.get("context_action") or ""),
        str(payload.get("action") if payload.get("action") is not None else -1),
        str(family_id if family_id is not None else ""),
    )
    if caches is not None and lookup_key in caches.prediction_context_level_lookup:
        return int(caches.prediction_context_level_lookup[lookup_key] or 0)
    tables = {row[0] for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "prediction_results" not in tables:
        return 0
    prediction_columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(prediction_results)").fetchall()}
    if "context_level" not in prediction_columns:
        return 0
    context_lookup_value = payload.get("context_signature") or payload.get("context_action")
    if "context_action" in prediction_columns:
        context_filter = "COALESCE(context_signature, context_action, '') = COALESCE(?, '')"
    else:
        context_filter = "COALESCE(context_signature, '') = COALESCE(?, '')"
    row = raw_conn.execute(
        f"""
        SELECT MAX(COALESCE(context_level, 0))
        FROM prediction_results
        WHERE {context_filter}
          AND COALESCE(action, -1) = COALESCE(?, -1)
          AND (
                COALESCE(actual_family, predicted_family) = ?
             OR COALESCE(actual_family, predicted_family) = ?
          )
        """,
        (
            context_lookup_value,
            payload.get("action"),
            family_id,
            str(family_id) if family_id is not None else None,
        ),
    ).fetchone()
    return int((row[0] or 0) if row is not None else 0)


def _upsert_transformation_family(
    connection: sqlite3.Connection,
    *,
    family_signature: str,
    support_count: int,
    member_count: int,
    first_seen: int,
    last_seen: int,
    stability_score: float,
    prediction_lift: float | None = None,
    prediction_accuracy_mean: float | None = None,
    prediction_error_before_mean: float | None = None,
    prediction_error_after_mean: float | None = None,
) -> None:
    stable_id = stable_family_int_id(family_signature)
    connection.execute(
        """
        INSERT INTO transformation_families (
            family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
            support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score,
            prediction_lift, prediction_accuracy_mean, prediction_error_before_mean, prediction_error_after_mean
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_signature) DO UPDATE SET
            support_count = transformation_families.support_count + excluded.support_count,
            member_count = transformation_families.member_count + excluded.member_count,
            first_seen_global_step = MIN(transformation_families.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
            stability_score = MAX(transformation_families.stability_score, excluded.stability_score),
            prediction_lift = COALESCE(excluded.prediction_lift, transformation_families.prediction_lift),
            prediction_accuracy_mean = COALESCE(excluded.prediction_accuracy_mean, transformation_families.prediction_accuracy_mean),
            prediction_error_before_mean = COALESCE(excluded.prediction_error_before_mean, transformation_families.prediction_error_before_mean),
            prediction_error_after_mean = COALESCE(excluded.prediction_error_after_mean, transformation_families.prediction_error_after_mean)
        """,
        (
            stable_id,
            family_signature,
            family_signature,
            "unknown",
            "unknown",
            "unknown",
            int(support_count),
            int(member_count),
            int(first_seen),
            int(last_seen),
            float(stability_score),
            None if prediction_lift is None else float(prediction_lift),
            None if prediction_accuracy_mean is None else float(prediction_accuracy_mean),
            None if prediction_error_before_mean is None else float(prediction_error_before_mean),
            None if prediction_error_after_mean is None else float(prediction_error_after_mean),
        ),
    )


def _graph_edge_cap_exceeded(
    graph_conn: sqlite3.Connection,
    *,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any] | None,
) -> str | None:
    del target_node_id, edge_type, totals
    total_edges = int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] or 0)
    if total_edges >= int(fold_config.max_graph_edges_per_fold):
        return "fold_cap"
    source_edges = int(
        graph_conn.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE source_node_id = ?",
            (source_node_id,),
        ).fetchone()[0]
        or 0
    )
    if source_edges >= int(fold_config.max_edges_per_source_node):
        return "source_cap"
    if source_node_id.startswith("carrier:"):
        carrier_edges = int(
            graph_conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE source_node_id = ? AND edge_type IN ('explains', 'anchors', 'appears_in')",
                (source_node_id,),
            ).fetchone()[0]
            or 0
        )
        if carrier_edges >= int(fold_config.max_edges_per_carrier):
            return "carrier_cap"
    if source_node_id.startswith("family:"):
        family_edges = int(
            graph_conn.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE source_node_id = ?",
                (source_node_id,),
            ).fetchone()[0]
            or 0
        )
        if family_edges >= int(fold_config.max_edges_per_family):
            return "family_cap"
    return None


def _ingest_live_graph_export(
    graph_conn: sqlite3.Connection,
    path: Path,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any] | None = None,
) -> None:
    if not bool(fold_config.fold_graph):
        return
    payload = _load_json(path) or {}
    for row in payload.get("nodes", []) or []:
        _upsert_graph_node(
            graph_conn,
            node_id=str(row.get("node_id")),
            node_type=str(row.get("node_type") or "Unknown"),
            canonical_key=None if row.get("canonical_key") is None else str(row.get("canonical_key")),
            fold_config=fold_config,
            support_count=int(row.get("support_count") or 1),
        )
    for row in payload.get("edges", []) or []:
        _upsert_graph_edge(
            graph_conn,
            source_node_id=str(row.get("source_node_id")),
            target_node_id=str(row.get("target_node_id")),
            edge_type=str(row.get("edge_type") or "related_to"),
            fold_config=fold_config,
            support_count=int(row.get("support_count") or 1),
            weight=float(row.get("weight") or 1.0),
            totals=totals,
        )


def _upsert_observation_graph(
    graph_conn: sqlite3.Connection,
    *,
    fold_config: CompactMemoryFoldConfig,
    game: str | None,
    sampler: str | None,
    context_signature: str | None,
    action: int | None,
    effect_signature: str | None,
    contingency_key: str | None,
    family_signature: str | None,
    carrier_signature: str | None,
    contradiction_key: str | None,
    replay_id: str | None,
    totals: dict[str, Any] | None = None,
) -> None:
    if not bool(fold_config.fold_graph):
        return
    nodes: list[tuple[str, str, str | None]] = []
    edges: list[tuple[str, str, str]] = []
    if game:
        nodes.append((f"game:{game}", "game", game))
    if sampler:
        nodes.append((f"sampler:{sampler}", "sampler", sampler))
    if context_signature:
        context_key = _normalize_jsonish(context_signature)
        nodes.append((f"context:{context_key}", "context", context_key))
        if game:
            edges.append((f"game:{game}", f"context:{context_key}", "observed_in"))
        if sampler:
            edges.append((f"sampler:{sampler}", f"context:{context_key}", "sampled_by"))
    if action is not None:
        nodes.append((f"action:{action}", "action", str(action)))
        if context_signature:
            edges.append((f"context:{_normalize_jsonish(context_signature)}", f"action:{action}", "offers_action"))
    if effect_signature:
        nodes.append((f"effect:{effect_signature}", "effect", effect_signature))
        if action is not None:
            edges.append((f"action:{action}", f"effect:{effect_signature}", "produces_effect"))
    if contingency_key:
        nodes.append((f"contingency:{contingency_key}", "contingency", contingency_key))
        if context_signature:
            edges.append((f"context:{_normalize_jsonish(context_signature)}", f"contingency:{contingency_key}", "supports"))
    if family_signature:
        nodes.append((f"family:{family_signature}", "family", family_signature))
        if contingency_key:
            edges.append((f"contingency:{contingency_key}", f"family:{family_signature}", "member_of"))
        if effect_signature:
            edges.append((f"family:{family_signature}", f"effect:{effect_signature}", "explains_effect"))
    if contradiction_key:
        nodes.append((f"contradiction:{contradiction_key}", "contradiction", contradiction_key))
        if context_signature:
            edges.append((f"context:{_normalize_jsonish(context_signature)}", f"contradiction:{contradiction_key}", "contradicts"))
        if family_signature:
            edges.append((f"family:{family_signature}", f"contradiction:{contradiction_key}", "contradicted_by"))
            edges.append((f"contradiction:{contradiction_key}", f"family:{family_signature}", "challenges"))
    if replay_id:
        nodes.append((f"replay:{replay_id}", "replay", replay_id))
        if contradiction_key:
            edges.append((f"contradiction:{contradiction_key}", f"replay:{replay_id}", "prioritizes"))
        if contingency_key:
            edges.append((f"replay:{replay_id}", f"contingency:{contingency_key}", "replays"))
        if family_signature:
            edges.append((f"replay:{replay_id}", f"family:{family_signature}", "replays"))
        if context_signature:
            edges.append((f"replay:{replay_id}", f"context:{_normalize_jsonish(context_signature)}", "replays"))
    if carrier_signature:
        nodes.append((f"carrier:{carrier_signature}", "carrier", carrier_signature))
        if family_signature:
            edges.append((f"carrier:{carrier_signature}", f"family:{family_signature}", "explains"))
        if context_signature:
            edges.append((f"carrier:{carrier_signature}", f"context:{_normalize_jsonish(context_signature)}", "appears_in"))
        if contingency_key:
            edges.append((f"carrier:{carrier_signature}", f"contingency:{contingency_key}", "anchors"))
    for node_id, node_type, canonical_key in nodes:
        _upsert_graph_node(graph_conn, node_id=node_id, node_type=node_type, canonical_key=canonical_key, fold_config=fold_config)
    for source, target, edge_type in edges:
        _upsert_graph_edge(
            graph_conn,
            source_node_id=source,
            target_node_id=target,
            edge_type=edge_type,
            fold_config=fold_config,
            totals=totals,
        )


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
        ORDER BY
            CASE example_kind
                WHEN 'first' THEN 0
                WHEN 'recent' THEN 1
                WHEN 'prediction_error' THEN 2
                WHEN 'priority' THEN 3
                ELSE 4
            END,
            priority_score DESC,
            global_step DESC,
            example_id ASC
        """,
        (owner_type, owner_id),
    ).fetchall()
    for row in rows[limit:]:
        connection.execute("DELETE FROM representative_examples WHERE example_id = ?", (row[0],))
    connection.execute(
        """
        UPDATE stable_contingencies
        SET representative_example_count = (
            SELECT COUNT(*) FROM representative_examples WHERE owner_type = 'contingency' AND owner_id = ?
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


def _upsert_graph_node(
    graph_conn: sqlite3.Connection,
    *,
    node_id: str,
    node_type: str,
    canonical_key: str | None,
    fold_config: CompactMemoryFoldConfig,
    support_count: int = 1,
) -> None:
    graph_conn.execute(
        """
        INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            node_type = COALESCE(graph_nodes.node_type, excluded.node_type),
            canonical_key = COALESCE(graph_nodes.canonical_key, excluded.canonical_key),
            first_seen_global_step = MIN(graph_nodes.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX(graph_nodes.last_seen_global_step, excluded.last_seen_global_step),
            support_count = MAX(graph_nodes.support_count, excluded.support_count)
        """,
        (node_id, node_type, canonical_key, fold_config.global_step_start, fold_config.global_step_end, int(support_count)),
    )


def _upsert_graph_edge(
    graph_conn: sqlite3.Connection,
    *,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    fold_config: CompactMemoryFoldConfig,
    support_count: int = 1,
    weight: float = 1.0,
    totals: dict[str, Any] | None = None,
) -> bool:
    if totals is not None:
        totals["graph_edges_attempted"] = int(totals.get("graph_edges_attempted", 0) or 0) + 1
    edge_id = f"{source_node_id}->{edge_type}->{target_node_id}"
    existing_edge = graph_conn.execute(
        "SELECT 1 FROM graph_edges WHERE edge_id = ? LIMIT 1",
        (edge_id,),
    ).fetchone()
    if bool(fold_config.enable_graph_edge_caps) and existing_edge is None:
        reason = _graph_edge_cap_exceeded(
            graph_conn,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            fold_config=fold_config,
            totals=totals,
        )
        if reason is not None:
            if totals is not None:
                counter_name = {
                    "fold_cap": "graph_edges_skipped_by_fold_cap",
                    "source_cap": "graph_edges_skipped_by_source_cap",
                    "carrier_cap": "graph_edges_skipped_by_carrier_cap",
                    "family_cap": "graph_edges_skipped_by_family_cap",
                }[reason]
                totals[counter_name] = int(totals.get(counter_name, 0) or 0) + 1
            return False
    graph_conn.execute(
        """
        INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(edge_id) DO UPDATE SET
            first_seen_global_step = MIN(graph_edges.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX(graph_edges.last_seen_global_step, excluded.last_seen_global_step),
            support_count = COALESCE(graph_edges.support_count, 0) + COALESCE(excluded.support_count, 0),
            weight = MAX(graph_edges.weight, excluded.weight)
        """,
        (edge_id, source_node_id, target_node_id, edge_type, fold_config.global_step_start, fold_config.global_step_end, int(support_count), float(weight)),
    )
    if totals is not None:
        totals["graph_edges_written"] = int(totals.get("graph_edges_written", 0) or 0) + 1
    return True


def _write_memory_summary_table(connection: sqlite3.Connection, summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        connection.execute(
            """
            INSERT INTO memory_summary (key, value_json)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (str(key), json.dumps(value, sort_keys=True)),
        )


def _build_memory_summary_from_connections(
    *,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    replay_conn: sqlite3.Connection,
    paths: CompactMemoryPaths,
) -> dict[str, Any]:
    stable_count = int(state_conn.execute("SELECT COUNT(*) FROM stable_contingencies WHERE support_count >= 20").fetchone()[0])
    stable_substrate_count = int(state_conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0])
    family_count = _count_rows(state_conn, "transformation_families")
    family_members_count = _count_rows(state_conn, "family_members")
    carrier_count = _count_rows(state_conn, "carrier_candidates")
    role_candidate_count = _count_rows(state_conn, "role_candidates")
    emergent_role_count = _safe_scalar(state_conn, "SELECT COUNT(*) FROM role_candidates WHERE COALESCE(is_emergent, 0) = 1")
    role_transfer_attempt_count = _count_rows(state_conn, "role_transfer_attempts")
    successful_role_transfer_count = _safe_scalar(
        state_conn,
        "SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1",
    )
    concept_candidate_count = _count_rows(state_conn, "concept_candidates")
    promoted_concept_count = _safe_scalar(state_conn, "SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1")
    world_model_component_count = _count_rows(state_conn, "world_model_components")
    coherent_world_model_component_count = _safe_scalar(
        state_conn,
        "SELECT COUNT(*) FROM world_model_components WHERE COALESCE(is_coherent, 0) = 1",
    )
    future_option_event_count = _count_rows(state_conn, "future_option_events")
    future_option_motif_count = _count_rows(state_conn, "future_option_motifs")
    emergent_future_option_motif_count = _safe_scalar(
        state_conn,
        "SELECT COUNT(*) FROM future_option_motifs WHERE COALESCE(is_emergent, 0) = 1",
    )
    future_option_attention_link_count = _count_rows(state_conn, "future_option_attention_links")
    future_option_transfer_link_count = _count_rows(state_conn, "future_option_transfer_links")
    compact_trajectory_event_count = _count_rows(state_conn, "compact_interaction_trajectory_events")
    contradiction_count = _count_rows(state_conn, "contradiction_clusters")
    example_count = _count_rows(state_conn, "representative_examples")
    emergent_carrier_count = int(
        state_conn.execute("SELECT COUNT(*) FROM carrier_candidates WHERE COALESCE(is_emergent, 0) = 1").fetchone()[0]
    )
    memory_record_count = _safe_scalar(
        state_conn,
        """
        SELECT COUNT(*)
        FROM memory_scores
        WHERE node_id LIKE 'M0:interaction:%'
          AND (
            isf_total IS NOT NULL
            OR retention_status IS NOT NULL
            OR replay_priority IS NOT NULL
          )
        """,
    )
    if int(memory_record_count or 0) <= 0:
        memory_record_count = _safe_scalar(
            state_conn,
            "SELECT COUNT(*) FROM memory_nodes WHERE node_type = 'InteractionMemory'",
        )
    memory_replay_candidate_count = _safe_scalar(
        state_conn,
        "SELECT COUNT(*) FROM memory_edges WHERE edge_type = 'selected_for_replay'",
    )
    high_replay_priority_count = _safe_scalar(
        state_conn,
        """
        SELECT COUNT(*)
        FROM memory_scores
        WHERE node_id LIKE 'M0:interaction:%'
          AND COALESCE(replay_priority, 0.0) >= 0.50
        """,
    )
    retention_rows = _safe_rows(
        state_conn,
        """
        SELECT retention_status, COUNT(*)
        FROM memory_scores
        WHERE node_id LIKE 'M0:interaction:%'
          AND retention_status IS NOT NULL
        GROUP BY retention_status
        """,
    )
    retention_counts = {
        str(row[0]): int(row[1] or 0)
        for row in retention_rows
        if row and row[0] not in (None, "")
    }
    summary = {
        "stable_contingency_count": stable_count,
        "stable_contingencies_count": stable_substrate_count,
        "transformation_family_count": family_count,
        "transformation_families_count": family_count,
        "family_members_count": family_members_count,
        "carrier_candidate_count": carrier_count,
        "emergent_carrier_count": emergent_carrier_count,
        "role_candidate_count": role_candidate_count,
        "emergent_role_count": emergent_role_count,
        "role_transfer_attempt_count": role_transfer_attempt_count,
        "successful_role_transfer_count": successful_role_transfer_count,
        "concept_candidate_count": concept_candidate_count,
        "promoted_concept_count": promoted_concept_count,
        "world_model_component_count": world_model_component_count,
        "coherent_world_model_component_count": coherent_world_model_component_count,
        "future_option_event_count": future_option_event_count,
        "future_option_motif_count": future_option_motif_count,
        "emergent_future_option_motif_count": emergent_future_option_motif_count,
        "future_option_attention_link_count": future_option_attention_link_count,
        "future_option_transfer_link_count": future_option_transfer_link_count,
        "compact_trajectory_event_count": compact_trajectory_event_count,
        "contradiction_cluster_count": contradiction_count,
        "representative_example_count": example_count,
        "graph_node_count": _count_rows(graph_conn, "graph_nodes"),
        "graph_edge_count": _count_rows(graph_conn, "graph_edges"),
        "memory_substrate_node_count": _count_rows(state_conn, "memory_nodes"),
        "memory_substrate_edge_count": _count_rows(state_conn, "memory_edges"),
        "memory_substrate_evidence_count": _count_rows(state_conn, "memory_evidence"),
        "memory_substrate_score_count": _count_rows(state_conn, "memory_scores"),
        "memory_substrate_promotion_count": _count_rows(state_conn, "memory_promotions"),
        "memory_record_count": memory_record_count,
        "memory_replay_candidate_count": memory_replay_candidate_count,
        "high_replay_priority_count": high_replay_priority_count,
        "memory_retention_status_counts": retention_counts,
        "protected_memory_count": int(retention_counts.get("protected", 0)),
        "active_memory_count": int(retention_counts.get("active", 0)),
        "forgotten_memory_count": int(retention_counts.get("forgotten", 0)),
        "compressed_memory_count": int(retention_counts.get("compressed", 0)),
        "replay_queue_size": _count_rows(replay_conn, "replay_queue"),
        "current_state_path": str(paths.current_state),
        "graph_path": str(paths.graph),
        "replay_queue_path": str(paths.replay_queue),
    }
    for key in (
        "stable_contingency_count",
        "transformation_family_count",
        "carrier_candidate_count",
        "role_candidate_count",
        "emergent_role_count",
        "role_transfer_attempt_count",
        "successful_role_transfer_count",
        "concept_candidate_count",
        "promoted_concept_count",
        "world_model_component_count",
        "coherent_world_model_component_count",
        "future_option_event_count",
        "future_option_motif_count",
        "emergent_future_option_motif_count",
        "future_option_attention_link_count",
        "future_option_transfer_link_count",
        "compact_trajectory_event_count",
        "graph_node_count",
        "graph_edge_count",
        "memory_substrate_node_count",
        "memory_substrate_edge_count",
        "memory_substrate_evidence_count",
        "memory_substrate_score_count",
        "memory_substrate_promotion_count",
        "memory_record_count",
        "memory_replay_candidate_count",
        "high_replay_priority_count",
        "protected_memory_count",
        "active_memory_count",
        "forgotten_memory_count",
        "compressed_memory_count",
        "replay_queue_size",
    ):
        summary[key] = int(summary[key] or 0)
    return summary


def _safe_rows(connection: sqlite3.Connection, query: str) -> list[tuple[Any, ...]]:
    try:
        return list(connection.execute(query).fetchall())
    except sqlite3.DatabaseError:
        return []


def _count_rows(connection: sqlite3.Connection, table: str) -> int:
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.DatabaseError:
        return 0


def _repair_family_effect_type(effect_text: str) -> str:
    lowered = str(effect_text or "unknown").lower()
    if any(token in lowered for token in ("negative", "restrict", "collapse", "game_over")):
        return "negative_change"
    if any(token in lowered for token in ("positive", "expand", "enable", "win")):
        return "positive_change"
    if any(token in lowered for token in ("preserve", "stable", "neutral", "no_change")):
        return "no_change"
    return "mixed_change"


def _repair_family_polarity(effect_text: str) -> str:
    lowered = str(effect_text or "unknown").lower()
    if any(token in lowered for token in ("negative", "restrict", "collapse", "game_over")):
        return "negative"
    if any(token in lowered for token in ("positive", "expand", "enable", "win")):
        return "positive"
    return "mixed"


def _write_family_repair_summary(connection: sqlite3.Connection, summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        connection.execute(
            """
            INSERT INTO memory_summary (key, value_json)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (str(key), json.dumps(value)),
        )


def _stable_family_int_id(canonical_signature: str) -> int:
    return int(sha1(str(canonical_signature).encode("utf-8")).hexdigest()[:12], 16) % 2_000_000_000


def _safe_scalar(connection: sqlite3.Connection, query: str) -> int:
    try:
        row = connection.execute(query).fetchone()
    except sqlite3.DatabaseError:
        return 0
    if row is None:
        return 0
    return int(row[0] or 0)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _normalize_jsonish(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    text = str(value)
    try:
        loaded = json.loads(text)
    except Exception:
        return text
    return json.dumps(loaded, sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_jsonish_cached(value: Any, caches: RawDbFoldCaches | None = None) -> str:
    text = str(value)
    if caches is not None and text in caches.normalized_jsonish:
        return caches.normalized_jsonish[text]
    normalized = _normalize_jsonish(value)
    if caches is not None:
        caches.normalized_jsonish[text] = normalized
    return normalized


def _path_segment(path: Path, index_from_end: int) -> str:
    parts = path.parts
    try:
        return str(parts[index_from_end])
    except IndexError:
        return "unknown"


def _seed_from_db_path(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 0


def _db_interaction_scope(db_path: Path) -> dict[str, Any]:
    return {
        "game": _path_segment(db_path, -4),
        "sampler": _path_segment(db_path, -3),
        "seed": _seed_from_db_path(db_path),
    }


def _scope_raw_local_id(prefix: str, raw_id: Any, db_path: Path) -> str:
    value = str(raw_id)
    db_scope = sha1(str(Path(db_path).resolve()).encode("utf-8")).hexdigest()[:12]
    scoped_prefix = f"{prefix}:db:{db_scope}:"
    if value.startswith(f"{prefix}:"):
        return value if scoped_prefix in value else f"{scoped_prefix}{value}"
    return f"{scoped_prefix}{value}"


def _scope_memory_node_id(node_id: str, db_path: Path) -> str:
    for prefix in ("M0:interaction:", "M0:replay:", "M0:cost:", "M0:future_option_delta:"):
        if not str(node_id).startswith(prefix):
            continue
        suffix = str(node_id).split(prefix, 1)[1]
        if suffix.startswith("g") or suffix.startswith("local:"):
            return str(node_id)
        scope = _db_interaction_scope(db_path)
        scoped_key = scoped_interaction_key(
            interaction_id=suffix,
            game=scope["game"],
            sampler=scope["sampler"],
            seed=scope["seed"],
        )
        if prefix == "M0:interaction:":
            return interaction_node_id(scoped_key)
        return f"{prefix}{scoped_key}"
    return str(node_id)


def _scope_payload_interaction_nodes(value: Any, db_path: Path) -> Any:
    if isinstance(value, dict):
        return {str(key): _scope_payload_interaction_nodes(item, db_path) for key, item in value.items()}
    if isinstance(value, list):
        return [_scope_payload_interaction_nodes(item, db_path) for item in value]
    if isinstance(value, tuple):
        return [_scope_payload_interaction_nodes(item, db_path) for item in value]
    if isinstance(value, str) and value.startswith("M0:"):
        return _scope_memory_node_id(value, db_path)
    return value


def _scope_memory_canonical_key(node_id: str, node_type: str | None, canonical_key: Any, db_path: Path) -> str | None:
    if canonical_key is None:
        return None
    if str(node_type or "") == "InteractionMemory" or str(node_id).startswith("M0:interaction:"):
        return _scope_raw_local_id("canonical", canonical_key, db_path)
    return str(canonical_key)


def _json_loads_or_none(value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def _json_dumps_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_carrier_timing_source(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"real_evidence", "fold_start_fallback", "mixed", "unknown"}:
        return text
    return "unknown"


def _ensure_current_state_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        configure_compact_sqlite_connection(connection, write=True)
        existing_tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        legacy_transfer_provenance = False
        if "role_transfer_attempts" in existing_tables:
            attempt_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(role_transfer_attempts)").fetchall()
            }
            legacy_transfer_provenance = (
                (
                    "source_game_key" not in attempt_columns
                    or "predicted_target_role_signature" not in attempt_columns
                    or "provenance_status" not in attempt_columns
                )
                and bool(connection.execute("SELECT 1 FROM role_transfer_attempts LIMIT 1").fetchone())
            )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS stable_contingencies (
                contingency_id INTEGER,
                canonical_key TEXT PRIMARY KEY,
                game TEXT,
                sampler TEXT,
                context_level INTEGER DEFAULT 0,
                action INTEGER,
                effect_signature TEXT,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                stability_score REAL,
                mean_prediction_error REAL,
                mean_replay_priority REAL,
                representative_example_count INTEGER,
                prediction_attempt_count INTEGER DEFAULT 0,
                prediction_success_count INTEGER DEFAULT 0,
                prediction_accuracy REAL,
                prediction_error_before REAL,
                prediction_error_after REAL,
                normalized_contingency_key TEXT
            );
            CREATE TABLE IF NOT EXISTS transformation_families (
                family_id INTEGER,
                canonical_signature TEXT PRIMARY KEY,
                relaxed_signature TEXT,
                effect_type TEXT,
                action_group TEXT,
                polarity TEXT,
                support_count INTEGER,
                member_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                stability_score REAL,
                prediction_lift REAL,
                prediction_accuracy_mean REAL,
                prediction_error_before_mean REAL,
                prediction_error_after_mean REAL
            );
            CREATE TABLE IF NOT EXISTS family_identity_map (
                canonical_signature TEXT PRIMARY KEY,
                stable_family_id INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS family_members (
                family_signature TEXT,
                contingency_key TEXT,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                PRIMARY KEY (family_signature, contingency_key)
            );
            CREATE TABLE IF NOT EXISTS carrier_candidates (
                carrier_id TEXT,
                carrier_signature TEXT PRIMARY KEY,
                carrier_source TEXT,
                support_count INTEGER,
                linked_family_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                carrier_timing_source TEXT DEFAULT 'unknown',
                stability_score REAL,
                is_emergent INTEGER
            );
            CREATE TABLE IF NOT EXISTS carrier_links (
                carrier_signature TEXT,
                linked_type TEXT,
                linked_key TEXT,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                PRIMARY KEY (carrier_signature, linked_type, linked_key)
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
            CREATE TABLE IF NOT EXISTS trajectory_efficiency (
                trajectory_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                level_id TEXT,
                sampler TEXT,
                seed INTEGER,
                epoch INTEGER,
                outcome_class TEXT NOT NULL,
                comparable_outcome_group_id TEXT NOT NULL,
                efficiency_active INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 0,
                terminal INTEGER NOT NULL DEFAULT 0,
                trajectory_length INTEGER NOT NULL,
                steps_to_success INTEGER,
                best_known_solution_length INTEGER,
                normalized_solve_efficiency REAL,
                future_option_gain REAL,
                future_option_gain_per_action REAL,
                equivalent_outcome_cost_gap REAL,
                loop_count INTEGER,
                loop_ratio REAL,
                repeated_state_count INTEGER,
                repeated_state_ratio REAL,
                blocked_action_count INTEGER,
                blocked_action_ratio REAL,
                wasted_action_count INTEGER,
                wasted_action_ratio REAL,
                unique_state_count INTEGER,
                efficiency_score REAL,
                efficiency_memory_bonus REAL,
                efficiency_replay_bonus REAL,
                efficiency_retention_bonus REAL,
                efficiency_promotion_bonus REAL
            );
            CREATE TABLE IF NOT EXISTS compact_interaction_trajectory_events (
                event_id TEXT PRIMARY KEY,
                interaction_id TEXT NOT NULL,
                game_id TEXT,
                level_id TEXT,
                sampler TEXT,
                seed INTEGER,
                epoch INTEGER,
                episode_id INTEGER,
                global_step INTEGER,
                outcome_state TEXT,
                level_completed_event INTEGER,
                state_hash_before TEXT,
                state_hash_after TEXT,
                action INTEGER,
                no_effect_action INTEGER,
                future_option_gain REAL
            );
            CREATE TABLE IF NOT EXISTS memory_nodes (
                node_id TEXT PRIMARY KEY,
                memory_level TEXT NOT NULL,
                node_type TEXT NOT NULL,
                canonical_key TEXT,
                support_count INTEGER DEFAULT 0,
                first_seen_step INTEGER,
                last_seen_step INTEGER,
                attrs_json TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                support_count INTEGER DEFAULT 1,
                evidence_json TEXT,
                UNIQUE(source_node_id, target_node_id, edge_type)
            );
            CREATE TABLE IF NOT EXISTS memory_evidence (
                evidence_id TEXT PRIMARY KEY,
                target_node_id TEXT NOT NULL,
                source_interaction_id INTEGER,
                evidence_type TEXT NOT NULL,
                payload_json TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_scores (
                node_id TEXT PRIMARY KEY,
                isf_total REAL,
                prediction_lift REAL,
                transfer_score REAL,
                explanatory_reach REAL,
                compression_gain REAL,
                future_option_delta REAL,
                replay_priority REAL,
                retention_status TEXT,
                memory_state TEXT,
                stored_epoch INTEGER,
                last_replayed_epoch INTEGER,
                last_promoted_epoch INTEGER,
                retention_score REAL,
                forgetting_score REAL,
                compressed_into_id TEXT,
                superseded_by_id TEXT,
                forgetting_reason TEXT,
                updated_step INTEGER
            );
            CREATE TABLE IF NOT EXISTS memory_promotions (
                promotion_id TEXT PRIMARY KEY,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                promotion_type TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                promotion_score REAL NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT
            );
            CREATE TABLE IF NOT EXISTS role_neighborhood_signatures (
                carrier_signature TEXT PRIMARY KEY,
                role_signature TEXT,
                role_type TEXT,
                token_json TEXT,
                diagnostic_token_json TEXT,
                linked_family_count INTEGER,
                linked_context_count INTEGER,
                linked_game_count INTEGER,
                in_edge_count INTEGER,
                out_edge_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                stability_score REAL
            );
            CREATE TABLE IF NOT EXISTS role_candidates (
                role_signature TEXT PRIMARY KEY,
                role_type TEXT,
                support_count INTEGER,
                linked_carrier_count INTEGER,
                linked_family_count INTEGER,
                linked_context_count INTEGER,
                cross_game_count INTEGER,
                cross_context_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                role_stability_score REAL,
                is_emergent INTEGER,
                role_signature_token_count INTEGER,
                diagnostic_token_count INTEGER,
                exact_family_token_count INTEGER,
                exact_identity_token_count INTEGER,
                role_explanatory_coverage REAL,
                role_compression_gain REAL,
                role_prediction_lift REAL,
                role_future_option_lift REAL,
                role_transfer_lift REAL,
                promotion_status TEXT,
                promotion_failure_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS role_links (
                role_signature TEXT,
                linked_type TEXT,
                linked_key TEXT,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                PRIMARY KEY (role_signature, linked_type, linked_key)
            );
            CREATE TABLE IF NOT EXISTS role_transfer_attempts (
                attempt_id TEXT PRIMARY KEY,
                role_signature TEXT,
                transfer_kind TEXT,
                source_scope_type TEXT,
                source_scope_key TEXT,
                target_scope_type TEXT,
                target_scope_key TEXT,
                source_game_key TEXT,
                target_game_key TEXT,
                source_context_key TEXT,
                target_context_key TEXT,
                source_carrier_signature TEXT,
                source_role_signature TEXT,
                predicted_target_role_signature TEXT,
                observed_target_role_signature TEXT,
                source_carrier_signatures_json TEXT,
                source_game_keys_json TEXT,
                source_context_keys_json TEXT,
                provenance_mode TEXT,
                provenance_status TEXT,
                target_carrier_signature TEXT,
                predicted_role_signature TEXT,
                observed_role_signature TEXT,
                similarity_score REAL,
                transfer_score REAL,
                reuse_success INTEGER,
                failure_reason TEXT,
                best_margin REAL,
                source_carrier_count INTEGER,
                candidate_role_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER
            );
            CREATE TABLE IF NOT EXISTS concept_candidates (
                concept_signature TEXT PRIMARY KEY,
                concept_type TEXT,
                support_count INTEGER,
                linked_role_count INTEGER,
                linked_carrier_count INTEGER,
                linked_family_count INTEGER,
                transfer_success_count INTEGER,
                strong_transfer_success_count INTEGER,
                cross_game_count INTEGER,
                cross_context_count INTEGER,
                compression_gain REAL,
                explanatory_reach REAL,
                promotion_score REAL,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                is_promoted INTEGER,
                concept_incremental_coverage REAL,
                concept_incremental_compression_gain REAL,
                concept_prediction_lift REAL,
                concept_future_option_prediction_lift REAL,
                concept_cross_game_transfer_lift REAL,
                validation_scope TEXT,
                validation_prediction_lift REAL,
                validation_action_selection_lift REAL,
                validation_transfer_lift REAL,
                validation_contradiction_resolution REAL,
                validation_explanatory_gain REAL,
                validation_evidence_count INTEGER,
                promotion_status TEXT,
                promotion_failure_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS concept_links (
                concept_signature TEXT,
                linked_type TEXT,
                linked_key TEXT,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                PRIMARY KEY (concept_signature, linked_type, linked_key)
            );
            CREATE TABLE IF NOT EXISTS concept_promotion_validation_diagnostics (
                concept_signature TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concept_incremental_coverage_state (
                candidate_signature TEXT PRIMARY KEY,
                epoch_id TEXT,
                structure_fingerprint TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_global_step INTEGER
            );
            CREATE TABLE IF NOT EXISTS world_model_components (
                component_signature TEXT PRIMARY KEY,
                component_type TEXT,
                node_count INTEGER,
                edge_count INTEGER,
                linked_concept_count INTEGER,
                linked_role_count INTEGER,
                linked_family_count INTEGER,
                linked_carrier_count INTEGER,
                cross_context_count INTEGER,
                cross_game_count INTEGER,
                explanatory_coverage REAL,
                prediction_support_count INTEGER,
                contradiction_coverage_count INTEGER,
                coherence_score REAL,
                candidate_only INTEGER,
                predicted_outcome_count INTEGER,
                predicted_outcome_count_is_proxy INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                is_coherent INTEGER,
                validation_prediction_lift REAL,
                validation_action_selection_lift REAL,
                validation_transfer_lift REAL,
                promotion_status TEXT,
                promotion_failure_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS world_model_links (
                component_signature TEXT,
                linked_type TEXT,
                linked_key TEXT,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                PRIMARY KEY (component_signature, linked_type, linked_key)
            );
            CREATE TABLE IF NOT EXISTS higher_order_milestones (
                milestone_name TEXT PRIMARY KEY,
                first_global_step INTEGER,
                evidence_key TEXT
            );
            CREATE TABLE IF NOT EXISTS higher_order_milestone_history (
                milestone_name TEXT PRIMARY KEY,
                first_global_step INTEGER,
                evidence_key TEXT
            );
            CREATE TABLE IF NOT EXISTS promotion_validation_state (
                candidate_type TEXT,
                candidate_signature TEXT,
                failure_count INTEGER NOT NULL DEFAULT 0,
                promotion_status TEXT NOT NULL DEFAULT 'candidate',
                last_validation_scope TEXT,
                last_validation_prediction_lift REAL,
                last_validation_action_selection_lift REAL,
                last_validation_transfer_lift REAL,
                last_validation_epoch TEXT,
                last_validation_global_step INTEGER,
                last_validation_result TEXT,
                updated_global_step INTEGER,
                PRIMARY KEY (candidate_type, candidate_signature)
            );
            CREATE TABLE IF NOT EXISTS future_option_events (
                event_id TEXT PRIMARY KEY,
                owner_type TEXT,
                owner_key TEXT,
                game TEXT,
                sampler TEXT,
                context_key TEXT,
                action_key TEXT,
                source_kind TEXT,
                motif_type TEXT,
                option_delta REAL,
                option_delta_bucket TEXT,
                option_count_before REAL,
                option_count_after REAL,
                novelty_score REAL,
                reversibility_score REAL,
                branching_score REAL,
                termination_score REAL,
                contradiction_score REAL,
                replay_priority_score REAL,
                memory_priority_score REAL,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                source_interaction_id TEXT,
                source_family_id TEXT,
                source_carrier_id TEXT,
                source_role_id TEXT,
                source_concept_id TEXT,
                source_context_signature TEXT,
                source_action TEXT,
                source_game_id TEXT,
                source_sampler TEXT,
                future_option_development_stage TEXT,
                survival_delta REAL,
                movement_freedom_delta REAL,
                environmental_influence_delta REAL,
                graph_expansion_delta REAL,
                role_discovery_delta REAL,
                concept_transfer_delta REAL,
                developmental_option_value REAL,
                motif_classification_reason TEXT,
                classification_source TEXT,
                classification_rule TEXT,
                classification_evidence_id TEXT,
                evidence_json TEXT
            );
            CREATE TABLE IF NOT EXISTS future_option_motifs (
                motif_signature TEXT PRIMARY KEY,
                motif_type TEXT,
                support_count INTEGER,
                linked_event_count INTEGER,
                linked_family_count INTEGER,
                linked_carrier_count INTEGER,
                linked_role_count INTEGER,
                linked_concept_count INTEGER,
                cross_context_count INTEGER,
                cross_game_count INTEGER,
                mean_option_delta REAL,
                mean_abs_option_delta REAL,
                mean_novelty_score REAL,
                mean_reversibility_score REAL,
                mean_branching_score REAL,
                mean_termination_score REAL,
                mean_replay_priority_score REAL,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                motif_stability_score REAL,
                is_emergent INTEGER,
                source_interaction_ids_json TEXT,
                source_family_ids_json TEXT,
                source_carrier_ids_json TEXT,
                source_role_ids_json TEXT,
                source_concept_ids_json TEXT,
                future_option_development_stage TEXT,
                development_component_means_json TEXT,
                motif_classification_reason TEXT,
                classification_source TEXT,
                classification_rule TEXT,
                classification_evidence_id TEXT,
                source_game_keys_json TEXT,
                target_game_keys_json TEXT,
                source_context_keys_json TEXT,
                target_context_keys_json TEXT
            );
            CREATE TABLE IF NOT EXISTS future_option_links (
                motif_signature TEXT,
                linked_type TEXT,
                linked_key TEXT,
                support_count INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                PRIMARY KEY (motif_signature, linked_type, linked_key)
            );
            CREATE TABLE IF NOT EXISTS future_option_attention_links (
                event_id TEXT PRIMARY KEY,
                motif_signature TEXT,
                owner_type TEXT,
                owner_key TEXT,
                option_delta_abs REAL,
                replay_priority_score REAL,
                memory_priority_score REAL,
                contradiction_score REAL,
                high_option_change INTEGER,
                high_attention INTEGER,
                raw_high_attention INTEGER,
                calibrated_high_attention INTEGER,
                source_label TEXT,
                attention_signal_source TEXT,
                attention_score REAL,
                attention_score_percentile REAL,
                attention_threshold_method TEXT,
                attention_calibration_degenerate INTEGER,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER
            );
            CREATE TABLE IF NOT EXISTS future_option_transfer_links (
                motif_signature TEXT,
                role_signature TEXT,
                concept_signature TEXT NOT NULL DEFAULT '__none__',
                transfer_attempt_count INTEGER,
                successful_transfer_count INTEGER,
                strong_transfer_success_count INTEGER,
                promoted_concept_count INTEGER,
                mean_transfer_score REAL,
                mean_best_margin REAL,
                source_role_signature TEXT,
                source_game_key TEXT,
                target_game_key TEXT,
                source_context_key TEXT,
                target_context_key TEXT,
                provenance_mode TEXT,
                motif_provenance_status TEXT,
                transfer_provenance_status TEXT,
                concept_validation_status TEXT,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                PRIMARY KEY (motif_signature, role_signature, concept_signature)
            );
            CREATE TABLE IF NOT EXISTS graph_epoch_summary (
                epoch_key TEXT PRIMARY KEY,
                global_step_start INTEGER,
                global_step_end INTEGER,
                graph_edges_attempted INTEGER DEFAULT 0,
                graph_edges_written INTEGER DEFAULT 0,
                graph_edges_skipped_by_fold_cap INTEGER DEFAULT 0,
                graph_edges_skipped_by_source_cap INTEGER DEFAULT 0,
                graph_edges_skipped_by_carrier_cap INTEGER DEFAULT 0,
                graph_edges_skipped_by_family_cap INTEGER DEFAULT 0,
                graph_edges_pruned_by_post_merge_fold_cap INTEGER DEFAULT 0,
                graph_edges_pruned_by_post_merge_source_cap INTEGER DEFAULT 0,
                graph_edges_pruned_by_post_merge_carrier_cap INTEGER DEFAULT 0,
                graph_edges_pruned_by_post_merge_family_cap INTEGER DEFAULT 0,
                graph_edges_written_by_set_based_merge INTEGER DEFAULT 0,
                graph_edges_before_set_based_merge INTEGER DEFAULT 0,
                graph_edges_after_set_based_merge INTEGER DEFAULT 0,
                cumulative_graph_node_count INTEGER DEFAULT 0,
                cumulative_graph_edge_count INTEGER DEFAULT 0,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_role_links_type_key
            ON role_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_carrier_links_signature_type_key
            ON carrier_links(carrier_signature, linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_carrier_links_type_key
            ON carrier_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_role_neighborhood_carrier_signature
            ON role_neighborhood_signatures(carrier_signature);
            CREATE INDEX IF NOT EXISTS idx_role_neighborhood_role
            ON role_neighborhood_signatures(role_signature);
            CREATE INDEX IF NOT EXISTS idx_role_links_signature_type_key
            ON role_links(role_signature, linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_role_transfer_role
            ON role_transfer_attempts(role_signature);
            CREATE INDEX IF NOT EXISTS idx_role_transfer_kind_scope
            ON role_transfer_attempts(transfer_kind, target_scope_key);
            CREATE INDEX IF NOT EXISTS idx_role_transfer_success
            ON role_transfer_attempts(reuse_success, transfer_kind);
            CREATE INDEX IF NOT EXISTS idx_concept_links_type_key
            ON concept_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_concept_links_signature_type_key
            ON concept_links(concept_signature, linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_world_model_links_type_key
            ON world_model_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_future_option_events_owner
            ON future_option_events(owner_type, owner_key);
            CREATE INDEX IF NOT EXISTS idx_future_option_events_motif
            ON future_option_events(motif_type, option_delta_bucket);
            CREATE INDEX IF NOT EXISTS idx_future_option_events_source_family
            ON future_option_events(source_family_id);
            CREATE INDEX IF NOT EXISTS idx_future_option_events_source_carrier
            ON future_option_events(source_carrier_id);
            CREATE INDEX IF NOT EXISTS idx_future_option_events_source_role
            ON future_option_events(source_role_id);
            CREATE INDEX IF NOT EXISTS idx_future_option_events_development_stage
            ON future_option_events(future_option_development_stage, motif_classification_reason);
            CREATE INDEX IF NOT EXISTS idx_future_option_motifs_emergent
            ON future_option_motifs(is_emergent, motif_type);
            CREATE INDEX IF NOT EXISTS idx_future_option_links_type_key
            ON future_option_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_future_option_links_motif_type_key
            ON future_option_links(motif_signature, linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_graph_epoch_summary_steps
            ON graph_epoch_summary(global_step_start, global_step_end);
            CREATE INDEX IF NOT EXISTS idx_future_option_attention_flags
            ON future_option_attention_links(high_option_change, high_attention);
            CREATE INDEX IF NOT EXISTS idx_future_option_transfer_motif
            ON future_option_transfer_links(motif_signature);
            CREATE INDEX IF NOT EXISTS idx_trajectory_efficiency_scope
            ON trajectory_efficiency(game_id, level_id, sampler, seed, epoch);
            CREATE INDEX IF NOT EXISTS idx_carrier_candidates_timing_source
            ON carrier_candidates(carrier_timing_source);
            CREATE INDEX IF NOT EXISTS idx_compact_trajectory_events_scope
            ON compact_interaction_trajectory_events(game_id, level_id, sampler, seed, epoch, episode_id, global_step);
            """
        )
        _ensure_column(connection, "stable_contingencies", "context_level", "INTEGER DEFAULT 0")
        _ensure_column(connection, "stable_contingencies", "prediction_attempt_count", "INTEGER DEFAULT 0")
        _ensure_column(connection, "stable_contingencies", "prediction_success_count", "INTEGER DEFAULT 0")
        _ensure_column(connection, "stable_contingencies", "prediction_accuracy", "REAL")
        _ensure_column(connection, "stable_contingencies", "prediction_error_before", "REAL")
        _ensure_column(connection, "stable_contingencies", "prediction_error_after", "REAL")
        _ensure_column(connection, "stable_contingencies", "normalized_contingency_key", "TEXT")
        _ensure_column(connection, "carrier_candidates", "carrier_timing_source", "TEXT DEFAULT 'unknown'")
        _ensure_column(connection, "graph_epoch_summary", "graph_edges_written_by_set_based_merge", "INTEGER DEFAULT 0")
        _ensure_column(connection, "graph_epoch_summary", "graph_edges_before_set_based_merge", "INTEGER DEFAULT 0")
        _ensure_column(connection, "graph_epoch_summary", "graph_edges_after_set_based_merge", "INTEGER DEFAULT 0")
        _ensure_column(connection, "transformation_families", "canonical_signature", "TEXT")
        _ensure_column(connection, "transformation_families", "relaxed_signature", "TEXT")
        _ensure_column(connection, "transformation_families", "effect_type", "TEXT")
        _ensure_column(connection, "transformation_families", "action_group", "TEXT")
        _ensure_column(connection, "transformation_families", "polarity", "TEXT")
        _ensure_column(connection, "transformation_families", "member_count", "INTEGER")
        _ensure_column(connection, "transformation_families", "first_seen_global_step", "INTEGER")
        _ensure_column(connection, "transformation_families", "last_seen_global_step", "INTEGER")
        _ensure_column(connection, "transformation_families", "stability_score", "REAL")
        _ensure_column(connection, "transformation_families", "prediction_lift", "REAL")
        _ensure_column(connection, "transformation_families", "prediction_accuracy_mean", "REAL")
        _ensure_column(connection, "transformation_families", "prediction_error_before_mean", "REAL")
        _ensure_column(connection, "transformation_families", "prediction_error_after_mean", "REAL")
        _ensure_column(connection, "family_members", "family_signature", "TEXT")
        _ensure_column(connection, "family_members", "contingency_key", "TEXT")
        _ensure_column(connection, "family_members", "first_seen_global_step", "INTEGER")
        _ensure_column(connection, "family_members", "last_seen_global_step", "INTEGER")
        _ensure_column(connection, "role_neighborhood_signatures", "diagnostic_token_json", "TEXT")
        _ensure_column(connection, "role_candidates", "role_signature_token_count", "INTEGER")
        _ensure_column(connection, "role_candidates", "diagnostic_token_count", "INTEGER")
        _ensure_column(connection, "role_candidates", "exact_family_token_count", "INTEGER")
        _ensure_column(connection, "role_candidates", "exact_identity_token_count", "INTEGER")
        _ensure_column(connection, "role_candidates", "role_explanatory_coverage", "REAL")
        _ensure_column(connection, "role_candidates", "role_compression_gain", "REAL")
        _ensure_column(connection, "role_candidates", "role_prediction_lift", "REAL")
        _ensure_column(connection, "role_candidates", "role_future_option_lift", "REAL")
        _ensure_column(connection, "role_candidates", "role_transfer_lift", "REAL")
        _ensure_column(connection, "role_candidates", "promotion_status", "TEXT")
        _ensure_column(connection, "role_candidates", "promotion_failure_count", "INTEGER DEFAULT 0")
        _ensure_column(connection, "role_transfer_attempts", "best_margin", "REAL")
        _ensure_column(connection, "role_transfer_attempts", "source_carrier_count", "INTEGER")
        _ensure_column(connection, "role_transfer_attempts", "candidate_role_count", "INTEGER")
        _ensure_column(connection, "role_transfer_attempts", "source_game_key", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "target_game_key", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "source_context_key", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "target_context_key", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "source_carrier_signature", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "source_role_signature", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "predicted_target_role_signature", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "observed_target_role_signature", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "source_carrier_signatures_json", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "source_game_keys_json", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "source_context_keys_json", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "provenance_mode", "TEXT")
        _ensure_column(connection, "role_transfer_attempts", "provenance_status", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "source_role_signature", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "source_game_key", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "target_game_key", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "source_context_key", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "target_context_key", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "provenance_mode", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "motif_provenance_status", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "transfer_provenance_status", "TEXT")
        _ensure_column(connection, "future_option_transfer_links", "concept_validation_status", "TEXT")
        _ensure_column(connection, "concept_candidates", "strong_transfer_success_count", "INTEGER")
        _ensure_column(connection, "concept_candidates", "transfer_success_concentration", "REAL")
        _ensure_column(connection, "concept_candidates", "is_overconcentrated", "INTEGER DEFAULT 0")
        _ensure_column(connection, "concept_candidates", "concept_incremental_coverage", "REAL")
        _ensure_column(connection, "concept_candidates", "concept_incremental_compression_gain", "REAL")
        _ensure_column(connection, "concept_candidates", "concept_prediction_lift", "REAL")
        _ensure_column(connection, "concept_candidates", "concept_future_option_prediction_lift", "REAL")
        _ensure_column(connection, "concept_candidates", "concept_cross_game_transfer_lift", "REAL")
        _ensure_column(connection, "concept_candidates", "validation_scope", "TEXT")
        _ensure_column(connection, "concept_candidates", "validation_prediction_lift", "REAL")
        _ensure_column(connection, "concept_candidates", "validation_action_selection_lift", "REAL")
        _ensure_column(connection, "concept_candidates", "validation_transfer_lift", "REAL")
        _ensure_column(connection, "concept_candidates", "validation_evidence_count", "INTEGER")
        _ensure_column(connection, "concept_candidates", "promotion_status", "TEXT")
        _ensure_column(connection, "concept_candidates", "promotion_failure_count", "INTEGER DEFAULT 0")
        _ensure_column(connection, "promotion_validation_state", "last_validation_epoch", "TEXT")
        _ensure_column(connection, "promotion_validation_state", "last_validation_global_step", "INTEGER")
        _ensure_column(connection, "promotion_validation_state", "last_validation_result", "TEXT")
        _ensure_column(connection, "world_model_components", "candidate_only", "INTEGER DEFAULT 0")
        _ensure_column(connection, "world_model_components", "predicted_outcome_count", "INTEGER")
        _ensure_column(connection, "world_model_components", "predicted_outcome_count_is_proxy", "INTEGER DEFAULT 0")
        _ensure_column(connection, "world_model_components", "validation_prediction_lift", "REAL")
        _ensure_column(connection, "world_model_components", "validation_action_selection_lift", "REAL")
        _ensure_column(connection, "world_model_components", "validation_transfer_lift", "REAL")
        _ensure_column(connection, "world_model_components", "validation_contradiction_resolution", "REAL")
        _ensure_column(connection, "world_model_components", "validation_explanatory_gain", "REAL")
        _ensure_column(connection, "world_model_components", "promotion_status", "TEXT")
        _ensure_column(connection, "world_model_components", "promotion_failure_count", "INTEGER DEFAULT 0")
        _ensure_column(connection, "future_option_attention_links", "attention_signal_source", "TEXT")
        _ensure_column(connection, "future_option_attention_links", "raw_high_attention", "INTEGER")
        _ensure_column(connection, "future_option_attention_links", "calibrated_high_attention", "INTEGER")
        _ensure_column(connection, "future_option_attention_links", "source_label", "TEXT")
        _ensure_column(connection, "future_option_attention_links", "attention_score", "REAL")
        _ensure_column(connection, "future_option_attention_links", "attention_score_percentile", "REAL")
        _ensure_column(connection, "future_option_attention_links", "attention_threshold_method", "TEXT")
        _ensure_column(connection, "future_option_attention_links", "attention_calibration_degenerate", "INTEGER")
        _ensure_column(connection, "future_option_events", "source_interaction_id", "TEXT")
        _ensure_column(connection, "future_option_events", "source_family_id", "TEXT")
        _ensure_column(connection, "future_option_events", "source_carrier_id", "TEXT")
        _ensure_column(connection, "future_option_events", "source_role_id", "TEXT")
        _ensure_column(connection, "future_option_events", "source_concept_id", "TEXT")
        _ensure_column(connection, "future_option_events", "source_context_signature", "TEXT")
        _ensure_column(connection, "future_option_events", "source_action", "TEXT")
        _ensure_column(connection, "future_option_events", "source_game_id", "TEXT")
        _ensure_column(connection, "future_option_events", "source_sampler", "TEXT")
        _ensure_column(connection, "future_option_events", "future_option_development_stage", "TEXT")
        _ensure_column(connection, "future_option_events", "survival_delta", "REAL")
        _ensure_column(connection, "future_option_events", "movement_freedom_delta", "REAL")
        _ensure_column(connection, "future_option_events", "environmental_influence_delta", "REAL")
        _ensure_column(connection, "future_option_events", "graph_expansion_delta", "REAL")
        _ensure_column(connection, "future_option_events", "role_discovery_delta", "REAL")
        _ensure_column(connection, "future_option_events", "concept_transfer_delta", "REAL")
        _ensure_column(connection, "future_option_events", "developmental_option_value", "REAL")
        _ensure_column(connection, "future_option_events", "motif_classification_reason", "TEXT")
        _ensure_column(connection, "future_option_events", "classification_source", "TEXT")
        _ensure_column(connection, "future_option_events", "classification_rule", "TEXT")
        _ensure_column(connection, "future_option_events", "classification_evidence_id", "TEXT")
        _ensure_column(connection, "future_option_motifs", "source_interaction_ids_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "source_family_ids_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "source_carrier_ids_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "source_role_ids_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "source_concept_ids_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "future_option_development_stage", "TEXT")
        _ensure_column(connection, "future_option_motifs", "development_component_means_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "motif_classification_reason", "TEXT")
        _ensure_column(connection, "future_option_motifs", "classification_source", "TEXT")
        _ensure_column(connection, "future_option_motifs", "classification_rule", "TEXT")
        _ensure_column(connection, "future_option_motifs", "classification_evidence_id", "TEXT")
        _ensure_column(connection, "future_option_motifs", "source_game_keys_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "target_game_keys_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "source_context_keys_json", "TEXT")
        _ensure_column(connection, "future_option_motifs", "target_context_keys_json", "TEXT")
        _ensure_column(connection, "memory_scores", "memory_state", "TEXT")
        _ensure_column(connection, "memory_scores", "stored_epoch", "INTEGER")
        _ensure_column(connection, "memory_scores", "last_replayed_epoch", "INTEGER")
        _ensure_column(connection, "memory_scores", "last_promoted_epoch", "INTEGER")
        _ensure_column(connection, "memory_scores", "retention_score", "REAL")
        _ensure_column(connection, "memory_scores", "forgetting_score", "REAL")
        _ensure_column(connection, "memory_scores", "compressed_into_id", "TEXT")
        _ensure_column(connection, "memory_scores", "superseded_by_id", "TEXT")
        _ensure_column(connection, "memory_scores", "forgetting_reason", "TEXT")
        if legacy_transfer_provenance:
            # Preserve old rows for audit, but never let the former
            # target-relative pseudo-scopes support verified transfer claims.
            connection.execute(
                """
                UPDATE role_transfer_attempts
                SET provenance_mode = 'legacy', provenance_status = 'legacy'
                WHERE provenance_mode IS NULL OR provenance_mode = ''
                """
            )
            # Invalidate all artifacts derived from the old attempts.  A fresh
            # higher-order pass replaces the retained audit rows transactionally.
            for table in (
                "concept_candidates",
                "concept_links",
                "world_model_components",
                "world_model_links",
                "concept_promotion_validation_diagnostics",
                "future_option_transfer_links",
            ):
                if table in existing_tables:
                    connection.execute(f"DELETE FROM {table}")
            connection.execute(
                """
                INSERT INTO memory_summary (key, value_json)
                VALUES ('role_transfer_provenance_migration', ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                """,
                (json.dumps({"rebuild_required": True, "schema": "concrete_source_scope_v2"}, sort_keys=True),),
            )
        connection.execute("DROP INDEX IF EXISTS idx_carrier_links_carrier_type_key")
        connection.execute("DROP INDEX IF EXISTS idx_role_neighborhood_carrier")
        connection.commit()


def _ensure_graph_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        configure_compact_sqlite_connection(connection, write=True)
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
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source_node
            ON graph_edges(source_node_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target_node
            ON graph_edges(target_node_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_type_target
            ON graph_edges(edge_type, target_node_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_type_source
            ON graph_edges(edge_type, source_node_id);
            """
        )
        connection.execute("DROP INDEX IF EXISTS idx_graph_edges_source")
        connection.execute("DROP INDEX IF EXISTS idx_graph_edges_target")
        connection.commit()


def _ensure_table_column(connection: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if any(str(row[1]) == column for row in rows):
        return
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _ensure_replay_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        configure_compact_sqlite_connection(connection, write=True)
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


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing_columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing_columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
