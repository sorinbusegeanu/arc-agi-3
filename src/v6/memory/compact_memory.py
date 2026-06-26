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
    live_graph_paths = sorted(raw_dir.rglob("live_graph_compact.json"))
    report = _load_json(raw_dir / "interaction_sampling_v05c_report.json") or {}
    temporal_rows = list(((report.get("temporal_milestones") or {}).get("by_game_sampler_seed") or []))
    current_summary = load_memory_summary(paths.summary_json)
    totals = {
        "stable_contingencies_added": 0,
        "transformation_families_added": 0,
        "carrier_candidates_added": 0,
        "contradiction_clusters_added": 0,
        "replay_queue_size": 0,
        "representative_examples_retained": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "graph_live_exports_ingested": 0,
        "db_files_folded": len(db_paths),
        "total_interactions_seen": int(current_summary.get("total_interactions_seen", 0) or 0)
        + int((report.get("validation") or {}).get("memory_record_count", 0) or 0),
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
            _ingest_live_graph_export(graph_conn, graph_path, fold_config)
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
        _write_memory_summary_table(state_conn, summary)
        paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_conn.commit()
        graph_conn.commit()
        replay_conn.commit()
    return totals


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
                )
            for graph_path in live_graph_paths:
                _ingest_live_graph_export(graph_conn, graph_path, fold_config)
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
                "transformation_families_added",
                "carrier_candidates_added",
                "contradiction_clusters_added",
            ):
                totals[key] = sum(int(item.get(key, 0) or 0) for item in chunk_totals)
            totals["replay_queue_size"] = int(summary.get("replay_queue_size", 0) or 0)
            totals["representative_examples_retained"] = int(summary.get("representative_example_count", 0) or 0)
            totals["graph_node_count"] = int(summary.get("graph_node_count", 0) or 0)
            totals["graph_edge_count"] = int(summary.get("graph_edge_count", 0) or 0)
            totals["db_files_folded"] = len(db_paths)
            summary.update(totals)
            summary["fold_summary"] = dict(totals)
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
        "transformation_families_added": 0,
        "carrier_candidates_added": 0,
        "contradiction_clusters_added": 0,
        "replay_queue_size": 0,
        "representative_examples_retained": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "graph_live_exports_ingested": 0,
        "db_files_folded": len(db_paths),
        "total_interactions_seen": 0,
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
                    mean_replay_priority, representative_example_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_key) DO UPDATE SET
                    support_count = MAX(stable_contingencies.support_count, excluded.support_count),
                    context_level = MAX(stable_contingencies.context_level, excluded.context_level),
                    last_seen_global_step = MAX(stable_contingencies.last_seen_global_step, excluded.last_seen_global_step),
                    stability_score = MAX(stable_contingencies.stability_score, excluded.stability_score)
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
                    support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_signature) DO UPDATE SET
                    support_count = MAX(transformation_families.support_count, excluded.support_count),
                    member_count = MAX(transformation_families.member_count, excluded.member_count),
                    last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
                    stability_score = MAX(transformation_families.stability_score, excluded.stability_score)
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
                ),
            )
        for replay in getattr(system.memory_lifecycle, "replay_candidates", {}).values():
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
                    json.dumps(replay.to_dict(), sort_keys=True),
                ),
            )
        for candidate in getattr(system.carrier_tracker, "build_candidates", lambda: [])():
            state_conn.execute(
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
                    candidate.carrier_id,
                    candidate.carrier_signature,
                    candidate.carrier_source,
                    int(candidate.support_count),
                    int(candidate.distinct_family_count),
                    global_step_start,
                    global_step_end,
                    float(candidate.prediction_lift),
                    int(candidate.status == "emergent_carrier" and candidate.carrier_source != "context_action_fallback"),
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
            )
            _upsert_carrier_link(
                state_conn,
                carrier_signature=str(candidate.carrier_signature),
                linked_type="context",
                linked_key=None if candidate.context_signature is None else _normalize_jsonish(candidate.context_signature),
                fold_config=CompactMemoryFoldConfig(global_step_start=global_step_start, global_step_end=global_step_end),
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


def canonical_family_signature_from_family(family: Any) -> str:
    centroid = getattr(family, "centroid_vector", None)
    if centroid is not None:
        return "centroid:" + json.dumps([round(float(value), 6) for value in np.asarray(centroid, dtype=float).tolist()], separators=(",", ":"))
    return f"family:{getattr(family, 'id', 'unknown')}"


def canonical_family_signature_from_raw_db(raw_conn: sqlite3.Connection, family_id: Any, row_payload: dict[str, Any] | None) -> str:
    payload = dict(row_payload or {})
    transformation_tables = {row[0] for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "transformation_families" in transformation_tables and family_id is not None:
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
            return f"{key}:{_normalize_jsonish(value)}"
    db_identity = str(raw_conn.execute("PRAGMA database_list").fetchone()[2])
    return f"local_family:{db_identity}:{family_id}"


def stable_family_int_id(canonical_signature: str) -> int:
    return int.from_bytes(sha1(str(canonical_signature).encode("utf-8")).digest()[:8], "big") % 2_147_483_647


def _merge_compact_memory_dir_into_main(
    *,
    temp_dir: Path,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    replay_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
) -> None:
    temp_paths = ensure_memory_layout(temp_dir)
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


def _merge_state_tables(temp_state: sqlite3.Connection, state_conn: sqlite3.Connection, fold_config: CompactMemoryFoldConfig) -> None:
    for row in temp_state.execute("SELECT * FROM family_identity_map ORDER BY canonical_signature ASC").fetchall():
        _upsert_family_identity_map(state_conn, str(row["canonical_signature"]), int(row["stable_family_id"]))
    for row in temp_state.execute("SELECT * FROM stable_contingencies ORDER BY canonical_key ASC").fetchall():
        state_conn.execute(
            """
            INSERT INTO stable_contingencies (
                contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                mean_replay_priority, representative_example_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_key) DO UPDATE SET
                support_count = MAX(stable_contingencies.support_count, excluded.support_count),
                context_level = MAX(stable_contingencies.context_level, excluded.context_level),
                first_seen_global_step = MIN(stable_contingencies.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(stable_contingencies.last_seen_global_step, excluded.last_seen_global_step),
                stability_score = MAX(stable_contingencies.stability_score, excluded.stability_score),
                representative_example_count = MAX(stable_contingencies.representative_example_count, excluded.representative_example_count)
            """,
            tuple(row[column] for column in row.keys()),
        )
    for row in temp_state.execute("SELECT * FROM transformation_families ORDER BY canonical_signature ASC").fetchall():
        state_conn.execute(
            """
            INSERT INTO transformation_families (
                family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_signature) DO UPDATE SET
                support_count = transformation_families.support_count + excluded.support_count,
                member_count = transformation_families.member_count + excluded.member_count,
                first_seen_global_step = MIN(transformation_families.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
                stability_score = MAX(transformation_families.stability_score, excluded.stability_score)
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
                first_seen_global_step, last_seen_global_step, stability_score, is_emergent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(carrier_signature) DO UPDATE SET
                support_count = MAX(carrier_candidates.support_count, excluded.support_count),
                linked_family_count = MAX(carrier_candidates.linked_family_count, excluded.linked_family_count),
                first_seen_global_step = MIN(carrier_candidates.first_seen_global_step, excluded.first_seen_global_step),
                last_seen_global_step = MAX(carrier_candidates.last_seen_global_step, excluded.last_seen_global_step),
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


def _merge_graph_tables(temp_graph: sqlite3.Connection, graph_conn: sqlite3.Connection, fold_config: CompactMemoryFoldConfig) -> None:
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


def _fold_single_db(
    *,
    db_path: Path,
    state_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    replay_conn: sqlite3.Connection,
    fold_config: CompactMemoryFoldConfig,
    totals: dict[str, Any],
) -> None:
    with sqlite3.connect(db_path) as raw_conn:
        raw_conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        game = _path_segment(db_path, -4)
        sampler = _path_segment(db_path, -3)
        seed = _seed_from_db_path(db_path)
        stable_threshold = 20
        prediction_payload_by_interaction = _prediction_payload_by_interaction_id(raw_conn) if "prediction_results" in tables else {}

        family_members_by_signature: dict[str, set[str]] = {}
        if "contingencies" in tables:
            for row in raw_conn.execute("SELECT * FROM contingencies").fetchall():
                payload = dict(row)
                family_signature = canonical_family_signature_from_raw_db(raw_conn, payload.get("transformation_family"), payload)
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
                    ),
                    action=int(payload.get("action") or 0),
                    effect_signature=family_signature,
                    support_count=support_count,
                    fold_config=fold_config,
                    stable_threshold=stable_threshold,
                )
                if support_count >= stable_threshold:
                    totals["stable_contingencies_added"] += 1
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
                )
        if "prediction_results" in tables:
            family_supports: dict[str, dict[str, Any]] = {}
            contradiction_supports: dict[str, dict[str, Any]] = {}
            for row in raw_conn.execute("SELECT * FROM prediction_results").fetchall():
                payload = dict(row)
                family_signature = canonical_family_signature_from_raw_db(raw_conn, payload.get("actual_family") or payload.get("predicted_family"), payload)
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
                    },
                )
                info["member_count"] += 1
                info["support_count"] += 1
                global_step = int(payload.get("global_step") or payload.get("interaction_id") or fold_config.global_step_end)
                info["first_step"] = global_step if info["first_step"] is None else min(int(info["first_step"]), global_step)
                info["last_step"] = global_step if info["last_step"] is None else max(int(info["last_step"]), global_step)
                info["mean_error_total"] += float(payload.get("isf_prediction_error") or payload.get("prediction_error") or 0.0)
                info["mean_replay_total"] += float(payload.get("memory_replay_priority") or payload.get("replay_priority") or 0.0)
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
                )
            for family_signature, info in family_supports.items():
                member_count = int(info["member_count"])
                _upsert_transformation_family(
                    state_conn,
                    family_signature=family_signature,
                    support_count=int(info["support_count"]),
                    member_count=member_count,
                    first_seen=int(info["first_step"] or fold_config.global_step_start),
                    last_seen=int(info["last_step"] or fold_config.global_step_end),
                    stability_score=float(member_count),
                )
                totals["transformation_families_added"] += 1
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
        if "interactions" in tables:
            for row in raw_conn.execute("SELECT * FROM interactions WHERE COALESCE(memory_replay_candidate, 0) = 1 OR COALESCE(memory_replay_priority, 0.0) > 0.0").fetchall():
                payload = dict(row)
                replay_id = str(payload.get("id"))
                priority = float(payload.get("memory_replay_priority") or 0.0)
                reason = "carrier_linked" if payload.get("carrier_signature") else "priority"
                prediction_payload = dict(prediction_payload_by_interaction.get(replay_id, {}))
                merged_payload = {**prediction_payload, **payload}
                family_signature = None
                family_id = prediction_payload.get("actual_family") or prediction_payload.get("predicted_family")
                if family_id not in (None, ""):
                    family_signature = canonical_family_signature_from_raw_db(raw_conn, family_id, prediction_payload)
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
                        json.dumps(_json_safe(merged_payload), sort_keys=True),
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
                )
        carrier_path = db_path.with_name("carrier_candidates.json")
        if carrier_path.exists():
            for item in json.loads(carrier_path.read_text(encoding="utf-8")):
                carrier_signature = str(item.get("carrier_signature") or item.get("carrier_id") or "")
                if not carrier_signature:
                    continue
                carrier_source = str(item.get("carrier_source") or "unknown")
                is_emergent = str(item.get("status") or "") == "emergent_carrier" and carrier_source != "context_action_fallback"
                state_conn.execute(
                    """
                    INSERT INTO carrier_candidates (
                        carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                        first_seen_global_step, last_seen_global_step, stability_score, is_emergent
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(carrier_signature) DO UPDATE SET
                        support_count = MAX(carrier_candidates.support_count, excluded.support_count),
                        linked_family_count = MAX(carrier_candidates.linked_family_count, excluded.linked_family_count),
                        first_seen_global_step = MIN(carrier_candidates.first_seen_global_step, excluded.first_seen_global_step),
                        last_seen_global_step = MAX(carrier_candidates.last_seen_global_step, excluded.last_seen_global_step),
                        stability_score = MAX(carrier_candidates.stability_score, excluded.stability_score),
                        is_emergent = MAX(carrier_candidates.is_emergent, excluded.is_emergent)
                    """,
                    (
                        str(item.get("carrier_id") or carrier_signature),
                        carrier_signature,
                        carrier_source,
                        int(item.get("support_count", 0) or 0),
                        int(item.get("distinct_family_count", 0) or item.get("linked_family_count", 0) or 0),
                        fold_config.global_step_start,
                        fold_config.global_step_end,
                        float(item.get("prediction_lift", 0.0) or item.get("stability_score", 0.0) or 0.0),
                        int(is_emergent),
                    ),
                )
                totals["carrier_candidates_added"] += 1
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
                    global_step=fold_config.global_step_end,
                    priority_score=float(item.get("prediction_lift", 0.0) or 0.0) + float(item.get("support_count", 0) or 0),
                    compact_payload_json=json.dumps(item, sort_keys=True),
                )
                _upsert_carrier_link(
                    state_conn,
                    carrier_signature=carrier_signature,
                    linked_type="family",
                    linked_key=family_signature,
                    fold_config=fold_config,
                )
                _upsert_carrier_link(
                    state_conn,
                    carrier_signature=carrier_signature,
                    linked_type="context",
                    linked_key=None if item.get("context_signature") is None else _normalize_jsonish(item.get("context_signature")),
                    fold_config=fold_config,
                )
                _upsert_carrier_link(
                    state_conn,
                    carrier_signature=carrier_signature,
                    linked_type="contingency",
                    linked_key=contingency_key,
                    fold_config=fold_config,
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
    fold_config: CompactMemoryFoldConfig,
    stable_threshold: int,
) -> None:
    connection.execute(
        """
        INSERT INTO stable_contingencies (
            contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
            first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
            mean_replay_priority, representative_example_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_key) DO UPDATE SET
            support_count = MAX(stable_contingencies.support_count, excluded.support_count),
            context_level = MAX(stable_contingencies.context_level, excluded.context_level),
            first_seen_global_step = MIN(stable_contingencies.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX(stable_contingencies.last_seen_global_step, excluded.last_seen_global_step),
            stability_score = MAX(stable_contingencies.stability_score, excluded.stability_score)
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
        ),
    )


def _upsert_carrier_link(
    connection: sqlite3.Connection,
    *,
    carrier_signature: str,
    linked_type: str,
    linked_key: str | None,
    fold_config: CompactMemoryFoldConfig,
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
        (str(carrier_signature), str(linked_type), str(linked_key), 1, fold_config.global_step_start, fold_config.global_step_end),
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


def _context_level_from_raw(raw_conn: sqlite3.Connection, *, payload: dict[str, Any], family_id: Any) -> int:
    if payload.get("context_level") not in (None, ""):
        return int(payload.get("context_level") or 0)
    tables = {row[0] for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "prediction_results" not in tables:
        return 0
    prediction_columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(prediction_results)").fetchall()}
    if "context_level" not in prediction_columns:
        return 0
    row = raw_conn.execute(
        """
        SELECT MAX(COALESCE(context_level, 0))
        FROM prediction_results
        WHERE COALESCE(context_signature, '') = COALESCE(?, '')
          AND COALESCE(action, -1) = COALESCE(?, -1)
          AND (
                COALESCE(actual_family, predicted_family) = ?
             OR COALESCE(actual_family, predicted_family) = ?
          )
        """,
        (
            payload.get("context_signature") or payload.get("context_action"),
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
) -> None:
    stable_id = stable_family_int_id(family_signature)
    connection.execute(
        """
        INSERT INTO transformation_families (
            family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
            support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_signature) DO UPDATE SET
            support_count = transformation_families.support_count + excluded.support_count,
            member_count = transformation_families.member_count + excluded.member_count,
            first_seen_global_step = MIN(transformation_families.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX(transformation_families.last_seen_global_step, excluded.last_seen_global_step),
            stability_score = MAX(transformation_families.stability_score, excluded.stability_score)
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
        ),
    )


def _ingest_live_graph_export(graph_conn: sqlite3.Connection, path: Path, fold_config: CompactMemoryFoldConfig) -> None:
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
) -> None:
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
        _upsert_graph_edge(graph_conn, source_node_id=source, target_node_id=target, edge_type=edge_type, fold_config=fold_config)


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
) -> None:
    edge_id = f"{source_node_id}->{edge_type}->{target_node_id}"
    graph_conn.execute(
        """
        INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(edge_id) DO UPDATE SET
            first_seen_global_step = MIN(graph_edges.first_seen_global_step, excluded.first_seen_global_step),
            last_seen_global_step = MAX(graph_edges.last_seen_global_step, excluded.last_seen_global_step),
            support_count = MAX(graph_edges.support_count, excluded.support_count),
            weight = MAX(graph_edges.weight, excluded.weight)
        """,
        (edge_id, source_node_id, target_node_id, edge_type, fold_config.global_step_start, fold_config.global_step_end, int(support_count), float(weight)),
    )


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
    family_count = _count_rows(state_conn, "transformation_families")
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
    contradiction_count = _count_rows(state_conn, "contradiction_clusters")
    example_count = _count_rows(state_conn, "representative_examples")
    emergent_carrier_count = int(
        state_conn.execute("SELECT COUNT(*) FROM carrier_candidates WHERE COALESCE(is_emergent, 0) = 1").fetchone()[0]
    )
    summary = {
        "stable_contingency_count": stable_count,
        "transformation_family_count": family_count,
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
        "contradiction_cluster_count": contradiction_count,
        "representative_example_count": example_count,
        "graph_node_count": _count_rows(graph_conn, "graph_nodes"),
        "graph_edge_count": _count_rows(graph_conn, "graph_edges"),
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
        "graph_node_count",
        "graph_edge_count",
        "replay_queue_size",
    ):
        summary[key] = int(summary[key] or 0)
    return summary


def _count_rows(connection: sqlite3.Connection, table: str) -> int:
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.DatabaseError:
        return 0


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


def _ensure_current_state_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
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
                representative_example_count INTEGER
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
                stability_score REAL
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
                exact_identity_token_count INTEGER
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
                is_promoted INTEGER
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
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                is_coherent INTEGER
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
                is_emergent INTEGER
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
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER
            );
            CREATE TABLE IF NOT EXISTS future_option_transfer_links (
                motif_signature TEXT,
                role_signature TEXT,
                concept_signature TEXT,
                transfer_attempt_count INTEGER,
                successful_transfer_count INTEGER,
                strong_transfer_success_count INTEGER,
                promoted_concept_count INTEGER,
                mean_transfer_score REAL,
                mean_best_margin REAL,
                first_seen_global_step INTEGER,
                last_seen_global_step INTEGER,
                PRIMARY KEY (motif_signature, role_signature, concept_signature)
            );
            CREATE INDEX IF NOT EXISTS idx_role_links_type_key
            ON role_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_role_transfer_role
            ON role_transfer_attempts(role_signature);
            CREATE INDEX IF NOT EXISTS idx_role_transfer_success
            ON role_transfer_attempts(reuse_success, transfer_kind);
            CREATE INDEX IF NOT EXISTS idx_concept_links_type_key
            ON concept_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_world_model_links_type_key
            ON world_model_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_future_option_events_owner
            ON future_option_events(owner_type, owner_key);
            CREATE INDEX IF NOT EXISTS idx_future_option_events_motif
            ON future_option_events(motif_type, option_delta_bucket);
            CREATE INDEX IF NOT EXISTS idx_future_option_motifs_emergent
            ON future_option_motifs(is_emergent, motif_type);
            CREATE INDEX IF NOT EXISTS idx_future_option_links_type_key
            ON future_option_links(linked_type, linked_key);
            CREATE INDEX IF NOT EXISTS idx_future_option_attention_flags
            ON future_option_attention_links(high_option_change, high_attention);
            CREATE INDEX IF NOT EXISTS idx_future_option_transfer_motif
            ON future_option_transfer_links(motif_signature);
            """
        )
        _ensure_column(connection, "stable_contingencies", "context_level", "INTEGER DEFAULT 0")
        _ensure_column(connection, "transformation_families", "canonical_signature", "TEXT")
        _ensure_column(connection, "transformation_families", "relaxed_signature", "TEXT")
        _ensure_column(connection, "transformation_families", "effect_type", "TEXT")
        _ensure_column(connection, "transformation_families", "action_group", "TEXT")
        _ensure_column(connection, "transformation_families", "polarity", "TEXT")
        _ensure_column(connection, "transformation_families", "member_count", "INTEGER")
        _ensure_column(connection, "transformation_families", "first_seen_global_step", "INTEGER")
        _ensure_column(connection, "transformation_families", "last_seen_global_step", "INTEGER")
        _ensure_column(connection, "transformation_families", "stability_score", "REAL")
        _ensure_column(connection, "family_members", "family_signature", "TEXT")
        _ensure_column(connection, "family_members", "contingency_key", "TEXT")
        _ensure_column(connection, "family_members", "first_seen_global_step", "INTEGER")
        _ensure_column(connection, "family_members", "last_seen_global_step", "INTEGER")
        _ensure_column(connection, "role_neighborhood_signatures", "diagnostic_token_json", "TEXT")
        _ensure_column(connection, "role_candidates", "role_signature_token_count", "INTEGER")
        _ensure_column(connection, "role_candidates", "diagnostic_token_count", "INTEGER")
        _ensure_column(connection, "role_candidates", "exact_family_token_count", "INTEGER")
        _ensure_column(connection, "role_candidates", "exact_identity_token_count", "INTEGER")
        _ensure_column(connection, "role_transfer_attempts", "best_margin", "REAL")
        _ensure_column(connection, "role_transfer_attempts", "source_carrier_count", "INTEGER")
        _ensure_column(connection, "role_transfer_attempts", "candidate_role_count", "INTEGER")
        _ensure_column(connection, "concept_candidates", "strong_transfer_success_count", "INTEGER")
        _ensure_column(connection, "world_model_components", "candidate_only", "INTEGER DEFAULT 0")
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


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing_columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing_columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
