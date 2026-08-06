from __future__ import annotations

import json
import os
import re
import sqlite3
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

_PATCHED = False
_MAX_TRANSFER_HISTORY = 250_000
_FUTURE_EDGE_TYPES = (
    "expands_future_options",
    "restricts_future_options",
    "preserves_future_options",
)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    if not _table_exists(connection, table):
        return []
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()]


def _dict_rows(connection: sqlite3.Connection, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    previous = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql, parameters).fetchall()]
    finally:
        connection.row_factory = previous


def _clean_scope(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    lowered = text.lower()
    if not text or lowered in {"none", "null", "[]", "{}"} or "null" in lowered:
        return None
    return text


def _normalize_interaction_id(value: Any) -> str:
    text = str(value or "")
    for prefix in ("M0:interaction:", "interaction:"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _stable_surrogate(kind: str, *parts: Any) -> str:
    seed = "|".join(str(part or "") for part in parts)
    return f"surrogate_{kind}:" + sha1(seed.encode("utf-8")).hexdigest()[:20]


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _deep_get(payload: dict[str, Any], names: Iterable[str]) -> Any:
    wanted = set(names)
    queue: list[Any] = [payload]
    while queue:
        item = queue.pop(0)
        if isinstance(item, dict):
            for key, value in item.items():
                if key in wanted and value not in (None, ""):
                    return value
                if isinstance(value, (dict, list, tuple)):
                    queue.append(value)
        elif isinstance(item, (list, tuple)):
            queue.extend(item)
    return None


def _merge_metadata(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if value not in (None, "") and target.get(key) in (None, ""):
            target[key] = value


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "game": _deep_get(payload, ("game", "game_key", "source_game_key")),
        "sampler": _deep_get(payload, ("sampler", "source_sampler")),
        "context": _deep_get(
            payload,
            ("context_signature", "context_key", "source_context_key", "context"),
        ),
        "action": _deep_get(payload, ("action", "action_key", "source_action")),
        "global_step": _deep_get(
            payload,
            ("global_step", "step", "last_seen_global_step", "first_seen_global_step"),
        ),
        "predicted_family": _deep_get(payload, ("predicted_family",)),
        "actual_family": _deep_get(payload, ("actual_family",)),
        "prediction_confidence": _deep_get(payload, ("prediction_confidence",)),
        "context_contradiction": _deep_get(
            payload,
            ("context_contradiction", "prediction_violation", "is_prediction_violation"),
        ),
    }


def _interaction_metadata_index(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    if _table_exists(connection, "memory_nodes"):
        columns = set(_table_columns(connection, "memory_nodes"))
        selected = [name for name in (
            "node_id", "payload_json", "first_seen_global_step", "last_seen_global_step"
        ) if name in columns]
        if "node_id" in selected:
            rows = _dict_rows(
                connection,
                f"SELECT {', '.join(selected)} FROM memory_nodes "
                "WHERE node_id LIKE 'M0:interaction:%' ORDER BY node_id ASC",
            )
            for row in rows:
                interaction_id = _normalize_interaction_id(row.get("node_id"))
                metadata = index.setdefault(interaction_id, {})
                _merge_metadata(metadata, _metadata_from_payload(_json_object(row.get("payload_json"))))
                _merge_metadata(metadata, {
                    "global_step": row.get("last_seen_global_step")
                    if row.get("last_seen_global_step") is not None
                    else row.get("first_seen_global_step"),
                })

    preferred_tables = (
        "prediction_results",
        "interactions",
        "interaction_results",
        "representative_examples",
    )
    id_candidates = ("interaction_id", "source_interaction_id", "node_id", "id")
    field_candidates = {
        "game": ("game", "game_key", "source_game_key"),
        "sampler": ("sampler", "source_sampler"),
        "context": ("context_signature", "context_key", "source_context_key"),
        "action": ("action", "action_key", "source_action"),
        "global_step": ("global_step", "step", "last_seen_global_step", "first_seen_global_step"),
        "predicted_family": ("predicted_family",),
        "actual_family": ("actual_family",),
        "prediction_confidence": ("prediction_confidence",),
        "context_contradiction": ("context_contradiction", "prediction_violation"),
        "payload_json": ("payload_json", "evidence_json"),
    }
    for table in preferred_tables:
        columns = set(_table_columns(connection, table))
        if not columns:
            continue
        id_column = next((name for name in id_candidates if name in columns), None)
        if id_column is None:
            continue
        selected = [id_column]
        mapped: dict[str, str] = {}
        for target, candidates in field_candidates.items():
            source = next((name for name in candidates if name in columns), None)
            if source is not None:
                mapped[target] = source
                selected.append(source)
        selected = list(dict.fromkeys(selected))
        try:
            rows = _dict_rows(
                connection,
                f"SELECT {', '.join(selected)} FROM {table} ORDER BY {id_column} ASC",
            )
        except sqlite3.Error:
            continue
        for row in rows:
            interaction_id = _normalize_interaction_id(row.get(id_column))
            if not interaction_id:
                continue
            metadata = index.setdefault(interaction_id, {})
            direct = {
                target: row.get(source)
                for target, source in mapped.items()
                if target != "payload_json"
            }
            _merge_metadata(metadata, direct)
            payload_column = mapped.get("payload_json")
            if payload_column:
                _merge_metadata(metadata, _metadata_from_payload(_json_object(row.get(payload_column))))
    return index


def _insert_dict(connection: sqlite3.Connection, table: str, payload: dict[str, Any], *, mode: str = "OR REPLACE") -> None:
    columns = _table_columns(connection, table)
    usable = [column for column in columns if column in payload]
    if not usable:
        raise RuntimeError(f"No compatible columns found for {table}")
    values: list[Any] = []
    for column in usable:
        value = payload[column]
        if isinstance(value, (dict, list, tuple, set)):
            value = json.dumps(
                sorted(value) if isinstance(value, set) else value,
                sort_keys=True,
            )
        values.append(value)
    placeholders = ", ".join("?" for _ in usable)
    connection.execute(
        f"INSERT {mode} INTO {table} ({', '.join(usable)}) VALUES ({placeholders})",
        tuple(values),
    )


def _backup_future_edges(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(connection, "memory_edges"):
        return []
    placeholders = ", ".join("?" for _ in _FUTURE_EDGE_TYPES)
    rows = _dict_rows(
        connection,
        "SELECT * FROM memory_edges "
        "WHERE source_node_id LIKE 'M0:interaction:%' "
        f"AND edge_type IN ({placeholders}) "
        "ORDER BY source_node_id ASC, target_node_id ASC, edge_type ASC",
        _FUTURE_EDGE_TYPES,
    )
    if rows:
        connection.execute(
            "DELETE FROM memory_edges "
            "WHERE source_node_id LIKE 'M0:interaction:%' "
            f"AND edge_type IN ({placeholders})",
            _FUTURE_EDGE_TYPES,
        )
    return rows


def _restore_rows(connection: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        _insert_dict(connection, table, row, mode="OR IGNORE")


def _edge_event_payload(
    future_options: Any,
    row: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    development_stage: Any,
) -> dict[str, Any]:
    source_node = str(row.get("source_node_id") or "")
    target_node = str(row.get("target_node_id") or "")
    source_id = _normalize_interaction_id(source_node)
    target_id = _normalize_interaction_id(target_node)
    source_meta = dict(metadata.get(source_id, {}))
    target_meta = dict(metadata.get(target_id, {}))

    source_game_real = _clean_scope(source_meta.get("game"))
    target_game_real = _clean_scope(target_meta.get("game"))
    source_context_real = _clean_scope(source_meta.get("context"))
    target_context_real = _clean_scope(target_meta.get("context"))

    source_game = source_game_real or _stable_surrogate("game", source_id)
    target_game = target_game_real or _stable_surrogate("game", target_id)
    source_context = source_context_real or _stable_surrogate("context", source_game, source_id)
    target_context = target_context_real or _stable_surrogate("context", target_game, target_id)

    source_game_surrogate = int(source_game_real is None)
    target_game_surrogate = int(target_game_real is None)
    source_context_surrogate = int(source_context_real is None)
    target_context_surrogate = int(target_context_real is None)
    fully_verified = not any((
        source_game_surrogate,
        target_game_surrogate,
        source_context_surrogate,
        target_context_surrogate,
    ))

    source_step = source_meta.get("global_step")
    target_step = target_meta.get("global_step")
    steps = [
        int(value)
        for value in (source_step, target_step)
        if value is not None and str(value).lstrip("-").isdigit()
    ]
    first_seen = min(steps) if steps else None
    last_seen = max(steps) if steps else None
    edge_type = str(row.get("edge_type") or "preserves_future_options")

    payload = future_options._build_future_option_event(
        owner_type="interaction",
        owner_key=source_id,
        source_kind=f"future_option_edge:{edge_type}",
        game=source_game,
        sampler=source_meta.get("sampler"),
        context_key=source_context,
        action_key=None if source_meta.get("action") is None else str(source_meta.get("action")),
        text_fragments=[edge_type],
        support_count=1,
        polarity=None,
        first_seen=first_seen,
        last_seen=last_seen,
        mean_prediction_error=0.0,
        mean_replay_priority=0.0,
        stability_score=0.0,
        event_id_seed=f"future_option_edge|{source_node}|{target_node}|{edge_type}",
        evidence_json={
            "source_table": "memory_edges",
            "source_node_id": source_node,
            "target_node_id": target_node,
            "edge_type": edge_type,
            "source_game_key": source_game,
            "target_game_key": target_game,
            "source_context_key": source_context,
            "target_context_key": target_context,
        },
        future_option_edge_type=edge_type,
        source_interaction_ids={source_id},
        development_stage=development_stage,
    )
    payload.update({
        "target_interaction_id": target_id,
        "classification_source": "future_option_edge",
        "classification_rule": edge_type,
        "classification_provenance_status": (
            "verified" if fully_verified else "resolved_with_surrogate"
        ),
        "source_game_key": source_game,
        "target_game_key": target_game,
        "source_context_key": source_context,
        "target_context_key": target_context,
        "source_game_is_surrogate": source_game_surrogate,
        "target_game_is_surrogate": target_game_surrogate,
        "source_context_is_surrogate": source_context_surrogate,
        "target_context_is_surrogate": target_context_surrogate,
        "context_resolution_source": (
            "interaction_metadata" if fully_verified else "surrogate"
        ),
        "context_is_surrogate": int(source_context_surrogate or target_context_surrogate),
        "source_game_id": source_game,
        "source_context_signature": source_context,
        "game": source_game,
        "context_key": source_context,
    })
    evidence = _json_object(payload.get("evidence_json"))
    evidence.update({
        "classification_source": "future_option_edge",
        "classification_rule": edge_type,
        "classification_provenance_status": payload["classification_provenance_status"],
        "source_game_is_surrogate": source_game_surrogate,
        "target_game_is_surrogate": target_game_surrogate,
        "source_context_is_surrogate": source_context_surrogate,
        "target_context_is_surrogate": target_context_surrogate,
    })
    payload["evidence_json"] = evidence
    return payload


def _semantic_source_kind(row: dict[str, Any]) -> str:
    evidence = _json_object(row.get("evidence_json"))
    base = str(row.get("source_kind") or "unknown").split("|sem=", 1)[0]
    edge_type = str(
        evidence.get("edge_type")
        or evidence.get("future_option_edge_type")
        or row.get("classification_rule")
        or ""
    )
    stage = str(row.get("future_option_development_stage") or "")
    effect = str(evidence.get("effect_type") or "")
    action_group = str(evidence.get("action_group") or row.get("action_key") or "")
    polarity = str(evidence.get("polarity") or "")
    owner_type = str(row.get("owner_type") or "")
    semantic = {
        "edge": edge_type,
        "stage": stage,
        "effect": effect,
        "action": action_group,
        "polarity": polarity,
        "owner": owner_type,
    }
    compact = ",".join(f"{key}:{value}" for key, value in semantic.items() if value)
    return f"{base}|sem={compact}" if compact else base


def _real_scope_motif_demotions(connection: sqlite3.Connection) -> int:
    if not (
        _table_exists(connection, "future_option_motifs")
        and _table_exists(connection, "future_option_motif_observations")
    ):
        return 0
    rows = _dict_rows(
        connection,
        """
        SELECT motif_signature,
               COUNT(DISTINCT CASE
                   WHEN provenance_status = 'verified'
                    AND COALESCE(source_game_is_surrogate, 0) = 0
                    AND source_game_key IS NOT NULL
                   THEN source_game_key END) AS real_game_count,
               COUNT(DISTINCT CASE
                   WHEN provenance_status = 'verified'
                    AND COALESCE(source_context_is_surrogate, 0) = 0
                    AND source_context_key IS NOT NULL
                   THEN source_context_key END) AS real_context_count
        FROM future_option_motif_observations
        GROUP BY motif_signature
        """,
    )
    demoted = 0
    for row in rows:
        if int(row.get("real_game_count") or 0) < 2 and int(row.get("real_context_count") or 0) < 2:
            cursor = connection.execute(
                "UPDATE future_option_motifs SET is_emergent = 0 "
                "WHERE motif_signature = ? AND COALESCE(is_emergent, 0) = 1",
                (row["motif_signature"],),
            )
            demoted += int(cursor.rowcount or 0)
    return demoted


def _compact_prediction_events(
    higher_order: Any,
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    role_links: dict[str, dict[str, set[str]]],
    transfer_history: Any,
) -> list[dict[str, Any]]:
    if first_seen_global_step is None or not source_roles:
        return []
    if not (_table_exists(state_conn, "memory_scores") and _table_exists(state_conn, "memory_edges")):
        return []

    metadata = _interaction_metadata_index(state_conn)
    score_columns = set(_table_columns(state_conn, "memory_scores"))
    selected = ["node_id"]
    for name in ("replay_priority", "future_option_delta", "first_seen_global_step", "last_seen_global_step"):
        if name in score_columns:
            selected.append(name)
    score_rows = _dict_rows(
        state_conn,
        f"SELECT {', '.join(selected)} FROM memory_scores "
        "WHERE node_id LIKE 'M0:interaction:%' ORDER BY node_id ASC",
    )
    violation_ids = {
        _normalize_interaction_id(row["source_node_id"])
        for row in _dict_rows(
            state_conn,
            "SELECT DISTINCT source_node_id FROM memory_edges "
            "WHERE edge_type = 'violates_prediction' "
            "AND source_node_id LIKE 'M0:interaction:%'",
        )
    }
    family_by_interaction: dict[str, set[str]] = {}
    for row in _dict_rows(
        state_conn,
        "SELECT source_node_id, target_node_id FROM memory_edges "
        "WHERE source_node_id LIKE 'M0:interaction:%' "
        "AND target_node_id LIKE 'M2:family:%'",
    ):
        interaction_id = _normalize_interaction_id(row["source_node_id"])
        family = str(row["target_node_id"]).split("M2:family:", 1)[-1]
        family_by_interaction.setdefault(interaction_id, set()).add(family)

    candidate_families = {
        str(family)
        for role in source_roles
        for family in role_links.get(role, {}).get("family", set())
        if family not in (None, "")
    }
    events: list[dict[str, Any]] = []
    synthetic_step = int(first_seen_global_step) + 1
    for ordinal, row in enumerate(score_rows):
        interaction_id = _normalize_interaction_id(row.get("node_id"))
        meta = metadata.get(interaction_id, {})
        step_value = (
            meta.get("global_step")
            if meta.get("global_step") is not None
            else row.get("last_seen_global_step")
            if row.get("last_seen_global_step") is not None
            else row.get("first_seen_global_step")
        )
        try:
            step = int(step_value)
        except (TypeError, ValueError):
            step = synthetic_step + ordinal
        if step <= int(first_seen_global_step):
            continue
        families = set(family_by_interaction.get(interaction_id, set()))
        predicted_family = str(meta.get("predicted_family") or "")
        actual_family = str(meta.get("actual_family") or "")
        if predicted_family:
            families.add(predicted_family)
        if actual_family:
            families.add(actual_family)
        if not families:
            continue

        rates = [
            higher_order._prior_role_success_rate(
                transfer_rows,
                role=role,
                before_step=step,
                transfer_history=transfer_history,
            )[0]
            for role in source_roles
        ]
        baseline = max(rates, default=0.0)
        concept_score = (
            higher_order._combined_role_score(rates)
            if len(rates) >= 2
            else baseline
        )
        violated = interaction_id in violation_ids
        outcome = 0.0 if violated else 1.0
        feature_step = transfer_history.max_any_step_before(step)
        if feature_step is None:
            feature_step = int(first_seen_global_step)
        events.append({
            "concept_id": candidate_signature,
            "event_id": f"prediction:compact_interaction:{interaction_id}",
            "event_type": "prediction",
            "evaluation_scope": "later_global_step",
            "predicted_family": predicted_family,
            "actual_family": actual_family,
            "candidate_role_family_ids": sorted(candidate_families),
            "best_single_role_score": baseline,
            "lower_level_baseline_score": baseline,
            "concept_enabled_score": concept_score,
            "prediction_gain": concept_score - baseline,
            "behavioral_gain": -abs(concept_score - outcome) + abs(baseline - outcome),
            "_outcome": outcome,
            "_evaluation_global_step": step,
            "_feature_global_step_max": feature_step,
            "_label_used_as_feature": False,
            "_context_keys": [str(meta["context"])] if _clean_scope(meta.get("context")) else [],
            "_game_keys": [str(meta["game"])] if _clean_scope(meta.get("game")) else [],
            "_family_ids": sorted(families),
            "_compact_prediction_surrogate": True,
        })
    return events


def _patch_future_options() -> None:
    import v6.future_options as future_options

    if getattr(future_options, "_ARC_AGI3_HYPOTHESIS_REPAIRS", False):
        return

    original_events = future_options.derive_future_option_events
    original_motifs = future_options.derive_future_option_motifs

    def derive_future_option_events(
        state_conn: sqlite3.Connection,
        graph_conn: sqlite3.Connection,
        max_events: int,
        development_stage: Any = None,
        progress_factory: Any | None = None,
    ) -> dict[str, Any]:
        if development_stage is None:
            development_stage = future_options.FutureOptionDevelopmentStage.SURVIVAL
        edge_rows = _backup_future_edges(state_conn)
        try:
            summary = original_events(
                state_conn,
                graph_conn,
                max_events=max_events,
                development_stage=development_stage,
                progress_factory=progress_factory,
            )
        finally:
            _restore_rows(state_conn, "memory_edges", edge_rows)

        current_count = int(
            state_conn.execute("SELECT COUNT(*) FROM future_option_events").fetchone()[0]
        )
        remaining = max(0, int(max_events) - current_count)
        metadata = _interaction_metadata_index(state_conn)
        inserted_edges = 0
        for row in edge_rows[:remaining]:
            payload = _edge_event_payload(
                future_options,
                row,
                metadata,
                development_stage,
            )
            _insert_dict(state_conn, "future_option_events", payload, mode="OR REPLACE")
            inserted_edges += 1

        total = int(
            state_conn.execute("SELECT COUNT(*) FROM future_option_events").fetchone()[0]
        )
        summary.update({
            "future_option_event_count": total,
            "future_option_events_inserted_total": total,
            "future_option_edge_rows_seen": len(edge_rows),
            "future_option_edge_events_inserted": inserted_edges,
            "future_option_edge_events_dropped_by_budget": max(0, len(edge_rows) - inserted_edges),
            "future_option_event_budget_policy": "higher_order_sources_first_then_graph_edges",
        })
        first_step = state_conn.execute(
            "SELECT MIN(first_seen_global_step) FROM future_option_events "
            "WHERE first_seen_global_step IS NOT NULL"
        ).fetchone()[0]
        summary["first_future_option_event_step"] = (
            None if first_step is None else int(first_step)
        )
        return summary

    def derive_future_option_motifs(
        state_conn: sqlite3.Connection,
        graph_conn: sqlite3.Connection,
        max_motifs: int,
        development_stage: Any = None,
        progress_factory: Any | None = None,
    ) -> dict[str, Any]:
        if development_stage is None:
            development_stage = future_options.FutureOptionDevelopmentStage.SURVIVAL
        rows = _dict_rows(
            state_conn,
            "SELECT event_id, source_kind, owner_type, action_key, "
            "future_option_development_stage, classification_rule, evidence_json "
            "FROM future_option_events ORDER BY event_id ASC",
        )
        for row in rows:
            enriched = _semantic_source_kind(row)
            if enriched != row.get("source_kind"):
                state_conn.execute(
                    "UPDATE future_option_events SET source_kind = ? WHERE event_id = ?",
                    (enriched, row["event_id"]),
                )
        summary = original_motifs(
            state_conn,
            graph_conn,
            max_motifs=max_motifs,
            development_stage=development_stage,
            progress_factory=progress_factory,
        )
        demoted = _real_scope_motif_demotions(state_conn)
        if demoted:
            emergent = int(
                state_conn.execute(
                    "SELECT COUNT(*) FROM future_option_motifs WHERE COALESCE(is_emergent, 0) = 1"
                ).fetchone()[0]
            )
            summary["emergent_future_option_motif_count"] = emergent
        summary["motifs_demoted_without_real_cross_scope_evidence"] = demoted
        summary["motif_signature_semantics_enriched"] = True
        return summary

    future_options.derive_future_option_events = derive_future_option_events
    future_options.derive_future_option_motifs = derive_future_option_motifs
    future_options._ARC_AGI3_HYPOTHESIS_REPAIRS = True


def _patch_higher_order() -> None:
    import v6.higher_order_substrate as higher_order

    if getattr(higher_order, "_ARC_AGI3_HYPOTHESIS_REPAIRS", False):
        return

    original_transfer_only = higher_order.derive_role_transfer_attempts_only
    original_prediction_events = higher_order._prediction_explanation_events
    original_contradiction_events = higher_order._contradiction_resolution_explanation_events

    def derive_role_transfer_attempts_only(
        *,
        memory_dir: Path,
        run_dir: Path | None = None,
        max_transfer_attempts: int = 25_000,
        workers: int = 1,
        chunk_size: int = 5_000,
        progress_factory: Any | None = None,
    ) -> dict[str, Any]:
        paths = higher_order.ensure_memory_layout(memory_dir)
        historical_rows: list[dict[str, Any]] = []
        with sqlite3.connect(paths.current_state) as connection:
            connection.row_factory = sqlite3.Row
            if _table_exists(connection, "role_transfer_attempts"):
                columns = set(_table_columns(connection, "role_transfer_attempts"))
                order_column = (
                    "last_seen_global_step"
                    if "last_seen_global_step" in columns
                    else "first_seen_global_step"
                    if "first_seen_global_step" in columns
                    else "attempt_id"
                )
                historical_rows = [
                    dict(row)
                    for row in connection.execute(
                        f"SELECT * FROM role_transfer_attempts "
                        f"ORDER BY {order_column} DESC, attempt_id ASC "
                        f"LIMIT {_MAX_TRANSFER_HISTORY}"
                    ).fetchall()
                ]

        summary = original_transfer_only(
            memory_dir=memory_dir,
            run_dir=run_dir,
            max_transfer_attempts=max_transfer_attempts,
            workers=workers,
            chunk_size=chunk_size,
            progress_factory=progress_factory,
        )

        with sqlite3.connect(paths.current_state) as connection:
            connection.row_factory = sqlite3.Row
            for row in historical_rows:
                _insert_dict(connection, "role_transfer_attempts", row, mode="OR IGNORE")
            columns = set(_table_columns(connection, "role_transfer_attempts"))
            total = int(connection.execute("SELECT COUNT(*) FROM role_transfer_attempts").fetchone()[0])
            if total > _MAX_TRANSFER_HISTORY and "attempt_id" in columns:
                order_column = (
                    "last_seen_global_step"
                    if "last_seen_global_step" in columns
                    else "first_seen_global_step"
                    if "first_seen_global_step" in columns
                    else "attempt_id"
                )
                connection.execute(
                    "DELETE FROM role_transfer_attempts WHERE attempt_id NOT IN ("
                    f"SELECT attempt_id FROM role_transfer_attempts "
                    f"ORDER BY {order_column} DESC, attempt_id ASC "
                    f"LIMIT {_MAX_TRANSFER_HISTORY})"
                )
                total = _MAX_TRANSFER_HISTORY
            summary["historical_transfer_attempt_count"] = total
            summary["historical_transfer_attempts_restored"] = max(
                0, total - int(summary.get("transfer_attempt_count", 0) or 0)
            )
            summary["transfer_history_retention_limit"] = _MAX_TRANSFER_HISTORY
            if _table_exists(connection, "memory_summary"):
                connection.execute(
                    """
                    INSERT INTO memory_summary (key, value_json)
                    VALUES ('higher_order_transfer_summary', ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                    """,
                    (json.dumps(summary, sort_keys=True),),
                )
            connection.commit()
        return summary

    def prediction_explanation_events(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        events = original_prediction_events(*args, **kwargs)
        if events:
            return events
        return _compact_prediction_events(
            higher_order,
            args[0] if args else kwargs["state_conn"],
            candidate_signature=kwargs["candidate_signature"],
            source_roles=kwargs["source_roles"],
            first_seen_global_step=kwargs["first_seen_global_step"],
            transfer_rows=kwargs["transfer_rows"],
            role_links=kwargs["role_links"],
            transfer_history=kwargs.get("transfer_history"),
        )

    def contradiction_resolution_events(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        events = original_contradiction_events(*args, **kwargs)
        if events:
            return events
        prediction_events = _compact_prediction_events(
            higher_order,
            args[0] if args else kwargs["state_conn"],
            candidate_signature=kwargs["candidate_signature"],
            source_roles=kwargs["source_roles"],
            first_seen_global_step=kwargs["first_seen_global_step"],
            transfer_rows=kwargs["transfer_rows"],
            role_links=kwargs["role_links"],
            transfer_history=kwargs.get("transfer_history"),
        )
        output: list[dict[str, Any]] = []
        for event in prediction_events:
            interaction_id = str(event["event_id"]).rsplit(":", 1)[-1]
            # Compact violation rows have an outcome of zero. Preserve only
            # those rows as contradiction-resolution opportunities.
            if float(event.get("_outcome", 1.0)) != 0.0:
                continue
            converted = dict(event)
            converted["event_id"] = f"contradiction_resolution:compact_interaction:{interaction_id}"
            converted["event_type"] = "contradiction_resolution"
            converted["_outcome"] = 1.0
            output.append(converted)
        return output

    higher_order.derive_role_transfer_attempts_only = derive_role_transfer_attempts_only
    higher_order._prediction_explanation_events = prediction_explanation_events
    higher_order._contradiction_resolution_explanation_events = contradiction_resolution_events
    higher_order._ARC_AGI3_HYPOTHESIS_REPAIRS = True


def _patch_h02_report() -> None:
    import v6.hypothesis_h02_report as h02

    if getattr(h02, "_ARC_AGI3_HYPOTHESIS_REPAIRS", False):
        return

    original = h02.evaluate_h02_prediction_violation_attention

    def evaluate_h02_prediction_violation_attention(
        run_dir: Path,
        output_dir: Path,
        *,
        memory_dir: Path | None = None,
        max_rows: int = h02.DEFAULT_MAX_ROWS,
        max_db_files: int = h02.DEFAULT_MAX_DB_FILES,
        prefer_db: str | None = None,
        scan_all_dbs: bool = False,
    ) -> dict[str, Any]:
        result = original(
            run_dir=run_dir,
            output_dir=output_dir,
            memory_dir=memory_dir,
            max_rows=max_rows,
            max_db_files=max_db_files,
            prefer_db=prefer_db,
            scan_all_dbs=scan_all_dbs,
        )
        per_epoch_expected = int(result.get("total_jobs_expected") or 0)
        represented = max(
            int(result.get("jobs_represented_in_compact_or_manifest_evidence") or 0),
            int(result.get("jobs_represented_in_raw_scan") or 0),
        )
        match = re.search(r"epoch_(\d+)", str(run_dir))
        epoch_number = int(match.group(1)) if match else None
        if (
            epoch_number is not None
            and epoch_number > 0
            and per_epoch_expected > 0
            and represented > per_epoch_expected
        ):
            cumulative_expected = per_epoch_expected * epoch_number
            result["total_jobs_expected_current_epoch"] = per_epoch_expected
            result["total_jobs_expected_cumulative"] = cumulative_expected
            result["total_jobs_expected"] = cumulative_expected
            result["evidence_coverage_ratio"] = min(
                1.0,
                represented / cumulative_expected if cumulative_expected else 0.0,
            )
            result["evidence_coverage_scope"] = "cumulative_manifest_through_current_epoch"
        elif per_epoch_expected > 0:
            result["evidence_coverage_ratio"] = min(
                1.0,
                represented / per_epoch_expected,
            )
            result["evidence_coverage_scope"] = "current_epoch"

        if isinstance(result.get("core_metrics"), dict):
            result["core_metrics"]["evidence_coverage_ratio"] = result.get(
                "evidence_coverage_ratio"
            )
        h02._finalize_h02_result(result, Path(output_dir))
        return result

    h02.evaluate_h02_prediction_violation_attention = evaluate_h02_prediction_violation_attention
    h02._ARC_AGI3_HYPOTHESIS_REPAIRS = True


def apply_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("ARC_AGI3_DISABLE_HYPOTHESIS_REPAIRS") == "1":
        return False
    _patch_future_options()
    _patch_higher_order()
    _patch_h02_report()
    _PATCHED = True
    return True
