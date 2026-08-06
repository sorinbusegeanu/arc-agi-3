from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

_PATCHED = False


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(connection, table):
        return set()
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _rows(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    previous = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in connection.execute(sql, parameters).fetchall()
        ]
    finally:
        connection.row_factory = previous


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _deep_value(payload: Any, keys: tuple[str, ...]) -> Any:
    wanted = set(keys)
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


def _clean(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "[]", "{}"}:
        return None
    if "surrogate:" in text.lower():
        return None
    return text


def _interaction_suffix(value: Any) -> str:
    text = str(value or "")
    for prefix in ("M0:interaction:", "interaction:"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _step_from_identifier(value: Any) -> int | None:
    text = _interaction_suffix(value)
    match = re.fullmatch(r"g(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|:)g(\d+)(?:$|:)", text)
    return int(match.group(1)) if match else None


def _node_metadata_index(
    connection: sqlite3.Connection,
    *,
    table: str,
) -> dict[str, dict[str, Any]]:
    columns = _columns(connection, table)
    if not columns or "node_id" not in columns:
        return {}

    selected = ["node_id"]
    for name in (
        "attrs_json",
        "payload_json",
        "evidence_json",
        "canonical_key",
        "first_seen_step",
        "last_seen_step",
        "first_seen_global_step",
        "last_seen_global_step",
    ):
        if name in columns:
            selected.append(name)

    index: dict[str, dict[str, Any]] = {}
    for row in _rows(
        connection,
        f"SELECT {', '.join(selected)} FROM {table} ORDER BY node_id ASC",
    ):
        node_id = str(row.get("node_id") or "")
        payload: dict[str, Any] = {}
        for name in ("attrs_json", "payload_json", "evidence_json"):
            payload.update(_json_dict(row.get(name)))

        step = _deep_value(
            payload,
            ("global_step", "step", "last_seen_global_step"),
        )
        if step is None:
            for name in (
                "last_seen_global_step",
                "last_seen_step",
                "first_seen_global_step",
                "first_seen_step",
            ):
                if row.get(name) is not None:
                    step = row.get(name)
                    break
        if step is None:
            step = _step_from_identifier(node_id)

        metadata = {
            "node_id": node_id,
            "game": _clean(
                _deep_value(
                    payload,
                    ("game", "game_key", "source_game_key"),
                )
            ),
            "sampler": _clean(
                _deep_value(payload, ("sampler", "source_sampler"))
            ),
            "context": _clean(
                _deep_value(
                    payload,
                    (
                        "context_signature",
                        "context_key",
                        "source_context_key",
                        "context",
                    ),
                )
            ),
            "action": _deep_value(
                payload,
                ("action", "action_key", "source_action"),
            ),
            "step": step,
        }
        aliases = {
            node_id,
            _interaction_suffix(node_id),
        }
        if metadata["step"] is not None:
            aliases.add(f"g{int(metadata['step'])}")
            aliases.add(f"M0:interaction:g{int(metadata['step'])}")
        for alias in aliases:
            if alias:
                existing = index.get(alias)
                if existing is None or (
                    existing.get("game") is None
                    and metadata.get("game") is not None
                ):
                    index[alias] = metadata
    return index


def _compact_h02_metadata(memory_dir: Path) -> dict[str, Any]:
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return {}

    with sqlite3.connect(current_state) as connection:
        if not (
            _table_exists(connection, "memory_scores")
            and _table_exists(connection, "memory_edges")
        ):
            return {}
        row = connection.execute(
            """
            WITH score_base AS (
                SELECT
                    node_id,
                    COALESCE(replay_priority, 0.0) AS replay_priority
                FROM memory_scores
                WHERE node_id LIKE 'M0:interaction:%'
            ),
            violation_base AS (
                SELECT DISTINCT source_node_id AS node_id
                FROM memory_edges
                WHERE edge_type='violates_prediction'
                  AND source_node_id LIKE 'M0:interaction:%'
            )
            SELECT
                COUNT(*) AS total_count,
                SUM(
                    CASE WHEN violation_base.node_id IS NOT NULL
                    THEN 1 ELSE 0 END
                ) AS violating_count,
                AVG(
                    CASE WHEN violation_base.node_id IS NOT NULL
                    THEN score_base.replay_priority END
                ) AS violating_mean,
                AVG(
                    CASE WHEN violation_base.node_id IS NULL
                    THEN score_base.replay_priority END
                ) AS non_violating_mean
            FROM score_base
            LEFT JOIN violation_base USING (node_id)
            """
        ).fetchone()

    total = int((row[0] if row else 0) or 0)
    violating = int((row[1] if row else 0) or 0)
    violating_mean = (
        None if row is None or row[2] is None else float(row[2])
    )
    non_violating_mean = (
        None if row is None or row[3] is None else float(row[3])
    )
    return {
        "row_count_available": total,
        "row_count_used": total,
        "prediction_violation_row_count": violating,
        "non_prediction_violation_row_count": max(0, total - violating),
        "mean_replay_priority_for_prediction_violating_interactions":
            violating_mean,
        "mean_replay_priority_for_non_prediction_violating_interactions":
            non_violating_mean,
        "candidate_tables_used": ["memory_scores", "memory_edges"],
        "prediction_violation_metric_source":
            "memory_edges.edge_type=violates_prediction",
        "replay_priority_metric_source":
            "memory_scores.replay_priority",
        "db_found": True,
        "db_path": str(current_state),
        "selected_db_path": str(current_state),
        "schema_inspected": True,
        "tables_seen": ["memory_scores", "memory_edges"],
        "compact_join_row_count": total,
        "compact_join_violating_row_count": violating,
        "compact_metric_metadata_complete": total > 0,
    }


def _repair_h02_result(
    h02: Any,
    *,
    result: dict[str, Any],
    output_dir: Path,
    memory_dir: Path | None,
) -> dict[str, Any]:
    if memory_dir is None:
        return result
    metadata = _compact_h02_metadata(Path(memory_dir))
    if not metadata:
        return result

    result.update(metadata)
    if result.get("direct_replay_lift_available") is True:
        result["direct_replay_lift_evidence_source"] = (
            "compact_interaction_join"
        )
        result["evidence_source"] = (
            "direct_streaming_manifest_and_compact_memory"
            if "direct_streaming" in str(result.get("evidence_source") or "")
            else "compact_memory_interaction_join"
        )
    h02._finalize_h02_result(result, Path(output_dir))
    return result


def _family_strength(record: dict[str, Any]) -> float:
    support = max(0.0, float(record.get("family_link_support_count") or 0))
    roles = max(0.0, float(record.get("family_link_role_count") or 0))
    events = max(0.0, float(record.get("family_link_event_count") or 0))
    gain = max(
        0.0,
        float(record.get("family_link_prediction_gain") or 0.0),
    )
    return (
        0.30 * math.log1p(events)
        + 0.25 * math.log1p(roles)
        + 0.20 * math.log1p(support)
        + 0.25 * gain
    )


def _strong_family(record: dict[str, Any]) -> bool:
    verified = (
        str(record.get("family_link_provenance_status") or "")
        == "verified"
    )
    support = int(record.get("family_link_support_count") or 0)
    roles = int(record.get("family_link_role_count") or 0)
    events = int(record.get("family_link_event_count") or 0)
    gain = float(record.get("family_link_prediction_gain") or 0.0)

    return bool(
        verified
        and support >= 2
        and events >= 2
        and (roles >= 2 or gain > 0.0)
    )


def _rerank_world_model_family_links(
    memory_dir: Path,
    *,
    configured_cap: int,
) -> dict[str, Any]:
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return {"world_model_family_selection_repair_applied": False}

    with sqlite3.connect(current_state) as connection:
        connection.row_factory = sqlite3.Row
        if not all(
            _table_exists(connection, table)
            for table in (
                "world_model_components",
                "world_model_family_links",
                "world_model_links",
            )
        ):
            return {"world_model_family_selection_repair_applied": False}

        component_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    component_signature,
                    candidate_family_link_count,
                    retained_family_link_count,
                    family_links_dropped_low_support,
                    family_links_dropped_limit
                FROM world_model_components
                ORDER BY component_signature
                """
            ).fetchall()
        ]

        summaries: list[dict[str, Any]] = []
        total_before = 0
        total_after = 0
        total_pruned_weak = 0
        total_pruned_adaptive = 0

        for component in component_rows:
            signature = str(component["component_signature"])
            records = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT
                        component_signature,
                        family_signature,
                        family_link_support_count,
                        family_link_role_count,
                        family_link_event_count,
                        family_link_prediction_gain,
                        family_link_provenance_status
                    FROM world_model_family_links
                    WHERE component_signature=?
                    ORDER BY family_signature
                    """,
                    (signature,),
                ).fetchall()
            ]
            total_before += len(records)

            strong = [record for record in records if _strong_family(record)]
            strong.sort(
                key=lambda record: (
                    -_family_strength(record),
                    -int(record.get("family_link_event_count") or 0),
                    -int(record.get("family_link_role_count") or 0),
                    -int(record.get("family_link_support_count") or 0),
                    str(record.get("family_signature") or ""),
                )
            )

            candidate_count = int(
                component.get("candidate_family_link_count") or len(records)
            )
            adaptive_cap = min(
                max(1, int(configured_cap)),
                max(8, int(round(math.sqrt(max(1, candidate_count))))),
            )
            selected = strong[:adaptive_cap]
            selected_families = {
                str(record["family_signature"]) for record in selected
            }
            pruned_weak = len(records) - len(strong)
            pruned_adaptive = max(0, len(strong) - len(selected))
            total_pruned_weak += pruned_weak
            total_pruned_adaptive += pruned_adaptive
            total_after += len(selected)

            connection.execute(
                "DELETE FROM world_model_family_links "
                "WHERE component_signature=?",
                (signature,),
            )
            connection.execute(
                "DELETE FROM world_model_links "
                "WHERE component_signature=? AND linked_type='family'",
                (signature,),
            )
            for record in selected:
                connection.execute(
                    """
                    INSERT INTO world_model_family_links (
                        component_signature,
                        family_signature,
                        family_link_support_count,
                        family_link_role_count,
                        family_link_event_count,
                        family_link_prediction_gain,
                        family_link_provenance_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signature,
                        str(record["family_signature"]),
                        int(record.get("family_link_support_count") or 0),
                        int(record.get("family_link_role_count") or 0),
                        int(record.get("family_link_event_count") or 0),
                        float(
                            record.get("family_link_prediction_gain") or 0.0
                        ),
                        str(
                            record.get("family_link_provenance_status")
                            or "missing"
                        ),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO world_model_links (
                        component_signature,
                        linked_type,
                        linked_key,
                        support_count,
                        first_seen_global_step,
                        last_seen_global_step
                    )
                    SELECT
                        ?,
                        'family',
                        ?,
                        1,
                        first_seen_global_step,
                        last_seen_global_step
                    FROM world_model_components
                    WHERE component_signature=?
                    ON CONFLICT(component_signature, linked_type, linked_key)
                    DO UPDATE SET support_count=excluded.support_count
                    """,
                    (
                        signature,
                        str(record["family_signature"]),
                        signature,
                    ),
                )

            old_low = int(
                component.get("family_links_dropped_low_support") or 0
            )
            old_limit = int(
                component.get("family_links_dropped_limit") or 0
            )
            retained = len(selected)
            connection.execute(
                """
                UPDATE world_model_components
                SET
                    linked_family_count=?,
                    retained_family_link_count=?,
                    dropped_family_link_count=
                        MAX(0, candidate_family_link_count - ?),
                    family_links_dropped_low_support=?,
                    family_links_dropped_limit=?
                WHERE component_signature=?
                """,
                (
                    retained,
                    retained,
                    retained,
                    old_low + pruned_weak,
                    old_limit + pruned_adaptive,
                    signature,
                ),
            )
            summaries.append(
                {
                    "component_signature": signature,
                    "candidate_family_count": candidate_count,
                    "retained_before": len(records),
                    "strong_family_count": len(strong),
                    "adaptive_cap": adaptive_cap,
                    "retained_after": retained,
                    "pruned_weak": pruned_weak,
                    "pruned_adaptive_cap": pruned_adaptive,
                    "selected_families_sample": sorted(
                        selected_families
                    )[:20],
                }
            )

        summary = {
            "world_model_family_selection_repair_applied": True,
            "world_model_family_selection_policy":
                "verified_support_and_event_gate_then_adaptive_sqrt_cap",
            "world_model_family_configured_cap": int(configured_cap),
            "world_model_family_link_count_before": total_before,
            "world_model_family_link_count_after": total_after,
            "world_model_family_links_pruned_weak": total_pruned_weak,
            "world_model_family_links_pruned_adaptive_cap":
                total_pruned_adaptive,
            "world_model_family_component_summaries": summaries[:100],
        }
        connection.execute(
            """
            INSERT INTO memory_summary (key, value_json)
            VALUES ('world_model_family_selection_repair', ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
            """,
            (json.dumps(summary, sort_keys=True),),
        )
        connection.commit()
        return summary


def _metadata_lookup(
    state_index: dict[str, dict[str, Any]],
    graph_index: dict[str, dict[str, Any]],
    value: Any,
) -> dict[str, Any] | None:
    text = str(value or "")
    candidates = (
        text,
        _interaction_suffix(text),
        f"M0:interaction:{_interaction_suffix(text)}",
    )
    for candidate in candidates:
        if candidate in state_index:
            return state_index[candidate]
        if candidate in graph_index:
            return graph_index[candidate]
    return None


def _backfill_future_option_edge_provenance(
    state_connection: sqlite3.Connection,
    graph_connection: sqlite3.Connection,
) -> dict[str, Any]:
    if not _table_exists(state_connection, "future_option_events"):
        return {
            "future_option_edge_provenance_backfill_applied": False,
        }

    state_index = _node_metadata_index(
        state_connection,
        table="memory_nodes",
    )
    graph_index = _node_metadata_index(
        graph_connection,
        table="graph_nodes",
    )

    events = _rows(
        state_connection,
        """
        SELECT
            event_id,
            source_interaction_id,
            target_interaction_id,
            evidence_json,
            classification_provenance_status
        FROM future_option_events
        WHERE classification_source='future_option_edge'
        ORDER BY event_id
        """,
    )
    resolved_source = 0
    resolved_target = 0
    verified = 0

    for event in events:
        evidence = _json_dict(event.get("evidence_json"))
        source_value = (
            event.get("source_interaction_id")
            or evidence.get("source_node_id")
            or evidence.get("source_interaction_id")
        )
        target_value = (
            event.get("target_interaction_id")
            or evidence.get("target_node_id")
            or evidence.get("target_interaction_id")
        )
        source = _metadata_lookup(
            state_index,
            graph_index,
            source_value,
        )
        target = _metadata_lookup(
            state_index,
            graph_index,
            target_value,
        )

        source_game = _clean(None if source is None else source.get("game"))
        source_context = _clean(
            None if source is None else source.get("context")
        )
        target_game = _clean(None if target is None else target.get("game"))
        target_context = _clean(
            None if target is None else target.get("context")
        )
        if source_game and source_context:
            resolved_source += 1
        if target_game and target_context:
            resolved_target += 1

        # The future-option classification belongs to the source interaction.
        # A concrete source game/context is sufficient to verify that event.
        status = (
            "verified"
            if source_game and source_context
            else str(
                event.get("classification_provenance_status")
                or "resolved_with_surrogate"
            )
        )
        if status == "verified":
            verified += 1

        evidence["provenance_backfill"] = {
            "source_metadata_resolved": bool(
                source_game and source_context
            ),
            "target_metadata_resolved": bool(
                target_game and target_context
            ),
            "source_metadata_node_id": (
                None if source is None else source.get("node_id")
            ),
            "target_metadata_node_id": (
                None if target is None else target.get("node_id")
            ),
            "verification_scope": "source_interaction",
        }

        state_connection.execute(
            """
            UPDATE future_option_events
            SET
                game=COALESCE(?, game),
                sampler=COALESCE(?, sampler),
                context_key=COALESCE(?, context_key),
                action_key=COALESCE(?, action_key),
                first_seen_global_step=
                    COALESCE(?, first_seen_global_step),
                last_seen_global_step=
                    COALESCE(?, last_seen_global_step),
                source_game_id=COALESCE(?, source_game_id),
                source_sampler=COALESCE(?, source_sampler),
                source_context_signature=
                    COALESCE(?, source_context_signature),
                source_action=COALESCE(?, source_action),
                source_game_key=COALESCE(?, source_game_key),
                source_context_key=COALESCE(?, source_context_key),
                target_game_key=COALESCE(?, target_game_key),
                target_context_key=COALESCE(?, target_context_key),
                source_game_is_surrogate=
                    CASE WHEN ? IS NOT NULL THEN 0
                         ELSE source_game_is_surrogate END,
                source_context_is_surrogate=
                    CASE WHEN ? IS NOT NULL THEN 0
                         ELSE source_context_is_surrogate END,
                target_game_is_surrogate=
                    CASE WHEN ? IS NOT NULL THEN 0
                         ELSE target_game_is_surrogate END,
                target_context_is_surrogate=
                    CASE WHEN ? IS NOT NULL THEN 0
                         ELSE target_context_is_surrogate END,
                context_resolution_source=
                    CASE WHEN ? IS NOT NULL
                         THEN 'interaction_node_metadata'
                         ELSE context_resolution_source END,
                context_is_surrogate=
                    CASE WHEN ? IS NOT NULL THEN 0
                         ELSE context_is_surrogate END,
                classification_provenance_status=?,
                evidence_json=?
            WHERE event_id=?
            """,
            (
                source_game,
                None if source is None else source.get("sampler"),
                source_context,
                None if source is None else source.get("action"),
                None if source is None else source.get("step"),
                None if source is None else source.get("step"),
                source_game,
                None if source is None else source.get("sampler"),
                source_context,
                None if source is None else source.get("action"),
                source_game,
                source_context,
                target_game,
                target_context,
                source_game,
                source_context,
                target_game,
                target_context,
                source_context,
                source_context,
                status,
                json.dumps(evidence, sort_keys=True),
                str(event["event_id"]),
            ),
        )

    summary = {
        "future_option_edge_provenance_backfill_applied": True,
        "future_option_edge_event_count_seen": len(events),
        "future_option_edge_source_scope_resolved": resolved_source,
        "future_option_edge_target_scope_resolved": resolved_target,
        "future_option_edge_verified_after_backfill": verified,
        "future_option_edge_provenance_source":
            "memory_nodes_or_graph_nodes_metadata",
    }
    state_connection.commit()
    return summary


def apply_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("ARC_AGI3_DISABLE_REMAINING_REPORT_REPAIRS") == "1":
        return False

    import v6.hypothesis_h02_report as h02
    import v6.higher_order_substrate as higher_order
    import v6.future_options as future_options

    if not getattr(h02, "_ARC_AGI3_COMPACT_METADATA_FIX", False):
        original_h02 = h02.evaluate_h02_prediction_violation_attention

        def evaluate_h02(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original_h02(*args, **kwargs)
            output_dir = (
                kwargs.get("output_dir")
                if kwargs.get("output_dir") is not None
                else (args[1] if len(args) > 1 else None)
            )
            memory_dir = kwargs.get("memory_dir")
            if output_dir is None:
                return result
            return _repair_h02_result(
                h02,
                result=result,
                output_dir=Path(output_dir),
                memory_dir=(
                    None if memory_dir is None else Path(memory_dir)
                ),
            )

        h02.evaluate_h02_prediction_violation_attention = evaluate_h02
        h02._ARC_AGI3_COMPACT_METADATA_FIX = True

    if not getattr(
        higher_order,
        "_ARC_AGI3_WORLD_MODEL_FAMILY_SELECTIVITY_FIX",
        False,
    ):
        original_world = higher_order.derive_world_model_components_only

        def derive_world_models(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original_world(*args, **kwargs)
            memory_dir = kwargs.get("memory_dir")
            if memory_dir is None and args:
                memory_dir = args[0]
            if memory_dir is None:
                return result
            configured_cap = int(
                kwargs.get(
                    "max_world_model_family_links",
                    higher_order.MAX_WORLD_MODEL_FAMILY_LINKS,
                )
            )
            repair = _rerank_world_model_family_links(
                Path(memory_dir),
                configured_cap=configured_cap,
            )
            return {**result, **repair}

        higher_order.derive_world_model_components_only = derive_world_models
        higher_order._ARC_AGI3_WORLD_MODEL_FAMILY_SELECTIVITY_FIX = True

    if not getattr(
        future_options,
        "_ARC_AGI3_EDGE_PROVENANCE_BACKFILL_FIX",
        False,
    ):
        original_events = future_options.derive_future_option_events

        def derive_future_events(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original_events(*args, **kwargs)
            state_connection = args[0]
            graph_connection = args[1]
            repair = _backfill_future_option_edge_provenance(
                state_connection,
                graph_connection,
            )
            return {**result, **repair}

        future_options.derive_future_option_events = derive_future_events
        future_options._ARC_AGI3_EDGE_PROVENANCE_BACKFILL_FIX = True

    import v6.hypothesis_suite_report as suite

    suite.evaluate_h02_prediction_violation_attention = (
        h02.evaluate_h02_prediction_violation_attention
    )
    suite.derive_world_model_components_only = (
        higher_order.derive_world_model_components_only
    )

    _PATCHED = True
    return True
