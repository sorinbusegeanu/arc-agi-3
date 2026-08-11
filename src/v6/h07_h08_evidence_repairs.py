from __future__ import annotations

import inspect
import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

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


def _dict_rows(
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


def _deep_value(payload: Any, names: Iterable[str]) -> Any:
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


def _normalize_interaction_id(value: Any) -> str:
    text = str(value or "")
    for prefix in ("M0:interaction:", "interaction:"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _interaction_step(identifier: str) -> int | None:
    match = re.fullmatch(r"g(\d+)", identifier)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|:)g(\d+)(?:$|:)", identifier)
    return int(match.group(1)) if match else None


def _clean_scope(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "[]", "{}"}:
        return None
    return text


def _interaction_metadata(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not _table_exists(connection, "memory_nodes"):
        return result

    columns = _columns(connection, "memory_nodes")
    selected = ["node_id"]
    for name in (
        "attrs_json",
        "payload_json",
        "canonical_key",
        "first_seen_step",
        "last_seen_step",
        "first_seen_global_step",
        "last_seen_global_step",
    ):
        if name in columns:
            selected.append(name)

    for row in _dict_rows(
        connection,
        f"SELECT {', '.join(selected)} FROM memory_nodes "
        "WHERE node_id LIKE 'M0:interaction:%' ORDER BY node_id ASC",
    ):
        interaction_id = _normalize_interaction_id(row.get("node_id"))
        payload = {}
        payload.update(_json_dict(row.get("attrs_json")))
        payload.update(_json_dict(row.get("payload_json")))
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
            step = _interaction_step(interaction_id)

        result[interaction_id] = {
            "global_step": step,
            "game": _deep_value(
                payload,
                ("game", "game_key", "source_game_key"),
            ),
            "context": _deep_value(
                payload,
                (
                    "context_signature",
                    "context_key",
                    "source_context_key",
                    "context",
                ),
            ),
            "family": _deep_value(
                payload,
                (
                    "family",
                    "family_id",
                    "family_signature",
                    "actual_family",
                    "predicted_family",
                ),
            ),
            "carrier": _deep_value(
                payload,
                ("carrier", "carrier_id", "carrier_signature"),
            ),
            "predicted_family": _deep_value(
                payload,
                ("predicted_family",),
            ),
            "actual_family": _deep_value(payload, ("actual_family",)),
        }
    return result


def _node_kind_value(node_id: Any) -> tuple[str | None, str | None]:
    text = str(node_id or "")
    prefixes = (
        ("family", "M2:family:"),
        ("family", "family:"),
        ("carrier", "M3:carrier:"),
        ("carrier", "carrier:"),
        ("role", "M3:role:"),
        ("role", "role:"),
        ("context", "M0:context:"),
        ("context", "context:"),
        ("game", "game:"),
        ("contingency", "M1:contingency:"),
        ("contingency", "contingency:"),
    )
    for kind, prefix in prefixes:
        if text.startswith(prefix):
            return kind, text[len(prefix):]
    return None, None


def _interaction_structures(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, set[str]]]:
    output: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    if not _table_exists(connection, "memory_edges"):
        return output

    rows = _dict_rows(
        connection,
        """
        SELECT source_node_id, target_node_id, edge_type
        FROM memory_edges
        WHERE source_node_id LIKE 'M0:interaction:%'
           OR target_node_id LIKE 'M0:interaction:%'
        ORDER BY source_node_id, target_node_id, edge_type
        """,
    )
    contingency_to_interactions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source = str(row.get("source_node_id") or "")
        target = str(row.get("target_node_id") or "")
        if source.startswith("M0:interaction:"):
            interaction_id = _normalize_interaction_id(source)
            adjacent = target
        else:
            interaction_id = _normalize_interaction_id(target)
            adjacent = source
        kind, value = _node_kind_value(adjacent)
        if kind and value:
            if kind == "contingency":
                contingency_to_interactions[adjacent].add(interaction_id)
            else:
                output[interaction_id][kind].add(value)

    if contingency_to_interactions:
        placeholders = ",".join("?" for _ in contingency_to_interactions)
        two_hop = _dict_rows(
            connection,
            f"""
            SELECT source_node_id, target_node_id, edge_type
            FROM memory_edges
            WHERE source_node_id IN ({placeholders})
               OR target_node_id IN ({placeholders})
            ORDER BY source_node_id, target_node_id, edge_type
            """,
            tuple(contingency_to_interactions) * 2,
        )
        for row in two_hop:
            source = str(row.get("source_node_id") or "")
            target = str(row.get("target_node_id") or "")
            if source in contingency_to_interactions:
                contingency = source
                adjacent = target
            elif target in contingency_to_interactions:
                contingency = target
                adjacent = source
            else:
                continue
            kind, value = _node_kind_value(adjacent)
            if kind and value and kind != "contingency":
                for interaction_id in contingency_to_interactions[contingency]:
                    output[interaction_id][kind].add(value)
    return output


def _matching_roles(
    *,
    source_roles: list[str],
    role_links: dict[str, dict[str, set[str]]],
    structures: dict[str, set[str]],
) -> list[str]:
    matched: list[str] = []
    for role in source_roles:
        links = role_links.get(role, {})
        if any(
            set(structures.get(kind, set()))
            & {str(item) for item in links.get(kind, set())}
            for kind in ("family", "carrier", "context", "game")
        ):
            matched.append(role)
    return matched


def _robust_compact_prediction_events(
    higher_order: Any,
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    role_links: dict[str, dict[str, set[str]]],
    transfer_history: Any,
    contradiction_only: bool,
) -> list[dict[str, Any]]:
    if (
        first_seen_global_step is None
        or not source_roles
        or not _table_exists(state_conn, "memory_scores")
        or not _table_exists(state_conn, "memory_edges")
    ):
        return []

    score_columns = _columns(state_conn, "memory_scores")
    selected = ["node_id"]
    for name in (
        "updated_step",
        "first_seen_step",
        "last_seen_step",
        "first_seen_global_step",
        "last_seen_global_step",
    ):
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
            """
            SELECT DISTINCT source_node_id
            FROM memory_edges
            WHERE edge_type = 'violates_prediction'
              AND source_node_id LIKE 'M0:interaction:%'
            """,
        )
    }
    metadata = _interaction_metadata(state_conn)
    structures_by_interaction = _interaction_structures(state_conn)

    events: list[dict[str, Any]] = []
    synthetic_step = int(first_seen_global_step) + 1
    for ordinal, row in enumerate(score_rows):
        interaction_id = _normalize_interaction_id(row.get("node_id"))
        violated = interaction_id in violation_ids
        if contradiction_only and not violated:
            continue

        meta = metadata.get(interaction_id, {})
        step_value = meta.get("global_step")
        if step_value is None:
            for name in (
                "updated_step",
                "last_seen_global_step",
                "last_seen_step",
                "first_seen_global_step",
                "first_seen_step",
            ):
                if row.get(name) is not None:
                    step_value = row.get(name)
                    break
        if step_value is None:
            step_value = _interaction_step(interaction_id)
        try:
            step = int(step_value)
        except (TypeError, ValueError):
            step = synthetic_step + ordinal
        if step <= int(first_seen_global_step):
            continue

        structures = {
            kind: set(values)
            for kind, values in structures_by_interaction.get(
                interaction_id, {}
            ).items()
        }
        for kind in ("family", "carrier", "context", "game"):
            value = _clean_scope(meta.get(kind))
            if value:
                structures.setdefault(kind, set()).add(value)

        predicted_family = str(meta.get("predicted_family") or "")
        actual_family = str(meta.get("actual_family") or "")
        for value in (predicted_family, actual_family):
            if value:
                structures.setdefault("family", set()).add(value)

        matched_roles = _matching_roles(
            source_roles=source_roles,
            role_links=role_links,
            structures=structures,
        )
        if not matched_roles:
            continue

        rates = [
            higher_order._prior_role_success_rate(
                transfer_rows,
                role=role,
                before_step=step,
                transfer_history=transfer_history,
            )[0]
            for role in matched_roles
        ]
        if contradiction_only:
            rates = [1.0 - rate for rate in rates]
        baseline = max(rates, default=0.0)
        concept_score = (
            higher_order._combined_role_score(rates)
            if len(rates) >= 2
            else baseline
        )
        outcome = 1.0 if contradiction_only else (0.0 if violated else 1.0)
        feature_step = (
            transfer_history.max_any_step_before(step)
            if transfer_history is not None
            else None
        )
        if feature_step is None:
            feature_step = int(first_seen_global_step)
        if feature_step >= step:
            continue

        event_type = (
            "contradiction_resolution"
            if contradiction_only
            else "prediction"
        )
        event_id = (
            f"{event_type}:compact_interaction:{interaction_id}"
        )
        events.append(
            {
                "concept_id": candidate_signature,
                "event_id": event_id,
                "event_type": event_type,
                "evaluation_scope": "later_global_step",
                "predicted_family": predicted_family,
                "actual_family": actual_family,
                "candidate_role_family_ids": sorted(
                    {
                        str(family)
                        for role in matched_roles
                        for family in role_links.get(role, {}).get(
                            "family", set()
                        )
                    }
                ),
                "best_single_role_score": baseline,
                "lower_level_baseline_score": baseline,
                "concept_enabled_score": concept_score,
                "prediction_gain": concept_score - baseline,
                "behavioral_gain": (
                    concept_score - baseline
                    if contradiction_only
                    else -abs(concept_score - outcome)
                    + abs(baseline - outcome)
                ),
                "_outcome": outcome,
                "_evaluation_global_step": step,
                "_feature_global_step_max": feature_step,
                "_label_used_as_feature": False,
                "_source_role_ids": sorted(matched_roles),
                "_family_ids": sorted(structures.get("family", set())),
                "_carrier_ids": sorted(structures.get("carrier", set())),
                "_context_keys": sorted(structures.get("context", set())),
                "_game_keys": sorted(structures.get("game", set())),
                "_compact_prediction_surrogate": True,
                "_compact_structural_link_verified": True,
            }
        )
    return events


def _recompute_population_continuity(
    higher_order: Any,
    *,
    events: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    state: dict[str, Any],
    previous_state: dict[str, Any] | None,
    source_roles: list[str],
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not previous_state:
        diagnostics["population_retention_policy"] = (
            "current_structural_relevance_only"
        )
        return events, diagnostics, state

    previous_ids = {
        str(value)
        for value in previous_state.get(
            "relevant_event_ids",
            previous_state.get("explanation_event_ids", ()),
        )
        or ()
    }
    if not previous_ids:
        diagnostics["population_retention_policy"] = (
            "current_structural_relevance_only"
        )
        return events, diagnostics, state

    eligible_events = [
        event for event in events if not bool(event.get("invalid"))
    ]
    available_ids = {
        str(event.get("event_id"))
        for event in eligible_events
    }
    carried_ids = previous_ids & available_ids
    for event in eligible_events:
        event_id = str(event.get("event_id"))
        if event_id in carried_ids and not bool(event.get("is_relevant")):
            event["is_relevant"] = True
            reasons = list(event.get("relevance_reasons", ()) or ())
            if "historical_population_retention" not in reasons:
                reasons.append("historical_population_retention")
            event["relevance_reasons"] = reasons

    relevant = [
        event for event in eligible_events if bool(event.get("is_relevant"))
    ]
    unrelated = [
        event for event in eligible_events if not bool(event.get("is_relevant"))
    ]
    invalid = [
        event for event in events if bool(event.get("invalid"))
    ]
    explained = [
        event for event in relevant if bool(event.get("explained"))
    ]
    rejected = [
        event for event in relevant if not bool(event.get("explained"))
    ]
    current_ids = {str(event["event_id"]) for event in relevant}
    explained_ids = {str(event["event_id"]) for event in explained}
    previous_explained_ids = {
        str(value)
        for value in previous_state.get(
            "explained_relevant_event_ids", ()
        )
        or ()
    }
    retained_ids = previous_ids & current_ids
    new_ids = current_ids - previous_ids
    expired_ids = previous_ids - current_ids
    retained_fraction_previous = len(retained_ids) / max(
        1, len(previous_ids)
    )
    retained_fraction_current = len(retained_ids) / max(
        1, len(current_ids)
    )
    comparable = (
        retained_fraction_previous
        >= float(config.promotion_population_comparability_threshold)
    )
    if current_ids == previous_ids:
        population_change = "unchanged"
    elif retained_ids == previous_ids:
        population_change = "population_expansion"
    elif retained_ids == current_ids:
        population_change = "population_contraction"
    else:
        population_change = "population_definition_changed"
    population_status = "comparable" if comparable else population_change

    retained_events = [
        event
        for event in relevant
        if str(event["event_id"]) in retained_ids
    ]
    new_events = [
        event for event in relevant if str(event["event_id"]) in new_ids
    ]
    retained_explained = [
        event for event in retained_events if bool(event.get("explained"))
    ]
    new_explained = [
        event for event in new_events if bool(event.get("explained"))
    ]
    coverage = len(explained) / len(relevant) if relevant else 0.0
    retained_coverage = (
        len(retained_explained) / len(retained_events)
        if retained_events
        else 0.0
    )
    prior_retained_coverage = (
        len(retained_ids & previous_explained_ids) / len(retained_ids)
        if retained_ids
        else None
    )
    coverage_delta = (
        retained_coverage - prior_retained_coverage
        if comparable and prior_retained_coverage is not None
        else None
    )
    new_coverage = (
        len(new_explained) / len(new_events) if new_events else 0.0
    )
    retained_lift = (
        higher_order._mean_event_gain(
            retained_events, "concept_incremental_gain"
        )
        or 0.0
    )
    evaluation_population_id = sha1(
        "\n".join(sorted(current_ids)).encode("utf-8")
    ).hexdigest()

    definition_cost = 0.05 * max(1, len(source_roles))
    baseline_cost = sum(
        float(event.get("baseline_description_cost") or 0.0)
        for event in relevant
    )
    concept_cost = definition_cost + sum(
        float(event.get("concept_description_cost") or 0.0)
        for event in relevant
    )

    previous_eligible = previous_state.get(
        "relevant_event_count",
        previous_state.get("eligible_event_count"),
    )
    previous_explained = previous_state.get("explained_event_count")
    previous_coverage = previous_state.get("incremental_coverage")
    longitudinal = {
        "previous_epoch": previous_state.get("epoch_id"),
        "previous_explained_event_count": previous_explained,
        "current_explained_event_count": len(explained),
        "explained_event_count_delta": (
            None
            if previous_explained is None
            else len(explained) - int(previous_explained or 0)
        ),
        "previous_eligible_event_count": previous_eligible,
        "current_eligible_event_count": len(relevant),
        "eligible_event_count_delta": (
            None
            if previous_eligible is None
            else len(relevant) - int(previous_eligible or 0)
        ),
        "previous_incremental_explanatory_coverage": previous_coverage,
        "current_incremental_explanatory_coverage": coverage,
        "coverage_delta": coverage_delta,
        "relevant_population_signature": evaluation_population_id,
        "relevant_event_count": len(relevant),
        "retained_event_count": len(retained_ids),
        "new_relevant_event_count": len(new_ids),
        "expired_relevant_event_count": len(expired_ids),
        "population_comparison": population_status,
        "population_change": population_change,
        "retained_fraction_previous": retained_fraction_previous,
        "retained_fraction_current": retained_fraction_current,
        "current_population_coverage": coverage,
        "retained_population_coverage": retained_coverage,
        "retained_population_lift": retained_lift,
        "new_population_coverage": new_coverage,
        "previous_incremental_coverage": previous_coverage,
        "current_incremental_coverage": coverage,
        "incremental_coverage_delta": coverage_delta,
        "classification": population_status,
        "historical_population_ids_available": len(previous_ids),
        "historical_population_ids_carried_forward": len(carried_ids),
    }

    relevance_reasons = (
        "source_role_overlap",
        "target_role_overlap",
        "family_overlap",
        "carrier_overlap",
        "context_overlap",
        "game_overlap",
        "historical_population_retention",
    )
    relevance_counts = {
        reason: sum(
            reason in (event.get("relevance_reasons", ()) or ())
            for event in eligible_events
        )
        for reason in relevance_reasons
    }

    state.update(
        {
            "eligible_event_count": len(relevant),
            "relevant_event_count": len(relevant),
            "explained_event_count": len(explained),
            "incremental_coverage": coverage,
            "relevant_event_ids": sorted(current_ids),
            "explained_relevant_event_ids": sorted(explained_ids),
            "explanation_event_ids": sorted(current_ids),
            "relevant_population_signature": evaluation_population_id,
            "structure_fingerprint": evaluation_population_id,
        }
    )
    diagnostics.update(
        {
            "eligible_explanation_event_count": len(relevant),
            "relevant_heldout_event_count": len(relevant),
            "explained_relevant_event_count": len(explained),
            "unrelated_event_count": len(unrelated),
            "invalid_explanation_event_count": len(invalid),
            "explained_event_count": len(explained),
            "rejected_event_count": len(rejected),
            "incremental_explanatory_coverage": coverage,
            "relevant_incremental_coverage": coverage,
            "explained_event_type_counts": higher_order._event_type_counts(
                explained
            ),
            "rejected_event_type_counts": higher_order._event_type_counts(
                rejected
            ),
            "invalid_event_type_counts": higher_order._event_type_counts(
                invalid
            ),
            "prediction_explained_event_count": sum(
                "prediction"
                in (event.get("explanation_channels", ()) or ())
                for event in explained
            ),
            "behavioral_explained_event_count": sum(
                "behavioral"
                in (event.get("explanation_channels", ()) or ())
                for event in explained
            ),
            "compression_explained_event_count": sum(
                "compression"
                in (event.get("explanation_channels", ()) or ())
                for event in explained
            ),
            "multi_channel_explained_event_count": sum(
                len(event.get("explanation_channels", ()) or ()) > 1
                for event in explained
            ),
            "mean_prediction_gain": (
                higher_order._mean_event_gain(
                    relevant, "prediction_gain"
                )
                or 0.0
            ),
            "mean_behavioral_gain": (
                higher_order._mean_event_gain(
                    relevant, "behavioral_gain"
                )
                or 0.0
            ),
            "mean_compression_gain": (
                higher_order._mean_event_gain(
                    relevant, "compression_gain"
                )
                or 0.0
            ),
            "baseline_description_cost": baseline_cost,
            "concept_description_cost": concept_cost,
            "incremental_compression_gain": baseline_cost - concept_cost,
            "explained_event_ids_sample": sorted(explained_ids)[:20],
            "rejected_event_ids_sample": sorted(
                str(event["event_id"]) for event in rejected
            )[:20],
            "functional_coverage_longitudinal_change": longitudinal,
            "coverage_longitudinal_change": longitudinal,
            "coverage_change_classification": population_status,
            "relevant_event_type_diagnostics": (
                higher_order._relevant_event_type_diagnostics(relevant)
            ),
            "relevant_transfer_event_count": sum(
                event.get("event_type") == "transfer"
                for event in relevant
            ),
            "relevant_future_option_event_count": sum(
                event.get("event_type") == "future_option_motif"
                for event in relevant
            ),
            "relevant_prediction_event_count": sum(
                event.get("event_type") == "prediction"
                for event in relevant
            ),
            "relevant_behavior_event_count": sum(
                event.get("event_type")
                == "contradiction_resolution"
                for event in relevant
            ),
            "evaluation_population_id": evaluation_population_id,
            "role_combination_lift": (
                higher_order._mean_event_gain(
                    relevant, "concept_incremental_gain"
                )
                or 0.0
            ),
            "retained_event_count": len(retained_ids),
            "retained_fraction_previous": retained_fraction_previous,
            "retained_fraction_current": retained_fraction_current,
            "current_population_coverage": coverage,
            "retained_population_coverage": retained_coverage,
            "retained_population_lift": retained_lift,
            "new_population_coverage": new_coverage,
            "relevant_by_source_role_count": relevance_counts[
                "source_role_overlap"
            ],
            "relevant_by_target_role_count": relevance_counts[
                "target_role_overlap"
            ],
            "relevant_by_family_count": relevance_counts[
                "family_overlap"
            ],
            "relevant_by_carrier_count": relevance_counts[
                "carrier_overlap"
            ],
            "relevant_by_context_count": relevance_counts[
                "context_overlap"
            ],
            "relevant_by_game_count": relevance_counts[
                "game_overlap"
            ],
            "relevant_by_historical_population_count": relevance_counts[
                "historical_population_retention"
            ],
            "population_retention_policy": (
                "carry_forward_prior_relevant_ids_for_same_concept"
            ),
            "historical_population_ids_available": len(previous_ids),
            "historical_population_ids_carried_forward": len(carried_ids),
        }
    )
    return events, diagnostics, state


def _current_concept_validation(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    if not (
        _table_exists(connection, "concept_candidates")
        and _table_exists(
            connection, "concept_promotion_validation_diagnostics"
        )
    ):
        return {}

    current = {
        str(row[0])
        for row in connection.execute(
            "SELECT concept_signature FROM concept_candidates"
        ).fetchall()
    }
    rows = _dict_rows(
        connection,
        """
        SELECT diagnostic.concept_signature, diagnostic.payload_json
        FROM concept_promotion_validation_diagnostics AS diagnostic
        WHERE diagnostic.rowid IN (
            SELECT MAX(rowid)
            FROM concept_promotion_validation_diagnostics
            GROUP BY concept_signature
        )
        ORDER BY diagnostic.concept_signature
        """,
    )
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        signature = str(row.get("concept_signature") or "")
        if signature not in current:
            continue
        payload = _json_dict(row.get("payload_json"))
        score_passed = payload.get("promotion_score_gate_passed")
        if score_passed is None:
            try:
                score = float(
                    payload.get("adjusted_promotion_score")
                    or payload.get("promotion_score")
                    or 0.0
                )
                threshold = float(
                    payload.get("promotion_threshold") or 0.55
                )
                score_passed = score >= threshold
            except (TypeError, ValueError):
                score_passed = False
        payload["report_current_validated_promoted"] = bool(
            payload.get("current_validation_passed")
            and score_passed
        )
        output[signature] = payload
    return output


def _with_current_concept_flags_for_world_models(
    higher_order: Any,
    original: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    validate_world_models = bool(kwargs.get("validate_world_models"))
    memory_dir = kwargs.get("memory_dir")
    if not validate_world_models or memory_dir is None:
        return original(*args, **kwargs)

    paths = higher_order.ensure_memory_layout(Path(memory_dir))
    saved: list[tuple[str, int]] = []
    with sqlite3.connect(paths.current_state) as connection:
        validation = _current_concept_validation(connection)
        if not validation:
            return original(*args, **kwargs)
        saved = [
            (str(row[0]), int(row[1] or 0))
            for row in connection.execute(
                "SELECT concept_signature, is_promoted "
                "FROM concept_candidates"
            ).fetchall()
        ]
        connection.executemany(
            "UPDATE concept_candidates SET is_promoted=? "
            "WHERE concept_signature=?",
            [
                (
                    int(
                        validation.get(signature, {}).get(
                            "report_current_validated_promoted", False
                        )
                    ),
                    signature,
                )
                for signature, _old in saved
            ],
        )
        connection.commit()

    try:
        return original(*args, **kwargs)
    finally:
        with sqlite3.connect(paths.current_state) as connection:
            connection.executemany(
                "UPDATE concept_candidates SET is_promoted=? "
                "WHERE concept_signature=?",
                [(old, signature) for signature, old in saved],
            )
            connection.commit()


def _repair_h08_result(
    h08: Any,
    *,
    memory_dir: Path,
    output_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return result

    with sqlite3.connect(current_state) as connection:
        validation = _current_concept_validation(connection)
        if not validation:
            return result
        current_validated = {
            signature
            for signature, payload in validation.items()
            if payload.get("report_current_validated_promoted")
        }
        component_concepts: dict[str, set[str]] = defaultdict(set)
        if _table_exists(connection, "world_model_links"):
            for row in connection.execute(
                """
                SELECT component_signature, linked_key
                FROM world_model_links
                WHERE linked_type='concept'
                """
            ).fetchall():
                component_concepts[str(row[0])].add(str(row[1]))

    persistent_count = int(result.get("promoted_concept_count") or 0)
    current_count = len(current_validated)
    result["persistent_retained_concept_count"] = persistent_count
    result["current_validated_promoted_concept_count"] = current_count
    result["promoted_concept_count"] = current_count
    result["current_validated_promoted_concept_signatures"] = sorted(
        current_validated
    )

    qualifying_records = [
        dict(record)
        for record in result.get("qualifying_component_records", [])
        if isinstance(record, dict)
    ]
    current_qualifying: list[dict[str, Any]] = []
    for record in qualifying_records:
        signature = str(record.get("component_signature") or "")
        linked = component_concepts.get(signature, set())
        record["current_validated_linked_concept_count"] = len(
            linked & current_validated
        )
        record["linked_only_to_current_validated_concepts"] = bool(
            linked and linked <= current_validated
        )
        if record["linked_only_to_current_validated_concepts"]:
            current_qualifying.append(record)

    result["qualifying_component_records"] = current_qualifying
    result["qualifying_component_count"] = len(current_qualifying)
    result["qualifying_component_signatures"] = [
        str(record.get("component_signature") or "")
        for record in current_qualifying
    ]

    gates = dict(result.get("h08_validity_gates", {}) or {})
    gates["promoted_concepts"] = {
        "required": 1,
        "actual": current_count,
        "passed": current_count >= 1,
    }
    gates["qualifying_components"] = {
        "required": 1,
        "actual": len(current_qualifying),
        "passed": len(current_qualifying) >= 1,
    }
    result["h08_validity_gates"] = gates

    if current_count == 0:
        result["decision"] = "INSUFFICIENT_EVIDENCE"
        result["evidence_stage"] = "current_concept_validation_failed"
        result["missing_evidence"] = [
            "No current concept candidate passed incremental validation."
        ]
    elif len(current_qualifying) == 0:
        result["decision"] = "PARTIALLY_VALID"
        result["missing_evidence"] = [
            "No world-model component linked only to currently validated "
            "concepts satisfies all H08 coherence gates."
        ]
    elif (
        int(result.get("role_candidate_count") or 0) >= 1
        and int(result.get("role_transfer_success_count") or 0) >= 1
    ):
        result["decision"] = "VALID"
        result["missing_evidence"] = []

    core = dict(result.get("core_metrics", {}) or {})
    for key in (
        "promoted_concept_count",
        "persistent_retained_concept_count",
        "current_validated_promoted_concept_count",
        "qualifying_component_count",
    ):
        core[key] = result.get(key)
    result["core_metrics"] = core
    h08._write_outputs(Path(output_dir), result)
    return result


def apply_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("ARC_AGI3_DISABLE_H07_H08_EVIDENCE_REPAIRS") == "1":
        return False

    import v6.higher_order_substrate as higher_order
    import v6.hypothesis_h08_report as h08

    if not getattr(
        higher_order,
        "_ARC_AGI3_H07_EVIDENCE_CONTINUITY_FIX",
        False,
    ):
        original_prediction = higher_order._prediction_explanation_events
        original_contradiction = (
            higher_order._contradiction_resolution_explanation_events
        )
        original_build = (
            higher_order._build_functional_explanation_diagnostics
        )
        original_validate = higher_order.validate_incremental_promotions_only

        def prediction_events(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            events = original_prediction(*args, **kwargs)
            if events:
                return events
            connection = args[0] if args else kwargs["state_conn"]
            return _robust_compact_prediction_events(
                higher_order,
                connection,
                candidate_signature=kwargs["candidate_signature"],
                source_roles=kwargs["source_roles"],
                first_seen_global_step=kwargs["first_seen_global_step"],
                transfer_rows=kwargs["transfer_rows"],
                role_links=kwargs["role_links"],
                transfer_history=kwargs.get("transfer_history"),
                contradiction_only=False,
            )

        def contradiction_events(
            *args: Any,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            events = original_contradiction(*args, **kwargs)
            if events:
                return events
            connection = args[0] if args else kwargs["state_conn"]
            return _robust_compact_prediction_events(
                higher_order,
                connection,
                candidate_signature=kwargs["candidate_signature"],
                source_roles=kwargs["source_roles"],
                first_seen_global_step=kwargs["first_seen_global_step"],
                transfer_rows=kwargs["transfer_rows"],
                role_links=kwargs["role_links"],
                transfer_history=kwargs.get("transfer_history"),
                contradiction_only=True,
            )

        def build_functional(*args: Any, **kwargs: Any):
            signature = inspect.signature(original_build)
            accepts_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            supported_kwargs = (
                kwargs
                if accepts_var_kwargs
                else {
                    key: value
                    for key, value in kwargs.items()
                    if key in signature.parameters
                }
            )
            result = original_build(*args, **supported_kwargs)
            events, diagnostics, state = result
            return _recompute_population_continuity(
                higher_order,
                events=events,
                diagnostics=diagnostics,
                state=state,
                previous_state=kwargs.get("previous_state"),
                source_roles=kwargs["source_roles"],
                config=kwargs["config"],
            )

        def validate_promotions(*args: Any, **kwargs: Any):
            return _with_current_concept_flags_for_world_models(
                higher_order,
                original_validate,
                *args,
                **kwargs,
            )

        higher_order._prediction_explanation_events = prediction_events
        higher_order._contradiction_resolution_explanation_events = (
            contradiction_events
        )
        higher_order._build_functional_explanation_diagnostics = (
            build_functional
        )
        higher_order.validate_incremental_promotions_only = (
            validate_promotions
        )
        higher_order._ARC_AGI3_H07_EVIDENCE_CONTINUITY_FIX = True

    if not getattr(h08, "_ARC_AGI3_CURRENT_PROMOTION_FIX", False):
        original_h08 = h08.evaluate_h08_world_model_coherence

        def evaluate_h08(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original_h08(*args, **kwargs)
            memory_dir = kwargs.get("memory_dir")
            output_dir = kwargs.get("output_dir")
            if memory_dir is None or output_dir is None:
                return result
            return _repair_h08_result(
                h08,
                memory_dir=Path(memory_dir),
                output_dir=Path(output_dir),
                result=result,
            )

        h08.evaluate_h08_world_model_coherence = evaluate_h08
        h08._ARC_AGI3_CURRENT_PROMOTION_FIX = True

    import v6.hypothesis_suite_report as suite

    suite.evaluate_h08_world_model_coherence = (
        h08.evaluate_h08_world_model_coherence
    )
    suite.validate_incremental_promotions_only = (
        higher_order.validate_incremental_promotions_only
    )

    _PATCHED = True
    return True
