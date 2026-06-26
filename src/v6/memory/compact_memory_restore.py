from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from v6.carrier_emergence import CarrierEmergenceTracker
from v6.contingency.contingency_learner import Contingency
from v6.memory.compact_memory import stable_family_int_id
from v6.memory_lifecycle import MemoryRecord, ReplayCandidate
from v6.memory.substrate import MemoryEdge, MemoryEvidence, MemoryNode, MemoryPromotion, MemoryScore
from v6.transformation.transformation_clusterer import TransformationFamily


READONLY_RETRY_ATTEMPTS = 5
READONLY_RETRY_BASE_DELAY_SECONDS = 0.05
READONLY_CONNECT_TIMEOUT_SECONDS = 1.0


def load_compact_memory_into_system(system: Any, memory_dir: Path) -> dict[str, Any]:
    paths = {
        "current_state": Path(memory_dir) / "current_state.sqlite",
        "graph": Path(memory_dir) / "graph.sqlite",
        "replay_queue": Path(memory_dir) / "replay_queue.sqlite",
        "summary_json": Path(memory_dir) / "memory_summary.json",
    }
    summary = {
        "stable_contingencies_restored": 0,
        "transformation_families_restored": 0,
        "family_members_restored": 0,
        "carrier_candidates_restored": 0,
        "replay_candidates_restored": 0,
        "graph_nodes_restored": 0,
        "graph_edges_restored": 0,
        "memory_nodes_restored": 0,
        "memory_edges_restored": 0,
        "memory_evidence_restored": 0,
        "memory_scores_restored": 0,
        "memory_promotions_restored": 0,
        "memory_summary_loaded": False,
        "restore_warnings": [],
    }
    if not paths["current_state"].exists():
        summary["restore_warnings"].append(f"missing current_state sqlite: {paths['current_state']}")
        return summary

    with _connect_readonly_with_retry(paths["current_state"]) as state_conn:
        state_conn.row_factory = sqlite3.Row
        family_id_map = {
            str(row["canonical_signature"]): int(row["stable_family_id"])
            for row in state_conn.execute("SELECT canonical_signature, stable_family_id FROM family_identity_map").fetchall()
        }
        contingencies = []
        for row in state_conn.execute(
            """
            SELECT canonical_key, context_level, action, effect_signature, support_count, stability_score
            FROM stable_contingencies
            """
        ).fetchall():
            context_signature = _context_signature_from_canonical_key(str(row["canonical_key"]))
            transformation_family = int(family_id_map.get(str(row["effect_signature"]), stable_family_int_id(str(row["effect_signature"]))))
            contingencies.append(
                Contingency(
                    id=max(1, summary["stable_contingencies_restored"] + 1),
                    context_level=int(row["context_level"] or 0),
                    context_signature=context_signature,
                    action=int(row["action"]),
                    transformation_family=transformation_family,
                    support_count=int(row["support_count"] or 0),
                    confidence=max(0.0, min(1.0, float(row["stability_score"] or 0.0))),
                )
            )
        system.contingency_learner.import_contingencies(contingencies)
        summary["stable_contingencies_restored"] = len(contingencies)

        family_member_rows = state_conn.execute(
            """
            SELECT family_signature, contingency_key, support_count
            FROM family_members
            """
        ).fetchall()
        summary["family_members_restored"] = len(family_member_rows)
        family_columns = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in state_conn.execute("PRAGMA table_info(transformation_families)").fetchall()}
        family_rows = state_conn.execute("SELECT * FROM transformation_families").fetchall()
        for row in family_rows:
            signature = str(
                row["canonical_signature"]
                if "canonical_signature" in family_columns
                else row["family_id"]
            )
            stable_family_id = int(
                family_id_map.get(
                    signature,
                    row["family_id"] if "family_id" in family_columns and row["family_id"] is not None else stable_family_int_id(signature),
                )
            )
            system.clusterer.import_family(
                TransformationFamily(
                    id=stable_family_id,
                    centroid_vector=_centroid_from_signature(signature),
                    support_count=int(row["support_count"] if "support_count" in family_columns and row["support_count"] is not None else 0),
                    member_delta_ids=[],
                )
            )
        for row in family_member_rows:
            system.clusterer.import_family_member(str(row["family_signature"]), str(row["contingency_key"]))
        summary["transformation_families_restored"] = len(family_rows)

        if hasattr(system, "carrier_tracker") and isinstance(system.carrier_tracker, CarrierEmergenceTracker):
            for row in state_conn.execute(
                """
                SELECT carrier_signature, carrier_source, support_count, linked_family_count,
                       first_seen_global_step, last_seen_global_step, stability_score, is_emergent
                FROM carrier_candidates
                """
            ).fetchall():
                system.carrier_tracker.import_candidate(
                    carrier_signature=str(row["carrier_signature"]),
                    carrier_source=str(row["carrier_source"] or "unknown"),
                    support_count=int(row["support_count"] or 0),
                    linked_family_count=int(row["linked_family_count"] or 0),
                    first_seen_global_step=row["first_seen_global_step"],
                    last_seen_global_step=row["last_seen_global_step"],
                    stability_score=float(row["stability_score"] or 0.0),
                    is_emergent=bool(row["is_emergent"]),
                )
                summary["carrier_candidates_restored"] += 1

        for key, value_json in state_conn.execute("SELECT key, value_json FROM memory_summary").fetchall():
            try:
                system._isf_counts[str(key)] = int(json.loads(value_json))
            except Exception:
                continue
        if hasattr(system, "memory"):
            for row in state_conn.execute(
                """
                SELECT node_id, memory_level, node_type, canonical_key, attrs_json, first_seen_step, support_count
                FROM memory_nodes
                ORDER BY node_id ASC
                """
            ).fetchall():
                system.memory.upsert_node(
                    MemoryNode(
                        node_id=str(row["node_id"]),
                        memory_level=str(row["memory_level"]),
                        node_type=str(row["node_type"]),
                        canonical_key=None if row["canonical_key"] is None else str(row["canonical_key"]),
                        attrs=_parse_json(row["attrs_json"]),
                    ),
                    step=row["first_seen_step"],
                    support_increment=max(1, int(row["support_count"] or 1)),
                )
                summary["memory_nodes_restored"] += 1
            for row in state_conn.execute(
                """
                SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                FROM memory_edges
                ORDER BY source_node_id ASC, target_node_id ASC, edge_type ASC
                """
            ).fetchall():
                system.memory.upsert_edge(
                    MemoryEdge(
                        source_node_id=str(row["source_node_id"]),
                        target_node_id=str(row["target_node_id"]),
                        edge_type=str(row["edge_type"]),
                        weight=float(row["weight"] or 1.0),
                        evidence=_parse_json(row["evidence_json"]),
                    ),
                    support_increment=max(1, int(row["support_count"] or 1)),
                )
                summary["memory_edges_restored"] += 1
            for row in state_conn.execute(
                """
                SELECT node_id, isf_total, prediction_lift, transfer_score, explanatory_reach,
                       compression_gain, future_option_delta, replay_priority, retention_status, updated_step
                FROM memory_scores
                ORDER BY node_id ASC
                """
            ).fetchall():
                system.memory.upsert_score(
                    MemoryScore(
                        node_id=str(row["node_id"]),
                        isf_total=None if row["isf_total"] is None else float(row["isf_total"]),
                        prediction_lift=None if row["prediction_lift"] is None else float(row["prediction_lift"]),
                        transfer_score=None if row["transfer_score"] is None else float(row["transfer_score"]),
                        explanatory_reach=None if row["explanatory_reach"] is None else float(row["explanatory_reach"]),
                        compression_gain=None if row["compression_gain"] is None else float(row["compression_gain"]),
                        future_option_delta=None if row["future_option_delta"] is None else float(row["future_option_delta"]),
                        replay_priority=None if row["replay_priority"] is None else float(row["replay_priority"]),
                        retention_status=None if row["retention_status"] is None else str(row["retention_status"]),
                    ),
                    step=row["updated_step"],
                )
                summary["memory_scores_restored"] += 1
            for row in state_conn.execute(
                """
                SELECT evidence_id, target_node_id, source_interaction_id, evidence_type, payload_json
                FROM memory_evidence
                ORDER BY evidence_id ASC
                """
            ).fetchall():
                system.memory.add_evidence(
                    MemoryEvidence(
                        evidence_id=str(row["evidence_id"]),
                        target_node_id=str(row["target_node_id"]),
                        source_interaction_id=None if row["source_interaction_id"] is None else int(row["source_interaction_id"]),
                        evidence_type=str(row["evidence_type"]),
                        payload=_parse_json(row["payload_json"]),
                    )
                )
                summary["memory_evidence_restored"] += 1
            for row in state_conn.execute(
                """
                SELECT promotion_id, source_node_id, target_node_id, promotion_type,
                       evidence_count, promotion_score, status
                FROM memory_promotions
                ORDER BY promotion_id ASC
                """
            ).fetchall():
                system.memory.record_promotion(
                    MemoryPromotion(
                        promotion_id=str(row["promotion_id"]),
                        source_node_id=str(row["source_node_id"]),
                        target_node_id=str(row["target_node_id"]),
                        promotion_type=str(row["promotion_type"]),
                        evidence_count=int(row["evidence_count"] or 0),
                        promotion_score=float(row["promotion_score"] or 0.0),
                        status=str(row["status"]),
                    )
                )
                summary["memory_promotions_restored"] += 1

    if paths["replay_queue"].exists():
        with _connect_readonly_with_retry(paths["replay_queue"]) as replay_conn:
            replay_conn.row_factory = sqlite3.Row
            for row in replay_conn.execute(
                """
                SELECT replay_id, owner_type, owner_id, priority_score, reason,
                       first_seen_global_step, last_seen_global_step, compact_payload_json
                FROM replay_queue
                """
            ).fetchall():
                payload = _parse_json(row["compact_payload_json"])
                interaction_id = str(row["replay_id"])
                record = MemoryRecord(
                    interaction_id=interaction_id,
                    family_id=None if payload.get("family_id") is None else str(payload.get("family_id")),
                    context_signature=None if payload.get("context_signature") is None else str(payload.get("context_signature")),
                    action_signature=None if payload.get("action_signature") is None else str(payload.get("action_signature")),
                    carrier_signature=None if payload.get("carrier_signature") is None else str(payload.get("carrier_signature")),
                    isf_total=float(payload.get("isf_total", row["priority_score"]) or 0.0),
                    prediction_error=float(payload.get("prediction_error", row["priority_score"]) or 0.0),
                    learning_value=float(payload.get("learning_value", 0.0) or 0.0),
                    transfer_potential=float(payload.get("transfer_potential", 0.0) or 0.0),
                    explanatory_potential=float(payload.get("explanatory_potential", 0.0) or 0.0),
                    context_contradiction=bool(payload.get("context_contradiction", False)),
                    timestamp_step=int(row["last_seen_global_step"] or 0),
                    replay_count=int(payload.get("replay_count", 0) or 0),
                    status="protected" if float(row["priority_score"] or 0.0) >= 0.8 else str(payload.get("status", "active")),
                    retention_reason=str(row["reason"]),
                )
                system.memory_lifecycle.import_record(record)
                system.memory_lifecycle.import_replay_candidate(
                    ReplayCandidate(
                        interaction_id=interaction_id,
                        replay_priority=float(row["priority_score"] or 0.0),
                        reason=str(row["reason"]),
                        family_id=record.family_id,
                        context_signature=record.context_signature,
                        status=record.status,
                    )
                )
                summary["replay_candidates_restored"] += 1

    if paths["graph"].exists():
        with _connect_readonly_with_retry(paths["graph"]) as graph_conn:
            graph_conn.row_factory = sqlite3.Row
            for row in graph_conn.execute(
                """
                SELECT node_id, node_type, canonical_key, support_count
                FROM graph_nodes
                """
            ).fetchall():
                system.graph.import_node(
                    str(row["node_id"]),
                    str(row["node_type"] or "Unknown"),
                    None if row["canonical_key"] is None else str(row["canonical_key"]),
                    attrs={"support_count": int(row["support_count"] or 0)},
                )
                summary["graph_nodes_restored"] += 1
            for row in graph_conn.execute(
                """
                SELECT source_node_id, target_node_id, edge_type, support_count, weight
                FROM graph_edges
                """
            ).fetchall():
                system.graph.import_edge(
                    str(row["source_node_id"]),
                    str(row["target_node_id"]),
                    str(row["edge_type"]),
                    weight=float(row["weight"] or 1.0),
                    support_count=int(row["support_count"] or 0),
                )
                summary["graph_edges_restored"] += 1

    if paths["summary_json"].exists():
        summary["memory_summary_loaded"] = True
    return summary


def _connect_readonly_with_retry(
    path: Path,
    *,
    attempts: int = READONLY_RETRY_ATTEMPTS,
    base_delay_seconds: float = READONLY_RETRY_BASE_DELAY_SECONDS,
    timeout_seconds: float = READONLY_CONNECT_TIMEOUT_SECONDS,
) -> sqlite3.Connection:
    db_uri = f"{path.resolve().as_uri()}?mode=ro"
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            connection = sqlite3.connect(
                db_uri,
                uri=True,
                timeout=float(timeout_seconds),
            )
            connection.execute(f"PRAGMA busy_timeout = {int(max(0.0, float(timeout_seconds)) * 1000)}")
            return connection
        except sqlite3.OperationalError as exc:
            last_error = exc
            if "locked" not in str(exc).lower() or attempt >= max(1, int(attempts)) - 1:
                raise
            time.sleep(float(base_delay_seconds) * (2 ** attempt))
    if last_error is not None:
        raise last_error
    raise sqlite3.OperationalError(f"failed to open readonly sqlite database: {path}")


def _parse_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def _context_signature_from_canonical_key(value: str) -> tuple:
    context, _sep, _rest = value.partition("|a")
    try:
        loaded = json.loads(context)
    except Exception:
        return (context,)
    if isinstance(loaded, list):
        return tuple(loaded)
    return (loaded,)


def _centroid_from_signature(signature: str) -> np.ndarray:
    if signature.startswith("centroid:"):
        raw = signature.split(":", 1)[1]
        try:
            data = json.loads(raw)
            return np.asarray(data, dtype=float)
        except Exception:
            pass
    return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
