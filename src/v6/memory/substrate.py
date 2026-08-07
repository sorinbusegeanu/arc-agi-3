from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Iterable

from v6.memory.migrations.v61 import migrate_connection


@dataclass(frozen=True)
class MemoryNode:
    node_id: str
    memory_level: str
    node_type: str
    canonical_key: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "v6.1"
    evidence_version: str = "v1"
    created_epoch: int | None = None
    updated_epoch: int | None = None
    status: str = "active"


@dataclass(frozen=True)
class MemoryEdge:
    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)
    edge_status: str = "accepted"
    edge_confidence: float | None = None
    edge_source: str | None = "runtime"
    specificity_score: float | None = None
    last_validated_epoch: int | None = None


@dataclass(frozen=True)
class MemoryEvidence:
    evidence_id: str
    target_node_id: str
    source_interaction_id: int | None
    evidence_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryScore:
    node_id: str
    isf_total: float | None = None
    prediction_lift: float | None = None
    transfer_score: float | None = None
    explanatory_reach: float | None = None
    compression_gain: float | None = None
    future_option_delta: float | None = None
    replay_priority: float | None = None
    retention_status: str | None = None
    memory_state: str | None = None
    stored_epoch: int | None = None
    last_replayed_epoch: int | None = None
    last_promoted_epoch: int | None = None
    retention_score: float | None = None
    forgetting_score: float | None = None
    compressed_into_id: str | None = None
    superseded_by_id: str | None = None
    forgetting_reason: str | None = None


@dataclass(frozen=True)
class MemoryPromotion:
    promotion_id: str
    source_node_id: str
    target_node_id: str
    promotion_type: str
    evidence_count: int
    promotion_score: float
    status: str
    source_memory_ids: tuple[str, ...] = ()
    compression_gain: float | None = None
    prediction_lift: float | None = None
    transfer_score: float | None = None
    explanatory_reach: float | None = None
    epoch: int | None = None
    global_step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryLifecycleEvent:
    lifecycle_event_id: str
    memory_id: str
    event_type: str
    reason: str | None = None
    previous_status: str | None = None
    new_status: str | None = None
    score: float | None = None
    epoch: int | None = None
    global_step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class MemorySubstrate:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        auto_commit: bool = True,
    ) -> None:
        self.connection = connection
        self.auto_commit = bool(auto_commit)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_nodes (
                node_id TEXT PRIMARY KEY,
                memory_level TEXT NOT NULL,
                node_type TEXT NOT NULL,
                canonical_key TEXT,
                support_count INTEGER DEFAULT 0,
                first_seen_step INTEGER,
                last_seen_step INTEGER,
                attrs_json TEXT,
                schema_version TEXT NOT NULL DEFAULT 'v6.1',
                evidence_version TEXT NOT NULL DEFAULT 'v1',
                created_epoch INTEGER,
                updated_epoch INTEGER,
                status TEXT NOT NULL DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS memory_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                support_count INTEGER DEFAULT 1,
                evidence_json TEXT,
                edge_status TEXT NOT NULL DEFAULT 'accepted',
                edge_confidence REAL,
                edge_source TEXT,
                specificity_score REAL,
                last_validated_epoch INTEGER,
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
                payload_json TEXT,
                source_memory_ids_json TEXT,
                compression_gain REAL,
                prediction_lift REAL,
                transfer_score REAL,
                explanatory_reach REAL,
                epoch INTEGER,
                global_step INTEGER
            );
            CREATE TABLE IF NOT EXISTS memory_versions (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_nodes_level_type
            ON memory_nodes(memory_level, node_type);
            CREATE INDEX IF NOT EXISTS idx_memory_edges_source
            ON memory_edges(source_node_id, edge_type);
            CREATE INDEX IF NOT EXISTS idx_memory_edges_target
            ON memory_edges(target_node_id, edge_type);
            CREATE INDEX IF NOT EXISTS idx_memory_scores_replay
            ON memory_scores(replay_priority);
            CREATE INDEX IF NOT EXISTS idx_memory_promotions_type
            ON memory_promotions(promotion_type);
            """
        )
        migrate_connection(self.connection)
        if self.auto_commit:
            self.connection.commit()

    def upsert_node(
        self,
        node: MemoryNode,
        *,
        step: int | None = None,
        support_increment: int = 1,
    ) -> None:
        attrs_json = json.dumps(_json_safe(node.attrs), sort_keys=True)
        self.connection.execute(
            """
            INSERT INTO memory_nodes (
                node_id, memory_level, node_type, canonical_key,
                support_count, first_seen_step, last_seen_step, attrs_json,
                schema_version, evidence_version, created_epoch,
                updated_epoch, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                memory_level=excluded.memory_level,
                node_type=excluded.node_type,
                canonical_key=COALESCE(
                    memory_nodes.canonical_key,
                    excluded.canonical_key
                ),
                support_count=COALESCE(memory_nodes.support_count, 0) + ?,
                first_seen_step=CASE
                    WHEN memory_nodes.first_seen_step IS NULL
                        THEN excluded.first_seen_step
                    WHEN excluded.first_seen_step IS NULL
                        THEN memory_nodes.first_seen_step
                    ELSE MIN(
                        memory_nodes.first_seen_step,
                        excluded.first_seen_step
                    )
                END,
                last_seen_step=CASE
                    WHEN memory_nodes.last_seen_step IS NULL
                        THEN excluded.last_seen_step
                    WHEN excluded.last_seen_step IS NULL
                        THEN memory_nodes.last_seen_step
                    ELSE MAX(
                        memory_nodes.last_seen_step,
                        excluded.last_seen_step
                    )
                END,
                attrs_json=excluded.attrs_json,
                schema_version=excluded.schema_version,
                evidence_version=excluded.evidence_version,
                created_epoch=COALESCE(
                    memory_nodes.created_epoch,
                    excluded.created_epoch
                ),
                updated_epoch=COALESCE(
                    excluded.updated_epoch,
                    memory_nodes.updated_epoch
                ),
                status=excluded.status
            """,
            (
                str(node.node_id),
                str(node.memory_level),
                str(node.node_type),
                None if node.canonical_key is None else str(node.canonical_key),
                int(max(0, support_increment)),
                None if step is None else int(step),
                None if step is None else int(step),
                attrs_json,
                str(node.schema_version or "v6.1"),
                str(node.evidence_version or "v1"),
                node.created_epoch,
                node.updated_epoch,
                str(node.status or "active"),
                int(max(0, support_increment)),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def update_node_support_and_attrs(
        self,
        node_id: str,
        attrs: dict[str, Any],
        *,
        support_increment: int = 0,
        step: int | None = None,
    ) -> None:
        existing = self.get_node(node_id)
        merged_attrs = dict((existing or {}).get("attrs", {}))
        merged_attrs.update(attrs)
        self.upsert_node(
            MemoryNode(
                node_id=str(node_id),
                memory_level=str(
                    (existing or {}).get("memory_level", "M0")
                ),
                node_type=str((existing or {}).get("node_type", "Unknown")),
                canonical_key=(
                    None if existing is None
                    else existing.get("canonical_key")
                ),
                attrs=merged_attrs,
                schema_version=str(
                    (existing or {}).get("schema_version", "v6.1")
                ),
                evidence_version=str(
                    (existing or {}).get("evidence_version", "v1")
                ),
                created_epoch=(existing or {}).get("created_epoch"),
                updated_epoch=(existing or {}).get("updated_epoch"),
                status=str((existing or {}).get("status", "active")),
            ),
            step=step,
            support_increment=support_increment,
        )

    def upsert_edge(
        self,
        edge: MemoryEdge,
        *,
        support_increment: int = 1,
    ) -> None:
        evidence_json = json.dumps(_json_safe(edge.evidence), sort_keys=True)
        confidence = (
            float(edge.weight)
            if edge.edge_confidence is None
            else float(edge.edge_confidence)
        )
        self.connection.execute(
            """
            INSERT INTO memory_edges (
                source_node_id, target_node_id, edge_type, weight,
                support_count, evidence_json, edge_status,
                edge_confidence, edge_source, specificity_score,
                last_validated_epoch
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_node_id, target_node_id, edge_type)
            DO UPDATE SET
                weight=MAX(memory_edges.weight, excluded.weight),
                support_count=COALESCE(memory_edges.support_count, 0) + ?,
                evidence_json=excluded.evidence_json,
                edge_status=excluded.edge_status,
                edge_confidence=MAX(
                    COALESCE(memory_edges.edge_confidence, 0.0),
                    COALESCE(excluded.edge_confidence, 0.0)
                ),
                edge_source=COALESCE(
                    excluded.edge_source,
                    memory_edges.edge_source
                ),
                specificity_score=COALESCE(
                    excluded.specificity_score,
                    memory_edges.specificity_score
                ),
                last_validated_epoch=COALESCE(
                    excluded.last_validated_epoch,
                    memory_edges.last_validated_epoch
                )
            """,
            (
                str(edge.source_node_id),
                str(edge.target_node_id),
                str(edge.edge_type),
                float(edge.weight),
                int(max(0, support_increment)),
                evidence_json,
                str(edge.edge_status or "accepted"),
                confidence,
                None if edge.edge_source is None else str(edge.edge_source),
                _float_or_none(edge.specificity_score),
                edge.last_validated_epoch,
                int(max(0, support_increment)),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def validate_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        *,
        accepted: bool,
        confidence: float,
        specificity_score: float | None = None,
        epoch: int | None = None,
        source: str = "validation",
    ) -> None:
        self.connection.execute(
            """
            UPDATE memory_edges
            SET
                edge_status=?,
                edge_confidence=?,
                specificity_score=COALESCE(?, specificity_score),
                last_validated_epoch=COALESCE(?, last_validated_epoch),
                edge_source=?
            WHERE source_node_id=? AND target_node_id=? AND edge_type=?
            """,
            (
                "accepted" if accepted else "rejected",
                float(confidence),
                _float_or_none(specificity_score),
                epoch,
                str(source),
                str(source_node_id),
                str(target_node_id),
                str(edge_type),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def add_evidence(self, evidence: MemoryEvidence) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO memory_evidence (
                evidence_id, target_node_id, source_interaction_id,
                evidence_type, payload_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(evidence.evidence_id),
                str(evidence.target_node_id),
                (
                    None if evidence.source_interaction_id is None
                    else int(evidence.source_interaction_id)
                ),
                str(evidence.evidence_type),
                json.dumps(_json_safe(evidence.payload), sort_keys=True),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def upsert_score(
        self,
        score: MemoryScore,
        *,
        step: int | None = None,
    ) -> None:
        previous = self.connection.execute(
            """
            SELECT memory_state, retention_status
            FROM memory_scores WHERE node_id=?
            """,
            (str(score.node_id),),
        ).fetchone()
        self.connection.execute(
            """
            INSERT INTO memory_scores (
                node_id, isf_total, prediction_lift, transfer_score,
                explanatory_reach, compression_gain, future_option_delta,
                replay_priority, retention_status, memory_state,
                stored_epoch, last_replayed_epoch, last_promoted_epoch,
                retention_score, forgetting_score, compressed_into_id,
                superseded_by_id, forgetting_reason, updated_step
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                isf_total=COALESCE(
                    excluded.isf_total, memory_scores.isf_total
                ),
                prediction_lift=COALESCE(
                    excluded.prediction_lift,
                    memory_scores.prediction_lift
                ),
                transfer_score=COALESCE(
                    excluded.transfer_score,
                    memory_scores.transfer_score
                ),
                explanatory_reach=COALESCE(
                    excluded.explanatory_reach,
                    memory_scores.explanatory_reach
                ),
                compression_gain=COALESCE(
                    excluded.compression_gain,
                    memory_scores.compression_gain
                ),
                future_option_delta=COALESCE(
                    excluded.future_option_delta,
                    memory_scores.future_option_delta
                ),
                replay_priority=COALESCE(
                    excluded.replay_priority,
                    memory_scores.replay_priority
                ),
                retention_status=COALESCE(
                    excluded.retention_status,
                    memory_scores.retention_status
                ),
                memory_state=COALESCE(
                    excluded.memory_state,
                    memory_scores.memory_state
                ),
                stored_epoch=COALESCE(
                    excluded.stored_epoch,
                    memory_scores.stored_epoch
                ),
                last_replayed_epoch=COALESCE(
                    excluded.last_replayed_epoch,
                    memory_scores.last_replayed_epoch
                ),
                last_promoted_epoch=COALESCE(
                    excluded.last_promoted_epoch,
                    memory_scores.last_promoted_epoch
                ),
                retention_score=COALESCE(
                    excluded.retention_score,
                    memory_scores.retention_score
                ),
                forgetting_score=COALESCE(
                    excluded.forgetting_score,
                    memory_scores.forgetting_score
                ),
                compressed_into_id=COALESCE(
                    excluded.compressed_into_id,
                    memory_scores.compressed_into_id
                ),
                superseded_by_id=COALESCE(
                    excluded.superseded_by_id,
                    memory_scores.superseded_by_id
                ),
                forgetting_reason=COALESCE(
                    excluded.forgetting_reason,
                    memory_scores.forgetting_reason
                ),
                updated_step=COALESCE(
                    excluded.updated_step,
                    memory_scores.updated_step
                )
            """,
            (
                str(score.node_id),
                _float_or_none(score.isf_total),
                _float_or_none(score.prediction_lift),
                _float_or_none(score.transfer_score),
                _float_or_none(score.explanatory_reach),
                _float_or_none(score.compression_gain),
                _float_or_none(score.future_option_delta),
                _float_or_none(score.replay_priority),
                score.retention_status,
                score.memory_state,
                score.stored_epoch,
                score.last_replayed_epoch,
                score.last_promoted_epoch,
                _float_or_none(score.retention_score),
                _float_or_none(score.forgetting_score),
                score.compressed_into_id,
                score.superseded_by_id,
                score.forgetting_reason,
                None if step is None else int(step),
            ),
        )

        previous_status = None
        if previous is not None:
            previous_status = str(previous[0] or previous[1] or "")
        new_status = str(
            score.memory_state or score.retention_status or previous_status or ""
        )
        if new_status and new_status != previous_status:
            self.record_lifecycle_event(
                MemoryLifecycleEvent(
                    lifecycle_event_id=(
                        f"lifecycle:{_short_hash(
                            f'{score.node_id}|{previous_status}|{new_status}|{step}'
                        )}"
                    ),
                    memory_id=str(score.node_id),
                    event_type=_lifecycle_event_type(new_status),
                    reason=score.forgetting_reason,
                    previous_status=previous_status or None,
                    new_status=new_status,
                    score=score.retention_score,
                    epoch=score.stored_epoch,
                    global_step=step,
                )
            )
        if self.auto_commit:
            self.connection.commit()

    def record_promotion(self, promotion: MemoryPromotion) -> None:
        source_ids = tuple(
            dict.fromkeys(
                (
                    *promotion.source_memory_ids,
                    str(promotion.source_node_id),
                )
            )
        )
        payload = {
            "source_node_id": str(promotion.source_node_id),
            "target_node_id": str(promotion.target_node_id),
            **dict(promotion.payload),
        }
        self.connection.execute(
            """
            INSERT OR REPLACE INTO memory_promotions (
                promotion_id, source_node_id, target_node_id,
                promotion_type, evidence_count, promotion_score,
                status, payload_json, source_memory_ids_json,
                compression_gain, prediction_lift, transfer_score,
                explanatory_reach, epoch, global_step
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(promotion.promotion_id),
                str(promotion.source_node_id),
                str(promotion.target_node_id),
                str(promotion.promotion_type),
                int(promotion.evidence_count),
                float(promotion.promotion_score),
                str(promotion.status),
                json.dumps(_json_safe(payload), sort_keys=True),
                json.dumps(list(source_ids), sort_keys=True),
                _float_or_none(promotion.compression_gain),
                _float_or_none(promotion.prediction_lift),
                _float_or_none(promotion.transfer_score),
                _float_or_none(promotion.explanatory_reach),
                promotion.epoch,
                promotion.global_step,
            ),
        )
        self.record_lifecycle_event(
            MemoryLifecycleEvent(
                lifecycle_event_id=(
                    f"lifecycle:promotion:{promotion.promotion_id}"
                ),
                memory_id=str(promotion.target_node_id),
                event_type="promoted",
                reason=str(promotion.promotion_type),
                new_status=str(promotion.status),
                score=float(promotion.promotion_score),
                epoch=promotion.epoch,
                global_step=promotion.global_step,
                payload=payload,
            )
        )
        if self.auto_commit:
            self.connection.commit()

    def record_lifecycle_event(
        self,
        event: MemoryLifecycleEvent,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO memory_lifecycle_events (
                lifecycle_event_id, memory_id, event_type, reason,
                previous_status, new_status, score, epoch, global_step,
                payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event.lifecycle_event_id),
                str(event.memory_id),
                str(event.event_type),
                event.reason,
                event.previous_status,
                event.new_status,
                _float_or_none(event.score),
                event.epoch,
                event.global_step,
                json.dumps(_json_safe(event.payload), sort_keys=True),
                time.time(),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def record_context_split(
        self,
        *,
        split_id: str,
        parent_context_key: str,
        child_context_key: str,
        action_key: str | None,
        contradiction_key: str | None,
        differentiating_features: list[Any],
        prediction_lift_before: float | None = None,
        prediction_lift_after: float | None = None,
        validation_status: str = "candidate",
        epoch: int | None = None,
        global_step: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO context_split_lineage (
                split_id, parent_context_key, child_context_key,
                action_key, contradiction_key,
                differentiating_features_json,
                prediction_lift_before, prediction_lift_after,
                validation_status, epoch, global_step, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(split_id),
                str(parent_context_key),
                str(child_context_key),
                action_key,
                contradiction_key,
                json.dumps(
                    _json_safe(differentiating_features),
                    sort_keys=True,
                ),
                _float_or_none(prediction_lift_before),
                _float_or_none(prediction_lift_after),
                str(validation_status),
                epoch,
                global_step,
                json.dumps(_json_safe(payload or {}), sort_keys=True),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def record_strategy_reuse(
        self,
        *,
        reuse_event_id: str,
        strategy_id: str,
        game: str | None,
        level_key: str | None,
        context_key: str | None,
        success: bool,
        cost: float | None,
        epoch: int | None,
        global_step: int | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO strategy_reuse_events (
                reuse_event_id, strategy_id, game, level_key,
                context_key, success, cost, epoch, global_step,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(reuse_event_id),
                str(strategy_id),
                game,
                level_key,
                context_key,
                int(bool(success)),
                _float_or_none(cost),
                epoch,
                global_step,
                json.dumps(_json_safe(payload or {}), sort_keys=True),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT
                node_id, memory_level, node_type, canonical_key,
                support_count, first_seen_step, last_seen_step,
                attrs_json, schema_version, evidence_version,
                created_epoch, updated_epoch, status
            FROM memory_nodes
            WHERE node_id=?
            """,
            (str(node_id),),
        ).fetchone()
        return None if row is None else _node_row_to_dict(row)

    def edges_from(
        self,
        node_id: str,
        edge_type: str | None = None,
        *,
        accepted_only: bool = False,
    ) -> list[dict[str, Any]]:
        return self._edges(
            "source_node_id",
            node_id,
            edge_type,
            accepted_only=accepted_only,
        )

    def edges_to(
        self,
        node_id: str,
        edge_type: str | None = None,
        *,
        accepted_only: bool = False,
    ) -> list[dict[str, Any]]:
        return self._edges(
            "target_node_id",
            node_id,
            edge_type,
            accepted_only=accepted_only,
        )

    def _edges(
        self,
        key: str,
        node_id: str,
        edge_type: str | None,
        *,
        accepted_only: bool,
    ) -> list[dict[str, Any]]:
        query = f"""
            SELECT
                source_node_id, target_node_id, edge_type, weight,
                support_count, evidence_json, edge_status,
                edge_confidence, edge_source, specificity_score,
                last_validated_epoch
            FROM memory_edges
            WHERE {key}=?
        """
        params: list[Any] = [str(node_id)]
        if edge_type is not None:
            query += " AND edge_type=?"
            params.append(str(edge_type))
        if accepted_only:
            query += " AND edge_status='accepted'"
        query += " ORDER BY source_node_id, target_node_id, edge_type"
        return [
            _edge_row_to_dict(row)
            for row in self.connection.execute(
                query, tuple(params)
            ).fetchall()
        ]

    def query_nodes(
        self,
        memory_level: str | None = None,
        node_type: str | None = None,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                node_id, memory_level, node_type, canonical_key,
                support_count, first_seen_step, last_seen_step,
                attrs_json, schema_version, evidence_version,
                created_epoch, updated_epoch, status
            FROM memory_nodes
            WHERE 1=1
        """
        params: list[Any] = []
        if memory_level is not None:
            query += " AND memory_level=?"
            params.append(str(memory_level))
        if node_type is not None:
            query += " AND node_type=?"
            params.append(str(node_type))
        if status is not None:
            query += " AND status=?"
            params.append(str(status))
        query += " ORDER BY node_id"
        return [
            _node_row_to_dict(row)
            for row in self.connection.execute(
                query, tuple(params)
            ).fetchall()
        ]


def interaction_node_id(identifier: int | str) -> str:
    value = str(identifier)
    return (
        value if value.startswith("M0:interaction:")
        else f"M0:interaction:{value}"
    )


def scoped_interaction_key(
    *,
    interaction_id: int | str,
    global_step: int | None = None,
    game: str | None = None,
    sampler: str | None = None,
    seed: int | str | None = None,
) -> str:
    if global_step is not None:
        return f"g{int(global_step)}"
    return ":".join(
        (
            "local",
            str(game or "unknown_game"),
            str(sampler or "unknown_sampler"),
            str(seed or "unknown_seed"),
            str(interaction_id),
        )
    )


def observation_node_id(signature: str) -> str:
    return f"M0:observation:{_short_hash(signature)}"


def action_node_id(action: int | str) -> str:
    return f"M0:action:{action}"


def delta_node_id(delta_id: int | str) -> str:
    return f"M0:delta:{delta_id}"


def contingency_node_id(
    context_level: int | str,
    context_signature: str,
    action: int | str,
    family: int | str,
) -> str:
    return (
        "M1:contingency:"
        + _short_hash(
            f"{context_level}|{context_signature}|{action}|{family}"
        )
    )


def family_node_id(family_id: int | str) -> str:
    return f"M2:family:{family_id}"


def carrier_node_id(signature: str) -> str:
    return f"M3:carrier:{_short_hash(signature)}"


def role_node_id(role_signature: str) -> str:
    return f"M3:role:{_short_hash(role_signature)}"


def concept_node_id(concept_signature: str) -> str:
    return f"M4:concept:{_short_hash(concept_signature)}"


def world_model_node_id(signature: str) -> str:
    return f"M5:world_model:{_short_hash(signature)}"


def strategy_node_id(signature: str) -> str:
    return f"M6:strategy:{_short_hash(signature)}"


def trajectory_node_id(episode_id: int | str) -> str:
    return f"M0:trajectory:{episode_id}"


def _short_hash(value: str) -> str:
    return sha1(str(value).encode("utf-8")).hexdigest()[:20]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def _lifecycle_event_type(status: str) -> str:
    normalized = str(status).lower()
    mapping = {
        "active": "retained",
        "protected": "retained",
        "compressed": "compressed",
        "forgotten": "forgotten",
        "demoted": "demoted",
        "superseded": "superseded",
        "reactivated": "reactivated",
        "merged": "merged",
    }
    return mapping.get(normalized, "status_changed")


def _node_row_to_dict(
    row: sqlite3.Row | tuple[Any, ...],
) -> dict[str, Any]:
    values = list(row)
    try:
        attrs = json.loads(str(values[7])) if values[7] else {}
    except Exception:
        attrs = {}
    return {
        "node_id": str(values[0]),
        "memory_level": str(values[1]),
        "node_type": str(values[2]),
        "canonical_key": values[3],
        "support_count": int(values[4] or 0),
        "first_seen_step": values[5],
        "last_seen_step": values[6],
        "attrs": attrs,
        "schema_version": str(values[8] or "v6.1"),
        "evidence_version": str(values[9] or "v1"),
        "created_epoch": values[10],
        "updated_epoch": values[11],
        "status": str(values[12] or "active"),
    }


def _edge_row_to_dict(
    row: sqlite3.Row | tuple[Any, ...],
) -> dict[str, Any]:
    values = list(row)
    try:
        evidence = json.loads(str(values[5])) if values[5] else {}
    except Exception:
        evidence = {}
    return {
        "source_node_id": str(values[0]),
        "target_node_id": str(values[1]),
        "edge_type": str(values[2]),
        "weight": float(values[3] or 1.0),
        "support_count": int(values[4] or 0),
        "evidence": evidence,
        "edge_status": str(values[6] or "accepted"),
        "edge_confidence": (
            None if values[7] is None else float(values[7])
        ),
        "edge_source": values[8],
        "specificity_score": (
            None if values[9] is None else float(values[9])
        ),
        "last_validated_epoch": values[10],
    }
