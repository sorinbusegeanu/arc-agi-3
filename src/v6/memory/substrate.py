from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any


@dataclass(frozen=True)
class MemoryNode:
    node_id: str
    memory_level: str
    node_type: str
    canonical_key: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEdge:
    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class MemoryPromotion:
    promotion_id: str
    source_node_id: str
    target_node_id: str
    promotion_type: str
    evidence_count: int
    promotion_score: float
    status: str


class MemorySubstrate:
    def __init__(self, connection: sqlite3.Connection, *, auto_commit: bool = True) -> None:
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
            CREATE TABLE IF NOT EXISTS memory_versions (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_nodes_level_type
            ON memory_nodes(memory_level, node_type);
            CREATE INDEX IF NOT EXISTS idx_memory_nodes_type
            ON memory_nodes(node_type);
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
        self.connection.execute(
            """
            INSERT INTO memory_versions (key, value)
            VALUES ('memory_substrate_schema', 'phase1')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )
        if self.auto_commit:
            self.connection.commit()

    def upsert_node(self, node: MemoryNode, *, step: int | None = None, support_increment: int = 1) -> None:
        attrs_json = json.dumps(_json_safe(node.attrs), sort_keys=True)
        self.connection.execute(
            """
            INSERT INTO memory_nodes (
                node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                memory_level = excluded.memory_level,
                node_type = excluded.node_type,
                canonical_key = COALESCE(memory_nodes.canonical_key, excluded.canonical_key),
                support_count = COALESCE(memory_nodes.support_count, 0) + ?,
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
                str(node.node_id),
                str(node.memory_level),
                str(node.node_type),
                None if node.canonical_key is None else str(node.canonical_key),
                int(max(0, support_increment)),
                None if step is None else int(step),
                None if step is None else int(step),
                attrs_json,
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
        memory_level = str((existing or {}).get("memory_level", "M0"))
        node_type = str((existing or {}).get("node_type", "Unknown"))
        canonical_key = None if existing is None else existing.get("canonical_key")
        self.upsert_node(
            MemoryNode(
                node_id=str(node_id),
                memory_level=memory_level,
                node_type=node_type,
                canonical_key=canonical_key,
                attrs=merged_attrs,
            ),
            step=step,
            support_increment=support_increment,
        )

    def upsert_edge(self, edge: MemoryEdge, *, support_increment: int = 1) -> None:
        evidence_json = json.dumps(_json_safe(edge.evidence), sort_keys=True)
        self.connection.execute(
            """
            INSERT INTO memory_edges (
                source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_node_id, target_node_id, edge_type) DO UPDATE SET
                weight = MAX(memory_edges.weight, excluded.weight),
                support_count = COALESCE(memory_edges.support_count, 0) + ?,
                evidence_json = excluded.evidence_json
            """,
            (
                str(edge.source_node_id),
                str(edge.target_node_id),
                str(edge.edge_type),
                float(edge.weight),
                int(max(0, support_increment)),
                evidence_json,
                int(max(0, support_increment)),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def add_evidence(self, evidence: MemoryEvidence) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO memory_evidence (
                evidence_id, target_node_id, source_interaction_id, evidence_type, payload_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(evidence.evidence_id),
                str(evidence.target_node_id),
                None if evidence.source_interaction_id is None else int(evidence.source_interaction_id),
                str(evidence.evidence_type),
                json.dumps(_json_safe(evidence.payload), sort_keys=True),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def upsert_score(self, score: MemoryScore, *, step: int | None = None) -> None:
        self.connection.execute(
            """
            INSERT INTO memory_scores (
                node_id, isf_total, prediction_lift, transfer_score, explanatory_reach,
                compression_gain, future_option_delta, replay_priority, retention_status, updated_step
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                isf_total = COALESCE(excluded.isf_total, memory_scores.isf_total),
                prediction_lift = COALESCE(excluded.prediction_lift, memory_scores.prediction_lift),
                transfer_score = COALESCE(excluded.transfer_score, memory_scores.transfer_score),
                explanatory_reach = COALESCE(excluded.explanatory_reach, memory_scores.explanatory_reach),
                compression_gain = COALESCE(excluded.compression_gain, memory_scores.compression_gain),
                future_option_delta = COALESCE(excluded.future_option_delta, memory_scores.future_option_delta),
                replay_priority = COALESCE(excluded.replay_priority, memory_scores.replay_priority),
                retention_status = COALESCE(excluded.retention_status, memory_scores.retention_status),
                updated_step = COALESCE(excluded.updated_step, memory_scores.updated_step)
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
                None if step is None else int(step),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def record_promotion(self, promotion: MemoryPromotion) -> None:
        payload = {
            "source_node_id": str(promotion.source_node_id),
            "target_node_id": str(promotion.target_node_id),
        }
        self.connection.execute(
            """
            INSERT OR REPLACE INTO memory_promotions (
                promotion_id, source_node_id, target_node_id, promotion_type,
                evidence_count, promotion_score, status, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(promotion.promotion_id),
                str(promotion.source_node_id),
                str(promotion.target_node_id),
                str(promotion.promotion_type),
                int(promotion.evidence_count),
                float(promotion.promotion_score),
                str(promotion.status),
                json.dumps(payload, sort_keys=True),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json
            FROM memory_nodes
            WHERE node_id = ?
            """,
            (str(node_id),),
        ).fetchone()
        return None if row is None else _node_row_to_dict(row)

    def edges_from(self, node_id: str, edge_type: str | None = None) -> list[dict[str, Any]]:
        if edge_type is None:
            rows = self.connection.execute(
                """
                SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                FROM memory_edges
                WHERE source_node_id = ?
                ORDER BY target_node_id ASC, edge_type ASC
                """,
                (str(node_id),),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                FROM memory_edges
                WHERE source_node_id = ? AND edge_type = ?
                ORDER BY target_node_id ASC
                """,
                (str(node_id), str(edge_type)),
            ).fetchall()
        return [_edge_row_to_dict(row) for row in rows]

    def edges_to(self, node_id: str, edge_type: str | None = None) -> list[dict[str, Any]]:
        if edge_type is None:
            rows = self.connection.execute(
                """
                SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                FROM memory_edges
                WHERE target_node_id = ?
                ORDER BY source_node_id ASC, edge_type ASC
                """,
                (str(node_id),),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json
                FROM memory_edges
                WHERE target_node_id = ? AND edge_type = ?
                ORDER BY source_node_id ASC
                """,
                (str(node_id), str(edge_type)),
            ).fetchall()
        return [_edge_row_to_dict(row) for row in rows]

    def query_nodes(self, memory_level: str | None = None, node_type: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json
            FROM memory_nodes
            WHERE 1 = 1
        """
        params: list[Any] = []
        if memory_level is not None:
            query += " AND memory_level = ?"
            params.append(str(memory_level))
        if node_type is not None:
            query += " AND node_type = ?"
            params.append(str(node_type))
        query += " ORDER BY node_id ASC"
        rows = self.connection.execute(query, tuple(params)).fetchall()
        return [_node_row_to_dict(row) for row in rows]


def interaction_node_id(identifier: int | str) -> str:
    return f"M0:interaction:{int(identifier)}"


def observation_node_id(signature: str) -> str:
    return f"M0:observation:{_short_hash(signature)}"


def action_node_id(action: int | str) -> str:
    return f"M0:action:{action}"


def delta_node_id(delta_id: int | str) -> str:
    return f"M0:delta:{delta_id}"


def contingency_node_id(context_level: int | str, context_signature: str, action: int | str, family: int | str) -> str:
    return f"M1:contingency:{_short_hash(f'{context_level}|{context_signature}|{action}|{family}')}"


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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def _node_row_to_dict(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    attrs_json = row["attrs_json"] if isinstance(row, sqlite3.Row) else row[7]
    try:
        attrs = json.loads(str(attrs_json)) if attrs_json else {}
    except Exception:
        attrs = {}
    return {
        "node_id": str(row["node_id"] if isinstance(row, sqlite3.Row) else row[0]),
        "memory_level": str(row["memory_level"] if isinstance(row, sqlite3.Row) else row[1]),
        "node_type": str(row["node_type"] if isinstance(row, sqlite3.Row) else row[2]),
        "canonical_key": row["canonical_key"] if isinstance(row, sqlite3.Row) else row[3],
        "support_count": int((row["support_count"] if isinstance(row, sqlite3.Row) else row[4]) or 0),
        "first_seen_step": row["first_seen_step"] if isinstance(row, sqlite3.Row) else row[5],
        "last_seen_step": row["last_seen_step"] if isinstance(row, sqlite3.Row) else row[6],
        "attrs": attrs,
    }


def _edge_row_to_dict(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    evidence_json = row["evidence_json"] if isinstance(row, sqlite3.Row) else row[5]
    try:
        evidence = json.loads(str(evidence_json)) if evidence_json else {}
    except Exception:
        evidence = {}
    return {
        "source_node_id": str(row["source_node_id"] if isinstance(row, sqlite3.Row) else row[0]),
        "target_node_id": str(row["target_node_id"] if isinstance(row, sqlite3.Row) else row[1]),
        "edge_type": str(row["edge_type"] if isinstance(row, sqlite3.Row) else row[2]),
        "weight": float((row["weight"] if isinstance(row, sqlite3.Row) else row[3]) or 1.0),
        "support_count": int((row["support_count"] if isinstance(row, sqlite3.Row) else row[4]) or 0),
        "evidence": evidence,
    }
