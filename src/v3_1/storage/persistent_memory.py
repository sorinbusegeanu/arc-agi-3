from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v3_1.contracts.messages import (
    DurableMemoryUpdateBatch,
    PersistentMemoryFlushRequest,
    PersistentMemoryFlushResult,
    PersistentMemoryLoadRequest,
    PersistentMemoryLoadResult,
)
from v3_1.storage.paths import get_persistent_memory_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)


def _decode(payload: str | None, default: Any) -> Any:
    if not payload:
        return default
    return json.loads(payload)


_DURABLE_FIELD_DEFS = {
    "mechanic_type": "text",
    "maturity_stage": "text",
    "evidence_basis": "text",
    "observed_support_count": "integer not null default 0",
    "hypothesis_support_count": "integer not null default 0",
    "contradiction_count": "integer not null default 0",
    "cross_round_stability": "integer not null default 0",
    "last_evidence_tier": "text",
}


def _durable_value(payload: dict[str, Any], key: str):
    if key in {"observed_support_count", "hypothesis_support_count", "contradiction_count", "cross_round_stability"}:
        return int(payload.get(key, 0) or 0)
    return payload.get(key)


def _validate_durable_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    metadata = dict(payload.get("metadata", {}) or {})
    for key in _DURABLE_FIELD_DEFS:
        if key not in payload and key in metadata:
            payload[key] = metadata[key]
    missing = [key for key in _DURABLE_FIELD_DEFS if key not in payload]
    if missing:
        raise ValueError(f"incomplete durable row; missing fields: {sorted(missing)}")
    return payload


@dataclass(frozen=True)
class PersistentMemoryStore:
    db_path: str

    @classmethod
    def from_storage_root(cls, storage_root: str, override_path: str | None = None) -> "PersistentMemoryStore":
        resolved = Path(override_path) if override_path else get_persistent_memory_db_path(storage_root)
        return cls(db_path=str(resolved))

    def __post_init__(self) -> None:
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                create table if not exists sessions (
                    session_id text primary key,
                    game_id text not null,
                    created_at text not null,
                    updated_at text not null,
                    last_memory_version text,
                    metadata_json text not null default '{}'
                );
                create table if not exists games (
                    game_id text primary key,
                    created_at text not null,
                    updated_at text not null,
                    metadata_json text not null default '{}'
                );
                create table if not exists memory_snapshots (
                    snapshot_id integer primary key autoincrement,
                    session_id text not null,
                    round_id integer not null,
                    pass_id integer not null,
                    memory_version text not null,
                    snapshot_path text,
                    created_at text not null,
                    metadata_json text not null default '{}'
                );
                create table if not exists skills (
                    skill_id text not null,
                    game_id text not null,
                    skill_type text,
                    usefulness real not null default 0.0,
                    confidence real not null default 0.0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (skill_id, game_id)
                );
                create table if not exists skill_stats (
                    skill_id text not null,
                    game_id text not null,
                    attempts integer not null default 0,
                    successes integer not null default 0,
                    failures integer not null default 0,
                    usefulness_total real not null default 0.0,
                    confidence_total real not null default 0.0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (skill_id, game_id)
                );
                create table if not exists candidate_outcomes (
                    candidate_class text not null,
                    game_id text not null,
                    attempts integer not null default 0,
                    successes integer not null default 0,
                    failures integer not null default 0,
                    progress_total real not null default 0.0,
                    route_failures integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (candidate_class, game_id)
                );
                create table if not exists failure_patterns (
                    pattern_key text not null,
                    game_id text not null,
                    count integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (pattern_key, game_id)
                );
                create table if not exists recovery_patterns (
                    pattern_key text not null,
                    game_id text not null,
                    attempts integer not null default 0,
                    successes integer not null default 0,
                    failures integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (pattern_key, game_id)
                );
                create table if not exists poi_patterns (
                    poi_key text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    utility_total real not null default 0.0,
                    persistence_total real not null default 0.0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (poi_key, game_id)
                );
                create table if not exists trigger_patterns (
                    trigger_key text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    confidence_total real not null default 0.0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (trigger_key, game_id)
                );
                create table if not exists consequence_patterns (
                    consequence_key text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    reward_total real not null default 0.0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (consequence_key, game_id)
                );
                create table if not exists entity_signatures (
                    signature text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    success_signals integer not null default 0,
                    failure_signals integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (signature, game_id)
                );
                create table if not exists area_signatures (
                    signature text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (signature, game_id)
                );
                create table if not exists mechanic_hypotheses (
                    hypothesis_key text not null,
                    game_id text not null,
                    evidence_count integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (hypothesis_key, game_id)
                );
                create table if not exists ranker_state (
                    game_id text primary key,
                    ranker_version text,
                    payload_json text not null default '{}',
                    updated_at text not null
                );
                create table if not exists mechanic_graph_nodes (
                    node_id text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (node_id, game_id)
                );
                create table if not exists mechanic_graph_edges (
                    edge_id text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (edge_id, game_id)
                );
                create table if not exists durable_dependency_paths (
                    path_id text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (path_id, game_id)
                );
                create table if not exists deterministic_hypothesis_proposals (
                    proposal_id text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (proposal_id, game_id)
                );
                create table if not exists llm_hypothesis_proposals (
                    proposal_id text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (proposal_id, game_id)
                );
                create table if not exists proposal_validation_state (
                    proposal_id text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (proposal_id, game_id)
                );
                create table if not exists proposal_agreement_groups (
                    agreement_key text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (agreement_key, game_id)
                );
                create table if not exists proposal_outcome_summaries (
                    proposal_id text not null,
                    game_id text not null,
                    observations integer not null default 0,
                    metadata_json text not null default '{}',
                    updated_at text not null,
                    primary key (proposal_id, game_id)
                );
                """
            )
            self._ensure_columns(conn, "skills")
            self._ensure_columns(conn, "skill_stats")
            self._ensure_columns(conn, "candidate_outcomes")
            self._ensure_columns(conn, "failure_patterns")
            self._ensure_columns(conn, "recovery_patterns")
            self._ensure_columns(conn, "poi_patterns")
            self._ensure_columns(conn, "trigger_patterns")
            self._ensure_columns(conn, "consequence_patterns")
            self._ensure_columns(conn, "entity_signatures")
            self._ensure_columns(conn, "area_signatures")
            self._ensure_columns(conn, "mechanic_hypotheses")
            self._ensure_columns(conn, "mechanic_graph_nodes")
            self._ensure_columns(conn, "mechanic_graph_edges")
            self._ensure_columns(conn, "durable_dependency_paths")
            self._ensure_columns(conn, "deterministic_hypothesis_proposals")
            self._ensure_columns(conn, "llm_hypothesis_proposals")
            self._ensure_columns(conn, "proposal_validation_state")
            self._ensure_columns(conn, "proposal_agreement_groups")
            self._ensure_columns(conn, "proposal_outcome_summaries")
            self._ensure_columns(conn, "ranker_state")
            conn.commit()

    def _ensure_columns(self, conn: sqlite3.Connection, table: str) -> None:
        existing = {str(row["name"]) for row in conn.execute(f"pragma table_info({table})").fetchall()}
        for column, definition in _DURABLE_FIELD_DEFS.items():
            if column not in existing:
                conn.execute(f"alter table {table} add column {column} {definition}")

    def load_priors(self, request: PersistentMemoryLoadRequest) -> PersistentMemoryLoadResult:
        priors = {
            "skill_stats": self.query_skill_stats(request.game_id),
            "candidate_outcomes": self.query_candidate_outcomes(request.game_id),
            "failure_patterns": self.query_table_rows("failure_patterns", "pattern_key", request.game_id),
            "recovery_patterns": self.query_table_rows("recovery_patterns", "pattern_key", request.game_id),
            "poi_patterns": self.query_table_rows("poi_patterns", "poi_key", request.game_id),
            "trigger_patterns": self.query_table_rows("trigger_patterns", "trigger_key", request.game_id),
            "consequence_patterns": self.query_table_rows("consequence_patterns", "consequence_key", request.game_id),
            "entity_signatures": self.query_table_rows("entity_signatures", "signature", request.game_id),
            "area_signatures": self.query_table_rows("area_signatures", "signature", request.game_id),
            "mechanic_hypotheses": self.query_table_rows("mechanic_hypotheses", "hypothesis_key", request.game_id),
            "mechanic_graph_nodes": self.query_table_rows("mechanic_graph_nodes", "node_id", request.game_id),
            "mechanic_graph_edges": self.query_table_rows("mechanic_graph_edges", "edge_id", request.game_id),
            "durable_dependency_paths": self.query_table_rows("durable_dependency_paths", "path_id", request.game_id),
            "deterministic_hypothesis_proposals": self.query_table_rows("deterministic_hypothesis_proposals", "proposal_id", request.game_id),
            "llm_hypothesis_proposals": self.query_table_rows("llm_hypothesis_proposals", "proposal_id", request.game_id),
            "proposal_validation_state": self.query_table_rows("proposal_validation_state", "proposal_id", request.game_id),
            "proposal_agreement_groups": self.query_table_rows("proposal_agreement_groups", "agreement_key", request.game_id),
            "proposal_outcome_summaries": self.query_table_rows("proposal_outcome_summaries", "proposal_id", request.game_id),
            "ranker_state": self.query_ranker_state(request.game_id),
        }
        metadata = {
            "db_path": self.db_path,
            "loaded_tables": sorted(key for key, value in priors.items() if value),
        }
        return PersistentMemoryLoadResult(
            session_id=request.session_id,
            run_id=request.run_id,
            game_id=request.game_id,
            db_path=self.db_path,
            loaded=any(bool(value) for value in priors.values()),
            priors=priors,
            metadata=metadata,
        )

    def flush(self, request: PersistentMemoryFlushRequest) -> PersistentMemoryFlushResult:
        batch = request.batch
        counts: dict[str, int] = {}
        now = _utc_now()
        with self._connect() as conn:
            self._upsert_session(conn, batch.session_id, batch.game_id, batch.source_memory_version, request.metadata, now)
            self._upsert_game(conn, batch.game_id, now)
            if request.session_snapshot_path:
                conn.execute(
                    "insert into memory_snapshots(session_id, round_id, pass_id, memory_version, snapshot_path, created_at, metadata_json) values (?, ?, ?, ?, ?, ?, ?)",
                    (
                        batch.session_id,
                        batch.round_id,
                        batch.pass_id,
                        batch.source_memory_version,
                        request.session_snapshot_path,
                        now,
                        _json({"flush_id": request.flush_id, **request.metadata}),
                    ),
                )
            counts["skills"] = self._merge_rows(
                conn,
                "skills",
                batch.game_id,
                "skill_id",
                batch.skills,
                merge_sql="usefulness = skills.usefulness + excluded.usefulness, confidence = skills.confidence + excluded.confidence, metadata_json = excluded.metadata_json, updated_at = excluded.updated_at",
                columns=("skill_id", "skill_type", "usefulness", "confidence", "metadata_json", "updated_at"),
            )
            counts["skill_stats"] = self._merge_rows(
                conn,
                "skill_stats",
                batch.game_id,
                "skill_id",
                batch.skill_stats,
                merge_sql="attempts = skill_stats.attempts + excluded.attempts, successes = skill_stats.successes + excluded.successes, failures = skill_stats.failures + excluded.failures, usefulness_total = skill_stats.usefulness_total + excluded.usefulness_total, confidence_total = skill_stats.confidence_total + excluded.confidence_total, metadata_json = excluded.metadata_json, updated_at = excluded.updated_at",
                columns=("skill_id", "attempts", "successes", "failures", "usefulness_total", "confidence_total", "metadata_json", "updated_at"),
            )
            counts["candidate_outcomes"] = self._merge_rows(
                conn,
                "candidate_outcomes",
                batch.game_id,
                "candidate_class",
                batch.candidate_outcomes,
                merge_sql="attempts = candidate_outcomes.attempts + excluded.attempts, successes = candidate_outcomes.successes + excluded.successes, failures = candidate_outcomes.failures + excluded.failures, progress_total = candidate_outcomes.progress_total + excluded.progress_total, route_failures = candidate_outcomes.route_failures + excluded.route_failures, metadata_json = excluded.metadata_json, updated_at = excluded.updated_at",
                columns=("candidate_class", "attempts", "successes", "failures", "progress_total", "route_failures", "metadata_json", "updated_at"),
            )
            counts["failure_patterns"] = self._merge_pattern_rows(conn, "failure_patterns", batch.game_id, "pattern_key", batch.failure_patterns, count_column="count", now=now)
            counts["recovery_patterns"] = self._merge_recovery_rows(conn, batch.game_id, batch.recovery_patterns, now)
            counts["poi_patterns"] = self._merge_pattern_rows(conn, "poi_patterns", batch.game_id, "poi_key", batch.poi_patterns, count_column="observations", now=now, real_sums=("utility_total", "persistence_total"))
            counts["trigger_patterns"] = self._merge_pattern_rows(conn, "trigger_patterns", batch.game_id, "trigger_key", batch.trigger_patterns, count_column="observations", now=now, real_sums=("confidence_total",))
            counts["consequence_patterns"] = self._merge_pattern_rows(conn, "consequence_patterns", batch.game_id, "consequence_key", batch.consequence_patterns, count_column="observations", now=now, real_sums=("reward_total",))
            counts["entity_signatures"] = self._merge_entity_rows(conn, "entity_signatures", batch.game_id, batch.entity_signatures, now)
            counts["area_signatures"] = self._merge_pattern_rows(conn, "area_signatures", batch.game_id, "signature", batch.area_signatures, count_column="observations", now=now)
            counts["mechanic_hypotheses"] = self._merge_pattern_rows(conn, "mechanic_hypotheses", batch.game_id, "hypothesis_key", batch.mechanic_hypotheses, count_column="evidence_count", now=now)
            counts["mechanic_graph_nodes"] = self._merge_pattern_rows(conn, "mechanic_graph_nodes", batch.game_id, "node_id", batch.mechanic_graph_nodes, count_column="observations", now=now)
            counts["mechanic_graph_edges"] = self._merge_pattern_rows(conn, "mechanic_graph_edges", batch.game_id, "edge_id", batch.mechanic_graph_edges, count_column="observations", now=now)
            counts["durable_dependency_paths"] = self._merge_pattern_rows(conn, "durable_dependency_paths", batch.game_id, "path_id", batch.durable_dependency_paths, count_column="observations", now=now)
            counts["deterministic_supported_paths"] = self._merge_pattern_rows(conn, "durable_dependency_paths", batch.game_id, "path_id", batch.deterministic_supported_paths, count_column="observations", now=now)
            counts["llm_supported_paths"] = self._merge_pattern_rows(conn, "durable_dependency_paths", batch.game_id, "path_id", batch.llm_supported_paths, count_column="observations", now=now)
            counts["deterministic_llm_agreements"] = self._merge_pattern_rows(conn, "proposal_agreement_groups", batch.game_id, "agreement_key", batch.deterministic_llm_agreements, count_column="observations", now=now)
            counts["repeated_validated_hypotheses"] = self._merge_pattern_rows(conn, "proposal_validation_state", batch.game_id, "proposal_id", batch.repeated_validated_hypotheses, count_column="observations", now=now)
            counts["contradicted_llm_proposals"] = self._merge_pattern_rows(conn, "proposal_validation_state", batch.game_id, "proposal_id", batch.contradicted_llm_proposals, count_column="observations", now=now)
            counts["deterministic_hypothesis_proposals"] = self._merge_pattern_rows(conn, "deterministic_hypothesis_proposals", batch.game_id, "proposal_id", batch.deterministic_hypothesis_proposals, count_column="observations", now=now)
            counts["llm_hypothesis_proposals"] = self._merge_pattern_rows(conn, "llm_hypothesis_proposals", batch.game_id, "proposal_id", batch.llm_hypothesis_proposals, count_column="observations", now=now)
            counts["proposal_validation_state"] = self._merge_pattern_rows(conn, "proposal_validation_state", batch.game_id, "proposal_id", batch.proposal_validation_state, count_column="observations", now=now)
            counts["proposal_agreement_groups"] = self._merge_pattern_rows(conn, "proposal_agreement_groups", batch.game_id, "agreement_key", batch.proposal_agreement_groups, count_column="observations", now=now)
            counts["proposal_outcome_summaries"] = self._merge_pattern_rows(conn, "proposal_outcome_summaries", batch.game_id, "proposal_id", batch.proposal_outcome_summaries, count_column="observations", now=now)
            counts["ranker_state"] = self._upsert_ranker_state(conn, batch.game_id, batch.ranker_state, now)
            conn.commit()
        return PersistentMemoryFlushResult(
            session_id=request.session_id,
            run_id=request.run_id,
            game_id=request.game_id,
            flush_id=request.flush_id,
            db_path=self.db_path,
            source_memory_version=batch.source_memory_version,
            rows_written=counts,
            metadata={"db_path": self.db_path, "session_snapshot_path": request.session_snapshot_path, **request.metadata},
        )

    def record_memory_snapshot_reference(self, *, session_id: str, round_id: int, pass_id: int, memory_version: str, snapshot_path: str, metadata: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert into memory_snapshots(session_id, round_id, pass_id, memory_version, snapshot_path, created_at, metadata_json) values (?, ?, ?, ?, ?, ?, ?)",
                (session_id, round_id, pass_id, memory_version, snapshot_path, _utc_now(), _json(metadata or {})),
            )
            conn.commit()

    def query_skill_stats(self, game_id: str) -> dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "select skill_stats.skill_id, skill_stats.attempts, skill_stats.successes, skill_stats.failures, skill_stats.usefulness_total, skill_stats.confidence_total, skill_stats.metadata_json, skill_stats.mechanic_type, skill_stats.maturity_stage, skill_stats.evidence_basis, skill_stats.observed_support_count, skill_stats.hypothesis_support_count, skill_stats.contradiction_count, skill_stats.cross_round_stability, skill_stats.last_evidence_tier, skills.skill_type, skills.usefulness, skills.confidence from skill_stats left join skills on skills.skill_id = skill_stats.skill_id and skills.game_id = skill_stats.game_id where skill_stats.game_id = ?",
                (game_id,),
            ).fetchall()
        result = {}
        for row in rows:
            result[str(row["skill_id"])] = {
                "attempts": int(row["attempts"]),
                "successes": int(row["successes"]),
                "failures": int(row["failures"]),
                "usefulness_total": float(row["usefulness_total"]),
                "confidence_total": float(row["confidence_total"]),
                "skill_type": row["skill_type"],
                "usefulness": float(row["usefulness"] or 0.0),
                "confidence": float(row["confidence"] or 0.0),
                "mechanic_type": row["mechanic_type"],
                "maturity_stage": row["maturity_stage"],
                "evidence_basis": row["evidence_basis"],
                "observed_support_count": int(row["observed_support_count"] or 0),
                "hypothesis_support_count": int(row["hypothesis_support_count"] or 0),
                "contradiction_count": int(row["contradiction_count"] or 0),
                "cross_round_stability": int(row["cross_round_stability"] or 0),
                "last_evidence_tier": row["last_evidence_tier"],
                **_decode(row["metadata_json"], {}),
            }
        return result

    def query_candidate_outcomes(self, game_id: str) -> dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute("select * from candidate_outcomes where game_id = ?", (game_id,)).fetchall()
        return {
            str(row["candidate_class"]): {
                "attempts": int(row["attempts"]),
                "successes": int(row["successes"]),
                "failures": int(row["failures"]),
                "progress_total": float(row["progress_total"]),
                "route_failures": int(row["route_failures"]),
                "mechanic_type": row["mechanic_type"],
                "maturity_stage": row["maturity_stage"],
                "evidence_basis": row["evidence_basis"],
                "observed_support_count": int(row["observed_support_count"] or 0),
                "hypothesis_support_count": int(row["hypothesis_support_count"] or 0),
                "contradiction_count": int(row["contradiction_count"] or 0),
                "cross_round_stability": int(row["cross_round_stability"] or 0),
                "last_evidence_tier": row["last_evidence_tier"],
                **_decode(row["metadata_json"], {}),
            }
            for row in rows
        }

    def query_table_rows(self, table: str, key_column: str, game_id: str) -> dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute(f"select * from {table} where game_id = ?", (game_id,)).fetchall()
        result = {}
        for row in rows:
            key = str(row[key_column])
            payload = dict(row)
            payload.pop("game_id", None)
            payload.pop(key_column, None)
            payload["metadata"] = _decode(payload.pop("metadata_json", None), {})
            result[key] = payload
        return result

    def query_ranker_state(self, game_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("select * from ranker_state where game_id = ?", (game_id,)).fetchone()
        if row is None:
            return {}
        return {
            "ranker_version": row["ranker_version"],
            "mechanic_type": row["mechanic_type"],
            "maturity_stage": row["maturity_stage"],
            "evidence_basis": row["evidence_basis"],
            "observed_support_count": int(row["observed_support_count"] or 0),
            "hypothesis_support_count": int(row["hypothesis_support_count"] or 0),
            "contradiction_count": int(row["contradiction_count"] or 0),
            "cross_round_stability": int(row["cross_round_stability"] or 0),
            "last_evidence_tier": row["last_evidence_tier"],
            **_decode(row["payload_json"], {}),
        }

    def _upsert_session(self, conn: sqlite3.Connection, session_id: str, game_id: str, memory_version: str, metadata: dict[str, Any], now: str) -> None:
        conn.execute(
            """
            insert into sessions(session_id, game_id, created_at, updated_at, last_memory_version, metadata_json)
            values (?, ?, ?, ?, ?, ?)
            on conflict(session_id) do update set
                updated_at = excluded.updated_at,
                last_memory_version = excluded.last_memory_version,
                metadata_json = excluded.metadata_json
            """,
            (session_id, game_id, now, now, memory_version, _json(metadata)),
        )

    def _upsert_game(self, conn: sqlite3.Connection, game_id: str, now: str) -> None:
        conn.execute(
            """
            insert into games(game_id, created_at, updated_at, metadata_json)
            values (?, ?, ?, '{}')
            on conflict(game_id) do update set
                updated_at = excluded.updated_at
            """,
            (game_id, now, now),
        )

    def _merge_rows(self, conn: sqlite3.Connection, table: str, game_id: str, key_column: str, rows: tuple[dict[str, Any], ...], *, merge_sql: str, columns: tuple[str, ...]) -> int:
        if not rows:
            return 0
        now = _utc_now()
        durable_columns = tuple(_DURABLE_FIELD_DEFS.keys())
        all_columns = tuple(columns) + durable_columns
        durable_merge = ", ".join(f"{column} = excluded.{column}" for column in durable_columns)
        for row in rows:
            payload = _validate_durable_row(row)
            payload.setdefault("metadata_json", _json(payload.get("metadata", {})))
            payload["updated_at"] = now
            values = [payload.get(column) for column in columns]
            values.extend(_durable_value(payload, column) for column in durable_columns)
            conn.execute(
                f"""
                insert into {table}({key_column}, game_id, {', '.join(all_columns[1:])})
                values (?, ?, {', '.join('?' for _ in all_columns[1:])})
                on conflict({key_column}, game_id) do update set
                    {merge_sql}, {durable_merge}
                """,
                [payload.get(key_column), game_id, *values[1:]],
            )
        return len(rows)

    def _merge_pattern_rows(self, conn: sqlite3.Connection, table: str, game_id: str, key_column: str, rows: tuple[dict[str, Any], ...], *, count_column: str, now: str, real_sums: tuple[str, ...] = ()) -> int:
        if not rows:
            return 0
        extra_columns = [count_column, *real_sums, "metadata_json", "updated_at", *_DURABLE_FIELD_DEFS.keys()]
        merge_parts = [f"{count_column} = {table}.{count_column} + excluded.{count_column}"]
        for column in real_sums:
            merge_parts.append(f"{column} = {table}.{column} + excluded.{column}")
        merge_parts.append("metadata_json = excluded.metadata_json")
        merge_parts.append("updated_at = excluded.updated_at")
        merge_parts.extend(f"{column} = excluded.{column}" for column in _DURABLE_FIELD_DEFS)
        for row in rows:
            payload = _validate_durable_row(row)
            metadata_json = _json(payload.get("metadata", {}))
            values = [int(payload.get(count_column, 0))]
            values.extend(float(payload.get(column, 0.0)) for column in real_sums)
            values.extend([metadata_json, now])
            values.extend(_durable_value(payload, column) for column in _DURABLE_FIELD_DEFS)
            conn.execute(
                f"""
                insert into {table}({key_column}, game_id, {', '.join(extra_columns)})
                values (?, ?, {', '.join('?' for _ in extra_columns)})
                on conflict({key_column}, game_id) do update set
                    {', '.join(merge_parts)}
                """,
                [payload.get(key_column), game_id, *values],
            )
        return len(rows)

    def _merge_recovery_rows(self, conn: sqlite3.Connection, game_id: str, rows: tuple[dict[str, Any], ...], now: str) -> int:
        if not rows:
            return 0
        for row in rows:
            payload = _validate_durable_row(row)
            conn.execute(
                """
                insert into recovery_patterns(pattern_key, game_id, attempts, successes, failures, metadata_json, updated_at, mechanic_type, maturity_stage, evidence_basis, observed_support_count, hypothesis_support_count, contradiction_count, cross_round_stability, last_evidence_tier)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(pattern_key, game_id) do update set
                    attempts = recovery_patterns.attempts + excluded.attempts,
                    successes = recovery_patterns.successes + excluded.successes,
                    failures = recovery_patterns.failures + excluded.failures,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    mechanic_type = excluded.mechanic_type,
                    maturity_stage = excluded.maturity_stage,
                    evidence_basis = excluded.evidence_basis,
                    observed_support_count = excluded.observed_support_count,
                    hypothesis_support_count = excluded.hypothesis_support_count,
                    contradiction_count = excluded.contradiction_count,
                    cross_round_stability = excluded.cross_round_stability,
                    last_evidence_tier = excluded.last_evidence_tier
                """,
                (
                    payload.get("pattern_key"),
                    game_id,
                    int(payload.get("attempts", 0)),
                    int(payload.get("successes", 0)),
                    int(payload.get("failures", 0)),
                    _json(payload.get("metadata", {})),
                    now,
                    payload.get("mechanic_type"),
                    payload.get("maturity_stage"),
                    payload.get("evidence_basis"),
                    int(payload.get("observed_support_count", 0)),
                    int(payload.get("hypothesis_support_count", 0)),
                    int(payload.get("contradiction_count", 0)),
                    int(payload.get("cross_round_stability", 0)),
                    payload.get("last_evidence_tier"),
                ),
            )
        return len(rows)

    def _merge_entity_rows(self, conn: sqlite3.Connection, table: str, game_id: str, rows: tuple[dict[str, Any], ...], now: str) -> int:
        if not rows:
            return 0
        for row in rows:
            payload = _validate_durable_row(row)
            conn.execute(
                f"""
                insert into {table}(signature, game_id, observations, success_signals, failure_signals, metadata_json, updated_at, mechanic_type, maturity_stage, evidence_basis, observed_support_count, hypothesis_support_count, contradiction_count, cross_round_stability, last_evidence_tier)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(signature, game_id) do update set
                    observations = {table}.observations + excluded.observations,
                    success_signals = {table}.success_signals + excluded.success_signals,
                    failure_signals = {table}.failure_signals + excluded.failure_signals,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    mechanic_type = excluded.mechanic_type,
                    maturity_stage = excluded.maturity_stage,
                    evidence_basis = excluded.evidence_basis,
                    observed_support_count = excluded.observed_support_count,
                    hypothesis_support_count = excluded.hypothesis_support_count,
                    contradiction_count = excluded.contradiction_count,
                    cross_round_stability = excluded.cross_round_stability,
                    last_evidence_tier = excluded.last_evidence_tier
                """,
                (
                    payload.get("signature"),
                    game_id,
                    int(payload.get("observations", 0)),
                    int(payload.get("success_signals", 0)),
                    int(payload.get("failure_signals", 0)),
                    _json(payload.get("metadata", {})),
                    now,
                    payload.get("mechanic_type"),
                    payload.get("maturity_stage"),
                    payload.get("evidence_basis"),
                    int(payload.get("observed_support_count", 0)),
                    int(payload.get("hypothesis_support_count", 0)),
                    int(payload.get("contradiction_count", 0)),
                    int(payload.get("cross_round_stability", 0)),
                    payload.get("last_evidence_tier"),
                ),
            )
        return len(rows)

    def _upsert_ranker_state(self, conn: sqlite3.Connection, game_id: str, rows: tuple[dict[str, Any], ...], now: str) -> int:
        if not rows:
            return 0
        payload = _validate_durable_row(rows[-1])
        conn.execute(
            """
            insert into ranker_state(game_id, ranker_version, payload_json, updated_at, mechanic_type, maturity_stage, evidence_basis, observed_support_count, hypothesis_support_count, contradiction_count, cross_round_stability, last_evidence_tier)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(game_id) do update set
                ranker_version = excluded.ranker_version,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at,
                mechanic_type = excluded.mechanic_type,
                maturity_stage = excluded.maturity_stage,
                evidence_basis = excluded.evidence_basis,
                observed_support_count = excluded.observed_support_count,
                hypothesis_support_count = excluded.hypothesis_support_count,
                contradiction_count = excluded.contradiction_count,
                cross_round_stability = excluded.cross_round_stability,
                last_evidence_tier = excluded.last_evidence_tier
            """,
            (
                game_id,
                payload.get("ranker_version"),
                _json(payload.get("payload", payload)),
                now,
                payload.get("mechanic_type"),
                payload.get("maturity_stage"),
                payload.get("evidence_basis"),
                int(payload.get("observed_support_count", 0)),
                int(payload.get("hypothesis_support_count", 0)),
                int(payload.get("contradiction_count", 0)),
                int(payload.get("cross_round_stability", 0)),
                payload.get("last_evidence_tier"),
            ),
        )
        return 1
