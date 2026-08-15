from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Iterable, Mapping

from v7.memory.gate_validation import GateTrialSummary
from v7.memory.ids import MemoryId
from v7.memory.schema import ensure_v7_schema
from v7.memory.state import GateId


_SQLITE_ID_CHUNK_CAP = 900


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    memory_id: MemoryId
    generation_id: int
    parent_memory_id: MemoryId | None = None
    relation_type: int = 0
    source_game: str | None = None
    source_context: str | None = None
    source_global_step: int | None = None


@dataclass(frozen=True, slots=True)
class TransferTrialRecord:
    memory_id: MemoryId
    generation_id: int
    source_game: str
    target_game: str
    success: bool
    score: float = 0.0
    payload: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ContradictionRecord:
    memory_id: MemoryId
    generation_id: int
    severity: float
    source_game: str | None = None
    source_context: str | None = None
    source_global_step: int | None = None
    payload: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CandidateProvenanceRecord:
    memory_id: MemoryId
    candidate_generation: int
    provenance_games: tuple[str, ...]
    provenance_contexts: tuple[str, ...]
    scope_hash: str


@dataclass(frozen=True, slots=True)
class GateTrialRecord:
    memory_id: MemoryId
    generation_id: int
    gate_id: GateId | int
    candidate_generation: int
    target_game: str | None = None
    target_context: str | None = None
    participated: bool = True
    contribution: float = 0.0
    causal_gain: float = 0.0
    prediction_gain: float = 0.0
    planning_gain: float = 0.0
    future_option_gain: float = 0.0
    terminal_gain: float = 0.0
    efficiency_gain: float = 0.0
    intervention_type: str = ""
    paired_trial_id: str = ""
    payload: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class LifecycleWindowRecord:
    memory_id: MemoryId
    consecutive_low_windows: int
    consecutive_harm_windows: int
    consecutive_positive_windows: int
    last_utility: float
    last_generation: int


@dataclass(frozen=True, slots=True)
class MemoryTombstoneRecord:
    memory_id: MemoryId
    level_id: int
    type_id: int
    retired_generation: int
    reason: str
    canonical_key: str | None = None
    replacement_memory_id: MemoryId | None = None
    provenance_pointer: str | None = None


class EvidenceLifecycleStore:
    """Append-only provenance, transfer, gate-validation and lifecycle ledger."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        ensure_v7_schema(self.connection)

    def close(self) -> None:
        self.connection.close()

    def _memory_id_chunks(
        self,
        memory_ids: Iterable[MemoryId],
    ) -> tuple[tuple[int, ...], ...]:
        ids = tuple(sorted(set(int(memory_id) for memory_id in memory_ids)))
        if not ids:
            return ()
        try:
            variable_limit = int(
                self.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
            )
        except (AttributeError, TypeError, ValueError, sqlite3.Error):
            variable_limit = 999
        chunk_size = max(1, min(_SQLITE_ID_CHUNK_CAP, variable_limit))
        return tuple(
            ids[offset : offset + chunk_size]
            for offset in range(0, len(ids), chunk_size)
        )

    def append_provenance(self, records: Iterable[ProvenanceRecord]) -> int:
        rows = [
            (
                int(r.memory_id),
                None if r.parent_memory_id is None else int(r.parent_memory_id),
                int(r.relation_type),
                r.source_game,
                r.source_context,
                r.source_global_step,
                int(r.generation_id),
            )
            for r in records
        ]
        if not rows:
            return 0
        with self.connection:
            self.connection.executemany(
                "INSERT INTO provenance_records(memory_id,parent_memory_id,relation_type,source_game,source_context,source_global_step,generation_id) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def append_transfer_trials(self, records: Iterable[TransferTrialRecord]) -> int:
        rows = [
            (
                int(r.memory_id),
                r.source_game,
                r.target_game,
                1 if r.success else 0,
                float(r.score),
                json.dumps(r.payload or {}, separators=(",", ":"), sort_keys=True),
                int(r.generation_id),
            )
            for r in records
        ]
        if not rows:
            return 0
        with self.connection:
            self.connection.executemany(
                "INSERT INTO transfer_trials(memory_id,source_game,target_game,success,score,payload_json,generation_id) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def append_contradictions(self, records: Iterable[ContradictionRecord]) -> int:
        rows = [
            (
                int(r.memory_id),
                float(r.severity),
                r.source_game,
                r.source_context,
                r.source_global_step,
                json.dumps(r.payload or {}, separators=(",", ":"), sort_keys=True),
                int(r.generation_id),
            )
            for r in records
        ]
        if not rows:
            return 0
        with self.connection:
            self.connection.executemany(
                "INSERT INTO contradiction_records(memory_id,severity,source_game,source_context,source_global_step,payload_json,generation_id) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def transfer_summary(
        self,
        memory_ids: Iterable[MemoryId],
    ) -> dict[MemoryId, tuple[int, int, float]]:
        summaries: dict[MemoryId, tuple[int, int, float]] = {}
        for ids in self._memory_id_chunks(memory_ids):
            placeholders = ",".join("?" for _ in ids)
            rows = self.connection.execute(
                f"SELECT memory_id, COUNT(*), SUM(success), AVG(score) FROM transfer_trials WHERE memory_id IN ({placeholders}) GROUP BY memory_id",
                ids,
            ).fetchall()
            summaries.update(
                {
                    MemoryId(int(memory_id)): (
                        int(total),
                        int(successes or 0),
                        float(avg_score or 0.0),
                    )
                    for memory_id, total, successes, avg_score in rows
                }
            )
        return summaries

    def heldout_transfer_summary(
        self,
        memory_ids: Iterable[MemoryId],
        *,
        formation_generations: Mapping[MemoryId, int] | None = None,
    ) -> dict[MemoryId, tuple[int, int, float]]:
        """Summarize unique transfer into games outside frozen formation scope."""
        ids = tuple(
            sorted(
                set(MemoryId(int(memory_id)) for memory_id in memory_ids),
                key=int,
            )
        )
        if not ids:
            return {}
        cutoffs = formation_generations or {}
        scopes = {
            memory_id: set(
                self.provenance_source_games_at(
                    memory_id,
                    int(cutoffs[memory_id]),
                )
                if memory_id in cutoffs
                else self.provenance_source_games(memory_id)
            )
            for memory_id in ids
        }
        rows: list[tuple[object, ...]] = []
        for chunk in self._memory_id_chunks(ids):
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                self.connection.execute(
                    f"SELECT memory_id, source_game, target_game, success, score, payload_json FROM transfer_trials WHERE memory_id IN ({placeholders}) ORDER BY transfer_trial_id",
                    chunk,
                ).fetchall()
            )
        unique: dict[
            tuple[MemoryId, str, int | None],
            tuple[str, str, int, float, str],
        ] = {}
        for raw_id, source_game, target_game, success, score, payload_json in rows:
            memory_id = MemoryId(int(raw_id))
            source = str(source_game or "")
            target = str(target_game or "")
            try:
                payload = json.loads(str(payload_json or "{}"))
            except (TypeError, json.JSONDecodeError):
                payload = {}
            step = payload.get("source_global_step")
            step_value = None if step is None else int(step)
            attribution = str(payload.get("attribution") or "")
            key = (memory_id, target, step_value)
            candidate = (
                source,
                target,
                int(success),
                float(score),
                attribution,
            )
            current = unique.get(key)
            if current is None or (
                attribution == "trajectory_usage"
                and current[4] != "trajectory_usage"
            ):
                unique[key] = candidate

        grouped: dict[MemoryId, list[tuple[int, float]]] = {}
        for (memory_id, _target_key, _step), (
            source,
            target,
            success,
            score,
            _attribution,
        ) in unique.items():
            scope = scopes.get(memory_id, set())
            if not scope or not target or target in scope or source == target:
                continue
            if source and source not in scope:
                continue
            grouped.setdefault(memory_id, []).append((success, score))
        return {
            memory_id: (
                len(values),
                sum(success for success, _score in values),
                sum(score for _success, score in values) / len(values),
            )
            for memory_id, values in grouped.items()
            if values
        }

    def contradiction_summary(
        self,
        memory_ids: Iterable[MemoryId],
    ) -> dict[MemoryId, tuple[int, float]]:
        summaries: dict[MemoryId, tuple[int, float]] = {}
        for ids in self._memory_id_chunks(memory_ids):
            placeholders = ",".join("?" for _ in ids)
            rows = self.connection.execute(
                f"SELECT memory_id, COUNT(*), MAX(severity) FROM contradiction_records WHERE memory_id IN ({placeholders}) GROUP BY memory_id",
                ids,
            ).fetchall()
            summaries.update(
                {
                    MemoryId(int(memory_id)): (
                        int(total),
                        float(max_severity or 0.0),
                    )
                    for memory_id, total, max_severity in rows
                }
            )
        return summaries

    def provenance_parents(self, memory_id: MemoryId) -> tuple[MemoryId, ...]:
        rows = self.connection.execute(
            "SELECT DISTINCT parent_memory_id FROM provenance_records WHERE memory_id=? AND parent_memory_id IS NOT NULL ORDER BY parent_memory_id",
            (int(memory_id),),
        ).fetchall()
        return tuple(MemoryId(int(row[0])) for row in rows)

    def provenance_source_games(self, memory_id: MemoryId) -> tuple[str, ...]:
        rows = self.connection.execute(
            """
            WITH RECURSIVE ancestry(memory_id) AS (
                SELECT ?
                UNION
                SELECT p.parent_memory_id FROM provenance_records AS p JOIN ancestry AS a ON p.memory_id=a.memory_id WHERE p.parent_memory_id IS NOT NULL
            )
            SELECT DISTINCT p.source_game FROM provenance_records AS p JOIN ancestry AS a ON p.memory_id=a.memory_id
            WHERE p.source_game IS NOT NULL AND p.source_game<>'' ORDER BY p.source_game
            """,
            (int(memory_id),),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def provenance_source_games_at(
        self,
        memory_id: MemoryId,
        generation_id: int,
    ) -> tuple[str, ...]:
        cutoff = int(generation_id)
        rows = self.connection.execute(
            """
            WITH RECURSIVE ancestry(memory_id) AS (
                SELECT ?
                UNION
                SELECT p.parent_memory_id
                FROM provenance_records AS p
                JOIN ancestry AS a ON p.memory_id=a.memory_id
                WHERE p.parent_memory_id IS NOT NULL AND p.generation_id<=?
            )
            SELECT DISTINCT p.source_game
            FROM provenance_records AS p
            JOIN ancestry AS a ON p.memory_id=a.memory_id
            WHERE p.generation_id<=?
              AND p.source_game IS NOT NULL
              AND p.source_game<>''
            ORDER BY p.source_game
            """,
            (int(memory_id), cutoff, cutoff),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def provenance_source_contexts_at(
        self,
        memory_id: MemoryId,
        generation_id: int,
    ) -> tuple[str, ...]:
        cutoff = int(generation_id)
        rows = self.connection.execute(
            """
            WITH RECURSIVE ancestry(memory_id) AS (
                SELECT ?
                UNION
                SELECT p.parent_memory_id
                FROM provenance_records AS p
                JOIN ancestry AS a ON p.memory_id=a.memory_id
                WHERE p.parent_memory_id IS NOT NULL AND p.generation_id<=?
            )
            SELECT DISTINCT p.source_context
            FROM provenance_records AS p
            JOIN ancestry AS a ON p.memory_id=a.memory_id
            WHERE p.generation_id<=?
              AND p.source_context IS NOT NULL
              AND p.source_context<>''
            ORDER BY p.source_context
            """,
            (int(memory_id), cutoff, cutoff),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    @staticmethod
    def _scope_hash(
        memory_id: MemoryId,
        generation_id: int,
        games: Iterable[str],
        contexts: Iterable[str],
    ) -> str:
        digest = blake2b(digest_size=16)
        digest.update(str(int(memory_id)).encode("ascii"))
        digest.update(str(int(generation_id)).encode("ascii"))
        for game in sorted(set(str(value) for value in games)):
            digest.update(b"g:")
            digest.update(game.encode("utf-8"))
        for context in sorted(set(str(value) for value in contexts)):
            digest.update(b"c:")
            digest.update(context.encode("utf-8"))
        return digest.hexdigest()

    def freeze_candidate_scope(
        self,
        memory_id: MemoryId,
        candidate_generation: int,
    ) -> CandidateProvenanceRecord:
        existing = self.candidate_scope(memory_id)
        if existing is not None:
            return existing
        generation = int(candidate_generation)
        games = self.provenance_source_games_at(memory_id, generation)
        contexts = self.provenance_source_contexts_at(memory_id, generation)
        scope_hash = self._scope_hash(memory_id, generation, games, contexts)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO candidate_provenance(memory_id,candidate_generation,provenance_games_json,provenance_contexts_json,scope_hash) VALUES (?,?,?,?,?)",
                (
                    int(memory_id),
                    generation,
                    json.dumps(list(games), separators=(",", ":")),
                    json.dumps(list(contexts), separators=(",", ":")),
                    scope_hash,
                ),
            )
        return self.candidate_scope(memory_id) or CandidateProvenanceRecord(
            memory_id, generation, games, contexts, scope_hash
        )

    def candidate_scope(
        self,
        memory_id: MemoryId,
    ) -> CandidateProvenanceRecord | None:
        row = self.connection.execute(
            "SELECT candidate_generation,provenance_games_json,provenance_contexts_json,scope_hash FROM candidate_provenance WHERE memory_id=?",
            (int(memory_id),),
        ).fetchone()
        if row is None:
            return None
        generation, games_json, contexts_json, scope_hash = row
        try:
            games = tuple(str(value) for value in json.loads(games_json or "[]"))
        except (TypeError, json.JSONDecodeError):
            games = ()
        try:
            contexts = tuple(str(value) for value in json.loads(contexts_json or "[]"))
        except (TypeError, json.JSONDecodeError):
            contexts = ()
        return CandidateProvenanceRecord(
            memory_id,
            int(generation),
            tuple(sorted(set(games))),
            tuple(sorted(set(contexts))),
            str(scope_hash),
        )

    @staticmethod
    def _gate_transfer_score(record: GateTrialRecord) -> float:
        return (
            0.35 * float(record.causal_gain)
            + 0.20 * float(record.prediction_gain)
            + 0.15 * float(record.planning_gain)
            + 0.10 * float(record.future_option_gain)
            + 0.10 * float(record.terminal_gain)
            + 0.10 * float(record.efficiency_gain)
        )

    @staticmethod
    def _target_is_heldout(
        gate_id: GateId,
        scope: CandidateProvenanceRecord,
        target_game: str | None,
        target_context: str | None,
    ) -> bool:
        game = str(target_game or "")
        context = str(target_context or "")
        unseen_game = bool(game) and game not in set(scope.provenance_games)
        unseen_context = bool(context) and context not in set(scope.provenance_contexts)
        if gate_id in {GateId.G34, GateId.G45, GateId.G56}:
            return unseen_game
        return unseen_game or unseen_context

    def append_gate_trials(self, records: Iterable[GateTrialRecord]) -> int:
        rows = []
        for record in records:
            gate_id = GateId(int(record.gate_id))
            scope = self.freeze_candidate_scope(
                record.memory_id,
                record.candidate_generation,
            )
            heldout = self._target_is_heldout(
                gate_id,
                scope,
                record.target_game,
                record.target_context,
            )
            paired = bool(str(record.paired_trial_id or ""))
            intervention = bool(str(record.intervention_type or ""))
            genuine = bool(
                heldout
                and record.participated
                and abs(float(record.contribution)) > 0.0
                and paired
                and intervention
            )
            score = self._gate_transfer_score(record)
            success = bool(genuine and float(record.causal_gain) > 0.0)
            rows.append(
                (
                    int(record.memory_id),
                    int(gate_id),
                    int(scope.candidate_generation),
                    record.target_game,
                    record.target_context,
                    1 if record.participated else 0,
                    float(record.contribution),
                    float(record.causal_gain),
                    float(record.prediction_gain),
                    float(record.planning_gain),
                    float(record.future_option_gain),
                    float(record.terminal_gain),
                    float(record.efficiency_gain),
                    str(record.intervention_type),
                    str(record.paired_trial_id),
                    1 if genuine else 0,
                    1 if success else 0,
                    float(score),
                    json.dumps(record.payload or {}, separators=(",", ":"), sort_keys=True),
                    int(record.generation_id),
                )
            )
        if not rows:
            return 0
        before = self.connection.total_changes
        with self.connection:
            self.connection.executemany(
                """
                INSERT OR IGNORE INTO gate_trials(
                    memory_id,gate_id,candidate_generation,target_game,target_context,
                    participated,contribution,causal_gain,prediction_gain,planning_gain,
                    future_option_gain,terminal_gain,efficiency_gain,intervention_type,
                    paired_trial_id,genuine,success,transfer_score,payload_json,generation_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return int(self.connection.total_changes - before)

    def gate_trial_summary(
        self,
        memory_ids: Iterable[MemoryId],
    ) -> dict[MemoryId, GateTrialSummary]:
        rows: list[tuple[object, ...]] = []
        for ids in self._memory_id_chunks(memory_ids):
            placeholders = ",".join("?" for _ in ids)
            rows.extend(
                self.connection.execute(
                    f"""
                    SELECT memory_id,target_game,target_context,success,causal_gain,transfer_score,
                           terminal_gain,prediction_gain,planning_gain,future_option_gain,efficiency_gain
                    FROM gate_trials
                    WHERE genuine=1 AND memory_id IN ({placeholders})
                    ORDER BY gate_trial_id
                    """,
                    ids,
                ).fetchall()
            )
        grouped: dict[MemoryId, list[tuple[object, ...]]] = {}
        for row in rows:
            grouped.setdefault(MemoryId(int(row[0])), []).append(row[1:])
        summaries: dict[MemoryId, GateTrialSummary] = {}
        for memory_id, values in grouped.items():
            targets = {
                (str(row[0] or ""), str(row[1] or ""))
                for row in values
                if row[0] or row[1]
            }
            count = len(values)
            summaries[memory_id] = GateTrialSummary(
                trials=count,
                successes=sum(int(row[2]) for row in values),
                independent_targets=len(targets),
                mean_causal_gain=sum(float(row[3]) for row in values) / count,
                mean_transfer_score=sum(float(row[4]) for row in values) / count,
                positive_terminal_gain=sum(float(row[5]) for row in values) / count,
                prediction_gain=sum(float(row[6]) for row in values) / count,
                planning_gain=sum(float(row[7]) for row in values) / count,
                future_option_gain=sum(float(row[8]) for row in values) / count,
                efficiency_gain=sum(float(row[9]) for row in values) / count,
            )
        return summaries

    def update_lifecycle_window(
        self,
        memory_id: MemoryId,
        *,
        generation_id: int,
        utility: float,
        harm: bool,
        low_threshold: float = 0.10,
        positive_threshold: float = 0.20,
    ) -> LifecycleWindowRecord:
        prior = self.lifecycle_window(memory_id)
        low = 0 if prior is None else prior.consecutive_low_windows
        harm_count = 0 if prior is None else prior.consecutive_harm_windows
        positive = 0 if prior is None else prior.consecutive_positive_windows
        value = float(utility)
        low = low + 1 if value < float(low_threshold) else 0
        harm_count = harm_count + 1 if bool(harm) else 0
        positive = positive + 1 if value >= float(positive_threshold) and not harm else 0
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO lifecycle_windows(memory_id,consecutive_low_windows,consecutive_harm_windows,consecutive_positive_windows,last_utility,last_generation,updated_at)
                VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(memory_id) DO UPDATE SET
                    consecutive_low_windows=excluded.consecutive_low_windows,
                    consecutive_harm_windows=excluded.consecutive_harm_windows,
                    consecutive_positive_windows=excluded.consecutive_positive_windows,
                    last_utility=excluded.last_utility,
                    last_generation=excluded.last_generation,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    int(memory_id),
                    low,
                    harm_count,
                    positive,
                    value,
                    int(generation_id),
                ),
            )
        return LifecycleWindowRecord(
            memory_id,
            low,
            harm_count,
            positive,
            value,
            int(generation_id),
        )

    def lifecycle_window(self, memory_id: MemoryId) -> LifecycleWindowRecord | None:
        row = self.connection.execute(
            "SELECT consecutive_low_windows,consecutive_harm_windows,consecutive_positive_windows,last_utility,last_generation FROM lifecycle_windows WHERE memory_id=?",
            (int(memory_id),),
        ).fetchone()
        if row is None:
            return None
        return LifecycleWindowRecord(
            memory_id,
            int(row[0]),
            int(row[1]),
            int(row[2]),
            float(row[3]),
            int(row[4]),
        )

    def append_tombstone(self, record: MemoryTombstoneRecord) -> int:
        before = self.connection.total_changes
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO memory_tombstones(memory_id,level_id,type_id,canonical_key,retired_generation,reason,replacement_memory_id,provenance_pointer)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    int(record.memory_id),
                    int(record.level_id),
                    int(record.type_id),
                    record.canonical_key,
                    int(record.retired_generation),
                    str(record.reason),
                    None if record.replacement_memory_id is None else int(record.replacement_memory_id),
                    record.provenance_pointer,
                ),
            )
        return int(self.connection.total_changes - before)

    def transfer_trial_exists(
        self,
        memory_id: MemoryId,
        *,
        target_game: str,
        source_global_step: int | None,
        attribution: str | None = None,
    ) -> bool:
        rows = self.connection.execute(
            "SELECT payload_json FROM transfer_trials WHERE memory_id=? AND target_game=?",
            (int(memory_id), str(target_game)),
        ).fetchall()
        for (payload_json,) in rows:
            try:
                payload = json.loads(str(payload_json or "{}"))
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("source_global_step") != source_global_step:
                continue
            if attribution is not None and str(payload.get("attribution") or "") != str(attribution):
                continue
            return True
        return False
