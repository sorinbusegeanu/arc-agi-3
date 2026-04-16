from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from v5_0.contracts.avatar_types import SavedLevelTrace

GLOBAL_TRACE_STORE_PATH = Path("/home/zodrak/zod/runs_v5_0/trace_store.sqlite")

def get_global_trace_store_path() -> str:
    return str(GLOBAL_TRACE_STORE_PATH)


def _resolve_db_path(db_path: str | Path | None) -> Path:
    return Path(db_path) if db_path is not None else Path(get_global_trace_store_path())


def initialize_trace_store(db_path: str | Path | None = None) -> str:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS level_traces (
                game_id TEXT NOT NULL,
                level_id TEXT NOT NULL,
                trace_id TEXT PRIMARY KEY,
                trace_version INTEGER NOT NULL,
                step_count INTEGER NOT NULL,
                solved INTEGER NOT NULL,
                replay_verified INTEGER NOT NULL,
                source_run_id TEXT,
                created_at TEXT NOT NULL,
                action_trace_json TEXT NOT NULL,
                optimized INTEGER NOT NULL DEFAULT 0,
                optimized_at TEXT,
                parent_trace_id TEXT,
                notes TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trace_optimization_runs (
                run_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                level_id TEXT NOT NULL,
                trace_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_level_traces_schema_extensions(conn)
        conn.commit()
    return str(path)


def save_level_trace(*, db_path: str | Path | None = None, trace: SavedLevelTrace, trace_id: str | None = None) -> str:
    tid = trace_id or f"{trace.game_id}:{trace.level_id}:{int(datetime.now(timezone.utc).timestamp()*1000)}"
    if bool(trace.solved) and bool(trace.replay_verified):
        strict_trace = SavedLevelTrace(
            game_id=trace.game_id,
            level_id=trace.level_id,
            solved=trace.solved,
            action_trace=trace.action_trace,
            step_count=trace.step_count,
            source_run_id=trace.source_run_id,
            trace_version=trace.trace_version,
            replay_verified=trace.replay_verified,
            action_sources=getattr(trace, "action_sources", None),
            trace_id=tid,
            optimized=getattr(trace, "optimized", None),
            optimized_at=getattr(trace, "optimized_at", None),
            parent_trace_id=getattr(trace, "parent_trace_id", None),
        )
        if not validate_trace_for_best_use(strict_trace):
            raise ValueError("invalid_verified_trace_for_save")
    resolved_db_path = initialize_trace_store(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    if bool(trace.solved) and bool(trace.replay_verified):
        if int(trace.step_count) != len(tuple(trace.action_trace)):
            raise ValueError("verified_trace_step_count_mismatch")
        if not tuple(trace.action_trace):
            raise ValueError("verified_trace_empty_action_trace")
        if not tid:
            raise ValueError("verified_trace_missing_trace_id")
    with sqlite3.connect(resolved_db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO level_traces (
                game_id, level_id, trace_id, trace_version, step_count, solved, replay_verified,
                source_run_id, created_at, action_trace_json, optimized, optimized_at, parent_trace_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.game_id,
                trace.level_id,
                tid,
                int(trace.trace_version),
                int(trace.step_count),
                1 if trace.solved else 0,
                1 if trace.replay_verified else 0,
                trace.source_run_id,
                created_at,
                json.dumps(list(trace.action_trace)),
                1 if bool(getattr(trace, "optimized", False)) else 0,
                getattr(trace, "optimized_at", None),
                getattr(trace, "parent_trace_id", None),
                None,
            ),
        )
        conn.commit()
    return tid


def save_trace_history_row(*, db_path: str | Path | None = None, trace: SavedLevelTrace, trace_id: str | None = None) -> str:
    resolved_db_path = initialize_trace_store(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    tid = trace_id or str(getattr(trace, "trace_id", "") or f"{trace.game_id}:{trace.level_id}:{int(datetime.now(timezone.utc).timestamp()*1000)}")
    with sqlite3.connect(resolved_db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO level_traces (
                game_id, level_id, trace_id, trace_version, step_count, solved, replay_verified,
                source_run_id, created_at, action_trace_json, optimized, optimized_at, parent_trace_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.game_id,
                trace.level_id,
                tid,
                int(trace.trace_version),
                int(trace.step_count),
                1 if trace.solved else 0,
                1 if trace.replay_verified else 0,
                trace.source_run_id,
                created_at,
                json.dumps(list(trace.action_trace)),
                1 if bool(getattr(trace, "optimized", False)) else 0,
                getattr(trace, "optimized_at", None),
                getattr(trace, "parent_trace_id", None),
                None,
            ),
        )
        conn.commit()
    return tid


def get_best_trace_for_level(*, db_path: str | Path | None = None, game_id: str, level_id: str) -> SavedLevelTrace | None:
    resolved_db_path = initialize_trace_store(db_path)
    with sqlite3.connect(resolved_db_path) as conn:
        row = conn.execute(
            """
            SELECT game_id, level_id, trace_id, step_count, solved, replay_verified, source_run_id, trace_version, action_trace_json,
                   optimized, optimized_at, parent_trace_id
            FROM level_traces
            WHERE game_id = ? AND level_id = ? AND solved = 1 AND replay_verified = 1
            ORDER BY step_count ASC, created_at ASC, trace_id ASC
            LIMIT 1
            """,
            (game_id, level_id),
        ).fetchone()
    if row is None:
        return None
    return _row_to_trace(row)


def get_all_traces_for_game(*, db_path: str | Path | None = None, game_id: str) -> tuple[SavedLevelTrace, ...]:
    resolved_db_path = initialize_trace_store(db_path)
    with sqlite3.connect(resolved_db_path) as conn:
        rows = conn.execute(
            """
            SELECT game_id, level_id, trace_id, step_count, solved, replay_verified, source_run_id, trace_version, action_trace_json,
                   optimized, optimized_at, parent_trace_id
            FROM level_traces
            WHERE game_id = ?
            ORDER BY level_id ASC, step_count ASC, created_at ASC, trace_id ASC
            """,
            (game_id,),
        ).fetchall()
    return tuple(_row_to_trace(row) for row in rows)


def get_solved_levels_for_game(*, db_path: str | Path | None = None, game_id: str) -> tuple[str, ...]:
    return get_verified_solved_levels_for_game(db_path=db_path, game_id=game_id)


def get_verified_solved_levels_for_game(*, db_path: str | Path | None = None, game_id: str) -> tuple[str, ...]:
    resolved_db_path = initialize_trace_store(db_path)
    with sqlite3.connect(resolved_db_path) as conn:
        rows = conn.execute(
            """
            SELECT game_id, level_id, trace_id, step_count, solved, replay_verified, source_run_id, trace_version, action_trace_json,
                   optimized, optimized_at, parent_trace_id
            FROM level_traces
            WHERE game_id = ? AND solved = 1 AND replay_verified = 1
            ORDER BY level_id ASC, step_count ASC, created_at ASC, trace_id ASC
            """,
            (game_id,),
        ).fetchall()
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        trace = _row_to_trace(row)
        level_id = str(trace.level_id)
        if level_id in seen:
            continue
        if not validate_trace_for_best_use(trace):
            continue
        seen.add(level_id)
        out.append(level_id)
    return tuple(out)


def mark_trace_verified(*, db_path: str | Path | None = None, trace_id: str, replay_verified: bool = True) -> None:
    resolved_db_path = initialize_trace_store(db_path)
    with sqlite3.connect(resolved_db_path) as conn:
        conn.execute(
            "UPDATE level_traces SET replay_verified = ? WHERE trace_id = ?",
            (1 if replay_verified else 0, trace_id),
        )
        conn.commit()


def replace_best_trace_if_shorter(*, db_path: str | Path | None = None, trace: SavedLevelTrace) -> tuple[bool, str]:
    existing = get_best_trace_for_level(db_path=db_path, game_id=trace.game_id, level_id=trace.level_id)
    if existing is None or int(trace.step_count) < int(existing.step_count):
        trace_id = save_level_trace(db_path=db_path, trace=trace)
        return True, trace_id
    return False, ""


def save_or_replace_best_trace(*, db_path: str | Path | None = None, trace: SavedLevelTrace) -> tuple[bool, str]:
    existing = get_best_trace_for_level(db_path=db_path, game_id=trace.game_id, level_id=trace.level_id)
    if existing is None:
        trace_id = save_level_trace(db_path=db_path, trace=trace)
        return True, trace_id
    if int(trace.step_count) < int(existing.step_count):
        trace_id = save_level_trace(db_path=db_path, trace=trace)
        return True, trace_id
    return False, str(existing.trace_id or "")


def upsert_verified_best_trace(*, db_path: str | Path | None = None, trace: SavedLevelTrace) -> tuple[bool, str | None]:
    if not validate_trace_for_best_use(trace):
        return False, None
    trace_id = str(getattr(trace, "trace_id", "") or "")
    existing = get_best_trace_for_level(db_path=db_path, game_id=trace.game_id, level_id=trace.level_id)
    if existing is not None and int(trace.step_count) >= int(existing.step_count):
        return False, str(existing.trace_id or None)
    save_level_trace(db_path=db_path, trace=trace, trace_id=trace_id)
    return True, trace_id


def mark_trace_optimized(
    *,
    db_path: str | Path | None = None,
    trace_id: str,
    optimized: bool,
    optimized_at: str,
    notes: str | None = None,
) -> None:
    resolved_db_path = initialize_trace_store(db_path)
    with sqlite3.connect(resolved_db_path) as conn:
        conn.execute(
            """
            UPDATE level_traces
            SET optimized = ?, optimized_at = ?, notes = ?
            WHERE trace_id = ?
            """,
            (1 if optimized else 0, optimized_at, notes, trace_id),
        )
        conn.commit()


def get_best_verified_trace_prefix(
    *,
    game_id: str,
    level_ids: tuple[str, ...] | list[str],
    db_path: str | Path | None = None,
) -> tuple[SavedLevelTrace, ...]:
    traces: list[SavedLevelTrace] = []
    for level_id in tuple(str(item) for item in level_ids):
        best = get_best_trace_for_level(db_path=db_path, game_id=game_id, level_id=level_id)
        if best is None:
            break
        traces.append(best)
    return tuple(traces)


def rebuild_trace_store_index(
    *,
    db_path: str | Path | None = None,
    game_id: str,
) -> dict[str, dict[str, object]]:
    resolved_db_path = initialize_trace_store(db_path)
    with sqlite3.connect(resolved_db_path) as conn:
        rows = conn.execute(
            """
            SELECT game_id, level_id, trace_id, step_count, solved, replay_verified, source_run_id, trace_version, action_trace_json,
                   optimized, optimized_at, parent_trace_id
            FROM level_traces
            WHERE game_id = ? AND solved = 1 AND replay_verified = 1
            ORDER BY level_id ASC, step_count ASC, created_at ASC, trace_id ASC
            """,
            (game_id,),
        ).fetchall()
    index: dict[str, dict[str, object]] = {}
    for row in rows:
        trace = _row_to_trace(row)
        if not validate_trace_for_best_use(trace):
            continue
        level_id = str(trace.level_id)
        if level_id in index:
            continue
        index[level_id] = {
            "best_step_count": int(trace.step_count),
            "best_trace_id": str(trace.trace_id),
            "optimization_status": "known",
        }
    return index


def validate_trace_for_best_use(trace: SavedLevelTrace) -> bool:
    if not bool(getattr(trace, "solved", False)):
        return False
    if not bool(getattr(trace, "replay_verified", False)):
        return False
    if not str(getattr(trace, "trace_id", "") or ""):
        return False
    actions = tuple(getattr(trace, "action_trace", ()) or ())
    if not actions:
        return False
    if int(getattr(trace, "step_count", 0) or 0) != len(actions):
        return False
    return True


def _row_to_trace(row) -> SavedLevelTrace:
    (
        game_id,
        level_id,
        trace_id,
        step_count,
        solved,
        replay_verified,
        source_run_id,
        trace_version,
        action_trace_json,
        optimized,
        optimized_at,
        parent_trace_id,
    ) = row
    actions = json.loads(action_trace_json)
    return SavedLevelTrace(
        game_id=str(game_id),
        level_id=str(level_id),
        solved=bool(solved),
        action_trace=tuple(str(item) for item in actions),
        step_count=int(step_count),
        source_run_id=source_run_id,
        trace_version=int(trace_version),
        replay_verified=bool(replay_verified),
        trace_id=str(trace_id) if trace_id is not None else None,
        optimized=bool(optimized),
        optimized_at=str(optimized_at) if optimized_at is not None else None,
        parent_trace_id=str(parent_trace_id) if parent_trace_id is not None else None,
    )


def _ensure_level_traces_schema_extensions(conn: sqlite3.Connection) -> None:
    columns = {str(item[1]) for item in conn.execute("PRAGMA table_info(level_traces)").fetchall()}
    if "optimized" not in columns:
        conn.execute("ALTER TABLE level_traces ADD COLUMN optimized INTEGER NOT NULL DEFAULT 0")
    if "optimized_at" not in columns:
        conn.execute("ALTER TABLE level_traces ADD COLUMN optimized_at TEXT")
    if "parent_trace_id" not in columns:
        conn.execute("ALTER TABLE level_traces ADD COLUMN parent_trace_id TEXT")
    if "notes" not in columns:
        conn.execute("ALTER TABLE level_traces ADD COLUMN notes TEXT")
