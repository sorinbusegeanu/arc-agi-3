from __future__ import annotations

"""Durable generic replay and startup-frozen restored-competence metrics."""

import json
import os
import threading
import time
from pathlib import Path
from typing import Iterable


_SCHEMA_VERSION = 1
_MANIFEST_FILE = "restored_competence.json"
_GENERIC_STORE_FILE = "generic_best_successful.json"
_GENERIC_GAMES = frozenset(
    ("FrozenLake-v1", "ArcAgi/Chess-v0", "ArcAgi/Sudoku-v0")
)
_STORE_LOCK = threading.Lock()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _runtime_root_from_success_root(success_root: str | Path | None) -> Path | None:
    if success_root is None:
        return None
    path = Path(success_root)
    if path.parent.name != "verified_success":
        return None
    return path.parent.parent


def _candidate_record(raw: object, *, game_id: str | None = None) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    game = str(game_id or raw.get("game_id", "")).strip()
    actions = raw.get("actions")
    if game not in _GENERIC_GAMES or not isinstance(actions, list) or not actions:
        return None
    try:
        action_values = [int(value) for value in actions]
        seed = int(raw.get("seed", 0) or 0)
        recorded_ns = max(0, int(raw.get("recorded_ns", 0) or 0))
    except (TypeError, ValueError):
        return None
    trajectory_id = str(raw.get("trajectory_id", "")).strip()
    if not trajectory_id:
        return None
    return {
        "game_id": game,
        "trajectory_id": trajectory_id,
        "seed": seed,
        "actions": action_values,
        "recorded_ns": recorded_ns,
        "source": str(raw.get("source", "verified_generic")),
    }


def _candidate_key(record: dict[str, object]) -> tuple[int, int, str]:
    return (
        len(record.get("actions", ())),
        max(0, int(record.get("recorded_ns", 0) or 0)),
        str(record.get("trajectory_id", "")),
    )


def _generic_store_path(runtime_root: str | Path) -> Path:
    return Path(runtime_root) / "trajectory_optimizer" / _GENERIC_STORE_FILE


def _load_generic_store(runtime_root: str | Path) -> dict[str, dict[str, object]]:
    raw = _load_json(_generic_store_path(runtime_root))
    games = raw.get("games")
    if not isinstance(games, dict):
        return {}
    result: dict[str, dict[str, object]] = {}
    for game, value in games.items():
        record = _candidate_record(value, game_id=str(game))
        if record is not None:
            result[str(game)] = record
    return result


def persist_generic_win_v872(
    success_root: str | Path | None,
    event: dict[str, object],
) -> bool:
    """Promote one verified generic WIN into the durable replay store."""

    runtime_root = _runtime_root_from_success_root(success_root)
    record = _candidate_record(event)
    if runtime_root is None or record is None:
        return False
    path = _generic_store_path(runtime_root)
    with _STORE_LOCK:
        games = _load_generic_store(runtime_root)
        prior = games.get(str(record["game_id"]))
        if prior is None or _candidate_key(record) < _candidate_key(prior):
            games[str(record["game_id"])] = record
        else:
            record = prior
        _atomic_json(
            path,
            {
                "schema_version": _SCHEMA_VERSION,
                "games": dict(sorted(games.items())),
            },
        )
    return True


def _historical_generic_wins(
    runtime_root: str | Path,
    *,
    exclude_run: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    verified_root = Path(runtime_root) / "verified_success"
    excluded = None if exclude_run is None else Path(exclude_run)
    best: dict[str, dict[str, object]] = {}
    if not verified_root.is_dir():
        return best
    for run_root in sorted(verified_root.glob("run-*")):
        if not run_root.is_dir() or (excluded is not None and run_root == excluded):
            continue
        event_root = run_root / "events"
        if not event_root.is_dir():
            continue
        for path in sorted(event_root.glob("*.json")):
            raw = _load_json(path)
            if str(raw.get("terminal_state", "")).upper() != "WIN":
                continue
            record = _candidate_record(raw)
            if record is None:
                continue
            game = str(record["game_id"])
            prior = best.get(game)
            if prior is None or _candidate_key(record) < _candidate_key(prior):
                best[game] = record
    return best


def _arc_complete_games(runtime_root: str | Path) -> dict[str, dict[str, object]]:
    path = Path(runtime_root) / "trajectory_optimizer" / "best_successful.json"
    raw = _load_json(path)
    games = raw.get("games")
    if not isinstance(games, dict):
        return {}
    result: dict[str, dict[str, object]] = {}
    for game, value in games.items():
        if not isinstance(value, dict) or str(game) in _GENERIC_GAMES:
            continue
        levels = value.get("levels")
        try:
            successes = int(value.get("successes", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not isinstance(levels, list) or not levels or successes <= 0:
            continue
        solved_levels = min(5, len(levels))
        result[str(game)] = {
            "game_id": str(game),
            "trajectory_id": str(
                value.get("trajectory_id", value.get("variant_id", ""))
            ),
            "source": "arc_best_successful",
            "restored_levels_solved": solved_levels,
            "restored_game_solved": solved_levels >= 5,
        }
    return result


def capture_startup_restored_competence_v872(
    runtime_root: str | Path,
    success_root: str | Path,
) -> dict[str, object]:
    """Freeze durable competence before current-run evidence can change it."""

    success_path = Path(success_root)
    generic = _load_generic_store(runtime_root)
    for game, record in _historical_generic_wins(
        runtime_root, exclude_run=success_path
    ).items():
        prior = generic.get(game)
        if prior is None or _candidate_key(record) < _candidate_key(prior):
            generic[game] = record

    games = _arc_complete_games(runtime_root)
    for game, record in generic.items():
        games[game] = {
            **record,
            "source": "generic_verified_win",
            "restored_levels_solved": 1,
            "restored_game_solved": True,
        }
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "source": "STARTUP_DURABLE_COMPETENCE",
        "captured_ns": time.time_ns(),
        "games": dict(sorted(games.items())),
    }
    _atomic_json(success_path / _MANIFEST_FILE, payload)
    return payload


def _manifest(success_root: str | Path | None) -> dict[str, object]:
    if success_root is None:
        return {}
    return _load_json(Path(success_root) / _MANIFEST_FILE)


def generic_replay_candidate_v872(
    success_root: str | Path | None,
    game_id: str,
) -> dict[str, object] | None:
    games = _manifest(success_root).get("games")
    if not isinstance(games, dict):
        return None
    record = games.get(str(game_id))
    if not isinstance(record, dict) or record.get("source") != "generic_verified_win":
        return None
    return _candidate_record(record, game_id=str(game_id))


def restored_competence_snapshot_v872(
    success_root: str | Path | None,
    games: Iterable[str],
) -> dict[str, object]:
    selected = tuple(dict.fromkeys(str(game) for game in games))
    manifest = _manifest(success_root)
    durable = manifest.get("games")
    if not isinstance(durable, dict):
        durable = {}

    restored_levels = 0
    restored_games: list[str] = []
    target_levels = 0
    for game in selected:
        target = 1 if game in _GENERIC_GAMES else 5
        target_levels += target
        record = durable.get(game)
        if not isinstance(record, dict):
            continue
        try:
            solved = min(
                target,
                max(0, int(record.get("restored_levels_solved", 0) or 0)),
            )
        except (TypeError, ValueError):
            solved = 0
        restored_levels += solved
        if bool(record.get("restored_game_solved", False)) and solved >= target:
            restored_games.append(game)

    game_count = len(selected)
    return {
        "schema_version": _SCHEMA_VERSION,
        "source": str(manifest.get("source", "STARTUP_DURABLE_COMPETENCE")),
        "selected_games": list(selected),
        "restored_game_ids": restored_games,
        "restored_levels_solved": restored_levels,
        "restored_games_solved": len(restored_games),
        "level_target_count": target_levels,
        "game_target_count": game_count,
        "restored_level_solve_rate_pct": (
            0.0 if target_levels <= 0 else 100.0 * restored_levels / target_levels
        ),
        "restored_game_solve_rate_pct": (
            0.0 if game_count <= 0 else 100.0 * len(restored_games) / game_count
        ),
    }
