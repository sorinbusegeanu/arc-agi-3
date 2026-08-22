from __future__ import annotations

"""v8.49 action-space learning diagnostics and click-aware allocation telemetry.

The report is intentionally observational.  It records action-space exposure,
productive/no-op outcomes, mixed click/movement episodes, frontier condition and
allocation share without introducing reward semantics or changing canonical memory.
Actor processes write compact per-process deltas; the parent consumes them
incrementally for allocation and reporting, avoiding per-action IPC and shared-file
contention.
"""

import json
import math
import os
import statistics
import threading
import time
from pathlib import Path


_INSTALLED = False
_BASE_ENV_INIT = None
_BASE_ENV_AVAILABLE = None
_BASE_WRITE_ALLOCATION_LOG = None
_BASE_ALLOCATION_STDOUT = None

_TRAJECTORY_ROOT_ENV = "ARC_AGI3_V8_TRAJECTORY_ROOT"
_EVENT_DIR = "action_learning_events_v849"
_REPORT_FILE = "action_learning.log"
_FLUSH_STEPS = 64
_RUN_STARTED_AT = time.time()
_REFRESH_LOCK = threading.RLock()
_FILE_OFFSETS: dict[str, int] = {}
_SPACE: dict[str, dict[str, object]] = {}
_RUN: dict[str, dict[str, object]] = {}
_FRONTIER_FILE_CACHE: dict[
    str,
    tuple[tuple[int, int, int], str, dict[str, int]],
] = {}
_LAST_REFRESH = 0.0
_REFRESH_SECONDS = 0.50
_REFRESH_CURSOR = 0
_REFRESH_MAX_EVENTS = 2048
_REFRESH_MAX_EVENTS_PER_FILE = 128
_REFRESH_MAX_SECONDS = 0.05

_COUNTER_FIELDS = (
    "steps",
    "click_actions_executed",
    "click_noops",
    "click_productive",
    "click_level_advances",
    "click_wins",
    "movement_actions_executed",
    "movement_productive",
    "movement_level_advances",
    "mixed_sequences_observed",
    "mixed_sequences_productive",
    "mixed_sequences_level_advancing",
)


def _click():
    from v8 import click_exploration_v848 as click

    return click


def _empty_aggregate() -> dict[str, object]:
    row: dict[str, object] = {
        "native_types": set(),
        "click_targets_available": set(),
        "movement_actions_available": set(),
        "exact_click_targets_available": set(),
        "click_targets_tested": set(),
        "exact_click_targets_tested": set(),
        "grid_coordinate_capacity": 0,
        "max_branching": 0,
    }
    for field in _COUNTER_FIELDS:
        row[field] = 0
    return row


def _aggregate(target: dict[str, dict[str, object]], game_id: str) -> dict[str, object]:
    return target.setdefault(str(game_id), _empty_aggregate())


def _event_root() -> Path | None:
    raw = os.environ.get(_TRAJECTORY_ROOT_ENV)
    if raw is None or not str(raw).strip():
        return None
    return Path(raw).parent / _EVENT_DIR


def _ensure_env_metrics(env) -> None:
    if hasattr(env, "_v849_metrics"):
        return
    env._v849_metrics = {field: 0 for field in _COUNTER_FIELDS}
    env._v849_available_native_types = set()
    env._v849_available_clicks = set()
    env._v849_available_exact_clicks = set()
    env._v849_available_movements = set()
    env._v849_tested_clicks = set()
    env._v849_tested_exact_clicks = set()
    env._v849_reported_native_types = set()
    env._v849_reported_clicks = set()
    env._v849_reported_exact_available = set()
    env._v849_reported_movements = set()
    env._v849_reported_tested_clicks = set()
    env._v849_reported_tested_exact = set()
    env._v849_grid_coordinate_capacity = 0
    env._v849_max_branching = 0
    env._v849_steps_since_flush = 0
    env._v849_episode_kinds = set()
    env._v849_episode_mixed_counted = False
    env._v849_episode_mixed_productive = False
    env._v849_episode_mixed_level = False
    env._v849_last_available = ()


def _env_init_v849(self, *args, **kwargs) -> None:
    game_id = str(kwargs.get("game_id", ""))
    _BASE_ENV_INIT(self, *args, **kwargs)
    self._v849_game_id = game_id
    _ensure_env_metrics(self)


def _observe_available(env, actions) -> None:
    import numpy as np

    _ensure_env_metrics(env)
    values = tuple(sorted({int(value) for value in actions}))
    env._v849_last_available = values
    env._v849_max_branching = max(int(env._v849_max_branching), len(values))
    click = _click()
    for action in values:
        native = int(click._targeting().native_action_id(int(action)))
        env._v849_available_native_types.add(native)
        if native == 6:
            env._v849_available_clicks.add(int(action))
            if click._is_exact_click_token(int(action)):
                env._v849_available_exact_clicks.add(int(action))
        else:
            env._v849_available_movements.add(int(action))
    grid = np.asarray(getattr(env, "_last_grid", ()))
    if grid.ndim == 2 and grid.size > 0:
        env._v849_grid_coordinate_capacity = max(
            int(env._v849_grid_coordinate_capacity),
            int(grid.shape[0]) * int(grid.shape[1]),
        )


def _env_available_v849(self):
    result = _BASE_ENV_AVAILABLE(self)
    _observe_available(self, result)
    return result


def _changed(before, after) -> bool:
    import numpy as np

    left = np.asarray(before)
    right = np.asarray(after)
    if left.shape != right.shape:
        return True
    return bool(left.size and np.any(left != right))


def _record_episode_kind(env, kind: str, *, productive: bool, level_advanced: bool) -> None:
    env._v849_episode_kinds.add(str(kind))
    mixed = "click" in env._v849_episode_kinds and "movement" in env._v849_episode_kinds
    if mixed and not bool(env._v849_episode_mixed_counted):
        env._v849_metrics["mixed_sequences_observed"] += 1
        env._v849_episode_mixed_counted = True
    if mixed and productive and not bool(env._v849_episode_mixed_productive):
        env._v849_metrics["mixed_sequences_productive"] += 1
        env._v849_episode_mixed_productive = True
    if mixed and level_advanced and not bool(env._v849_episode_mixed_level):
        env._v849_metrics["mixed_sequences_level_advancing"] += 1
        env._v849_episode_mixed_level = True


def _reset_episode_metrics(env) -> None:
    env._v849_episode_kinds.clear()
    env._v849_episode_mixed_counted = False
    env._v849_episode_mixed_productive = False
    env._v849_episode_mixed_level = False


def _event_payload(env) -> dict[str, object] | None:
    _ensure_env_metrics(env)
    game_id = str(getattr(env, "_v849_game_id", ""))
    if not game_id:
        return None
    new_native = env._v849_available_native_types - env._v849_reported_native_types
    new_clicks = env._v849_available_clicks - env._v849_reported_clicks
    new_exact_available = (
        env._v849_available_exact_clicks - env._v849_reported_exact_available
    )
    new_movements = env._v849_available_movements - env._v849_reported_movements
    new_tested = env._v849_tested_clicks - env._v849_reported_tested_clicks
    new_tested_exact = env._v849_tested_exact_clicks - env._v849_reported_tested_exact
    counters = {field: int(env._v849_metrics[field]) for field in _COUNTER_FIELDS}
    if (
        not any(counters.values())
        and not new_native
        and not new_clicks
        and not new_movements
        and not new_tested
        and not new_exact_available
    ):
        return None
    payload: dict[str, object] = {
        "schema": 1,
        "time": time.time(),
        "pid": os.getpid(),
        "game_id": game_id,
        "native_types": sorted(int(value) for value in new_native),
        "click_targets_available": sorted(int(value) for value in new_clicks),
        "exact_click_targets_available": sorted(
            int(value) for value in new_exact_available
        ),
        "movement_actions_available": sorted(int(value) for value in new_movements),
        "click_targets_tested": sorted(int(value) for value in new_tested),
        "exact_click_targets_tested": sorted(
            int(value) for value in new_tested_exact
        ),
        "grid_coordinate_capacity": int(env._v849_grid_coordinate_capacity),
        "max_branching": int(env._v849_max_branching),
    }
    payload.update(counters)
    return payload


def _mark_payload_reported(env) -> None:
    env._v849_reported_native_types.update(env._v849_available_native_types)
    env._v849_reported_clicks.update(env._v849_available_clicks)
    env._v849_reported_exact_available.update(env._v849_available_exact_clicks)
    env._v849_reported_movements.update(env._v849_available_movements)
    env._v849_reported_tested_clicks.update(env._v849_tested_clicks)
    env._v849_reported_tested_exact.update(env._v849_tested_exact_clicks)
    for field in _COUNTER_FIELDS:
        env._v849_metrics[field] = 0
    env._v849_steps_since_flush = 0


def _flush_env_metrics(env, *, force: bool = False) -> None:
    _ensure_env_metrics(env)
    if not force and int(env._v849_steps_since_flush) < _FLUSH_STEPS:
        return
    root = _event_root()
    if root is None:
        if force:
            env._v849_steps_since_flush = 0
        return
    payload = _event_payload(env)
    if payload is None:
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"actor-{os.getpid()}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError:
        return
    _mark_payload_reported(env)


def _env_step_v849(self, action):
    import numpy as np

    _ensure_env_metrics(self)
    if not self._v849_last_available:
        try:
            _observe_available(self, _BASE_ENV_AVAILABLE(self))
        except BaseException:
            pass
    token = int(action)
    click = _click()
    native = int(click._targeting().native_action_id(token))
    before = np.asarray(getattr(self, "_last_grid", ()))
    before_level = int(getattr(self, "last_levels_completed", 0))

    result = click._env_step_v848(self, token)

    after = np.asarray(getattr(self, "_last_grid", ()))
    level_advanced = bool(
        int(getattr(self, "last_levels_completed", before_level)) > before_level
        or bool(getattr(self, "level_completed_event", False))
    )
    win = str(getattr(self, "last_outcome_state", "")) == "WIN"
    productive = bool(_changed(before, after) or level_advanced or win)
    self._v849_metrics["steps"] += 1
    self._v849_steps_since_flush += 1

    if native == 6:
        self._v849_metrics["click_actions_executed"] += 1
        self._v849_tested_clicks.add(token)
        if click._is_exact_click_token(token):
            self._v849_tested_exact_clicks.add(token)
        if productive:
            self._v849_metrics["click_productive"] += 1
        else:
            self._v849_metrics["click_noops"] += 1
        if level_advanced:
            self._v849_metrics["click_level_advances"] += 1
        if win:
            self._v849_metrics["click_wins"] += 1
        kind = "click"
    else:
        self._v849_metrics["movement_actions_executed"] += 1
        if productive:
            self._v849_metrics["movement_productive"] += 1
        if level_advanced:
            self._v849_metrics["movement_level_advances"] += 1
        kind = "movement"

    _record_episode_kind(
        self,
        kind,
        productive=productive,
        level_advanced=level_advanced,
    )

    terminal = bool(
        win
        or str(getattr(self, "last_outcome_state", "")) == "GAME_OVER"
        or bool(getattr(self, "last_step_was_reset_boundary", False))
    )
    _flush_env_metrics(self, force=terminal)
    if terminal:
        _reset_episode_metrics(self)
    return result


def _env_reset_v849(self, *args, **kwargs):
    _ensure_env_metrics(self)
    _flush_env_metrics(self, force=True)
    _reset_episode_metrics(self)
    return _click()._env_reset_v848(self, *args, **kwargs)


def _merge_event(target: dict[str, dict[str, object]], raw: dict[str, object]) -> None:
    game = str(raw.get("game_id", ""))
    if not game:
        return
    row = _aggregate(target, game)
    for field in _COUNTER_FIELDS:
        row[field] = int(row[field]) + max(0, int(raw.get(field, 0)))
    for field, key in (
        ("native_types", "native_types"),
        ("click_targets_available", "click_targets_available"),
        ("movement_actions_available", "movement_actions_available"),
        ("exact_click_targets_available", "exact_click_targets_available"),
        ("click_targets_tested", "click_targets_tested"),
        ("exact_click_targets_tested", "exact_click_targets_tested"),
    ):
        values = row[field]
        if isinstance(values, set):
            values.update(int(value) for value in raw.get(key, ()))
    row["grid_coordinate_capacity"] = max(
        int(row["grid_coordinate_capacity"]),
        max(0, int(raw.get("grid_coordinate_capacity", 0))),
    )
    row["max_branching"] = max(
        int(row["max_branching"]),
        max(0, int(raw.get("max_branching", 0))),
    )


def _refresh_events(*, force: bool = False) -> None:
    global _LAST_REFRESH, _REFRESH_CURSOR
    now = time.monotonic()
    with _REFRESH_LOCK:
        if not force and now - float(_LAST_REFRESH) < _REFRESH_SECONDS:
            return
        started = now
        events = 0
        checked = 0
        try:
            root = _event_root()
            if root is None or not root.exists():
                _REFRESH_CURSOR = 0
                return
            paths = tuple(sorted(root.glob("actor-*.jsonl")))
            if not paths:
                _REFRESH_CURSOR = 0
                return
            start = int(_REFRESH_CURSOR) % len(paths)
            for relative_index in range(len(paths)):
                index = (start + relative_index) % len(paths)
                if checked > 0 and time.monotonic() - started >= _REFRESH_MAX_SECONDS:
                    _REFRESH_CURSOR = (index + 1) % len(paths)
                    return
                path = paths[index]
                key = str(path)
                offset = int(_FILE_OFFSETS.get(key, 0))
                checked += 1
                try:
                    size = int(path.stat().st_size)
                    if size < offset:
                        offset = 0
                    if size == offset:
                        _FILE_OFFSETS[key] = offset
                        _REFRESH_CURSOR = (index + 1) % len(paths)
                        continue
                    with path.open("r", encoding="utf-8") as stream:
                        stream.seek(offset)
                        file_events = 0
                        while True:
                            position = stream.tell()
                            line = stream.readline()
                            if not line:
                                break
                            try:
                                raw = json.loads(line)
                            except (ValueError, TypeError):
                                stream.seek(position)
                                break
                            if isinstance(raw, dict) and int(raw.get("schema", 0)) == 1:
                                _merge_event(_SPACE, raw)
                                if float(raw.get("time", 0.0)) >= _RUN_STARTED_AT:
                                    _merge_event(_RUN, raw)
                            events += 1
                            file_events += 1
                            if (
                                events >= _REFRESH_MAX_EVENTS
                                or time.monotonic() - started >= _REFRESH_MAX_SECONDS
                            ):
                                _FILE_OFFSETS[key] = stream.tell()
                                _REFRESH_CURSOR = (index + 1) % len(paths)
                                return
                            if file_events >= _REFRESH_MAX_EVENTS_PER_FILE:
                                break
                        _FILE_OFFSETS[key] = stream.tell()
                    _REFRESH_CURSOR = (index + 1) % len(paths)
                except OSError:
                    _REFRESH_CURSOR = (index + 1) % len(paths)
                    continue
            _REFRESH_CURSOR = 0
        finally:
            # Rate-limit from completion, not from the beginning of a possibly
            # slow scan. Otherwise a scan slower than the interval immediately
            # triggers another scan and starves lease dispatch.
            _LAST_REFRESH = time.monotonic()


def _probe_game_action_space_v849(
    game_id: str,
    *,
    refresh_events: bool = True,
):
    """Non-invasive replacement for v8.48's environment-construction probe."""
    if refresh_events:
        _refresh_events()
    row = _SPACE.get(str(game_id))
    if row is None:
        return None
    native = row["native_types"]
    if not isinstance(native, set) or not native:
        return None
    return 6 in native, max(1, int(row["max_branching"]))


def _click_complexity_multiplier_v849(
    coordinator,
    game_id: str,
    *,
    refresh_events: bool = True,
) -> float:
    if refresh_events:
        _refresh_events()
    game = str(game_id)
    measured = None
    with coordinator._lock:
        prior = getattr(coordinator, "_v848_action_spaces", {}).get(game)
        if prior is not None:
            measured = prior
    if measured is None:
        measured = _probe_game_action_space_v849(game, refresh_events=False)
    if not measured or not bool(measured[0]):
        return 1.0

    row = _SPACE.get(game)
    click_branching = max(1, int(measured[1]))
    references = []
    for other in _SPACE.values():
        native = other["native_types"]
        movements = other["movement_actions_available"]
        if isinstance(native, set) and 6 not in native and isinstance(movements, set) and movements:
            references.append(len(movements))
    if not references:
        # Mixed games still provide an observed non-click branch baseline.
        for other in _SPACE.values():
            movements = other["movement_actions_available"]
            if isinstance(movements, set) and movements:
                references.append(len(movements))
    reference = max(1.0, float(statistics.median(references))) if references else 1.0
    return math.sqrt(max(1.0, float(click_branching) / reference))


def _space_type(row: dict[str, object] | None) -> str:
    if row is None:
        return "unknown"
    native = row.get("native_types", set())
    if not isinstance(native, set) or not native:
        return "unknown"
    has_click = 6 in native
    has_other = any(int(value) != 6 for value in native)
    if has_click and has_other:
        return "mixed"
    if has_click:
        return "click"
    return "movement"


def _frontier_metrics(game_id: str) -> dict[str, int]:
    from v8 import sampling_evidence_frontier_v847 as frontier

    raw_root = os.environ.get(_TRAJECTORY_ROOT_ENV)
    result = {
        "click_frontier_nodes": 0,
        "click_frontier_expandable": 0,
        "suppressed_click_noop_frontiers": 0,
    }
    if raw_root is None:
        return result
    root = Path(raw_root) / frontier._STATE_DIR
    if not root.exists():
        return result
    token = frontier._game_token(str(game_id))
    for path in sorted(root.glob(f"{token}-*.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        signature = (int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))
        cached = _FRONTIER_FILE_CACHE.get(str(path))
        if cached is not None and cached[0] == signature:
            cached_game, metrics = cached[1], cached[2]
            if cached_game == str(game_id):
                for field, value in metrics.items():
                    result[field] += int(value)
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        cached_game = str(raw.get("game_id", ""))
        metrics = {
            "click_frontier_nodes": 0,
            "click_frontier_expandable": 0,
            "suppressed_click_noop_frontiers": 0,
        }
        if cached_game != str(game_id):
            _FRONTIER_FILE_CACHE[str(path)] = (signature, cached_game, metrics)
            continue
        for item in raw.get("nodes", ()):
            if not isinstance(item, dict):
                continue
            available = {int(value) for value in item.get("available_actions", ())}
            tried = {int(value) for value in item.get("tried_actions", ())}
            anchor = tuple(int(value) for value in item.get("anchor", ()))
            click_available = {value for value in available if _click()._is_click_token(value)}
            click_expandable = click_available - tried
            if click_available or (anchor and _click()._is_click_token(anchor[-1])):
                metrics["click_frontier_nodes"] += 1
            if click_expandable:
                metrics["click_frontier_expandable"] += 1
            if bool(item.get("latent", False)) and anchor and _click()._is_click_token(anchor[-1]):
                metrics["suppressed_click_noop_frontiers"] += 1
        try:
            after = path.stat()
            after_signature = (int(after.st_ino), int(after.st_size), int(after.st_mtime_ns))
        except OSError:
            after_signature = ()
        if after_signature == signature:
            _FRONTIER_FILE_CACHE[str(path)] = (signature, cached_game, metrics)
        for field, value in metrics.items():
            result[field] += int(value)
    return result


def _game_row(
    coordinator,
    game_id: str,
    *,
    refresh_events: bool = True,
) -> dict[str, object]:
    from v8 import plateau_progress_v846 as progress

    if refresh_events:
        _refresh_events(force=True)
    game = str(game_id)
    run = _RUN.get(game, _empty_aggregate())
    space = _SPACE.get(game, run)
    kind = _space_type(space)
    click_actions = int(run["click_actions_executed"])
    movement_actions = int(run["movement_actions_executed"])
    exact_tested = run["exact_click_targets_tested"]
    click_tested = run["click_targets_tested"]
    click_available = run["click_targets_available"]
    movement_available = run["movement_actions_available"]
    capacity = int(run["grid_coordinate_capacity"])
    coverage = (
        100.0 * len(exact_tested) / float(capacity)
        if capacity > 0 and isinstance(exact_tested, set)
        else 0.0
    )
    productive_click_rate = (
        float(run["click_productive"]) / float(click_actions)
        if click_actions > 0
        else 0.0
    )
    productive_movement_rate = (
        float(run["movement_productive"]) / float(movement_actions)
        if movement_actions > 0
        else 0.0
    )
    revisit_rate = (
        1.0 - float(len(click_tested)) / float(click_actions)
        if click_actions > 0 and isinstance(click_tested, set)
        else 0.0
    )
    won = bool(getattr(coordinator, "_game_won", {}).get(game, False))
    deepest = max(0, int(progress._MAX_LEVEL_REACHED.get(game, 0)))
    levels_solved = 5 if won else min(4, deepest)
    with coordinator._lock:
        allocation = coordinator._run.get(game)
        total_steps = max(1, sum(int(row.sample_steps) for row in coordinator._run.values()))
        allocation_steps = 0 if allocation is None else int(allocation.sample_steps)
    row: dict[str, object] = {
        "game_id": game,
        "action_space_type": kind,
        "steps": int(run["steps"]),
        "levels_solved": int(levels_solved),
        "wins": int(won),
        "movement_actions_available": len(movement_available) if isinstance(movement_available, set) else 0,
        "click_targets_available": len(click_available) if isinstance(click_available, set) else 0,
        "click_targets_tested": len(click_tested) if isinstance(click_tested, set) else 0,
        "click_target_coverage_pct": min(100.0, max(0.0, coverage)),
        "click_actions_executed": click_actions,
        "click_noops": int(run["click_noops"]),
        "click_productive": int(run["click_productive"]),
        "click_level_advances": int(run["click_level_advances"]),
        "click_wins": int(run["click_wins"]),
        "movement_actions_executed": movement_actions,
        "movement_productive": int(run["movement_productive"]),
        "movement_level_advances": int(run["movement_level_advances"]),
        "productive_click_rate": productive_click_rate,
        "productive_movement_rate": productive_movement_rate,
        "unique_click_targets_tested": len(click_tested) if isinstance(click_tested, set) else 0,
        "unique_productive_click_targets": 0,
        "click_revisit_rate": max(0.0, min(1.0, revisit_rate)),
        "mixed_sequences_observed": int(run["mixed_sequences_observed"]),
        "mixed_sequences_productive": int(run["mixed_sequences_productive"]),
        "mixed_sequences_level_advancing": int(run["mixed_sequences_level_advancing"]),
        "allocation_steps": allocation_steps,
        "allocation_share": float(allocation_steps) / float(total_steps),
        "click_complexity_multiplier": float(
            _click_complexity_multiplier_v849(
                coordinator,
                game,
                refresh_events=False,
            )
        ),
    }
    if kind in {"click", "mixed"}:
        row.update(_frontier_metrics(game))
    else:
        row.update(
            {
                "click_frontier_nodes": 0,
                "click_frontier_expandable": 0,
                "suppressed_click_noop_frontiers": 0,
            }
        )
    return row


def action_learning_snapshot_v849(coordinator) -> dict[str, object]:
    _refresh_events(force=True)
    games = tuple(sorted(getattr(coordinator, "_games", ())))
    rows = [
        _game_row(coordinator, game, refresh_events=False)
        for game in games
    ]
    click_capable = [row for row in rows if row["action_space_type"] in {"click", "mixed"}]
    click_only = [row for row in rows if row["action_space_type"] == "click"]
    mixed = [row for row in rows if row["action_space_type"] == "mixed"]
    total_click_actions = sum(int(row["click_actions_executed"]) for row in click_capable)
    total_click_productive = sum(int(row["click_productive"]) for row in click_capable)
    total_click_tested = sum(int(row["unique_click_targets_tested"]) for row in click_capable)
    total_click_capacity = sum(
        int(_RUN.get(str(row["game_id"]), {}).get("grid_coordinate_capacity", 0))
        for row in click_capable
    )
    summary = {
        "click_capable_games": len(click_capable),
        "click_games": len(click_only),
        "mixed_games": len(mixed),
        "click_levels_solved": sum(int(row["levels_solved"]) for row in click_capable),
        "click_levels_total": 5 * len(click_capable),
        "click_games_solved": sum(int(row["wins"]) for row in click_capable),
        "mixed_levels_solved": sum(int(row["levels_solved"]) for row in mixed),
        "mixed_levels_total": 5 * len(mixed),
        "click_coverage_pct": (
            100.0 * float(total_click_tested) / float(total_click_capacity)
            if total_click_capacity > 0
            else 0.0
        ),
        "click_productive_rate": (
            float(total_click_productive) / float(total_click_actions)
            if total_click_actions > 0
            else 0.0
        ),
    }
    return {
        "time": time.time(),
        "summary": summary,
        "games": rows,
    }


def _write_action_learning_log(runtime, coordinator) -> None:
    payload = action_learning_snapshot_v849(coordinator)
    payload["generation"] = int(getattr(runtime, "generation", 0))
    target = Path(runtime.root) / _REPORT_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        return


def _write_allocation_log_v849(runtime, coordinator, completed_by_game, active_progress, active_leases) -> None:
    _BASE_WRITE_ALLOCATION_LOG(
        runtime,
        coordinator,
        completed_by_game,
        active_progress,
        active_leases,
    )
    _write_action_learning_log(runtime, coordinator)


def _allocation_stdout_v849(coordinator, completed_by_game, active_progress, active_leases) -> None:
    _BASE_ALLOCATION_STDOUT(
        coordinator,
        completed_by_game,
        active_progress,
        active_leases,
    )
    summary = action_learning_snapshot_v849(coordinator)["summary"]
    print(
        f'[{time.strftime("%H:%M")}] action learning '
        f"click_capable={int(summary['click_capable_games'])} "
        f"click_games={int(summary['click_games'])} mixed={int(summary['mixed_games'])} "
        f"click_levels_solved={int(summary['click_levels_solved'])}/{int(summary['click_levels_total'])} "
        f"click_games_solved={int(summary['click_games_solved'])}/{int(summary['click_capable_games'])} "
        f"click_coverage={float(summary['click_coverage_pct']):.1f}% "
        f"click_productive={100.0 * float(summary['click_productive_rate']):.1f}% "
        f"mixed_levels_solved={int(summary['mixed_levels_solved'])}/{int(summary['mixed_levels_total'])}",
        flush=True,
    )


def _reset_action_learning_state_v849() -> None:
    global _LAST_REFRESH, _REFRESH_CURSOR, _RUN_STARTED_AT
    with _REFRESH_LOCK:
        _FILE_OFFSETS.clear()
        _SPACE.clear()
        _RUN.clear()
        _FRONTIER_FILE_CACHE.clear()
        _LAST_REFRESH = 0.0
        _REFRESH_CURSOR = 0
        _RUN_STARTED_AT = time.time()


def install_action_learning_report_v849() -> None:
    global _INSTALLED, _BASE_ENV_INIT, _BASE_ENV_AVAILABLE
    global _BASE_WRITE_ALLOCATION_LOG, _BASE_ALLOCATION_STDOUT
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import adaptive_learning_allocation_v819 as allocation
    from v8 import adaptive_learning_allocation_v819_performance_fix as performance
    from v8 import click_exploration_v848 as click
    from v8 import runtime_repair_v822 as v822
    from v8 import sampling_progress_control_v829 as repair
    from v8 import solved_game_recovery_v821 as recovery
    from v8 import within_action_temporal_v88 as temporal

    # v8.48 originally wrapped the public adapter directly. Recompose it beneath
    # v8.8 while restoring every historical public authority exactly:
    # adapter -> v8.22 -> v8.21 -> v8.29 -> v8.8 -> v8.49/v8.48 -> lower adapter.
    lower_step = temporal._BASE_ARC_STEP
    click._BASE_ENV_STEP = lower_step
    temporal._BASE_ARC_STEP = _env_step_v849
    repair._BASE_ENV_STEP = temporal._adapter_step_v88
    recovery._BASE_ENV_STEP = repair._env_step_v829
    v822._BASE_ENV_STEP = recovery._tracked_env_step
    ArcGridEnvironment.step = v822._runtime_env_step

    lower_reset = temporal._BASE_ARC_RESET
    click._BASE_ENV_RESET = lower_reset
    temporal._BASE_ARC_RESET = _env_reset_v849
    repair._BASE_ENV_RESET = temporal._adapter_reset_v88
    recovery._BASE_ENV_RESET = repair._env_reset_v829
    v822._BASE_ENV_RESET = recovery._tracked_env_reset
    ArcGridEnvironment.reset = v822._runtime_env_reset

    _BASE_ENV_INIT = ArcGridEnvironment.__init__
    _BASE_ENV_AVAILABLE = ArcGridEnvironment.available_actions
    ArcGridEnvironment.__init__ = _env_init_v849
    ArcGridEnvironment.available_actions = _env_available_v849

    # Stop constructing environments in the parent merely to classify action
    # spaces. Real actor observations now drive the same v8.48 sampling wrapper.
    click._probe_game_action_space = _probe_game_action_space_v849
    click._click_complexity_multiplier = _click_complexity_multiplier_v849

    _BASE_WRITE_ALLOCATION_LOG = performance._write_allocation_log_live
    _BASE_ALLOCATION_STDOUT = performance._allocation_stdout_live
    performance._write_allocation_log_live = _write_allocation_log_v849
    performance._allocation_stdout_live = _allocation_stdout_v849
    _INSTALLED = True
