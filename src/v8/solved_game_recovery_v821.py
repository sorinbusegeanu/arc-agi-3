from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


_INSTALLED = False
_BASE_VISIBLE_SOLUTION = None
_BASE_ENV_INIT = None
_BASE_ENV_STEP = None
_BASE_ENV_RESET = None
_RUNTIME_ACTIONS: dict[int, list[int]] = {}
_RUNTIME_BOUNDARIES: dict[int, list[int]] = {}


def _level_payload(levels: tuple[tuple[int, ...], ...]) -> list[dict[str, object]]:
    return [
        {"level": index, "actions": [int(value) for value in actions]}
        for index, actions in enumerate(levels)
    ]


def _flatten_record(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, dict):
        return ()
    levels = raw.get("levels")
    if not isinstance(levels, list):
        return ()
    result: list[int] = []
    for level in levels:
        if not isinstance(level, dict) or not isinstance(level.get("actions"), list):
            return ()
        try:
            result.extend(int(value) for value in level["actions"])
        except (TypeError, ValueError):
            return ()
    return tuple(result)


def _already_visible(optimizer_root: Path, game_id: str, actions: tuple[int, ...]) -> bool:
    best_path = optimizer_root / "best_successful.json"
    try:
        payload = json.loads(best_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        payload = {}
    games = payload.get("games", {}) if isinstance(payload, dict) else {}
    if isinstance(games, dict) and _flatten_record(games.get(str(game_id))) == actions:
        return True

    inbox = optimizer_root / "solutions_inbox"
    for path in sorted(inbox.glob("*.json"))[-128:]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if str(raw.get("game_id", "")) == str(game_id) and _flatten_record(raw) == actions:
            return True
    return False


def _split_runtime_levels(
    actions: tuple[int, ...], boundaries: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    cuts = sorted({int(value) for value in boundaries if 0 < int(value) < len(actions)})
    if not cuts:
        return (actions,)
    levels = []
    start = 0
    for stop in cuts:
        levels.append(tuple(actions[start:stop]))
        start = stop
    levels.append(tuple(actions[start:]))
    return tuple(level for level in levels if level)


def _publish_runtime_win(game_id: str, actions: tuple[int, ...], boundaries=()) -> None:
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_v814 as optimizer

    game = str(game_id)
    full = tuple(int(value) for value in actions)
    if not game or not full:
        return
    levels = _split_runtime_levels(full, tuple(int(value) for value in boundaries))
    record = {
        "game_id": game,
        "trajectory_id": f"runtime-{optimizer.action_sequence_hash(full):016x}",
        "source": "observed",
        "terminal_state": "WIN",
        "total_cost": len(full),
        "levels": _level_payload(levels),
        "attempts": 1,
        "successes": 1,
        "reliability": 1.0,
    }
    if inspection._validated_solution_record(record) is None:
        return

    root_raw = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
    if root_raw:
        optimizer_root = Path(root_raw)
        if _already_visible(optimizer_root, game, full):
            return
        target = optimizer_root / "solutions_inbox" / (
            f"runtime-{game}-{os.getpid()}-{time.time_ns()}.json"
        )
        optimizer._atomic_json(target, record)
    else:
        inspection._CAPTURED_SOLUTIONS_FOR_TESTS.append(record)


def _tracked_env_init(self, *args, **kwargs) -> None:
    _BASE_ENV_INIT(self, *args, **kwargs)
    key = id(self)
    self._v821_recovery_game_id = str(kwargs.get("game_id", ""))
    _RUNTIME_ACTIONS[key] = []
    _RUNTIME_BOUNDARIES[key] = []


def _tracked_env_reset(self):
    result = _BASE_ENV_RESET(self)
    key = id(self)
    _RUNTIME_ACTIONS[key] = []
    _RUNTIME_BOUNDARIES[key] = []
    return result


def _tracked_env_step(self, action):
    from v8 import trajectory_optimizer_v814 as optimizer

    key = id(self)
    prior_level = int(getattr(self, "last_levels_completed", 0))
    result = _BASE_ENV_STEP(self, action)

    # This fallback is actor-execution-only. Validators and synthetic calls to the
    # trajectory writer never satisfy the capture gate and cannot manufacture a WIN.
    if not bool(getattr(optimizer, "_CAPTURE_ACTIVE", False)):
        return result

    actions = _RUNTIME_ACTIONS.setdefault(key, [])
    boundaries = _RUNTIME_BOUNDARIES.setdefault(key, [])
    actions.append(int(action))
    current_level = int(getattr(self, "last_levels_completed", prior_level))
    if current_level > prior_level:
        boundaries.append(len(actions))

    state = str(getattr(self, "last_outcome_state", ""))
    if state == "WIN":
        game = str(getattr(self, "_v821_recovery_game_id", "")) or str(
            getattr(optimizer, "_CAPTURE_SOURCE_ID", "")
        )
        _publish_runtime_win(game, tuple(actions), tuple(boundaries))
        _RUNTIME_ACTIONS[key] = []
        _RUNTIME_BOUNDARIES[key] = []
    elif state == "GAME_OVER" or bool(getattr(self, "last_step_was_reset_boundary", False)):
        _RUNTIME_ACTIONS[key] = []
        _RUNTIME_BOUNDARIES[key] = []
    return result


def _validated_rows(optimizer_root: Path, game_id: str):
    from v8 import trajectory_optimizer_v814 as optimizer

    path = optimizer_root / "validated.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return ()
    rows = payload.get("validated", ()) if isinstance(payload, dict) else ()
    result = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            row = optimizer.ValidatedTrajectory.from_dict(raw)
        except (TypeError, ValueError, KeyError):
            continue
        if str(row.anchor.source_id) == str(game_id):
            result.append(row)
    return tuple(result)


def _validated_levels(rows, win_row) -> tuple[tuple[int, ...], ...]:
    full = tuple(int(value) for value in win_row.anchor.prefix_actions) + tuple(
        int(value) for value in win_row.actions
    )
    count = max(1, int(win_row.target.levels_completed))
    if count <= 1:
        return (full,)

    cumulative: list[tuple[int, ...]] = [()]
    for level in range(1, count):
        candidates = []
        for row in rows:
            if int(row.target.levels_completed) != level:
                continue
            prefix = tuple(int(value) for value in row.anchor.prefix_actions) + tuple(
                int(value) for value in row.actions
            )
            if (
                len(prefix) > len(cumulative[-1])
                and len(prefix) < len(full)
                and tuple(full[: len(prefix)]) == prefix
            ):
                candidates.append(prefix)
        if not candidates:
            return (full,)
        cumulative.append(min(candidates, key=lambda value: (len(value), value)))
    cumulative.append(full)
    return tuple(
        current[len(previous) :]
        for previous, current in zip(cumulative, cumulative[1:])
    )


def _best_from_validated(optimizer_root: Path, game_id: str, best):
    from v8 import trajectory_inspection_v819 as inspection

    rows = _validated_rows(optimizer_root, game_id)
    for row in rows:
        if str(row.target.terminal_state) != "WIN":
            continue
        full = tuple(int(value) for value in row.anchor.prefix_actions) + tuple(
            int(value) for value in row.actions
        )
        if not full:
            continue
        levels = _validated_levels(rows, row)
        attempts = max(1, int(getattr(row, "attempts", 1)))
        successes = max(0, int(getattr(row, "successes", 1)))
        record = {
            "game_id": str(game_id),
            "variant_id": str(row.variant_id),
            "source": "optimized",
            "terminal_state": "WIN",
            "total_cost": len(full),
            "levels": _level_payload(levels),
            "attempts": attempts,
            "successes": successes,
            "reliability": float(successes) / float(attempts),
        }
        record = inspection._validated_solution_record(record)
        if record is not None and inspection._is_better_solution(record, best):
            best = record
    return best


def _best_visible_solution_v821(root: str | Path, game_id: str):
    best = _BASE_VISIBLE_SOLUTION(root, game_id)
    optimizer_root = Path(root) / "trajectory_optimizer"
    return _best_from_validated(optimizer_root, str(game_id), best)


def install_solved_game_recovery_v821() -> None:
    global _INSTALLED, _BASE_VISIBLE_SOLUTION, _BASE_ENV_INIT, _BASE_ENV_STEP, _BASE_ENV_RESET
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import trajectory_inspection_v819_fixups as visibility

    _BASE_VISIBLE_SOLUTION = visibility._best_visible_solution
    visibility._best_visible_solution = _best_visible_solution_v821

    _BASE_ENV_INIT = ArcGridEnvironment.__init__
    _BASE_ENV_STEP = ArcGridEnvironment.step
    _BASE_ENV_RESET = ArcGridEnvironment.reset
    ArcGridEnvironment.__init__ = _tracked_env_init
    ArcGridEnvironment.step = _tracked_env_step
    ArcGridEnvironment.reset = _tracked_env_reset

    # A runtime WIN is enough to stop spending DISCOVERY credits on the same game
    # while its trajectory is being validated. Scientific solved state still comes
    # only from the existing validated frontier path.
    perf._PROVISIONAL_WIN_SECONDS = math.inf
    _INSTALLED = True
