from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


_INSTALLED = False
_BASE_WRITE_SUCCESSFUL_TRAJECTORY = None
_BASE_VISIBLE_SOLUTION = None
_RECENT_LEVEL_PREFIXES: dict[str, dict[int, tuple[int, ...]]] = {}


def _level_payload(levels: tuple[tuple[int, ...], ...]) -> list[dict[str, object]]:
    return [
        {"level": index, "actions": [int(value) for value in actions]}
        for index, actions in enumerate(levels)
    ]


def _levels_from_prefixes(
    game_id: str,
    full_actions: tuple[int, ...],
    levels_completed: int,
) -> tuple[tuple[int, ...], ...]:
    count = max(1, int(levels_completed))
    if count <= 1:
        return (tuple(full_actions),)

    by_level = _RECENT_LEVEL_PREFIXES.get(str(game_id), {})
    cumulative: list[tuple[int, ...]] = [()]
    for level in range(1, count):
        prefix = tuple(by_level.get(level, ()))
        previous = cumulative[-1]
        if (
            not prefix
            or len(prefix) <= len(previous)
            or len(prefix) >= len(full_actions)
            or tuple(full_actions[: len(prefix)]) != prefix
        ):
            # Exact level boundaries are optional for visibility.  The complete
            # winning action sequence is still authoritative and must not be lost.
            return (tuple(full_actions),)
        cumulative.append(prefix)
    cumulative.append(tuple(full_actions))
    return tuple(
        current[len(previous) :]
        for previous, current in zip(cumulative, cumulative[1:])
    )


def _publish_complete_win(row) -> None:
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_v814 as optimizer

    game = str(row.anchor.source_id)
    full = tuple(int(value) for value in row.full_actions)
    if not game or not full:
        return

    level = max(1, int(row.target.levels_completed))
    if not tuple(row.anchor.prefix_actions) and level <= 1:
        _RECENT_LEVEL_PREFIXES[game] = {}
    _RECENT_LEVEL_PREFIXES.setdefault(game, {})[level] = full

    if str(row.target.terminal_state) != "WIN":
        return

    levels = _levels_from_prefixes(game, full, level)
    record = {
        "game_id": game,
        "trajectory_id": str(row.trajectory_id),
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
        inbox = Path(root_raw) / "solutions_inbox"
        target = inbox / (
            f"complete-{game}-{row.trajectory_id}-{os.getpid()}-{time.time_ns()}.json"
        )
        optimizer._atomic_json(target, record)
    else:
        inspection._CAPTURED_SOLUTIONS_FOR_TESTS.append(record)
    _RECENT_LEVEL_PREFIXES.pop(game, None)


def _write_successful_trajectory_v821(row) -> None:
    # Preserve every existing optimizer/inspection side effect first.  Then publish
    # a complete WIN directly from the cumulative SuccessfulTrajectory row before
    # actor capture state is cleared by the environment hook.
    _BASE_WRITE_SUCCESSFUL_TRAJECTORY(row)
    _publish_complete_win(row)


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
    global _INSTALLED, _BASE_WRITE_SUCCESSFUL_TRAJECTORY, _BASE_VISIBLE_SOLUTION
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import trajectory_inspection_v819_fixups as visibility
    from v8 import trajectory_optimizer_v814 as optimizer

    _BASE_WRITE_SUCCESSFUL_TRAJECTORY = optimizer._write_successful_trajectory
    optimizer._write_successful_trajectory = _write_successful_trajectory_v821

    _BASE_VISIBLE_SOLUTION = visibility._best_visible_solution
    visibility._best_visible_solution = _best_visible_solution_v821

    # A runtime WIN is already enough evidence to stop spending DISCOVERY credits
    # on the same game while its trajectory is being validated.  The previous
    # 60-second expiry allowed a fast solved game to become UNSOLVED-priority again
    # if persistence/validation lagged, starving harder games of freed workers.
    perf._PROVISIONAL_WIN_SECONDS = math.inf
    _INSTALLED = True
