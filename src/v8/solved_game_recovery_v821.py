from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


_INSTALLED = False
_BASE_VISIBLE_SOLUTION = None
_BASE_IS_BETTER_SOLUTION = None
_BASE_ENV_INIT = None
_BASE_ENV_STEP = None
_BASE_ENV_RESET = None
_RUNTIME_SEGMENT_ACTIONS: dict[int, list[int]] = {}
_RUNTIME_LEVEL_SEGMENTS: dict[int, dict[int, tuple[int, ...]]] = {}
_RUNTIME_CURRENT_LEVEL: dict[int, int] = {}


def _level_payload(levels: tuple[tuple[int, ...], ...]) -> list[dict[str, object]]:
    return [
        {"level": index, "actions": [int(value) for value in actions]}
        for index, actions in enumerate(levels)
    ]


def _flatten_levels(levels) -> tuple[int, ...]:
    result: list[int] = []
    for level in levels:
        result.extend(int(value) for value in level)
    return tuple(result)


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


def _record_level_count(raw: object) -> int:
    if not isinstance(raw, dict):
        return 0
    levels = raw.get("levels")
    return len(levels) if isinstance(levels, list) else 0


def _prefer_complete_solution(candidate: dict[str, object], prior: dict[str, object] | None) -> bool:
    """Within one game, complete level decomposition outranks a shorter fragment."""

    if prior is None:
        return True
    candidate_levels = _record_level_count(candidate)
    prior_levels = _record_level_count(prior)
    if candidate_levels != prior_levels:
        return candidate_levels > prior_levels
    return bool(_BASE_IS_BETTER_SOLUTION(candidate, prior))


def _already_visible(optimizer_root: Path, game_id: str, levels) -> bool:
    actions = _flatten_levels(levels)
    expected_levels = len(tuple(levels))
    best_path = optimizer_root / "best_successful.json"
    try:
        payload = json.loads(best_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        payload = {}
    games = payload.get("games", {}) if isinstance(payload, dict) else {}
    prior = games.get(str(game_id)) if isinstance(games, dict) else None
    if (
        _record_level_count(prior) == expected_levels
        and _flatten_record(prior) == actions
    ):
        return True

    inbox = optimizer_root / "solutions_inbox"
    for path in sorted(inbox.glob("*.json"))[-128:]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if (
            str(raw.get("game_id", "")) == str(game_id)
            and _record_level_count(raw) == expected_levels
            and _flatten_record(raw) == actions
        ):
            return True
    return False


def _publish_runtime_levels(game_id: str, levels) -> None:
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_v814 as optimizer

    canonical = tuple(tuple(int(value) for value in level) for level in levels)
    if not canonical or any(not level for level in canonical):
        return
    full = _flatten_levels(canonical)
    if not full:
        return
    game = str(game_id)
    if not game:
        return

    record = {
        "game_id": game,
        "trajectory_id": f"runtime-{optimizer.action_sequence_hash(full):016x}",
        "source": "observed",
        "terminal_state": "WIN",
        "total_cost": len(full),
        "levels": _level_payload(canonical),
        "attempts": 1,
        "successes": 1,
        "reliability": 1.0,
    }
    if inspection._validated_solution_record(record) is None:
        return

    root_raw = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
    if root_raw:
        optimizer_root = Path(root_raw)
        if _already_visible(optimizer_root, game, canonical):
            return
        target = optimizer_root / "solutions_inbox" / (
            f"runtime-{game}-{os.getpid()}-{time.time_ns()}.json"
        )
        optimizer._atomic_json(target, record)
    else:
        inspection._CAPTURED_SOLUTIONS_FOR_TESTS.append(record)


def _publish_runtime_win(
    game_id: str,
    actions: tuple[int, ...],
    boundaries=(),
    *,
    expected_levels: int | None = None,
) -> None:
    """Compatibility helper; never publish a known-incomplete level decomposition."""

    full = tuple(int(value) for value in actions)
    if not full:
        return
    cuts = sorted({int(value) for value in boundaries if 0 < int(value) < len(full)})
    levels = []
    start = 0
    for stop in cuts:
        levels.append(tuple(full[start:stop]))
        start = stop
    levels.append(tuple(full[start:]))
    canonical = tuple(level for level in levels if level)
    if expected_levels is not None and len(canonical) != max(1, int(expected_levels)):
        return
    _publish_runtime_levels(game_id, canonical)


def _store_level_segment(key: int, level: int, actions) -> None:
    segment = tuple(int(value) for value in actions)
    if not segment or int(level) < 0:
        return
    by_level = _RUNTIME_LEVEL_SEGMENTS.setdefault(int(key), {})
    prior = by_level.get(int(level))
    if prior is None or (len(segment), segment) < (len(prior), prior):
        by_level[int(level)] = segment


def _complete_runtime_levels(key: int, expected_levels: int):
    count = max(1, int(expected_levels))
    by_level = _RUNTIME_LEVEL_SEGMENTS.get(int(key), {})
    if any(index not in by_level for index in range(count)):
        return None
    return tuple(by_level[index] for index in range(count))


def _tracked_env_init(self, *args, **kwargs) -> None:
    _BASE_ENV_INIT(self, *args, **kwargs)
    key = id(self)
    self._v821_recovery_game_id = str(kwargs.get("game_id", ""))
    _RUNTIME_SEGMENT_ACTIONS[key] = []
    _RUNTIME_LEVEL_SEGMENTS[key] = {}
    _RUNTIME_CURRENT_LEVEL[key] = int(getattr(self, "last_levels_completed", 0))


def _tracked_env_reset(self):
    result = _BASE_ENV_RESET(self)
    key = id(self)
    # v8.21 deliberately resets while probing/replaying decision points.  Those
    # resets end only the current level attempt; already observed successful level
    # segments remain valid evidence for reconstructing a complete game path.
    _RUNTIME_SEGMENT_ACTIONS[key] = []
    _RUNTIME_CURRENT_LEVEL[key] = int(getattr(self, "last_levels_completed", 0))
    _RUNTIME_LEVEL_SEGMENTS.setdefault(key, {})
    return result


def _tracked_env_step(self, action):
    from v8 import trajectory_optimizer_v814 as optimizer

    key = id(self)
    prior_level = int(getattr(self, "last_levels_completed", 0))
    result = _BASE_ENV_STEP(self, action)

    if not bool(getattr(optimizer, "_CAPTURE_ACTIVE", False)):
        return result

    tracked_level = int(_RUNTIME_CURRENT_LEVEL.get(key, prior_level))
    if tracked_level != prior_level:
        _RUNTIME_SEGMENT_ACTIONS[key] = []
        _RUNTIME_CURRENT_LEVEL[key] = prior_level

    segment = _RUNTIME_SEGMENT_ACTIONS.setdefault(key, [])
    segment.append(int(action))
    current_level = int(getattr(self, "last_levels_completed", prior_level))
    state = str(getattr(self, "last_outcome_state", ""))

    if current_level == prior_level + 1:
        _store_level_segment(key, prior_level, segment)
        _RUNTIME_SEGMENT_ACTIONS[key] = []
        _RUNTIME_CURRENT_LEVEL[key] = current_level
    elif state == "WIN":
        # Some environments report WIN without incrementing the completion counter
        # on the final transition.  Treat the current segment as the final level.
        _store_level_segment(key, prior_level, segment)

    if state == "WIN":
        expected_levels = max(current_level, prior_level + 1)
        levels = _complete_runtime_levels(key, expected_levels)
        if levels is not None:
            game = str(getattr(self, "_v821_recovery_game_id", "")) or str(
                getattr(optimizer, "_CAPTURE_SOURCE_ID", "")
            )
            _publish_runtime_levels(game, levels)
        _RUNTIME_SEGMENT_ACTIONS[key] = []
        _RUNTIME_LEVEL_SEGMENTS[key] = {}
        _RUNTIME_CURRENT_LEVEL[key] = current_level
    elif state == "GAME_OVER":
        # Failed attempts do not invalidate successful segments learned for earlier
        # levels; only the failed current-level attempt is discarded.
        _RUNTIME_SEGMENT_ACTIONS[key] = []
        _RUNTIME_CURRENT_LEVEL[key] = current_level
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


def _validated_levels(rows, win_row):
    count = max(1, int(win_row.target.levels_completed))
    levels: list[tuple[int, ...]] = []
    for completed in range(1, count + 1):
        candidates = []
        for row in rows:
            if int(row.target.levels_completed) != completed or not row.actions:
                continue
            if completed == count and str(row.target.terminal_state) != "WIN":
                continue
            if completed < count and str(row.target.terminal_state) == "WIN":
                continue
            reliability = float(max(0, int(row.successes))) / float(max(1, int(row.attempts)))
            actions = tuple(int(value) for value in row.actions)
            candidates.append((len(actions), -reliability, actions, row))
        if not candidates:
            return None
        levels.append(min(candidates, key=lambda value: value[:3])[2])
    return tuple(levels)


def _known_level_count(optimizer_root: Path, game_id: str, rows=()) -> int:
    from v8 import trajectory_optimizer_v814 as optimizer

    known = max(
        (int(row.target.levels_completed) for row in rows),
        default=0,
    )
    inbox = optimizer_root / "inbox"
    try:
        paths = sorted(inbox.glob("*.json"))[-256:]
    except OSError:
        paths = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            row = optimizer.SuccessfulTrajectory.from_dict(raw)
        except (OSError, TypeError, ValueError, KeyError):
            continue
        if str(row.anchor.source_id) == str(game_id):
            known = max(known, int(row.target.levels_completed))
    return max(0, int(known))


def _best_from_validated(optimizer_root: Path, game_id: str, best, rows=None):
    from v8 import trajectory_inspection_v819 as inspection

    rows = _validated_rows(optimizer_root, game_id) if rows is None else tuple(rows)
    for row in rows:
        if str(row.target.terminal_state) != "WIN":
            continue
        levels = _validated_levels(rows, row)
        if levels is None:
            continue
        attempts = max(1, int(getattr(row, "attempts", 1)))
        successes = max(0, int(getattr(row, "successes", 1)))
        record = {
            "game_id": str(game_id),
            "variant_id": str(row.variant_id),
            "source": "optimized",
            "terminal_state": "WIN",
            "total_cost": sum(len(level) for level in levels),
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
    from v8 import trajectory_inspection_v819 as inspection

    game = str(game_id)
    optimizer_root = Path(root) / "trajectory_optimizer"
    rows = _validated_rows(optimizer_root, game)
    best = _BASE_VISIBLE_SOLUTION(root, game)
    known_levels = _known_level_count(optimizer_root, game, rows)
    if best is not None and known_levels > 0 and _record_level_count(best) < known_levels:
        best = None
    return _best_from_validated(optimizer_root, game, best, rows)


def install_solved_game_recovery_v821() -> None:
    global _INSTALLED, _BASE_VISIBLE_SOLUTION, _BASE_IS_BETTER_SOLUTION
    global _BASE_ENV_INIT, _BASE_ENV_STEP, _BASE_ENV_RESET
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_inspection_v819_fixups as visibility

    _BASE_VISIBLE_SOLUTION = visibility._best_visible_solution
    visibility._best_visible_solution = _best_visible_solution_v821

    _BASE_IS_BETTER_SOLUTION = inspection._is_better_solution
    inspection._is_better_solution = _prefer_complete_solution

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
