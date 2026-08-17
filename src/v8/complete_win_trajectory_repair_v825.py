from __future__ import annotations

import json
from pathlib import Path


_INSTALLED = False
_BASE_PUBLISH_RUNTIME_LEVELS = None
_BASE_RESET_SOLVE_METRICS = None
_BASE_SERVICE_START = None
_BASE_VISIBLE_SOLUTION = None
_COMPLETE_BEST_WIN_STEPS = 0
_COMPLETE_LAST_WIN_STEPS = 0
_SNAPSHOT_RECOVERY_LIMIT = 32


def _canonical_levels(levels) -> tuple[tuple[int, ...], ...]:
    try:
        rows = tuple(tuple(int(value) for value in level) for level in levels)
    except (TypeError, ValueError):
        return ()
    if not rows or any(not row for row in rows):
        return ()
    return rows


def _reset_complete_solve_metrics_v825() -> None:
    global _COMPLETE_BEST_WIN_STEPS, _COMPLETE_LAST_WIN_STEPS

    _COMPLETE_BEST_WIN_STEPS = 0
    _COMPLETE_LAST_WIN_STEPS = 0
    _BASE_RESET_SOLVE_METRICS()


def _publish_runtime_levels_v825(game_id: str, levels) -> None:
    """Publish a reconstructable WIN and make its complete cost the solve metric."""

    global _COMPLETE_BEST_WIN_STEPS, _COMPLETE_LAST_WIN_STEPS

    canonical = _canonical_levels(levels)
    if not str(game_id) or not canonical:
        return _BASE_PUBLISH_RUNTIME_LEVELS(game_id, levels)

    _BASE_PUBLISH_RUNTIME_LEVELS(game_id, canonical)
    total = sum(len(level) for level in canonical)
    if total <= 0:
        return

    _COMPLETE_LAST_WIN_STEPS = total
    _COMPLETE_BEST_WIN_STEPS = (
        total
        if _COMPLETE_BEST_WIN_STEPS <= 0
        else min(_COMPLETE_BEST_WIN_STEPS, total)
    )

    # v8.23 transports these process-local values with the completed lease result.
    # Replace any final-level fragment measured by the older episode counter with
    # the cost of the complete replayable WIN path.
    from v8 import learning_fixes_v088 as learning

    learning._BEST_WIN_STEPS = int(_COMPLETE_BEST_WIN_STEPS)
    learning._LAST_WIN_STEPS = int(_COMPLETE_LAST_WIN_STEPS)


def _snapshot_auxiliary_states(root: str | Path):
    """Yield recent verified auxiliary states, newest first, without mutating them."""

    from v8 import snapshot

    directory = Path(root) / "snapshots"
    if not directory.is_dir():
        return
    try:
        paths = sorted(directory.glob("snapshot-*"), key=lambda path: path.name, reverse=True)
    except OSError:
        return

    yielded = 0
    for path in paths:
        if yielded >= _SNAPSHOT_RECOVERY_LIMIT:
            break
        if not path.is_dir() or not (path / "COMPLETE").is_file() or not (path / "manifest.json").is_file():
            continue
        try:
            manifest_payload = (path / "manifest.json").read_bytes()
            expected = (path / "COMPLETE").read_text(encoding="ascii").strip()
            if snapshot._sha(manifest_payload) != expected:
                continue
            manifest = json.loads(manifest_payload)
            spec = manifest.get("auxiliary_state")
            if not isinstance(spec, dict):
                continue
            payload = (path / str(spec["file"])).read_bytes()
            if snapshot._sha(payload) != str(spec["sha256"]):
                continue
            state = json.loads(payload)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if isinstance(state, dict):
            yielded += 1
            yield state


def _validated_rows_from_state(state: object):
    from v8 import trajectory_optimizer_v814 as optimizer

    if not isinstance(state, dict):
        return ()
    optimizer_state = state.get("trajectory_optimizer")
    if not isinstance(optimizer_state, dict):
        return ()
    raw_rows = optimizer_state.get("validated", ())
    if not isinstance(raw_rows, (list, tuple)):
        return ()
    rows = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        try:
            rows.append(optimizer.ValidatedTrajectory.from_dict(raw))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(rows)


def _merge_validated_rows(service, rows) -> int:
    from v8 import trajectory_optimizer_v814 as optimizer

    changed = 0
    with service._lock:
        for row in rows:
            key = optimizer._frontier_key(row.anchor, row.target)
            prior = service._validated.get(key)
            if prior is not None and (
                prior.cost,
                -prior.saved_actions,
                prior.variant_id,
            ) <= (
                row.cost,
                -row.saved_actions,
                row.variant_id,
            ):
                continue
            service._validated[key] = row
            changed += 1
    return changed


def _restore_persisted_validated_v825(service) -> int:
    """Keep validated trajectories across ordinary runs, not only --restore runs."""

    from v8 import trajectory_optimizer_v814 as optimizer

    restored = 0
    restored += _merge_validated_rows(service, optimizer._load_validated_rows(service.validated_path))
    # v8.14 used to overwrite validated.json with an empty in-memory map at start.
    # Recover recent snapshot copies as well so a prior run can repair that loss.
    for state in _snapshot_auxiliary_states(service.root.parent):
        restored += _merge_validated_rows(service, _validated_rows_from_state(state))
    return restored


def _service_start_v825(self) -> None:
    _restore_persisted_validated_v825(self)
    return _BASE_SERVICE_START(self)


def _record_from_validated_rows(game_id: str, rows, best):
    from v8 import solved_game_recovery_v821 as recovery
    from v8 import trajectory_inspection_v819 as inspection

    game = str(game_id)
    rows = tuple(row for row in rows if str(row.anchor.source_id) == game)
    for win_row in rows:
        if str(win_row.target.terminal_state) != "WIN":
            continue
        levels = recovery._validated_levels(rows, win_row)
        if levels is None:
            continue
        attempts = max(1, int(getattr(win_row, "attempts", 1)))
        successes = max(0, int(getattr(win_row, "successes", 1)))
        raw = {
            "game_id": game,
            "variant_id": str(win_row.variant_id),
            "source": "optimized",
            "terminal_state": "WIN",
            "total_cost": sum(len(level) for level in levels),
            "levels": recovery._level_payload(levels),
            "attempts": attempts,
            "successes": successes,
            "reliability": float(successes) / float(attempts),
        }
        record = inspection._validated_solution_record(raw)
        if record is not None and inspection._is_better_solution(record, best):
            best = record
    return best


def _best_visible_solution_v825(root: str | Path, game_id: str):
    """Fall back to durable snapshot validation if live sidecars were overwritten."""

    game = str(game_id)
    best = _BASE_VISIBLE_SOLUTION(root, game)
    for state in _snapshot_auxiliary_states(root):
        rows = _validated_rows_from_state(state)
        if rows:
            best = _record_from_validated_rows(game, rows, best)
    return best


def install_complete_win_trajectory_repair_v825() -> None:
    global _INSTALLED
    global _BASE_PUBLISH_RUNTIME_LEVELS, _BASE_RESET_SOLVE_METRICS
    global _BASE_SERVICE_START, _BASE_VISIBLE_SOLUTION
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import runtime_repair_v822 as v822
    from v8 import solved_game_recovery_v821 as recovery
    from v8 import trajectory_inspection_v819_fixups as visibility
    from v8 import trajectory_optimizer_v814 as optimizer

    # v8.22 accidentally bypassed v8.21's successful-level accumulator by calling
    # recovery._BASE_ENV_STEP/_RESET directly. Compose through the installed v8.21
    # hooks so probe/replay resets do not erase already solved level segments.
    v822._BASE_ENV_STEP = recovery._tracked_env_step
    v822._BASE_ENV_RESET = recovery._tracked_env_reset
    ArcGridEnvironment.step = v822._runtime_env_step
    ArcGridEnvironment.reset = v822._runtime_env_reset

    _BASE_PUBLISH_RUNTIME_LEVELS = recovery._publish_runtime_levels
    recovery._publish_runtime_levels = _publish_runtime_levels_v825

    _BASE_RESET_SOLVE_METRICS = v822._reset_solve_metrics
    v822._reset_solve_metrics = _reset_complete_solve_metrics_v825

    # Do not destroy prior validated trajectory evidence when a normal run starts.
    _BASE_SERVICE_START = optimizer.TrajectoryOptimizationService.start
    optimizer.TrajectoryOptimizationService.start = _service_start_v825

    # show-best should still work after an older run sidecar was overwritten when a
    # complete validated trajectory survives in a durable runtime snapshot.
    _BASE_VISIBLE_SOLUTION = visibility._best_visible_solution
    visibility._best_visible_solution = _best_visible_solution_v825

    _INSTALLED = True
