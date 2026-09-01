from __future__ import annotations

"""v8.66 verified-success authority for current-run outcome metrics.

Actor counters, restored competence, level-depth proxies, and terminal reward counters
remain diagnostics.  L/G/firstWin/step-per-level are credited only from successful
trajectory evidence persisted under the unique current-run success root.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Iterable

_INSTALLED = False

SUCCESS_ROOT_ENV = "ARC_AGI3_V8_VERIFIED_SUCCESS_ROOT"
_SCHEMA_VERSION = 1
_GENERIC_GAMES = frozenset(
    ("FrozenLake-v1", "ArcAgi/Chess-v0", "ArcAgi/Sudoku-v0")
)

_BASE_RUNTIME_START = None
_BASE_RUNTIME_CLOSE = None
_BASE_RUNTIME_METRICS = None
_BASE_ACTOR_RUN_JOBS = None
_BASE_MIXED_RUN_JOBS = None
_BASE_MAKE_ADAPTER = None
_BASE_ARC_ENV_STEP = None
_BASE_TRAJECTORY_RESET_CAPTURE = None
_BASE_TRAJECTORY_CAPTURE_STEP = None
_BASE_TRAJECTORY_WRITE = None
_BASE_EFFECTIVENESS_SNAPSHOT = None
_BASE_PERIODIC_PROGRESS = None
_BASE_EFFECTIVENESS_LOG = None

_ARC_CAPTURE = threading.local()
_WRITE_LOCK = threading.Lock()


def _root_path(value) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _runtime_success_root(runtime) -> Path | None:
    return _root_path(getattr(runtime, "_v866_success_root", None))


def _configured_success_root() -> Path | None:
    return _root_path(os.environ.get(SUCCESS_ROOT_ENV))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    temp.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _trajectory_id(
    *,
    game_id: str,
    seed: int,
    terminal_state: str,
    levels_completed: int,
    actions: tuple[int, ...],
) -> str:
    payload = json.dumps(
        [
            str(game_id),
            int(seed),
            str(terminal_state),
            int(levels_completed),
            list(actions),
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16, person=b"v8.66-success").hexdigest()


def record_verified_success_v866(
    *,
    game_id: str,
    seed: int,
    terminal_state: str,
    levels_completed: int,
    actions: Iterable[int],
    capture_step: int | None,
    trajectory_id: str | None = None,
    root: str | Path | None = None,
) -> bool:
    """Persist one current-run successful trajectory atomically.

    Returning False means no authoritative current-run evidence was persisted, so
    callers must not credit the outcome metric.
    """
    success_root = _root_path(root) or _configured_success_root()
    if success_root is None:
        return False
    action_values = tuple(int(value) for value in actions)
    state = str(terminal_state).upper()
    if state not in {"LEVEL", "WIN"}:
        return False
    trajectory = str(
        trajectory_id
        or _trajectory_id(
            game_id=str(game_id),
            seed=int(seed),
            terminal_state=state,
            levels_completed=max(0, int(levels_completed)),
            actions=action_values,
        )
    )
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "trajectory_id": trajectory,
        "game_id": str(game_id),
        "seed": int(seed),
        "terminal_state": state,
        "levels_completed": max(0, int(levels_completed)),
        "actions": list(action_values),
        "action_count": len(action_values),
        "capture_step": None if capture_step is None else max(0, int(capture_step)),
        "recorded_ns": time.time_ns(),
    }
    target = (
        success_root
        / "events"
        / f"{trajectory}-{os.getpid()}-{time.time_ns()}.json"
    )
    try:
        with _WRITE_LOCK:
            _atomic_json(target, payload)
    except OSError:
        return False
    if state == "WIN":
        try:
            from v8.restored_competence_v872 import persist_generic_win_v872

            persist_generic_win_v872(success_root, payload)
        except OSError:
            # Current-run verified evidence remains authoritative even if the
            # optional cross-run generic replay promotion cannot be written.
            pass
    return True


def _read_events(root: Path | None) -> tuple[dict[str, object], ...]:
    if root is None:
        return ()
    event_root = root / "events"
    if not event_root.is_dir():
        return ()
    rows: list[dict[str, object]] = []
    for path in sorted(event_root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(raw, dict) and int(raw.get("schema_version", 0) or 0) == _SCHEMA_VERSION:
            rows.append(raw)
    return tuple(rows)


def _target_levels(game_id: str) -> int:
    return 1 if str(game_id) in _GENERIC_GAMES else 5


def verified_success_snapshot_v866(
    runtime_or_root,
    games: Iterable[str],
) -> dict[str, object]:
    games = tuple(dict.fromkeys(str(game) for game in games))
    selected = set(games)
    if isinstance(runtime_or_root, (str, Path)):
        root = _root_path(runtime_or_root)
    else:
        root = _runtime_success_root(runtime_or_root)
    rows = tuple(
        row for row in _read_events(root) if str(row.get("game_id", "")) in selected
    )

    won_games: set[str] = set()
    deepest_by_game: dict[str, int] = {}
    first_win_steps: dict[str, int] = {}
    level_events = 0
    win_events = 0
    for row in rows:
        game = str(row.get("game_id", ""))
        state = str(row.get("terminal_state", "")).upper()
        levels = max(0, int(row.get("levels_completed", 0) or 0))
        if state == "LEVEL":
            level_events += 1
            deepest_by_game[game] = max(deepest_by_game.get(game, 0), levels)
        elif state == "WIN":
            level_events += 1
            win_events += 1
            won_games.add(game)
            deepest_by_game[game] = _target_levels(game)
            step = row.get("capture_step")
            if step is not None:
                try:
                    step_value = max(0, int(step))
                except (TypeError, ValueError):
                    step_value = 0
                if step_value > 0:
                    previous = first_win_steps.get(game)
                    if previous is None or step_value < previous:
                        first_win_steps[game] = step_value

    solved_levels = 0
    target_levels = 0
    for game in games:
        target = _target_levels(game)
        target_levels += target
        solved_levels += min(target, max(0, deepest_by_game.get(game, 0)))

    games_won = len(won_games)
    first_values = tuple(first_win_steps.values())
    return {
        "schema_version": _SCHEMA_VERSION,
        "source": "CURRENT_RUN_VERIFIED_SUCCESS_TRAJECTORIES",
        "success_root": None if root is None else str(root),
        "selected_games": list(games),
        "successful_trajectories": len(rows),
        "verified_level_advance_events": level_events,
        "verified_win_trajectory_events": win_events,
        "current_run_levels_solved": solved_levels,
        "current_run_games_won": games_won,
        "level_target_count": target_levels,
        "game_target_count": len(games),
        "level_solve_rate_pct": (
            0.0 if target_levels <= 0 else 100.0 * solved_levels / target_levels
        ),
        "game_solve_rate_pct": (
            0.0 if not games else 100.0 * games_won / len(games)
        ),
        "first_win_steps_by_game": dict(sorted(first_win_steps.items())),
        "mean_first_win_step": (
            None
            if not first_values
            else float(sum(first_values)) / float(len(first_values))
        ),
    }


def verified_success_from_summary_v866(summary) -> dict[str, object]:
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    if isinstance(metrics, dict):
        verified = metrics.get("verified_success")
        if isinstance(verified, dict):
            return dict(verified)
    verified = summary.get("verified_success") if isinstance(summary, dict) else None
    return dict(verified) if isinstance(verified, dict) else {}


def _prepare_runtime_success_root(runtime) -> Path:
    existing = _runtime_success_root(runtime)
    if existing is not None:
        os.environ[SUCCESS_ROOT_ENV] = str(existing)
        return existing
    base = Path(getattr(runtime, "root"))
    root = base / "verified_success" / f"run-{os.getpid()}-{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=False)
    runtime._v866_success_root = str(root)
    os.environ[SUCCESS_ROOT_ENV] = str(root)
    _atomic_json(
        root / "run.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "pid": os.getpid(),
            "started_ns": time.time_ns(),
            "source": "CURRENT_RUN_ONLY",
        },
    )
    from v8.restored_competence_v872 import (
        capture_startup_restored_competence_v872,
    )

    capture_startup_restored_competence_v872(base, root)
    return root


def _runtime_start_v866(self, *args, **kwargs):
    _prepare_runtime_success_root(self)
    return _BASE_RUNTIME_START(self, *args, **kwargs)


def _runtime_close_v866(self, *args, **kwargs):
    root = _runtime_success_root(self)
    try:
        return _BASE_RUNTIME_CLOSE(self, *args, **kwargs)
    finally:
        if root is not None and os.environ.get(SUCCESS_ROOT_ENV) == str(root):
            os.environ.pop(SUCCESS_ROOT_ENV, None)


def _selected_games(runtime) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(game)
            for game in getattr(runtime, "_v866_selected_games", ())
        )
    )


def _runtime_metrics_v866(self):
    from v8.restored_competence_v872 import restored_competence_snapshot_v872

    payload = _BASE_RUNTIME_METRICS(self)
    if not isinstance(payload, dict):
        payload = dict(payload)
    payload["verified_success"] = verified_success_snapshot_v866(
        self, _selected_games(self)
    )
    payload["restored_competence"] = restored_competence_snapshot_v872(
        _runtime_success_root(self), _selected_games(self)
    )
    return payload


def _set_run_scope(runtime, jobs) -> None:
    jobs = tuple(jobs)
    runtime._v866_selected_games = tuple(
        dict.fromkeys(str(getattr(job, "game_id", "")) for job in jobs)
    )
    runtime._v866_total_step_budget = sum(
        max(0, int(getattr(job, "steps", 0))) for job in jobs
    )


def _run_actor_jobs_v866(runtime, jobs, **kwargs):
    jobs = tuple(jobs)
    if not bool(getattr(runtime, "_v866_mixed_scope_active", False)):
        _set_run_scope(runtime, jobs)
    return _BASE_ACTOR_RUN_JOBS(runtime, jobs, **kwargs)


def _run_mixed_actor_jobs_v866(runtime, jobs, **kwargs):
    jobs = tuple(jobs)
    _set_run_scope(runtime, jobs)
    runtime._v866_mixed_scope_active = True
    try:
        return _BASE_MIXED_RUN_JOBS(runtime, jobs, **kwargs)
    finally:
        runtime._v866_mixed_scope_active = False


class _VerifiedAdapterProxy:
    __slots__ = ("_inner", "_game_id", "_seed", "_actions", "_steps")

    def __init__(self, inner, game_id: str, seed: int):
        self._inner = inner
        self._game_id = str(game_id)
        self._seed = int(seed)
        self._actions: list[int] = []
        self._steps = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def step(self, action):
        self._steps += 1
        self._actions.append(int(action))
        result = self._inner.step(action)
        boundary = self._inner.cognitive_boundary_event()
        if not bool(boundary.continuation) and int(boundary.primary_valence) > 0:
            record_verified_success_v866(
                game_id=self._game_id,
                seed=self._seed,
                terminal_state="WIN",
                levels_completed=1,
                actions=tuple(self._actions),
                capture_step=self._steps,
            )
        return result

    def reset(self):
        self._actions = []
        return self._inner.reset()

    def close(self):
        return self._inner.close()


def _make_adapter_v866(game_id: str, *, seed: int = 0):
    return _VerifiedAdapterProxy(
        _BASE_MAKE_ADAPTER(game_id, seed=seed),
        str(game_id),
        int(seed),
    )


def _reset_capture_v866(job=None):
    _ARC_CAPTURE.step = 0
    return _BASE_TRAJECTORY_RESET_CAPTURE(job)


def _capture_env_step_v866(self, action):
    try:
        from v8 import trajectory_optimizer_v814 as trajectory
        active = bool(getattr(trajectory, "_CAPTURE_ACTIVE", False))
    except BaseException:
        active = False
    if active:
        _ARC_CAPTURE.step = int(getattr(_ARC_CAPTURE, "step", 0)) + 1
    return _BASE_TRAJECTORY_CAPTURE_STEP(self, action)


def _write_successful_trajectory_v866(row) -> None:
    _BASE_TRAJECTORY_WRITE(row)
    from v8 import information_flow_diagnostics as flow

    flow.add_counters(
        "trajectory_optimizer",
        successful_trajectories_produced=1,
        trajectories_submitted=1,
    )
    flow.emit_bounded(
        "trajectory_optimizer", "successful_trajectory_produced",
        input_count=1, output_count=1,
        examples=(
            {"trajectory_id": str(row.trajectory_id),
             "producer_source_stage": "verified_success_capture",
             "source_world": str(row.anchor.source_id),
             "submitted_to_optimizer": True,
             "optimizer_received": None,
             "counted_in_trajectories_seen": None,
             "optimizer_candidates_generated": None,
             "validation_attempts": None,
             "validation_result": None,
             "accepted_variant_id": None,
             "rejection_reason": None},
        ),
    )
    try:
        record_verified_success_v866(
            game_id=str(row.anchor.source_id),
            seed=int(row.anchor.seed),
            terminal_state=str(row.target.terminal_state),
            levels_completed=int(row.target.levels_completed),
            actions=tuple(row.actions),
            capture_step=max(0, int(getattr(_ARC_CAPTURE, "step", 0))) or None,
            trajectory_id=str(row.trajectory_id),
        )
    except (AttributeError, TypeError, ValueError):
        # Optimizer capture remains operational; malformed/noncanonical rows earn no
        # verified-success credit.
        return


def _runtime_has_verified_scope(runtime) -> bool:
    return _runtime_success_root(runtime) is not None


def learning_effectiveness_snapshot_v866(
    runtime,
    coordinator,
    completed_by_game,
    active_progress,
    active_leases,
):
    base = _BASE_EFFECTIVENESS_SNAPSHOT(
        runtime,
        coordinator,
        completed_by_game,
        active_progress,
        active_leases,
    )
    if not _runtime_has_verified_scope(runtime):
        # Preserve direct legacy helper behavior for isolated callers. Production
        # runtime.start() always establishes the v8.66 run scope.
        return base

    games = _selected_games(runtime)
    if not games:
        outcomes = getattr(coordinator, "_games", ())
        games = tuple(dict.fromkeys(str(game) for game in outcomes))
    verified = verified_success_snapshot_v866(runtime, games)
    from v8 import information_flow_diagnostics as flow

    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    if service is not None:
        with service._lock:
            optimizer_counters = {
                "successful_trajectories_produced": int(verified["successful_trajectories"]),
                "trajectories_submitted": int(verified["successful_trajectories"]),
                "trajectories_received": int(flow.counter_snapshot("trajectory_optimizer").get("trajectories_received", 0)),
                "trajectories_seen": int(service._trajectories_seen),
                "candidates_generated": int(service._candidates_generated),
                "validations": int(service._validations),
                "validation_successes": int(service._validation_successes),
                "accepted_variants": len(service._validated),
            }
        bypassed = max(
            0,
            optimizer_counters["trajectories_received"]
            - optimizer_counters["trajectories_seen"],
        )
        flow.emit(
            "trajectory_optimizer", "pipeline_summary",
            input_count=optimizer_counters["successful_trajectories_produced"],
            output_count=optimizer_counters["accepted_variants"],
            rejection_counts=(
                {"received_without_legacy_trajectories_seen_increment": bypassed}
                if bypassed else {}
            ),
            fields={"counters": optimizer_counters,
                    "trajectories_seen_counter_owner": "legacy_edit_source_queue",
                    "source_validation_bypasses_trajectories_seen": True,
                    "counter_scope": "current_run_verified_and_optimizer_service"},
        )
    from v8.restored_competence_v872 import restored_competence_snapshot_v872

    restored = restored_competence_snapshot_v872(_runtime_success_root(runtime), games)

    effectiveness = base.get("effectiveness", {})
    outcome = effectiveness.get("outcome_effectiveness", {})
    efficiency = effectiveness.get("efficiency", {})
    scope = base.get("scope", {})

    raw_level_events = int(outcome.get("current_run_level_advance_events", 0) or 0)
    raw_games_won = int(outcome.get("current_run_games_won", 0) or 0)
    outcome.update(
        {
            "level_solve_rate_pct": float(verified["level_solve_rate_pct"]),
            "game_solve_rate_pct": float(verified["game_solve_rate_pct"]),
            "current_run_levels_solved": int(
                verified["current_run_levels_solved"]
            ),
            "current_run_level_advance_events": int(
                verified["verified_level_advance_events"]
            ),
            "current_run_games_won": int(verified["current_run_games_won"]),
            "successful_trajectories": int(
                verified["successful_trajectories"]
            ),
            "outcome_source": verified["source"],
            "actor_reported_level_advance_events": raw_level_events,
            "actor_reported_games_won": raw_games_won,
        }
    )

    solved_levels = int(verified["current_run_levels_solved"])
    used_steps = max(0, int(scope.get("steps", 0) or 0))
    efficiency["steps_per_solved_level"] = (
        None if solved_levels <= 0 else float(used_steps) / float(solved_levels)
    )
    efficiency["mean_first_win_step"] = verified["mean_first_win_step"]
    effectiveness["restored_competence"] = restored
    scope["games"] = len(games)
    scope["outcome_source"] = verified["source"]
    base["verified_success"] = verified
    base["restored_competence"] = restored
    return base


def _write_learning_effectiveness_log_v866(
    runtime,
    coordinator,
    completed_by_game,
    active_progress,
    active_leases,
) -> None:
    """Production v8.66 log/stdout path with full-run budget authority."""
    from v8 import learning_effectiveness_report_v850 as report

    payload = learning_effectiveness_snapshot_v866(
        runtime,
        coordinator,
        completed_by_game,
        active_progress,
        active_leases,
    )
    target = Path(runtime.root) / report._REPORT_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
    except OSError:
        pass

    # During a mixed run this callback belongs to the inner ARC allocator.  Its
    # progress excludes concurrent generic actors while the runtime budget covers
    # every job, so printing that ratio beside the all-actor dedicated reporter
    # creates a second, lower percentage that appears to move backwards.  Keep
    # the detailed JSONL record, but leave mixed-run stdout to the all-actor
    # reporter. Pure ARC runs retain this authoritative presentation.
    if bool(getattr(runtime, "_v866_mixed_scope_active", False)):
        return

    scope = payload.get("scope", {})
    used_steps = int(scope.get("steps", 0)) if isinstance(scope, dict) else 0
    budget = max(
        0,
        int(
            getattr(
                runtime,
                "_v866_total_step_budget",
                getattr(runtime, "_v850_total_step_budget", 0),
            )
        ),
    )
    budget_pct = None if budget <= 0 else report._pct(used_steps, budget)

    now = time.monotonic()
    last_steps = int(getattr(runtime, "_v850_last_effectiveness_stdout_steps", -1))
    last_time = float(getattr(runtime, "_v850_last_effectiveness_stdout_time", -1.0))
    if used_steps != last_steps or last_time < 0.0 or now - last_time >= 30.0:
        report._emit_effectiveness_stdout(
            payload, budget_consumed_pct=budget_pct
        )
        runtime._v850_last_effectiveness_stdout_steps = used_steps
        runtime._v850_last_effectiveness_stdout_time = now


def _periodic_progress_line_v866(rows, total_steps, baseline=None) -> str:
    from v8.restored_competence_v872 import restored_competence_snapshot_v872

    values = tuple(rows)
    root = _configured_success_root()
    if root is None:
        return _BASE_PERIODIC_PROGRESS(values, total_steps, baseline)
    games = tuple(
        dict.fromkeys(str(getattr(row, "game_id", "")) for row in values)
    )
    verified = verified_success_snapshot_v866(root, games)
    restored = restored_competence_snapshot_v872(root, games)
    used_steps = sum(max(0, int(getattr(row, "steps", 0))) for row in values)
    planned_steps = sum(
        max(0, int(getattr(row, "planned_steps", 0))) for row in values
    )
    budget = max(0, int(total_steps or 0))
    budget_pct = 0.0 if budget <= 0 else 100.0 * used_steps / budget
    m7 = 0.0 if used_steps <= 0 else 100.0 * planned_steps / used_steps
    solved = int(verified["current_run_levels_solved"])
    step_per_level = None if solved <= 0 else float(used_steps) / solved

    from v8 import learning_effectiveness_report_v850 as report

    return (
        f"{max(0.0, min(100.0, budget_pct)):.0f}% - effectiveness "
        f"L={float(verified['level_solve_rate_pct']):.1f}% "
        f"G={float(verified['game_solve_rate_pct']):.1f}% "
        f"RestL={float(restored['restored_level_solve_rate_pct']):.1f}% "
        f"RestG={float(restored['restored_game_solve_rate_pct']):.1f}% "
        f"M7={m7:.1f}% "
        "M7val=- XferVal=- Opt=- Prod=- "
        f"step/L={report._compact_number(step_per_level)} "
        f"firstWin={report._compact_number(verified['mean_first_win_step'])}"
    )


def install_verified_success_metrics_v866() -> None:
    global _INSTALLED
    global _BASE_RUNTIME_START, _BASE_RUNTIME_CLOSE, _BASE_RUNTIME_METRICS
    global _BASE_ACTOR_RUN_JOBS, _BASE_MIXED_RUN_JOBS, _BASE_MAKE_ADAPTER
    global _BASE_ARC_ENV_STEP
    global _BASE_TRAJECTORY_RESET_CAPTURE, _BASE_TRAJECTORY_CAPTURE_STEP
    global _BASE_TRAJECTORY_WRITE, _BASE_EFFECTIVENESS_SNAPSHOT
    global _BASE_PERIODIC_PROGRESS, _BASE_EFFECTIVENESS_LOG
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import learning_effectiveness_report_v850 as report
    from v8 import mixed_environment_v859 as mixed
    from v8 import reporter
    from v8 import trajectory_optimizer_v814 as trajectory
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_RUNTIME_START = V82ContinuousMemoryRuntime.start
    _BASE_RUNTIME_CLOSE = V82ContinuousMemoryRuntime.close
    _BASE_RUNTIME_METRICS = V82ContinuousMemoryRuntime.metrics
    V82ContinuousMemoryRuntime.start = _runtime_start_v866
    V82ContinuousMemoryRuntime.close = _runtime_close_v866
    V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v866

    _BASE_ACTOR_RUN_JOBS = actor_module.run_actor_jobs
    actor_module.run_actor_jobs = _run_actor_jobs_v866

    _BASE_MIXED_RUN_JOBS = mixed.run_mixed_actor_jobs
    mixed.run_mixed_actor_jobs = _run_mixed_actor_jobs_v866
    _BASE_MAKE_ADAPTER = mixed.make_adapter
    mixed.make_adapter = _make_adapter_v866

    _BASE_TRAJECTORY_RESET_CAPTURE = trajectory._reset_capture
    trajectory._reset_capture = _reset_capture_v866
    # v8.14's original capture wrapper is still in the composed production chain,
    # even though later layers own the public adapter method. Insert beneath that
    # wrapper so all later temporal, recovery, click, and sampling authorities keep
    # their exact bindings while every real captured ARC step reaches v8.66.
    _BASE_ARC_ENV_STEP = trajectory._BASE_ENV_STEP
    _BASE_TRAJECTORY_CAPTURE_STEP = _BASE_ARC_ENV_STEP
    trajectory._BASE_ENV_STEP = _capture_env_step_v866
    _BASE_TRAJECTORY_WRITE = trajectory._write_successful_trajectory
    trajectory._write_successful_trajectory = _write_successful_trajectory_v866

    _BASE_EFFECTIVENESS_SNAPSHOT = report.learning_effectiveness_snapshot_v850
    report.learning_effectiveness_snapshot_v850 = learning_effectiveness_snapshot_v866

    _BASE_EFFECTIVENESS_LOG = report._write_learning_effectiveness_log
    report._write_learning_effectiveness_log = _write_learning_effectiveness_log_v866

    _BASE_PERIODIC_PROGRESS = reporter.format_periodic_progress_line
    reporter.format_periodic_progress_line = _periodic_progress_line_v866

    _INSTALLED = True
