from __future__ import annotations

import os
import threading
from dataclasses import dataclass


_INSTALLED = False
_LIFECYCLE_START_DELAY_SECONDS = 300.0
_BASE_LIFECYCLE_WORKER = None
_BASE_ACTOR_WORKER = None
_BASE_PLAN_CANDIDATES = None
_BASE_FORCED_ACTION = None
_BASE_DISCOVERY_ACTION = None
_BASE_OBSERVE_TRANSITION = None
_BASE_ENV_STEP = None
_BASE_ENV_RESET = None
_PROBE_STATE = threading.local()
_RUNTIME_EPISODE_BOUNDARIES: dict[int, list[int]] = {}


def _delayed_lifecycle_worker(supervisor) -> None:
    delay = max(
        0.0,
        float(
            getattr(
                supervisor,
                "_v822_lifecycle_start_delay_seconds",
                _LIFECYCLE_START_DELAY_SECONDS,
            )
        ),
    )
    if delay > 0.0 and supervisor._stop.wait(delay):
        return
    if _BASE_LIFECYCLE_WORKER is not None:
        _BASE_LIFECYCLE_WORKER(supervisor)


def _probe_required(sampler, *, level: int, context: int, actions) -> bool:
    available = {int(value) for value in actions}
    if not available:
        return False
    point = sampler.points.get((int(level), int(context)))
    if point is None:
        return True
    if point.successful_action is not None:
        return False
    untested = available - set(int(value) for value in point.tested_actions)
    if not untested:
        return False
    transfer = sampler.transfer_action
    if (
        transfer is not None
        and int(level) > int(sampler.transfer_from_level)
        and int(transfer) in untested
    ):
        return True
    return bool(untested)


def _forced_action_v822(self, **kwargs):
    action = _BASE_FORCED_ACTION(self, **kwargs)
    if action is not None:
        _PROBE_STATE.before_plan = False
        return action
    _PROBE_STATE.before_plan = _probe_required(
        self,
        level=int(kwargs.get("level", 0)),
        context=int(kwargs.get("context", 0)),
        actions=kwargs.get("actions", ()),
    )
    return None


def _plan_candidates_v822(self, context_signature, action_ids, **kwargs):
    from v8 import decision_point_sampling_v821 as sampling

    if sampling._decision_mode_enabled() and bool(
        getattr(_PROBE_STATE, "before_plan", False)
    ):
        return ()
    return _BASE_PLAN_CANDIDATES(self, context_signature, action_ids, **kwargs)


def _discovery_action_v822(self, **kwargs):
    try:
        return _BASE_DISCOVERY_ACTION(self, **kwargs)
    finally:
        _PROBE_STATE.before_plan = False


def _best_frontier_v822(self):
    candidates = [
        row
        for row in self.points.values()
        if row.successful_action is None and row.untested()
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            int(row.priority),
            -len(row.anchor),
            int(row.level),
            -int(row.context),
        ),
    )


def _observe_transition_v822(self, **kwargs) -> None:
    intervention = self.current
    verification = self.verification
    failed_verification = bool(
        intervention is not None
        and intervention.kind == "VERIFICATION"
        and not (
            bool(kwargs.get("level_advanced", False))
            or str(kwargs.get("terminal_state", "")) == "WIN"
        )
    )
    _BASE_OBSERVE_TRANSITION(self, **kwargs)
    if not failed_verification or verification is None:
        return
    point = self.points.get(verification.point_key)
    if point is not None:
        point.successful_action = None
        point.priority = min(int(point.priority), 3)
    self.pending_reset = None
    self._schedule_point(_best_frontier_v822(self))


def _episode_levels(actions, boundaries, expected_levels: int):
    full = tuple(int(value) for value in actions)
    expected = max(1, int(expected_levels))
    if not full:
        return None
    cuts = sorted({int(value) for value in boundaries if 0 < int(value) < len(full)})
    if len(cuts) != expected - 1:
        return None
    levels = []
    start = 0
    for stop in cuts:
        levels.append(tuple(full[start:stop]))
        start = stop
    levels.append(tuple(full[start:]))
    if len(levels) != expected or any(not level for level in levels):
        return None
    return tuple(levels)


def _runtime_env_reset(self, *args, **kwargs):
    result = _BASE_ENV_RESET(self, *args, **kwargs)
    _RUNTIME_EPISODE_BOUNDARIES[id(self)] = []
    return result


def _runtime_env_step(self, action):
    from v8 import solved_game_recovery_v821 as recovery
    from v8 import trajectory_optimizer_v814 as optimizer

    key = id(self)
    capture_active = bool(getattr(optimizer, "_CAPTURE_ACTIVE", False))
    history_before = (
        tuple(int(value) for value in optimizer._ACTOR_ACTION_HISTORY)
        if capture_active
        else ()
    )
    prior_level = int(getattr(self, "last_levels_completed", 0))
    result = _BASE_ENV_STEP(self, action)
    if not capture_active:
        return result

    state = str(getattr(self, "last_outcome_state", ""))
    current_level = int(getattr(self, "last_levels_completed", prior_level))
    full_actions = (*history_before, int(action))
    boundaries = _RUNTIME_EPISODE_BOUNDARIES.setdefault(key, [])

    if current_level > prior_level:
        cutoff = len(full_actions)
        if cutoff > 0 and cutoff not in boundaries:
            boundaries.append(cutoff)

    if state == "WIN":
        expected_levels = max(current_level, prior_level + 1)
        levels = _episode_levels(full_actions, boundaries, expected_levels)
        if levels is not None:
            game_id = str(getattr(self, "_v821_recovery_game_id", "")) or str(
                getattr(optimizer, "_CAPTURE_SOURCE_ID", "")
            )
            recovery._publish_runtime_levels(game_id, levels)
        _RUNTIME_EPISODE_BOUNDARIES[key] = []
    elif state == "GAME_OVER" or bool(
        getattr(self, "last_step_was_reset_boundary", False)
    ):
        _RUNTIME_EPISODE_BOUNDARIES[key] = []
    return result


def _reset_solve_metrics() -> None:
    from v8 import learning_fixes_v088 as learning

    learning._EPISODE_STEPS = 0
    learning._FIRST_WIN_STEPS = 0
    learning._BEST_WIN_STEPS = 0
    learning._LAST_WIN_STEPS = 0


def _actor_worker_v822(*, job, **kwargs):
    from v8 import actor as actor_module
    from v8 import behavior_recovery as behavior
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import model as model_module
    from v8 import primary_valence as primary

    if not sampling._decision_mode_enabled():
        return _BASE_ACTOR_WORKER(job=job, **kwargs)

    prior_mode = os.environ.get(behavior._ACTOR_MODE_ENV)
    prior_epsilon = os.environ.get(behavior._ACTOR_EPSILON_ENV)
    prior_seed = os.environ.get(behavior._ACTOR_SEED_ENV)
    prior_experience = model_module.ExperienceEvent
    prior_capture = bool(primary._CAPTURE_ACTIVE)

    os.environ[behavior._ACTOR_MODE_ENV] = "1"
    os.environ[behavior._ACTOR_EPSILON_ENV] = str(
        max(0.0, min(1.0, float(job.epsilon)))
    )
    os.environ[behavior._ACTOR_SEED_ENV] = str(int(job.seed))
    _reset_solve_metrics()
    primary._reset_actor_capture()
    primary._CAPTURE_ACTIVE = True
    # v8.21 imported ExperienceEvent directly from v8.model, bypassing the
    # installed actor factory chain.  Route that import through the chain for the
    # duration of this one actor lease so primary-valence/progress capture is live.
    model_module.ExperienceEvent = actor_module.ExperienceEvent
    try:
        return _BASE_ACTOR_WORKER(job=job, **kwargs)
    finally:
        model_module.ExperienceEvent = prior_experience
        primary._CAPTURE_ACTIVE = prior_capture
        primary._reset_actor_capture()
        for key, prior in (
            (behavior._ACTOR_MODE_ENV, prior_mode),
            (behavior._ACTOR_EPSILON_ENV, prior_epsilon),
            (behavior._ACTOR_SEED_ENV, prior_seed),
        ):
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _game_summary_v822(rows, *, levels_per_game: int = 5):
    from v8 import diagnostics as diagnostics

    grouped = diagnostics._group_games(rows)
    if not grouped:
        return 0.0, 0.0, 0, 0
    games = len(grouped)
    denominator = max(1, int(levels_per_game))
    won = 0
    solved_levels = 0
    for lane_rows in grouped.values():
        game_won = any(int(row.wins) > 0 for row in lane_rows)
        won += int(game_won)
        deepest = max((diagnostics._deepest_level(row) for row in lane_rows), default=0)
        if not game_won:
            # Reaching the environment's final level counter is not a solved final
            # level until a terminal WIN is observed.
            deepest = min(deepest, max(0, denominator - 1))
        solved_levels += min(denominator, deepest)
    total_levels = games * denominator
    return (
        100.0 * won / games,
        100.0 * solved_levels / total_levels,
        int(won),
        int(games),
    )


def _format_game_rate_line_v822(rows) -> str:
    from v8 import diagnostics as diagnostics

    rows = tuple(rows)
    win_rate, level_rate, solved_games, games = diagnostics.game_summary(rows)
    grouped = diagnostics._group_games(rows)
    details = []
    for game_id, lane_rows in sorted(grouped.items()):
        solved_rows = [row for row in lane_rows if int(getattr(row, "wins", 0)) > 0]
        if not solved_rows:
            continue
        best_values = [int(getattr(row, "best_win_steps", 0) or 0) for row in solved_rows]
        last_values = [int(getattr(row, "last_win_steps", 0) or 0) for row in solved_rows]
        best = min((value for value in best_values if value > 0), default=0)
        latest = max(solved_rows, key=lambda row: int(getattr(row, "steps", 0) or 0))
        last = int(getattr(latest, "last_win_steps", 0) or 0)
        if best > 0 and last > 0 and best != last:
            details.append(
                f"{game_id}:best_win_actions={best},last_win_actions={last}"
            )
        elif best > 0:
            details.append(f"{game_id}:win_actions={best}")
        else:
            details.append(f"{game_id}:win_observed")
    suffix = "" if not details else " (" + "; ".join(details) + ")"
    return (
        f"current_run_wins={win_rate:.1f}% current_run_levels_solved={level_rate:.1f}% "
        f"current_run_solved_games={solved_games}/{games}{suffix}"
    )


def install_runtime_repair_v822() -> None:
    global _INSTALLED, _BASE_LIFECYCLE_WORKER, _BASE_ACTOR_WORKER
    global _BASE_PLAN_CANDIDATES, _BASE_FORCED_ACTION, _BASE_DISCOVERY_ACTION
    global _BASE_OBSERVE_TRANSITION, _BASE_ENV_STEP, _BASE_ENV_RESET
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import actor as actor_module
    from v8 import dedicated_lifecycle_v813 as lifecycle
    from v8 import diagnostics
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import solved_game_recovery_v821 as recovery
    from v8.publication import LiveReadView

    _BASE_LIFECYCLE_WORKER = lifecycle._lifecycle_worker
    lifecycle._lifecycle_worker = _delayed_lifecycle_worker

    _BASE_FORCED_ACTION = sampling.DecisionPointSampler.forced_action
    _BASE_DISCOVERY_ACTION = sampling.DecisionPointSampler.discovery_action
    _BASE_OBSERVE_TRANSITION = sampling.DecisionPointSampler.observe_transition
    sampling.DecisionPointSampler.forced_action = _forced_action_v822
    sampling.DecisionPointSampler.discovery_action = _discovery_action_v822
    sampling.DecisionPointSampler.observe_transition = _observe_transition_v822
    sampling.DecisionPointSampler._best_frontier = _best_frontier_v822

    _BASE_PLAN_CANDIDATES = LiveReadView.plan_candidates
    LiveReadView.plan_candidates = _plan_candidates_v822

    _BASE_ENV_STEP = recovery._BASE_ENV_STEP
    _BASE_ENV_RESET = recovery._BASE_ENV_RESET
    ArcGridEnvironment.step = _runtime_env_step
    ArcGridEnvironment.reset = _runtime_env_reset

    _BASE_ACTOR_WORKER = actor_module.actor_worker
    actor_module.actor_worker = _actor_worker_v822

    diagnostics.game_summary = _game_summary_v822
    diagnostics.game_rates = lambda rows: tuple(_game_summary_v822(rows)[:2])
    diagnostics.format_game_rate_line = _format_game_rate_line_v822
    _INSTALLED = True
