from __future__ import annotations

import inspect
import os
import textwrap
from dataclasses import dataclass


_INSTALLED = False
_ACTOR_POOL_ENV = "ARC_AGI3_V8_ACTOR_POOL_SIZE"
_INITIAL_BREADTH_LEASE_STEPS = 2048
_PREWIN_VALIDATION_MAX_UNSOLVED_GAMES = 8
_BASE_PROGRESSIVE_SUBMIT = None
_BASE_TARGET_COMPATIBLE = None


@dataclass(frozen=True, slots=True)
class V823ActorResult:
    actor_id: int
    game_id: str
    steps: int
    wins: int
    failures: int
    levels_completed: int
    resets: int
    replans: int = 0
    planned_steps: int = 0
    strategy_stats: tuple = ()
    preference_probes: tuple = ()
    replanning_trials: tuple = ()
    pending_learning: object | None = None
    best_win_steps: int = 0
    last_win_steps: int = 0


class _ResultAdapterV823:
    """Attach process-local solve metrics to the lease result before aggregation."""

    def __init__(self, target, worker_id: int, lease) -> None:
        self.target = target
        self.worker_id = int(worker_id)
        self.lease = lease

    def put(self, row) -> None:
        from v8 import adaptive_learning_allocation_v819 as v819
        from v8 import learning_fixes_v088 as learning

        enriched = V823ActorResult(
            int(getattr(row, "actor_id", self.worker_id)),
            str(getattr(row, "game_id", self.lease.game_id)),
            int(getattr(row, "steps", 0)),
            int(getattr(row, "wins", 0)),
            int(getattr(row, "failures", 0)),
            int(getattr(row, "levels_completed", 0)),
            int(getattr(row, "resets", 0)),
            int(getattr(row, "replans", 0)),
            int(getattr(row, "planned_steps", 0)),
            tuple(getattr(row, "strategy_stats", ())),
            tuple(getattr(row, "preference_probes", ())),
            tuple(getattr(row, "replanning_trials", ())),
            getattr(row, "pending_learning", None),
            int(getattr(learning, "_BEST_WIN_STEPS", 0) or 0),
            int(getattr(learning, "_LAST_WIN_STEPS", 0) or 0),
        )
        self.target.put(v819._LeaseResult(self.worker_id, self.lease, enriched))


def requested_actor_pool(job_count: int) -> int:
    jobs = max(1, int(job_count))
    raw = os.environ.get(_ACTOR_POOL_ENV)
    if raw is None or not raw.strip():
        return jobs
    try:
        requested = int(raw)
    except ValueError:
        return jobs
    return max(1, min(jobs, requested))


def initial_unsolved_lease_steps(
    *,
    available: int,
    base_steps: int,
    initial_probe: bool,
    worker_count: int,
    game_count: int,
) -> int:
    limit = min(max(1, int(available)), max(1, int(base_steps)))
    if bool(initial_probe) and int(worker_count) < int(game_count):
        return min(limit, _INITIAL_BREADTH_LEASE_STEPS)
    return limit


def _adaptive_progress_rows_v823(
    actor_module,
    jobs,
    completed_by_game,
    active_progress,
    active_leases,
):
    """Preserve winning-path action counts through adaptive per-game aggregation."""

    totals: dict[str, dict[str, int]] = {
        game: dict(values) for game, values in completed_by_game.items()
    }
    for worker_id, progress in active_progress.items():
        lease = active_leases.get(worker_id)
        if lease is None or progress is None:
            continue
        bucket = totals.setdefault(
            lease.game_id,
            {
                "steps": 0,
                "wins": 0,
                "failures": 0,
                "levels_completed": 0,
                "replans": 0,
                "planned_steps": 0,
                "first_win_step": 0,
                "best_win_steps": 0,
                "last_win_steps": 0,
                "resets": 0,
            },
        )
        base_steps = int(bucket["steps"])
        bucket["steps"] += int(getattr(progress, "steps", 0))
        bucket["wins"] += int(getattr(progress, "wins", 0))
        bucket["failures"] += int(getattr(progress, "failures", 0))
        bucket["levels_completed"] += int(getattr(progress, "levels_completed", 0))
        bucket["replans"] += int(getattr(progress, "replans", 0))
        bucket["planned_steps"] += int(getattr(progress, "planned_steps", 0))
        local_first = int(getattr(progress, "first_win_step", 0) or 0)
        if bucket["first_win_step"] <= 0 and local_first > 0:
            bucket["first_win_step"] = base_steps + local_first
        local_best = int(getattr(progress, "best_win_steps", 0) or 0)
        if local_best > 0:
            prior_best = int(bucket.get("best_win_steps", 0) or 0)
            bucket["best_win_steps"] = (
                local_best if prior_best <= 0 else min(prior_best, local_best)
            )
        local_last = int(getattr(progress, "last_win_steps", 0) or 0)
        if local_last > 0:
            bucket["last_win_steps"] = local_last

    first_job: dict[str, object] = {}
    for job in jobs:
        first_job.setdefault(str(job.game_id), job)
    rows = []
    for job in jobs:
        game = str(job.game_id)
        values = totals.get(
            game,
            {
                "steps": 0,
                "wins": 0,
                "failures": 0,
                "levels_completed": 0,
                "replans": 0,
                "planned_steps": 0,
                "first_win_step": 0,
                "best_win_steps": 0,
                "last_win_steps": 0,
            },
        )
        if first_job[game] is not job:
            values = {key: 0 for key in values}
        kwargs = dict(
            actor_id=int(job.actor_id),
            game_id=game,
            steps=int(values.get("steps", 0)),
            wins=int(values.get("wins", 0)),
            failures=int(values.get("failures", 0)),
            levels_completed=int(values.get("levels_completed", 0)),
            replans=int(values.get("replans", 0)),
            planned_steps=int(values.get("planned_steps", 0)),
        )
        try:
            row = actor_module.ActorProgress(
                **kwargs,
                first_win_step=int(values.get("first_win_step", 0)),
                best_win_steps=int(values.get("best_win_steps", 0)),
                last_win_steps=int(values.get("last_win_steps", 0)),
            )
        except TypeError:
            try:
                row = actor_module.ActorProgress(
                    **kwargs,
                    first_win_step=int(values.get("first_win_step", 0)),
                )
            except TypeError:
                row = actor_module.ActorProgress(**kwargs)
        rows.append(row)
    return tuple(rows)


def _install_bounded_adaptive_runner() -> None:
    """Patch scaling and completed-lease metric defects in the adaptive runner."""

    from v8 import adaptive_learning_allocation_v819_performance_fix as perf

    source = textwrap.dedent(inspect.getsource(perf._adaptive_run_actor_jobs_perf))
    worker_old = "    worker_count = max(1, len(jobs))"
    worker_new = "    worker_count = _v823_requested_actor_pool(len(jobs))"
    if worker_old not in source:
        raise RuntimeError("v8.23 could not locate adaptive worker-count authority")
    source = source.replace(worker_old, worker_new, 1)

    lease_old = """        if initial_games:\n            game = initial_games.pop(0)\n        else:\n            game = choose_game()\n        mode = coordinator.choose_mode(game)\n        if coordinator.game_state(game) == v819.GameLearningState.UNSOLVED:\n            steps = min(int(available), max(1, int(base_budget_by_game.get(game, 1))))\n"""
    lease_new = """        initial_probe = bool(initial_games)\n        if initial_probe:\n            game = initial_games.pop(0)\n        else:\n            game = choose_game()\n        mode = coordinator.choose_mode(game)\n        if coordinator.game_state(game) == v819.GameLearningState.UNSOLVED:\n            steps = _v823_initial_unsolved_lease_steps(\n                available=int(available),\n                base_steps=max(1, int(base_budget_by_game.get(game, 1))),\n                initial_probe=initial_probe,\n                worker_count=worker_count,\n                game_count=len(games),\n            )\n"""
    if lease_old not in source:
        raise RuntimeError("v8.23 could not locate adaptive initial-lease authority")
    source = source.replace(lease_old, lease_new, 1)

    bucket_old = '                "first_win_step": 0,\n                "resets": 0,'
    bucket_new = '                "first_win_step": 0,\n                "best_win_steps": 0,\n                "last_win_steps": 0,\n                "resets": 0,'
    if bucket_old not in source:
        raise RuntimeError("v8.23 could not locate adaptive completed-game metric bucket")
    source = source.replace(bucket_old, bucket_new, 1)

    result_old = '                values["resets"] += int(getattr(result, "resets", 0))\n                pending = getattr(result, "pending_learning", None)'
    result_new = '''                values["resets"] += int(getattr(result, "resets", 0))
                result_best = int(getattr(result, "best_win_steps", 0) or 0)
                if result_best > 0:
                    prior_best = int(values.get("best_win_steps", 0) or 0)
                    values["best_win_steps"] = (
                        result_best if prior_best <= 0 else min(prior_best, result_best)
                    )
                result_last = int(getattr(result, "last_win_steps", 0) or 0)
                if result_last > 0:
                    values["last_win_steps"] = result_last
                pending = getattr(result, "pending_learning", None)'''
    if result_old not in source:
        raise RuntimeError("v8.23 could not locate adaptive lease-result aggregation")
    source = source.replace(result_old, result_new, 1)

    perf.__dict__["_v823_requested_actor_pool"] = requested_actor_pool
    perf.__dict__["_v823_initial_unsolved_lease_steps"] = initial_unsolved_lease_steps
    exec(compile(source, perf.__file__ or "<v8.23-adaptive>", "exec"), perf.__dict__)


def _unsolved_game_count(service) -> int:
    runtime = getattr(service, "_v819_runtime", None)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None:
        return 0
    try:
        from v8 import adaptive_learning_allocation_v819 as v819

        return sum(
            1
            for game in tuple(coordinator._games)
            if coordinator.game_state(str(game)) == v819.GameLearningState.UNSOLVED
        )
    except BaseException:
        return 0


def _bounded_progressive_submit(service, trajectory) -> bool:
    from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix
    from v8 import progressive_level_learning_v820 as progressive

    if not progressive._is_runtime_unsolved_partial(service, trajectory):
        return _BASE_PROGRESSIVE_SUBMIT(service, trajectory)

    game = str(trajectory.anchor.source_id)
    level = max(1, int(trajectory.target.levels_completed))
    # Always preserve the best partial source for post-WIN optimization.
    solve_fix._defer_pre_win_source(service, trajectory)

    # Large learning sets should spend CPU on sampling, not replay validators.
    if _unsolved_game_count(service) > _PREWIN_VALIDATION_MAX_UNSOLVED_GAMES:
        return True

    with service._v819_lock:
        stored = getattr(service, "_v819_pre_win_sources", {}).get(game, {}).get(level)
        if stored is None or str(stored.trajectory_id) != str(trajectory.trajectory_id):
            return True
        admitted = getattr(service, "_v823_pre_win_admitted_once", None)
        if admitted is None:
            admitted = set()
            service._v823_pre_win_admitted_once = admitted
        key = (game, level)
        if key in admitted:
            return True
        admitted.add(key)

    routed = bool(_BASE_PROGRESSIVE_SUBMIT(service, trajectory))
    if not routed:
        with service._v819_lock:
            getattr(service, "_v823_pre_win_admitted_once", set()).discard((game, level))
    return routed


def _target_compatible_variant_v823(view, plans, action_ids):
    """Allow cross-prefix activation only when the live planner agrees on strategy.

    Exact-prefix rows remain executable.  A seedless row from another prefix may
    still transfer, but a coincidentally matching first action is no longer enough.
    """

    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    optimizer._refresh_view_variants(view)
    source_id = str(getattr(optimizer, "_CAPTURE_SOURCE_ID", ""))
    if not source_id:
        return None
    available = {int(value) for value in action_ids}
    history = tuple(int(value) for value in getattr(optimizer, "_ACTOR_ACTION_HISTORY", ()))
    plan_outcomes = {plan.outcome_uid for plan in plans if not plan.outcome_uid.is_zero}
    plan_strategies = {plan.strategy_uid for plan in plans}
    plan_actions = {int(plan.action_id) for plan in plans}
    attempted = set(getattr(view, "_v814_attempted_variants", set()))
    rows = []
    for row in tuple(getattr(view, "_v814_variants", ())):
        if row.variant_id in attempted or row.anchor.source_id != source_id or not row.actions:
            continue
        if row.target_outcome_uid.is_zero or row.target_outcome_uid not in plan_outcomes:
            continue
        if int(row.actions[0]) not in available:
            continue
        attempts = max(1, int(getattr(row, "attempts", 1)))
        successes = max(0, int(getattr(row, "successes", 0)))
        if successes / attempts < float(v818._MIN_VALIDATED_RELIABILITY):
            continue

        prefix_exact = tuple(int(value) for value in row.anchor.prefix_actions) == history
        strategy_match = row.parent_strategy_uid in plan_strategies
        action_match = int(row.actions[0]) in plan_actions
        if not prefix_exact and not strategy_match:
            continue
        if not strategy_match and not action_match:
            continue
        rows.append(row)
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (-int(row.saved_actions), int(row.cost), -int(row.successes), row.variant_id),
    )


def install_sampling_control_repair_v823() -> None:
    global _INSTALLED, _BASE_PROGRESSIVE_SUBMIT, _BASE_TARGET_COMPATIBLE
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import runtime_repair_v822 as v822
    from v8 import trajectory_optimizer_v818 as v818
    from v8.publication import LiveReadView

    # Restore the original v8.21 contract: replay/verification can force an action,
    # but ordinary decision-point discovery is consulted only when the planner has
    # no executable plan.
    LiveReadView.plan_candidates = v822._BASE_PLAN_CANDIDATES
    sampling._VERIFICATION_REPEATS = 1

    _BASE_PROGRESSIVE_SUBMIT = v819._service_submit_v819
    v819._service_submit_v819 = _bounded_progressive_submit

    _BASE_TARGET_COMPATIBLE = v818._target_compatible_variant
    v818._target_compatible_variant = _target_compatible_variant_v823

    v819._ResultAdapter = _ResultAdapterV823
    v819._adaptive_progress_rows = _adaptive_progress_rows_v823
    _install_bounded_adaptive_runner()
    _INSTALLED = True
