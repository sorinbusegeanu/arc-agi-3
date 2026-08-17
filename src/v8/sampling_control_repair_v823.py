from __future__ import annotations

import inspect
import os
import textwrap


_INSTALLED = False
_ACTOR_POOL_ENV = "ARC_AGI3_V8_ACTOR_POOL_SIZE"
_INITIAL_BREADTH_LEASE_STEPS = 2048
_PREWIN_VALIDATION_MAX_UNSOLVED_GAMES = 8
_BASE_PROGRESSIVE_SUBMIT = None
_BASE_TARGET_COMPATIBLE = None


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


def _install_bounded_adaptive_runner() -> None:
    """Patch only the two scaling defects in the default v8.19 adaptive runner.

    The source rewrite is intentionally guarded by exact snippets so a future
    implementation change fails loudly instead of silently applying a stale patch.
    Explicit custom lease configurations still go through the existing v8.19
    performance-fixup dispatcher unchanged.
    """

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

    _install_bounded_adaptive_runner()
    _INSTALLED = True
