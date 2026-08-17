from __future__ import annotations

from typing import Iterable


_INSTALLED = False
_BASE_SERVICE_SUBMIT = None
_BASE_GENERATE_V820 = None
_BASE_SUBMIT_NEXT_SOURCE = None
_BASE_RUN_ACTOR_JOBS = None
_PREWIN_CHEAP_TRAJECTORIES: set[str] = set()


def _full_cost(trajectory) -> int:
    return len(tuple(trajectory.anchor.prefix_actions)) + len(tuple(trajectory.actions))


def _is_runtime_unsolved_partial(service, trajectory) -> bool:
    runtime = getattr(service, "_v819_runtime", None)
    if runtime is None:
        return False
    if int(getattr(trajectory, "round_index", 0)) != 0:
        return False
    if str(getattr(trajectory.target, "terminal_state", "")) == "WIN":
        return False
    try:
        from v8 import adaptive_learning_allocation_v819 as v819

        return (
            runtime._v819_adaptive_learning.game_state(str(trajectory.anchor.source_id))
            == v819.GameLearningState.UNSOLVED
        )
    except BaseException:
        return False


def _submit_progressive_partial(service, trajectory) -> bool:
    """Validate/minimize only the current best pre-WIN level source.

    Partial level successes remain stored for post-WIN full optimization, but the
    best observed source for each game/level is admitted immediately to source
    validation and then to v8.20 DIRECT_ACTION/TRUNCATE minimization.  Expensive
    deletion search stays disabled until the game has a validated WIN.
    """

    if not _is_runtime_unsolved_partial(service, trajectory):
        return _BASE_SERVICE_SUBMIT(service, trajectory)

    from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix

    game = str(trajectory.anchor.source_id)
    level = max(1, int(trajectory.target.levels_completed))
    solve_fix._defer_pre_win_source(service, trajectory)

    with service._v819_lock:
        pending = getattr(service, "_v819_pre_win_sources", {})
        stored = pending.get(game, {}).get(level)
        if stored is None or str(stored.trajectory_id) != str(trajectory.trajectory_id):
            return True

        admitted = getattr(service, "_v820_pre_win_admitted", None)
        if admitted is None:
            admitted = {}
            service._v820_pre_win_admitted = admitted
        key = (game, level)
        rank = (_full_cost(trajectory), str(trajectory.trajectory_id))
        prior = admitted.get(key)
        if prior is not None and tuple(rank) >= tuple(prior):
            return True
        admitted[key] = rank

    trajectory_id = str(trajectory.trajectory_id)
    _PREWIN_CHEAP_TRAJECTORIES.add(trajectory_id)
    routed = bool(solve_fix._BASE_SERVICE_SUBMIT_V819(service, trajectory))
    if not routed:
        _PREWIN_CHEAP_TRAJECTORIES.discard(trajectory_id)
        with service._v819_lock:
            admitted = getattr(service, "_v820_pre_win_admitted", {})
            if admitted.get((game, level)) == rank:
                admitted.pop((game, level), None)
    return routed


def _generate_progressive_v820(source, config=None):
    trajectory_id = str(source.trajectory_id)
    if trajectory_id in _PREWIN_CHEAP_TRAJECTORIES:
        # The optimizer loop has already routed TARGET_MINIMIZE before calling the
        # generator.  Returning no deletion candidates gives pre-WIN levels only
        # DIRECT_ACTION plus successful-prefix truncation.
        _PREWIN_CHEAP_TRAJECTORIES.discard(trajectory_id)
        return ()
    return _BASE_GENERATE_V820(source, config)


def _submit_next_source_progressive(service, candidate, validated) -> None:
    runtime = getattr(service, "_v819_runtime", None)
    if runtime is not None and str(candidate.source.target.terminal_state) != "WIN":
        try:
            from v8 import adaptive_learning_allocation_v819 as v819

            if (
                runtime._v819_adaptive_learning.game_state(
                    str(candidate.source.anchor.source_id)
                )
                == v819.GameLearningState.UNSOLVED
            ):
                # Do not recursively enter ddmin from a successful pre-WIN direct
                # action/truncation.  The validated prefix is already published and
                # reusable; full recursive minimization begins after a validated WIN.
                return
        except BaseException:
            pass
    _BASE_SUBMIT_NEXT_SOURCE(service, candidate, validated)


class _BatchingRuntimeProxy:
    """Merge the adaptive runner's one-at-a-time learning callbacks in small bursts."""

    def __init__(self, runtime, *, batch_size: int) -> None:
        self._runtime = runtime
        self._batch_size = max(1, int(batch_size))
        self._pending: list[object] = []

    def __getattr__(self, name: str):
        return getattr(self._runtime, name)

    def record_actor_results(self, rows: Iterable[object]) -> None:
        from v8 import actor as actor_module

        values = tuple(rows)
        if not values:
            return
        if not all(isinstance(row, actor_module.ActorLearningBatch) for row in values):
            self.flush()
            self._runtime.record_actor_results(values)
            return
        self._pending.extend(values)
        if len(self._pending) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        from v8 import actor as actor_module

        rows = tuple(self._pending)
        self._pending.clear()
        self._runtime.record_actor_results(actor_module._merge_learning_batches(rows))


def _run_actor_jobs_batched(runtime, jobs, **kwargs):
    from v8 import adaptive_learning_allocation_v819 as v819

    jobs = tuple(jobs)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if (
        coordinator is None
        or int(coordinator.config.lease_steps) != int(v819._DEFAULT_LEASE_STEPS)
        or len(jobs) <= 1
    ):
        return _BASE_RUN_ACTOR_JOBS(runtime, jobs, **kwargs)

    # Four is intentionally small: it cuts repeated expensive parent updates while
    # keeping live learning latency low even when only a subset of actors has useful
    # strategy/probe evidence in a publication interval.
    proxy = _BatchingRuntimeProxy(runtime, batch_size=min(4, len(jobs)))
    progress_callback = kwargs.get("progress_callback")
    if progress_callback is not None:
        def flushing_progress_callback(rows):
            proxy.flush()
            return progress_callback(rows)

        kwargs = dict(kwargs)
        kwargs["progress_callback"] = flushing_progress_callback
    try:
        return _BASE_RUN_ACTOR_JOBS(proxy, jobs, **kwargs)
    finally:
        proxy.flush()


def install_progressive_level_learning_v820() -> None:
    global _INSTALLED
    global _BASE_SERVICE_SUBMIT, _BASE_GENERATE_V820, _BASE_SUBMIT_NEXT_SOURCE
    global _BASE_RUN_ACTOR_JOBS
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_target_minimization_v820 as v820

    _BASE_SERVICE_SUBMIT = v819._service_submit_v819
    _BASE_GENERATE_V820 = v820._generate_v820
    _BASE_SUBMIT_NEXT_SOURCE = v820._submit_next_source
    _BASE_RUN_ACTOR_JOBS = actor_module.run_actor_jobs

    v819._service_submit_v819 = _submit_progressive_partial
    v820._generate_v820 = _generate_progressive_v820
    v820._submit_next_source = _submit_next_source_progressive
    actor_module.run_actor_jobs = _run_actor_jobs_batched
    _INSTALLED = True
