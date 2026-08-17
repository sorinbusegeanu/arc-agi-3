from __future__ import annotations

import os
import time


_INSTALLED = False
_UNSOLVED_LEASE_STEPS = 2048
_LIFECYCLE_MIN_INTERVAL_SECONDS = 60.0
_BASE_PLAN_CHAIN = None
_BASE_PREWIN_SUBMIT = None
_FOREIGN_TRANSFER_STRATEGIES: set[tuple[int, int, int]] = set()


def unsolved_lease_steps_v824(
    *,
    available: int,
    base_steps: int,
    initial_probe: bool,
    worker_count: int,
    game_count: int,
) -> int:
    """Bound every unsolved lease so allocation can react during a run."""
    del initial_probe, worker_count, game_count
    return min(
        max(1, int(available)),
        max(1, int(base_steps)),
        _UNSOLVED_LEASE_STEPS,
    )


def _prewin_submit_v824(service, trajectory) -> bool:
    """Validate one cheap pre-WIN source per game/level regardless of set size."""
    from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix
    from v8 import progressive_level_learning_v820 as progressive

    if not progressive._is_runtime_unsolved_partial(service, trajectory):
        return _BASE_PREWIN_SUBMIT(service, trajectory)

    game = str(trajectory.anchor.source_id)
    level = max(1, int(trajectory.target.levels_completed))
    solve_fix._defer_pre_win_source(service, trajectory)
    with service._v819_lock:
        stored = getattr(service, "_v819_pre_win_sources", {}).get(game, {}).get(level)
        if stored is None or str(stored.trajectory_id) != str(trajectory.trajectory_id):
            return True
        admitted = getattr(service, "_v824_pre_win_admitted_once", None)
        if admitted is None:
            admitted = set()
            service._v824_pre_win_admitted_once = admitted
        key = (game, level)
        if key in admitted:
            return True
        admitted.add(key)

    routed = bool(_BASE_PREWIN_SUBMIT(service, trajectory))
    if not routed:
        with service._v819_lock:
            getattr(service, "_v824_pre_win_admitted_once", set()).discard((game, level))
    return routed


def _game_hash(game_id: str) -> int:
    from v8.model import stable_u64

    return int(stable_u64(str(game_id), person=b"v8-game"))


def _foreign_key(game_id: str, strategy_uid) -> tuple[int, int, int]:
    return (
        _game_hash(game_id),
        int(strategy_uid.hi),
        int(strategy_uid.lo),
    )


def _plan_candidates_v824(self, context_signature, action_ids, **kwargs):
    """Probe unsolved decisions first and make TRANSFER provenance-explicit."""
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import runtime_repair_v822 as v822
    from v8 import trajectory_optimizer_v814 as optimizer

    if sampling._decision_mode_enabled() and bool(
        getattr(v822._PROBE_STATE, "before_plan", False)
    ):
        return ()

    rows = _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)
    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    if mode != v819.SamplingMode.TRANSFER.value:
        return rows

    game_id = str(getattr(optimizer, "_CAPTURE_SOURCE_ID", ""))
    if not game_id:
        return ()
    current_game_hash = _game_hash(game_id)
    selected = []
    for row in rows:
        source_games = self.source_games(row.strategy_uid)
        if not source_games or not any(int(value) != current_game_hash for value in source_games):
            continue
        _FOREIGN_TRANSFER_STRATEGIES.add(_foreign_key(game_id, row.strategy_uid))
        selected.append(row)
    return tuple(selected)


def _success_to_dict_v824(self):
    from v8 import adaptive_learning_allocation_v819 as v819

    raw = v819._BASE_SUCCESS_TO_DICT(self)
    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    foreign = bool(
        mode == v819.SamplingMode.TRANSFER.value
        and not self.parent_strategy_uid.is_zero
        and _foreign_key(self.anchor.source_id, self.parent_strategy_uid)
        in _FOREIGN_TRANSFER_STRATEGIES
    )
    raw["frontier_source"] = (
        v819.FrontierSource.TRANSFER.value
        if foreign
        else v819.FrontierSource.SAMPLER.value
    )
    raw["sampling_mode"] = mode
    return raw


def _lifecycle_worker_v824(supervisor) -> None:
    """Complete active windows quickly but start new windows at most once/minute."""
    from v8 import dedicated_lifecycle_v813 as lifecycle

    try:
        completed_window = int(
            getattr(supervisor.lifecycle, "_v812_last_completed_window", -1)
        )
        last_completed_at = time.monotonic() if completed_window >= 0 else -1.0
        while not supervisor._stop.is_set():
            if supervisor._pause.is_set():
                supervisor._stop.wait(0.05)
                continue

            current_completed = int(
                getattr(supervisor.lifecycle, "_v812_last_completed_window", -1)
            )
            if current_completed != completed_window:
                completed_window = current_completed
                last_completed_at = time.monotonic()

            active_window = int(
                getattr(supervisor.lifecycle, "_v812_active_window", -1)
            )
            interval = max(
                0.0,
                float(
                    getattr(
                        supervisor,
                        "_v824_lifecycle_min_interval_seconds",
                        _LIFECYCLE_MIN_INTERVAL_SECONDS,
                    )
                ),
            )
            if active_window < 0 and last_completed_at >= 0.0 and interval > 0.0:
                remaining = interval - (time.monotonic() - last_completed_at)
                if remaining > 0.0:
                    supervisor._stop.wait(min(0.50, remaining))
                    continue

            acquired = supervisor._v813_lifecycle_run_lock.acquire(timeout=0.05)
            if not acquired:
                continue
            try:
                if not supervisor._pause.is_set() and not supervisor._stop.is_set():
                    before = int(
                        getattr(supervisor.lifecycle, "_v812_last_completed_window", -1)
                    )
                    lifecycle._run_lifecycle_iteration(supervisor)
                    after = int(
                        getattr(supervisor.lifecycle, "_v812_last_completed_window", -1)
                    )
                    if after != before:
                        completed_window = after
                        last_completed_at = time.monotonic()
            finally:
                supervisor._v813_lifecycle_run_lock.release()
            supervisor._stop.wait(0.10)
    except BaseException as exc:
        supervisor._v813_lifecycle_error = f"{type(exc).__name__}: {exc}"
        supervisor._stop.set()


def install_learning_performance_repair_v824() -> None:
    global _INSTALLED, _BASE_PLAN_CHAIN, _BASE_PREWIN_SUBMIT
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import runtime_repair_v822 as v822
    from v8 import sampling_control_repair_v823 as v823
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.publication import LiveReadView

    # 1) Every unsolved lease is short enough for real mid-run reallocation.
    perf.__dict__["_v823_initial_unsolved_lease_steps"] = unsolved_lease_steps_v824

    # 2) Keep one cheap DIRECT_ACTION/TRUNCATE validation per unsolved game/level,
    # but remove v8.23's game-count gate that disabled it above eight unsolved games.
    _BASE_PREWIN_SUBMIT = v823._BASE_PROGRESSIVE_SUBMIT
    v819._service_submit_v819 = _prewin_submit_v824

    # 3) Restore bounded decision-point probing before planner reuse. TRANSFER
    # additionally accepts only strategies with provenance from another game.
    _BASE_PLAN_CHAIN = v822._BASE_PLAN_CANDIDATES
    v822._BASE_PLAN_CANDIDATES = _plan_candidates_v824
    LiveReadView.plan_candidates = v822._BASE_PLAN_CANDIDATES

    # Label a trajectory TRANSFER only when its successful parent strategy was
    # actually admitted by the foreign-game provenance filter above.
    optimizer.SuccessfulTrajectory.to_dict = _success_to_dict_v824
    v819._success_to_dict_v819 = _success_to_dict_v824

    # 4) Preserve 64-generation lifecycle semantics, but throttle new lifecycle
    # windows in wall-clock time. The existing v8.22 five-minute startup delay stays.
    v822._BASE_LIFECYCLE_WORKER = _lifecycle_worker_v824

    _INSTALLED = True
