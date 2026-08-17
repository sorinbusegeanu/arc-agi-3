from __future__ import annotations


_INSTALLED = False
_BASE_RUN_ACTOR_JOBS = None
_BASE_CHOOSE_GAME = None
_BASE_RECOMMENDED_LEASE_STEPS = None
_BASE_RECORD_LEASE = None


def _ensure_tracking_state(coordinator) -> None:
    if not hasattr(coordinator, "_v819_inflight_tracking_active"):
        coordinator._v819_inflight_tracking_active = False
    if not hasattr(coordinator, "_v819_inflight_steps"):
        coordinator._v819_inflight_steps = {}
    if not hasattr(coordinator, "_v819_inflight_reservations"):
        coordinator._v819_inflight_reservations = {}


def _begin_inflight_tracking(coordinator) -> None:
    _ensure_tracking_state(coordinator)
    with coordinator._lock:
        coordinator._v819_inflight_tracking_active = True
        coordinator._v819_inflight_steps = {}
        coordinator._v819_inflight_reservations = {}


def _clear_inflight_tracking(coordinator) -> None:
    _ensure_tracking_state(coordinator)
    with coordinator._lock:
        coordinator._v819_inflight_tracking_active = False
        coordinator._v819_inflight_steps.clear()
        coordinator._v819_inflight_reservations.clear()


def _reserve_inflight(coordinator, game_id: str, steps: int) -> None:
    _ensure_tracking_state(coordinator)
    requested = max(0, int(steps))
    if requested <= 0:
        return
    with coordinator._lock:
        if not bool(coordinator._v819_inflight_tracking_active):
            return
        game = str(game_id)
        reservations = coordinator._v819_inflight_reservations.setdefault(game, [])
        reservations.append(requested)
        coordinator._v819_inflight_steps[game] = (
            int(coordinator._v819_inflight_steps.get(game, 0)) + requested
        )


def _release_inflight(coordinator, game_id: str, actual_steps: int) -> int:
    _ensure_tracking_state(coordinator)
    with coordinator._lock:
        if not bool(coordinator._v819_inflight_tracking_active):
            return 0
        game = str(game_id)
        reservations = coordinator._v819_inflight_reservations.get(game)
        if not reservations:
            return 0
        actual = max(0, int(actual_steps))
        exact = next(
            (index for index, value in enumerate(reservations) if int(value) == actual),
            None,
        )
        if exact is None:
            exact = min(
                range(len(reservations)),
                key=lambda index: (abs(int(reservations[index]) - actual), index),
            )
        reserved = int(reservations.pop(exact))
        if not reservations:
            coordinator._v819_inflight_reservations.pop(game, None)
        remaining = max(
            0,
            int(coordinator._v819_inflight_steps.get(game, 0)) - reserved,
        )
        if remaining:
            coordinator._v819_inflight_steps[game] = remaining
        else:
            coordinator._v819_inflight_steps.pop(game, None)
        return reserved


def _inflight_steps_for(coordinator, game_id: str) -> int:
    _ensure_tracking_state(coordinator)
    with coordinator._lock:
        return int(coordinator._v819_inflight_steps.get(str(game_id), 0))


def _choose_game_with_inflight(self, games) -> str:
    _ensure_tracking_state(self)
    if not bool(self._v819_inflight_tracking_active):
        return _BASE_CHOOSE_GAME(self, games)
    candidates = tuple(dict.fromkeys(str(game) for game in games))
    if not candidates:
        raise ValueError("adaptive allocator requires at least one game")
    self.register_games(candidates)
    with self._lock:
        return min(
            candidates,
            key=lambda game: (
                (
                    float(self._run[game].sample_steps)
                    + float(self._v819_inflight_steps.get(game, 0))
                )
                / max(1e-9, float(self.sampling_weight(game))),
                int(self._run[game].leases)
                + len(self._v819_inflight_reservations.get(game, ())),
                game,
            ),
        )


def _recommended_lease_steps_with_inflight(self, game_id: str, remaining: int) -> int:
    steps = int(_BASE_RECOMMENDED_LEASE_STEPS(self, game_id, remaining))
    _reserve_inflight(self, str(game_id), steps)
    return steps


def _record_lease_with_inflight(self, game_id, mode, steps) -> None:
    _release_inflight(self, str(game_id), int(steps))
    _BASE_RECORD_LEASE(self, game_id, mode, steps)


def _run_actor_jobs_with_inflight(runtime, jobs, **kwargs):
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None:
        return _BASE_RUN_ACTOR_JOBS(runtime, jobs, **kwargs)
    _begin_inflight_tracking(coordinator)
    try:
        return _BASE_RUN_ACTOR_JOBS(runtime, jobs, **kwargs)
    finally:
        _clear_inflight_tracking(coordinator)


def install_adaptive_learning_allocation_v819_inflight_fix() -> None:
    global _INSTALLED
    global _BASE_RUN_ACTOR_JOBS, _BASE_CHOOSE_GAME
    global _BASE_RECOMMENDED_LEASE_STEPS, _BASE_RECORD_LEASE
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import adaptive_learning_allocation_v819 as v819

    _BASE_RUN_ACTOR_JOBS = actor_module.run_actor_jobs
    _BASE_CHOOSE_GAME = v819.AdaptiveLearningCoordinator.choose_game
    _BASE_RECOMMENDED_LEASE_STEPS = (
        v819.AdaptiveLearningCoordinator.recommended_lease_steps
    )
    _BASE_RECORD_LEASE = v819.AdaptiveLearningCoordinator.record_lease

    v819.AdaptiveLearningCoordinator.choose_game = _choose_game_with_inflight
    v819.AdaptiveLearningCoordinator.recommended_lease_steps = (
        _recommended_lease_steps_with_inflight
    )
    v819.AdaptiveLearningCoordinator.record_lease = _record_lease_with_inflight
    actor_module.run_actor_jobs = _run_actor_jobs_with_inflight
    _INSTALLED = True
