from __future__ import annotations

import os
from collections import defaultdict


_INSTALLED = False
_BASE_SERVICE_SUBMIT_V819 = None
_BASE_PUBLISH_VALIDATED_SOURCE_V819 = None


def _cached_game_state(coordinator, game_id: str):
    """Read allocation/lifecycle state without rebuilding a multi-million-node index."""
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import lifecycle_competence_integration_v827 as lifecycle

    game = str(game_id)
    if not all(
        hasattr(coordinator, name)
        for name in ("_lock", "_game_won", "_records")
    ):
        return coordinator.game_state(game)
    with coordinator._lock:
        if not bool(coordinator._game_won.get(game, False)):
            return v819.GameLearningState.UNSOLVED
        rows = [
            row
            for (owner, _level), row in coordinator._records.items()
            if owner == game
        ]
        if not rows:
            return v819.GameLearningState.UNSOLVED
        state = (
            v819.GameLearningState.SOLVED_OPTIMIZING
            if any(
                row.state == v819.GameLearningState.SOLVED_OPTIMIZING
                for row in rows
            )
            else v819.GameLearningState.SOLVED_STABLE
        )

    lifecycle_class = lifecycle._cached_frontier_lifecycle_class(coordinator, game)
    if lifecycle_class in {"UNKNOWN", "ACTIVE"}:
        return state
    if lifecycle_class == "QUARANTINED":
        return v819.GameLearningState.SOLVED_OPTIMIZING
    return v819.GameLearningState.UNSOLVED


def _cached_sampling_weight(coordinator, game_id: str, state) -> float:
    """Report the installed allocation weight without another lifecycle refresh."""
    from v8 import adaptive_learning_allocation_v819 as v819

    base = {
        v819.GameLearningState.UNSOLVED: float(coordinator.config.unsolved_weight),
        v819.GameLearningState.SOLVED_OPTIMIZING: float(
            coordinator.config.optimizing_weight
        ),
        v819.GameLearningState.SOLVED_STABLE: float(coordinator.config.stable_weight),
    }[state]
    enabled = str(os.environ.get("ARC_AGI3_V8_PLATEAU_PRIORITY_ENABLED", "0")).lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return max(1e-9, base)
    with coordinator._lock:
        signals = coordinator._signals.setdefault(
            str(game_id), v819.GamePrioritySignals()
        )
        return max(1e-9, base * signals.multiplier)


def _game_is_validated_solved(service, game_id: str) -> bool:
    runtime = getattr(service, "_v819_runtime", None)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None:
        return False
    try:
        from v8 import adaptive_learning_allocation_v819 as v819

        return _cached_game_state(coordinator, str(game_id)) != v819.GameLearningState.UNSOLVED
    except BaseException:
        return False


def _defer_pre_win_source(service, trajectory) -> bool:
    """Keep only the shortest observed prefix-chain source for each reached level."""

    game = str(trajectory.anchor.source_id)
    level = max(1, int(trajectory.target.levels_completed))
    full_cost = len(tuple(trajectory.anchor.prefix_actions)) + len(tuple(trajectory.actions))
    with service._v819_lock:
        pending = getattr(service, "_v819_pre_win_sources", None)
        if pending is None:
            pending = defaultdict(dict)
            service._v819_pre_win_sources = pending
        by_level = pending[game]
        prior = by_level.get(level)
        if prior is None:
            by_level[level] = trajectory
        else:
            prior_cost = len(tuple(prior.anchor.prefix_actions)) + len(tuple(prior.actions))
            if (full_cost, str(trajectory.trajectory_id)) < (
                prior_cost,
                str(prior.trajectory_id),
            ):
                by_level[level] = trajectory
    # SuccessfulTrajectory.from_dict records this only so v8.19 source validation
    # can distinguish SAMPLER from TRANSFER. Pre-WIN level sources are not source-
    # validated, so do not retain an unbounded process-global provenance entry.
    try:
        from v8 import adaptive_learning_allocation_v819 as v819

        v819._SOURCE_KIND_BY_TRAJECTORY.pop(str(trajectory.trajectory_id), None)
    except BaseException:
        pass
    # True means the optimizer inbox may delete the file: the best source for this
    # level is now retained in memory until a complete WIN is independently validated.
    return True


def _release_pre_win_sources(service, game_id: str) -> int:
    from v8 import adaptive_learning_allocation_v819 as v819

    game = str(game_id)
    with service._v819_lock:
        pending = getattr(service, "_v819_pre_win_sources", None)
        if not pending:
            return 0
        by_level = dict(pending.pop(game, {}))
    released = 0
    for _level, trajectory in sorted(by_level.items()):
        if v819._BASE_SERVICE_SUBMIT(service, trajectory):
            released += 1
    return released


def _service_submit_solve_first(service, trajectory) -> bool:
    """Do no replay/optimization work for partial levels of unsolved games."""

    if int(getattr(trajectory, "round_index", 0)) != 0:
        return _BASE_SERVICE_SUBMIT_V819(service, trajectory)

    terminal = str(getattr(trajectory.target, "terminal_state", ""))
    game = str(trajectory.anchor.source_id)
    if terminal == "WIN":
        # One complete WIN is the admission event for validation. Until it validates,
        # partial level trajectories remain cheap in-memory observations only.
        return _BASE_SERVICE_SUBMIT_V819(service, trajectory)

    if _game_is_validated_solved(service, game):
        # Once the game is solved, later level paths are optimization material and
        # can enter the existing v8.18 optimizer directly without another baseline
        # source-validation replay.
        try:
            from v8 import adaptive_learning_allocation_v819 as v819

            v819._SOURCE_KIND_BY_TRAJECTORY.pop(str(trajectory.trajectory_id), None)
        except BaseException:
            pass
        return v819._BASE_SERVICE_SUBMIT(service, trajectory)

    return _defer_pre_win_source(service, trajectory)


def _publish_validated_source_solve_first(runtime, candidate, result, target_uid) -> None:
    service = runtime._v814_trajectory_optimizer
    _BASE_PUBLISH_VALIDATED_SOURCE_V819(runtime, candidate, result, target_uid)
    if str(getattr(candidate.source.target, "terminal_state", "")) == "WIN":
        _release_pre_win_sources(service, str(candidate.source.anchor.source_id))


def install_adaptive_learning_allocation_v819_solve_fix() -> None:
    global _INSTALLED, _BASE_SERVICE_SUBMIT_V819, _BASE_PUBLISH_VALIDATED_SOURCE_V819
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819

    _BASE_SERVICE_SUBMIT_V819 = v819._service_submit_v819
    _BASE_PUBLISH_VALIDATED_SOURCE_V819 = v819._publish_validated_source
    v819.AdaptiveLearningCoordinator._v819_telemetry_game_state = _cached_game_state
    v819.AdaptiveLearningCoordinator._v819_telemetry_sampling_weight = (
        _cached_sampling_weight
    )
    v819._service_submit_v819 = _service_submit_solve_first
    v819._publish_validated_source = _publish_validated_source_solve_first
    _INSTALLED = True
