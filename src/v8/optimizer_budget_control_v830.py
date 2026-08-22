from __future__ import annotations

import threading
from dataclasses import dataclass


_INSTALLED = False
_BASE_RESERVE_OPTIMIZATION = None
_BASE_RECORD_OPTIMIZER_VALIDATION = None
_BASE_ROUTE_CANDIDATE = None
_BASE_VALIDATE_TRACKED = None
_BASE_PROCESS_CANDIDATE = None
_BASE_START_WAITING_VALIDATORS = None

# Give every new frontier a bounded probe, then scale from the amount of work
# that can still be removed.  The old floor of 64 made very short solutions
# consume nearly the same no-progress budget as trajectories hundreds of
# actions long.
_MIN_SEARCH_BUDGET = 16
_HEADROOM_BUDGET_SCALE = 8
_SAVED_BUDGET_SCALE = 4
_PROGRESS_REPORT_VALIDATIONS = 32

_BUDGET_CONTEXT = threading.local()
_PROCESS_CONTEXT = threading.local()


@dataclass(slots=True)
class OptimizerBudgetStats:
    source_cost: int = 0
    best_cost: int = 0
    validations: int = 0
    successes: int = 0
    failed_attempts: int = 0
    validations_since_improvement: int = 0
    saved_actions: int = 0
    improvements: int = 0
    last_report_bucket: int = 0
    failure_reported: bool = False
    started: bool = False
    terminal_status: str = ""


def _ensure_state(coordinator) -> None:
    if not hasattr(coordinator, "_v830_optimizer_budget_stats"):
        coordinator._v830_optimizer_budget_stats = {}


def _stats_for(coordinator, game_id: str, level: int) -> OptimizerBudgetStats:
    _ensure_state(coordinator)
    key = (str(game_id), max(1, int(level)))
    row = coordinator._v830_optimizer_budget_stats.get(key)
    if row is None:
        record = coordinator._record(*key)
        row = OptimizerBudgetStats(
            validations_since_improvement=max(
                0, int(record.validations_since_improvement)
            )
        )
        coordinator._v830_optimizer_budget_stats[key] = row
    return row


def _candidate_scope(candidate) -> tuple[str, int, int]:
    source = candidate.source
    return (
        str(source.anchor.source_id),
        max(1, int(source.target.levels_completed)),
        max(1, int(source.cost)),
    )


def _budget_limits(coordinator, game_id: str, level: int) -> tuple[int, int, int]:
    with coordinator._lock:
        stats = _stats_for(coordinator, game_id, level)
        hard_cap = max(1, int(coordinator.config.optimization_validation_budget))
        configured_stall = max(
            1, int(coordinator.config.max_validations_without_improvement)
        )
        best_cost = max(0, int(stats.best_cost))
        if best_cost <= 0:
            return hard_cap, configured_stall, 0
        potential = max(0, best_cost - 1)
        if potential <= 0:
            return 0, 0, 0

        floor = min(
            hard_cap,
            max(1, min(_MIN_SEARCH_BUDGET, configured_stall)),
        )
        dynamic_cap = min(
            hard_cap,
            floor
            + potential * _HEADROOM_BUDGET_SCALE
            + max(0, int(stats.saved_actions)) * _SAVED_BUDGET_SCALE,
        )
        stall_limit = min(
            dynamic_cap,
            max(
                1,
                min(
                    configured_stall,
                    floor + potential,
                ),
            ),
        )
        return int(dynamic_cap), int(stall_limit), int(potential)


def _game_potential(coordinator, game_id: str) -> int:
    game = str(game_id)
    with coordinator._lock:
        _ensure_state(coordinator)
        return sum(
            max(0, int(stats.best_cost) - 1)
            for (owner, _level), stats in coordinator._v830_optimizer_budget_stats.items()
            if owner == game
        )


def _status_message(coordinator, game_id: str, level: int, status: str) -> str:
    with coordinator._lock:
        stats = _stats_for(coordinator, game_id, level)
        record = coordinator._record(game_id, level)
        budget, stall, potential = _budget_limits(coordinator, game_id, level)
        best = max(0, int(stats.best_cost))
        source = max(0, int(stats.source_cost))
        return (
            f"optimizer game={str(game_id)} level={max(1, int(level))} "
            f"status={str(status)} cost={best or source} potential={potential} "
            f"game_potential={_game_potential(coordinator, game_id)} "
            f"validations={int(stats.validations)} successes={int(stats.successes)} "
            f"saved={int(stats.saved_actions)} "
            f"budget={int(record.consumed_optimization_budget)}/{budget} "
            f"no_progress={int(record.validations_since_improvement)}/{stall}"
        )


def _emit_status(
    coordinator,
    game_id: str,
    level: int,
    status: str,
    *,
    terminal: bool = False,
) -> None:
    with coordinator._lock:
        stats = _stats_for(coordinator, game_id, level)
        if terminal and stats.terminal_status == str(status):
            return
        if terminal:
            stats.terminal_status = str(status)
        elif stats.terminal_status:
            stats.terminal_status = ""
    coordinator._emit(_status_message(coordinator, game_id, level, status))


def _register_candidate(coordinator, candidate) -> tuple[str, int]:
    game, level, source_cost = _candidate_scope(candidate)
    emit_start = False
    with coordinator._lock:
        stats = _stats_for(coordinator, game, level)
        if stats.source_cost <= 0:
            stats.source_cost = source_cost
        else:
            stats.source_cost = max(int(stats.source_cost), source_cost)
        if stats.best_cost <= 0:
            stats.best_cost = source_cost
        else:
            stats.best_cost = min(int(stats.best_cost), source_cost)
        if not stats.started:
            stats.started = True
            emit_start = True
    if emit_start:
        _emit_status(coordinator, game, level, "START")
    return game, level


def _context_set(target, **values):
    prior = {name: getattr(target, name, None) for name in values}
    present = {name: hasattr(target, name) for name in values}
    for name, value in values.items():
        setattr(target, name, value)
    return prior, present


def _context_restore(target, prior, present) -> None:
    for name, value in prior.items():
        if present[name]:
            setattr(target, name, value)
        else:
            try:
                delattr(target, name)
            except AttributeError:
                pass


def _reserve_optimization_v830(
    self,
    *,
    game_id: str,
    level: int,
    attempts: int,
) -> bool:
    mode = getattr(_BUDGET_CONTEXT, "mode", None)
    key = getattr(_BUDGET_CONTEXT, "key", None)
    expected = (str(game_id), max(1, int(level)))
    if mode not in {"precheck", "consume"} or key != expected:
        return _BASE_RESERVE_OPTIMIZATION(
            self,
            game_id=game_id,
            level=level,
            attempts=attempts,
        )

    requested = max(1, int(attempts))
    status = None
    terminal = False
    with self._lock:
        stats = _stats_for(self, game_id, level)
        record = self._record(game_id, level)
        budget, stall, potential = _budget_limits(self, game_id, level)

        if int(record.optimizer_exhausted_version) == int(record.frontier_version):
            status, terminal = "EXHAUSTED", True
            allowed = False
        elif potential <= 0:
            record.optimizer_exhausted_version = int(record.frontier_version)
            status, terminal = "MINIMAL", True
            allowed = False
        elif int(record.validations_since_improvement) >= int(stall):
            record.optimizer_exhausted_version = int(record.frontier_version)
            status, terminal = "STALLED", True
            allowed = False
        elif int(record.consumed_optimization_budget) + requested > int(budget):
            record.optimizer_exhausted_version = int(record.frontier_version)
            status, terminal = "BUDGET_EXHAUSTED", True
            allowed = False
        elif mode == "precheck":
            allowed = True
        else:
            record.consumed_optimization_budget += requested
            from v8 import adaptive_learning_allocation_v819 as v819

            run = self._run.setdefault(str(game_id), v819.GameRunTelemetry())
            run.optimizer_candidates += 1
            allowed = True

    if status is not None:
        _emit_status(self, game_id, level, status, terminal=terminal)
    return bool(allowed)


def _record_optimizer_validation_v830(
    self,
    *,
    game_id: str,
    level: int,
    attempts: int,
    successes: int,
    saved_actions: int,
    improved: bool,
    generation: int,
) -> None:
    _BASE_RECORD_OPTIMIZER_VALIDATION(
        self,
        game_id=game_id,
        level=level,
        attempts=attempts,
        successes=successes,
        saved_actions=saved_actions,
        improved=improved,
        generation=generation,
    )

    meaningful = bool(
        improved
        and int(saved_actions) >= int(self.config.min_meaningful_improvement)
    )
    if not meaningful:
        return

    source_cost = getattr(_PROCESS_CONTEXT, "source_cost", None)
    with self._lock:
        stats = _stats_for(self, game_id, level)
        saved = max(0, int(saved_actions))
        stats.saved_actions += saved
        stats.improvements += 1
        stats.validations_since_improvement = 0
        stats.failure_reported = False
        stats.terminal_status = ""
        if source_cost is not None:
            candidate_cost = max(1, int(source_cost) - saved)
            if stats.best_cost <= 0:
                stats.best_cost = candidate_cost
            else:
                stats.best_cost = min(int(stats.best_cost), candidate_cost)

    _emit_status(self, game_id, level, "IMPROVED")
    if _budget_limits(self, game_id, level)[2] <= 0:
        _emit_status(self, game_id, level, "MINIMAL", terminal=True)


def _route_candidate_v830(service, candidate) -> bool:
    runtime = getattr(service, "_v819_runtime", None)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None or str(getattr(candidate, "edit_kind", "")) == "VALIDATE_SOURCE":
        return _BASE_ROUTE_CANDIDATE(service, candidate)

    game, level = _register_candidate(coordinator, candidate)
    prior, present = _context_set(
        _BUDGET_CONTEXT,
        mode="precheck",
        key=(game, level),
    )
    try:
        routed = bool(_BASE_ROUTE_CANDIDATE(service, candidate))
    finally:
        _context_restore(_BUDGET_CONTEXT, prior, present)

    if not routed:
        try:
            service._log(
                "optimizer_budget_route_rejected",
                game=game,
                level=level,
                edit=str(getattr(candidate, "edit_kind", "")),
            )
        except BaseException:
            pass
    return routed


def _adjust_reserved_attempts(coordinator, game: str, level: int, delta: int) -> None:
    if int(delta) == 0:
        return
    with coordinator._lock:
        row = coordinator._record(game, level)
        row.consumed_optimization_budget = max(
            0,
            int(row.consumed_optimization_budget) + int(delta),
        )


def _validate_tracked_v830(service, validator, candidate, *, skip_attempted: bool = True):
    if str(getattr(candidate, "edit_kind", "")) == "VALIDATE_SOURCE":
        return _BASE_VALIDATE_TRACKED(
            service,
            validator,
            candidate,
            skip_attempted=skip_attempted,
        )

    runtime = getattr(service, "_v819_runtime", None)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None:
        return _BASE_VALIDATE_TRACKED(
            service,
            validator,
            candidate,
            skip_attempted=skip_attempted,
        )

    if skip_attempted:
        with service._lock:
            if candidate.candidate_id in service._attempted:
                return None

    game, level = _register_candidate(coordinator, candidate)
    try:
        from v8 import trajectory_optimizer_v818 as v818

        expected_attempts = max(1, len(tuple(v818._VALIDATION_SEEDS)))
    except BaseException:
        expected_attempts = 1

    prior, present = _context_set(
        _BUDGET_CONTEXT,
        mode="consume",
        key=(game, level),
    )
    try:
        if not coordinator.reserve_optimization(
            game_id=game,
            level=level,
            attempts=expected_attempts,
        ):
            try:
                service._log(
                    "optimizer_budget_skip",
                    game=game,
                    level=level,
                    edit=str(getattr(candidate, "edit_kind", "")),
                    candidate_id=str(getattr(candidate, "candidate_id", "")),
                )
            except BaseException:
                pass
            return None
    finally:
        _context_restore(_BUDGET_CONTEXT, prior, present)

    result = _BASE_VALIDATE_TRACKED(
        service,
        validator,
        candidate,
        skip_attempted=skip_attempted,
    )
    if result is None:
        _adjust_reserved_attempts(coordinator, game, level, -expected_attempts)
        return None

    actual_attempts = max(1, int(getattr(result, "attempts", expected_attempts)))
    _adjust_reserved_attempts(
        coordinator,
        game,
        level,
        actual_attempts - expected_attempts,
    )
    successes = max(
        0,
        int(
            getattr(
                result,
                "successes",
                int(bool(getattr(result, "success", False))),
            )
        ),
    )

    emit_status = None
    with coordinator._lock:
        stats = _stats_for(coordinator, game, level)
        stats.validations += actual_attempts
        stats.successes += successes
        stats.failed_attempts += max(0, actual_attempts - successes)
        stats.validations_since_improvement += actual_attempts
        bucket = stats.validations // _PROGRESS_REPORT_VALIDATIONS
        if successes <= 0 and not stats.failure_reported:
            stats.failure_reported = True
            emit_status = "NO_PROGRESS"
        elif bucket > int(stats.last_report_bucket):
            stats.last_report_bucket = bucket
            emit_status = "SEARCHING"

    if emit_status is not None:
        _emit_status(coordinator, game, level, emit_status)

    return result


def _process_candidate_v830(service, validator, candidate):
    source_cost = max(1, int(candidate.source.cost))
    game = str(candidate.source.anchor.source_id)
    level = max(1, int(candidate.source.target.levels_completed))
    prior, present = _context_set(
        _PROCESS_CONTEXT,
        source_cost=source_cost,
        key=(game, level),
    )
    try:
        return _BASE_PROCESS_CANDIDATE(service, validator, candidate)
    finally:
        _context_restore(_PROCESS_CONTEXT, prior, present)


def _start_waiting_validators_v830(service) -> None:
    runtime = getattr(service, "_v819_runtime", None)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None:
        return _BASE_START_WAITING_VALIDATORS(service)

    from v8 import trajectory_optimizer_v818 as v818

    with service._v818_validator_lock:
        waiting = tuple(service._v818_waiting_games)
    ordered = sorted(
        waiting,
        key=lambda game: (-_game_potential(coordinator, str(game)), str(game)),
    )
    for game in ordered:
        v818._ensure_validator(service, str(game))


def optimizer_budget_snapshot(coordinator, game_id: str) -> tuple[dict[str, int], ...]:
    game = str(game_id)
    rows = []
    with coordinator._lock:
        _ensure_state(coordinator)
        for (owner, level), stats in sorted(
            coordinator._v830_optimizer_budget_stats.items()
        ):
            if owner != game:
                continue
            budget, stall, potential = _budget_limits(coordinator, owner, level)
            record = coordinator._record(owner, level)
            rows.append(
                {
                    "level": int(level),
                    "source_cost": int(stats.source_cost),
                    "best_cost": int(stats.best_cost),
                    "potential": int(potential),
                    "validations": int(stats.validations),
                    "successes": int(stats.successes),
                    "saved_actions": int(stats.saved_actions),
                    "budget_used": int(record.consumed_optimization_budget),
                    "budget_limit": int(budget),
                    "no_progress": int(record.validations_since_improvement),
                    "no_progress_limit": int(stall),
                }
            )
    return tuple(rows)


def install_optimizer_budget_control_v830() -> None:
    global _INSTALLED
    global _BASE_RESERVE_OPTIMIZATION, _BASE_RECORD_OPTIMIZER_VALIDATION
    global _BASE_ROUTE_CANDIDATE, _BASE_VALIDATE_TRACKED
    global _BASE_PROCESS_CANDIDATE, _BASE_START_WAITING_VALIDATORS
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v818 as v818
    from v8 import trajectory_target_minimization_v820 as v820

    _BASE_RESERVE_OPTIMIZATION = v819.AdaptiveLearningCoordinator.reserve_optimization
    _BASE_RECORD_OPTIMIZER_VALIDATION = (
        v819.AdaptiveLearningCoordinator.record_optimizer_validation
    )
    _BASE_ROUTE_CANDIDATE = v818._route_candidate
    _BASE_VALIDATE_TRACKED = v820._validate_tracked
    _BASE_PROCESS_CANDIDATE = v820._process_candidate
    _BASE_START_WAITING_VALIDATORS = v818._start_waiting_validators

    v819.AdaptiveLearningCoordinator.reserve_optimization = _reserve_optimization_v830
    v819.AdaptiveLearningCoordinator.record_optimizer_validation = (
        _record_optimizer_validation_v830
    )
    v818._route_candidate = _route_candidate_v830
    v820._validate_tracked = _validate_tracked_v830
    v820._process_candidate = _process_candidate_v830
    v818._start_waiting_validators = _start_waiting_validators_v830
    _INSTALLED = True
