from __future__ import annotations

import json
import os
import threading
import time


_INSTALLED = False
_RUN_SESSION_ENV = "ARC_AGI3_V8_RUN_SESSION"
_FULL_WIN_SCOPE_LEVEL = 1_000_000_000

_BASE_RUNTIME_INIT = None
_BASE_CANDIDATE_SCOPE = None
_BASE_STATUS_MESSAGE = None
_BASE_RESERVE_OPTIMIZATION = None
_BASE_RECORD_OPTIMIZER_VALIDATION = None
_BASE_PROCESS_CANDIDATE = None
_BASE_SUBMIT_NEXT_SOURCE = None

_FULL_WIN_CONTEXT = threading.local()


def _new_run_session() -> str:
    return f"{os.getpid()}-{time.time_ns()}"


def _ensure_run_session() -> str:
    session = str(os.environ.get(_RUN_SESSION_ENV, "")).strip()
    if session:
        return session
    session = _new_run_session()
    os.environ[_RUN_SESSION_ENV] = session
    return session


def _runtime_init_base_v835(self, *args, **kwargs):
    prior = os.environ.get(_RUN_SESSION_ENV)
    session = _new_run_session()
    os.environ[_RUN_SESSION_ENV] = session
    try:
        result = _BASE_RUNTIME_INIT(self, *args, **kwargs)
    except BaseException:
        if prior is None:
            os.environ.pop(_RUN_SESSION_ENV, None)
        else:
            os.environ[_RUN_SESSION_ENV] = prior
        raise
    self._v835_run_session = session
    return result


def _write_runtime_win_marker_v835(game_id: str, result) -> None:
    if int(getattr(result, "wins", 0)) <= 0:
        return

    from v8 import runtime_win_optimization_v834 as v834
    from v8 import trajectory_optimizer_v814 as optimizer

    path = v834._marker_path(game_id)
    if path is None:
        return
    optimizer._atomic_json(
        path,
        {
            "game_id": str(game_id),
            "run_session": _ensure_run_session(),
            "observed_levels": max(0, int(getattr(result, "levels_completed", 0))),
            "steps": max(1, int(getattr(result, "steps", 0))),
            "time_ns": time.time_ns(),
        },
    )


def _promote_runtime_win_if_present_v835(coordinator, game_id: str) -> bool:
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import runtime_win_optimization_v834 as v834

    session = str(os.environ.get(_RUN_SESSION_ENV, "")).strip()
    if not session:
        return False
    path = v834._marker_path(game_id)
    if path is None or not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False

    game = str(game_id)
    if str(raw.get("game_id", "")) != game:
        return False
    if str(raw.get("run_session", "")) != session:
        return False

    runtime = getattr(coordinator, "_v834_runtime", None)
    generation = max(1, int(getattr(runtime, "generation", 1)))
    with coordinator._lock:
        coordinator.register_games((game,))
        row = coordinator._record(game, _FULL_WIN_SCOPE_LEVEL)
        previous = row.state
        coordinator._game_won[game] = True
        if row.first_success_generation <= 0:
            row.first_success_generation = generation
        row.last_success_generation = generation
        row.last_frontier_improvement_generation = max(
            int(row.last_frontier_improvement_generation), generation
        )
        row.optimizer_exhausted_version = -1
        if row.state == v819.GameLearningState.UNSOLVED:
            row.state = v819.GameLearningState.SOLVED_OPTIMIZING
        if previous != row.state:
            coordinator._emit(
                f"learning state game={game} target=FULL_WIN "
                f"{previous.value}->{row.state.value} observed_runtime_win=1"
            )
    return True


def _is_full_win_source(source) -> bool:
    try:
        return bool(
            str(source.target.terminal_state) == "WIN"
            and not tuple(source.anchor.prefix_actions)
        )
    except (AttributeError, TypeError):
        return False


def _candidate_scope_v835(candidate) -> tuple[str, int, int]:
    if not _is_full_win_source(candidate.source):
        return _BASE_CANDIDATE_SCOPE(candidate)
    return (
        str(candidate.source.anchor.source_id),
        _FULL_WIN_SCOPE_LEVEL,
        max(1, int(candidate.source.cost)),
    )


def _status_message_v835(coordinator, game_id: str, level: int, status: str) -> str:
    if int(level) != _FULL_WIN_SCOPE_LEVEL:
        return _BASE_STATUS_MESSAGE(coordinator, game_id, level, status)

    from v8 import optimizer_budget_control_v830 as v830

    with coordinator._lock:
        stats = v830._stats_for(coordinator, game_id, level)
        record = coordinator._record(game_id, level)
        budget, stall, potential = v830._budget_limits(coordinator, game_id, level)
        best = max(0, int(stats.best_cost))
        source = max(0, int(stats.source_cost))
        return (
            f"optimizer game={str(game_id)} target=FULL_WIN "
            f"status={str(status)} cost={best or source} potential={potential} "
            f"game_potential={v830._game_potential(coordinator, game_id)} "
            f"validations={int(stats.validations)} successes={int(stats.successes)} "
            f"saved={int(stats.saved_actions)} "
            f"budget={int(record.consumed_optimization_budget)}/{budget} "
            f"no_progress={int(stats.validations_since_improvement)}/{stall}"
        )


def _scope_override(game_id: str, level: int) -> int:
    from v8 import optimizer_budget_control_v830 as v830

    game = str(game_id)
    budget_key = getattr(v830._BUDGET_CONTEXT, "key", None)
    if budget_key == (game, _FULL_WIN_SCOPE_LEVEL):
        return _FULL_WIN_SCOPE_LEVEL
    process_key = getattr(_FULL_WIN_CONTEXT, "key", None)
    if process_key == (game, _FULL_WIN_SCOPE_LEVEL):
        return _FULL_WIN_SCOPE_LEVEL
    return max(1, int(level))


def _reserve_optimization_v835(
    self,
    *,
    game_id: str,
    level: int,
    attempts: int,
) -> bool:
    return _BASE_RESERVE_OPTIMIZATION(
        self,
        game_id=str(game_id),
        level=_scope_override(game_id, level),
        attempts=attempts,
    )


def _record_optimizer_validation_v835(
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
    return _BASE_RECORD_OPTIMIZER_VALIDATION(
        self,
        game_id=str(game_id),
        level=_scope_override(game_id, level),
        attempts=attempts,
        successes=successes,
        saved_actions=saved_actions,
        improved=improved,
        generation=generation,
    )


def _process_candidate_base_v835(service, validator, candidate):
    if not _is_full_win_source(candidate.source):
        return _BASE_PROCESS_CANDIDATE(service, validator, candidate)

    key = (str(candidate.source.anchor.source_id), _FULL_WIN_SCOPE_LEVEL)
    present = hasattr(_FULL_WIN_CONTEXT, "key")
    prior = getattr(_FULL_WIN_CONTEXT, "key", None)
    _FULL_WIN_CONTEXT.key = key
    try:
        return _BASE_PROCESS_CANDIDATE(service, validator, candidate)
    finally:
        if present:
            _FULL_WIN_CONTEXT.key = prior
        else:
            try:
                delattr(_FULL_WIN_CONTEXT, "key")
            except AttributeError:
                pass


def _submit_next_source_v835(service, candidate, validated) -> None:
    if not _is_full_win_source(candidate.source):
        return _BASE_SUBMIT_NEXT_SOURCE(service, candidate, validated)
    if validated is None or int(validated.cost) <= 1:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v814 as optimizer

    next_source = optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(validated.anchor, validated.target, validated.actions),
        validated.anchor,
        validated.target,
        validated.actions,
        validated.strategy_uid,
        validated.target_outcome_uid,
        int(candidate.source.round_index) + 1,
    )
    v819._BASE_SERVICE_SUBMIT(service, next_source)


def install_runtime_win_scope_v835() -> None:
    global _INSTALLED
    global _BASE_RUNTIME_INIT, _BASE_CANDIDATE_SCOPE, _BASE_STATUS_MESSAGE
    global _BASE_RESERVE_OPTIMIZATION, _BASE_RECORD_OPTIMIZER_VALIDATION
    global _BASE_PROCESS_CANDIDATE, _BASE_SUBMIT_NEXT_SOURCE
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import optimizer_budget_control_v830 as v830
    from v8 import runtime_win_optimization_v834 as v834
    from v8 import trajectory_target_minimization_v820 as v820

    _BASE_RUNTIME_INIT = v834._BASE_RUNTIME_INIT
    _BASE_CANDIDATE_SCOPE = v830._candidate_scope
    _BASE_STATUS_MESSAGE = v830._status_message
    _BASE_RESERVE_OPTIMIZATION = v819.AdaptiveLearningCoordinator.reserve_optimization
    _BASE_RECORD_OPTIMIZER_VALIDATION = (
        v819.AdaptiveLearningCoordinator.record_optimizer_validation
    )
    _BASE_PROCESS_CANDIDATE = v830._BASE_PROCESS_CANDIDATE
    _BASE_SUBMIT_NEXT_SOURCE = v820._submit_next_source

    # Keep v8.34's public hooks installed and repair their internal semantics.
    v834._BASE_RUNTIME_INIT = _runtime_init_base_v835
    v834._write_runtime_win_marker = _write_runtime_win_marker_v835
    v834._promote_runtime_win_if_present = _promote_runtime_win_if_present_v835

    # Full-game WIN optimization owns a distinct budget/statistics scope while
    # v8.30 remains the public candidate-processing authority.
    v830._candidate_scope = _candidate_scope_v835
    v830._status_message = _status_message_v835
    v819.AdaptiveLearningCoordinator.reserve_optimization = _reserve_optimization_v835
    v819.AdaptiveLearningCoordinator.record_optimizer_validation = (
        _record_optimizer_validation_v835
    )
    v830._BASE_PROCESS_CANDIDATE = _process_candidate_base_v835
    v820._submit_next_source = _submit_next_source_v835
    _INSTALLED = True
