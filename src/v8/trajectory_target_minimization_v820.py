from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, replace


_INSTALLED = False
_BASE_GAME_VALIDATE = None
_BASE_GENERATE_V818 = None
_TARGET_MINIMIZE = "TARGET_MINIMIZE"
_DIRECT_ACTION = "DIRECT_ACTION"
_TRUNCATE_SUCCESS_PREFIX = "TRUNCATE_SUCCESS_PREFIX"
_DELTA_DELETE = "DELTA_DELETE"
_CANCELLED_SOURCE_MARKER = "__v841_shutdown_cancelled__"


class _ValidationCancelled(RuntimeError):
    """Internal control flow for a replay interrupted by final shutdown drain."""


def _validation_cancel_requested(service) -> bool:
    cancel = getattr(service, "_v841_validation_cancel", None)
    if cancel is None:
        return False
    is_set = getattr(cancel, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancel)


def _raise_if_validation_cancelled(service) -> None:
    if _validation_cancel_requested(service):
        raise _ValidationCancelled("trajectory validation cancelled for final drain")


def _preserve_cancelled_source(service, candidate) -> None:
    """Keep an interrupted optimizer source restartable in the next snapshot."""

    source = candidate.source
    source_id = str(source.trajectory_id)
    with service._lock:
        service._attempted.discard(str(candidate.candidate_id))
        pending = service._v818_pending_sources.get(source_id)
        if pending is None:
            pending = {
                "source": source,
                "candidate_ids": set(),
                "routing_open": False,
            }
            service._v818_pending_sources[source_id] = pending
        pending["candidate_ids"].add(f"{_CANCELLED_SOURCE_MARKER}:{source_id}")

    # Round-zero source-validation trajectories do not pass through the normal
    # v8.18 pending-source map. Keep an atomic inbox copy as well, so cancellation
    # never turns a successful sampler observation into volatile-only work.
    if int(getattr(source, "round_index", 0)) == 0:
        from v8 import trajectory_optimizer_v814 as optimizer

        service.inbox.mkdir(parents=True, exist_ok=True)
        target = service.inbox / f"shutdown-{source_id}.json"
        optimizer._atomic_json(target, source.to_dict())


@dataclass(frozen=True, slots=True)
class V820ValidationResult:
    success: bool
    actions_executed: int
    reason: str
    terminal_state: str
    levels_completed: int
    attempts: int
    successes: int
    prefix_actions: tuple[int, ...] = ()
    terminal_context: int = 0
    terminal_action: int = 0
    outcome_signature: int = 0
    successful_prefix_lengths: tuple[int, ...] = ()


def _is_arc_validator(service) -> bool:
    from v8.trajectory_validation_v814 import validate_arc_candidate

    return getattr(service, "validator", None) is validate_arc_candidate


def _validation_result_from_legacy(raw) -> V820ValidationResult:
    success = bool(getattr(raw, "success", False))
    lengths = tuple(int(value) for value in getattr(raw, "successful_prefix_lengths", ()) or ())
    return V820ValidationResult(
        success,
        int(getattr(raw, "actions_executed", 0)),
        str(getattr(raw, "reason", "target_preserved" if success else "target_not_reached")),
        str(getattr(raw, "terminal_state", "")),
        int(getattr(raw, "levels_completed", 0)),
        max(1, int(getattr(raw, "attempts", 1))),
        max(0, int(getattr(raw, "successes", int(success)))),
        tuple(int(value) for value in getattr(raw, "prefix_actions", ()) or ()),
        int(getattr(raw, "terminal_context", 0)),
        int(getattr(raw, "terminal_action", 0)),
        int(getattr(raw, "outcome_signature", 0)),
        lengths,
    )


def _game_validate_v820(self, candidate) -> V820ValidationResult:
    from v8 import trajectory_optimizer_v818 as v818

    if not _is_arc_validator(self.service):
        return _validation_result_from_legacy(_BASE_GAME_VALIDATE(self, candidate))

    prefix = tuple(self.service._v818_prefix_for(candidate))
    successes = 0
    attempts = 0
    total_actions = 0
    successful_lengths: list[int] = []
    last_reason = "target_not_reached"
    terminal_context = terminal_action = outcome_signature = 0
    terminal_state = ""
    levels_completed = 0

    for execution_seed in v818._VALIDATION_SEEDS:
        _raise_if_validation_cancelled(self.service)
        attempts += 1
        try:
            ok, steps, reason, context, action, outcome, _prefix_steps = self._trial(
                candidate,
                execution_seed,
                prefix,
            )
        except _ValidationCancelled:
            raise
        except BaseException as exc:
            ok = False
            steps = 0
            reason = f"{type(exc).__name__}: {exc}"
            context = action = outcome = 0
        total_actions += int(steps)
        last_reason = str(reason)
        if ok:
            successes += 1
            successful_lengths.append(int(steps))
            terminal_context = int(context)
            terminal_action = int(action)
            outcome_signature = int(outcome)
            terminal_state = str(candidate.source.target.terminal_state)
            levels_completed = int(candidate.source.target.levels_completed)

    required = max(1, (len(v818._VALIDATION_SEEDS) + 1) // 2)
    accepted = successes >= required
    return V820ValidationResult(
        accepted,
        total_actions,
        "target_preserved" if accepted else last_reason,
        terminal_state,
        levels_completed,
        attempts,
        successes,
        prefix,
        terminal_context,
        terminal_action,
        outcome_signature,
        tuple(successful_lengths),
    )


def _candidate(optimizer, source, kind: str, actions, start: int = 0, removed: int = 0):
    values = tuple(int(value) for value in actions)
    return optimizer.TrajectoryCandidate(
        optimizer._candidate_id(source, kind, values),
        source,
        kind,
        values,
        int(start),
        int(removed),
    )


def _generate_v820(source, config=None):
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    cfg = config or optimizer.TrajectoryOptimizerConfig()
    actions = tuple(int(value) for value in source.actions)
    n = len(actions)
    limit = max(0, int(cfg.max_candidates_per_round))
    if n <= 1 or limit <= 0:
        return ()

    rows: list[object] = []
    seen: set[tuple[int, ...]] = set()

    def add(kind: str, candidate_actions: tuple[int, ...], start: int, removed: int) -> None:
        if (
            len(rows) >= limit
            or not candidate_actions
            or len(candidate_actions) >= n
            or candidate_actions in seen
        ):
            return
        seen.add(candidate_actions)
        rows.append(_candidate(optimizer, source, kind, candidate_actions, start, removed))

    cleanup_reserve = min(8, max(2, limit // 3))
    repeat_reserve = min(4, max(0, limit - cleanup_reserve - 1))
    delta_budget = max(1, limit - cleanup_reserve - repeat_reserve)

    # True coarse delta-debugging order: halves, quarters, eighths, ...
    granularity = 2
    while granularity <= n and len(rows) < delta_budget:
        chunk = max(1, int(math.ceil(n / granularity)))
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            add(_DELTA_DELETE, actions[:start] + actions[end:], start, end - start)
            if len(rows) >= delta_budget:
                break
        if chunk <= 1:
            break
        granularity *= 2

    # Exact repetition is still useful, but only after target-aware coarse search.
    for row in v818._BASE_GENERATE(source, cfg):
        if len(rows) >= limit - cleanup_reserve:
            break
        if row.edit_kind != "REDUCE_REPEAT":
            continue
        values = tuple(int(value) for value in row.actions)
        if values in seen:
            continue
        seen.add(values)
        rows.append(row)

    local_slots = max(1, cleanup_reserve // 2)
    local_limit = min(max(2, int(cfg.max_segment_delete)), max(2, n - 1))
    local_added = 0
    for length in range(local_limit, 1, -1):
        for start in v818._sample_positions(n - length + 1, max(2, local_slots)):
            before = len(rows)
            add("DELETE_SEGMENT", actions[:start] + actions[start + length :], start, length)
            local_added += int(len(rows) > before)
            if local_added >= local_slots or len(rows) >= limit:
                break
        if local_added >= local_slots or len(rows) >= limit:
            break

    remaining = max(0, limit - len(rows))
    for index in v818._sample_positions(n, remaining):
        add("DELETE_ACTION", actions[:index] + actions[index + 1 :], index, 1)
        if len(rows) >= limit:
            break
    return tuple(rows)


def _available_actions_at_anchor(validator, candidate) -> tuple[int, ...]:
    from v8 import trajectory_optimizer_v818 as v818

    prefix = tuple(validator.service._v818_prefix_for(candidate))
    actions: set[int] = set()
    for execution_seed in v818._VALIDATION_SEEDS:
        _raise_if_validation_cancelled(validator.service)
        env = validator._environment(execution_seed, candidate.source.anchor.env_root)
        valid = True
        for action in prefix:
            _raise_if_validation_cancelled(validator.service)
            available = {int(value) for value in env.available_actions()}
            if int(action) not in available:
                valid = False
                break
            env.step(int(action))
            if str(getattr(env, "last_outcome_state", "")) == "GAME_OVER":
                valid = False
                break
        if not valid or validator._target_reached(env, candidate.source):
            continue
        actions.update(int(value) for value in env.available_actions())
    return tuple(sorted(actions))


def _validate_tracked(service, validator, candidate, *, skip_attempted: bool = True):
    _raise_if_validation_cancelled(service)
    with service._lock:
        if skip_attempted and candidate.candidate_id in service._attempted:
            return None
        service._attempted.add(candidate.candidate_id)
        service._active_validations += 1
    try:
        try:
            result = validator.validate(candidate)
            _raise_if_validation_cancelled(service)
        except _ValidationCancelled:
            _preserve_cancelled_source(service, candidate)
            raise
    finally:
        with service._lock:
            service._active_validations -= 1
    with service._lock:
        service._validations += int(result.attempts)
        service._validation_successes += int(result.successes)
    return result


def _canonicalize_success(service, validator, candidate, result):
    from v8 import trajectory_optimizer_v814 as optimizer

    current_candidate = candidate
    current_result = result
    while bool(current_result.success):
        lengths = tuple(
            int(value)
            for value in getattr(current_result, "successful_prefix_lengths", ())
            if int(value) > 0
        )
        if not lengths:
            break
        required = max(lengths)
        if required >= len(current_candidate.actions):
            break
        shortened_actions = tuple(current_candidate.actions[:required])
        shortened = _candidate(
            optimizer,
            current_candidate.source,
            _TRUNCATE_SUCCESS_PREFIX,
            shortened_actions,
            required,
            len(current_candidate.actions) - required,
        )
        shortened_result = _validate_tracked(
            service,
            validator,
            shortened,
            skip_attempted=False,
        )
        if shortened_result is None or not bool(shortened_result.success):
            break
        current_candidate = shortened
        current_result = shortened_result
    return current_candidate, current_result


def _accept_frontier(service, candidate, result):
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    if not bool(result.success) or int(candidate.cost) >= int(candidate.source.cost):
        return False, None
    row = service._accept(candidate)
    accepted = getattr(row, "variant_id", None) == candidate.candidate_id
    if not accepted:
        return False, None

    validated = replace(
        row,
        attempts=int(result.attempts),
        successes=int(result.successes),
    )
    key = optimizer._frontier_key(validated.anchor, validated.target)
    target_key = v818._target_key(candidate.source)
    with service._lock:
        service._validated[key] = validated
    with service._v818_validator_lock:
        service._v818_frontier_cost[target_key] = min(
            int(candidate.cost),
            int(service._v818_frontier_cost.get(target_key, candidate.source.cost)),
        )
        service._v818_successful_frontiers += 1
        saved = max(0, int(candidate.source.cost) - int(candidate.cost))
        service._v818_saved_actions += saved
        if saved > service._v818_best_saved:
            service._v818_best_saved = saved
            service._v818_best_parent = int(candidate.source.cost)
            service._v818_best_candidate = int(candidate.cost)
    v818._record_validated_prefix(service, candidate, result)
    service._publish_validated()
    return True, validated


def _submit_next_source(service, candidate, validated) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    if validated is None or int(validated.cost) <= 1:
        return
    next_source = optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(validated.anchor, validated.target, validated.actions),
        validated.anchor,
        validated.target,
        validated.actions,
        validated.strategy_uid,
        validated.target_outcome_uid,
        int(candidate.source.round_index) + 1,
    )
    service.submit_trajectory(next_source)


def _log_validation(service, candidate, result, *, accepted: bool, original_cost: int | None = None) -> None:
    service._log(
        "validation",
        candidate_id=candidate.candidate_id,
        game=str(candidate.source.anchor.source_id),
        edit=candidate.edit_kind,
        parent_cost=candidate.source.cost,
        candidate_cost=candidate.cost,
        original_candidate_cost=(candidate.cost if original_cost is None else int(original_cost)),
        attempts=result.attempts,
        successes=result.successes,
        successful_prefix_lengths=list(getattr(result, "successful_prefix_lengths", ()) or ()),
        success=bool(result.success),
        frontier=bool(accepted),
        reason=result.reason,
    )


def _process_candidate(service, validator, candidate):
    result = _validate_tracked(service, validator, candidate)
    if result is None:
        return False, None

    effective_candidate = candidate
    effective_result = result
    if (
        bool(result.success)
        and candidate.edit_kind != "VALIDATE_SOURCE"
        and int(candidate.cost) < int(candidate.source.cost)
    ):
        effective_candidate, effective_result = _canonicalize_success(
            service,
            validator,
            candidate,
            result,
        )

    accepted, validated = _accept_frontier(service, effective_candidate, effective_result)
    _log_validation(
        service,
        effective_candidate,
        effective_result,
        accepted=accepted,
        original_cost=int(candidate.cost),
    )
    if service.on_validation is not None:
        service.on_validation(effective_candidate, effective_result, validated)
    if accepted:
        _submit_next_source(service, effective_candidate, validated)
    return accepted, validated


def _process_target_minimize(service, validator, marker) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    source = marker.source
    target_key = v818._target_key(source)
    with service._v818_validator_lock:
        frontier = int(service._v818_frontier_cost.get(target_key, source.cost))
    if frontier <= 1:
        return

    direct_actions = _available_actions_at_anchor(validator, marker)
    with service._lock:
        service._candidates_generated += len(direct_actions) + 1
    for action in direct_actions:
        _raise_if_validation_cancelled(service)
        direct = _candidate(
            optimizer,
            source,
            _DIRECT_ACTION,
            (int(action),),
            0,
            max(0, int(source.cost) - 1),
        )
        accepted, validated = _process_candidate(service, validator, direct)
        if accepted and validated is not None and int(validated.cost) <= 1:
            return
        with service._v818_validator_lock:
            if int(service._v818_frontier_cost.get(target_key, source.cost)) <= 1:
                return

    # Replay the original source once and canonicalize the first successful prefix.
    source_probe = _candidate(
        optimizer,
        source,
        _TRUNCATE_SUCCESS_PREFIX,
        tuple(source.actions),
        0,
        0,
    )
    source_result = _validate_tracked(service, validator, source_probe)
    if source_result is None:
        return
    shortened, shortened_result = _canonicalize_success(
        service,
        validator,
        source_probe,
        source_result,
    )
    if int(shortened.cost) >= int(source.cost):
        service._log(
            "target_minimize",
            game=str(source.anchor.source_id),
            trajectory_id=str(source.trajectory_id),
            parent_cost=int(source.cost),
            canonical_cost=int(shortened.cost),
            direct_actions=len(direct_actions),
            improved=False,
        )
        return

    accepted, validated = _accept_frontier(service, shortened, shortened_result)
    _log_validation(
        service,
        shortened,
        shortened_result,
        accepted=accepted,
        original_cost=int(source.cost),
    )
    if service.on_validation is not None:
        service.on_validation(shortened, shortened_result, validated)
    if accepted:
        _submit_next_source(service, shortened, validated)


def _optimizer_loop_v820(service) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    try:
        while not service._stop.is_set():
            v818._restore_pending_sources(service)
            v818._ingest_inbox_v818(service)
            v818._start_waiting_validators(service)
            try:
                source = service._sources.get(timeout=float(service.config.poll_interval_seconds))
            except queue.Empty:
                continue
            try:
                v818._begin_source_work(service, source)
                if int(source.round_index) >= int(service.config.max_optimization_rounds):
                    continue
                v818._register_source_prefix(service, source)
                key = v818._target_key(source)
                with service._v818_validator_lock:
                    prior = service._v818_frontier_cost.get(key)
                    if prior is None or int(source.cost) < int(prior):
                        service._v818_frontier_cost[key] = int(source.cost)

                if _is_arc_validator(service):
                    marker = _candidate(
                        optimizer,
                        source,
                        _TARGET_MINIMIZE,
                        tuple(source.actions),
                        0,
                        0,
                    )
                    if not v818._route_candidate(service, marker):
                        continue
                    rows = _generate_v820(source, service.config)
                else:
                    rows = _BASE_GENERATE_V818(source, service.config)

                with service._lock:
                    service._candidates_generated += len(rows)
                service._log(
                    "candidates",
                    trajectory_id=source.trajectory_id,
                    game=str(source.anchor.source_id),
                    parent_cost=source.cost,
                    count=len(rows),
                    round=source.round_index,
                    target_aware=bool(_is_arc_validator(service)),
                )
                for candidate in rows:
                    if not v818._route_candidate(service, candidate):
                        break
            finally:
                v818._end_source_routing(service, source)
                service._sources.task_done()
    except BaseException as exc:
        service._fail(exc)


def _game_validator_loop_v820(service, game_id: str) -> None:
    from v8 import trajectory_optimizer_v818 as v818

    game = str(game_id)
    validator = v818._GameReplayValidator(service, game)
    idle_since = time.monotonic()
    processed = 0
    try:
        while not service._stop.is_set():
            if _validation_cancel_requested(service):
                break
            with service._v818_validator_lock:
                q = service._v818_game_queues.setdefault(
                    game, queue.Queue(maxsize=v818._PER_GAME_QUEUE_CAPACITY)
                )
                waiting_exists = bool(service._v818_waiting_games)
            try:
                candidate = q.get(timeout=0.20)
            except queue.Empty:
                if waiting_exists and time.monotonic() - idle_since >= v818._VALIDATION_IDLE_SECONDS:
                    break
                continue
            idle_since = time.monotonic()
            cancelled = False
            try:
                if candidate.edit_kind == _TARGET_MINIMIZE:
                    _process_target_minimize(service, validator, candidate)
                    continue

                target_key = v818._target_key(candidate.source)
                with service._v818_validator_lock:
                    frontier = service._v818_frontier_cost.get(
                        target_key,
                        candidate.source.cost,
                    )
                if (
                    candidate.edit_kind != "VALIDATE_SOURCE"
                    and int(candidate.cost) >= int(frontier)
                    and int(frontier) < int(candidate.source.cost)
                ):
                    service._log(
                        "validation_pruned",
                        candidate_id=candidate.candidate_id,
                        game=game,
                        candidate_cost=candidate.cost,
                        frontier_cost=frontier,
                    )
                    continue
                _process_candidate(service, validator, candidate)
            except _ValidationCancelled:
                cancelled = True
            finally:
                if not cancelled:
                    v818._finish_candidate_work(service, candidate)
                q.task_done()
                processed += 1
            if cancelled:
                break
            if processed >= v818._VALIDATION_QUANTUM:
                with service._v818_validator_lock:
                    if service._v818_waiting_games:
                        break
    except BaseException as exc:
        service._fail(exc)
    finally:
        v818._retire_game_validator(service, game)


def install_trajectory_target_minimization_v820() -> None:
    global _INSTALLED, _BASE_GAME_VALIDATE, _BASE_GENERATE_V818
    if _INSTALLED:
        return

    from v8 import trajectory_optimizer_v818 as v818

    _BASE_GAME_VALIDATE = v818._GameReplayValidator.validate
    _BASE_GENERATE_V818 = v818._generate_v818
    v818._GameReplayValidator.validate = _game_validate_v820
    v818._optimizer_loop_v818 = _optimizer_loop_v820
    v818._game_validator_loop = _game_validator_loop_v820
    _INSTALLED = True
