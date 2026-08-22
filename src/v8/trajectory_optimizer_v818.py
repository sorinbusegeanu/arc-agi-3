from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from v8.model import MemoryLevel, MemoryType, MemoryUid, stable_u64


_INSTALLED = False
_MAX_VALIDATORS = 10
_VALIDATION_SEEDS = (0, 1)
_ACTIVITY_INTERVAL_SECONDS = 300.0
_VALIDATION_IDLE_SECONDS = 5.0
_PER_GAME_QUEUE_CAPACITY = 256
_MIN_VALIDATED_RELIABILITY = 0.50

_BASE_GENERATE = None
_BASE_SERVICE_INIT = None
_BASE_SERVICE_ACCEPT = None
_BASE_STATE_DICT = None
_BASE_LOAD_STATE = None
_BASE_LIVE_PLAN = None
_BASE_RUNTIME_INIT = None


@dataclass(frozen=True, slots=True)
class V818ValidationResult:
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


class _GameReplayValidator:
    def __init__(self, service, game_id: str) -> None:
        self.service = service
        self.game_id = str(game_id)
        self._envs: dict[tuple[int, str | None], object] = {}

    def _environment(self, execution_seed: int, env_root: str | None):
        from v7.environment.arc_adapter import ArcGridEnvironment

        key = (int(execution_seed), env_root)
        env = self._envs.get(key)
        if env is None:
            env = ArcGridEnvironment(
                game_id=self.game_id,
                seed=int(execution_seed),
                env_root=env_root,
            )
            env.game_wait_seconds = 0.0
            self._envs[key] = env
        else:
            env.reset()
        return env

    @staticmethod
    def _target_reached(env, target) -> bool:
        state = str(getattr(env, "last_outcome_state", ""))
        levels = int(getattr(env, "last_levels_completed", 0))
        if str(target.terminal_state) == "WIN":
            return state == "WIN"
        return levels >= int(target.levels_completed)

    @staticmethod
    def _action_available(env, action: int) -> bool:
        token = int(action)
        if token in {int(value) for value in env.available_actions()}:
            return True
        # Exact-coordinate paging bounds exploration, not the native ACTION6
        # contract. Captured in-bounds coordinates remain executable in replay.
        try:
            from v8 import click_exploration_v848 as click

            return bool(
                click._is_exact_click_token(token)
                and click._valid_exact_click(token, env.observe())
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _trial(self, candidate, execution_seed: int, prefix: tuple[int, ...]):
        from v7.environment.encoding import structural_grid_signature, transition_signature

        env = self._environment(execution_seed, candidate.source.anchor.env_root)
        target = candidate.source.target
        prefix_executed = 0
        for action in prefix:
            if not self._action_available(env, int(action)):
                return False, 0, "prefix_action_unavailable", 0, 0, 0, prefix_executed
            env.step(int(action))
            prefix_executed += 1
            if str(getattr(env, "last_outcome_state", "")) == "GAME_OVER":
                return False, 0, "anchor_failed", 0, 0, 0, prefix_executed

        if self._target_reached(env, target):
            return False, 0, "anchor_already_reaches_target", 0, 0, 0, prefix_executed

        candidate_steps = 0
        for action in candidate.actions:
            if not self._action_available(env, int(action)):
                return False, candidate_steps, "candidate_action_unavailable", 0, 0, 0, prefix_executed
            before = env.observe()
            context = int(structural_grid_signature(before))
            after = env.step(int(action))
            candidate_steps += 1
            outcome = int(transition_signature(before, after))
            if self._target_reached(env, target):
                return (
                    True,
                    candidate_steps,
                    "target_preserved",
                    context,
                    int(action),
                    outcome,
                    prefix_executed,
                )
            if str(getattr(env, "last_outcome_state", "")) == "GAME_OVER":
                return False, candidate_steps, "candidate_failed", 0, 0, 0, prefix_executed

        return False, candidate_steps, "target_not_reached", 0, 0, 0, prefix_executed

    def validate(self, candidate) -> V818ValidationResult:
        prefix = self.service._v818_prefix_for(candidate)
        successes = 0
        attempts = 0
        total_actions = 0
        last_reason = "target_not_reached"
        terminal_context = terminal_action = outcome_signature = 0
        terminal_state = ""
        levels_completed = 0

        for execution_seed in _VALIDATION_SEEDS:
            attempts += 1
            try:
                ok, steps, reason, context, action, outcome, _prefix_steps = self._trial(
                    candidate,
                    execution_seed,
                    prefix,
                )
            except BaseException as exc:
                ok = False
                steps = 0
                reason = f"{type(exc).__name__}: {exc}"
                context = action = outcome = 0
            total_actions += int(steps)
            last_reason = str(reason)
            if ok:
                successes += 1
                terminal_context = int(context)
                terminal_action = int(action)
                outcome_signature = int(outcome)
                terminal_state = str(candidate.source.target.terminal_state)
                levels_completed = int(candidate.source.target.levels_completed)

        required = max(1, (len(_VALIDATION_SEEDS) + 1) // 2)
        accepted = successes >= required
        return V818ValidationResult(
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
        )


def _seedless_anchor_hash(optimizer, anchor, target) -> int:
    return stable_u64(
        str(anchor.source_id),
        int(optimizer.action_sequence_hash(anchor.prefix_actions)),
        int(target.levels_completed),
        str(target.terminal_state),
        person=b"v8.18-anchor",
    )


def _seedless_trajectory_id(optimizer, anchor, target, actions) -> str:
    value = stable_u64(
        _seedless_anchor_hash(optimizer, anchor, target),
        int(optimizer.action_sequence_hash(actions)),
        person=b"v8.18-trajectory",
    )
    return f"{value:016x}"


def _seedless_validated_id(optimizer, anchor, target, actions, edit_kind: str) -> str:
    value = stable_u64(
        _seedless_anchor_hash(optimizer, anchor, target),
        int(optimizer.action_sequence_hash(actions)),
        str(edit_kind),
        person=b"v8.18-validated",
    )
    return f"{value:016x}"


def _seedless_candidate_id(optimizer, source, kind: str, actions: tuple[int, ...]) -> str:
    value = stable_u64(
        str(source.trajectory_id),
        str(kind),
        int(optimizer.action_sequence_hash(actions)),
        person=b"v8.18-candidate",
    )
    return f"{value:016x}"


def _seedless_variant_key(optimizer, candidate) -> tuple[int, int, int, int]:
    sequence = optimizer._SEQUENCE_MARKER | (
        optimizer.action_sequence_hash(candidate.actions) & optimizer._SEQUENCE_MASK
    )
    target = candidate.source.target_outcome_uid
    return (
        int(sequence),
        int(target.hi),
        int(target.lo),
        int(_seedless_anchor_hash(optimizer, candidate.source.anchor, candidate.source.target)),
    )


def _seedless_strategy_uid_for_row(optimizer, row) -> MemoryUid:
    sequence = optimizer._SEQUENCE_MARKER | (
        optimizer.action_sequence_hash(row.actions) & optimizer._SEQUENCE_MASK
    )
    key = (
        int(sequence),
        int(row.target_outcome_uid.hi),
        int(row.target_outcome_uid.lo),
        int(_seedless_anchor_hash(optimizer, row.anchor, row.target)),
    )
    return MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, key)


def _sample_positions(length: int, count: int) -> tuple[int, ...]:
    n = max(0, int(length))
    cap = max(0, int(count))
    if n <= 0 or cap <= 0:
        return ()
    if n <= cap:
        return tuple(range(n))
    values = []
    for index in range(cap):
        value = min(n - 1, int(index * n / cap))
        if not values or values[-1] != value:
            values.append(value)
    return tuple(values)


def _generate_v818(source, config=None):
    from v8 import trajectory_optimizer_v814 as optimizer

    cfg = config or optimizer.TrajectoryOptimizerConfig()
    actions = tuple(int(value) for value in source.actions)
    n = len(actions)
    if n <= 1:
        return ()
    limit = max(0, int(cfg.max_candidates_per_round))
    if limit <= 0:
        return ()

    unique: dict[tuple[int, ...], object] = {}

    def add(kind: str, candidate_actions: tuple[int, ...], start: int, removed: int, *, block=0, repeats=0):
        if not candidate_actions or len(candidate_actions) >= n or candidate_actions in unique:
            return
        unique[candidate_actions] = optimizer.TrajectoryCandidate(
            _seedless_candidate_id(optimizer, source, kind, candidate_actions),
            source,
            kind,
            candidate_actions,
            int(start),
            int(removed),
            int(block),
            int(repeats),
        )

    # Exact repeat collapse remains the highest-value structural edit.
    for row in _BASE_GENERATE(source, cfg):
        if row.edit_kind != "REDUCE_REPEAT":
            continue
        add(
            row.edit_kind,
            tuple(row.actions),
            row.removed_start,
            row.removed_length,
            block=row.repeat_block_length,
            repeats=row.repeat_count,
        )
        if len(unique) >= max(8, limit // 2):
            break

    # Long solutions use coarse delta-debugging deletions before local edits.
    lengths = {
        max(2, n // divisor)
        for divisor in (2, 3, 4, 6, 8, 12, 16)
        if n // divisor >= 2
    }
    for length in sorted(lengths, reverse=True):
        positions = _sample_positions(n - length + 1, 6)
        for start in positions:
            add(
                "DELETE_SEGMENT",
                actions[:start] + actions[start + length :],
                start,
                length,
            )
            if len(unique) >= limit:
                return tuple(unique.values())

    # Then search bounded local segments, largest first.
    local_max = min(max(2, int(cfg.max_segment_delete)), max(2, n - 1))
    for length in range(local_max, 1, -1):
        for start in _sample_positions(n - length + 1, 8):
            add(
                "DELETE_SEGMENT",
                actions[:start] + actions[start + length :],
                start,
                length,
            )
            if len(unique) >= limit:
                return tuple(unique.values())

    # Single-action deletion is deliberately last for long trajectories.
    for index in _sample_positions(n, max(8, limit // 4)):
        add("DELETE_ACTION", actions[:index] + actions[index + 1 :], index, 1)
        if len(unique) >= limit:
            break

    return tuple(unique.values())


def _target_key(source) -> tuple[object, ...]:
    return (
        str(source.anchor.source_id),
        int(source.target.levels_completed),
        str(source.target.terminal_state),
        int(source.target_outcome_uid.hi),
        int(source.target_outcome_uid.lo),
    )


def _queue_depth(service) -> int:
    with service._v818_validator_lock:
        queues = tuple(service._v818_game_queues.values())
    return sum(int(q.qsize()) for q in queues)


def _alive_validators(service) -> dict[str, threading.Thread]:
    return {
        game: thread
        for game, thread in service._v818_validator_threads.items()
        if thread is not None and thread.is_alive()
    }


def _ensure_validator(service, game_id: str) -> None:
    game = str(game_id)
    thread_to_start = None
    with service._v818_validator_lock:
        service._v818_game_queues.setdefault(game, queue.Queue(maxsize=_PER_GAME_QUEUE_CAPACITY))
        alive = _alive_validators(service)
        service._v818_validator_threads = dict(alive)
        existing = alive.get(game)
        if existing is not None:
            service._v818_waiting_games.discard(game)
            return
        if len(alive) >= int(service._v818_max_validators):
            service._v818_waiting_games.add(game)
            return
        thread_to_start = threading.Thread(
            target=_game_validator_loop,
            args=(service, game),
            name=f"v8-trajectory-validator-{game}",
            daemon=True,
        )
        service._v818_validator_threads[game] = thread_to_start
        service._v818_waiting_games.discard(game)
    if thread_to_start is not None:
        thread_to_start.start()


def _start_waiting_validators(service) -> None:
    with service._v818_validator_lock:
        waiting = sorted(service._v818_waiting_games)
    for game in waiting:
        _ensure_validator(service, game)


def _route_candidate(service, candidate) -> bool:
    game = str(candidate.source.anchor.source_id)
    with service._v818_validator_lock:
        q = service._v818_game_queues.setdefault(
            game, queue.Queue(maxsize=_PER_GAME_QUEUE_CAPACITY)
        )
    _ensure_validator(service, game)
    while not service._stop.is_set():
        try:
            q.put(candidate, timeout=0.10)
            return True
        except queue.Full:
            _start_waiting_validators(service)
    return False


def _ingest_inbox_v818(service) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    service.inbox.mkdir(parents=True, exist_ok=True)
    for path in sorted(service.inbox.glob("*.json"))[:128]:
        remove = False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            row = optimizer.SuccessfulTrajectory.from_dict(raw)
            with service._lock:
                duplicate = row.trajectory_id in service._seen_sources
            # v8.19 routes sampler trajectories directly to source validation
            # and tracks them separately from the original optimizer sources.
            # Treat those as consumed too; otherwise submit_trajectory() rejects
            # every duplicate while the inbox file remains forever, preventing
            # runtime shutdown from ever reaching quiescence.
            v819_lock = getattr(service, "_v819_lock", None)
            if not duplicate and v819_lock is not None:
                with v819_lock:
                    duplicate = row.trajectory_id in getattr(
                        service, "_v819_source_seen", ()
                    )
            if duplicate:
                remove = True
            else:
                remove = bool(service.submit_trajectory(row))
        except BaseException as exc:
            service._log("inbox_error", path=str(path), error=f"{type(exc).__name__}: {exc}")
            remove = True
        if remove:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _register_source_prefix(service, source) -> None:
    level = max(0, int(source.target.levels_completed) - 1)
    prefix = tuple(int(value) for value in source.anchor.prefix_actions)
    if level <= 0:
        prefix = ()
    game = str(source.anchor.source_id)
    with service._v818_validator_lock:
        by_level = service._v818_best_prefixes.setdefault(game, {})
        prior = by_level.get(level)
        if prior is None or len(prefix) < len(prior):
            by_level[level] = prefix


def _prefix_for(service, candidate) -> tuple[int, ...]:
    game = str(candidate.source.anchor.source_id)
    level = max(0, int(candidate.source.target.levels_completed) - 1)
    original = tuple(int(value) for value in candidate.source.anchor.prefix_actions)
    with service._v818_validator_lock:
        optimized = service._v818_best_prefixes.get(game, {}).get(level)
    if optimized is None:
        return original
    if not original or len(optimized) <= len(original):
        return tuple(optimized)
    return original


def _record_validated_prefix(service, candidate, result) -> None:
    game = str(candidate.source.anchor.source_id)
    level = max(0, int(candidate.source.target.levels_completed))
    full = tuple(result.prefix_actions) + tuple(candidate.actions)
    with service._v818_validator_lock:
        by_level = service._v818_best_prefixes.setdefault(game, {})
        prior = by_level.get(level)
        if prior is None or len(full) < len(prior):
            by_level[level] = full


def _optimizer_loop_v818(service) -> None:
    try:
        while not service._stop.is_set():
            _ingest_inbox_v818(service)
            _start_waiting_validators(service)
            try:
                source = service._sources.get(timeout=float(service.config.poll_interval_seconds))
            except queue.Empty:
                continue
            try:
                if int(source.round_index) >= int(service.config.max_optimization_rounds):
                    continue
                _register_source_prefix(service, source)
                key = _target_key(source)
                with service._v818_validator_lock:
                    prior = service._v818_frontier_cost.get(key)
                    if prior is None or int(source.cost) < int(prior):
                        service._v818_frontier_cost[key] = int(source.cost)
                rows = _generate_v818(source, service.config)
                with service._lock:
                    service._candidates_generated += len(rows)
                service._log(
                    "candidates",
                    trajectory_id=source.trajectory_id,
                    game=str(source.anchor.source_id),
                    parent_cost=source.cost,
                    count=len(rows),
                    round=source.round_index,
                )
                for candidate in rows:
                    if not _route_candidate(service, candidate):
                        break
            finally:
                service._sources.task_done()
    except BaseException as exc:
        service._fail(exc)


def _game_validator_loop(service, game_id: str) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    game = str(game_id)
    validator = _GameReplayValidator(service, game)
    idle_since = time.monotonic()
    try:
        while not service._stop.is_set():
            with service._v818_validator_lock:
                q = service._v818_game_queues.setdefault(
                    game, queue.Queue(maxsize=_PER_GAME_QUEUE_CAPACITY)
                )
                waiting_exists = bool(service._v818_waiting_games)
            try:
                candidate = q.get(timeout=0.20)
            except queue.Empty:
                if waiting_exists and time.monotonic() - idle_since >= _VALIDATION_IDLE_SECONDS:
                    break
                continue
            idle_since = time.monotonic()
            try:
                target_key = _target_key(candidate.source)
                with service._v818_validator_lock:
                    frontier = service._v818_frontier_cost.get(target_key, candidate.source.cost)
                if int(candidate.cost) >= int(frontier) and int(frontier) < int(candidate.source.cost):
                    service._log(
                        "validation_pruned",
                        candidate_id=candidate.candidate_id,
                        game=game,
                        candidate_cost=candidate.cost,
                        frontier_cost=frontier,
                    )
                    continue

                with service._lock:
                    if candidate.candidate_id in service._attempted:
                        continue
                    service._attempted.add(candidate.candidate_id)
                    service._active_validations += 1
                try:
                    result = validator.validate(candidate)
                finally:
                    with service._lock:
                        service._active_validations -= 1

                with service._lock:
                    service._validations += int(result.attempts)
                    service._validation_successes += int(result.successes)

                accepted = False
                validated = None
                if bool(result.success) and int(candidate.cost) < int(candidate.source.cost):
                    row = service._accept(candidate)
                    accepted = getattr(row, "variant_id", None) == candidate.candidate_id
                    if accepted:
                        validated = replace(
                            row,
                            attempts=int(result.attempts),
                            successes=int(result.successes),
                        )
                        key = optimizer._frontier_key(validated.anchor, validated.target)
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
                        _record_validated_prefix(service, candidate, result)
                        service._publish_validated()

                service._log(
                    "validation",
                    candidate_id=candidate.candidate_id,
                    game=game,
                    edit=candidate.edit_kind,
                    parent_cost=candidate.source.cost,
                    candidate_cost=candidate.cost,
                    attempts=result.attempts,
                    successes=result.successes,
                    success=bool(result.success),
                    frontier=accepted,
                    reason=result.reason,
                )

                if service.on_validation is not None:
                    service.on_validation(candidate, result, validated)

                if accepted and validated is not None:
                    next_source = optimizer.SuccessfulTrajectory(
                        optimizer._trajectory_id(
                            validated.anchor,
                            validated.target,
                            validated.actions,
                        ),
                        validated.anchor,
                        validated.target,
                        validated.actions,
                        validated.strategy_uid,
                        validated.target_outcome_uid,
                        int(candidate.source.round_index) + 1,
                    )
                    service.submit_trajectory(next_source)
            finally:
                q.task_done()
    except BaseException as exc:
        service._fail(exc)
    finally:
        with service._v818_validator_lock:
            current = service._v818_validator_threads.get(game)
            if current is threading.current_thread():
                service._v818_validator_threads.pop(game, None)
        _start_waiting_validators(service)


def _drain_v818(service, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline:
        _ingest_inbox_v818(service)
        _start_waiting_validators(service)
        with service._lock:
            active = int(service._active_validations)
        with service._v818_validator_lock:
            queues = tuple(service._v818_game_queues.values())
        if (
            service._sources.unfinished_tasks == 0
            and all(q.unfinished_tasks == 0 for q in queues)
            and active == 0
            and not any(service.inbox.glob("*.json"))
        ):
            return True
        time.sleep(0.02)
    return False


def _activity_snapshot(service) -> tuple[int, int, int, int, int]:
    with service._lock:
        return (
            int(service._trajectories_seen),
            int(service._candidates_generated),
            int(service._validations),
            int(service._validation_successes),
            len(service._validated),
        )


def _emit_activity_if_due(service, *, force: bool = False) -> bool:
    now = time.monotonic()
    last = float(getattr(service, "_v818_last_activity_report", 0.0))
    if not force and now - last < _ACTIVITY_INTERVAL_SECONDS:
        return False
    current = _activity_snapshot(service)
    prior = tuple(getattr(service, "_v818_activity_baseline", (0, 0, 0, 0, 0)))
    delta = tuple(max(0, int(a) - int(b)) for a, b in zip(current, prior, strict=True))
    service._v818_last_activity_report = now
    service._v818_activity_baseline = current
    if not any(delta[:4]):
        return False
    with service._v818_validator_lock:
        validators = len(_alive_validators(service))
        games = sum(1 for q in service._v818_game_queues.values() if q.unfinished_tasks > 0)
        saved = int(service._v818_saved_actions)
        best_parent = int(service._v818_best_parent)
        best_candidate = int(service._v818_best_candidate)
    print(
        f'[{time.strftime("%H:%M")}] trajectory optimization '
        f"validators={validators} games={games} trajectories={delta[0]} "
        f"generated={delta[1]} validations={delta[2]} successes={delta[3]} "
        f"frontiers={current[4]} queued={_queue_depth(service)} saved={saved} "
        f"best={best_parent}->{best_candidate}",
        flush=True,
    )
    return True


def _reporter_loop_v818(service) -> None:
    from v8 import trajectory_optimizer_stdout_v816 as stdout

    while not service._stop.wait(1.0):
        try:
            stdout._emit_success_report_if_due(service)
            _emit_activity_if_due(service)
        except BaseException as exc:
            service._fail(exc)
            return


def _start_v818(service) -> None:
    if service._optimizer_thread is not None and service._optimizer_thread.is_alive():
        return
    service.root.mkdir(parents=True, exist_ok=True)
    service.inbox.mkdir(parents=True, exist_ok=True)
    service._publish_validated()
    service._optimizer_thread = threading.Thread(
        target=_optimizer_loop_v818,
        args=(service,),
        name="v8-trajectory-optimizer",
        daemon=True,
    )
    service._optimizer_thread.start()
    thread = threading.Thread(
        target=_reporter_loop_v818,
        args=(service,),
        name="v8-trajectory-optimizer-reporter",
        daemon=True,
    )
    service._v818_reporter_thread = thread
    thread.start()


def _stop_v818(service, *, drain: bool = True, timeout: float = 10.0) -> None:
    if drain:
        _drain_v818(service, timeout=max(0.0, float(timeout) * 0.7))
    service._stop.set()
    if service._optimizer_thread is not None:
        service._optimizer_thread.join(timeout=max(0.1, float(timeout) * 0.2))
    with service._v818_validator_lock:
        threads = tuple(service._v818_validator_threads.values())
    for thread in threads:
        thread.join(timeout=max(0.1, float(timeout) * 0.1))
    reporter = getattr(service, "_v818_reporter_thread", None)
    if reporter is not None:
        reporter.join(timeout=min(1.0, max(0.1, float(timeout))))
    _emit_activity_if_due(service, force=True)
    service._publish_validated()


def _state_dict_v818(service) -> dict[str, object]:
    payload = _BASE_STATE_DICT(service)
    payload["version"] = 2
    with service._v818_validator_lock:
        payload["best_prefixes"] = {
            game: {
                str(level): list(actions)
                for level, actions in sorted(by_level.items())
            }
            for game, by_level in sorted(service._v818_best_prefixes.items())
        }
    return payload


def _load_state_v818(service, state: dict[str, object] | None) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    if not state:
        return
    raw = dict(state)
    version = int(raw.get("version", 1))
    if version < 2:
        migrated = []
        for item in raw.get("validated", ()):
            if not isinstance(item, dict):
                continue
            row = optimizer.ValidatedTrajectory.from_dict(item)
            migrated.append(row.to_dict())
        raw["version"] = 2
        raw["validated"] = migrated
        # Old IDs encoded seeds; rediscover/revalidate under seedless identity.
        raw["seen_sources"] = []
        raw["attempted"] = []
    _BASE_LOAD_STATE(service, raw)
    best = raw.get("best_prefixes", {})
    if isinstance(best, dict):
        with service._v818_validator_lock:
            for game, levels in best.items():
                if not isinstance(levels, dict):
                    continue
                target = service._v818_best_prefixes.setdefault(str(game), {})
                for level, actions in levels.items():
                    try:
                        target[int(level)] = tuple(int(value) for value in actions)
                    except (TypeError, ValueError):
                        continue


def _resolve_target_outcome(runtime, candidate, result) -> MemoryUid:
    current = candidate.source.target_outcome_uid
    if not current.is_zero:
        if any(row.uid == current for row in runtime.read_view.node_records(level=MemoryLevel.M6)):
            return current
    if not bool(getattr(result, "success", False)):
        return MemoryUid.zero()
    if not int(getattr(result, "terminal_context", 0)) or not int(getattr(result, "outcome_signature", 0)):
        return MemoryUid.zero()

    try:
        from v8 import behavior_recovery as behavior
        from v8.normalized_memory_v086_fixups import _grounded_context

        source_hash = stable_u64(candidate.source.anchor.source_id, person=b"v8-game")
        raw_context = int(result.terminal_context)
        contexts = (
            int(_grounded_context(source_hash, raw_context)),
            raw_context,
        )
        for context in contexts:
            matches = behavior.observed_outcome_uids(
                runtime.read_view,
                context_signature=context,
                action_id=int(result.terminal_action),
                outcome_signature=int(result.outcome_signature),
            )
            if matches:
                return sorted(matches)[0]
    except BaseException:
        return MemoryUid.zero()
    return MemoryUid.zero()


def _publish_resolved_validation(runtime, candidate, result, validated, target_uid: MemoryUid) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    source = replace(candidate.source, target_outcome_uid=target_uid)
    resolved_candidate = replace(candidate, source=source)
    strategy_uid = optimizer.variant_strategy_uid(resolved_candidate)
    resolved = replace(
        validated,
        target_outcome_uid=target_uid,
        strategy_uid=strategy_uid,
    )
    service = runtime._v814_trajectory_optimizer
    key = optimizer._frontier_key(resolved.anchor, resolved.target)
    with service._lock:
        service._validated[key] = resolved
    service._publish_validated()
    optimizer._runtime_validation_callback(runtime, resolved_candidate, result, resolved)


def _retry_deferred(runtime) -> None:
    pending = list(getattr(runtime, "_v818_deferred_trajectory_bindings", ()))
    if not pending:
        return
    remaining = []
    for candidate, result, validated in pending:
        target_uid = _resolve_target_outcome(runtime, candidate, result)
        if target_uid.is_zero:
            remaining.append((candidate, result, validated))
            continue
        _publish_resolved_validation(runtime, candidate, result, validated, target_uid)
    runtime._v818_deferred_trajectory_bindings = remaining


def _runtime_validation_callback_v818(runtime, candidate, result, validated) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    _retry_deferred(runtime)
    if not bool(getattr(result, "success", False)):
        optimizer._runtime_validation_callback(runtime, candidate, result, None)
        return
    if validated is None:
        # A successful but non-frontier candidate is useful search information but
        # does not need another canonical M7 strategy.
        return
    target_uid = _resolve_target_outcome(runtime, candidate, result)
    if target_uid.is_zero:
        runtime._v818_deferred_trajectory_bindings.append((candidate, result, validated))
        return
    _publish_resolved_validation(runtime, candidate, result, validated, target_uid)


def _target_compatible_variant(view, plans, action_ids):
    from v8 import trajectory_optimizer_v814 as optimizer

    optimizer._refresh_view_variants(view)
    source_id = str(getattr(optimizer, "_CAPTURE_SOURCE_ID", ""))
    if not source_id:
        return None
    available = {int(value) for value in action_ids}
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
        if successes / attempts < _MIN_VALIDATED_RELIABILITY:
            continue
        if (
            row.parent_strategy_uid not in plan_strategies
            and int(row.actions[0]) not in plan_actions
        ):
            continue
        rows.append(row)
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (-int(row.saved_actions), int(row.cost), -int(row.successes), row.variant_id),
    )


def _plan_candidates_v818(view, context_signature, action_ids, **kwargs):
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.publication import PlannedAction

    available = {int(value) for value in action_ids}
    active = getattr(view, "_v814_active_variant", None)
    remaining = tuple(getattr(view, "_v814_active_actions", ()))
    if active is not None and remaining and int(remaining[0]) not in available:
        view._v814_active_variant = None
        view._v814_active_actions = ()

    base = tuple(_BASE_LIVE_PLAN(view, context_signature, action_ids, **kwargs))
    if base and float(getattr(base[0], "score", 0.0)) >= 900_000.0:
        if int(base[0].action_id) in available:
            return base
        view._v814_active_variant = None
        view._v814_active_actions = ()
        base = tuple(_BASE_LIVE_PLAN(view, context_signature, action_ids, **kwargs))

    if not bool(getattr(view, "_behavior_actor_mode", False)) or not base:
        return base

    selected = _target_compatible_variant(view, base, action_ids)
    if selected is None:
        return base
    view._v814_attempted_variants.add(selected.variant_id)
    view._v814_active_variant = selected
    view._v814_active_actions = tuple(selected.actions[1:])
    plan = PlannedAction(
        int(selected.actions[0]),
        selected.target_outcome_uid,
        selected.strategy_uid,
        950_000.0,
        False,
    )
    view._behavior_last_plans = (plan,)
    return (plan,)


def install_trajectory_optimizer_v818() -> None:
    global _INSTALLED
    global _BASE_GENERATE, _BASE_SERVICE_INIT, _BASE_SERVICE_ACCEPT
    global _BASE_STATE_DICT, _BASE_LOAD_STATE, _BASE_LIVE_PLAN, _BASE_RUNTIME_INIT
    if _INSTALLED:
        return

    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.publication import LiveReadView
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_GENERATE = optimizer.generate_optimization_candidates
    _BASE_SERVICE_INIT = optimizer.TrajectoryOptimizationService.__init__
    _BASE_SERVICE_ACCEPT = optimizer.TrajectoryOptimizationService._accept
    _BASE_STATE_DICT = optimizer.TrajectoryOptimizationService.state_dict
    _BASE_LOAD_STATE = optimizer.TrajectoryOptimizationService.load_state
    _BASE_LIVE_PLAN = LiveReadView.plan_candidates
    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__

    # Seed remains an execution parameter on the legacy compatibility object, but
    # it is removed from serialization, identity, frontiering and activation.
    def anchor_to_dict(self):
        return {
            "source_id": str(self.source_id),
            "prefix_actions": list(self.prefix_actions),
            "env_root": self.env_root,
        }

    def anchor_from_dict(cls, raw):
        return cls(
            str(raw.get("source_id", "")),
            0,
            tuple(int(value) for value in raw.get("prefix_actions", ())),
            None if raw.get("env_root") is None else str(raw.get("env_root")),
        )

    def successful_from_dict(cls, raw):
        anchor = optimizer.ReplayAnchor.from_dict(dict(raw.get("anchor", {})))
        target = optimizer.TrajectoryTarget.from_dict(dict(raw.get("target", {})))
        actions = tuple(int(value) for value in raw.get("actions", ()))
        return cls(
            _seedless_trajectory_id(optimizer, anchor, target, actions),
            anchor,
            target,
            actions,
            optimizer._uid_from_raw(raw.get("parent_strategy_uid")),
            optimizer._uid_from_raw(raw.get("target_outcome_uid")),
            int(raw.get("round_index", 0)),
        )

    def validated_from_dict(cls, raw):
        anchor = optimizer.ReplayAnchor.from_dict(dict(raw.get("anchor", {})))
        target = optimizer.TrajectoryTarget.from_dict(dict(raw.get("target", {})))
        actions = tuple(int(value) for value in raw.get("actions", ()))
        target_uid = optimizer._uid_from_raw(raw.get("target_outcome_uid"))
        provisional = cls(
            _seedless_validated_id(
                optimizer,
                anchor,
                target,
                actions,
                str(raw.get("edit_kind", "")),
            ),
            anchor,
            target,
            actions,
            MemoryUid.zero(),
            target_uid,
            optimizer._uid_from_raw(raw.get("parent_strategy_uid")),
            int(raw.get("parent_cost", 0)),
            str(raw.get("edit_kind", "")),
            int(raw.get("attempts", 1)),
            int(raw.get("successes", 1)),
        )
        return replace(provisional, strategy_uid=_seedless_strategy_uid_for_row(optimizer, provisional))

    def anchor_hash(anchor, target):
        return _seedless_anchor_hash(optimizer, anchor, target)

    def trajectory_id(anchor, target, actions):
        return _seedless_trajectory_id(optimizer, anchor, target, actions)

    def candidate_id(source, kind, actions):
        return _seedless_candidate_id(optimizer, source, kind, tuple(actions))

    def variant_key(candidate):
        return _seedless_variant_key(optimizer, candidate)

    def variant_uid(candidate):
        return MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, variant_key(candidate))

    def frontier_key(anchor, target):
        return f"{_seedless_anchor_hash(optimizer, anchor, target):016x}"

    def select_variant(rows, *, source_id, seed=None, action_history=(), attempted=None):
        del seed
        history = tuple(int(value) for value in action_history)
        blocked = attempted or set()
        candidates = [
            row
            for row in rows
            if row.variant_id not in blocked
            and row.anchor.source_id == str(source_id)
            and tuple(row.anchor.prefix_actions) == history
            and not row.target_outcome_uid.is_zero
            and row.actions
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda row: (-row.saved_actions, row.cost, -row.successes, row.variant_id),
        )

    optimizer.ReplayAnchor.to_dict = anchor_to_dict
    optimizer.ReplayAnchor.from_dict = classmethod(anchor_from_dict)
    optimizer.SuccessfulTrajectory.from_dict = classmethod(successful_from_dict)
    optimizer.ValidatedTrajectory.from_dict = classmethod(validated_from_dict)
    optimizer._anchor_hash = anchor_hash
    optimizer._trajectory_id = trajectory_id
    optimizer._candidate_id = candidate_id
    optimizer.variant_strategy_key = variant_key
    optimizer.variant_strategy_uid = variant_uid
    optimizer._frontier_key = frontier_key
    optimizer.select_validated_variant = select_variant
    optimizer.generate_optimization_candidates = _generate_v818

    def service_init(self, *args, **kwargs):
        _BASE_SERVICE_INIT(self, *args, **kwargs)
        self._v818_validator_lock = threading.RLock()
        self._v818_game_queues: dict[str, queue.Queue] = {}
        self._v818_validator_threads: dict[str, threading.Thread] = {}
        self._v818_waiting_games: set[str] = set()
        self._v818_max_validators = _MAX_VALIDATORS
        self._v818_best_prefixes: dict[str, dict[int, tuple[int, ...]]] = {}
        self._v818_frontier_cost: dict[tuple[object, ...], int] = {}
        self._v818_successful_frontiers = 0
        self._v818_saved_actions = 0
        self._v818_best_saved = 0
        self._v818_best_parent = 0
        self._v818_best_candidate = 0
        self._v818_last_activity_report = time.monotonic()
        self._v818_activity_baseline = _activity_snapshot(self)
        self._v818_reporter_thread = None
        self._v818_prefix_for = lambda candidate: _prefix_for(self, candidate)

    optimizer.TrajectoryOptimizationService.__init__ = service_init
    optimizer.TrajectoryOptimizationService._optimizer_loop = _optimizer_loop_v818
    optimizer.TrajectoryOptimizationService._validator_loop = lambda self: None
    optimizer.TrajectoryOptimizationService._ingest_inbox = _ingest_inbox_v818
    optimizer.TrajectoryOptimizationService.start = _start_v818
    optimizer.TrajectoryOptimizationService.stop = _stop_v818
    optimizer.TrajectoryOptimizationService.drain = _drain_v818
    optimizer.TrajectoryOptimizationService.state_dict = _state_dict_v818
    optimizer.TrajectoryOptimizationService.load_state = _load_state_v818
    optimizer.TrajectoryOptimizationService._v818_emit_activity_if_due = _emit_activity_if_due

    LiveReadView.plan_candidates = _plan_candidates_v818

    def runtime_init(self, *args, **kwargs):
        _BASE_RUNTIME_INIT(self, *args, **kwargs)
        self._v818_deferred_trajectory_bindings = []
        service = getattr(self, "_v814_trajectory_optimizer", None)
        if service is not None:
            service.on_validation = lambda candidate, result, validated: _runtime_validation_callback_v818(
                self,
                candidate,
                result,
                validated,
            )

    V82ContinuousMemoryRuntime.__init__ = runtime_init
    _INSTALLED = True
