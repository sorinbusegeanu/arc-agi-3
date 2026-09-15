from __future__ import annotations

import time
import contextlib
import importlib
import logging
import os
import traceback
import warnings
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Any
from random import Random

from v9.cognition.action_selection import choose_action
from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.memory.identity import stable_u64


_FORKSERVER_READY = False
_PROCESS_JOIN_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class EncodedTransition:
    actor_id: int
    producer_sequence: int
    global_step: int
    environment_identity: tuple[str, str, str, str]
    episode_id: int
    observation_schema_id: int
    before_signature: int
    action_id: int
    after_signature: int
    available_actions_after: int
    primary_valence: int
    symbols: tuple[object, ...]
    curriculum_step: str | None
    game_scenario: str
    symbols_only: bool = False
    action_schema_id: int = 0
    available_action_set_signature: int = 0
    boundary_scope: str = "NONE"
    task_success: bool = False
    task_failure: bool = False
    task_truncated: bool = False
    level_index: int = 0
    levels_completed: int = 0
    semantic_before: tuple[tuple[int, int, int, int, float], ...] = ()
    semantic_action: tuple[tuple[int, int, int, int, float], ...] = ()
    semantic_options: tuple[tuple[int, int, int, int, float], ...] = ()
    semantic_after: tuple[tuple[int, int, int, int, float], ...] = ()
    semantic_delta: tuple[tuple[int, int, int, int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ActorDone:
    actor_id: int
    game_id: str
    steps: int
    positive_boundaries: int
    negative_boundaries: int
    episode_boundaries: int
    resets: int
    policy_refreshes: int = 0
    task_successes: int = 0
    task_failures: int = 0
    task_truncations: int = 0
    levels_completed: int = 0
    grounded_action_influence: int = 0
    unique_contexts: int = 0
    unique_context_actions: int = 0
    mean_branching_factor: float = 0.0
    max_branching_factor: int = 0
    changed_transitions: int = 0


@dataclass(frozen=True, slots=True)
class ActorError:
    actor_id: int
    game_id: str
    message: str
    traceback_text: str


@dataclass(frozen=True, slots=True)
class WorkerStop:
    reason: str = "stop"


def _forkserver_bootstrap_main() -> None:
    return None


def resolve_start_method(start_method: str | None = None) -> str:
    return str(start_method or ("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"))


def ensure_process_server_ready(start_method: str | None = None) -> str:
    """Start the multiprocessing server while the parent is still small."""
    global _FORKSERVER_READY
    method = resolve_start_method(start_method)
    if method != "forkserver" or _FORKSERVER_READY:
        return method
    if mp.current_process().name != "MainProcess":
        return method
    ctx = mp.get_context(method)
    process = ctx.Process(target=_forkserver_bootstrap_main, name="v9-forkserver-bootstrap")
    process.start()
    process.join(timeout=15.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        raise RuntimeError("v9 forkserver bootstrap timed out")
    if process.exitcode != 0:
        raise RuntimeError(f"v9 forkserver bootstrap failed with code {process.exitcode}")
    process.close()
    _FORKSERVER_READY = True
    return method


def _close_queue(queue_obj: Any, *, drain: bool) -> None:
    if queue_obj is None:
        return
    # All application-level payloads are explicitly drained before topology
    # shutdown. Never wait on multiprocessing's private feeder thread here:
    # Queue.join_thread() has no timeout and can deadlock the epoch boundary.
    try:
        queue_obj.cancel_join_thread()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        queue_obj.close()
    except (AttributeError, OSError, ValueError):
        pass


def _close_process(process: Any) -> None:
    try:
        if process.is_alive():
            return
    except (AttributeError, ValueError):
        return
    try:
        process.close()
    except (AttributeError, OSError, ValueError):
        pass


def _load_factory(path: str):
    module_name, sep, attr = path.partition(":")
    if not sep:
        raise ValueError("factory path must use module:function")
    return getattr(importlib.import_module(module_name), attr)


def _action_set_signature(action_schema_id: int, actions: tuple[int, ...]) -> int:
    normalized = tuple(sorted(set(int(action) for action in actions)))
    return stable_u64(int(action_schema_id), *normalized, person=b"v9-action-set")


def _flush_child_queue(queue_obj: Any) -> None:
    """Ensure this child process has flushed all payloads it put on a queue."""
    try:
        queue_obj.close()
        queue_obj.join_thread()
    except (AttributeError, AssertionError, OSError, ValueError):
        pass


def actor_process_main(*, spec: Any, actor_id: int, steps: int, seed: int, env_root: str | None, initial_policy: ActorPolicySnapshot, policy_updates: Any, epsilon: float, stagnation: float, policy_refresh_steps: int, policy_refresh_ms: float, stage_queue: Any, result_queue: Any, adapter_factory_path: str, alfred_backend_factory: str | None, run_nonce: int) -> None:
    game_id = str(getattr(spec, "display_name", getattr(spec, "game_id", "unknown")))
    adapter = None
    devnull = open(os.devnull, "w", encoding="utf-8")
    completion_sent = False
    try:
        logging.disable(logging.INFO)
        warnings.filterwarnings("ignore")
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            factory = _load_factory(adapter_factory_path)
            adapter = factory(spec, seed=seed, env_root=env_root, alfred_backend_factory=alfred_backend_factory)
            identity = adapter.identity()
            environment_instance_id = int(identity.instance_id.value)
            positives = negatives = episode_boundaries = resets = completed = 0
            task_successes = task_failures = task_truncations = levels_completed = 0
            grounded_action_influence = 0
            context_action_counts: dict[int, dict[int, int]] = {}
    active_strategy_uid = None
    active_strategy_actions: tuple[int, ...] = ()
    active_strategy_position = 0
    active_strategy_outcome = None
            branching_total = 0
            branching_samples = 0
            max_branching_factor = 0
            changed_transitions = 0
            episode_ordinal = 1
            policy = initial_policy
            policy_refreshes = 0
            rng = Random(seed)
            refresh_steps = max(1, int(policy_refresh_steps))
            refresh_seconds = max(0.001, float(policy_refresh_ms) / 1000.0)
            next_refresh_time = __import__("time").monotonic() + refresh_seconds

            for index in range(int(steps)):
                actions = tuple(sorted(set(int(v) for v in adapter.available_actions())))
                branching_total += len(actions)
                branching_samples += 1
                max_branching_factor = max(max_branching_factor, len(actions))
                if not actions:
                    adapter.reset()
                    episode_ordinal += 1
                    resets += 1
                    actions = tuple(sorted(set(int(v) for v in adapter.available_actions())))
                    if not actions:
                        continue
                now = __import__("time").monotonic()
                if index % refresh_steps == 0 or now >= next_refresh_time:
                    newest = None
                    while True:
                        try:
                            newest = policy_updates.get_nowait()
                        except queue.Empty:
                            break
                    if newest is not None and int(newest.generation) > int(policy.generation):
                        policy = newest
                        policy_refreshes += 1
                    next_refresh_time = now + refresh_seconds

                before = adapter.observe()
                before_signature = int(adapter.encode_observation(before))
                semantic_observer = getattr(adapter, "semantic_observation", None)
                semantic_before = tuple(semantic_observer(before)) if callable(semantic_observer) else ()
                learned_scores = policy.learned_scores(environment_instance_id, actions, environment_type=identity.environment_type, context_signature=before_signature)
                grounded_scores = policy.grounded_scores(
                    actions,
                    environment_type=identity.environment_type,
                    context_signature=before_signature,
                )
                action_schema_id = int(adapter.action_schema().schema_id)
                base_preference = min(actions, key=lambda value: (-float(learned_scores.get(int(value), 0.0)), int(value)))
                enriched_scores = {
                    int(value): float(learned_scores.get(int(value), 0.0)) + float(grounded_scores.get(int(value), 0.0))
                    for value in actions
                }
                grounded_preference = min(actions, key=lambda value: (-float(enriched_scores.get(int(value), 0.0)), int(value)))
                grounded_action_influence += int(base_preference != grounded_preference)
                context_counts = context_action_counts.setdefault(before_signature, {})
                strategy_rows = policy.strategies(environment_instance_id)
                if active_strategy_uid is not None:
                    current = next((row for row in strategy_rows if row.strategy_uid == active_strategy_uid), None)
                    if current is None or active_strategy_position >= len(active_strategy_actions) or int(active_strategy_actions[active_strategy_position]) not in actions:
                        alternatives = tuple(row for row in strategy_rows if row.target_outcome_uid == active_strategy_outcome and row.strategy_uid != active_strategy_uid and row.native_actions and int(row.native_actions[0]) in actions)
                        current = alternatives[0] if alternatives else None
                        if current is None:
                            active_strategy_uid = None
                            active_strategy_actions = ()
                            active_strategy_position = 0
                            active_strategy_outcome = None
                        else:
                            active_strategy_uid = current.strategy_uid
                            active_strategy_actions = current.native_actions
                            active_strategy_position = 0
                if active_strategy_uid is None and strategy_rows:
                    candidates = tuple(row for row in strategy_rows if row.native_actions and int(row.native_actions[0]) in actions)
                    if candidates:
                        selected = min(candidates, key=lambda row: (-row.reliability, -row.primary_valence, float("inf") if row.expected_cost is None else row.expected_cost, row.strategy_uid))
                        active_strategy_uid = selected.strategy_uid
                        active_strategy_actions = selected.native_actions
                        active_strategy_position = 0
                        active_strategy_outcome = selected.target_outcome_uid
                planned_action = None
                if active_strategy_uid is not None and active_strategy_position < len(active_strategy_actions):
                    candidate_action = int(active_strategy_actions[active_strategy_position])
                    if candidate_action in actions:
                        planned_action = candidate_action
                action = planned_action if planned_action is not None else choose_action(
                    policy,
                    actions,
                    rng=rng,
                    epsilon=float(epsilon),
                    learned_scores=enriched_scores,
                    target_environment_id=environment_instance_id,
                    action_schema_id=action_schema_id,
                    environment_type=identity.environment_type,
                    context_action_counts=context_counts,
                    stagnation=float(stagnation),
                )
                context_counts[int(action)] = int(context_counts.get(int(action), 0)) + 1
                if planned_action is not None:
                    active_strategy_position += 1
                    if active_strategy_position >= len(active_strategy_actions):
                        active_strategy_uid = None
                        active_strategy_actions = ()
                        active_strategy_position = 0
                        active_strategy_outcome = None
                semantic_action_fn = getattr(adapter, "semantic_action", None)
                semantic_delta_fn = getattr(adapter, "semantic_delta", None)
                semantic_action = tuple(semantic_action_fn(action)) if callable(semantic_action_fn) else ()
                semantic_options = tuple(
                    fact
                    for candidate in actions[:64]
                    for fact in (tuple(semantic_action_fn(candidate)) if callable(semantic_action_fn) else ())
                )
                after = adapter.step(int(action))
                semantic_after = tuple(semantic_observer(after)) if callable(semantic_observer) else ()
                semantic_delta = tuple(semantic_delta_fn(semantic_before, semantic_after)) if callable(semantic_delta_fn) else ()
                boundary = adapter.boundary_event()
                progress = adapter.task_progress()
                observation_schema_id = int(adapter.observation_schema().schema_id)
                available_after = tuple(sorted(set(int(v) for v in adapter.available_actions())))
                stage_queue.put(
                    EncodedTransition(
                        actor_id=actor_id,
                        producer_sequence=index + 1,
                        global_step=index,
                        environment_identity=(identity.family, identity.environment_type, identity.config, identity.instance),
                        episode_id=int(stable_u64(environment_instance_id, run_nonce, actor_id, episode_ordinal, person=b"v9-mp-episode")),
                        observation_schema_id=observation_schema_id,
                        action_schema_id=action_schema_id,
                        before_signature=before_signature,
                        action_id=int(adapter.encode_action(action)),
                        after_signature=int(adapter.encode_observation(after)),
                        available_actions_after=len(available_after),
                        available_action_set_signature=_action_set_signature(action_schema_id, available_after),
                        primary_valence=int(boundary.primary_valence),
                        boundary_scope=str(boundary.scope.value),
                        task_success=bool(progress.success),
                        task_failure=bool(progress.failure),
                        task_truncated=bool(progress.truncated),
                        level_index=int(progress.level_index),
                        levels_completed=int(progress.levels_completed),
                        semantic_before=semantic_before,
                        semantic_action=semantic_action,
                        semantic_options=semantic_options,
                        semantic_after=semantic_after,
                        semantic_delta=semantic_delta,
                        symbols=tuple(adapter.optional_symbol_stream()),
                        curriculum_step=getattr(spec, "curriculum_step", None),
                        game_scenario=str(getattr(spec, "game_id", identity.environment_type)),
                        symbols_only=str(getattr(spec, "condition", "") or "").upper() == "C1",
                    )
                )
                completed += 1
                changed_transitions += int(before_signature != int(adapter.encode_observation(after)))
                positives += int(boundary.primary_valence > 0)
                negatives += int(boundary.primary_valence < 0)
                episode_boundaries += int(not boundary.continuation)
                task_successes += int(progress.success)
                task_failures += int(progress.failure)
                task_truncations += int(progress.truncated)
                levels_completed = max(levels_completed, int(progress.levels_completed))
                if not boundary.continuation:
                    adapter.reset()
                    episode_ordinal += 1
                    resets += 1
            if completed and adapter.boundary_event().continuation:
                task_truncations += 1
                episode_boundaries += 1

            # ActorDone is an end-of-stream marker. Ensure every transition this
            # actor produced has left its feeder before publishing completion.
            _flush_child_queue(stage_queue)
            result_queue.put(ActorDone(
                actor_id, game_id, completed, positives, negatives, episode_boundaries, resets,
                policy_refreshes, task_successes, task_failures, task_truncations, levels_completed,
                grounded_action_influence, len(context_action_counts),
                sum(len(counts) for counts in context_action_counts.values()),
                float(branching_total) / max(1, branching_samples), max_branching_factor,
                changed_transitions,
            ))
            completion_sent = True
    except BaseException as exc:
        try:
            _flush_child_queue(stage_queue)
        except BaseException:
            pass
        try:
            result_queue.put(ActorError(actor_id, game_id, repr(exc), traceback.format_exc()))
            completion_sent = True
        except BaseException:
            pass
        raise SystemExit(1)
    finally:
        if completion_sent:
            _flush_child_queue(result_queue)
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        close()
                except BaseException:
                    pass
        devnull.close()


def stage_worker_main(stage_queue: Any, shard_queues: tuple[Any, ...]) -> None:
    while True:
        item = stage_queue.get()
        if isinstance(item, WorkerStop):
            return
        if not isinstance(item, EncodedTransition):
            continue
        shard = int(stable_u64(item.environment_identity[1], item.actor_id, item.producer_sequence, person=b"v9-stage-route") % len(shard_queues))
        shard_queues[shard].put(item)


def shard_worker_main(shard_id: int, shard_queue: Any, publication_queue: Any) -> None:
    sequence = 0
    while True:
        item = shard_queue.get()
        if isinstance(item, WorkerStop):
            publication_queue.put(("shard_done", int(shard_id), int(sequence)))
            return
        if not isinstance(item, EncodedTransition):
            continue
        sequence += 1
        publication_queue.put(("transition", int(shard_id), int(sequence), item))


class ProcessTopology:
    def __init__(self, *, actors: int, stage_workers: int, shards: int, queue_capacity: int, start_method: str | None = None) -> None:
        method = resolve_start_method(start_method)
        self.ctx = mp.get_context(method)
        self.actor_ctx = self.ctx
        self.worker_start_method = str(method)
        self.actor_start_method = str(method)
        self.actors = int(actors)
        self.stage_workers = int(stage_workers)
        self.shards = int(shards)
        self.stage_queue = self.ctx.Queue(maxsize=int(queue_capacity))
        self.publication_queue = self.ctx.Queue(maxsize=int(queue_capacity))
        self.result_queue = self.ctx.Queue(maxsize=max(64, int(actors) * 2))
        self.shard_queues = tuple(self.ctx.Queue(maxsize=int(queue_capacity)) for _ in range(self.shards))
        self.policy_updates = tuple(self.ctx.Queue(maxsize=1) for _ in range(self.actors))
        self.stage_processes: list[Any] = []
        self.shard_processes: list[Any] = []
        self.actor_processes: list[Any] = []
        self._closed = False

    def start_workers(self) -> None:
        for shard_id, shard_queue in enumerate(self.shard_queues):
            process = self.ctx.Process(target=shard_worker_main, args=(shard_id, shard_queue, self.publication_queue), name=f"v9-shard-{shard_id}")
            process.start()
            self.shard_processes.append(process)
        for index in range(self.stage_workers):
            process = self.ctx.Process(target=stage_worker_main, args=(self.stage_queue, self.shard_queues), name=f"v9-stage-{index}")
            process.start()
            self.stage_processes.append(process)

    def start_actor(self, *, index: int, spec: Any, actor_id: int, steps: int, seed: int, env_root: str | None, adapter_factory_path: str, alfred_backend_factory: str | None, run_nonce: int, initial_policy: ActorPolicySnapshot, epsilon: float, stagnation: float, policy_refresh_steps: int, policy_refresh_ms: float) -> None:
        process = self.actor_ctx.Process(
            target=actor_process_main,
            kwargs={
                "spec": spec,
                "actor_id": actor_id,
                "steps": steps,
                "seed": seed,
                "env_root": env_root,
                "initial_policy": initial_policy,
                "policy_updates": self.policy_updates[index],
                "epsilon": float(epsilon),
                "stagnation": float(stagnation),
                "policy_refresh_steps": int(policy_refresh_steps),
                "policy_refresh_ms": float(policy_refresh_ms),
                "stage_queue": self.stage_queue,
                "result_queue": self.result_queue,
                "adapter_factory_path": adapter_factory_path,
                "alfred_backend_factory": alfred_backend_factory,
                "run_nonce": int(run_nonce),
            },
            name=f"v9-actor-{actor_id}",
        )
        process.start()
        self.actor_processes.append(process)

    def publish_policy_snapshot(self, slots: tuple[int, ...], snapshot: ActorPolicySnapshot) -> None:
        for slot in slots:
            target = self.policy_updates[int(slot)]
            try:
                target.put_nowait(snapshot)
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    pass
                try:
                    target.put_nowait(snapshot)
                except queue.Full:
                    pass

    def signal_stage_stop(self) -> None:
        for _ in self.stage_processes:
            self.stage_queue.put(WorkerStop())

    @staticmethod
    def _join_checked(processes: tuple[Any, ...] | list[Any], *, role: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout)
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        stragglers = [process for process in processes if process.is_alive()]
        for process in stragglers:
            process.terminate()
        for process in stragglers:
            process.join(timeout=2.0)
        failed = [process for process in processes if process.exitcode not in (0, -15)]
        if failed:
            details = ", ".join(f"{process.name}:{process.exitcode}" for process in failed)
            raise RuntimeError(f"{role} worker shutdown failed: {details}")

    def join_actor_workers(self) -> None:
        self._join_checked(self.actor_processes, role="actor")

    def join_stage_workers(self) -> None:
        self._join_checked(self.stage_processes, role="stage", timeout=30.0)

    def stop_shard_workers(self) -> None:
        for shard_queue in self.shard_queues:
            shard_queue.put(WorkerStop())

    def join_shard_workers(self) -> None:
        self._join_checked(self.shard_processes, role="shard", timeout=30.0)

    def terminate(self) -> None:
        for process in (*self.actor_processes, *self.stage_processes, *self.shard_processes):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    def close(self, *, drain: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        queues = (
            self.stage_queue,
            self.publication_queue,
            self.result_queue,
            *self.shard_queues,
            *self.policy_updates,
        )
        for queue_obj in queues:
            _close_queue(queue_obj, drain=drain)
        for process in (*self.actor_processes, *self.stage_processes, *self.shard_processes):
            _close_process(process)

    @property
    def process_count(self) -> int:
        return len(self.actor_processes) + len(self.stage_processes) + len(self.shard_processes)
