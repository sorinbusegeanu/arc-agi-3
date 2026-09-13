from __future__ import annotations

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
    if not drain:
        try:
            queue_obj.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
    try:
        queue_obj.close()
    except (AttributeError, OSError, ValueError):
        pass
    if drain:
        try:
            queue_obj.join_thread()
        except (AttributeError, AssertionError, OSError, ValueError):
            pass


def _publish_actor_terminal(stage_queue: Any, result_queue: Any, terminal: ActorDone | ActorError) -> None:
    """Flush all actor transitions before publishing a terminal record, then flush it too."""
    _close_queue(stage_queue, drain=True)
    result_queue.put(terminal)
    _close_queue(result_queue, drain=True)


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


def actor_process_main(*, spec: Any, actor_id: int, steps: int, seed: int, env_root: str | None, initial_policy: ActorPolicySnapshot, policy_updates: Any, epsilon: float, policy_refresh_steps: int, policy_refresh_ms: float, stage_queue: Any, result_queue: Any, adapter_factory_path: str, alfred_backend_factory: str | None, run_nonce: int) -> None:
    game_id = str(getattr(spec, "display_name", getattr(spec, "game_id", "unknown")))
    adapter = None
    devnull = open(os.devnull, "w", encoding="utf-8")
    try:
        logging.disable(logging.INFO)
        warnings.filterwarnings("ignore")
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            factory = _load_factory(adapter_factory_path)
            adapter = factory(spec, seed=seed, env_root=env_root, alfred_backend_factory=alfred_backend_factory)
            identity = adapter.identity()
            environment_instance_id = int(identity.instance_id.value)
            positives = negatives = episode_boundaries = resets = completed = 0
            episode_ordinal = 1
            policy = initial_policy
            policy_refreshes = 0
            rng = Random(seed)
            refresh_steps = max(1, int(policy_refresh_steps))
            refresh_seconds = max(0.001, float(policy_refresh_ms) / 1000.0)
            next_refresh_time = __import__("time").monotonic() + refresh_seconds

            for index in range(int(steps)):
                actions = tuple(sorted(set(int(v) for v in adapter.available_actions())))
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
                learned_scores = policy.learned_scores(environment_instance_id, actions, environment_type=identity.environment_type, context_signature=before_signature)
                action = choose_action(policy, actions, rng=rng, epsilon=float(epsilon), learned_scores=learned_scores, target_environment_id=environment_instance_id)
                after = adapter.step(int(action))
                boundary = adapter.boundary_event()
                observation_schema_id = int(adapter.observation_schema().schema_id)
                stage_queue.put(
                    EncodedTransition(
                        actor_id=actor_id,
                        producer_sequence=index + 1,
                        global_step=index,
                        environment_identity=(identity.family, identity.environment_type, identity.config, identity.instance),
                        episode_id=int(stable_u64(environment_instance_id, run_nonce, actor_id, episode_ordinal, person=b"v9-mp-episode")),
                        observation_schema_id=observation_schema_id,
                        before_signature=before_signature,
                        action_id=int(adapter.encode_action(action)),
                        after_signature=int(adapter.encode_observation(after)),
                        available_actions_after=len(tuple(adapter.available_actions())),
                        primary_valence=int(boundary.primary_valence),
                        symbols=tuple(adapter.optional_symbol_stream()),
                        curriculum_step=getattr(spec, "curriculum_step", None),
                        game_scenario=str(getattr(spec, "game_id", identity.environment_type)),
                        symbols_only=str(getattr(spec, "condition", "") or "").upper() == "C1",
                    )
                )
                completed += 1
                positives += int(boundary.primary_valence > 0)
                negatives += int(boundary.primary_valence < 0)
                episode_boundaries += int(not boundary.continuation)
                if not boundary.continuation:
                    adapter.reset()
                    episode_ordinal += 1
                    resets += 1
            _publish_actor_terminal(
                stage_queue,
                result_queue,
                ActorDone(actor_id, game_id, completed, positives, negatives, episode_boundaries, resets, policy_refreshes),
            )
    except BaseException as exc:
        try:
            _publish_actor_terminal(stage_queue, result_queue, ActorError(actor_id, game_id, repr(exc), traceback.format_exc()))
        except BaseException:
            pass
        raise SystemExit(1)
    finally:
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

    def start_actor(self, *, index: int, spec: Any, actor_id: int, steps: int, seed: int, env_root: str | None, adapter_factory_path: str, alfred_backend_factory: str | None, run_nonce: int, initial_policy: ActorPolicySnapshot, epsilon: float, policy_refresh_steps: int, policy_refresh_ms: float) -> None:
        process = self.actor_ctx.Process(target=actor_process_main, kwargs={"spec": spec, "actor_id": actor_id, "steps": steps, "seed": seed, "env_root": env_root, "initial_policy": initial_policy, "policy_updates": self.policy_updates[index], "epsilon": float(epsilon), "policy_refresh_steps": int(policy_refresh_steps), "policy_refresh_ms": float(policy_refresh_ms), "stage_queue": self.stage_queue, "result_queue": self.result_queue, "adapter_factory_path": adapter_factory_path, "alfred_backend_factory": alfred_backend_factory, "run_nonce": int(run_nonce)}, name=f"v9-actor-{actor_id}")
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

    def join_actor_workers(self) -> None:
        for process in self.actor_processes:
            process.join(timeout=0)
            if process.is_alive():
                raise RuntimeError(f"actor process {process.name} has not finished flushing")
            if process.exitcode != 0:
                raise RuntimeError(f"actor process {process.name} exited with code {process.exitcode}")

    def join_stage_workers(self) -> None:
        for process in self.stage_processes:
            process.join(timeout=30)

    def stop_shard_workers(self) -> None:
        for shard_queue in self.shard_queues:
            shard_queue.put(WorkerStop())

    def join_shard_workers(self) -> None:
        for process in self.shard_processes:
            process.join(timeout=30)

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
