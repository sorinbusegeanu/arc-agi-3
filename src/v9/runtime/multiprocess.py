from __future__ import annotations

import importlib
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Any

from v9.memory.identity import stable_u64


@dataclass(frozen=True, slots=True)
class ActionRequest:
    actor_id: int
    request_id: int
    environment_instance_id: int
    actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EncodedTransition:
    actor_id: int
    producer_sequence: int
    global_step: int
    environment_identity: tuple[str, str, str, str]
    episode_ordinal: int
    observation_schema_id: int
    before_signature: int
    action_id: int
    after_signature: int
    available_actions_after: int
    primary_valence: int
    symbols: tuple[object, ...]
    curriculum_step: str | None
    game_scenario: str


@dataclass(frozen=True, slots=True)
class ActorDone:
    actor_id: int
    game_id: str
    steps: int
    positive_boundaries: int
    negative_boundaries: int
    resets: int


@dataclass(frozen=True, slots=True)
class WorkerStop:
    reason: str = "stop"


def _load_factory(path: str):
    module_name, sep, attr = path.partition(":")
    if not sep:
        raise ValueError("factory path must use module:function")
    return getattr(importlib.import_module(module_name), attr)


def actor_process_main(
    *,
    spec: Any,
    actor_id: int,
    steps: int,
    seed: int,
    env_root: str | None,
    action_requests: Any,
    action_responses: Any,
    stage_queue: Any,
    result_queue: Any,
    adapter_factory_path: str,
    alfred_backend_factory: str | None,
) -> None:
    factory = _load_factory(adapter_factory_path)
    adapter = factory(spec, seed=seed, env_root=env_root, alfred_backend_factory=alfred_backend_factory)
    identity = adapter.identity()
    environment_instance_id = int(identity.instance_id.value)
    positives = negatives = resets = completed = 0
    episode_ordinal = 1
    try:
        for index in range(int(steps)):
            actions = tuple(sorted(set(int(v) for v in adapter.available_actions())))
            if not actions:
                adapter.reset()
                episode_ordinal += 1
                resets += 1
                actions = tuple(sorted(set(int(v) for v in adapter.available_actions())))
                if not actions:
                    continue
            request_id = index + 1
            action_requests.put(ActionRequest(actor_id, request_id, environment_instance_id, actions))
            response_request_id, action = action_responses.get()
            if int(response_request_id) != request_id:
                raise RuntimeError("actor received out-of-order action response")
            before = adapter.observe()
            after = adapter.step(int(action))
            boundary = adapter.boundary_event()
            observation_schema_id = int(adapter.observation_schema().schema_id)
            stage_queue.put(
                EncodedTransition(
                    actor_id=actor_id,
                    producer_sequence=index + 1,
                    global_step=index,
                    environment_identity=(identity.family, identity.environment_type, identity.config, identity.instance),
                    episode_ordinal=episode_ordinal,
                    observation_schema_id=observation_schema_id,
                    before_signature=int(adapter.encode_observation(before)),
                    action_id=int(adapter.encode_action(action)),
                    after_signature=int(adapter.encode_observation(after)),
                    available_actions_after=len(tuple(adapter.available_actions())),
                    primary_valence=int(boundary.primary_valence),
                    symbols=tuple(adapter.optional_symbol_stream()),
                    curriculum_step=getattr(spec, "curriculum_step", None),
                    game_scenario=str(getattr(spec, "game_id", identity.environment_type)),
                )
            )
            completed += 1
            positives += int(boundary.primary_valence > 0)
            negatives += int(boundary.primary_valence < 0)
            if not boundary.continuation:
                adapter.reset()
                episode_ordinal += 1
                resets += 1
        result_queue.put(ActorDone(actor_id, str(getattr(spec, "display_name", identity.environment_type)), completed, positives, negatives, resets))
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def stage_worker_main(stage_queue: Any, shard_queues: tuple[Any, ...]) -> None:
    while True:
        item = stage_queue.get()
        if isinstance(item, WorkerStop):
            return
        if not isinstance(item, EncodedTransition):
            continue
        shard = int(stable_u64(item.environment_instance_id if hasattr(item, "environment_instance_id") else item.environment_identity[1], item.actor_id, item.producer_sequence, person=b"v9-stage-route") % len(shard_queues))
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
        method = start_method or ("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        self.ctx = mp.get_context(method)
        self.actors = int(actors)
        self.stage_workers = int(stage_workers)
        self.shards = int(shards)
        self.stage_queue = self.ctx.Queue(maxsize=int(queue_capacity))
        self.action_requests = self.ctx.Queue(maxsize=max(64, int(actors) * 4))
        self.publication_queue = self.ctx.Queue(maxsize=int(queue_capacity))
        self.result_queue = self.ctx.Queue(maxsize=max(64, int(actors) * 2))
        self.shard_queues = tuple(self.ctx.Queue(maxsize=int(queue_capacity)) for _ in range(self.shards))
        self.action_responses = tuple(self.ctx.Queue(maxsize=2) for _ in range(self.actors))
        self.stage_processes: list[Any] = []
        self.shard_processes: list[Any] = []
        self.actor_processes: list[Any] = []

    def start_workers(self) -> None:
        for shard_id, shard_queue in enumerate(self.shard_queues):
            process = self.ctx.Process(target=shard_worker_main, args=(shard_id, shard_queue, self.publication_queue), name=f"v9-shard-{shard_id}")
            process.start()
            self.shard_processes.append(process)
        for index in range(self.stage_workers):
            process = self.ctx.Process(target=stage_worker_main, args=(self.stage_queue, self.shard_queues), name=f"v9-stage-{index}")
            process.start()
            self.stage_processes.append(process)

    def start_actor(self, *, index: int, spec: Any, actor_id: int, steps: int, seed: int, env_root: str | None, adapter_factory_path: str, alfred_backend_factory: str | None) -> None:
        process = self.ctx.Process(
            target=actor_process_main,
            kwargs={
                "spec": spec,
                "actor_id": actor_id,
                "steps": steps,
                "seed": seed,
                "env_root": env_root,
                "action_requests": self.action_requests,
                "action_responses": self.action_responses[index],
                "stage_queue": self.stage_queue,
                "result_queue": self.result_queue,
                "adapter_factory_path": adapter_factory_path,
                "alfred_backend_factory": alfred_backend_factory,
            },
            name=f"v9-actor-{actor_id}",
        )
        process.start()
        self.actor_processes.append(process)

    def stop_pipeline(self) -> None:
        for _ in self.stage_processes:
            self.stage_queue.put(WorkerStop())
        for process in self.stage_processes:
            process.join(timeout=30)
        for shard_queue in self.shard_queues:
            shard_queue.put(WorkerStop())
        for process in self.shard_processes:
            process.join(timeout=30)

    def terminate(self) -> None:
        for process in (*self.actor_processes, *self.stage_processes, *self.shard_processes):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    @property
    def process_count(self) -> int:
        return len(self.actor_processes) + len(self.stage_processes) + len(self.shard_processes)
