from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass
from random import Random
from typing import Iterable

from v8.model import EventId, ExperienceEvent, PipelineEvent, encode_pipeline, stable_u64
from v8.publication import LiveReadView, ShardReadDescriptor
from v8.ring import SharedRingBuffer


@dataclass(frozen=True, slots=True)
class ActorJob:
    actor_id: int
    game_id: str
    steps: int
    seed: int
    env_root: str | None = None
    epsilon: float = 0.10


@dataclass(frozen=True, slots=True)
class ActorResult:
    actor_id: int
    game_id: str
    steps: int
    wins: int
    failures: int
    levels_completed: int
    resets: int


def _polarity(value: str) -> int:
    if value == "positive":
        return 1
    if value == "negative":
        return -1
    return 0


def _local_significance(changed_cells: int, future_delta: float) -> float:
    structural = min(1.0, max(0, int(changed_cells)) / 32.0)
    option = math.tanh(abs(float(future_delta)))
    return 0.55 * structural + 0.45 * option


def actor_worker(
    *,
    job: ActorJob,
    experience_ring_args: dict[str, object],
    read_descriptors: tuple[ShardReadDescriptor, ...],
    watermark: mp.sharedctypes.Synchronized,
    stop_event: mp.synchronize.Event,
    result_queue: mp.Queue,
) -> None:
    from v7.environment.arc_adapter import ArcGridEnvironment
    from v7.environment.encoding import (
        carrier_signature,
        changed_cell_count,
        structural_grid_signature,
        transformation_family_signature,
        transition_signature,
    )

    ring = SharedRingBuffer(**experience_ring_args)
    view = LiveReadView(read_descriptors)
    rng = Random(job.seed)
    env = ArcGridEnvironment(game_id=job.game_id, seed=job.seed, env_root=job.env_root)
    sequence = 0
    rolling_trajectory = stable_u64(job.actor_id, job.seed, person=b"v8-traj-seed")
    local_overlay: dict[tuple[int, int], tuple[int, float]] = {}
    wins = failures = 0
    last_levels = int(env.last_levels_completed)
    try:
        for _ in range(int(job.steps)):
            if stop_event.is_set():
                break
            before = env.observe()
            before_actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            if not before_actions:
                env.reset()
                continue
            context = int(structural_grid_signature(before))
            scores = view.score_actions(context, before_actions)
            combined = []
            for score in scores:
                local_support, local_score = local_overlay.get((context, score.action_id), (0, 0.0))
                support = score.support_count + local_support
                if support <= 0:
                    value = 0.0
                else:
                    weighted = score.score * score.support_count + local_score * local_support
                    value = weighted / support
                combined.append((score.action_id, support, value))
            unseen = [action for action, support, _score in combined if support == 0]
            if unseen:
                action = unseen[rng.randrange(len(unseen))]
            elif rng.random() < float(job.epsilon):
                action = before_actions[rng.randrange(len(before_actions))]
            else:
                action = min(combined, key=lambda row: (-row[2], -row[1], row[0]))[0]

            after = env.step(action)
            after_actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            sequence += 1
            with watermark.get_lock():
                watermark.value += 1
                current_watermark = int(watermark.value)
            outcome = int(transition_signature(before, after))
            family = int(transformation_family_signature(before, after))
            carrier = int(carrier_signature(before, after) or 0)
            changed = int(changed_cell_count(before, after))
            future_delta = float(len(after_actions) - len(before_actions))
            rolling_trajectory = stable_u64(
                rolling_trajectory, context, action, outcome, person=b"v8-trajectory"
            )
            event = ExperienceEvent(
                event_id=EventId.from_producer(job.actor_id, sequence),
                watermark=current_watermark,
                producer_id=job.actor_id,
                producer_sequence=sequence,
                source_game_hash=stable_u64(job.game_id, person=b"v8-game"),
                global_step=current_watermark,
                context_signature=context,
                action_id=action,
                outcome_signature=outcome,
                family_signature=family,
                carrier_signature=carrier,
                future_option_delta=future_delta,
                changed_cells=changed,
                terminal_polarity=_polarity(env.last_outcome_polarity),
                trajectory_signature=rolling_trajectory,
            )
            packet = encode_pipeline(PipelineEvent(event))
            while not stop_event.is_set():
                if ring.put(packet, timeout=0.25):
                    break

            significance = _local_significance(changed, future_delta)
            old_support, old_score = local_overlay.get((context, action), (0, 0.0))
            new_support = old_support + 1
            local_overlay[(context, action)] = (
                new_support,
                (old_score * old_support + significance) / new_support,
            )

            if env.last_outcome_polarity == "positive":
                wins += 1
            elif env.last_outcome_polarity == "negative":
                failures += 1
            current_levels = int(env.last_levels_completed)
            if current_levels < last_levels or env.last_step_was_reset_boundary:
                rolling_trajectory = stable_u64(
                    job.actor_id, sequence, current_watermark, person=b"v8-traj-reset"
                )
            last_levels = current_levels
        result_queue.put(
            ActorResult(
                job.actor_id,
                job.game_id,
                sequence,
                wins,
                failures,
                int(env.last_levels_completed),
                int(env.reset_count),
            )
        )
    finally:
        view.close()
        ring.close()


def run_actor_jobs(runtime, jobs: Iterable[ActorJob], *, timeout: float | None = None) -> tuple[ActorResult, ...]:
    runtime.start()
    jobs = tuple(jobs)
    if not jobs:
        return ()
    results: mp.Queue = mp.Queue()
    processes = [
        mp.Process(
            target=actor_worker,
            kwargs={
                "job": job,
                "experience_ring_args": runtime._stage_rings[0].attachment_args(),
                "read_descriptors": runtime.shard_descriptors,
                "watermark": runtime._watermark,
                "stop_event": runtime._stop,
                "result_queue": results,
            },
            name=f"v8-actor-{job.actor_id:03d}-{job.game_id}",
            daemon=True,
        )
        for job in jobs
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=timeout)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            raise TimeoutError(f"actor timed out: {process.name}")
        if process.exitcode != 0:
            raise RuntimeError(f"actor failed: {process.name} exit={process.exitcode}")
    rows = [results.get(timeout=2.0) for _ in jobs]
    return tuple(sorted(rows, key=lambda row: row.actor_id))
