from __future__ import annotations

import math
import multiprocessing as mp
import queue
import time
from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Callable, Iterable

from v8.model import (
    EventId,
    ExperienceEvent,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    PipelineEvent,
    encode_pipeline,
    stable_u64,
)
from v8.publication import LiveReadView, ShardReadDescriptor
from v8.ring import SharedRingBuffer


_MAX_PARENT_MESSAGES_PER_CYCLE = 256


@dataclass(frozen=True, slots=True)
class ActorJob:
    actor_id: int
    game_id: str
    steps: int
    seed: int
    env_root: str | None = None
    epsilon: float = 0.10
    replanning_probe_rate: float = 0.02
    max_probe_records: int = 256


@dataclass(frozen=True, slots=True)
class ActorProgress:
    actor_id: int
    game_id: str
    steps: int
    wins: int
    failures: int
    levels_completed: int
    replans: int = 0
    planned_steps: int = 0


@dataclass(frozen=True, slots=True)
class StrategyRunStat:
    strategy_uid: MemoryUid
    attempts: int
    successes: int
    cost: float


@dataclass(frozen=True, slots=True)
class PreferenceProbeResult:
    outcome_a: MemoryUid
    outcome_b: MemoryUid
    context_bucket: int
    chosen_outcome: MemoryUid
    preference_influenced: bool


@dataclass(frozen=True, slots=True)
class ReplanningTrialResult:
    primary_strategy_uid: MemoryUid
    alternative_strategy_uid: MemoryUid
    outcome_uid: MemoryUid
    recovery_succeeded: bool


@dataclass(frozen=True, slots=True)
class ActorLearningBatch:
    actor_id: int
    game_id: str
    strategy_stats: tuple[StrategyRunStat, ...] = ()
    preference_probes: tuple[PreferenceProbeResult, ...] = ()
    replanning_trials: tuple[ReplanningTrialResult, ...] = ()
    replans: int = 0


@dataclass(frozen=True, slots=True)
class ActorResult:
    actor_id: int
    game_id: str
    steps: int
    wins: int
    failures: int
    levels_completed: int
    resets: int
    replans: int = 0
    planned_steps: int = 0
    strategy_stats: tuple[StrategyRunStat, ...] = ()
    preference_probes: tuple[PreferenceProbeResult, ...] = ()
    replanning_trials: tuple[ReplanningTrialResult, ...] = ()


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


def _trajectory_step_cost(
    *,
    context: int,
    after_context: int,
    changed_cells: int,
    negative_outcome: bool,
    recent_contexts: tuple[int, ...],
) -> float:
    cost = 1.0
    if int(changed_cells) <= 0:
        cost += 1.0
    if int(after_context) == int(context):
        cost += 0.5
    if int(after_context) in set(int(value) for value in recent_contexts):
        cost += 1.0
    if bool(negative_outcome):
        cost += 1.0
    return float(cost)


def _bucket(value: float, threshold: float = 1e-9) -> int:
    return 1 if value > threshold else -1 if value < -threshold else 0


def _changed_bucket(changed_cells: int) -> int:
    value = max(0, int(changed_cells))
    if value == 0:
        return 0
    if value <= 4:
        return 1
    if value <= 16:
        return 2
    if value <= 64:
        return 3
    return 4


def _outcome_bucket(changed_cells: int, terminal_polarity: int) -> int:
    polarity = 1 if int(terminal_polarity) > 0 else -1 if int(terminal_polarity) < 0 else 0
    return _changed_bucket(changed_cells) * 3 + (polarity + 1)


def _observed_outcome_uids(
    *,
    outcome_signature: int,
    family_signature: int,
    future_delta: float,
    changed_cells: int,
    terminal_polarity: int = 0,
) -> tuple[MemoryUid, MemoryUid]:
    future = _bucket(future_delta)
    outcome_bucket = _outcome_bucket(changed_cells, terminal_polarity)
    variant = stable_u64(
        outcome_signature, family_signature, person=b"v8-outcome-variant"
    ) & 0xF
    return (
        MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (future, outcome_bucket, variant)),
        MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (future, outcome_bucket)),
    )


def _stats_tuple(values: dict[MemoryUid, list[float]]) -> tuple[StrategyRunStat, ...]:
    return tuple(
        StrategyRunStat(uid, int(row[0]), int(row[1]), float(row[2]))
        for uid, row in sorted(values.items())
        if row[0] > 0
    )


def _publish_progress(
    progress_queue: mp.Queue | None,
    reporting_queue: mp.Queue | None = None,
    *,
    job: ActorJob,
    steps: int,
    wins: int,
    failures: int,
    levels_completed: int,
    replans: int,
    planned_steps: int,
) -> None:
    row = ActorProgress(
        job.actor_id,
        job.game_id,
        int(steps),
        int(wins),
        int(failures),
        int(levels_completed),
        int(replans),
        int(planned_steps),
    )
    for target in (progress_queue, reporting_queue):
        if target is None:
            continue
        try:
            target.put_nowait(row)
        except queue.Full:
            pass


def _publish_learning(
    progress_queue: mp.Queue | None,
    *,
    job: ActorJob,
    strategy_stats: dict[MemoryUid, list[float]],
    preference_probes: list[PreferenceProbeResult],
    replanning_trials: list[ReplanningTrialResult],
) -> None:
    if progress_queue is None:
        return
    stats = _stats_tuple(strategy_stats)
    if not stats and not preference_probes and not replanning_trials:
        return
    progress_queue.put(
        ActorLearningBatch(
            job.actor_id,
            job.game_id,
            stats,
            tuple(preference_probes),
            tuple(replanning_trials),
            len(replanning_trials),
        )
    )


def actor_worker(
    *,
    job: ActorJob,
    experience_ring_args: dict[str, object],
    read_descriptors: tuple[ShardReadDescriptor, ...],
    watermark: mp.sharedctypes.Synchronized,
    stop_event: mp.synchronize.Event,
    result_queue: mp.Queue,
    progress_queue: mp.Queue | None = None,
    reporting_queue: mp.Queue | None = None,
    actor_throttle: mp.sharedctypes.Synchronized | None = None,
    snapshot_freeze: mp.synchronize.Event | None = None,
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
    terminal_wait_seconds = max(0.0, float(env.game_wait_seconds))
    env.game_wait_seconds = 0.0
    sequence = 0
    with watermark.get_lock():
        sequence_base = int(watermark.value)
    rolling_trajectory = stable_u64(job.actor_id, job.seed, sequence_base, person=b"v8-traj-seed")
    local_overlay: dict[tuple[int, int], tuple[int, float]] = {}
    recent_contexts: deque[int] = deque(maxlen=8)
    wins = failures = levels_completed = 0
    replans = planned_steps = 0
    selected_outcome: MemoryUid | None = None
    selected_strategy: MemoryUid | None = None
    strategy_stats: dict[MemoryUid, list[float]] = {}
    pending_strategy_stats: dict[MemoryUid, list[float]] = {}
    preference_probes: list[PreferenceProbeResult] = []
    pending_preference_probes: list[PreferenceProbeResult] = []
    replanning_trials: list[ReplanningTrialResult] = []
    pending_replanning_trials: list[ReplanningTrialResult] = []
    last_levels = int(env.last_levels_completed)
    next_progress = time.monotonic() + 5.0
    try:
        for _ in range(int(job.steps)):
            if stop_event.is_set():
                break
            while snapshot_freeze is not None and snapshot_freeze.is_set() and not stop_event.is_set():
                time.sleep(0.002)
            if stop_event.is_set():
                break

            before = env.observe()
            before_actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            if not before_actions:
                env.reset()
                selected_outcome = selected_strategy = None
                recent_contexts.clear()
                local_overlay.clear()
                continue
            context = int(structural_grid_signature(before))

            plans = view.plan_candidates(context, before_actions)
            planned = plans[0] if plans else None
            explicit_replan: tuple[MemoryUid, MemoryUid, MemoryUid] | None = None
            if planned is not None:
                alternatives = [row for row in plans[1:] if row.outcome_uid != planned.outcome_uid]
                if alternatives and len(preference_probes) < int(job.max_probe_records):
                    probe = PreferenceProbeResult(
                        planned.outcome_uid,
                        alternatives[0].outcome_uid,
                        stable_u64(context, person=b"v8-context"),
                        planned.outcome_uid,
                        bool(planned.preference_influenced),
                    )
                    preference_probes.append(probe)
                    pending_preference_probes.append(probe)

                same_outcome = [
                    row
                    for row in plans[1:]
                    if row.outcome_uid == planned.outcome_uid
                    and row.strategy_uid != planned.strategy_uid
                ]
                if same_outcome and rng.random() < float(job.replanning_probe_rate):
                    alternative = same_outcome[0]
                    explicit_replan = (
                        planned.strategy_uid,
                        alternative.strategy_uid,
                        planned.outcome_uid,
                    )
                    planned = alternative
                    replans += 1

                action = int(planned.action_id)
                planned_steps += 1
                if (
                    selected_outcome is not None
                    and selected_outcome == planned.outcome_uid
                    and selected_strategy is not None
                    and selected_strategy != planned.strategy_uid
                ):
                    replans += 1
                selected_outcome = planned.outcome_uid
                selected_strategy = planned.strategy_uid
            else:
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

            prediction_distribution = view.outcome_distribution(context, action)

            if actor_throttle is not None:
                with actor_throttle.get_lock():
                    delay = float(actor_throttle.value)
                if delay > 0:
                    time.sleep(delay)

            after = env.step(action)
            after_actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            after_context = int(structural_grid_signature(after))
            outcome = int(transition_signature(before, after))
            family = int(transformation_family_signature(before, after))
            carrier = int(carrier_signature(before, after) or 0)
            changed = int(changed_cell_count(before, after))
            future_delta = float(len(after_actions) - len(before_actions))
            terminal_polarity = _polarity(env.last_outcome_polarity)
            prediction_error = (
                0.0
                if not prediction_distribution
                else max(0.0, 1.0 - float(prediction_distribution.get(outcome, 0.0)))
            )
            rolling_trajectory = stable_u64(
                rolling_trajectory, context, action, outcome, person=b"v8-trajectory"
            )

            next_sequence = sequence + 1
            producer_sequence = sequence_base + next_sequence
            accepted = False
            current_watermark = 0
            while not stop_event.is_set():
                if snapshot_freeze is not None and snapshot_freeze.is_set():
                    time.sleep(0.002)
                    continue
                with watermark.get_lock():
                    current_watermark = int(watermark.value) + 1
                    event = ExperienceEvent(
                        event_id=EventId.from_producer(job.actor_id, producer_sequence),
                        watermark=current_watermark,
                        producer_id=job.actor_id,
                        producer_sequence=producer_sequence,
                        source_game_hash=stable_u64(job.game_id, person=b"v8-game"),
                        global_step=current_watermark,
                        context_signature=context,
                        action_id=action,
                        outcome_signature=outcome,
                        family_signature=family,
                        carrier_signature=carrier,
                        future_option_delta=future_delta,
                        changed_cells=changed,
                        terminal_polarity=terminal_polarity,
                        trajectory_signature=rolling_trajectory,
                        next_context_signature=after_context,
                        prediction_error=prediction_error,
                    )
                    packet = encode_pipeline(PipelineEvent(event))
                    if ring.put(packet, timeout=0.05):
                        watermark.value = current_watermark
                        accepted = True
                        break
                time.sleep(0)
            if not accepted:
                break
            sequence = next_sequence

            significance = _local_significance(changed, future_delta)
            local_value = (
                0.30 * significance
                + 0.55 * terminal_polarity
                + 0.15 * math.tanh(future_delta)
            )
            old_support, old_score = local_overlay.get((context, action), (0, 0.0))
            new_support = old_support + 1
            local_overlay[(context, action)] = (
                new_support,
                (old_score * old_support + local_value) / new_support,
            )

            prior_levels = last_levels
            if env.last_outcome_state == "WIN":
                wins += 1
            elif env.last_outcome_state == "GAME_OVER":
                failures += 1
            current_levels = int(env.last_levels_completed)
            level_advanced = current_levels > prior_levels
            if level_advanced:
                levels_completed += current_levels - prior_levels

            observed_variant, observed_coarse = _observed_outcome_uids(
                outcome_signature=outcome,
                family_signature=family,
                future_delta=future_delta,
                changed_cells=changed,
                terminal_polarity=terminal_polarity,
            )
            if planned is not None:
                success = int(
                    env.last_outcome_state == "WIN"
                    or level_advanced
                    or planned.outcome_uid in {observed_variant, observed_coarse}
                )
                step_cost = _trajectory_step_cost(
                    context=context,
                    after_context=after_context,
                    changed_cells=changed,
                    negative_outcome=env.last_outcome_state == "GAME_OVER",
                    recent_contexts=tuple(recent_contexts),
                )
                for stats_map in (strategy_stats, pending_strategy_stats):
                    stat = stats_map.setdefault(planned.strategy_uid, [0.0, 0.0, 0.0])
                    stat[0] += 1.0
                    stat[1] += float(success)
                    stat[2] += step_cost

            if explicit_replan is not None and len(replanning_trials) < int(job.max_probe_records):
                primary_uid, alternative_uid, target_outcome = explicit_replan
                trial = ReplanningTrialResult(
                    primary_uid,
                    alternative_uid,
                    target_outcome,
                    target_outcome in {observed_variant, observed_coarse},
                )
                replanning_trials.append(trial)
                pending_replanning_trials.append(trial)

            recent_contexts.append(context)
            terminal_game = env.last_outcome_state in {"WIN", "GAME_OVER"}
            reset_boundary = current_levels < last_levels or env.last_step_was_reset_boundary
            if terminal_game:
                _publish_learning(
                    progress_queue,
                    job=job,
                    strategy_stats=pending_strategy_stats,
                    preference_probes=pending_preference_probes,
                    replanning_trials=pending_replanning_trials,
                )
                pending_strategy_stats.clear()
                pending_preference_probes.clear()
                pending_replanning_trials.clear()
                local_overlay.clear()
                if terminal_wait_seconds > 0:
                    time.sleep(terminal_wait_seconds)
            if reset_boundary:
                rolling_trajectory = stable_u64(
                    job.actor_id, producer_sequence, current_watermark, person=b"v8-traj-reset"
                )
                selected_outcome = selected_strategy = None
                recent_contexts.clear()
                local_overlay.clear()
            last_levels = current_levels

            now = time.monotonic()
            if now >= next_progress:
                _publish_progress(
                    progress_queue,
                    reporting_queue,
                    job=job,
                    steps=sequence,
                    wins=wins,
                    failures=failures,
                    levels_completed=levels_completed,
                    replans=replans,
                    planned_steps=planned_steps,
                )
                next_progress = now + 5.0

        _publish_learning(
            progress_queue,
            job=job,
            strategy_stats=pending_strategy_stats,
            preference_probes=pending_preference_probes,
            replanning_trials=pending_replanning_trials,
        )
        _publish_progress(
            progress_queue,
            reporting_queue,
            job=job,
            steps=sequence,
            wins=wins,
            failures=failures,
            levels_completed=levels_completed,
            replans=replans,
            planned_steps=planned_steps,
        )
        result_queue.put(
            ActorResult(
                job.actor_id,
                job.game_id,
                sequence,
                wins,
                failures,
                levels_completed,
                int(env.reset_count),
                replans,
                planned_steps,
                _stats_tuple(strategy_stats),
                tuple(preference_probes),
                tuple(replanning_trials),
            )
        )
    finally:
        view.close()
        ring.close()


def run_actor_jobs(
    runtime,
    jobs: Iterable[ActorJob],
    *,
    timeout: float | None = None,
    progress_interval_seconds: float = 60.0,
    progress_callback: Callable[[tuple[ActorProgress, ...]], None] | None = None,
    reporting_queue: mp.Queue | None = None,
) -> tuple[ActorResult, ...]:
    runtime.start()
    jobs = tuple(jobs)
    if not jobs:
        return ()
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")

    ctx = runtime._mp_ctx
    results = ctx.Queue()
    progress = ctx.Queue(maxsize=max(16, len(jobs) * 8))
    processes = [
        ctx.Process(
            target=actor_worker,
            kwargs={
                "job": job,
                "experience_ring_args": runtime._stage_rings[0].attachment_args(),
                "read_descriptors": runtime.shard_descriptors,
                "watermark": runtime._watermark,
                "stop_event": runtime._stop,
                "result_queue": results,
                "progress_queue": progress,
                "reporting_queue": reporting_queue,
                "actor_throttle": runtime._actor_throttle,
                "snapshot_freeze": runtime._snapshot_freeze,
            },
            name=f"v8-actor-{job.actor_id:03d}-{job.game_id}",
            daemon=True,
        )
        for job in jobs
    ]
    latest = {
        job.actor_id: ActorProgress(job.actor_id, job.game_id, 0, 0, 0, 0)
        for job in jobs
    }
    result_by_actor: dict[int, ActorResult] = {}
    started = time.monotonic()
    deadline = None if timeout is None else started + float(timeout)
    next_report = started + float(progress_interval_seconds)

    try:
        for process in processes:
            process.start()

        while True:
            for _ in range(_MAX_PARENT_MESSAGES_PER_CYCLE):
                try:
                    row = progress.get_nowait()
                except queue.Empty:
                    break
                if isinstance(row, ActorProgress):
                    latest[row.actor_id] = row
                elif isinstance(row, ActorLearningBatch):
                    runtime.record_actor_results((row,))

            while True:
                try:
                    row = results.get_nowait()
                except queue.Empty:
                    break
                if isinstance(row, ActorResult):
                    result_by_actor[row.actor_id] = row
                    latest[row.actor_id] = ActorProgress(
                        row.actor_id,
                        row.game_id,
                        row.steps,
                        row.wins,
                        row.failures,
                        row.levels_completed,
                        row.replans,
                        row.planned_steps,
                    )

            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                detail = ", ".join(f"{process.name}={process.exitcode}" for process in failed)
                raise RuntimeError(f"actor failed: {detail}")

            now = time.monotonic()
            if progress_callback is not None and now >= next_report:
                progress_callback(tuple(latest[key] for key in sorted(latest)))
                while next_report <= now:
                    next_report += float(progress_interval_seconds)

            if all(not process.is_alive() for process in processes):
                break
            if deadline is not None and now >= deadline:
                raise TimeoutError("actor jobs timed out")
            time.sleep(0.05)

        for process in processes:
            process.join(timeout=2.0)

        while True:
            try:
                row = progress.get_nowait()
            except queue.Empty:
                break
            if isinstance(row, ActorProgress):
                latest[row.actor_id] = row
            elif isinstance(row, ActorLearningBatch):
                runtime.record_actor_results((row,))

        result_deadline = time.monotonic() + 2.0
        while len(result_by_actor) < len(jobs) and time.monotonic() < result_deadline:
            try:
                row = results.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(row, ActorResult):
                result_by_actor[row.actor_id] = row
                latest[row.actor_id] = ActorProgress(
                    row.actor_id,
                    row.game_id,
                    row.steps,
                    row.wins,
                    row.failures,
                    row.levels_completed,
                    row.replans,
                    row.planned_steps,
                )

        if len(result_by_actor) != len(jobs):
            missing = sorted(set(latest) - set(result_by_actor))
            raise RuntimeError(f"actor result missing for ids: {missing}")

        if progress_callback is not None:
            progress_callback(tuple(latest[key] for key in sorted(latest)))

        return tuple(result_by_actor[key] for key in sorted(result_by_actor))
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2.0)
        raise
