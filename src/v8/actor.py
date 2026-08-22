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
_LEARNING_PUBLISH_INTERVAL_SECONDS = 5.0
_ACTOR_GRAPH_CHECK_INTERVAL_STEPS = 1_000
_ACTOR_STARTUP_TIMEOUT_SECONDS = 300.0
_ACTOR_RING_RETRY_SECONDS = 0.001


def _publish_actor_packet(
    ring,
    watermark,
    packet_for_watermark: Callable[[int], bytes],
    *,
    stop_event,
    snapshot_freeze=None,
    retry_seconds: float = _ACTOR_RING_RETRY_SECONDS,
) -> int | None:
    """Publish one ordered actor packet without waiting under the global lock."""

    delay = max(0.0, float(retry_seconds))
    while not stop_event.is_set():
        if snapshot_freeze is not None and snapshot_freeze.is_set():
            stop_event.wait(0.002)
            continue
        with watermark.get_lock():
            current = int(watermark.value) + 1
            packet = packet_for_watermark(current)
            # A timed wait here used to hold the global watermark lock for up
            # to 50 ms. Under pressure that serialized the complete actor pool.
            if ring.put(packet, timeout=0.0):
                watermark.value = current
                return current
        stop_event.wait(delay)
    return None


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
    graph_check_steps: int = _ACTOR_GRAPH_CHECK_INTERVAL_STEPS


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
    pending_learning: ActorLearningBatch | None = None


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


def _learning_batch(
    *,
    job: ActorJob,
    strategy_stats: dict[MemoryUid, list[float]],
    preference_probes: list[PreferenceProbeResult],
    replanning_trials: list[ReplanningTrialResult],
) -> ActorLearningBatch | None:
    stats = _stats_tuple(strategy_stats)
    if not stats and not preference_probes and not replanning_trials:
        return None
    return ActorLearningBatch(
        job.actor_id,
        job.game_id,
        stats,
        tuple(preference_probes),
        tuple(replanning_trials),
        len(replanning_trials),
    )


def _publish_learning(
    progress_queue: mp.Queue | None,
    *,
    job: ActorJob,
    strategy_stats: dict[MemoryUid, list[float]],
    preference_probes: list[PreferenceProbeResult],
    replanning_trials: list[ReplanningTrialResult],
) -> bool:
    batch = _learning_batch(
        job=job,
        strategy_stats=strategy_stats,
        preference_probes=preference_probes,
        replanning_trials=replanning_trials,
    )
    if batch is None:
        return True
    if progress_queue is None:
        return False
    try:
        progress_queue.put_nowait(batch)
    except queue.Full:
        return False
    return True


def _merge_learning_batches(
    rows: Iterable[ActorLearningBatch],
) -> tuple[ActorLearningBatch, ...]:
    grouped: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row.actor_id), str(row.game_id))
        bucket = grouped.setdefault(
            key,
            {"stats": {}, "probes": [], "trials": [], "replans": 0},
        )
        stats = bucket["stats"]
        assert isinstance(stats, dict)
        for stat in row.strategy_stats:
            values = stats.setdefault(stat.strategy_uid, [0.0, 0.0, 0.0])
            values[0] += float(stat.attempts)
            values[1] += float(stat.successes)
            values[2] += float(stat.cost)
        probes = bucket["probes"]
        trials = bucket["trials"]
        assert isinstance(probes, list) and isinstance(trials, list)
        probes.extend(row.preference_probes)
        trials.extend(row.replanning_trials)
        bucket["replans"] = int(bucket["replans"]) + int(row.replans)

    merged: list[ActorLearningBatch] = []
    for (actor_id, game_id), bucket in sorted(grouped.items()):
        stats = bucket["stats"]
        probes = bucket["probes"]
        trials = bucket["trials"]
        assert isinstance(stats, dict)
        assert isinstance(probes, list) and isinstance(trials, list)
        merged.append(
            ActorLearningBatch(
                actor_id,
                game_id,
                _stats_tuple(stats),
                tuple(probes),
                tuple(trials),
                int(bucket["replans"]),
            )
        )
    return tuple(merged)


def _is_new_terminal_game(env) -> bool:
    return bool(
        env.last_outcome_state in {"WIN", "GAME_OVER"}
        and not env.last_step_was_reset_boundary
    )


def _reset_after_terminal_game(env, wait_seconds: float) -> None:
    if wait_seconds > 0:
        time.sleep(float(wait_seconds))
    env.reset()


def open_actor_read_view(
    read_descriptors: tuple[ShardReadDescriptor, ...],
    *,
    refresh_interval_seconds: float | None,
    record_cuts: dict[tuple[str, str], tuple[tuple[object, ...], int]] | None = None,
) -> LiveReadView:
    """Construct the installed actor-only view through one runtime authority."""
    return LiveReadView(
        read_descriptors,
        refresh_interval_seconds=refresh_interval_seconds,
        record_cuts=record_cuts,
    )


def _refresh_actor_graph_if_due(
    read_view: LiveReadView,
    *,
    completed_steps: int,
    next_check_step: int,
    check_interval_steps: int = _ACTOR_GRAPH_CHECK_INTERVAL_STEPS,
) -> int:
    interval = int(check_interval_steps)
    if interval <= 0:
        raise ValueError("graph check interval must be positive")
    if int(completed_steps) < int(next_check_step):
        return int(next_check_step)
    read_view.invalidate_strategy_cache()
    return (int(completed_steps) // interval + 1) * interval


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
    startup_ready: mp.synchronize.Event | None = None,
    startup_gate: mp.synchronize.Event | None = None,
    record_cuts: dict[tuple[str, str], tuple[tuple[object, ...], int]] | None = None,
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
    view = open_actor_read_view(
        read_descriptors,
        refresh_interval_seconds=None,
        record_cuts=record_cuts,
    )
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
    next_learning_publish = time.monotonic() + _LEARNING_PUBLISH_INTERVAL_SECONDS
    next_graph_check_step = int(job.graph_check_steps)
    try:
        if startup_ready is not None:
            startup_ready.set()
        while (
            startup_gate is not None
            and not startup_gate.wait(0.05)
            and not stop_event.is_set()
        ):
            pass
        if stop_event.is_set():
            return

        for _ in range(int(job.steps)):
            if stop_event.is_set():
                break
            while snapshot_freeze is not None and snapshot_freeze.is_set() and not stop_event.is_set():
                time.sleep(0.002)
            if stop_event.is_set():
                break

            next_graph_check_step = _refresh_actor_graph_if_due(
                view,
                completed_steps=sequence,
                next_check_step=next_graph_check_step,
                check_interval_steps=job.graph_check_steps,
            )

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
            def packet_for_watermark(current_watermark: int) -> bytes:
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
                return encode_pipeline(PipelineEvent(event))

            published_watermark = _publish_actor_packet(
                ring,
                watermark,
                packet_for_watermark,
                stop_event=stop_event,
                snapshot_freeze=snapshot_freeze,
            )
            if published_watermark is None:
                break
            current_watermark = int(published_watermark)
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
            terminal_game = _is_new_terminal_game(env)
            if terminal_game and env.last_outcome_state == "WIN":
                wins += 1
            elif terminal_game and env.last_outcome_state == "GAME_OVER":
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
            reset_boundary = current_levels < last_levels or env.last_step_was_reset_boundary
            if terminal_game:
                now = time.monotonic()
                if now >= next_learning_publish and _publish_learning(
                    progress_queue,
                    job=job,
                    strategy_stats=pending_strategy_stats,
                    preference_probes=pending_preference_probes,
                    replanning_trials=pending_replanning_trials,
                ):
                    pending_strategy_stats.clear()
                    pending_preference_probes.clear()
                    pending_replanning_trials.clear()
                    next_learning_publish = now + _LEARNING_PUBLISH_INTERVAL_SECONDS
                local_overlay.clear()
                _reset_after_terminal_game(env, terminal_wait_seconds)
                reset_boundary = True
            if reset_boundary:
                rolling_trajectory = stable_u64(
                    job.actor_id, producer_sequence, current_watermark, person=b"v8-traj-reset"
                )
                selected_outcome = selected_strategy = None
                recent_contexts.clear()
                local_overlay.clear()
            last_levels = int(env.last_levels_completed) if reset_boundary else current_levels

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

        final_learning_published = _publish_learning(
            progress_queue,
            job=job,
            strategy_stats=pending_strategy_stats,
            preference_probes=pending_preference_probes,
            replanning_trials=pending_replanning_trials,
        )
        pending_learning = None
        if not final_learning_published:
            pending_learning = _learning_batch(
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
                pending_learning,
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
    jobs = tuple(jobs)
    peers = getattr(runtime, "peers", None)
    if peers is not None:
        peers.pause()
    runtime.start()
    if not jobs:
        if peers is not None:
            peers.resume()
        return ()
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")

    ctx = runtime._mp_ctx
    startup_gate = ctx.Event() if hasattr(ctx, "Event") else None
    startup_ready = (
        tuple(ctx.Event() for _job in jobs) if startup_gate is not None else ()
    )
    if peers is not None:
        startup_timeout = (
            _ACTOR_STARTUP_TIMEOUT_SECONDS
            if timeout is None
            else max(0.01, min(float(timeout), _ACTOR_STARTUP_TIMEOUT_SECONDS))
        )
        if not peers.wait_idle(startup_timeout):
            raise TimeoutError("v8 peers did not pause before actor startup")
        runtime.wait_quiescent(
            timeout=startup_timeout,
            resume_peers=False,
            settle_peers=False,
        )
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
                "startup_ready": (
                    None if startup_gate is None else startup_ready[index]
                ),
                "startup_gate": startup_gate,
            },
            name=f"v8-actor-{job.actor_id:03d}-{job.game_id}",
            daemon=True,
        )
        for index, job in enumerate(jobs)
    ]
    latest = {
        job.actor_id: ActorProgress(job.actor_id, job.game_id, 0, 0, 0, 0)
        for job in jobs
    }
    result_by_actor: dict[int, ActorResult] = {}
    started = time.monotonic()
    deadline = None if timeout is None else started + float(timeout)
    next_report = started + float(progress_interval_seconds)
    started_processes: list[mp.Process] = []

    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        if startup_gate is not None:
            while not all(event.is_set() for event in startup_ready):
                failed = [
                    process
                    for process in started_processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    detail = ", ".join(
                        f"{process.name}={process.exitcode}" for process in failed
                    )
                    raise RuntimeError(f"actor failed during graph startup: {detail}")
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("actor graph startup timed out")
                time.sleep(0.01)
            startup_gate.set()
        if peers is not None:
            peers.resume()

        while True:
            learning_batches: list[ActorLearningBatch] = []
            for _ in range(_MAX_PARENT_MESSAGES_PER_CYCLE):
                try:
                    row = progress.get_nowait()
                except queue.Empty:
                    break
                if isinstance(row, ActorProgress):
                    latest[row.actor_id] = row
                elif isinstance(row, ActorLearningBatch):
                    learning_batches.append(row)
            if learning_batches:
                runtime.record_actor_results(_merge_learning_batches(learning_batches))

            result_learning: list[ActorLearningBatch] = []
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
                    if row.pending_learning is not None:
                        result_learning.append(row.pending_learning)
            if result_learning:
                runtime.record_actor_results(_merge_learning_batches(result_learning))

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

        learning_batches = []
        while True:
            try:
                row = progress.get_nowait()
            except queue.Empty:
                break
            if isinstance(row, ActorProgress):
                latest[row.actor_id] = row
            elif isinstance(row, ActorLearningBatch):
                learning_batches.append(row)
        if learning_batches:
            runtime.record_actor_results(_merge_learning_batches(learning_batches))

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
                if row.pending_learning is not None:
                    runtime.record_actor_results((row.pending_learning,))

        if len(result_by_actor) != len(jobs):
            missing = sorted(set(latest) - set(result_by_actor))
            raise RuntimeError(f"actor result missing for ids: {missing}")

        if progress_callback is not None:
            progress_callback(tuple(latest[key] for key in sorted(latest)))

        return tuple(result_by_actor[key] for key in sorted(result_by_actor))
    except BaseException:
        if startup_gate is not None:
            startup_gate.set()
        for process in started_processes:
            if process.is_alive():
                process.terminate()
        for process in started_processes:
            process.join(timeout=2.0)
        raise
