from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from random import Random


_INSTALLED = False
_BASE_ACTOR_WORKER = None
_SAMPLING_MODE_ENV = "ARC_AGI3_V8_SAMPLING_MODE"
_MAX_DECISION_POINTS = 256
_VERIFICATION_REPEATS = 2


@dataclass(slots=True)
class DecisionPoint:
    level: int
    context: int
    anchor: tuple[int, ...]
    available_actions: set[int] = field(default_factory=set)
    tested_actions: set[int] = field(default_factory=set)
    priority: int = 1
    successful_action: int | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (int(self.level), int(self.context))

    def untested(self) -> tuple[int, ...]:
        return tuple(sorted(self.available_actions - self.tested_actions))


@dataclass(slots=True)
class VerificationTask:
    point_key: tuple[int, int]
    anchor: tuple[int, ...]
    action: int
    remaining: int = _VERIFICATION_REPEATS


@dataclass(slots=True)
class Intervention:
    kind: str
    point_key: tuple[int, int] | None
    action: int
    anchor: tuple[int, ...]


class DecisionPointSampler:
    """Bounded breadth-first intervention scheduler for one game.

    Canonical memory remains the learning authority.  This object is only an
    interaction-control sidecar: it remembers replayable decision points and
    decides where the next no-plan intervention should occur.
    """

    def __init__(self, game_id: str, *, seed: int = 0, max_points: int = _MAX_DECISION_POINTS) -> None:
        self.game_id = str(game_id)
        self.max_points = max(8, int(max_points))
        self.rng = Random(int(seed) ^ 0x821D15C0)
        self.points: dict[tuple[int, int], DecisionPoint] = {}
        self.pending_reset: tuple[tuple[int, ...], tuple[int, int]] | None = None
        self.replay_actions: deque[int] = deque()
        self.replay_target: tuple[int, int] | None = None
        self.verification: VerificationTask | None = None
        self.transfer_action: int | None = None
        self.transfer_from_level = -1
        self.current: Intervention | None = None
        self._incoming_priority = 3

    def begin_lease(self, seed: int) -> None:
        self.rng.seed(int(seed) ^ 0x821D15C0)
        self.pending_reset = None
        self.replay_actions.clear()
        self.replay_target = None
        self.verification = None
        self.current = None
        self._incoming_priority = 3

    def _evict_if_needed(self) -> None:
        if len(self.points) < self.max_points:
            return
        disposable = [
            row for row in self.points.values()
            if row.successful_action is None
            and (self.verification is None or row.key != self.verification.point_key)
        ]
        if not disposable:
            return
        victim = min(
            disposable,
            key=lambda row: (int(row.priority), -len(row.anchor), int(row.level), int(row.context)),
        )
        self.points.pop(victim.key, None)

    def register_point(
        self,
        *,
        level: int,
        context: int,
        anchor: tuple[int, ...],
        actions: tuple[int, ...],
        priority: int | None = None,
    ) -> DecisionPoint:
        key = (int(level), int(context))
        row = self.points.get(key)
        if row is None:
            self._evict_if_needed()
            row = DecisionPoint(int(level), int(context), tuple(anchor))
            self.points[key] = row
        elif len(anchor) < len(row.anchor):
            row.anchor = tuple(anchor)
        row.available_actions.update(int(value) for value in actions)
        if priority is not None:
            row.priority = max(int(row.priority), int(priority))
        return row

    def _best_frontier(self) -> DecisionPoint | None:
        candidates = [row for row in self.points.values() if row.untested()]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (
                int(row.priority),
                -len(row.anchor),
                int(row.level),
                -int(row.context),
            ),
        )

    def _schedule_point(self, point: DecisionPoint | None) -> None:
        if point is None:
            return
        self.pending_reset = (tuple(point.anchor), point.key)

    def prepare_step(self, env) -> bool:
        if self.pending_reset is None:
            return False
        anchor, target = self.pending_reset
        self.pending_reset = None
        env.reset()
        self.replay_actions = deque(int(value) for value in anchor)
        self.replay_target = target
        self.current = None
        return True

    def on_external_reset(self) -> None:
        if self.pending_reset is not None:
            anchor, target = self.pending_reset
            self.pending_reset = None
            self.replay_actions = deque(int(value) for value in anchor)
            self.replay_target = target
        else:
            self.replay_actions.clear()
            self.replay_target = None
        self.current = None

    def forced_action(
        self,
        *,
        level: int,
        context: int,
        actions: tuple[int, ...],
        history: tuple[int, ...],
    ) -> int | None:
        available = set(int(value) for value in actions)
        key = (int(level), int(context))

        if self.replay_actions:
            action = int(self.replay_actions[0])
            if action not in available:
                invalid = self.replay_target
                if invalid is not None:
                    self.points.pop(invalid, None)
                    if self.verification is not None and self.verification.point_key == invalid:
                        self.verification = None
                self.replay_actions.clear()
                self.replay_target = None
                self.current = None
                return None
            self.replay_actions.popleft()
            self.current = Intervention("REPLAY", self.replay_target, action, tuple(history))
            return action

        if self.replay_target is not None:
            target = self.replay_target
            self.replay_target = None
            if key != target:
                self.points.pop(target, None)
                if self.verification is not None and self.verification.point_key == target:
                    self.verification = None
                self.current = None
                return None

        task = self.verification
        if task is not None and key == task.point_key and int(task.action) in available:
            self.current = Intervention("VERIFICATION", key, int(task.action), tuple(history))
            return int(task.action)
        return None

    def discovery_action(
        self,
        *,
        level: int,
        context: int,
        actions: tuple[int, ...],
        history: tuple[int, ...],
    ) -> int | None:
        point = self.register_point(
            level=int(level),
            context=int(context),
            anchor=tuple(history),
            actions=tuple(actions),
            priority=self._incoming_priority,
        )
        self._incoming_priority = 1
        untested = list(point.untested())
        if not untested:
            self.current = None
            return None

        if (
            self.transfer_action is not None
            and int(level) > int(self.transfer_from_level)
            and int(self.transfer_action) in untested
        ):
            action = int(self.transfer_action)
            kind = "TRANSFER"
        else:
            action = int(untested[0])
            kind = "DISCOVERY"
        self.current = Intervention(kind, point.key, action, tuple(history))
        return action

    @staticmethod
    def signal_priority(
        *,
        success: bool,
        positive: bool,
        prediction_error: float,
        novel: bool,
        future_delta: float,
        bad: bool,
    ) -> int:
        if success:
            return 6
        if positive:
            return 5
        if float(prediction_error) >= 0.50:
            return 4
        if novel:
            return 3
        if float(future_delta) > 0.0:
            return 2
        if bad:
            return 0
        return 1

    def observe_transition(
        self,
        *,
        before_level: int,
        before_context: int,
        action: int,
        after_level: int,
        after_context: int,
        after_actions: tuple[int, ...],
        history_after: tuple[int, ...],
        changed_cells: int,
        terminal_state: str,
        terminal_polarity: int,
        level_advanced: bool,
        prediction_error: float,
        future_delta: float,
    ) -> None:
        intervention = self.current
        self.current = None
        if intervention is None:
            return
        if intervention.kind == "REPLAY":
            if str(terminal_state) == "GAME_OVER":
                invalid = intervention.point_key
                if invalid is not None:
                    self.points.pop(invalid, None)
                    if self.verification is not None and self.verification.point_key == invalid:
                        self.verification = None
                self.replay_actions.clear()
                self.replay_target = None
            return

        source = None if intervention.point_key is None else self.points.get(intervention.point_key)
        if source is not None and intervention.kind in {"DISCOVERY", "TRANSFER"}:
            source.tested_actions.add(int(action))

        success = bool(level_advanced or str(terminal_state) == "WIN")
        positive = bool(int(terminal_polarity) > 0)
        after_key = (int(after_level), int(after_context))
        novel = after_key not in self.points
        repeated = after_key in self.points and after_key != intervention.point_key
        bad = bool(
            str(terminal_state) == "GAME_OVER"
            or (
                not success
                and (
                    int(after_context) == int(before_context)
                    or int(changed_cells) <= 0
                    or repeated
                )
            )
        )
        priority = self.signal_priority(
            success=success,
            positive=positive,
            prediction_error=float(prediction_error),
            novel=novel,
            future_delta=float(future_delta),
            bad=bad,
        )

        destination = None
        if str(terminal_state) != "GAME_OVER" and after_actions:
            destination = self.register_point(
                level=int(after_level),
                context=int(after_context),
                anchor=tuple(history_after),
                actions=tuple(after_actions),
                priority=priority,
            )

        if intervention.kind == "VERIFICATION":
            task = self.verification
            if task is None:
                return
            if success:
                task.remaining -= 1
                self.transfer_action = int(task.action)
                self.transfer_from_level = max(int(self.transfer_from_level), int(before_level))
                if task.remaining > 0:
                    self.pending_reset = (tuple(task.anchor), task.point_key)
                else:
                    self.verification = None
                    if destination is not None:
                        destination.priority = max(destination.priority, 6)
                        self._incoming_priority = max(self._incoming_priority, 6)
            else:
                self.verification = None
                self._schedule_point(self._best_frontier())
            return

        if success and source is not None:
            source.successful_action = int(action)
            source.priority = max(source.priority, 6)
            self.transfer_action = int(action)
            self.transfer_from_level = max(int(self.transfer_from_level), int(before_level))
            self.verification = VerificationTask(
                source.key,
                tuple(source.anchor),
                int(action),
                _VERIFICATION_REPEATS,
            )
            self.pending_reset = (tuple(source.anchor), source.key)
            return

        self._incoming_priority = max(self._incoming_priority, priority)
        best = self._best_frontier()
        if best is None:
            return
        if destination is not None and best.key == destination.key and not bad:
            return
        self._schedule_point(best)


_SAMPLERS: dict[tuple[int, str], DecisionPointSampler] = {}


def _decision_mode_enabled() -> bool:
    mode = os.environ.get(_SAMPLING_MODE_ENV, "DISCOVERY").strip().upper()
    return mode in {"", "DISCOVERY"}


def _sampler_for(job) -> DecisionPointSampler:
    key = (int(job.actor_id), str(job.game_id))
    sampler = _SAMPLERS.get(key)
    if sampler is None:
        sampler = DecisionPointSampler(str(job.game_id), seed=int(job.seed))
        _SAMPLERS[key] = sampler
        if len(_SAMPLERS) > 64:
            for stale in tuple(_SAMPLERS)[:-64]:
                _SAMPLERS.pop(stale, None)
    sampler.begin_lease(int(job.seed))
    return sampler


def _fallback_action(view, context, before_actions, local_overlay, rng, epsilon: float) -> int:
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
        return int(unseen[rng.randrange(len(unseen))])
    if rng.random() < float(epsilon):
        return int(before_actions[rng.randrange(len(before_actions))])
    return int(min(combined, key=lambda row: (-row[2], -row[1], row[0]))[0])


def _decision_actor_worker(
    *,
    job,
    experience_ring_args,
    read_descriptors,
    watermark,
    stop_event,
    result_queue,
    progress_queue=None,
    reporting_queue=None,
    actor_throttle=None,
    snapshot_freeze=None,
    startup_ready=None,
    startup_gate=None,
    record_cuts=None,
) -> None:
    from v7.environment.arc_adapter import ArcGridEnvironment
    from v7.environment.encoding import (
        carrier_signature,
        changed_cell_count,
        structural_grid_signature,
        transformation_family_signature,
        transition_signature,
    )
    from v8 import actor as actor_module
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.model import EventId, ExperienceEvent, PipelineEvent, encode_pipeline, stable_u64
    from v8.persistent_identity import arc_world_id, trajectory_identity
    from v8.ring import SharedRingBuffer

    sampler = _sampler_for(job)
    optimizer._reset_capture(job)
    ring = SharedRingBuffer(**experience_ring_args)
    view = actor_module.open_actor_read_view(
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
    source_world_id = arc_world_id(job.game_id)
    rolling_trajectory = trajectory_identity(
        source_world_id,
        producer_id=job.actor_id,
        episode_ordinal=0,
        sequence_base=sequence_base,
        namespace=b"v8-traj-seed",
    )
    local_overlay: dict[tuple[int, int], tuple[int, float]] = {}
    recent_contexts: deque[int] = deque(maxlen=8)
    wins = failures = levels_completed = 0
    replans = planned_steps = 0
    selected_outcome = None
    selected_strategy = None
    strategy_stats = {}
    pending_strategy_stats = {}
    preference_probes = []
    pending_preference_probes = []
    replanning_trials = []
    pending_replanning_trials = []
    last_levels = int(env.last_levels_completed)
    next_progress = time.monotonic() + 5.0
    next_learning_publish = time.monotonic() + actor_module._LEARNING_PUBLISH_INTERVAL_SECONDS
    next_graph_check_step = int(job.graph_check_steps)

    try:
        if startup_ready is not None:
            startup_ready.set()
        while startup_gate is not None and not startup_gate.wait(0.05) and not stop_event.is_set():
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

            if sampler.prepare_step(env):
                rolling_trajectory = stable_u64(
                    job.actor_id,
                    sequence_base + sequence,
                    int(env.reset_count),
                    person=b"v8-traj-reset",
                )
                selected_outcome = selected_strategy = None
                recent_contexts.clear()
                local_overlay.clear()
                last_levels = int(env.last_levels_completed)

            next_graph_check_step = actor_module._refresh_actor_graph_if_due(
                view,
                completed_steps=sequence,
                next_check_step=next_graph_check_step,
                check_interval_steps=job.graph_check_steps,
            )

            before = env.observe()
            before_actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            if not before_actions:
                env.reset()
                sampler.on_external_reset()
                selected_outcome = selected_strategy = None
                recent_contexts.clear()
                local_overlay.clear()
                last_levels = int(env.last_levels_completed)
                continue
            context = int(structural_grid_signature(before))
            before_level = int(env.last_levels_completed)
            history_before = tuple(int(value) for value in optimizer._ACTOR_ACTION_HISTORY)

            action = sampler.forced_action(
                level=before_level,
                context=context,
                actions=before_actions,
                history=history_before,
            )
            planned = None
            explicit_replan = None
            plans = ()
            if action is None:
                plans = view.plan_candidates(context, before_actions)
                planned = plans[0] if plans else None
                if planned is not None:
                    alternatives = [row for row in plans[1:] if row.outcome_uid != planned.outcome_uid]
                    if alternatives and len(preference_probes) < int(job.max_probe_records):
                        probe = actor_module.PreferenceProbeResult(
                            planned.outcome_uid,
                            alternatives[0].outcome_uid,
                            stable_u64(context, person=b"v8-context"),
                            planned.outcome_uid,
                            bool(planned.preference_influenced),
                        )
                        preference_probes.append(probe)
                        pending_preference_probes.append(probe)

                    same_outcome = [
                        row for row in plans[1:]
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
                    action = sampler.discovery_action(
                        level=before_level,
                        context=context,
                        actions=before_actions,
                        history=history_before,
                    )
                    if action is None:
                        action = _fallback_action(
                            view,
                            context,
                            before_actions,
                            local_overlay,
                            rng,
                            float(job.epsilon),
                        )

            prediction_distribution = view.outcome_distribution(context, int(action))

            if actor_throttle is not None:
                with actor_throttle.get_lock():
                    delay = float(actor_throttle.value)
                if delay > 0:
                    time.sleep(delay)

            after = env.step(int(action))
            after_actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            after_context = int(structural_grid_signature(after))
            outcome = int(transition_signature(before, after))
            family = int(transformation_family_signature(before, after))
            carrier = int(carrier_signature(before, after) or 0)
            changed = int(changed_cell_count(before, after))
            future_delta = float(len(after_actions) - len(before_actions))
            terminal_polarity = actor_module._polarity(env.last_outcome_polarity)
            prediction_error = (
                0.0
                if not prediction_distribution
                else max(0.0, 1.0 - float(prediction_distribution.get(outcome, 0.0)))
            )
            rolling_trajectory = stable_u64(
                rolling_trajectory, context, int(action), outcome, person=b"v8-trajectory"
            )

            next_sequence = sequence + 1
            producer_sequence = sequence_base + next_sequence
            def packet_for_watermark(current_watermark: int) -> bytes:
                event = ExperienceEvent(
                    event_id=EventId.from_producer(job.actor_id, producer_sequence),
                    watermark=current_watermark,
                    producer_id=job.actor_id,
                    producer_sequence=producer_sequence,
                    source_game_hash=source_world_id,
                    global_step=current_watermark,
                    context_signature=context,
                    action_id=int(action),
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

            published_watermark = actor_module._publish_actor_packet(
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

            significance = actor_module._local_significance(changed, future_delta)
            local_value = 0.30 * significance + 0.55 * terminal_polarity + 0.15 * math.tanh(future_delta)
            old_support, old_score = local_overlay.get((context, int(action)), (0, 0.0))
            new_support = old_support + 1
            local_overlay[(context, int(action))] = (
                new_support,
                (old_score * old_support + local_value) / new_support,
            )

            prior_levels = last_levels
            terminal_game = actor_module._is_new_terminal_game(env)
            if terminal_game and env.last_outcome_state == "WIN":
                wins += 1
            elif terminal_game and env.last_outcome_state == "GAME_OVER":
                failures += 1
            current_levels = int(env.last_levels_completed)
            level_advanced = current_levels > prior_levels
            if level_advanced:
                levels_completed += current_levels - prior_levels

            history_after = tuple(int(value) for value in optimizer._ACTOR_ACTION_HISTORY)
            sampler.observe_transition(
                before_level=before_level,
                before_context=context,
                action=int(action),
                after_level=current_levels,
                after_context=after_context,
                after_actions=after_actions,
                history_after=history_after,
                changed_cells=changed,
                terminal_state=str(env.last_outcome_state),
                terminal_polarity=terminal_polarity,
                level_advanced=level_advanced,
                prediction_error=prediction_error,
                future_delta=future_delta,
            )

            observed_variant, observed_coarse = actor_module._observed_outcome_uids(
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
                step_cost = actor_module._trajectory_step_cost(
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
                trial = actor_module.ReplanningTrialResult(
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
                if now >= next_learning_publish and actor_module._publish_learning(
                    progress_queue,
                    job=job,
                    strategy_stats=pending_strategy_stats,
                    preference_probes=pending_preference_probes,
                    replanning_trials=pending_replanning_trials,
                ):
                    pending_strategy_stats.clear()
                    pending_preference_probes.clear()
                    pending_replanning_trials.clear()
                    next_learning_publish = now + actor_module._LEARNING_PUBLISH_INTERVAL_SECONDS
                local_overlay.clear()
                actor_module._reset_after_terminal_game(env, terminal_wait_seconds)
                sampler.on_external_reset()
                reset_boundary = True
            elif reset_boundary:
                sampler.on_external_reset()

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
                actor_module._publish_progress(
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

        final_learning_published = actor_module._publish_learning(
            progress_queue,
            job=job,
            strategy_stats=pending_strategy_stats,
            preference_probes=pending_preference_probes,
            replanning_trials=pending_replanning_trials,
        )
        pending_learning = None
        if not final_learning_published:
            pending_learning = actor_module._learning_batch(
                job=job,
                strategy_stats=pending_strategy_stats,
                preference_probes=pending_preference_probes,
                replanning_trials=pending_replanning_trials,
            )
        actor_module._publish_progress(
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
            actor_module.ActorResult(
                job.actor_id,
                job.game_id,
                sequence,
                wins,
                failures,
                levels_completed,
                int(env.reset_count),
                replans,
                planned_steps,
                actor_module._stats_tuple(strategy_stats),
                tuple(preference_probes),
                tuple(replanning_trials),
                pending_learning,
            )
        )
    finally:
        view.close()
        ring.close()
        optimizer._reset_capture(None)


def _actor_worker_v821(*, job, **kwargs):
    if not _decision_mode_enabled():
        return _BASE_ACTOR_WORKER(job=job, **kwargs)
    return _decision_actor_worker(job=job, **kwargs)


def install_decision_point_sampling_v821() -> None:
    global _INSTALLED, _BASE_ACTOR_WORKER
    if _INSTALLED:
        return
    from v8 import actor as actor_module

    _BASE_ACTOR_WORKER = actor_module.actor_worker
    actor_module.actor_worker = _actor_worker_v821
    _INSTALLED = True
