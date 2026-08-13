from __future__ import annotations

import math
from random import Random
from time import perf_counter

from v7.context_evidence import ContextEpisodeEvidence
from v7.environment.arc_adapter import ArcGridEnvironment
from v7.environment.cognition import ContextualActionScorer, LocalCognitionOverlay
from v7.environment.encoding import (
    carrier_signature,
    changed_cell_count,
    grid_signature,
    structural_grid_signature,
    transformation_family_signature,
    transition_signature,
)
from v7.environment.parallel_sampling import SamplingBatchResult, SamplingJob, TrajectoryEvidence
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport


def _select(decisions, rng: Random, epsilon: float):
    if not decisions:
        raise ValueError("environment returned no available actions")
    if rng.random() < max(0.0, min(1.0, float(epsilon))):
        return decisions[rng.randrange(len(decisions))]
    maximum = max(float(item.score) for item in decisions)
    weights = [math.exp((float(item.score) - maximum) / 0.35) for item in decisions]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return min(decisions, key=lambda item: (-item.score, item.action_id))
    return rng.choices(list(decisions), weights=weights, k=1)[0]


def _representative(actions: list[int]) -> int | None:
    if not actions:
        return None
    counts = {action: actions.count(action) for action in set(actions)}
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def sample_job(directory: str, handle, job: SamplingJob) -> SamplingBatchResult:
    """Sample one game with immutable long-term memory plus a worker-local online overlay."""
    started = perf_counter()
    view = SegmentedMmapReadViewTransport(directory).attach(handle)
    attach_seconds = perf_counter() - started
    env = ArcGridEnvironment(game_id=job.game_id, seed=job.seed, env_root=job.env_root)
    rng = Random(job.seed)
    overlay = LocalCognitionOverlay()
    scorer = ContextualActionScorer()
    evidence = []
    trajectories = []
    wins = failures = levels_completed = 0
    level_index = 0
    trajectory_actions: list[int] = []
    trajectory_contexts: list[int] = []
    trajectory_future = 0.0

    for local_step in range(1, job.steps + 1):
        if overlay.should_reset():
            env.reset()
            overlay.reset_episode_history(keep_statistics=True)
            trajectory_actions.clear()
            trajectory_contexts.clear()
            trajectory_future = 0.0

        before = env.observe()
        before_actions = env.available_actions()
        exact = grid_signature(before)
        structural = structural_grid_signature(before)
        contexts = overlay.build_context(structural_signature=structural, exact_signature=exact)
        decisions = scorer.score_actions(view=view, contexts=contexts, actions=before_actions, overlay=overlay)
        decision = _select(decisions, rng, job.epsilon)
        action = int(decision.action_id)
        max_score = max(float(item.score) for item in decisions)

        after = env.step(action)
        positive = env.last_outcome_polarity == "positive" or bool(env.level_completed_event) or env.last_outcome_state == "WIN"
        negative = env.last_outcome_polarity == "negative" or env.last_outcome_state == "GAME_OVER"
        terminal = 1 if positive else -1 if negative else 0
        transition_after = before if bool(env.last_step_was_reset_boundary) and terminal else after
        outcome = transformation_family_signature(before, transition_after)
        raw_transition = transition_signature(before, transition_after)
        changed = changed_cell_count(before, transition_after)
        prediction_error = overlay.prediction_error(contexts.signatures, action, outcome)
        after_actions = env.available_actions()
        raw_action_delta = float(len(set(after_actions)) - len(set(before_actions)))
        before_value = float(len(set(before_actions))) + 4.0 * float(decision.future_reachability)
        after_value = float(len(set(after_actions))) + (4.0 if terminal > 0 else -4.0 if terminal < 0 else 0.0)
        future_delta = after_value - before_value

        overlay.record_step(
            contexts=contexts.signatures,
            next_contexts=(),
            action_id=action,
            outcome_signature=outcome,
            terminal_polarity=terminal,
            prediction_error=prediction_error,
            future_option_delta=future_delta,
            changed=changed > 0,
        )
        next_signatures: tuple[int, ...] = ()
        if terminal == 0:
            next_context = overlay.build_context(
                structural_signature=structural_grid_signature(after),
                exact_signature=grid_signature(after),
            )
            next_signatures = next_context.signatures
            for source, target in zip(contexts.signatures, next_signatures, strict=True):
                overlay.transitions[(int(source), action)][int(target)] += 1
            overlay.recent_states.append(int(next_signatures[-1]))

        support = decision.support
        evidence.append(ContextEpisodeEvidence(
            context_signature=int(support.context_signature),
            action_id=action,
            outcome_signature=outcome,
            success=terminal >= 0,
            prediction_error=prediction_error,
            future_option_delta=future_delta,
            source_game=job.game_id,
            source_context=str(support.context_signature),
            source_global_step=job.global_step_offset + local_step,
            carrier_signature=carrier_signature(before, transition_after),
            decision_role_ids=tuple(int(v) for v in support.role_ids),
            decision_concept_ids=tuple(int(v) for v in support.concept_ids),
            terminal_polarity=terminal,
            raw_action_option_delta=raw_action_delta,
            decision_score=float(decision.score),
            max_action_score=max_score,
            memory_guided=support.contextual_support > 0 or support.local_support > 0,
            context_signatures=contexts.signatures,
            next_context_signatures=next_signatures,
            exact_context_signature=exact,
            structural_context_signature=structural,
            raw_transition_signature=raw_transition,
            decision_world_model_ids=tuple(int(v) for v in support.world_model_ids),
            decision_strategy_ids=tuple(int(v) for v in support.strategy_ids),
            changed_cells=changed,
        ))

        wins += int(env.last_outcome_state == "WIN")
        failures += int(env.last_outcome_state == "GAME_OVER")
        level_event = bool(env.level_completed_event)
        levels_completed += int(level_event)
        trajectory_actions.append(action)
        trajectory_contexts.append(int(contexts.planning_signature))
        trajectory_future += future_delta
        if level_event or env.last_outcome_state == "WIN" or terminal < 0:
            trajectories.append(TrajectoryEvidence(
                game_id=job.game_id,
                epoch=job.epoch,
                level_key=f"level_{level_index:04d}",
                steps_to_success=len(trajectory_actions),
                source_global_step=job.global_step_offset + local_step,
                future_option_sum=trajectory_future,
                representative_action=_representative(trajectory_actions),
                success=terminal >= 0,
            ))
            # Existing TrajectoryEvidence is kept ABI-compatible; full sequences
            # are carried by the episode stream and reconstructed at ingestion.
            if level_event or env.last_outcome_state == "WIN":
                level_index += 1
            trajectory_actions.clear()
            trajectory_contexts.clear()
            trajectory_future = 0.0

    return SamplingBatchResult(
        job_index=job.job_index,
        epoch=job.epoch,
        game_id=job.game_id,
        seed=job.seed,
        steps=job.steps,
        wins=wins,
        failures=failures,
        levels_completed=levels_completed,
        resets=env.reset_count,
        evidence=tuple(evidence),
        worker_seconds=perf_counter() - started,
        mmap_reattach_count=1,
        mmap_reattach_seconds=attach_seconds,
        trajectories=tuple(trajectories),
    )
