from __future__ import annotations

from random import Random
from time import perf_counter

from v7.context_evidence import ContextEpisodeEvidence
from v7.environment.ablation import CognitionAblation
from v7.environment.arc_adapter import ArcGridEnvironment
from v7.environment.cognition import LocalCognitionOverlay
from v7.environment.encoding import (
    carrier_signature,
    changed_cell_count,
    grid_signature,
    structural_grid_signature,
    transformation_family_signature,
    transition_signature,
)
from v7.environment.parallel_sampling import (
    SamplingBatchResult,
    SamplingJob,
    TrajectoryEvidence,
)
from v7.environment.phase1_policy import (
    Phase1ActionScorer,
    StrategyExecutionCursor,
    select_phase1_action,
)
from v7.memory.planning import PersistentPlanningGraph
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport


def _representative(actions: list[int]) -> int | None:
    if not actions:
        return None
    counts = {action: actions.count(action) for action in set(actions)}
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _future_option_ablation(
    decisions,
    action_id: int,
    *,
    future_option_ablated: bool,
) -> tuple[float, int, bool]:
    on_scores: dict[int, float] = {}
    off_scores: dict[int, float] = {}
    for row in decisions:
        component = float(
            getattr(row, "future_option_score_component", 0.0)
        )
        if future_option_ablated:
            off_score = float(row.score)
            on_score = off_score + component
        else:
            on_score = float(row.score)
            off_score = on_score - component
        on_scores[int(row.action_id)] = on_score
        off_scores[int(row.action_id)] = off_score
    action = int(action_id)
    if action not in on_scores or action not in off_scores:
        return 0.0, 0, False
    on_order = [
        key
        for key, _ in sorted(
            on_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    off_order = [
        key
        for key, _ in sorted(
            off_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return (
        float(on_scores[action] - off_scores[action]),
        int(off_order.index(action) - on_order.index(action)),
        bool(on_order and off_order and on_order[0] != off_order[0]),
    )


def sample_job(directory: str, handle, job: SamplingJob) -> SamplingBatchResult:
    """Sample one game with long-term planning plus worker-local online learning."""
    started = perf_counter()
    view = SegmentedMmapReadViewTransport(directory).attach(handle)
    attach_seconds = perf_counter() - started
    env = ArcGridEnvironment(
        game_id=job.game_id,
        seed=job.seed,
        env_root=job.env_root,
    )
    rng = Random(job.seed)
    overlay = LocalCognitionOverlay()
    planning = PersistentPlanningGraph.from_view(view)
    ablation_mask = int(getattr(job, "ablation_mask", 0) or 0)
    scorer = Phase1ActionScorer(planning, ablation_mask=ablation_mask)
    strategy_cursor = StrategyExecutionCursor()
    evidence = []
    trajectories = []
    wins = failures = levels_completed = 0
    level_index = 0
    segment_index = 0
    trajectory_actions: list[int] = []
    trajectory_contexts: list[int] = []
    trajectory_future = 0.0

    for local_step in range(1, job.steps + 1):
        reset_boundary_before_step = False
        if overlay.should_reset():
            env.reset()
            overlay.reset_episode_history(
                keep_statistics=True,
                clear_failure_streak=True,
            )
            strategy_cursor.reset()
            trajectory_actions.clear()
            trajectory_contexts.clear()
            trajectory_future = 0.0
            segment_index += 1
            reset_boundary_before_step = True

        trajectory_segment_id = (
            f"job_{int(job.job_index):06d}/segment_{segment_index:06d}"
        )
        before = env.observe()
        before_actions = env.available_actions()
        exact = grid_signature(before)
        structural = structural_grid_signature(before)
        contexts = overlay.build_context(
            structural_signature=structural,
            exact_signature=exact,
        )
        decisions = scorer.score_actions(
            view=view,
            contexts=contexts,
            actions=before_actions,
            overlay=overlay,
        )
        selection = select_phase1_action(
            view=view,
            planning=planning,
            cursor=strategy_cursor,
            contexts=contexts,
            decisions=decisions,
            rng=rng,
            epsilon=job.epsilon,
            ablation_mask=ablation_mask,
        )
        decision = selection.decision
        action = int(decision.action_id)
        (
            future_option_ablation_score_delta,
            future_option_ablation_rank_lift,
            future_option_ablation_choice_changed,
        ) = _future_option_ablation(
            decisions,
            action,
            future_option_ablated=bool(
                ablation_mask & int(CognitionAblation.FUTURE_OPTION)
            ),
        )
        max_score = max(float(item.score) for item in decisions)

        after = env.step(action)
        positive = (
            env.last_outcome_polarity == "positive"
            or bool(env.level_completed_event)
            or env.last_outcome_state == "WIN"
        )
        negative = (
            env.last_outcome_polarity == "negative"
            or env.last_outcome_state == "GAME_OVER"
        )
        terminal = 1 if positive else -1 if negative else 0
        transition_after = (
            before
            if bool(env.last_step_was_reset_boundary) and terminal
            else after
        )
        outcome = transformation_family_signature(before, transition_after)
        raw_transition = transition_signature(before, transition_after)
        changed = changed_cell_count(before, transition_after)
        prediction_error = overlay.prediction_error(
            contexts.signatures,
            action,
            outcome,
        )

        # Empty terminal frames are automatically reset by the adapter. The
        # resulting action list belongs to the next episode and must not be
        # measured as this action's future-option set.
        future_option_observable = not bool(env.last_step_was_reset_boundary)
        after_actions = env.available_actions() if future_option_observable else []
        raw_action_delta = (
            float(len(set(after_actions)) - len(set(before_actions)))
            if future_option_observable
            else 0.0
        )
        before_value = float(len(set(before_actions))) + 4.0 * float(
            decision.future_reachability
        )
        after_value = float(len(set(after_actions))) + (
            4.0 if terminal > 0 else -4.0 if terminal < 0 else 0.0
        )
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
        next_planning_context: int | None = None
        if terminal == 0:
            next_context = overlay.build_context(
                structural_signature=structural_grid_signature(after),
                exact_signature=grid_signature(after),
            )
            next_signatures = next_context.signatures
            next_planning_context = int(next_context.planning_signature)
            for source, target in zip(
                contexts.signatures,
                next_signatures,
                strict=True,
            ):
                overlay.transitions[(int(source), action)][int(target)] += 1
            overlay.recent_states.append(int(next_signatures[-1]))

        strategy_cursor.observe_outcome(
            selected_strategy_id=selection.strategy_id,
            terminal_polarity=terminal,
            next_context_signature=next_planning_context,
        )

        support = decision.support
        strategy_ids = {
            int(value)
            for value in getattr(support, "strategy_ids", ()) or ()
        }
        if selection.strategy_id is not None:
            strategy_ids.add(int(selection.strategy_id))
        evidence.append(
            ContextEpisodeEvidence(
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
                memory_guided=(
                    selection.mode in {"memory", "strategy"}
                    or support.contextual_support > 0
                    or support.local_support > 0
                ),
                context_signatures=contexts.signatures,
                next_context_signatures=next_signatures,
                exact_context_signature=exact,
                structural_context_signature=structural,
                raw_transition_signature=raw_transition,
                decision_world_model_ids=tuple(
                    int(v) for v in support.world_model_ids
                ),
                decision_strategy_ids=tuple(sorted(strategy_ids)),
                changed_cells=changed,
                selected_context_rank=int(
                    getattr(support, "context_rank", 0) or 0
                ),
                selection_mode=selection.mode,
                effective_epsilon=float(selection.effective_epsilon),
                development_stage=selection.development_stage,
                ablation_mask=ablation_mask,
                future_option_ablation_available=future_option_observable,
                future_option_ablation_score_delta=(
                    future_option_ablation_score_delta
                ),
                future_option_ablation_rank_lift=(
                    future_option_ablation_rank_lift
                ),
                future_option_ablation_choice_changed=(
                    future_option_ablation_choice_changed
                ),
                trajectory_segment_id=trajectory_segment_id,
                reset_boundary_before_step=reset_boundary_before_step,
                future_option_observable=future_option_observable,
            )
        )

        wins += int(env.last_outcome_state == "WIN")
        failures += int(env.last_outcome_state == "GAME_OVER")
        level_event = bool(env.level_completed_event)
        levels_completed += int(level_event)
        trajectory_actions.append(action)
        trajectory_contexts.append(int(contexts.planning_signature))
        trajectory_future += future_delta
        if level_event or env.last_outcome_state == "WIN" or terminal < 0:
            trajectories.append(
                TrajectoryEvidence(
                    game_id=job.game_id,
                    epoch=job.epoch,
                    level_key=f"level_{level_index:04d}",
                    steps_to_success=len(trajectory_actions),
                    source_global_step=job.global_step_offset + local_step,
                    future_option_sum=trajectory_future,
                    representative_action=_representative(trajectory_actions),
                    success=terminal >= 0,
                )
            )
            if level_event or env.last_outcome_state == "WIN":
                level_index += 1
            trajectory_actions.clear()
            trajectory_contexts.clear()
            trajectory_future = 0.0
            segment_index += 1

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
