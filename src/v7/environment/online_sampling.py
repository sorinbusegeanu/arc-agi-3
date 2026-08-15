from __future__ import annotations

import math
from dataclasses import replace
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
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.planning import PersistentPlanningGraph
from v7.memory.state import GateValidationState
from v7.memory.status import memory_is_active, memory_is_probe_eligible
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


def _raw_probe_strength(view, memory_id: MemoryId) -> float:
    """Counterfactual memory strength when a PROBE_ONLY node is enabled."""
    node = view.nodes.get(memory_id)
    if not memory_is_probe_eligible(node):
        return 0.0
    assert node is not None
    support = 1.0 - math.exp(-max(0, int(node.support_count)) / 3.0)
    score = view.scores.get(memory_id)
    semantic = 0.0
    if score is not None:
        semantic = max(
            float(score.significance),
            float(score.learning_value),
            float(score.transfer_prior),
            float(score.explanatory_potential),
        )
    strength = max(0.0, min(1.0, 0.5 * support + 0.5 * semantic))
    validation = int(
        getattr(node, "validation_state", GateValidationState.VALIDATED)
    )
    if validation == int(GateValidationState.REJECTED):
        return 0.0
    if validation == int(GateValidationState.TRUSTED):
        return strength
    if validation == int(GateValidationState.VALIDATED):
        return 0.90 * strength
    if validation in {
        int(GateValidationState.PROBE_ELIGIBLE),
        int(GateValidationState.TRANSFER_TESTED),
    }:
        # Probe lanes are already rare and explicitly isolated. Artificially
        # shrinking them below the gate threshold made validation impossible.
        return strength
    return 0.35 * strength


def _probe_component_weight(node) -> float:
    if node is None:
        return 0.0
    if node.level == MemoryLevel.M3 and int(node.type_id) == 300:
        return 0.08
    if node.level == MemoryLevel.M4:
        return 0.08
    if node.level == MemoryLevel.M5:
        return 0.10
    if node.level == MemoryLevel.M6:
        return 0.12
    return 0.0


def _active_group(decision, node) -> tuple[int, ...]:
    support = decision.support
    if node.level == MemoryLevel.M3:
        return tuple(int(value) for value in support.role_ids)
    if node.level == MemoryLevel.M4:
        return tuple(int(value) for value in support.concept_ids)
    if node.level == MemoryLevel.M5:
        return tuple(int(value) for value in support.world_model_ids)
    if node.level == MemoryLevel.M6:
        return tuple(int(value) for value in support.strategy_ids)
    return ()


def _maybe_select_probe(
    *,
    view,
    scorer: Phase1ActionScorer,
    contexts,
    decisions,
    selection,
    rng: Random,
    epsilon: float,
):
    """Occasionally enable one PROBE_ONLY memory without degrading action rank."""
    probe_probability = min(0.10, max(0.02, float(epsilon)))
    if probe_probability <= 0.0 or rng.random() >= probe_probability:
        return selection, None, 0.0
    by_action = {int(row.action_id): row for row in decisions}
    best: tuple[float, float, int, int] | None = None
    for context in contexts.signatures:
        rows = view.probe_score_inputs(
            context_signature=int(context),
            action_ids=tuple(sorted(by_action)),
        )
        for row in rows:
            decision = by_action.get(int(row.action_id))
            if decision is None:
                continue
            candidate_ids = tuple(row.role_ids) + tuple(row.concept_ids)
            for memory_id in candidate_ids:
                node = view.nodes.get(memory_id)
                if node is None or memory_is_active(node) or not memory_is_probe_eligible(node):
                    continue
                weight = _probe_component_weight(node)
                if weight <= 0.0:
                    continue
                active_strength = scorer.base._memory_strength(  # noqa: SLF001
                    view,
                    _active_group(decision, node),
                )
                enabled_strength = max(active_strength, _raw_probe_strength(view, memory_id))
                contribution = weight * max(0.0, enabled_strength - active_strength)
                if contribution <= 0.0:
                    continue
                augmented_score = float(decision.score) + float(contribution)
                candidate = (
                    augmented_score,
                    float(contribution),
                    -int(row.action_id),
                    -int(memory_id),
                )
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        return selection, None, 0.0

    augmented_score, contribution, neg_action, neg_memory = best
    action_id = -neg_action
    memory_id = MemoryId(-neg_memory)
    selected_action = int(selection.decision.action_id)
    selected_score = float(selection.decision.score)

    # A probe may change the action only if enabling the candidate makes that
    # action genuinely outrank the already selected action. Explicit exploration
    # is never cancelled by a probe for another action.
    if action_id != selected_action:
        if str(selection.mode) == "exploration" or augmented_score <= selected_score:
            return selection, None, 0.0

    decision = by_action[action_id]
    decision = replace(decision, score=float(augmented_score))
    return (
        replace(selection, decision=decision, mode="probe", strategy_id=None),
        memory_id,
        float(contribution),
    )


def _m1_component(features: tuple[float, float, float, float, float]) -> float:
    confidence, value, future, failure, contradiction = features
    return (
        0.22 * float(confidence)
        + 0.16 * float(value)
        + 0.16 * max(0.0, float(future))
        - 0.24 * float(failure)
        - 0.10 * float(contradiction)
    )


def _decision_memory_contributions(
    *,
    view,
    scorer: Phase1ActionScorer,
    decision,
    probe_memory_id: MemoryId | None,
    probe_contribution: float,
) -> tuple[tuple[int, ...], tuple[tuple[int, float], ...]]:
    """Counterfactual score delta when each selected memory is removed."""
    support = decision.support
    active_row = view.score_inputs(
        context_signature=int(support.context_signature),
        action_ids=(int(decision.action_id),),
    )[0]
    contingency_ids = tuple(active_row.contingency_ids)
    contributions: dict[int, float] = {}

    m1_on = _m1_component(
        scorer.base._m1_features(view, contingency_ids)  # noqa: SLF001
    )
    for memory_id in contingency_ids:
        off_ids = tuple(value for value in contingency_ids if value != memory_id)
        off = _m1_component(
            scorer.base._m1_features(view, off_ids)  # noqa: SLF001
        )
        delta = float(m1_on - off)
        if delta != 0.0:
            contributions[int(memory_id)] = delta

    groups = (
        (0.08, tuple(int(value) for value in support.role_ids)),
        (0.08, tuple(int(value) for value in support.concept_ids)),
        (0.10, tuple(int(value) for value in support.world_model_ids)),
        (0.12, tuple(int(value) for value in support.strategy_ids)),
    )
    for weight, ids in groups:
        if not ids:
            continue
        on = scorer.base._memory_strength(view, ids)  # noqa: SLF001
        for raw_memory_id in ids:
            off_ids = tuple(value for value in ids if value != raw_memory_id)
            off = scorer.base._memory_strength(view, off_ids)  # noqa: SLF001
            delta = float(weight * (on - off))
            if delta != 0.0:
                contributions[raw_memory_id] = contributions.get(raw_memory_id, 0.0) + delta

    if probe_memory_id is not None and probe_contribution != 0.0:
        contributions[int(probe_memory_id)] = float(probe_contribution)
    return (
        tuple(int(value) for value in contingency_ids),
        tuple(sorted(contributions.items())),
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
        selection, probe_memory_id, probe_contribution = _maybe_select_probe(
            view=view,
            scorer=scorer,
            contexts=contexts,
            decisions=decisions,
            selection=selection,
            rng=rng,
            epsilon=job.epsilon,
        )
        decision = selection.decision
        action = int(decision.action_id)
        (
            decision_contingency_ids,
            decision_memory_contributions,
        ) = _decision_memory_contributions(
            view=view,
            scorer=scorer,
            decision=decision,
            probe_memory_id=probe_memory_id,
            probe_contribution=probe_contribution,
        )
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
        if probe_memory_id is not None:
            max_score = max(max_score, float(decision.score))

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
        role_ids = {int(value) for value in support.role_ids}
        concept_ids = {int(value) for value in support.concept_ids}
        world_model_ids = {int(value) for value in support.world_model_ids}
        if probe_memory_id is not None:
            probe_node = view.nodes.get(probe_memory_id)
            if probe_node is not None:
                if probe_node.level == MemoryLevel.M3 and int(probe_node.type_id) == 300:
                    role_ids.add(int(probe_memory_id))
                elif probe_node.level == MemoryLevel.M4:
                    concept_ids.add(int(probe_memory_id))
                elif probe_node.level == MemoryLevel.M5:
                    world_model_ids.add(int(probe_memory_id))
                elif probe_node.level == MemoryLevel.M6:
                    strategy_ids.add(int(probe_memory_id))
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
                decision_role_ids=tuple(sorted(role_ids)),
                decision_concept_ids=tuple(sorted(concept_ids)),
                terminal_polarity=terminal,
                raw_action_option_delta=raw_action_delta,
                decision_score=float(decision.score),
                max_action_score=max_score,
                memory_guided=(
                    selection.mode in {"memory", "strategy", "probe"}
                    or support.contextual_support > 0
                    or support.local_support > 0
                ),
                context_signatures=contexts.signatures,
                next_context_signatures=next_signatures,
                exact_context_signature=exact,
                structural_context_signature=structural,
                raw_transition_signature=raw_transition,
                decision_world_model_ids=tuple(sorted(world_model_ids)),
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
                decision_contingency_ids=decision_contingency_ids,
                decision_memory_contributions=decision_memory_contributions,
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
