from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable

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
from v7.memory.evidence_store import EvidenceRecord
from v7.memory.evidence_types import EvidenceType
from v7.runtime import V7Runtime, V7RuntimeConfig


class MemoryGuidedActionSelector:
    def __init__(self, *, seed: int = 0, epsilon: float = 0.10) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.random = Random(int(seed))
        self.epsilon = float(epsilon)
        self.scorer = ContextualActionScorer()
        self.overlay = LocalCognitionOverlay()

    def choose(self, runtime: V7Runtime, actions: list[int], *, before=None):
        ordered = tuple(sorted(set(int(action) for action in actions)))
        if not ordered:
            raise ValueError("environment returned no available actions")
        if before is None:
            # Compatibility fallback for direct callers without an observation.
            return ordered[self.random.randrange(len(ordered))]
        contexts = self.overlay.build_context(
            structural_signature=structural_grid_signature(before),
            exact_signature=grid_signature(before),
        )
        decisions = self.scorer.score_actions(
            view=runtime.writer.published_view,
            contexts=contexts,
            actions=ordered,
            overlay=self.overlay,
        )
        if self.random.random() < self.epsilon:
            decision = decisions[self.random.randrange(len(decisions))]
        else:
            maximum = max(float(item.score) for item in decisions)
            weights = [math.exp((float(item.score) - maximum) / 0.35) for item in decisions]
            decision = self.random.choices(list(decisions), weights=weights, k=1)[0]
        return int(decision.action_id), contexts, decision


@dataclass(frozen=True, slots=True)
class ArcGameRunConfig:
    game_id: str
    steps: int = 1000
    seed: int = 0
    env_root: str | None = None
    commit_every: int = 1000
    epsilon: float = 0.10
    restore: bool = True

    def __post_init__(self) -> None:
        if self.steps < 1 or self.commit_every < 1:
            raise ValueError("steps and commit_every must be positive")


@dataclass(frozen=True, slots=True)
class ArcGameRunResult:
    game_id: str
    steps: int
    generation: int
    memories: int
    wins: int
    failures: int
    levels_completed: int
    resets: int


EnvFactory = Callable[..., ArcGridEnvironment]


def run_arc_game(
    root: str | Path,
    config: ArcGameRunConfig,
    *,
    env_factory: EnvFactory = ArcGridEnvironment,
) -> ArcGameRunResult:
    runtime = V7Runtime(V7RuntimeConfig.from_path(root, restore=config.restore))
    env = env_factory(game_id=config.game_id, seed=config.seed, env_root=config.env_root)
    selector = MemoryGuidedActionSelector(seed=config.seed, epsilon=config.epsilon)
    wins = failures = 0
    initial_levels = env.last_levels_completed
    generation = int(runtime.writer.published_view.generation_id)
    trajectory_actions: list[int] = []
    trajectory_contexts: list[int] = []
    trajectory_future = 0.0
    level_index = 0
    try:
        for step in range(1, config.steps + 1):
            if selector.overlay.should_reset():
                env.reset()
                selector.overlay.reset_episode_history(keep_statistics=True)
                trajectory_actions.clear()
                trajectory_contexts.clear()
                trajectory_future = 0.0
            before = env.observe()
            before_actions = env.available_actions()
            action, contexts, decision = selector.choose(runtime, before_actions, before=before)
            after = env.step(action)
            positive = env.last_outcome_polarity == "positive" or bool(getattr(env, "level_completed_event", False)) or env.last_outcome_state == "WIN"
            negative = env.last_outcome_polarity == "negative" or env.last_outcome_state == "GAME_OVER"
            terminal = 1 if positive else -1 if negative else 0
            transition_after = before if bool(getattr(env, "last_step_was_reset_boundary", False)) and terminal else after
            outcome = transformation_family_signature(before, transition_after)
            raw_transition = transition_signature(before, transition_after)
            changed = changed_cell_count(before, transition_after)
            error = selector.overlay.prediction_error(contexts.signatures, action, outcome)
            after_actions = env.available_actions()
            raw_action_delta = float(len(set(after_actions)) - len(set(before_actions)))
            future_delta = float(len(set(after_actions)) - len(set(before_actions)))
            if terminal > 0:
                future_delta += 4.0
            elif terminal < 0:
                future_delta -= 4.0
            selector.overlay.record_step(
                contexts=contexts.signatures,
                next_contexts=(),
                action_id=action,
                outcome_signature=outcome,
                terminal_polarity=terminal,
                prediction_error=error,
                future_option_delta=future_delta,
                changed=changed > 0,
            )
            next_signatures: tuple[int, ...] = ()
            if terminal == 0:
                next_context = selector.overlay.build_context(
                    structural_signature=structural_grid_signature(after),
                    exact_signature=grid_signature(after),
                )
                next_signatures = next_context.signatures
                for source, target in zip(contexts.signatures, next_signatures, strict=True):
                    selector.overlay.transitions[(int(source), int(action))][int(target)] += 1

            support = decision.support
            runtime.observe(ContextEpisodeEvidence(
                context_signature=int(support.context_signature),
                action_id=action,
                outcome_signature=outcome,
                success=terminal >= 0,
                prediction_error=error,
                future_option_delta=future_delta,
                source_game=config.game_id,
                source_context=str(support.context_signature),
                source_global_step=step,
                carrier_signature=carrier_signature(before, transition_after),
                decision_role_ids=tuple(int(v) for v in support.role_ids),
                decision_concept_ids=tuple(int(v) for v in support.concept_ids),
                terminal_polarity=terminal,
                raw_action_option_delta=raw_action_delta,
                decision_score=float(decision.score),
                max_action_score=float(decision.score),
                memory_guided=support.contextual_support > 0 or support.local_support > 0,
                context_signatures=contexts.signatures,
                next_context_signatures=next_signatures,
                exact_context_signature=grid_signature(before),
                structural_context_signature=structural_grid_signature(before),
                raw_transition_signature=raw_transition,
                decision_world_model_ids=tuple(int(v) for v in support.world_model_ids),
                decision_strategy_ids=tuple(int(v) for v in support.strategy_ids),
                changed_cells=changed,
            ))
            trajectory_actions.append(action)
            trajectory_contexts.append(int(contexts.planning_signature))
            trajectory_future += future_delta
            if terminal != 0:
                runtime.evidence.append_evidence(EvidenceRecord(
                    memory_id=None,
                    evidence_type=int(EvidenceType.TRAJECTORY),
                    generation_id=int(runtime.writer.mutable_generation_id),
                    source_game=config.game_id,
                    source_context=f"level_{level_index:04d}",
                    source_global_step=step,
                    payload={
                        "level_key": f"level_{level_index:04d}",
                        "steps_to_success": len(trajectory_actions),
                        "future_option_sum": trajectory_future,
                        "representative_action": trajectory_actions[-1] if trajectory_actions else None,
                        "action_sequence": list(trajectory_actions[:256]),
                        "context_sequence": list(trajectory_contexts[:256]),
                        "success": terminal > 0,
                    },
                ))
                if terminal > 0:
                    level_index += 1
                trajectory_actions.clear()
                trajectory_contexts.clear()
                trajectory_future = 0.0

            wins += int(env.last_outcome_state == "WIN")
            failures += int(env.last_outcome_state == "GAME_OVER")
            if step % config.commit_every == 0:
                generation = int(runtime.commit().state.generation_id)
        if config.steps % config.commit_every:
            generation = int(runtime.commit().state.generation_id)
        return ArcGameRunResult(
            config.game_id,
            config.steps,
            generation,
            len(runtime.writer.published_view.nodes),
            wins,
            failures,
            max(0, env.last_levels_completed - initial_levels),
            env.reset_count,
        )
    finally:
        runtime.close()
