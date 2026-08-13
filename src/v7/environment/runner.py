from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable

from v7.derivation.scientific import EpisodeEvidence
from v7.environment.arc_adapter import ArcGridEnvironment
from v7.environment.encoding import SupportedPredictionTracker, grid_signature, transition_signature
from v7.memory.scoring import VectorizedActionScorer
from v7.runtime import V7Runtime, V7RuntimeConfig


class MemoryGuidedActionSelector:
    def __init__(self, *, seed: int = 0, epsilon: float = 0.10) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.random = Random(int(seed))
        self.epsilon = float(epsilon)
        self.scorer = VectorizedActionScorer()

    def choose(self, runtime: V7Runtime, actions: list[int]) -> int:
        ordered = tuple(sorted(set(int(action) for action in actions)))
        if not ordered:
            raise ValueError("environment returned no available actions")
        batch = self.scorer.score(runtime.writer.published_view.packed_cognition, ordered)
        unseen = [int(action) for action, count in zip(batch.action_ids, batch.evidence_counts, strict=True) if int(count) == 0]
        if unseen:
            return unseen[self.random.randrange(len(unseen))]
        if self.random.random() < self.epsilon:
            return ordered[self.random.randrange(len(ordered))]
        best = batch.best_action()
        return ordered[0] if best is None else best


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


def run_arc_game(root: str | Path, config: ArcGameRunConfig, *, env_factory: EnvFactory = ArcGridEnvironment) -> ArcGameRunResult:
    runtime = V7Runtime(V7RuntimeConfig.from_path(root, restore=config.restore))
    env = env_factory(game_id=config.game_id, seed=config.seed, env_root=config.env_root)
    selector = MemoryGuidedActionSelector(seed=config.seed, epsilon=config.epsilon)
    predictor = SupportedPredictionTracker()
    wins = 0
    failures = 0
    initial_levels = env.last_levels_completed
    generation = int(runtime.writer.published_view.generation_id)
    try:
        for step in range(1, config.steps + 1):
            before = env.observe()
            before_actions = env.available_actions()
            action = selector.choose(runtime, before_actions)
            context = grid_signature(before)
            after = env.step(action)
            outcome = transition_signature(before, after)
            error = predictor.prediction_error(context, action, outcome)
            predictor.observe(context, action, outcome)
            after_actions = env.available_actions()
            runtime.observe(EpisodeEvidence(
                context_signature=context,
                action_id=action,
                outcome_signature=outcome,
                success=env.last_outcome_polarity != "negative",
                prediction_error=error,
                future_option_delta=float(len(set(after_actions)) - len(set(before_actions))),
                source_game=config.game_id,
                source_context=str(context),
                source_global_step=step,
            ))
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
