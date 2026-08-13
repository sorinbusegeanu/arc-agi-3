from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable

from v7.derivation.scientific import EpisodeEvidence
from v7.environment.arc_adapter import ArcGridEnvironment
from v7.environment.encoding import SupportedPredictionTracker, grid_signature, transition_signature
from v7.memory.evidence_lifecycle import TransferTrialRecord
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
        unseen = [
            int(action)
            for action, count in zip(batch.action_ids, batch.evidence_counts, strict=True)
            if int(count) == 0
        ]
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
    commit_every: int = 250
    epsilon: float = 0.10
    restore: bool = True
    op_mode: str = "normal"
    render_mode: str | None = None

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
    transfer_trials: int
    derived_families: int
    derived_roles: int
    derived_concepts: int
    derived_world_models: int
    derived_strategies: int


EnvFactory = Callable[..., ArcGridEnvironment]


def run_arc_game(
    root: str | Path,
    config: ArcGameRunConfig,
    *,
    env_factory: EnvFactory = ArcGridEnvironment,
) -> ArcGameRunResult:
    runtime = V7Runtime(V7RuntimeConfig.from_path(root, restore=config.restore))
    env = env_factory(
        game_id=config.game_id,
        seed=config.seed,
        env_root=config.env_root,
        op_mode=config.op_mode,
        render_mode=config.render_mode,
    )
    selector = MemoryGuidedActionSelector(seed=config.seed, epsilon=config.epsilon)
    predictor = SupportedPredictionTracker()
    wins = failures = levels_completed = transfer_trials = 0
    derived = [0, 0, 0, 0, 0]
    generation = int(runtime.writer.published_view.generation_id)
    try:
        for step in range(1, config.steps + 1):
            before = env.observe()
            before_actions = env.available_actions()
            action = selector.choose(runtime, before_actions)
            context = grid_signature(before)

            prior_input = runtime.writer.published_view.score_inputs(
                context_signature=context,
                action_ids=(action,),
            )[0]

            action_data = env.action_data(action, rng=selector.random) if hasattr(env, "action_data") else {}
            try:
                after = env.step(action, data=action_data)
            except TypeError:
                after = env.step(action)

            outcome = transition_signature(before, after)
            error = predictor.prediction_error(context, action, outcome)
            predictor.observe(context, action, outcome)
            after_actions = env.available_actions()
            future_option_delta = float(len(set(after_actions)) - len(set(before_actions)))

            runtime.observe(
                EpisodeEvidence(
                    context_signature=context,
                    action_id=action,
                    outcome_signature=outcome,
                    success=env.last_outcome_polarity != "negative",
                    prediction_error=error,
                    future_option_delta=future_option_delta,
                    source_game=config.game_id,
                    source_context=str(context),
                    source_global_step=step,
                )
            )

            if env.last_outcome_polarity in {"positive", "negative"}:
                transfer_trials += _record_transfer_trials(
                    runtime,
                    concept_ids=prior_input.concept_ids,
                    target_game=config.game_id,
                    success=env.last_outcome_polarity == "positive",
                    context=context,
                    action=action,
                    outcome=outcome,
                    source_global_step=step,
                )

            wins += int(env.last_outcome_state == "WIN")
            failures += int(env.last_outcome_state == "GAME_OVER")
            levels_completed += int(bool(env.level_completed_event))

            if step % config.commit_every == 0:
                generation = int(runtime.commit().state.generation_id)
                _accumulate_derived(derived, runtime)

        if config.steps % config.commit_every:
            generation = int(runtime.commit().state.generation_id)
            _accumulate_derived(derived, runtime)

        return ArcGameRunResult(
            game_id=config.game_id,
            steps=config.steps,
            generation=generation,
            memories=len(runtime.writer.published_view.nodes),
            wins=wins,
            failures=failures,
            levels_completed=levels_completed,
            resets=env.reset_count,
            transfer_trials=transfer_trials,
            derived_families=derived[0],
            derived_roles=derived[1],
            derived_concepts=derived[2],
            derived_world_models=derived[3],
            derived_strategies=derived[4],
        )
    finally:
        runtime.close()


def _record_transfer_trials(
    runtime: V7Runtime,
    *,
    concept_ids,
    target_game: str,
    success: bool,
    context: int,
    action: int,
    outcome: int,
    source_global_step: int,
) -> int:
    records: list[TransferTrialRecord] = []
    for concept_id in tuple(sorted(set(concept_ids), key=int)):
        source_games = tuple(
            game for game in runtime.lifecycle_evidence.provenance_source_games(concept_id)
            if game != target_game
        )
        if not source_games:
            continue
        records.append(
            TransferTrialRecord(
                memory_id=concept_id,
                generation_id=int(runtime.writer.mutable_generation_id),
                source_game=source_games[0],
                target_game=target_game,
                success=bool(success),
                score=1.0 if success else 0.0,
                payload={
                    "context_signature": int(context),
                    "action_id": int(action),
                    "outcome_signature": int(outcome),
                    "source_global_step": int(source_global_step),
                    "source_games": list(source_games),
                },
            )
        )
    return runtime.lifecycle_evidence.append_transfer_trials(records)


def _accumulate_derived(values: list[int], runtime: V7Runtime) -> None:
    stats = runtime.last_derivation_stats
    values[0] += stats.families
    values[1] += stats.roles
    values[2] += stats.concepts
    values[3] += stats.world_models
    values[4] += stats.strategies
