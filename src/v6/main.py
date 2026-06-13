from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Any

from v6.contingency.context_builder import ContextBuilder
from v6.contingency.contingency_learner import ContingencyLearner
from v6.delta.delta_extractor import DeltaExtractor
from v6.environment.env_interface import Environment
from v6.evaluation.metrics import MetricsSnapshot, compute_metrics
from v6.graph.graph_manager import GraphManager
from v6.memory.contingency_store import ContingencyStore
from v6.memory.interaction_store import Interaction, InteractionStore
from v6.memory.transformation_store import TransformationStore
from v6.prediction.predictor import Predictor
from v6.transformation.transformation_clusterer import TransformationClusterer


@dataclass(frozen=True)
class V6Config:
    database_path: str = ":memory:"
    recluster_every: int = 100
    min_cluster_size: int = 5
    context_length: int = 3
    contingency_support_threshold: int = 20
    contingency_confidence_threshold: float = 0.8
    random_seed: int | None = None
    database_commit_every: int = 1


@dataclass(frozen=True)
class StepResult:
    interaction_id: int
    delta_id: int
    action: int
    predicted_family: int | None
    actual_family: int | None
    prediction_error: int | None
    prediction_context_level: int | None = None


class V6System:
    def __init__(self, env: Environment, config: V6Config | None = None, action_sampler: Any | None = None) -> None:
        self.env = env
        self.config = config or V6Config()
        self.rng = random.Random(self.config.random_seed)
        self.action_sampler = action_sampler
        self.connection = sqlite3.connect(self.config.database_path)
        self._auto_commit = int(self.config.database_commit_every) <= 1
        self.interactions = InteractionStore(self.connection, auto_commit=self._auto_commit)
        self.transformations = TransformationStore(self.connection, auto_commit=self._auto_commit)
        self.contingency_store = ContingencyStore(self.connection, auto_commit=self._auto_commit)
        self.delta_extractor = DeltaExtractor()
        self.clusterer = TransformationClusterer(
            min_cluster_size=self.config.min_cluster_size,
            recluster_every=self.config.recluster_every,
        )
        self.context_builder = ContextBuilder(context_length=self.config.context_length)
        self.contingency_learner = ContingencyLearner(
            support_threshold=self.config.contingency_support_threshold,
            confidence_threshold=self.config.contingency_confidence_threshold,
        )
        self.predictor = Predictor(self.contingency_learner)
        self.graph = GraphManager()
        self.episode_id = 0

    def choose_action(self) -> int:
        actions = self.env.available_actions()
        if not actions:
            raise ValueError("environment returned no available actions")
        if self.action_sampler is not None:
            return int(self.action_sampler.choose_action(self, actions))
        return int(self.rng.choice(actions))

    def run_step(self) -> StepResult:
        for _attempt in range(100):
            if self.action_sampler is not None and hasattr(self.action_sampler, "before_step"):
                self.action_sampler.before_step(self)
            observation_before = self.env.observe()
            action = self.choose_action()
            context_signatures = self.context_builder.multi_scale_signatures(action, max_level=self.config.context_length)
            selected_contingency = self.contingency_learner.best_stable_for_action(context_signatures, action)
            prediction_context_level = None if selected_contingency is None else int(selected_contingency.context_level)
            predicted_family = self.predictor.predict_multi_scale(context_signatures, action)
            observation_after = self.env.step(action)
            if not bool(getattr(self.env, "last_step_was_reset_boundary", False)):
                break
            self.episode_id += 1
        else:
            raise RuntimeError("environment produced too many reset boundaries without an observable transition")

        delta_id = self.transformations.next_delta_id()
        delta = self.delta_extractor.extract(observation_before, observation_after, delta_id=delta_id)
        self.transformations.add_delta(delta)

        interaction_id = self.interactions.next_id()
        interaction = Interaction(
            id=interaction_id,
            timestamp=interaction_id,
            observation_before=observation_before,
            action=action,
            observation_after=observation_after,
            delta_id=delta.id,
        )
        self.interactions.add(interaction)
        self.graph.add_interaction(interaction)
        self.graph.add_delta(interaction.id, delta)

        interaction_count = self.interactions.count()
        clustered = False
        if interaction_count % self.config.recluster_every == 0:
            clustered = self.clusterer.maybe_recluster(
                self.transformations.all_deltas(),
                interaction_count=interaction_count,
            )
        if clustered:
            self.transformations.replace_families(self.clusterer.families)
            self.graph.replace_families(self.clusterer.families, self.clusterer.delta_to_family)

        actual_family = self.clusterer.family_for_delta(delta.id)
        if actual_family is None and self.clusterer.families:
            actual_family = self.clusterer.nearest_family(delta)

        prediction_error = self.contingency_store.add_prediction_result(
            interaction_id=interaction.id,
            context_level=prediction_context_level,
            context_signature=context_signatures.get(
                prediction_context_level,
                context_signatures.get(self.config.context_length, self.context_builder.signature()),
            ),
            action=action,
            predicted_family=predicted_family,
            actual_family=actual_family,
            episode_id=self.episode_id,
        )

        if actual_family is not None:
            contingency = self.contingency_learner.update_multi_scale(context_signatures, action, actual_family)
            if contingency is not None:
                for stable in self.contingency_learner.stable_contingencies():
                    self.contingency_store.upsert_contingency(stable)
                    self.graph.add_contingency(stable)
            self.context_builder.update(actual_family, action)

        if self.action_sampler is not None and hasattr(self.action_sampler, "record_result"):
            self.action_sampler.record_result(
                action=action,
                delta=delta,
                actual_family=actual_family,
                predicted_family=predicted_family,
                prediction_error=prediction_error,
                reset_boundary=bool(getattr(self.env, "last_step_was_reset_boundary", False)),
            )

        self._commit_if_needed(interaction.id)

        return StepResult(
            interaction_id=interaction.id,
            delta_id=delta.id,
            action=action,
            predicted_family=predicted_family,
            actual_family=actual_family,
            prediction_error=prediction_error,
            prediction_context_level=prediction_context_level,
        )

    def run(self, steps: int | None = None) -> list[StepResult]:
        results: list[StepResult] = []
        if steps is None:
            while True:
                results.append(self.run_step())
        for _ in range(int(steps)):
            results.append(self.run_step())
        return results

    def metrics(self) -> MetricsSnapshot:
        return compute_metrics(
            families=list(self.clusterer.families.values()),
            contingencies=self.contingency_learner.stable_contingencies(),
            connection=self.connection,
        )

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def _commit_if_needed(self, interaction_id: int) -> None:
        if self._auto_commit:
            return
        interval = max(1, int(self.config.database_commit_every))
        if int(interaction_id) % interval == 0:
            self.connection.commit()


def run_forever(env: Environment, config: V6Config | None = None) -> None:
    system = V6System(env=env, config=config)
    system.run(steps=None)
