from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v6.carrier_emergence import CarrierEmergenceTracker, extract_carrier_signature
from v6.contingency.context_builder import ContextBuilder
from v6.contingency.contingency_learner import ContingencyLearner
from v6.context.contradiction_tracker import ContextContradictionTracker
from v6.delta.delta_extractor import DeltaExtractor
from v6.efficiency_metrics import EfficiencyTracker
from v6.environment.env_interface import Environment
from v6.evaluation.metrics import MetricsSnapshot, compute_metrics
from v6.graph.graph_manager import GraphManager
from v6.interaction_significance import compute_interaction_significance
from v6.memory_lifecycle import MemoryLifecycleManager
from v6.memory.contingency_store import ContingencyStore
from v6.memory.compact_memory import fold_live_system_into_compact_memory
from v6.memory.compact_memory_restore import load_compact_memory_into_system
from v6.memory.interaction_store import Interaction, InteractionStore
from v6.memory.transformation_store import TransformationStore
from v6.prediction.predictor import Predictor
from v6.transformation.transformation_clusterer import TransformationClusterer


@dataclass(frozen=True)
class V6Config:
    database_path: str = ":memory:"
    memory_input_dir: str | None = None
    memory_output_dir: str | None = None
    restore_compact_memory: bool = False
    persist_compact_memory_on_close: bool = False
    global_step_offset: int = 0
    recluster_every: int = 100
    min_cluster_size: int = 5
    context_length: int = 3
    contingency_support_threshold: int = 20
    contingency_confidence_threshold: float = 0.8
    random_seed: int | None = None
    database_commit_every: int = 1
    context_contradiction_min_confidence: float = 0.5
    context_contradiction_min_repeats: int = 2
    max_context_depth: int | None = None
    adaptive_context_expansion: bool = False
    carrier_min_support: int = 3
    carrier_min_distinct_contexts: int = 2
    carrier_min_prediction_lift: float = 0.05
    carrier_min_compression_gain: float = 0.01
    memory_max_active_records: int = 50_000
    memory_replay_queue_size: int = 1_000
    memory_protect_isf_threshold: float = 0.70
    memory_forget_isf_threshold: float = 0.10
    memory_min_records_before_forgetting: int = 1_000
    efficiency_action_cost_default: float = 1.0
    efficiency_recent_window_size: int = 100


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
        self.base_context_depth = int(self.config.context_length)
        self.max_context_depth = int(self.config.max_context_depth or self.config.context_length)
        self.max_context_depth = max(self.base_context_depth, self.max_context_depth)
        self.context_builder = ContextBuilder(context_length=self.max_context_depth)
        self._adaptive_action_depths: dict[int, int] = {}
        self._adaptive_context_expansion_count = 0
        self._adaptive_context_expansion_events: list[dict] = []
        self.contingency_learner = ContingencyLearner(
            support_threshold=self.config.contingency_support_threshold,
            confidence_threshold=self.config.contingency_confidence_threshold,
        )
        self.predictor = Predictor(self.contingency_learner)
        self.graph = GraphManager()
        self.context_contradictions = ContextContradictionTracker(
            min_confidence=self.config.context_contradiction_min_confidence,
            min_repeats_for_expansion=self.config.context_contradiction_min_repeats,
        )
        self.carrier_tracker = CarrierEmergenceTracker(
            min_support=self.config.carrier_min_support,
            min_distinct_contexts=self.config.carrier_min_distinct_contexts,
            min_prediction_lift=self.config.carrier_min_prediction_lift,
            min_compression_gain=self.config.carrier_min_compression_gain,
        )
        self.memory_lifecycle = MemoryLifecycleManager(
            max_active_records=self.config.memory_max_active_records,
            replay_queue_size=self.config.memory_replay_queue_size,
            protect_isf_threshold=self.config.memory_protect_isf_threshold,
            forget_isf_threshold=self.config.memory_forget_isf_threshold,
            min_records_before_forgetting=self.config.memory_min_records_before_forgetting,
        )
        self.efficiency_tracker = EfficiencyTracker(
            action_cost_default=self.config.efficiency_action_cost_default,
            recent_window_size=self.config.efficiency_recent_window_size,
        )
        self.episode_id = 0
        self.step_count = 0
        self._isf_counts: dict[str, int] = {}
        self._current_level_interaction_ids: list[int] = []
        self.compact_memory_restore_summary: dict[str, Any] = {
            "stable_contingencies_restored": 0,
            "transformation_families_restored": 0,
            "family_members_restored": 0,
            "carrier_candidates_restored": 0,
            "replay_candidates_restored": 0,
            "graph_nodes_restored": 0,
            "graph_edges_restored": 0,
            "memory_summary_loaded": False,
            "restore_warnings": [],
        }
        if bool(self.config.restore_compact_memory) and self.config.memory_input_dir:
            self.compact_memory_restore_summary = load_compact_memory_into_system(self, Path(self.config.memory_input_dir))

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
            active_context_depth = self._context_depth_for_action(action)
            context_signatures = self.context_builder.multi_scale_signatures(action, max_level=active_context_depth)
            selected_contingency = self.contingency_learner.best_stable_for_action(context_signatures, action)
            prediction_context_level = None if selected_contingency is None else int(selected_contingency.context_level)
            predicted_family = self.predictor.predict_multi_scale(context_signatures, action)
            prediction_confidence = self._prediction_confidence(
                context_signatures=context_signatures,
                action=action,
                predicted_family=predicted_family,
                selected_contingency=selected_contingency,
            )
            observation_after = self.env.step(action)
            outcome_state_after_step = str(getattr(self.env, "last_outcome_state", "NOT_FINISHED") or "NOT_FINISHED")
            if (
                not bool(getattr(self.env, "last_step_was_reset_boundary", False))
                or outcome_state_after_step in {"WIN", "GAME_OVER"}
            ):
                break
            self.episode_id += 1
        else:
            raise RuntimeError("environment produced too many reset boundaries without an observable transition")

        outcome_state = str(getattr(self.env, "last_outcome_state", "NOT_FINISHED") or "NOT_FINISHED")
        outcome_polarity = str(getattr(self.env, "last_outcome_polarity", "neutral") or "neutral")
        level_completed_event = bool(getattr(self.env, "level_completed_event", False))
        terminal_reset_surrogate = (
            bool(getattr(self.env, "last_step_was_reset_boundary", False))
            and outcome_state in {"WIN", "GAME_OVER"}
        )
        observation_after_for_delta = observation_before.copy() if terminal_reset_surrogate else observation_after

        delta_id = self.transformations.next_delta_id()
        delta = self.delta_extractor.extract(observation_before, observation_after_for_delta, delta_id=delta_id)
        self.transformations.add_delta(delta)

        interaction_id = self.interactions.next_id()
        interaction = Interaction(
            id=interaction_id,
            timestamp=interaction_id,
            global_step=int(self.config.global_step_offset) + int(interaction_id),
            observation_before=observation_before,
            action=action,
            observation_after=observation_after_for_delta,
            delta_id=delta.id,
        )
        self.step_count = int(interaction_id)
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
        prediction_correct = None
        if predicted_family is not None and actual_family is not None:
            prediction_correct = int(predicted_family) == int(actual_family)
        prediction_context_signature = context_signatures.get(
            prediction_context_level,
            context_signatures.get(active_context_depth),
        )
        if prediction_context_signature is None:
            prediction_context_signature = context_signatures.get(0, (int(action),))
        serialized_context_signature = json.dumps(list(prediction_context_signature))
        action_signature = f"a{int(action)}"
        actual_family_id = None if actual_family is None else str(actual_family)
        stable_delta_key = self._stable_delta_key(delta, outcome_state=outcome_state, level_completed_event=level_completed_event)
        memory_counts = self._build_isf_memory_counts(
            delta_id=stable_delta_key,
            actual_family_id=actual_family_id,
            context_signature=serialized_context_signature,
            outcome_state=outcome_state,
            level_completed_event=level_completed_event,
        )
        reward = getattr(self.env, "last_reward", None)
        terminated = bool(getattr(self.env, "last_terminal_state", None))
        truncated = bool(getattr(self.env, "last_step_was_reset_boundary", False))
        isf_graph_counts = {
            "new_contingency": 0,
            "new_graph_edge": 0,
        }
        prediction_error: int | None

        if actual_family is not None:
            contingency = self.contingency_learner.update_multi_scale(context_signatures, action, actual_family)
            if contingency is not None:
                isf_graph_counts["new_contingency"] = 1
                isf_graph_counts["new_graph_edge"] = 1
                for stable in self.contingency_learner.stable_contingencies():
                    self.contingency_store.upsert_contingency(stable)
                    self.graph.add_contingency(stable)
            self.context_builder.update(actual_family, action)

        isf_score = compute_interaction_significance(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            prediction_correct=prediction_correct,
            prediction_confidence=prediction_confidence,
            actual_family_id=actual_family_id,
            delta_id=stable_delta_key,
            context_signature=serialized_context_signature,
            memory_counts=memory_counts,
            graph_counts=isf_graph_counts,
            outcome_state=outcome_state,
            outcome_polarity=outcome_polarity,
            level_completed_event=level_completed_event,
        )
        carrier_signature, carrier_source = extract_carrier_signature(
            before_observation=observation_before,
            after_observation=observation_after_for_delta,
            delta=delta,
            context_signature=serialized_context_signature,
            action_signature=action_signature,
        )
        carrier_event = self.carrier_tracker.record_interaction(
            interaction_id=str(interaction.id),
            carrier_signature=carrier_signature,
            context_signature=serialized_context_signature,
            action_signature=action_signature,
            family_id=actual_family_id,
            delta_signature=getattr(delta, "signature", None) or getattr(delta, "delta_id", None) or str(delta),
            prediction_correct=prediction_correct,
            carrier_source=carrier_source,
        )
        carrier_stats = (
            self.carrier_tracker.stats_for_carrier(carrier_signature)
            if carrier_signature is not None
            else {
                "carrier_signature": None,
                "carrier_source": carrier_source,
                "carrier_support_count": None,
                "carrier_distinct_family_count": None,
                "carrier_distinct_context_count": None,
                "carrier_prediction_lift": 0.0,
                "carrier_compression_gain": 0.0,
                "carrier_status": "candidate",
            }
        )
        efficiency_event = self.efficiency_tracker.record_interaction(
            interaction_id=str(interaction.id),
            before_observation=observation_before,
            after_observation=observation_after_for_delta,
            delta=delta,
            context_signature=serialized_context_signature,
            action_signature=action_signature,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            # Future-option deltas are post-run diagnostics only here; do not use future horizon data during sampling.
            future_option_delta=None,
        )
        contradiction_event = self.context_contradictions.record_prediction_result(
            interaction_id=str(interaction.id),
            context_signature=serialized_context_signature,
            action_signature=action_signature,
            predicted_family_id=None if predicted_family is None else str(predicted_family),
            actual_family_id=actual_family_id,
            prediction_correct=prediction_correct,
            prediction_confidence=prediction_confidence,
            context_depth=int(active_context_depth),
            max_context_depth=self.max_context_depth,
        )
        context_expansion_suggested = False
        suggested_context_depth: int | None = None
        context_contradiction_reason: str | None = None
        context_contradiction_key: str | None = None
        adaptive_context_depth_after = active_context_depth
        adaptive_context_expansion_applied = False
        if contradiction_event is not None:
            context_contradiction_key = contradiction_event.contradiction_key
            suggested_context_depth = int(contradiction_event.suggested_context_depth)
            context_contradiction_reason = contradiction_event.reason
            context_expansion_suggested = self.context_contradictions.should_expand_context(
                serialized_context_signature,
                action_signature,
            )
            adaptive_context_depth_after, adaptive_context_expansion_applied = self._apply_adaptive_context_expansion(
                action=action,
                old_depth=active_context_depth,
                suggested_depth=suggested_context_depth,
                reason=context_contradiction_reason,
                contradiction_key=context_contradiction_key,
            )
            self.graph.add_contradicts(
                f"Context:{serialized_context_signature}",
                f"TransformationFamily:{int(actual_family)}",
                weight=float(prediction_confidence or 1.0),
                evidence=contradiction_event.to_dict(),
            )
        memory_record = self.memory_lifecycle.register_interaction(
            interaction_id=str(interaction.id),
            family_id=actual_family_id,
            context_signature=serialized_context_signature,
            action_signature=action_signature,
            carrier_signature=carrier_signature,
            isf_total=isf_score.total,
            prediction_error=isf_score.prediction_error,
            learning_value=isf_score.learning_value,
            transfer_potential=isf_score.transfer_potential,
            explanatory_potential=isf_score.explanatory_potential,
            context_contradiction=bool(contradiction_event is not None),
            timestamp_step=int(self.step_count),
        )
        replay_candidate = self.memory_lifecycle.replay_candidates.get(str(interaction.id))
        prediction_error = self.contingency_store.add_prediction_result(
            interaction_id=interaction.id,
            global_step=int(self.config.global_step_offset) + int(interaction.id),
            context_level=prediction_context_level,
            context_signature=prediction_context_signature,
            action=action,
            predicted_family=predicted_family,
            actual_family=actual_family,
            episode_id=self.episode_id,
            isf_version=isf_score.version,
            isf_total=isf_score.total,
            isf_survival_impact=isf_score.survival_impact,
            isf_prediction_error=isf_score.prediction_error,
            isf_learning_value=isf_score.learning_value,
            isf_transfer_potential=isf_score.transfer_potential,
            isf_explanatory_potential=isf_score.explanatory_potential,
            isf_weights_json=json.dumps(isf_score.weights, sort_keys=True),
            context_contradiction=contradiction_event is not None,
            context_contradiction_key=context_contradiction_key,
            context_expansion_suggested=context_expansion_suggested,
            suggested_context_depth=suggested_context_depth,
            context_contradiction_reason=context_contradiction_reason,
            context_depth_used=active_context_depth,
            adaptive_context_expansion_applied=adaptive_context_expansion_applied,
            adaptive_context_depth_after=adaptive_context_depth_after,
            carrier_signature=carrier_signature,
            carrier_source=carrier_source,
            carrier_event_recorded=carrier_event is not None,
            carrier_support_count=carrier_stats["carrier_support_count"],
            carrier_distinct_family_count=carrier_stats["carrier_distinct_family_count"],
            carrier_distinct_context_count=carrier_stats["carrier_distinct_context_count"],
            memory_status=memory_record.status,
            memory_retention_reason=memory_record.retention_reason,
            memory_replay_priority=0.0 if replay_candidate is None else float(replay_candidate.replay_priority),
            memory_replay_candidate=replay_candidate is not None,
            memory_replay_count=int(memory_record.replay_count),
            efficiency_action_cost=float(efficiency_event.action_cost),
            efficiency_cumulative_cost=float(efficiency_event.cumulative_cost),
            efficiency_repeated_state=bool(efficiency_event.repeated_state),
            efficiency_repeated_context_action=bool(efficiency_event.repeated_context_action),
            efficiency_no_effect_action=bool(efficiency_event.no_effect_action),
            efficiency_terminal_outcome=bool(efficiency_event.terminal_outcome),
            efficiency_outcome_signature=efficiency_event.outcome_signature,
            efficiency_best_known_cost_for_outcome=efficiency_event.best_known_cost_for_outcome,
            efficiency_normalized_solve_efficiency=efficiency_event.normalized_solve_efficiency,
            efficiency_equivalent_outcome_cost_gap=efficiency_event.equivalent_outcome_cost_gap,
            efficiency_future_option_gain_per_cost=efficiency_event.future_option_gain_per_cost,
            outcome_state=outcome_state,
            outcome_polarity=outcome_polarity,
            level_completed_event=level_completed_event,
        )
        interaction = Interaction(
            id=interaction_id,
            timestamp=interaction_id,
            global_step=int(self.config.global_step_offset) + int(interaction_id),
            observation_before=observation_before,
            action=action,
            observation_after=observation_after_for_delta,
            delta_id=delta.id,
            isf_version=isf_score.version,
            isf_total=isf_score.total,
            isf_survival_impact=isf_score.survival_impact,
            isf_prediction_error=isf_score.prediction_error,
            isf_learning_value=isf_score.learning_value,
            isf_transfer_potential=isf_score.transfer_potential,
            isf_explanatory_potential=isf_score.explanatory_potential,
            isf_weights_json=json.dumps(isf_score.weights, sort_keys=True),
            carrier_signature=carrier_signature,
            carrier_source=carrier_source,
            carrier_event_recorded=carrier_event is not None,
            carrier_support_count=carrier_stats["carrier_support_count"],
            carrier_distinct_family_count=carrier_stats["carrier_distinct_family_count"],
            carrier_distinct_context_count=carrier_stats["carrier_distinct_context_count"],
            memory_status=memory_record.status,
            memory_retention_reason=memory_record.retention_reason,
            memory_replay_priority=0.0 if replay_candidate is None else float(replay_candidate.replay_priority),
            memory_replay_candidate=replay_candidate is not None,
            memory_replay_count=int(memory_record.replay_count),
            context_depth_used=active_context_depth,
            adaptive_context_expansion_applied=adaptive_context_expansion_applied,
            adaptive_context_depth_after=adaptive_context_depth_after,
            efficiency_action_cost=float(efficiency_event.action_cost),
            efficiency_cumulative_cost=float(efficiency_event.cumulative_cost),
            efficiency_repeated_state=bool(efficiency_event.repeated_state),
            efficiency_repeated_context_action=bool(efficiency_event.repeated_context_action),
            efficiency_no_effect_action=bool(efficiency_event.no_effect_action),
            efficiency_terminal_outcome=bool(efficiency_event.terminal_outcome),
            efficiency_outcome_signature=efficiency_event.outcome_signature,
            efficiency_best_known_cost_for_outcome=efficiency_event.best_known_cost_for_outcome,
            efficiency_normalized_solve_efficiency=efficiency_event.normalized_solve_efficiency,
            efficiency_equivalent_outcome_cost_gap=efficiency_event.equivalent_outcome_cost_gap,
            efficiency_future_option_gain_per_cost=efficiency_event.future_option_gain_per_cost,
            outcome_state=outcome_state,
            outcome_polarity=outcome_polarity,
            level_completed_event=level_completed_event,
        )
        self.connection.execute(
            """
            UPDATE interactions
            SET
                isf_version = ?,
                isf_total = ?,
                isf_survival_impact = ?,
                isf_prediction_error = ?,
                isf_learning_value = ?,
                isf_transfer_potential = ?,
                isf_explanatory_potential = ?,
                isf_weights_json = ?,
                carrier_signature = ?,
                carrier_source = ?,
                carrier_event_recorded = ?,
                carrier_support_count = ?,
                carrier_distinct_family_count = ?,
                carrier_distinct_context_count = ?,
                memory_status = ?,
                memory_retention_reason = ?,
                memory_replay_priority = ?,
                memory_replay_candidate = ?,
                memory_replay_count = ?,
                context_depth_used = ?,
                adaptive_context_expansion_applied = ?,
                adaptive_context_depth_after = ?,
                efficiency_action_cost = ?,
                efficiency_cumulative_cost = ?,
                efficiency_repeated_state = ?,
                efficiency_repeated_context_action = ?,
                efficiency_no_effect_action = ?,
                efficiency_terminal_outcome = ?,
                efficiency_outcome_signature = ?,
                efficiency_best_known_cost_for_outcome = ?,
                efficiency_normalized_solve_efficiency = ?,
                efficiency_equivalent_outcome_cost_gap = ?,
                efficiency_future_option_gain_per_cost = ?,
                outcome_state = ?,
                outcome_polarity = ?,
                level_completed_event = ?
            WHERE id = ?
            """,
            (
                interaction.isf_version,
                interaction.isf_total,
                interaction.isf_survival_impact,
                interaction.isf_prediction_error,
                interaction.isf_learning_value,
                interaction.isf_transfer_potential,
                interaction.isf_explanatory_potential,
                interaction.isf_weights_json,
                interaction.carrier_signature,
                interaction.carrier_source,
                int(bool(interaction.carrier_event_recorded)),
                interaction.carrier_support_count,
                interaction.carrier_distinct_family_count,
                interaction.carrier_distinct_context_count,
                interaction.memory_status,
                interaction.memory_retention_reason,
                interaction.memory_replay_priority,
                int(bool(interaction.memory_replay_candidate)),
                interaction.memory_replay_count,
                interaction.context_depth_used,
                int(bool(interaction.adaptive_context_expansion_applied)),
                interaction.adaptive_context_depth_after,
                interaction.efficiency_action_cost,
                interaction.efficiency_cumulative_cost,
                int(bool(interaction.efficiency_repeated_state)),
                int(bool(interaction.efficiency_repeated_context_action)),
                int(bool(interaction.efficiency_no_effect_action)),
                int(bool(interaction.efficiency_terminal_outcome)),
                interaction.efficiency_outcome_signature,
                interaction.efficiency_best_known_cost_for_outcome,
                interaction.efficiency_normalized_solve_efficiency,
                interaction.efficiency_equivalent_outcome_cost_gap,
                interaction.efficiency_future_option_gain_per_cost,
                interaction.outcome_state,
                interaction.outcome_polarity,
                int(bool(interaction.level_completed_event)),
                interaction.id,
            ),
        )
        self._update_isf_counts(
            delta_id=stable_delta_key,
            actual_family_id=actual_family_id,
            context_signature=serialized_context_signature,
            outcome_state=outcome_state,
            level_completed_event=level_completed_event,
        )
        self._add_carrier_edges(
            interaction_id=interaction.id,
            carrier_signature=carrier_signature,
            actual_family_id=actual_family_id,
            carrier_status=str(carrier_stats["carrier_status"]),
        )
        self._add_efficiency_edges(
            interaction_id=interaction.id,
            context_signature=serialized_context_signature,
            action_signature=action_signature,
            efficiency_event=efficiency_event,
        )
        self._add_prediction_explanation_edges(
            interaction_id=interaction.id,
            prediction_correct=prediction_correct,
            prediction_confidence=prediction_confidence,
            predicted_family=predicted_family,
            actual_family=actual_family,
            actual_context_signature=prediction_context_signature,
            selected_contingency=selected_contingency,
        )
        if self._auto_commit:
            self.connection.commit()

        self._current_level_interaction_ids.append(int(interaction.id))

        if level_completed_event:
            self._apply_post_factum_level_completion_credit(completion_interaction_id=interaction.id)
        elif outcome_state in {"GAME_OVER", "WIN"}:
            self._current_level_interaction_ids.clear()

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

    def _context_depth_for_action(self, action: int) -> int:
        if not bool(self.config.adaptive_context_expansion):
            depth = self.base_context_depth
        else:
            depth = int(self._adaptive_action_depths.get(int(action), self.base_context_depth))
        return max(0, min(self.max_context_depth, int(depth)))

    def _apply_adaptive_context_expansion(
        self,
        *,
        action: int,
        old_depth: int,
        suggested_depth: int | None,
        reason: str | None,
        contradiction_key: str | None,
    ) -> tuple[int, bool]:
        if not bool(self.config.adaptive_context_expansion):
            return int(old_depth), False
        if suggested_depth is None:
            return int(old_depth), False
        new_depth = min(self.max_context_depth, max(int(old_depth), int(suggested_depth)))
        if new_depth <= int(old_depth):
            return int(old_depth), False
        self._adaptive_action_depths[int(action)] = int(new_depth)
        self._adaptive_context_expansion_count += 1
        self._adaptive_context_expansion_events.append(
            {
                "action": int(action),
                "old_depth": int(old_depth),
                "new_depth": int(new_depth),
                "reason": reason,
                "contradiction_key": contradiction_key,
            }
        )
        return int(new_depth), True

    def adaptive_context_summary(self) -> dict:
        action_depths = {int(action): int(depth) for action, depth in sorted(self._adaptive_action_depths.items())}
        return {
            "adaptive_context_expansion_enabled": bool(self.config.adaptive_context_expansion),
            "base_context_depth": int(self.base_context_depth),
            "max_context_depth": int(self.max_context_depth),
            "adaptive_context_expansion_count": int(self._adaptive_context_expansion_count),
            "adaptive_context_active_action_count": len(action_depths),
            "adaptive_context_max_depth_reached": max([self.base_context_depth, *action_depths.values()]) if action_depths else int(self.base_context_depth),
            "adaptive_context_action_depths": action_depths,
        }

    def close(self) -> None:
        if bool(self.config.persist_compact_memory_on_close) and self.config.memory_output_dir:
            fold_live_system_into_compact_memory(self, Path(self.config.memory_output_dir))
        self.connection.commit()
        self.connection.close()

    def _commit_if_needed(self, interaction_id: int) -> None:
        if self._auto_commit:
            return
        interval = max(1, int(self.config.database_commit_every))
        if int(interaction_id) % interval == 0:
            self.connection.commit()

    def _prediction_confidence(
        self,
        *,
        context_signatures: dict[int, tuple],
        action: int,
        predicted_family: int | None,
        selected_contingency: Any | None,
    ) -> float | None:
        if predicted_family is None:
            return None
        if selected_contingency is not None and int(selected_contingency.transformation_family) == int(predicted_family):
            return float(selected_contingency.confidence)
        for level in sorted(context_signatures, reverse=True):
            distribution = self.contingency_learner.distribution_at_level(level, context_signatures[level], action)
            if not distribution:
                continue
            if int(predicted_family) in distribution:
                return float(distribution[int(predicted_family)])
        return None

    def _apply_post_factum_level_completion_credit(self, *, completion_interaction_id: int) -> None:
        ids = list(self._current_level_interaction_ids)
        if int(completion_interaction_id) not in ids:
            ids.append(int(completion_interaction_id))
        horizon = len(ids)
        if horizon <= 0:
            return
        reason = "levels_completed_increment"
        for distance_from_completion, interaction_id in enumerate(reversed(ids)):
            credit = 1.0 / (1.0 + float(distance_from_completion))
            self.connection.execute(
                """
                UPDATE interactions
                SET
                    post_factum_level_completion_credit = MAX(COALESCE(post_factum_level_completion_credit, 0.0), ?),
                    post_factum_level_completion_decay = ?,
                    post_factum_level_completion_step = ?,
                    post_factum_credit_reason = ?
                WHERE id = ?
                """,
                (
                    float(credit),
                    float(credit),
                    int(completion_interaction_id),
                    reason,
                    int(interaction_id),
                ),
            )
            self.connection.execute(
                """
                UPDATE prediction_results
                SET
                    post_factum_level_completion_credit = MAX(COALESCE(post_factum_level_completion_credit, 0.0), ?),
                    post_factum_level_completion_decay = ?,
                    post_factum_level_completion_step = ?,
                    post_factum_credit_reason = ?
                WHERE interaction_id = ?
                """,
                (
                    float(credit),
                    float(credit),
                    int(completion_interaction_id),
                    reason,
                    int(interaction_id),
                ),
            )
            self.memory_lifecycle.apply_post_factum_credit(
                str(interaction_id),
                learning_credit=float(credit),
                reason="post_factum_level_completion",
            )
        self._current_level_interaction_ids = []

    def _build_isf_memory_counts(
        self,
        *,
        delta_id: str | None,
        actual_family_id: str | None,
        context_signature: str | None,
        outcome_state: str | None,
        level_completed_event: bool = False,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        if delta_id:
            counts[f"delta_id:{delta_id}"] = int(self._isf_counts.get(f"delta_id:{delta_id}", 0))
        if actual_family_id:
            counts[f"actual_family_id:{actual_family_id}"] = int(self._isf_counts.get(f"actual_family_id:{actual_family_id}", 0))
        if context_signature:
            counts[f"context_signature:{context_signature}"] = int(self._isf_counts.get(f"context_signature:{context_signature}", 0))
        if context_signature and actual_family_id:
            key = f"context_family:{context_signature}|{actual_family_id}"
            counts[key] = int(self._isf_counts.get(key, 0))
        if outcome_state:
            key = f"outcome_state:{outcome_state}"
            counts[key] = int(self._isf_counts.get(key, 0))
        if context_signature and outcome_state:
            key = f"context_outcome:{context_signature}|{outcome_state}"
            counts[key] = int(self._isf_counts.get(key, 0))
        if level_completed_event:
            counts["level_completed_event:true"] = int(self._isf_counts.get("level_completed_event:true", 0))
        if context_signature and level_completed_event:
            key = f"context_level_completed:{context_signature}|true"
            counts[key] = int(self._isf_counts.get(key, 0))
        return counts

    def _update_isf_counts(
        self,
        *,
        delta_id: str | None,
        actual_family_id: str | None,
        context_signature: str | None,
        outcome_state: str | None,
        level_completed_event: bool = False,
    ) -> None:
        if delta_id:
            key = f"delta_id:{delta_id}"
            self._isf_counts[key] = int(self._isf_counts.get(key, 0)) + 1
        if actual_family_id:
            key = f"actual_family_id:{actual_family_id}"
            self._isf_counts[key] = int(self._isf_counts.get(key, 0)) + 1
        if context_signature:
            key = f"context_signature:{context_signature}"
            self._isf_counts[key] = int(self._isf_counts.get(key, 0)) + 1
        if context_signature and actual_family_id:
            key = f"context_family:{context_signature}|{actual_family_id}"
            self._isf_counts[key] = int(self._isf_counts.get(key, 0)) + 1
        if outcome_state:
            key = f"outcome_state:{outcome_state}"
            self._isf_counts[key] = int(self._isf_counts.get(key, 0)) + 1
        if context_signature and outcome_state:
            key = f"context_outcome:{context_signature}|{outcome_state}"
            self._isf_counts[key] = int(self._isf_counts.get(key, 0)) + 1
        if level_completed_event:
            key = "level_completed_event:true"
            self._isf_counts[key] = int(self._isf_counts.get(key, 0)) + 1
        if context_signature and level_completed_event:
            key = f"context_level_completed:{context_signature}|true"
            self._isf_counts[key] = int(self._isf_counts.get(key, 0)) + 1

    def _stable_delta_key(self, delta: Any, outcome_state: str, level_completed_event: bool) -> str:
        changed_cells = getattr(delta, "changed_cells", None)
        dx = getattr(delta, "dx", None)
        dy = getattr(delta, "dy", None)
        return (
            f"cells={changed_cells}|dx={dx}|dy={dy}|state={outcome_state}|"
            f"level_completed={int(bool(level_completed_event))}"
        )

    def _add_prediction_explanation_edges(
        self,
        *,
        interaction_id: int,
        prediction_correct: bool | None,
        prediction_confidence: float | None,
        predicted_family: int | None,
        actual_family: int | None,
        actual_context_signature: tuple,
        selected_contingency: Any | None,
    ) -> None:
        if actual_family is None:
            return
        context_node = f"Context:{json.dumps(list(actual_context_signature))}"
        family_node = f"TransformationFamily:{int(actual_family)}"
        self.graph.add_depends_on(
            family_node,
            context_node,
            weight=1.0,
            evidence={"interaction_id": int(interaction_id)},
        )
        if selected_contingency is None:
            return
        contingency_node = f"Contingency:{int(selected_contingency.id)}"
        base_evidence = {
            "interaction_id": int(interaction_id),
            "context_signature": json.dumps(list(actual_context_signature)),
            "prediction_confidence": None if prediction_confidence is None else float(prediction_confidence),
        }
        weight = float(prediction_confidence or 1.0)
        if prediction_correct and predicted_family is not None:
            self.graph.add_explains(
                contingency_node,
                family_node,
                weight=weight,
                evidence=base_evidence,
            )
            return
        if prediction_correct is False and predicted_family is not None:
            self.graph.add_contradicts(
                contingency_node,
                family_node,
                weight=weight,
                evidence={
                    **base_evidence,
                    "predicted_family_id": int(predicted_family),
                    "actual_family_id": int(actual_family),
                },
            )

    def _add_carrier_edges(
        self,
        *,
        interaction_id: int,
        carrier_signature: str | None,
        actual_family_id: str | None,
        carrier_status: str,
    ) -> None:
        if carrier_signature is None or actual_family_id is None:
            return
        family_node = f"TransformationFamily:{actual_family_id}"
        carrier_node = f"Carrier:{carrier_signature}"
        self.graph.add_depends_on(
            family_node,
            carrier_node,
            weight=1.0,
            evidence={"interaction_id": int(interaction_id)},
        )
        if carrier_status == "emergent_carrier":
            self.graph.add_explains(
                carrier_node,
                family_node,
                weight=1.0,
                evidence={"interaction_id": int(interaction_id), "carrier_status": "emergent_carrier"},
            )

    def _add_efficiency_edges(
        self,
        *,
        interaction_id: int,
        context_signature: str | None,
        action_signature: str | None,
        efficiency_event: Any,
    ) -> None:
        if bool(efficiency_event.no_effect_action) and context_signature:
            self.graph.add_blocks(
                f"Context:{context_signature}",
                f"Action:{action_signature}",
                weight=1.0,
                evidence={"interaction_id": int(interaction_id), "reason": "no_effect_action"},
            )
        if bool(efficiency_event.terminal_outcome) and efficiency_event.outcome_signature:
            self.graph.add_terminates(
                f"Interaction:{int(interaction_id)}",
                f"Outcome:{efficiency_event.outcome_signature}",
                weight=1.0,
                evidence={"cumulative_cost": float(efficiency_event.cumulative_cost)},
            )


def run_forever(env: Environment, config: V6Config | None = None) -> None:
    system = V6System(env=env, config=config)
    system.run(steps=None)
