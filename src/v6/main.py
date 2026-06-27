from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from hashlib import sha1
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
from v6.future_options import FutureOptionEstimator
from v6.memory_lifecycle import MemoryLifecycleManager
from v6.memory.contingency_store import ContingencyStore
from v6.memory.compact_memory import fold_live_system_into_compact_memory
from v6.memory.promotion_engine import MemoryPromotionConfig, MemoryPromotionEngine
from v6.memory.query_engine import MemoryQueryEngine
from v6.memory.compact_memory_restore import load_compact_memory_into_system
from v6.memory.interaction_store import Interaction, InteractionStore
from v6.memory.substrate import (
    MemoryEdge,
    MemoryNode,
    MemoryScore,
    MemorySubstrate,
    action_node_id,
    carrier_node_id,
    contingency_node_id,
    delta_node_id,
    family_node_id,
    interaction_node_id,
    observation_node_id,
    scoped_interaction_key,
    strategy_node_id,
    trajectory_node_id,
)
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
    restore_compact_graph: bool = True
    restore_compact_substrate: bool = True
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
    memory_promotion_every: int = 100
    future_options_enabled: bool = True
    future_option_depth: int = 1
    memory_query_enabled: bool = False
    memory_action_selection_enabled: bool = False
    memory_query_use_roles: bool = True
    memory_query_use_concepts: bool = True
    memory_query_use_future_options: bool = True
    memory_query_failure_penalty: float = 0.25


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
        self.memory = MemorySubstrate(self.connection, auto_commit=self._auto_commit)
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
        self.future_option_estimator = FutureOptionEstimator()
        self.promotion_engine = MemoryPromotionEngine(self.memory, MemoryPromotionConfig())
        self.memory_query = MemoryQueryEngine(
            self.memory,
            contingency_learner=self.contingency_learner,
            graph=self.graph,
        )
        self.episode_id = 0
        self.step_count = 0
        self._isf_counts: dict[str, int] = {}
        self._current_level_interaction_ids: list[int] = []
        self._last_memory_interaction_node_id: str | None = None
        self._interaction_memory_node_ids: dict[int, str] = {}
        self.compact_memory_restore_summary: dict[str, Any] = {
            "stable_contingencies_restored": 0,
            "transformation_families_restored": 0,
            "family_members_restored": 0,
            "carrier_candidates_restored": 0,
            "replay_candidates_restored": 0,
            "graph_nodes_restored": 0,
            "graph_edges_restored": 0,
            "memory_nodes_restored": 0,
            "memory_edges_restored": 0,
            "memory_evidence_restored": 0,
            "memory_scores_restored": 0,
            "memory_promotions_restored": 0,
            "memory_summary_loaded": False,
            "restore_warnings": [],
        }
        if bool(self.config.restore_compact_memory) and self.config.memory_input_dir:
            self.compact_memory_restore_summary = load_compact_memory_into_system(
                self,
                Path(self.config.memory_input_dir),
                restore_graph=bool(self.config.restore_compact_graph),
                restore_substrate=bool(self.config.restore_compact_substrate),
            )

    def choose_action(self) -> int:
        actions = self.env.available_actions()
        if not actions:
            raise ValueError("environment returned no available actions")
        if bool(self.config.memory_action_selection_enabled) and self.action_sampler is None:
            contexts_by_action: dict[int, dict[int, tuple]] = {}
            for candidate_action in actions:
                candidate_depth = self._context_depth_for_action(int(candidate_action))
                contexts_by_action[int(candidate_action)] = self.context_builder.multi_scale_signatures(
                    int(candidate_action),
                    max_level=candidate_depth,
                )
            ranked = self.memory_query.rank_actions(contexts_by_action, list(actions))
            if ranked:
                best_score = ranked[0].score
                best = [item.action for item in ranked if float(item.score) == float(best_score)]
                return int(self.rng.choice(best))
        if self.action_sampler is not None:
            return int(self.action_sampler.choose_action(self, actions))
        return int(self.rng.choice(actions))

    def run_step(self) -> StepResult:
        for _attempt in range(100):
            if self.action_sampler is not None and hasattr(self.action_sampler, "before_step"):
                self.action_sampler.before_step(self)
            observation_before = self.env.observe()
            available_actions_before = list(self.env.available_actions())
            future_option_before = None
            if bool(self.config.future_options_enabled):
                future_option_before = self.future_option_estimator.estimate_option_set(
                    observation_before,
                    depth=int(self.config.future_option_depth),
                    available_actions=available_actions_before,
                )
            action = self.choose_action()
            active_context_depth = self._context_depth_for_action(action)
            context_signatures = self.context_builder.multi_scale_signatures(action, max_level=active_context_depth)
            selected_contingency = self.contingency_learner.best_stable_for_action(context_signatures, action)
            prediction_context_level = None if selected_contingency is None else int(selected_contingency.context_level)
            if bool(self.config.memory_query_enabled):
                memory_prediction = self.memory_query.predict_family(context_signatures, action, record_query=False)
                predicted_family = memory_prediction.predicted_family
                prediction_confidence = float(memory_prediction.confidence)
            else:
                memory_prediction = None
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
        available_actions_after = list(self.env.available_actions())
        future_option_after = None
        future_option_delta = None
        if bool(self.config.future_options_enabled) and future_option_before is not None:
            future_option_after = self.future_option_estimator.estimate_option_set(
                observation_after_for_delta,
                depth=int(self.config.future_option_depth),
                available_actions=available_actions_after,
            )

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
        if future_option_before is not None and future_option_after is not None:
            future_option_delta = self.future_option_estimator.compare(future_option_before, future_option_after, interaction.id)

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
        self._write_m0_memory(
            interaction=interaction,
            observation_before=observation_before,
            observation_after=observation_after_for_delta,
            context_signature=serialized_context_signature,
            context_level=prediction_context_level if prediction_context_level is not None else active_context_depth,
        )
        if future_option_before is not None and future_option_after is not None and future_option_delta is not None:
            self._write_future_option_memory(
                interaction_id=interaction.id,
                before_option_set=future_option_before,
                after_option_set=future_option_after,
                option_delta=future_option_delta,
            )
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
            self._write_m1_m2_memory(
                interaction_id=interaction.id,
                stable_contingency=contingency,
                actual_family=actual_family,
                delta_id=delta.id,
            )

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
            future_option_delta=None if future_option_delta is None else float(future_option_delta.delta_score),
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
        self._write_carrier_memory(
            interaction_id=interaction.id,
            carrier_signature=carrier_signature,
            carrier_source=carrier_source,
            carrier_stats=carrier_stats,
            context_signature=serialized_context_signature,
            actual_family_id=actual_family_id,
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
            future_option_delta=None if future_option_delta is None else float(future_option_delta.delta_score),
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
            self._write_prediction_violation_memory(
                interaction_id=interaction.id,
                contradiction_event=contradiction_event,
                actual_family_id=actual_family_id,
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
        self._write_trajectory_and_efficiency_memory(
            interaction_id=interaction.id,
            efficiency_event=efficiency_event,
            context_signature=serialized_context_signature,
        )
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
        self.memory.upsert_score(
            MemoryScore(
                node_id=self._interaction_memory_node_id(interaction.id),
                isf_total=float(isf_score.total),
                future_option_delta=None if future_option_delta is None else float(future_option_delta.delta_score),
                replay_priority=0.0 if replay_candidate is None else float(replay_candidate.replay_priority),
                retention_status=memory_record.status,
            ),
            step=int(interaction.id),
        )
        self._write_memory_lifecycle_memory(
            interaction_id=interaction.id,
            memory_record=memory_record,
            replay_candidate=replay_candidate,
        )
        promotion_every = max(1, int(self.config.memory_promotion_every))
        if int(interaction.id) % promotion_every == 0:
            self.promotion_engine.run_all(step=int(interaction.id))
        if self._auto_commit:
            self.connection.commit()

        self._current_level_interaction_ids.append(int(interaction.id))
        trajectory_ids_snapshot = list(self._current_level_interaction_ids)

        if level_completed_event:
            self._apply_post_factum_level_completion_credit(completion_interaction_id=interaction.id)
        if outcome_state == "WIN":
            self._apply_post_factum_trajectory_credit(
                terminal_interaction_id=interaction.id,
                kind="WIN",
                polarity="positive",
                reason="win_terminal_trajectory",
                interaction_ids=trajectory_ids_snapshot,
            )
        elif outcome_state == "GAME_OVER":
            self._apply_post_factum_trajectory_credit(
                terminal_interaction_id=interaction.id,
                kind="GAME_OVER",
                polarity="negative",
                reason="game_over_failure_path",
                interaction_ids=trajectory_ids_snapshot,
            )

        if self.action_sampler is not None and hasattr(self.action_sampler, "record_result"):
            self.action_sampler.record_result(
                action=action,
                delta=delta,
                actual_family=actual_family,
                predicted_family=predicted_family,
                prediction_error=prediction_error,
                reset_boundary=bool(getattr(self.env, "last_step_was_reset_boundary", False)),
            )

        if bool(self.config.memory_query_enabled) or bool(self.config.memory_action_selection_enabled):
            self.memory_query.record_selected_action_query(
                context_signatures=context_signatures,
                action=action,
                prediction=memory_prediction,
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
            self._sync_post_factum_replay_fields(int(interaction_id))
            self._apply_post_factum_future_option_score(int(interaction_id), float(credit))
        self._current_level_interaction_ids = []

    def _apply_post_factum_trajectory_credit(
        self,
        *,
        terminal_interaction_id: int,
        kind: str,
        polarity: str,
        reason: str,
        interaction_ids: list[int] | None = None,
    ) -> None:
        ids = list(self._current_level_interaction_ids if interaction_ids is None else interaction_ids)
        if int(terminal_interaction_id) not in ids:
            ids.append(int(terminal_interaction_id))
        if not ids:
            return
        for distance_from_terminal, interaction_id in enumerate(reversed(ids)):
            credit = 1.0 / (1.0 + float(distance_from_terminal))
            self.connection.execute(
                """
                UPDATE interactions
                SET
                    post_factum_trajectory_credit = MAX(COALESCE(post_factum_trajectory_credit, 0.0), ?),
                    post_factum_trajectory_credit_kind = ?,
                    post_factum_trajectory_credit_polarity = ?,
                    post_factum_trajectory_credit_step = ?,
                    post_factum_trajectory_credit_reason = ?
                WHERE id = ?
                """,
                (
                    float(credit),
                    str(kind),
                    str(polarity),
                    int(terminal_interaction_id),
                    str(reason),
                    int(interaction_id),
                ),
            )
            self.connection.execute(
                """
                UPDATE prediction_results
                SET
                    post_factum_trajectory_credit = MAX(COALESCE(post_factum_trajectory_credit, 0.0), ?),
                    post_factum_trajectory_credit_kind = ?,
                    post_factum_trajectory_credit_polarity = ?,
                    post_factum_trajectory_credit_step = ?,
                    post_factum_trajectory_credit_reason = ?
                WHERE interaction_id = ?
                """,
                (
                    float(credit),
                    str(kind),
                    str(polarity),
                    int(terminal_interaction_id),
                    str(reason),
                    int(interaction_id),
                ),
            )
            self.memory_lifecycle.apply_post_factum_credit(
                str(interaction_id),
                learning_credit=float(credit),
                reason=str(reason),
            )
            self._sync_post_factum_replay_fields(int(interaction_id))
            signed_credit = float(credit) if str(kind) == "WIN" else -float(credit) if str(kind) == "GAME_OVER" else float(credit)
            self._apply_post_factum_future_option_score(int(interaction_id), signed_credit)
        self._current_level_interaction_ids = []

    def _sync_post_factum_replay_fields(self, interaction_id: int) -> None:
        candidate = self.memory_lifecycle.replay_candidates.get(str(interaction_id))
        if candidate is None:
            return
        self.connection.execute(
            """
            UPDATE interactions
            SET
                memory_replay_priority = ?,
                memory_replay_candidate = 1,
                memory_retention_reason = ?
            WHERE id = ?
            """,
            (
                float(candidate.replay_priority),
                str(candidate.reason),
                int(interaction_id),
            ),
        )
        self.connection.execute(
            """
            UPDATE prediction_results
            SET
                memory_replay_priority = ?,
                memory_replay_candidate = 1,
                memory_retention_reason = ?
            WHERE interaction_id = ?
            """,
            (
                float(candidate.replay_priority),
                str(candidate.reason),
                int(interaction_id),
            ),
        )

    def _apply_post_factum_future_option_score(self, interaction_id: int, future_option_delta: float) -> None:
        node_id = self._interaction_memory_node_id(interaction_id)
        row = self.connection.execute(
            "SELECT future_option_delta, replay_priority, retention_status FROM memory_scores WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        existing = None if row is None or row[0] is None else float(row[0])
        chosen = float(future_option_delta)
        if existing is not None and abs(existing) > abs(chosen):
            chosen = float(existing)
        replay_priority = None if row is None or row[1] is None else float(row[1])
        retention_status = None if row is None or row[2] is None else str(row[2])
        self.memory.upsert_score(
            MemoryScore(
                node_id=node_id,
                future_option_delta=chosen,
                replay_priority=replay_priority,
                retention_status=retention_status,
            ),
            step=int(interaction_id),
        )

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

    def _interaction_memory_node_id(self, interaction_id: int) -> str:
        return self._interaction_memory_node_ids.get(
            int(interaction_id),
            interaction_node_id(
                scoped_interaction_key(
                    interaction_id=interaction_id,
                    global_step=int(self.config.global_step_offset) + int(interaction_id)
                    if self.config.global_step_offset is not None
                    else None,
                )
            ),
        )

    def _observation_signature(self, observation: Any) -> str:
        encoded = json.dumps(observation.tolist(), separators=(",", ":"))
        return encoded

    def _write_m0_memory(
        self,
        *,
        interaction: Interaction,
        observation_before: Any,
        observation_after: Any,
        context_signature: str | None = None,
        context_level: int | None = None,
    ) -> None:
        memory_key = scoped_interaction_key(
            interaction_id=interaction.id,
            global_step=interaction.global_step,
        )
        interaction_node = interaction_node_id(memory_key)
        self._interaction_memory_node_ids[int(interaction.id)] = interaction_node
        before_signature = self._observation_signature(observation_before)
        after_signature = self._observation_signature(observation_after)
        before_hash = sha1(before_signature.encode("utf-8")).hexdigest()
        after_hash = sha1(after_signature.encode("utf-8")).hexdigest()
        before_node = observation_node_id(before_signature)
        after_node = observation_node_id(after_signature)
        action_node = action_node_id(interaction.action)
        delta_node = delta_node_id(interaction.delta_id)
        step = int(interaction.id)

        self.memory.upsert_node(
            MemoryNode(
                node_id=interaction_node,
                memory_level="M0",
                node_type="InteractionMemory",
                canonical_key=memory_key,
                attrs={
                    "interaction_id": int(interaction.id),
                    "local_interaction_id": int(interaction.id),
                    "action": int(interaction.action),
                    "delta_id": int(interaction.delta_id),
                    "global_step": None if interaction.global_step is None else int(interaction.global_step),
                    "memory_interaction_key": memory_key,
                    "context_signature": context_signature,
                    "context_level": None if context_level is None else int(context_level),
                },
            ),
            step=step,
        )
        self.memory.upsert_node(
            MemoryNode(
                node_id=before_node,
                memory_level="M0",
                node_type="ObservationMemory",
                canonical_key=before_hash,
                attrs={
                    "shape": list(observation_before.shape),
                    "observation_hash": before_hash,
                    "signature_len": len(before_signature),
                    "signature_preview": before_signature[:256],
                },
            ),
            step=step,
        )
        self.memory.upsert_node(
            MemoryNode(
                node_id=after_node,
                memory_level="M0",
                node_type="ObservationMemory",
                canonical_key=after_hash,
                attrs={
                    "shape": list(observation_after.shape),
                    "observation_hash": after_hash,
                    "signature_len": len(after_signature),
                    "signature_preview": after_signature[:256],
                },
            ),
            step=step,
        )
        self.memory.upsert_node(
            MemoryNode(
                node_id=action_node,
                memory_level="M0",
                node_type="ActionMemory",
                canonical_key=str(int(interaction.action)),
                attrs={"action": int(interaction.action)},
            ),
            step=step,
        )
        self.memory.upsert_node(
            MemoryNode(
                node_id=delta_node,
                memory_level="M0",
                node_type="DeltaMemory",
                canonical_key=str(int(interaction.delta_id)),
                attrs={"delta_id": int(interaction.delta_id)},
            ),
            step=step,
        )
        self.memory.upsert_edge(
            MemoryEdge(before_node, interaction_node, "observed_before", evidence={"interaction_id": int(interaction.id)})
        )
        self.memory.upsert_edge(
            MemoryEdge(interaction_node, action_node, "takes_action", evidence={"interaction_id": int(interaction.id)})
        )
        self.memory.upsert_edge(
            MemoryEdge(interaction_node, delta_node, "produces_delta", evidence={"interaction_id": int(interaction.id)})
        )
        self.memory.upsert_edge(
            MemoryEdge(interaction_node, after_node, "observed_after", evidence={"interaction_id": int(interaction.id)})
        )
        if self._last_memory_interaction_node_id is not None:
            self.memory.upsert_edge(
                MemoryEdge(
                    self._last_memory_interaction_node_id,
                    interaction_node,
                    "follows",
                    evidence={"interaction_id": int(interaction.id)},
                )
        )
        self._last_memory_interaction_node_id = interaction_node

    def _write_m1_m2_memory(
        self,
        *,
        interaction_id: int,
        stable_contingency: Any | None,
        actual_family: int,
        delta_id: int,
    ) -> None:
        family = getattr(self.clusterer, "families", {}).get(int(actual_family))
        family_attrs = {
            "family_id": int(actual_family),
            "support_count": None if family is None else int(getattr(family, "support_count", 0) or 0),
            "centroid_vector": None if family is None else getattr(family, "centroid_vector", None).tolist() if getattr(family, "centroid_vector", None) is not None else None,
        }
        family_node = family_node_id(actual_family)
        self.memory.upsert_node(
            MemoryNode(
                node_id=family_node,
                memory_level="M2",
                node_type="TransformationFamilyMemory",
                canonical_key=str(int(actual_family)),
                attrs=family_attrs,
            ),
            step=int(interaction_id),
        )
        self.memory.upsert_edge(
            MemoryEdge(
                delta_node_id(delta_id),
                family_node,
                "member_of",
                evidence={"interaction_id": int(interaction_id)},
            )
        )
        self.memory.upsert_edge(
            MemoryEdge(
                self._interaction_memory_node_id(interaction_id),
                family_node,
                "supports",
                evidence={"interaction_id": int(interaction_id)},
            )
        )
        if stable_contingency is None:
            return
        context_signature = json.dumps(list(stable_contingency.context_signature))
        contingency_node = contingency_node_id(
            int(stable_contingency.context_level),
            context_signature,
            int(stable_contingency.action),
            int(stable_contingency.transformation_family),
        )
        self.memory.upsert_node(
            MemoryNode(
                node_id=contingency_node,
                memory_level="M1",
                node_type="ContingencyMemory",
                canonical_key=f"{stable_contingency.context_level}|{context_signature}|{stable_contingency.action}|{stable_contingency.transformation_family}",
                attrs={
                    "context_level": int(stable_contingency.context_level),
                    "context_signature": context_signature,
                    "action": int(stable_contingency.action),
                    "transformation_family": int(stable_contingency.transformation_family),
                    "support_count": int(stable_contingency.support_count),
                    "confidence": float(stable_contingency.confidence),
                },
            ),
            step=int(interaction_id),
        )
        self.memory.upsert_edge(
            MemoryEdge(contingency_node, family_node, "predicts", evidence={"interaction_id": int(interaction_id)})
        )
        self.memory.upsert_edge(
            MemoryEdge(self._interaction_memory_node_id(interaction_id), contingency_node, "supports", evidence={"interaction_id": int(interaction_id)})
        )
        self.memory.upsert_edge(
            MemoryEdge(contingency_node, self._interaction_memory_node_id(interaction_id), "derived_from", evidence={"interaction_id": int(interaction_id)})
        )

    def _write_carrier_memory(
        self,
        *,
        interaction_id: int,
        carrier_signature: str | None,
        carrier_source: str,
        carrier_stats: dict[str, Any],
        context_signature: str,
        actual_family_id: str | None,
    ) -> None:
        if carrier_signature is None:
            return
        carrier_node = carrier_node_id(carrier_signature)
        self.memory.upsert_node(
            MemoryNode(
                node_id=carrier_node,
                memory_level="M3",
                node_type="CarrierMemory",
                canonical_key=str(carrier_signature),
                attrs={
                    "carrier_signature": str(carrier_signature),
                    "carrier_source": str(carrier_source),
                    "support_count": carrier_stats.get("carrier_support_count"),
                    "distinct_family_count": carrier_stats.get("carrier_distinct_family_count"),
                    "distinct_context_count": carrier_stats.get("carrier_distinct_context_count"),
                    "prediction_lift": carrier_stats.get("carrier_prediction_lift"),
                    "compression_gain": carrier_stats.get("carrier_compression_gain"),
                    "status": carrier_stats.get("carrier_status"),
                },
            ),
            step=int(interaction_id),
        )
        self.memory.upsert_edge(
            MemoryEdge(carrier_node, self._interaction_memory_node_id(interaction_id), "carried_by", evidence={"interaction_id": int(interaction_id)})
        )
        if actual_family_id is not None:
            self.memory.upsert_edge(
                MemoryEdge(
                    carrier_node,
                    family_node_id(actual_family_id),
                    "associated_with_family",
                    evidence={"interaction_id": int(interaction_id)},
                )
            )
        self.memory.upsert_edge(
            MemoryEdge(
                carrier_node,
                self._context_memory_node_id(context_signature),
                "appears_in_context",
                evidence={"interaction_id": int(interaction_id)},
            )
        )

    def _write_prediction_violation_memory(
        self,
        *,
        interaction_id: int,
        contradiction_event: Any,
        actual_family_id: str | None,
    ) -> None:
        violation_node = self._violation_memory_node_id(str(contradiction_event.contradiction_key))
        self.memory.upsert_node(
            MemoryNode(
                node_id=violation_node,
                memory_level="M1",
                node_type="PredictionViolationMemory",
                canonical_key=str(contradiction_event.contradiction_key),
                attrs={
                    "contradiction_key": str(contradiction_event.contradiction_key),
                    "predicted_family_id": contradiction_event.predicted_family_id,
                    "actual_family_id": contradiction_event.actual_family_id,
                    "prediction_confidence": contradiction_event.prediction_confidence,
                    "suggested_context_depth": contradiction_event.suggested_context_depth,
                    "reason": contradiction_event.reason,
                },
            ),
            step=int(interaction_id),
        )
        self.memory.upsert_edge(
            MemoryEdge(
                self._interaction_memory_node_id(interaction_id),
                violation_node,
                "violates_prediction",
                evidence={"interaction_id": int(interaction_id)},
            )
        )
        if contradiction_event.predicted_family_id is not None:
            self.memory.upsert_edge(
                MemoryEdge(
                    violation_node,
                    family_node_id(contradiction_event.predicted_family_id),
                    "contradicts",
                    evidence={"interaction_id": int(interaction_id)},
                )
            )
        if actual_family_id is not None:
            self.memory.upsert_edge(
                MemoryEdge(
                    violation_node,
                    family_node_id(actual_family_id),
                    "supports",
                    evidence={"interaction_id": int(interaction_id)},
                )
            )
        self.memory.upsert_edge(
            MemoryEdge(
                violation_node,
                self._context_memory_node_id(str(contradiction_event.context_signature)),
                "suggests_context_expansion",
                evidence={"interaction_id": int(interaction_id)},
            )
        )

    def _write_trajectory_and_efficiency_memory(
        self,
        *,
        interaction_id: int,
        efficiency_event: Any,
        context_signature: str,
    ) -> None:
        trajectory_node = trajectory_node_id(self.episode_id)
        self.memory.upsert_node(
            MemoryNode(
                node_id=trajectory_node,
                memory_level="M0",
                node_type="TrajectoryMemory",
                canonical_key=str(self.episode_id),
                attrs={
                    "episode_id": int(self.episode_id),
                    "outcome_signature": efficiency_event.outcome_signature,
                },
            ),
            step=int(interaction_id),
        )
        self.memory.upsert_edge(
            MemoryEdge(trajectory_node, self._interaction_memory_node_id(interaction_id), "contains", evidence={"interaction_id": int(interaction_id)})
        )
        cost_node = self._cost_memory_node_id(interaction_id)
        self.memory.upsert_node(
            MemoryNode(
                node_id=cost_node,
                memory_level="M0",
                node_type="CostMemory",
                canonical_key=str(interaction_id),
                attrs={
                    "action_cost": float(efficiency_event.action_cost),
                    "cumulative_cost": float(efficiency_event.cumulative_cost),
                },
            ),
            step=int(interaction_id),
        )
        self.memory.upsert_edge(
            MemoryEdge(self._interaction_memory_node_id(interaction_id), cost_node, "has_cost", evidence={"interaction_id": int(interaction_id)})
        )
        if efficiency_event.repeated_state and efficiency_event.state_signature is not None:
            self.memory.upsert_edge(
                MemoryEdge(
                    self._interaction_memory_node_id(interaction_id),
                    self._state_memory_node_id(efficiency_event.state_signature),
                    "repeats_state",
                    evidence={"interaction_id": int(interaction_id)},
                )
            )
        if efficiency_event.no_effect_action:
            interaction = self.interactions.get(interaction_id)
            if interaction is not None:
                self.memory.upsert_edge(
                    MemoryEdge(
                        self._interaction_memory_node_id(interaction_id),
                        delta_node_id(interaction.delta_id),
                        "no_effect",
                        evidence={"interaction_id": int(interaction_id)},
                    )
                )
        if efficiency_event.terminal_outcome or (
            getattr(self.env, "level_completed_event", False) and str(getattr(self.env, "last_outcome_state", "")) == "NOT_FINISHED"
        ):
            outcome_signature = efficiency_event.outcome_signature or f"{context_signature}|{getattr(self.env, 'last_outcome_state', 'NOT_FINISHED')}"
            outcome_node = self._outcome_memory_node_id(outcome_signature)
            self.memory.upsert_node(
                MemoryNode(
                    node_id=outcome_node,
                    memory_level="M0",
                    node_type="OutcomeMemory",
                    canonical_key=str(outcome_signature),
                    attrs={"outcome_signature": outcome_signature},
                ),
                step=int(interaction_id),
            )
            self.memory.upsert_edge(
                MemoryEdge(trajectory_node, outcome_node, "has_outcome", evidence={"interaction_id": int(interaction_id)})
            )
        if (
            efficiency_event.outcome_signature is not None
            and efficiency_event.best_known_cost_for_outcome is not None
            and efficiency_event.equivalent_outcome_cost_gap is not None
        ):
            strategy_signature = (
                f"{efficiency_event.outcome_signature}|best={efficiency_event.best_known_cost_for_outcome}|"
                f"current={efficiency_event.cumulative_cost}"
            )
            strategy_node = strategy_node_id(strategy_signature)
            self.memory.upsert_node(
                MemoryNode(
                    node_id=strategy_node,
                    memory_level="M6",
                    node_type="EfficientStrategyMemory",
                    canonical_key=strategy_signature,
                    attrs={
                        "outcome_signature": efficiency_event.outcome_signature,
                        "best_known_cost": efficiency_event.best_known_cost_for_outcome,
                        "current_cost": efficiency_event.cumulative_cost,
                        "normalized_solve_efficiency": efficiency_event.normalized_solve_efficiency,
                        "equivalent_outcome_cost_gap": efficiency_event.equivalent_outcome_cost_gap,
                    },
                ),
                step=int(interaction_id),
            )
            self.memory.upsert_edge(
                MemoryEdge(strategy_node, trajectory_node, "derived_from", evidence={"interaction_id": int(interaction_id)})
            )

    def _write_memory_lifecycle_memory(
        self,
        *,
        interaction_id: int,
        memory_record: Any,
        replay_candidate: Any | None,
    ) -> None:
        self.memory.upsert_score(
            MemoryScore(
                node_id=self._interaction_memory_node_id(interaction_id),
                isf_total=float(memory_record.isf_total),
                replay_priority=0.0 if replay_candidate is None else float(replay_candidate.replay_priority),
                retention_status=str(memory_record.status),
            ),
            step=int(interaction_id),
        )
        if replay_candidate is not None:
            replay_node = self._replay_queue_memory_node_id(interaction_id)
            self.memory.upsert_node(
                MemoryNode(
                    node_id=replay_node,
                    memory_level="M0",
                    node_type="ReplayQueueMemory",
                    canonical_key=str(interaction_id),
                    attrs={
                        "interaction_id": int(interaction_id),
                        "replay_priority": float(replay_candidate.replay_priority),
                        "reason": str(replay_candidate.reason),
                    },
                ),
                step=int(interaction_id),
            )
            self.memory.upsert_edge(
                MemoryEdge(
                    self._interaction_memory_node_id(interaction_id),
                    replay_node,
                    "selected_for_replay",
                    evidence={"interaction_id": int(interaction_id)},
                )
            )

    def _write_future_option_memory(
        self,
        *,
        interaction_id: int,
        before_option_set: Any,
        after_option_set: Any,
        option_delta: Any,
    ) -> None:
        before_node = f"M0:future_option_set:{before_option_set.option_set_id}"
        after_node = f"M0:future_option_set:{after_option_set.option_set_id}"
        scoped_key = self._interaction_memory_node_id(interaction_id).split("M0:interaction:", 1)[1]
        delta_node = f"M0:future_option_delta:{scoped_key}"
        self.memory.upsert_node(
            MemoryNode(
                node_id=before_node,
                memory_level="M0",
                node_type="FutureOptionSetMemory",
                canonical_key=str(before_option_set.option_set_id),
                attrs={
                    "state_signature": before_option_set.state_signature,
                    "available_actions": list(before_option_set.available_actions),
                    "reachable_signatures": list(before_option_set.reachable_signatures),
                    "estimated_branching_factor": int(before_option_set.estimated_branching_factor),
                    "depth": int(before_option_set.depth),
                },
            ),
            step=int(interaction_id),
        )
        self.memory.upsert_node(
            MemoryNode(
                node_id=after_node,
                memory_level="M0",
                node_type="FutureOptionSetMemory",
                canonical_key=str(after_option_set.option_set_id),
                attrs={
                    "state_signature": after_option_set.state_signature,
                    "available_actions": list(after_option_set.available_actions),
                    "reachable_signatures": list(after_option_set.reachable_signatures),
                    "estimated_branching_factor": int(after_option_set.estimated_branching_factor),
                    "depth": int(after_option_set.depth),
                },
            ),
            step=int(interaction_id),
        )
        self.memory.upsert_node(
            MemoryNode(
                node_id=delta_node,
                memory_level="M0",
                node_type="FutureOptionDeltaMemory",
                canonical_key=scoped_key,
                attrs={
                    "delta_score": float(option_delta.delta_score),
                    "added_options": list(option_delta.added_options),
                    "removed_options": list(option_delta.removed_options),
                    "preserved_options": list(option_delta.preserved_options),
                },
            ),
            step=int(interaction_id),
        )
        before_observation = observation_node_id(before_option_set.state_signature)
        after_observation = observation_node_id(after_option_set.state_signature)
        self.memory.upsert_edge(MemoryEdge(before_observation, before_node, "has_future_options"))
        self.memory.upsert_edge(MemoryEdge(after_observation, after_node, "has_future_options"))
        self.memory.upsert_edge(MemoryEdge(self._interaction_memory_node_id(interaction_id), delta_node, "changes_future_options"))
        self.memory.upsert_edge(MemoryEdge(delta_node, before_node, "from_option_set"))
        self.memory.upsert_edge(MemoryEdge(delta_node, after_node, "to_option_set"))
        if float(option_delta.delta_score) > 0.0:
            self.memory.upsert_edge(MemoryEdge(self._interaction_memory_node_id(interaction_id), after_node, "expands_future_options"))
        elif float(option_delta.delta_score) < 0.0:
            self.memory.upsert_edge(MemoryEdge(self._interaction_memory_node_id(interaction_id), after_node, "restricts_future_options"))
        else:
            self.memory.upsert_edge(MemoryEdge(self._interaction_memory_node_id(interaction_id), after_node, "preserves_future_options"))

    def _context_memory_node_id(self, context_signature: str) -> str:
        return f"M0:context:{sha1(str(context_signature).encode('utf-8')).hexdigest()[:20]}"

    def _violation_memory_node_id(self, contradiction_key: str) -> str:
        return f"M1:violation:{sha1(str(contradiction_key).encode('utf-8')).hexdigest()[:20]}"

    def _replay_queue_memory_node_id(self, interaction_id: int) -> str:
        scoped_key = self._interaction_memory_node_id(interaction_id).split("M0:interaction:", 1)[1]
        return f"M0:replay:{scoped_key}"

    def _cost_memory_node_id(self, interaction_id: int) -> str:
        scoped_key = self._interaction_memory_node_id(interaction_id).split("M0:interaction:", 1)[1]
        return f"M0:cost:{scoped_key}"

    def _state_memory_node_id(self, state_signature: str) -> str:
        return f"M0:state:{sha1(str(state_signature).encode('utf-8')).hexdigest()[:20]}"

    def _outcome_memory_node_id(self, outcome_signature: str) -> str:
        return f"M0:outcome:{sha1(str(outcome_signature).encode('utf-8')).hexdigest()[:20]}"

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
