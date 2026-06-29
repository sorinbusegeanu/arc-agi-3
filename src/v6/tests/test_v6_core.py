from __future__ import annotations

import json
import math
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import v6.evaluation.interaction_sampling as interaction_sampling

from v6.cli import build_parser
from v6.cli import _apply_interaction_sampling_experiment_preset
from v6.carrier_emergence import CarrierEmergenceTracker, extract_carrier_signature
from v6.game_sets import load_game_set_manifest
from v6.efficiency_metrics import EfficiencyTracker, is_no_effect_delta
from v6.contingency_memory import (
    ContingencyMemoryConfig,
    RunStreamingState,
    action_only_accuracy,
    build_input_manifest,
    build_m1_contingencies,
    build_episode_summary,
    build_context_signature,
    classify_outcome,
    context_model_accuracy,
    determine_manifest_path,
    discover_parquet_runs,
    format_v06_report,
    list_interaction_files,
    load_partition_events,
    print_progress,
    process_interaction_row,
    run_contingency_memory_v06,
)
from v6.context.contradiction_tracker import ContextContradictionTracker
from v6.context_depth_compare_v07 import (
    ContextDepthCompareConfig,
    format_context_depth_comparison,
    run_context_depth_compare_v07,
)
from v6.contingency.contingency_learner import Contingency, ContingencyLearner
from v6.delta.delta_extractor import Delta, extract_delta
from v6.environment.arc_adapter import ArcGridEnvironment
from v6.evaluation.broad_game_validation import (
    BroadValidationConfig,
    best_configs as broad_best_configs,
    family_for_game,
    game_passes,
    parse_game_selector,
    summary_by_family_rows,
    summary_by_game_rows,
    validation_summary as broad_validation_summary,
    write_broad_reports,
    _failed_row,
)
from v6.evaluation.failure_diagnostics import (
    FailureDiagnosticsConfig,
    classify_failure_reasons,
    compute_run_diagnostics,
    context_depth_sensitivity_summary,
    failure_reason_rows,
    family_diagnostic_summary,
    feature_repair_summary,
    parse_v05b_games,
    step_sensitivity_summary,
    write_failure_diagnostics_reports,
    _family_game_batches,
)
from v6.evaluation.id_free_prefuture_validation import (
    FORBIDDEN_ID_FEATURE_NAMES,
    ID_FREE_FEATURE_SETS,
    evaluate_id_free_config,
    feature_matrix_for_id_free,
    forbidden_future_feature_check,
    forbidden_id_feature_check,
    write_id_free_reports,
)
from v6.evaluation.interaction_sampling import (
    InteractionSamplingConfig,
    _apply_future_option_efficiency_diagnostics,
    _future_option_deltas_by_interaction_id,
    best_by_game as sampling_best_by_game,
    parse_v05c_games,
    parse_v05c_samplers,
    resolve_game_ids,
    resolve_interaction_sampling_scope,
    sampler_comparison_rows,
    sampling_db_path,
    validation_summary as sampling_validation_summary,
    write_interaction_sampling_reports,
)
from v6.evaluation.future_effects import (
    FutureEffect,
    InteractionEvent,
    analyze_future_effects,
    classify_future_effect,
    ensure_future_effects_schema,
    future_effect_for_occurrence,
    load_future_effects,
    replace_future_effects,
)
from v6.evaluation.prefuture_role_prediction import (
    PREFUTURE_FEATURE_SETS,
    PrefutureExample,
    classification_metrics,
    evaluate_prefuture_classifier,
    feature_matrix,
    forbidden_feature_check,
    nearest_centroid_predictions,
    normalization_stats,
    apply_normalization,
    write_prefuture_reports,
)
from v6.evaluation.role_candidates import (
    RoleCandidate,
    _matched_ratio,
    analyze_role_candidates,
    ensure_role_candidates_schema,
    load_role_candidates,
    replace_role_candidates,
)
from v6.evaluation.role_generalization import (
    FEATURE_SETS,
    feature_indices,
    project_feature_vectors,
    role_predictions_at_threshold,
)
from v6.evaluation.role_validation import (
    ALLOWED_ASSIGNMENT_FEATURE_NAMES,
    ValidationExample,
    TrainRole,
    _apply_normalization,
    _classification_metrics,
    _load_examples,
    _normalization_stats,
    validate_role_predictor,
)
from v6.evaluation.validation_report import build_validation_report
from v6.graph.graph_manager import GraphManager
from v6.hypothesis_suite_report import _write_aggregated_hypothesis_text
from v6.hypothesis_h07_report import evaluate_h07_concept_emergence
from v6.hypothesis_h08_report import evaluate_h08_world_model_coherence
from v6.hypothesis_h09_report import evaluate_h09_future_option_motifs
from v6.hypothesis_h10_report import evaluate_h10_future_option_attention
from v6.interaction_significance import compute_interaction_significance
from v6.main import V6Config, V6System
from v6.future_options import FutureOptionEstimator
from v6.memory.compact_memory import (
    CompactMemoryFoldConfig,
    _build_raw_db_fold_caches,
    _context_level_from_raw,
    derive_missing_transformation_families_from_stable_contingencies,
    canonical_family_signature_from_raw_db,
    ensure_memory_layout,
    finalize_main_compact_memory,
    fold_live_system_into_compact_memory,
    fold_single_sampling_db_into_main_compact_memory,
    fold_sampling_job_sidecars_into_compact_memory,
    fold_sampling_job_sidecars_into_compact_memory_shard,
    load_memory_summary,
    merge_compact_memory_shards_into_main,
    normalized_contingency_identity,
    normalized_contingency_identity_cached,
)
from v6.memory.contingency_store import ContingencyStore
from v6.memory.interaction_store import Interaction, InteractionStore, encode_array
from v6.memory.live_memory_queue import (
    LiveMemoryReadCache,
    LiveMemoryWriterConfig,
    make_live_memory_queue,
    start_live_memory_writer,
    stop_live_memory_writer,
)
from v6.memory.promotion_engine import MemoryPromotionConfig, MemoryPromotionEngine
from v6.memory.query_engine import MemoryQueryEngine
from v6.memory.substrate import MemoryEdge, MemoryEvidence, MemoryNode, MemoryPromotion, MemoryScore, MemorySubstrate, action_node_id, delta_node_id, family_node_id, interaction_node_id, strategy_node_id
from v6.memory.compact_memory_restore import _fetchall_readonly_with_retry
from v6.memory_lifecycle import MemoryLifecycleManager
from v6.memory_types import M3RoleCandidate
from v6.m2_expand_v08c import M2ExpandV08cConfig, run_m2_expand_v08c
from v6.role_candidates_v08 import (
    RoleCandidatesV08Config,
    build_neighborhoods,
    build_role_candidates,
    build_similarity_adjacency,
    cluster_role_candidates,
    evaluate_pairwise_similarity,
    run_role_candidates_v08,
)
from v6.role_candidates_v08d import (
    RoleCandidatesV08dConfig,
    build_discriminative_neighborhoods,
    build_role_candidates as build_role_candidates_v08d,
    build_similarity_adjacency as build_similarity_adjacency_v08d,
    cluster_role_candidates as cluster_role_candidates_v08d,
    evaluate_pairwise_similarity as evaluate_pairwise_similarity_v08d,
    load_m1_support as load_m1_support_v08d,
    load_m2_families as load_m2_families_v08d,
    role_status,
    run_role_candidates_v08d,
)
from v6.role_transfer_v09 import RoleTransferV09Config, all_features, build_source_role_prototypes, load_neighborhoods, load_roles, run_role_transfer_v09
from v6.role_transfer_v09b import (
    RoleTransferV09bConfig,
    build_prototype_entry,
    build_strategy_specs,
    medoid_index,
    passes_confidence_gate,
    rank_roles_for_target,
    run_role_transfer_v09b,
    select_best_strategy_payload,
    StrategySpec,
)
from v6.role_transfer_v09c import (
    RoleTransferV09cConfig,
    SingleFamilyContext,
    build_single_family_context,
    choose_parallel_worker_count,
    future_option_behavior_features,
    graph_position_features,
    prepare_family_context_stream,
    run_role_transfer_v09c,
    select_same_effect_different_role_case,
    select_same_role_different_effect_case,
    surface_similarity_bin,
)
from v6.concept_candidates_v10 import (
    ConceptCandidatesV10Config,
    discover_concept_candidates,
    extract_role_graph_motif,
    run_concept_candidates_v10,
    sequence_similarity,
)
from v6.concept_candidates_v10fix import (
    ConceptCandidatesV10FixConfig,
    apply_target_metrics,
    evaluate_target_projection,
    merge_concept_rows,
    run_concept_candidates_v10fix,
    stable_concept_signature,
    strict_label_candidate,
)
from v6.concept_candidates_v10fixb import (
    ConceptCandidatesV10FixBConfig,
    RESIDUAL_PROFILE_KEYS,
    average_role_fingerprint_similarity,
    build_collision_rows as build_collision_rows_fixb,
    build_effect_residual_profile_from_record,
    build_effect_residual_profile_from_target_rows,
    build_source_role_map as build_source_role_map_fixb,
    canonical_role_fingerprint,
    choose_effective_worker_count,
    concepts_are_fuzzy_compatible,
    evaluate_target_projection_by_family,
    fuzzy_group_candidates,
    generate_subcomposition_candidates,
    load_source_manifest_family_map,
    merge_exact_candidates,
    resolve_manifest_families_for_record,
    run_concept_candidates_v10fixb,
)
from v6.concept_candidates_v10fixc import (
    ConceptCandidatesV10FixCConfig,
    discover_source_only_candidates_fixc,
    evaluate_target_projection_by_family_fixc,
    run_concept_candidates_v10fixc,
    validate_completed_fixc_run,
)
from v6.concept_candidates_v10fixd import (
    ConceptCandidatesV10FixDConfig,
    collect_source_items_and_manifest_groups,
    family_fixd_paths,
    run_concept_candidates_v10fixd,
    run_single_family_fixd,
)
from v6.m4_role_concepts_v10e import (
    M4RoleConceptsV10eConfig,
    build_baseline_dominance_rows,
    build_projection_audit_rows,
    build_v10e_report,
    build_role_based_candidate_row,
    is_transferable_role_based_concept,
    run_m4_role_concepts_v10e,
    score_role_based_concept_against_target_family,
)
from v6.role_transfer_v09a import RoleTransferV09aConfig, run_role_transfer_v09a
from v6.sampling import (
    ActionBalanceSampler,
    LowConfidenceSampler,
    MemoryGuidedExploreSampler,
    MemoryGuidedSampler,
    MixedExplorerSampler,
    NoChangeAvoidanceSampler,
    NoveltyDeltaSampler,
    ResetAwareMixedExplorerSampler,
    make_sampler,
    sampler_registry,
)
from v6.storage.duckdb_queries import load_run_table
from v6.storage.migration import migrate_sqlite_to_parquet
from v6.storage.parquet_backend import ParquetStorageBackend
from v6.storage.sqlite_backend import SQLiteStorageBackend
from v6.transformation_families_v07 import (
    TransformationFamiliesV07Config,
    build_m2_families,
    compute_action_baselines,
    family_similarity,
    families_by_game,
    format_v07_report,
    load_m1_contingencies,
    run_transformation_families_v07,
    validation_summary as v07_validation_summary,
)
from v6.transformation.transformation_clusterer import TransformationClusterer


class ToggleEnv:
    def __init__(self) -> None:
        self.grid = np.zeros((3, 3), dtype=int)
        self.state = 0

    def observe(self) -> np.ndarray:
        return self.grid.copy()

    def step(self, action: int) -> np.ndarray:
        self.state = 1 - self.state
        self.grid = np.zeros((3, 3), dtype=int)
        self.grid[1, 1] = self.state
        return self.grid.copy()

    def available_actions(self) -> list[int]:
        return [1]


class OutcomeEnv(ToggleEnv):
    def __init__(self, *, outcome_state: str, outcome_polarity: str, starting_level: int = 0, next_level: int | None = None, state_name: str | None = None) -> None:
        super().__init__()
        self._configured_outcome_state = str(outcome_state)
        self._configured_outcome_polarity = str(outcome_polarity)
        self._starting_level = int(starting_level)
        self._next_level = self._starting_level if next_level is None else int(next_level)
        self._state_name = state_name
        self.last_outcome_state = "NOT_FINISHED"
        self.last_outcome_polarity = "neutral"
        self.last_terminal_state = None
        self.last_step_was_reset_boundary = False
        self.last_reward = 0
        self.last_levels_completed = self._starting_level
        self.level_completed_event = False

    def step(self, action: int) -> np.ndarray:
        grid = super().step(action)
        self.last_outcome_state = self._configured_outcome_state
        self.last_outcome_polarity = self._configured_outcome_polarity
        self.last_terminal_state = self._configured_outcome_state if self._configured_outcome_state in {"WIN", "GAME_OVER"} else None
        self.last_step_was_reset_boundary = self._configured_outcome_state in {"WIN", "GAME_OVER"} and self._state_name is None
        self.level_completed_event = self._next_level > self.last_levels_completed
        self.last_levels_completed = self._next_level
        self.state = self._state_name or self.state
        return grid


class LevelCompletionSequenceEnv(ToggleEnv):
    def __init__(self, *, completion_step: int = 4, terminal_state: str = "NOT_FINISHED", increment_levels: bool = False) -> None:
        super().__init__()
        self.step_index = 0
        self.completion_step = int(completion_step)
        self.terminal_state = str(terminal_state)
        self.increment_levels = bool(increment_levels)
        self.last_outcome_state = "NOT_FINISHED"
        self.last_outcome_polarity = "neutral"
        self.last_terminal_state = None
        self.last_step_was_reset_boundary = False
        self.last_reward = 0
        self.last_levels_completed = 0
        self.level_completed_event = False

    def step(self, action: int) -> np.ndarray:
        grid = super().step(action)
        self.step_index += 1
        self.level_completed_event = self.step_index == self.completion_step and self.increment_levels
        if self.level_completed_event:
            self.last_levels_completed += 1
        self.last_outcome_state = self.terminal_state if self.step_index >= self.completion_step and self.terminal_state != "NOT_FINISHED" else "NOT_FINISHED"
        if self.last_outcome_state == "WIN":
            self.last_outcome_polarity = "positive"
        elif self.last_outcome_state == "GAME_OVER":
            self.last_outcome_polarity = "negative"
        elif self.level_completed_event:
            self.last_outcome_polarity = "positive"
        else:
            self.last_outcome_polarity = "neutral"
        self.last_terminal_state = self.last_outcome_state if self.last_outcome_state in {"WIN", "GAME_OVER"} else None
        self.last_step_was_reset_boundary = False
        return grid


def test_delta_extractor_uses_cell_changes_without_components() -> None:
    before = np.zeros((4, 4), dtype=int)
    after = before.copy()
    before[1, 1] = 2
    after[1, 2] = 2

    delta = extract_delta(before, after, delta_id=7)

    assert delta.id == 7
    assert delta.changed_cells == 2
    assert sorted(delta.changed_positions) == [(1, 1), (1, 2)]
    assert delta.dx == 1.0
    assert delta.dy == 0.0


def test_transformation_clusterer_discovers_families_from_delta_features() -> None:
    deltas = []
    for index in range(10):
        deltas.append(_delta_with_features(index + 1, changed_cells=2, dx=1.0, dy=0.0))
    for index in range(10):
        deltas.append(_delta_with_features(index + 11, changed_cells=50, dx=20.0, dy=0.0))

    clusterer = TransformationClusterer(min_cluster_size=5, recluster_every=5)
    assignments = clusterer.fit(deltas)

    assert len(clusterer.families) >= 2
    assert set(assignments) == {delta.id for delta in deltas}


def test_contingency_learner_promotes_stable_high_confidence_rule() -> None:
    learner = ContingencyLearner(support_threshold=3, confidence_threshold=0.8)
    context = (None, None, 1, None, None, 2)

    contingency = None
    for _ in range(3):
        contingency = learner.update(context, action=4, transformation_family=9)

    assert contingency is not None
    assert contingency.context_level == 0
    assert contingency.support_count == 3
    assert contingency.confidence == 1.0


def test_contingency_learner_tracks_multi_scale_contexts_and_prefers_specific() -> None:
    learner = ContingencyLearner(support_threshold=3, confidence_threshold=0.8)
    contexts = {
        0: (4,),
        1: (2, 3, 4),
        2: (1, 2, 3, 3, 4),
    }

    for _ in range(3):
        learner.update_multi_scale(contexts, action=4, transformation_family=9)

    stable = learner.stable_contingencies()
    assert {item.context_level for item in stable} == {0, 1, 2}
    assert learner.best_stable_for_action(contexts, action=4).context_level == 2
    assert learner.predict(contexts, action=4) == 9


def test_v6_system_records_interactions_predictions_and_metrics() -> None:
    env = ToggleEnv()
    system = V6System(
        env=env,
        config=V6Config(
            recluster_every=5,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=2,
            contingency_confidence_threshold=0.8,
            random_seed=0,
        ),
    )

    results = system.run(steps=30)
    metrics = system.metrics()

    assert len(results) == 30
    assert system.interactions.count() == 30
    assert metrics.transformation_family_count >= 1
    assert metrics.stable_contingency_count >= 1
    assert metrics.prediction_accuracy is not None


def test_v6_system_context_builder_retains_max_depth_for_adaptive_expansion() -> None:
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            context_length=1,
            max_context_depth=3,
            adaptive_context_expansion=True,
            random_seed=0,
        ),
    )
    try:
        assert system.context_builder.context_length == 3
        system.context_builder.update(1, 1)
        system.context_builder.update(2, 2)
        system.context_builder.update(3, 1)
        signatures = system.context_builder.multi_scale_signatures(1, max_level=3)
        assert 3 in signatures
        assert any(value is not None for value in signatures[3][:-1])
    finally:
        system.close()


def test_v6_system_adaptive_context_expansion_disabled() -> None:
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            context_length=1,
            max_context_depth=3,
            adaptive_context_expansion=False,
            random_seed=0,
        ),
    )
    try:
        depth, applied = system._apply_adaptive_context_expansion(
            action=1,
            old_depth=1,
            suggested_depth=2,
            reason="r",
            contradiction_key="k",
        )
        assert depth == 1
        assert applied is False
    finally:
        system.close()


def test_v6_system_adaptive_context_expansion_enabled() -> None:
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            context_length=1,
            max_context_depth=3,
            adaptive_context_expansion=True,
            random_seed=0,
        ),
    )
    try:
        depth, applied = system._apply_adaptive_context_expansion(
            action=1,
            old_depth=1,
            suggested_depth=2,
            reason="r",
            contradiction_key="k",
        )
        assert depth == 2
        assert applied is True
        assert system._context_depth_for_action(1) == 2
    finally:
        system.close()


def test_v6_system_adaptive_context_expansion_clamps_to_max_depth() -> None:
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            context_length=1,
            max_context_depth=2,
            adaptive_context_expansion=True,
            random_seed=0,
        ),
    )
    try:
        depth, applied = system._apply_adaptive_context_expansion(
            action=1,
            old_depth=1,
            suggested_depth=5,
            reason="r",
            contradiction_key="k",
        )
        assert depth == 2
        assert applied is True
    finally:
        system.close()


def test_v6_system_batched_database_commit_persists_on_close(tmp_path) -> None:
    db_path = tmp_path / "batched.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
            database_commit_every=1000,
        ),
    )

    system.run(steps=3)
    system.close()

    connection = __import__("sqlite3").connect(db_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM deltas").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM prediction_results").fetchone()[0] == 3
    finally:
        connection.close()


def test_graph_manager_add_typed_edge_creates_typed_edge() -> None:
    graph = GraphManager()

    graph.add_typed_edge("A", "B", "explains", weight=0.8, evidence={"x": 1})

    assert graph.count_edges_of_type("explains") == 1
    assert graph.edge_type_counts()["explains"] == 1


def test_graph_manager_duplicate_typed_edge_updates_in_place() -> None:
    graph = GraphManager()

    graph.add_typed_edge("A", "B", "explains", weight=0.3, evidence={"a": 1})
    graph.add_typed_edge("A", "B", "explains", weight=0.9, evidence={"b": 2})

    assert graph.count_edges_of_type("explains") == 1
    edges = list(graph.graph.edges(data=True, keys=True))
    assert len(edges) == 1
    _, _, _, attrs = edges[0]
    assert attrs["weight"] == 0.9
    assert attrs["evidence"]["a"] == 1
    assert attrs["evidence"]["b"] == 2


def test_graph_manager_symmetric_edges_are_added_both_directions() -> None:
    graph = GraphManager()

    graph.add_reversible_with("A", "B")
    graph.add_similar_role_to("Role:A", "Role:B")

    assert graph.count_edges_of_type("reversible_with") == 2
    assert graph.count_edges_of_type("similar_role_to") == 2


def test_graph_manager_edge_counts_include_zero_buckets() -> None:
    graph = GraphManager()

    counts = graph.edge_type_counts()

    assert counts["enables"] == 0
    assert counts["blocks"] == 0
    assert counts["contradicts"] == 0


def test_interaction_significance_prediction_error_rewards_confident_misses() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=False,
        prediction_confidence=0.9,
        actual_family_id="f1",
        delta_id="d1",
        context_signature="c1",
        memory_counts={},
        graph_counts={},
    )

    assert score.prediction_error >= 0.9
    assert score.total > 0


def test_interaction_significance_no_longer_accepts_reward_argument() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id=None,
        context_signature=None,
        memory_counts={},
        graph_counts={},
    )

    assert score.total >= 0.0

    with pytest.raises(TypeError):
        compute_interaction_significance(
            reward=0,
            terminated=False,
            truncated=False,
            prediction_correct=None,
            prediction_confidence=None,
            actual_family_id=None,
            delta_id=None,
            context_signature=None,
            memory_counts={},
            graph_counts={},
        )


def test_context_contradiction_tracker_ignores_correct_predictions() -> None:
    tracker = ContextContradictionTracker()

    event = tracker.record_prediction_result(
        interaction_id="i1",
        context_signature="ctx1",
        action_signature="a1",
        predicted_family_id="f1",
        actual_family_id="f1",
        prediction_correct=True,
        prediction_confidence=0.9,
        context_depth=2,
    )

    assert event is None
    assert tracker.summary()["context_contradiction_count"] == 0


def test_context_contradiction_tracker_creates_event_for_confident_wrong_prediction() -> None:
    tracker = ContextContradictionTracker()

    event = tracker.record_prediction_result(
        interaction_id="i1",
        context_signature="ctx1",
        action_signature="a1",
        predicted_family_id="f1",
        actual_family_id="f2",
        prediction_correct=False,
        prediction_confidence=0.9,
        context_depth=2,
        max_context_depth=5,
    )

    assert event is not None
    assert event.contradiction_key == "ctx1|a1|f1->f2"
    assert event.suggested_context_depth == 3
    assert tracker.summary()["context_contradiction_count"] == 1


def test_context_contradiction_tracker_ignores_low_confidence_wrong_prediction() -> None:
    tracker = ContextContradictionTracker(min_confidence=0.7)

    event = tracker.record_prediction_result(
        interaction_id="i1",
        context_signature="ctx1",
        action_signature="a1",
        predicted_family_id="f1",
        actual_family_id="f2",
        prediction_correct=False,
        prediction_confidence=0.3,
        context_depth=2,
    )

    assert event is None
    summary = tracker.summary()
    assert summary["contradiction_suppressed_low_confidence_count"] == 1
    assert summary["wrong_prediction_count"] == 1
    assert summary["confident_wrong_prediction_count"] == 0


def test_context_contradiction_tracker_tracks_missing_prediction_and_actual() -> None:
    tracker = ContextContradictionTracker()

    event_missing_prediction = tracker.record_prediction_result(
        interaction_id="i1",
        context_signature="ctx1",
        action_signature="a1",
        predicted_family_id=None,
        actual_family_id="f2",
        prediction_correct=False,
        prediction_confidence=0.9,
        context_depth=2,
    )
    event_missing_actual = tracker.record_prediction_result(
        interaction_id="i2",
        context_signature="ctx1",
        action_signature="a1",
        predicted_family_id="f1",
        actual_family_id=None,
        prediction_correct=False,
        prediction_confidence=0.9,
        context_depth=2,
    )

    assert event_missing_prediction is None
    assert event_missing_actual is None
    summary = tracker.summary()
    assert summary["contradiction_suppressed_missing_prediction_count"] == 1
    assert summary["contradiction_suppressed_missing_actual_count"] == 1


def test_context_contradiction_tracker_tracks_correct_or_unknown_suppression() -> None:
    tracker = ContextContradictionTracker()
    event = tracker.record_prediction_result(
        interaction_id="i1",
        context_signature="ctx1",
        action_signature="a1",
        predicted_family_id="f1",
        actual_family_id="f2",
        prediction_correct=None,
        prediction_confidence=0.9,
        context_depth=2,
    )
    assert event is None
    assert tracker.summary()["contradiction_suppressed_correct_or_unknown_count"] == 1


def test_context_contradiction_tracker_repeated_contradictions_suggest_expansion() -> None:
    tracker = ContextContradictionTracker(min_repeats_for_expansion=2)

    tracker.record_prediction_result(
        interaction_id="i1",
        context_signature="ctx1",
        action_signature="a1",
        predicted_family_id="f1",
        actual_family_id="f2",
        prediction_correct=False,
        prediction_confidence=0.9,
        context_depth=2,
    )
    tracker.record_prediction_result(
        interaction_id="i2",
        context_signature="ctx1",
        action_signature="a1",
        predicted_family_id="f1",
        actual_family_id="f2",
        prediction_correct=False,
        prediction_confidence=0.9,
        context_depth=2,
    )

    assert tracker.should_expand_context("ctx1", "a1") is True
    assert tracker.summary()["repeated_contradiction_count"] >= 1


def test_extract_carrier_signature_uses_position() -> None:
    sig, source = extract_carrier_signature(
        before_observation=None,
        after_observation=None,
        delta={"position": [3, 4]},
        context_signature="ctx",
        action_signature="act",
    )

    assert sig is not None
    assert "3" in sig
    assert "4" in sig
    assert source == "cell"


def test_extract_carrier_signature_falls_back_to_context_action() -> None:
    sig, source = extract_carrier_signature(
        before_observation=None,
        after_observation=None,
        delta={},
        context_signature="ctx1",
        action_signature="a1",
    )

    assert sig == "context_action:ctx1|a1"
    assert source == "context_action_fallback"


def test_carrier_tracker_records_event() -> None:
    tracker = CarrierEmergenceTracker()

    event = tracker.record_interaction(
        interaction_id="i1",
        carrier_signature="cell:1,2",
        context_signature="ctx1",
        action_signature="a1",
        family_id="f1",
        delta_signature="d1",
        prediction_correct=True,
        carrier_source="cell",
    )

    assert event is not None
    assert len(tracker.events) == 1
    assert event.carrier_source == "cell"


def test_carrier_tracker_emergent_threshold() -> None:
    tracker = CarrierEmergenceTracker(
        min_support=3,
        min_distinct_contexts=2,
        min_prediction_lift=0.0,
        min_compression_gain=0.01,
    )

    tracker.record_interaction(
        interaction_id="i1",
        carrier_signature="cell:1,2",
        context_signature="ctx1",
        action_signature="a1",
        family_id="f1",
        delta_signature="d1",
        prediction_correct=True,
        carrier_source="cell",
    )
    tracker.record_interaction(
        interaction_id="i2",
        carrier_signature="cell:1,2",
        context_signature="ctx2",
        action_signature="a1",
        family_id="f1",
        delta_signature="d2",
        prediction_correct=True,
        carrier_source="cell",
    )
    tracker.record_interaction(
        interaction_id="i3",
        carrier_signature="cell:1,2",
        context_signature="ctx2",
        action_signature="a1",
        family_id="f1",
        delta_signature="d3",
        prediction_correct=True,
        carrier_source="cell",
    )

    assert any(candidate.status == "emergent_carrier" for candidate in tracker.build_candidates())


def test_carrier_tracker_ignores_missing_signature() -> None:
    tracker = CarrierEmergenceTracker()

    event = tracker.record_interaction(
        interaction_id="i1",
        carrier_signature=None,
        context_signature="ctx1",
        action_signature="a1",
        family_id="f1",
        delta_signature="d1",
        prediction_correct=True,
    )

    assert event is None
    assert len(tracker.events) == 0


def test_carrier_tracker_context_action_fallback_cannot_become_emergent() -> None:
    tracker = CarrierEmergenceTracker(
        min_support=2,
        min_distinct_contexts=1,
        min_prediction_lift=0.0,
        min_compression_gain=0.0,
    )

    for index in range(3):
        tracker.record_interaction(
            interaction_id=f"i{index}",
            carrier_signature="context_action:ctx|a1",
            context_signature=f"ctx{index}",
            action_signature="a1",
            family_id="f1",
            delta_signature=f"d{index}",
            prediction_correct=True,
            carrier_source="context_action_fallback",
        )

    candidates = tracker.build_candidates()

    assert candidates[0].status == "contextual_fallback_candidate"
    assert candidates[0].carrier_source == "context_action_fallback"
    assert not any(candidate.status == "emergent_carrier" for candidate in candidates)


def test_carrier_tracker_cell_source_can_still_become_emergent() -> None:
    tracker = CarrierEmergenceTracker(
        min_support=2,
        min_distinct_contexts=1,
        min_prediction_lift=0.0,
        min_compression_gain=0.0,
    )

    for index in range(3):
        tracker.record_interaction(
            interaction_id=f"i{index}",
            carrier_signature="position:1,2",
            context_signature=f"ctx{index}",
            action_signature="a1",
            family_id="f1",
            delta_signature=f"d{index}",
            prediction_correct=True,
            carrier_source="cell",
        )

    candidates = tracker.build_candidates()

    assert any(candidate.status == "emergent_carrier" for candidate in candidates)
    assert any(candidate.carrier_source == "cell" for candidate in candidates)


def test_memory_lifecycle_high_isf_becomes_protected() -> None:
    manager = MemoryLifecycleManager()

    record = manager.register_interaction(
        interaction_id="i1",
        family_id="f1",
        context_signature="ctx1",
        action_signature="a1",
        carrier_signature=None,
        isf_total=0.9,
        prediction_error=0.1,
        learning_value=0.2,
        transfer_potential=0.1,
        explanatory_potential=0.1,
        context_contradiction=False,
        timestamp_step=1,
    )

    assert record.status == "protected"
    assert record.retention_reason == "high_isf"


def test_memory_lifecycle_prediction_error_enters_replay_queue() -> None:
    manager = MemoryLifecycleManager()

    manager.register_interaction(
        interaction_id="i1",
        family_id="f1",
        context_signature="ctx1",
        action_signature="a1",
        carrier_signature=None,
        isf_total=0.4,
        prediction_error=0.9,
        learning_value=0.1,
        transfer_potential=0.1,
        explanatory_potential=0.1,
        context_contradiction=False,
        timestamp_step=1,
    )

    assert "i1" in manager.replay_candidates
    assert manager.replay_candidates["i1"].reason == "prediction_error"


def test_memory_lifecycle_context_contradiction_enters_replay_queue() -> None:
    manager = MemoryLifecycleManager()

    manager.register_interaction(
        interaction_id="i1",
        family_id="f1",
        context_signature="ctx1",
        action_signature="a1",
        carrier_signature=None,
        isf_total=0.2,
        prediction_error=0.1,
        learning_value=0.1,
        transfer_potential=0.1,
        explanatory_potential=0.1,
        context_contradiction=True,
        timestamp_step=1,
    )

    assert "i1" in manager.replay_candidates


def test_memory_lifecycle_apply_post_factum_credit_merges_reasons() -> None:
    manager = MemoryLifecycleManager()
    manager.register_interaction(
        interaction_id="i1",
        family_id="f1",
        context_signature="ctx1",
        action_signature="a1",
        carrier_signature=None,
        isf_total=0.2,
        prediction_error=0.1,
        learning_value=0.1,
        transfer_potential=0.1,
        explanatory_potential=0.1,
        context_contradiction=False,
        timestamp_step=1,
    )

    manager.apply_post_factum_credit("i1", learning_credit=0.5, reason="post_factum_level_completion")
    manager.apply_post_factum_credit("i1", learning_credit=0.8, reason="win_terminal_trajectory")

    assert manager.replay_candidates["i1"].reason == "post_factum_level_completion+win_terminal_trajectory"
    assert manager.replay_candidates["i1"].replay_priority == 0.8


def test_memory_lifecycle_replay_batch_returns_highest_priority_first() -> None:
    manager = MemoryLifecycleManager()
    manager.register_interaction(
        interaction_id="i1",
        family_id="f1",
        context_signature="ctx1",
        action_signature="a1",
        carrier_signature=None,
        isf_total=0.2,
        prediction_error=0.5,
        learning_value=0.1,
        transfer_potential=0.1,
        explanatory_potential=0.1,
        context_contradiction=False,
        timestamp_step=1,
    )
    manager.register_interaction(
        interaction_id="i2",
        family_id="f1",
        context_signature="ctx2",
        action_signature="a1",
        carrier_signature=None,
        isf_total=0.6,
        prediction_error=0.9,
        learning_value=0.1,
        transfer_potential=0.1,
        explanatory_potential=0.1,
        context_contradiction=False,
        timestamp_step=2,
    )
    manager.register_interaction(
        interaction_id="i3",
        family_id="f1",
        context_signature="ctx3",
        action_signature="a1",
        carrier_signature=None,
        isf_total=0.3,
        prediction_error=0.5,
        learning_value=0.6,
        transfer_potential=0.1,
        explanatory_potential=0.1,
        context_contradiction=False,
        timestamp_step=3,
    )

    batch = manager.get_replay_batch(limit=2)

    assert len(batch) == 2
    assert batch[0].replay_priority >= batch[1].replay_priority


def test_memory_lifecycle_forgetting_does_not_remove_protected_records() -> None:
    manager = MemoryLifecycleManager(
        max_active_records=2,
        min_records_before_forgetting=1,
        forget_isf_threshold=0.2,
    )

    protected = manager.register_interaction(
        interaction_id="protected",
        family_id="f1",
        context_signature="ctx1",
        action_signature="a1",
        carrier_signature=None,
        isf_total=0.95,
        prediction_error=0.1,
        learning_value=0.1,
        transfer_potential=0.1,
        explanatory_potential=0.1,
        context_contradiction=False,
        timestamp_step=1,
    )
    for index in range(4):
        manager.register_interaction(
            interaction_id=f"low-{index}",
            family_id="f1",
            context_signature=f"ctx{index + 2}",
            action_signature="a1",
            carrier_signature=None,
            isf_total=0.05,
            prediction_error=0.05,
            learning_value=0.05,
            transfer_potential=0.05,
            explanatory_potential=0.05,
            context_contradiction=False,
            timestamp_step=index + 2,
        )

    assert protected.status == "protected"
    assert manager.records["protected"].status == "protected"
    assert manager.summary()["memory_forgotten_count"] > 0


def test_efficiency_no_effect_detection() -> None:
    assert is_no_effect_delta(None) is True
    assert is_no_effect_delta({"changed_cells": []}) is True


def test_efficiency_tracker_detects_repeated_state() -> None:
    tracker = EfficiencyTracker()

    e1 = tracker.record_interaction(
        interaction_id="i1",
        before_observation={"x": 1},
        after_observation={"x": 2},
        delta={"position": [1, 2]},
        context_signature="ctx",
        action_signature="a",
        reward=0,
        terminated=False,
        truncated=False,
    )
    e2 = tracker.record_interaction(
        interaction_id="i2",
        before_observation={"x": 1},
        after_observation={"x": 3},
        delta={"position": [1, 3]},
        context_signature="ctx",
        action_signature="b",
        reward=0,
        terminated=False,
        truncated=False,
    )

    assert e1.repeated_state is False
    assert e2.repeated_state is True


def test_efficiency_tracker_detects_repeated_context_action() -> None:
    tracker = EfficiencyTracker()
    tracker.record_interaction(
        interaction_id="i1",
        before_observation={"x": 1},
        after_observation={"x": 2},
        delta={"position": [1, 2]},
        context_signature="ctx",
        action_signature="a",
        reward=0,
        terminated=False,
        truncated=False,
    )
    event = tracker.record_interaction(
        interaction_id="i2",
        before_observation={"x": 2},
        after_observation={"x": 3},
        delta={"position": [1, 3]},
        context_signature="ctx",
        action_signature="a",
        reward=0,
        terminated=False,
        truncated=False,
    )

    assert event.repeated_context_action is True


def test_efficiency_tracker_terminal_outcome_has_efficiency_score() -> None:
    tracker = EfficiencyTracker()

    event = tracker.record_interaction(
        interaction_id="i1",
        before_observation={"x": 1},
        after_observation={"x": 2},
        delta={"position": [1, 2]},
        context_signature="ctx",
        action_signature="a",
        reward=1,
        terminated=True,
        truncated=False,
    )

    assert event.terminal_outcome is True
    assert event.normalized_solve_efficiency is not None


def test_efficiency_tracker_equivalent_outcome_cost_gap_is_reported() -> None:
    tracker = EfficiencyTracker()
    tracker.record_interaction(
        interaction_id="i1",
        before_observation={"x": 1},
        after_observation={"x": 2},
        delta={"position": [1, 2]},
        context_signature="ctx",
        action_signature="a",
        reward=0,
        terminated=False,
        truncated=False,
        action_cost=1.0,
    )
    second = tracker.record_interaction(
        interaction_id="i2",
        before_observation={"x": 3},
        after_observation={"x": 2},
        delta={"position": [1, 2]},
        context_signature="ctx2",
        action_signature="b",
        reward=0,
        terminated=False,
        truncated=False,
        action_cost=2.0,
    )

    assert second.equivalent_outcome_cost_gap is not None
    assert second.equivalent_outcome_cost_gap >= 0


def test_efficiency_tracker_summary_contains_requested_fields() -> None:
    tracker = EfficiencyTracker()
    tracker.record_interaction(
        interaction_id="i1",
        before_observation={"x": 1},
        after_observation={"x": 2},
        delta={"changed_cells": []},
        context_signature="ctx",
        action_signature="a",
        reward=0,
        terminated=False,
        truncated=False,
    )

    summary = tracker.summary()

    assert "efficiency_event_count" in summary
    assert "no_effect_action_count" in summary
    assert "repeated_state_count" in summary
    assert "repeated_context_action_count" in summary
    assert "distinct_outcome_count" in summary


def test_efficiency_tracker_apply_future_option_deltas() -> None:
    tracker = EfficiencyTracker()
    tracker.record_interaction(
        interaction_id="i1",
        before_observation={"x": 1},
        after_observation={"x": 2},
        delta={"position": [1, 2]},
        context_signature="ctx",
        action_signature="a",
        reward=0,
        terminated=False,
        truncated=False,
        action_cost=1.0,
    )
    tracker.record_interaction(
        interaction_id="i2",
        before_observation={"x": 2},
        after_observation={"x": 3},
        delta={"position": [1, 3]},
        context_signature="ctx",
        action_signature="b",
        reward=0,
        terminated=False,
        truncated=False,
        action_cost=2.0,
    )

    tracker.apply_future_option_deltas({"i1": 2.0})

    assert tracker.events[0].future_option_gain_per_cost == 2.0
    assert tracker.events[1].future_option_gain_per_cost is None
    assert tracker.summary()["mean_future_option_gain_per_cost"] == 2.0


def test_v05c_post_run_future_option_efficiency_enrichment_is_legacy_removed(tmp_path) -> None:
    db_path = tmp_path / "future_option.sqlite"
    with sqlite3.connect(db_path) as connection:
        interaction_store = InteractionStore(connection)
        contingency_store = ContingencyStore(connection)
        interaction_store.add(
            Interaction(
                id=1,
                timestamp=1,
                observation_before=np.zeros((2, 2), dtype=int),
                action=1,
                observation_after=np.ones((2, 2), dtype=int),
                delta_id=1,
                efficiency_action_cost=2.0,
            )
        )
        contingency_store.add_prediction_result(
            interaction_id=1,
            context_level=0,
            context_signature=(1,),
            action=1,
            predicted_family=10,
            actual_family=10,
            episode_id=0,
            efficiency_action_cost=2.0,
        )
        connection.commit()

    deltas = _future_option_deltas_by_interaction_id(db_path, horizon=2)
    _apply_future_option_efficiency_diagnostics(db_path, deltas)

    assert deltas == {}
    with sqlite3.connect(db_path) as connection:
        interaction_count = connection.execute(
            "SELECT COUNT(*) FROM interactions WHERE efficiency_future_option_gain_per_cost IS NOT NULL"
        ).fetchone()[0]
        prediction_count = connection.execute(
            "SELECT COUNT(*) FROM prediction_results WHERE efficiency_future_option_gain_per_cost IS NOT NULL"
        ).fetchone()[0]
    assert interaction_count == 0
    assert prediction_count == 0


def test_v05c_future_option_delta_lookup_missing_future_effects_table(tmp_path) -> None:
    db_path = tmp_path / "missing_future_effects.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE prediction_results (interaction_id INTEGER)")
        connection.commit()

    assert _future_option_deltas_by_interaction_id(db_path, horizon=2) == {}


def test_carrier_source_is_stored_in_sqlite_stores(tmp_path) -> None:
    db_path = tmp_path / "carrier_source.sqlite"
    with sqlite3.connect(db_path) as connection:
        interaction_store = InteractionStore(connection)
        contingency_store = ContingencyStore(connection)
        interaction_store.add(
            Interaction(
                id=1,
                timestamp=1,
                observation_before=np.zeros((2, 2), dtype=int),
                action=1,
                observation_after=np.ones((2, 2), dtype=int),
                delta_id=1,
                carrier_signature="position:1,2",
                carrier_source="cell",
            )
        )
        contingency_store.add_prediction_result(
            interaction_id=1,
            context_signature=(1,),
            action=1,
            predicted_family=1,
            actual_family=1,
            carrier_signature="position:1,2",
            carrier_source="cell",
        )
        interaction_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
        prediction_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(prediction_results)").fetchall()}
        interaction_source = connection.execute("SELECT carrier_source FROM interactions WHERE id = 1").fetchone()[0]
        prediction_source = connection.execute(
            "SELECT carrier_source FROM prediction_results WHERE interaction_id = 1"
        ).fetchone()[0]

    assert "carrier_source" in interaction_columns
    assert "carrier_source" in prediction_columns
    assert interaction_source == "cell"
    assert prediction_source == "cell"


def test_adaptive_context_fields_are_stored_in_sqlite_stores(tmp_path) -> None:
    db_path = tmp_path / "adaptive_context.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=1,
            max_context_depth=3,
            adaptive_context_expansion=True,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )
    try:
        system.run_step()
    finally:
        system.close()

    with sqlite3.connect(db_path) as connection:
        interaction_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(interactions)").fetchall()}
        prediction_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(prediction_results)").fetchall()}

    assert "context_depth_used" in interaction_columns
    assert "adaptive_context_expansion_applied" in interaction_columns
    assert "adaptive_context_depth_after" in interaction_columns
    assert "context_depth_used" in prediction_columns
    assert "adaptive_context_expansion_applied" in prediction_columns
    assert "adaptive_context_depth_after" in prediction_columns


def test_interaction_significance_learning_value_drops_with_high_prior_counts() -> None:
    score_low = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id="d1",
        context_signature=None,
        memory_counts={"delta_id:d1": 0},
        graph_counts={},
    )
    score_high = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id="d1",
        context_signature=None,
        memory_counts={"delta_id:d1": 100},
        graph_counts={},
    )

    assert score_low.learning_value > score_high.learning_value


def test_interaction_significance_terminal_signal_boosts_survival_impact() -> None:
    score = compute_interaction_significance(
        terminated=True,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id="f1",
        delta_id="d1",
        context_signature="c1",
        memory_counts={},
        graph_counts={},
    )

    assert score.survival_impact >= 0.50


def test_interaction_significance_win_uses_engine_outcome_state() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id=None,
        context_signature="ctx",
        memory_counts={"outcome_state:WIN": 0, "context_outcome:ctx|WIN": 0},
        graph_counts={},
        outcome_state="WIN",
        outcome_polarity=None,
    )

    assert score.survival_impact == 0.75
    assert score.outcome_polarity == "positive"
    assert score.learning_value >= 0.99


def test_interaction_significance_game_over_uses_negative_outcome_polarity() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id=None,
        context_signature="ctx",
        memory_counts={"outcome_state:GAME_OVER": 0},
        graph_counts={},
        outcome_state="GAME_OVER",
        outcome_polarity=None,
    )

    assert score.survival_impact == 1.0
    assert score.outcome_polarity == "negative"


def test_interaction_significance_repeated_win_has_lower_learning_value() -> None:
    score_low = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id=None,
        context_signature="ctx",
        memory_counts={
            "outcome_state:WIN": 0,
            "context_outcome:ctx|WIN": 0,
            "context_signature:ctx": 0,
        },
        graph_counts={},
        outcome_state="WIN",
        outcome_polarity="positive",
    )
    score_high = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id=None,
        context_signature="ctx",
        memory_counts={
            "outcome_state:WIN": 100,
            "context_outcome:ctx|WIN": 100,
            "context_signature:ctx": 100,
        },
        graph_counts={},
        outcome_state="WIN",
        outcome_polarity="positive",
    )

    assert score_low.learning_value > score_high.learning_value


def test_interaction_significance_not_finished_keeps_low_survival_without_other_signal() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id=None,
        context_signature="ctx",
        memory_counts={},
        graph_counts={},
        outcome_state="NOT_FINISHED",
        outcome_polarity="neutral",
    )

    assert score.survival_impact == 0.0


def test_interaction_significance_graph_proxy_boosts_explanatory_potential() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id="f1",
        delta_id="d1",
        context_signature="c1",
        memory_counts={},
        graph_counts={"new_contingency": 1},
    )

    assert score.explanatory_potential >= 0.5


def test_v6_system_run_step_stores_isf_fields(tmp_path) -> None:
    db_path = tmp_path / "isf.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )

    system.run(steps=3)
    system.close()

    connection = __import__("sqlite3").connect(db_path)
    try:
        row = connection.execute(
            """
            SELECT
                isf_version,
                isf_total,
                isf_survival_impact,
                isf_prediction_error,
                isf_learning_value,
                isf_transfer_potential,
                isf_explanatory_potential,
                isf_weights_json
            FROM interactions
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        assert row[0] == "isf_v02"
        assert all(value is not None for value in row[1:])
    finally:
        connection.close()


def test_levels_completed_change_marks_level_completed_event_without_inventing_state() -> None:
    class Raw:
        def __init__(self, frame, state="NOT_FINISHED", levels_completed=0):
            self.frame = frame
            self.state = state
            self.levels_completed = levels_completed
            self.available_actions = [1]

    class FakeInnerEnv:
        def __init__(self):
            self.step_count = 0

        def reset(self):
            return Raw(np.zeros((2, 2), dtype=int), levels_completed=0)

        def step(self, _action):
            self.step_count += 1
            return Raw(np.ones((2, 2), dtype=int), state="NOT_FINISHED", levels_completed=1)

        def available_actions(self):
            return [1]

    previous = sys.modules.get("arcengine")
    sys.modules["arcengine"] = types.SimpleNamespace(GameAction=types.SimpleNamespace(from_id=lambda value: value))
    try:
        env = object.__new__(ArcGridEnvironment)
        env.env = FakeInnerEnv()
        env.auto_reset_on_empty_frame = True
        env.reset_count = 0
        env.skipped_terminal_steps = 0
        env.last_step_was_reset_boundary = False
        env.last_terminal_state = None
        env.last_outcome_state = "NOT_FINISHED"
        env.last_outcome_polarity = "neutral"
        env.last_levels_completed = 0
        env.level_completed_event = False
        env._last_raw = env.env.reset()
        env._last_grid = np.zeros((2, 2), dtype=int)

        env.step(1)

        assert env.last_outcome_state == "NOT_FINISHED"
        assert env.last_outcome_polarity == "positive"
        assert env.level_completed_event is True
        assert env.last_levels_completed == 1
    finally:
        if previous is None:
            sys.modules.pop("arcengine", None)
        else:
            sys.modules["arcengine"] = previous


def test_adapter_preserves_engine_states_without_invented_labels() -> None:
    class Raw:
        def __init__(self, frame, state, levels_completed=0):
            self.frame = frame
            self.state = state
            self.levels_completed = levels_completed
            self.available_actions = [1]

    class FakeInnerEnv:
        def __init__(self, raw):
            self._raw = raw

        def reset(self):
            return Raw(np.zeros((2, 2), dtype=int), "NOT_FINISHED", 0)

        def step(self, _action):
            return self._raw

    previous = sys.modules.get("arcengine")
    sys.modules["arcengine"] = types.SimpleNamespace(GameAction=types.SimpleNamespace(from_id=lambda value: value))
    try:
        for state_name, expected_polarity in (("WIN", "positive"), ("GAME_OVER", "negative"), ("NOT_FINISHED", "neutral")):
            env = object.__new__(ArcGridEnvironment)
            env.env = FakeInnerEnv(Raw(np.ones((2, 2), dtype=int), state_name, 0))
            env.auto_reset_on_empty_frame = True
            env.reset_count = 0
            env.skipped_terminal_steps = 0
            env.last_step_was_reset_boundary = False
            env.last_terminal_state = None
            env.last_outcome_state = "NOT_FINISHED"
            env.last_outcome_polarity = "neutral"
            env.last_levels_completed = 0
            env.level_completed_event = False
            env._last_raw = env.env.reset()
            env._last_grid = np.zeros((2, 2), dtype=int)

            env.step(1)

            assert env.last_outcome_state == state_name
            assert env.last_outcome_polarity == expected_polarity
    finally:
        if previous is None:
            sys.modules.pop("arcengine", None)
        else:
            sys.modules["arcengine"] = previous


def test_interaction_significance_level_completed_event_has_positive_survival_signal() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id=None,
        context_signature="ctx",
        memory_counts={"level_completed_event:true": 0, "context_level_completed:ctx|true": 0},
        graph_counts={},
        outcome_state="NOT_FINISHED",
        outcome_polarity=None,
        level_completed_event=True,
    )

    assert score.survival_impact >= 0.50
    assert score.outcome_polarity == "positive"
    assert score.version == "isf_v02"


def test_interaction_significance_level_completed_event_learning_value_is_high_when_novel() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id=None,
        context_signature="ctx",
        memory_counts={"level_completed_event:true": 0, "context_level_completed:ctx|true": 0},
        graph_counts={},
        outcome_state="NOT_FINISHED",
        outcome_polarity="positive",
        level_completed_event=True,
    )

    assert score.survival_impact >= 0.50
    assert score.learning_value >= 0.99


def test_v6_system_run_step_stores_level_completed_event_fields(tmp_path) -> None:
    db_path = tmp_path / "outcomes.sqlite"
    system = V6System(
        env=OutcomeEnv(outcome_state="NOT_FINISHED", outcome_polarity="positive", starting_level=0, next_level=1),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )

    system.run_step()
    system.close()

    connection = sqlite3.connect(db_path)
    try:
        interaction_row = connection.execute(
            """
            SELECT outcome_state, outcome_polarity, level_completed_event
            FROM interactions
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        prediction_row = connection.execute(
            """
            SELECT outcome_state, outcome_polarity, level_completed_event
            FROM prediction_results
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()

    assert interaction_row == ("NOT_FINISHED", "positive", 1)
    assert prediction_row == ("NOT_FINISHED", "positive", 1)


def test_v6_system_applies_post_factum_level_completion_credit(tmp_path) -> None:
    db_path = tmp_path / "post_factum.sqlite"
    system = V6System(
        env=LevelCompletionSequenceEnv(completion_step=4, terminal_state="NOT_FINISHED", increment_levels=True),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )

    system.run(steps=4)
    replay_candidates = dict(system.memory_lifecycle.replay_candidates)
    system.close()

    connection = sqlite3.connect(db_path)
    try:
        interaction_rows = connection.execute(
            """
            SELECT id, outcome_state, level_completed_event, post_factum_level_completion_credit, post_factum_credit_reason
            FROM interactions
            ORDER BY id
            """
        ).fetchall()
        prediction_rows = connection.execute(
            """
            SELECT interaction_id, post_factum_level_completion_credit, post_factum_level_completion_decay, post_factum_level_completion_step, post_factum_credit_reason,
                   memory_replay_candidate, memory_replay_priority, memory_retention_reason
            FROM prediction_results
            ORDER BY interaction_id
            """
        ).fetchall()
        interaction_replay_rows = connection.execute(
            """
            SELECT id, memory_replay_candidate, memory_replay_priority, memory_retention_reason
            FROM interactions
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    assert [row[0] for row in interaction_rows] == [1, 2, 3, 4]
    credits = [float(row[3]) for row in interaction_rows]
    assert credits[3] > credits[2] > credits[1] > credits[0] > 0.0
    assert all(row[1] == "NOT_FINISHED" for row in interaction_rows)
    assert [row[2] for row in interaction_rows] == [0, 0, 0, 1]
    assert all(row[4] == "levels_completed_increment" for row in interaction_rows)
    assert len(prediction_rows) == 4
    assert [float(row[1]) for row in prediction_rows] == credits
    assert all(row[3] == 4 for row in prediction_rows)
    assert all(row[4] == "levels_completed_increment" for row in prediction_rows)
    assert {"1", "2", "3", "4"}.issubset(set(replay_candidates))
    assert all(replay_candidates[str(i)].replay_priority > 0.0 for i in range(1, 5))
    for i in range(1, 5):
        parts = replay_candidates[str(i)].reason.split("+")
        assert "post_factum_level_completion" in parts
    assert all(int(row[1]) == 1 for row in interaction_replay_rows)
    assert all(float(row[2]) > 0.0 for row in interaction_replay_rows)
    for row in interaction_replay_rows:
        parts = str(row[3]).split("+")
        assert "post_factum_level_completion" in parts
    assert all(int(row[5]) == 1 for row in prediction_rows)
    assert all(float(row[6]) > 0.0 for row in prediction_rows)
    for row in prediction_rows:
        parts = str(row[7]).split("+")
        assert "post_factum_level_completion" in parts


def test_v6_system_game_over_without_progress_gets_no_post_factum_credit(tmp_path) -> None:
    db_path = tmp_path / "post_factum_game_over.sqlite"
    system = V6System(
        env=LevelCompletionSequenceEnv(completion_step=4, terminal_state="GAME_OVER"),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )

    system.run(steps=4)
    replay_candidates = dict(system.memory_lifecycle.replay_candidates)
    system.close()

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, outcome_state, level_completed_event, post_factum_level_completion_credit
            FROM interactions
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    assert rows[-1][1] == "GAME_OVER"
    assert all(int(row[2]) == 0 for row in rows)
    assert all(float(row[3]) == 0.0 for row in rows)
    assert not any(candidate.reason == "post_factum_level_completion" for candidate in replay_candidates.values())


def test_v6_system_applies_win_trajectory_credit(tmp_path) -> None:
    db_path = tmp_path / "trajectory_win.sqlite"
    system = V6System(
        env=LevelCompletionSequenceEnv(completion_step=4, terminal_state="WIN", increment_levels=False),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )
    system.run(steps=4)
    replay_candidates = dict(system.memory_lifecycle.replay_candidates)
    system.close()

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, post_factum_trajectory_credit, post_factum_trajectory_credit_kind,
                   post_factum_trajectory_credit_polarity, post_factum_trajectory_credit_reason,
                   memory_replay_candidate, memory_replay_priority, memory_retention_reason
            FROM interactions
            ORDER BY id
            """
        ).fetchall()
        prediction_rows = connection.execute(
            """
            SELECT interaction_id, memory_replay_candidate, memory_replay_priority, memory_retention_reason
            FROM prediction_results
            ORDER BY interaction_id
            """
        ).fetchall()
    finally:
        connection.close()

    credits = [float(row[1]) for row in rows]
    assert credits[3] > credits[2] > credits[1] > credits[0] > 0.0
    assert all(row[2] == "WIN" for row in rows)
    assert all(row[3] == "positive" for row in rows)
    assert all(row[4] == "win_terminal_trajectory" for row in rows)
    assert {"1", "2", "3", "4"}.issubset(set(replay_candidates))
    for i in range(1, 5):
        parts = replay_candidates[str(i)].reason.split("+")
        assert "win_terminal_trajectory" in parts
    assert all(int(row[5]) == 1 for row in rows)
    assert all(float(row[6]) > 0.0 for row in rows)
    for row in rows:
        parts = str(row[7]).split("+")
        assert "win_terminal_trajectory" in parts
    assert all(int(row[1]) == 1 for row in prediction_rows)
    assert all(float(row[2]) > 0.0 for row in prediction_rows)
    for row in prediction_rows:
        parts = str(row[3]).split("+")
        assert "win_terminal_trajectory" in parts


def test_v6_system_applies_game_over_trajectory_credit(tmp_path) -> None:
    db_path = tmp_path / "trajectory_game_over.sqlite"
    system = V6System(
        env=LevelCompletionSequenceEnv(completion_step=4, terminal_state="GAME_OVER", increment_levels=False),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )
    system.run(steps=4)
    replay_candidates = dict(system.memory_lifecycle.replay_candidates)
    system.close()

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, post_factum_trajectory_credit, post_factum_trajectory_credit_kind,
                   post_factum_trajectory_credit_polarity, post_factum_trajectory_credit_reason,
                   memory_replay_candidate, memory_replay_priority, memory_retention_reason
            FROM interactions
            ORDER BY id
            """
        ).fetchall()
        prediction_rows = connection.execute(
            """
            SELECT interaction_id, memory_replay_candidate, memory_replay_priority, memory_retention_reason
            FROM prediction_results
            ORDER BY interaction_id
            """
        ).fetchall()
    finally:
        connection.close()

    credits = [float(row[1]) for row in rows]
    assert credits[3] > credits[2] > credits[1] > credits[0] > 0.0
    assert all(row[2] == "GAME_OVER" for row in rows)
    assert all(row[3] == "negative" for row in rows)
    assert all(row[4] == "game_over_failure_path" for row in rows)
    assert {"1", "2", "3", "4"}.issubset(set(replay_candidates))
    for i in range(1, 5):
        parts = replay_candidates[str(i)].reason.split("+")
        assert "game_over_failure_path" in parts
    assert all(int(row[5]) == 1 for row in rows)
    assert all(float(row[6]) > 0.0 for row in rows)
    for row in rows:
        parts = str(row[7]).split("+")
        assert "game_over_failure_path" in parts
    assert all(int(row[1]) == 1 for row in prediction_rows)
    assert all(float(row[2]) > 0.0 for row in prediction_rows)
    for row in prediction_rows:
        parts = str(row[3]).split("+")
        assert "game_over_failure_path" in parts


def test_v6_system_not_finished_without_progress_has_no_trajectory_credit(tmp_path) -> None:
    db_path = tmp_path / "trajectory_none.sqlite"
    system = V6System(
        env=LevelCompletionSequenceEnv(completion_step=99, terminal_state="NOT_FINISHED", increment_levels=False),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )
    system.run(steps=4)
    system.close()

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT post_factum_trajectory_credit FROM interactions ORDER BY id"
        ).fetchall()
    finally:
        connection.close()

    assert all(float(row[0]) == 0.0 for row in rows)


def test_v6_system_combined_level_completion_and_win_populates_both_credit_paths(tmp_path) -> None:
    db_path = tmp_path / "trajectory_combined.sqlite"
    system = V6System(
        env=LevelCompletionSequenceEnv(completion_step=4, terminal_state="WIN", increment_levels=True),
        config=V6Config(
            database_path=str(db_path),
            recluster_every=2,
            min_cluster_size=2,
            context_length=1,
            contingency_support_threshold=1,
            contingency_confidence_threshold=0.0,
            random_seed=0,
        ),
    )
    system.run(steps=4)
    replay_candidates = dict(system.memory_lifecycle.replay_candidates)
    system.close()

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, level_completed_event, post_factum_level_completion_credit, post_factum_trajectory_credit,
                   memory_replay_candidate, memory_replay_priority, memory_retention_reason
            FROM interactions
            ORDER BY id
            """
        ).fetchall()
        prediction_rows = connection.execute(
            """
            SELECT interaction_id, memory_replay_candidate, memory_replay_priority, memory_retention_reason
            FROM prediction_results
            ORDER BY interaction_id
            """
        ).fetchall()
    finally:
        connection.close()

    assert rows[-1][1] == 1
    assert all(float(row[2]) > 0.0 for row in rows)
    assert all(float(row[3]) > 0.0 for row in rows)
    assert {"1", "2", "3", "4"}.issubset(set(replay_candidates))
    for i in range(1, 5):
        reason = replay_candidates[str(i)].reason
        assert "post_factum_level_completion" in reason.split("+")
        assert "win_terminal_trajectory" in reason.split("+")
    assert all(int(row[4]) == 1 for row in rows)
    assert all(float(row[5]) > 0.0 for row in rows)
    for row in rows:
        parts = str(row[6]).split("+")
        assert "post_factum_level_completion" in parts
        assert "win_terminal_trajectory" in parts
    assert all(int(row[1]) == 1 for row in prediction_rows)
    assert all(float(row[2]) > 0.0 for row in prediction_rows)
    for row in prediction_rows:
        parts = str(row[3]).split("+")
        assert "post_factum_level_completion" in parts
        assert "win_terminal_trajectory" in parts


def test_v6_system_prediction_edges_add_explains_and_depends_on() -> None:
    system = V6System(env=ToggleEnv(), config=V6Config(database_path=":memory:", random_seed=0))

    system._add_prediction_explanation_edges(
        interaction_id=1,
        prediction_correct=True,
        prediction_confidence=0.8,
        predicted_family=9,
        actual_family=9,
        actual_context_signature=("ctx",),
        selected_contingency=type("ContingencyRef", (), {"id": 7})(),
    )

    assert system.graph.count_edges_of_type("explains") >= 1
    assert system.graph.count_edges_of_type("depends_on") >= 1


def test_v6_system_prediction_edges_add_contradicts() -> None:
    system = V6System(env=ToggleEnv(), config=V6Config(database_path=":memory:", random_seed=0))

    system._add_prediction_explanation_edges(
        interaction_id=1,
        prediction_correct=False,
        prediction_confidence=0.6,
        predicted_family=8,
        actual_family=9,
        actual_context_signature=("ctx",),
        selected_contingency=type("ContingencyRef", (), {"id": 7})(),
    )

    assert system.graph.count_edges_of_type("contradicts") >= 1
    assert system.graph.count_edges_of_type("depends_on") >= 1


def test_v6_system_run_step_stores_context_contradiction_fields(tmp_path) -> None:
    from v6.contingency.contingency_learner import Contingency

    db_path = tmp_path / "contradictions.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=2,
            max_context_depth=5,
            random_seed=0,
        ),
    )

    fake_contingency = Contingency(
        id=7,
        context_level=2,
        context_signature=("ctx1",),
        action=1,
        transformation_family=1,
        support_count=3,
        confidence=0.9,
    )
    system.predictor.predict_multi_scale = lambda *_args, **_kwargs: 1
    system._prediction_confidence = lambda **_kwargs: 0.9
    system.contingency_learner.best_stable_for_action = lambda *_args, **_kwargs: fake_contingency
    system.contingency_learner.update_multi_scale = lambda *_args, **_kwargs: fake_contingency
    system.contingency_learner.stable_contingencies = lambda: [fake_contingency]
    system.clusterer.family_for_delta = lambda _delta_id: 2
    system.clusterer.families = {2: object()}

    system.run_step()
    system.close()

    connection = __import__("sqlite3").connect(db_path)
    try:
        row = connection.execute(
            """
            SELECT
                context_contradiction,
                context_contradiction_key,
                context_expansion_suggested,
                suggested_context_depth,
                context_contradiction_reason
            FROM prediction_results
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        assert row[0] == 1
        assert "|a1|1->2" in str(row[1])
        assert row[2] == 0
        assert row[3] == 3
        assert row[4] == "confident_wrong_prediction_same_context"
    finally:
        connection.close()

    assert system.graph.count_edges_of_type("contradicts") >= 1
    assert any(
        str(source).startswith("Context:")
        and str(attrs.get("type", attrs.get("edge_type"))) == "contradicts"
        for source, _target, attrs in system.graph.graph.edges(data=True)
    )


def test_v6_system_run_step_stores_carrier_fields(tmp_path) -> None:
    db_path = tmp_path / "carrier.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=2,
            carrier_min_support=1,
            carrier_min_distinct_contexts=1,
            carrier_min_prediction_lift=0.0,
            carrier_min_compression_gain=0.0,
            random_seed=0,
        ),
    )

    system.run_step()
    system.close()

    connection = __import__("sqlite3").connect(db_path)
    try:
        interaction_row = connection.execute(
            """
            SELECT
                carrier_signature,
                carrier_event_recorded,
                carrier_support_count,
                carrier_distinct_family_count,
                carrier_distinct_context_count
            FROM interactions
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        prediction_row = connection.execute(
            """
            SELECT
                carrier_signature,
                carrier_event_recorded,
                carrier_support_count,
                carrier_distinct_family_count,
                carrier_distinct_context_count
            FROM prediction_results
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        assert interaction_row is not None
        assert prediction_row is not None
        assert interaction_row[0] is not None
        assert interaction_row[1] == 1
        assert interaction_row[2] >= 1
        assert prediction_row[0] is not None
        assert prediction_row[1] == 1
    finally:
        connection.close()


def test_v6_system_run_step_stores_memory_lifecycle_fields(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=2,
            random_seed=0,
            memory_min_records_before_forgetting=1,
            memory_max_active_records=10,
        ),
    )

    system.run_step()
    system.close()

    connection = __import__("sqlite3").connect(db_path)
    try:
        interaction_row = connection.execute(
            """
            SELECT
                memory_status,
                memory_retention_reason,
                memory_replay_priority,
                memory_replay_candidate,
                memory_replay_count
            FROM interactions
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        prediction_row = connection.execute(
            """
            SELECT
                memory_status,
                memory_retention_reason,
                memory_replay_priority,
                memory_replay_candidate,
                memory_replay_count
            FROM prediction_results
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        assert interaction_row is not None
        assert prediction_row is not None
        assert interaction_row[0] in {"active", "protected", "compressed", "forgotten"}
        assert interaction_row[1] is not None
        assert interaction_row[2] is not None
        assert prediction_row[0] in {"active", "protected", "compressed", "forgotten"}
        assert prediction_row[1] is not None
    finally:
        connection.close()


def test_memory_substrate_creates_all_tables() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {"memory_nodes", "memory_edges", "memory_evidence", "memory_scores", "memory_promotions", "memory_versions"}.issubset(tables)
        assert substrate.get_node("missing") is None
    finally:
        connection.close()


def test_hypothesis_suite_aggregated_text_concatenates_hypothesis_reports(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    expected_bodies: list[str] = []
    for hypothesis_id in range(1, 12):
        label = f"h{hypothesis_id:02d}"
        body = f"H{hypothesis_id:02d} report body"
        expected_bodies.append(body)
        subdir = reports_dir / label
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / f"{label}_report.txt").write_text(body + "\n", encoding="utf-8")

    _write_aggregated_hypothesis_text(reports_dir)

    aggregated = (reports_dir / "hypothesis_suite_aggregated.txt").read_text(encoding="utf-8")
    for body in expected_bodies:
        assert body in aggregated
    positions = [aggregated.index(body) for body in expected_bodies]
    assert positions == sorted(positions)


def test_hypothesis_suite_aggregated_text_uses_result_dicts_without_unavailable(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    _write_aggregated_hypothesis_text(
        reports_dir,
        hypothesis_results={
            "H01": {
                "decision": "VALID",
                "core_metrics": {"stable_contingency_count": 3},
                "missing_evidence": [],
                "evidence_diagnostics": {"source": "compact"},
            },
            "H02": {
                "decision": "INSUFFICIENT_EVIDENCE",
                "core_metrics": {"evidence_coverage_ratio": 0.4},
                "missing_evidence": ["low coverage"],
                "evidence_diagnostics": {"source": "manifest"},
            },
        },
    )
    aggregated = (reports_dir / "hypothesis_suite_aggregated.txt").read_text(encoding="utf-8")
    assert "H01" in aggregated
    assert "stable_contingency_count" in aggregated
    assert "report unavailable" not in aggregated


def test_hypothesis_suite_aggregated_text_on_epoch01_real_reports() -> None:
    reports_dir = Path("runs/v6/continuous/full_breadth_continuous/epochs/epoch_0001/reports")
    if not reports_dir.exists():
        pytest.skip("epoch_0001 real reports directory not present")
    _write_aggregated_hypothesis_text(reports_dir)
    aggregated_path = reports_dir / "hypothesis_suite_aggregated.txt"
    assert aggregated_path.exists()
    aggregated = aggregated_path.read_text(encoding="utf-8")
    expected_fragments = [
        "H01",
        "H02",
        "H03",
        "H04",
        "H05",
        "H06",
        "H07",
        "H08",
        "H09",
        "H10",
        "H11",
    ]
    for fragment in expected_fragments:
        assert fragment in aggregated
    txt_paths = [
        reports_dir / f"h{hypothesis_id:02d}" / next(
            path.name
            for path in sorted((reports_dir / f"h{hypothesis_id:02d}").glob("*.txt"))
        )
        for hypothesis_id in range(1, 12)
    ]
    first_lines = [path.read_text(encoding="utf-8").strip().splitlines()[0] for path in txt_paths]
    positions = [aggregated.index(line) for line in first_lines]
    assert positions == sorted(positions)


def test_h01_txt_checklist_matches_json_checklist(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence

    run_dir = tmp_path / "run_h01_txt"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "game": "g1",
                        "sampler_name": "s1",
                        "total_interactions": 10,
                        "memory_record_count": 10,
                        "stable_contingency_count": 3,
                        "discovered_contingency_count": 3,
                        "prediction_accuracy": 0.5,
                        "context_lift": 0.2,
                    },
                    {
                        "game": "g2",
                        "sampler_name": "s2",
                        "total_interactions": 12,
                        "memory_record_count": 12,
                        "stable_contingency_count": 2,
                        "discovered_contingency_count": 2,
                        "prediction_accuracy": 0.6,
                        "context_lift": 0.1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_h01_contingency_emergence(run_dir, tmp_path / "out_h01_txt")
    txt = (tmp_path / "out_h01_txt" / "h01_contingency_emergence_report.txt").read_text(encoding="utf-8")
    for key, value in result["acceptance_checks"].items():
        assert f"- {key}: {value}" in txt


def test_h01_compact_only_can_be_valid(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence

    run_dir = tmp_path / "run_h01_compact"
    run_dir.mkdir()
    memory_dir = tmp_path / "memory_h01_compact"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error, mean_replay_priority, representative_example_count, prediction_attempt_count, prediction_success_count, prediction_accuracy, prediction_error_before, prediction_error_after, normalized_contingency_key) VALUES (1,'c1','g1','s1',1,1,'e1',25,1,10,0.9,0.1,0.5,1,10,8,0.8,0.3,0.1,'k1')")
        conn.execute("INSERT INTO stable_contingencies (contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error, mean_replay_priority, representative_example_count, prediction_attempt_count, prediction_success_count, prediction_accuracy, prediction_error_before, prediction_error_after, normalized_contingency_key) VALUES (2,'c2','g2','s2',1,1,'e1',25,1,10,0.9,0.1,0.5,1,10,7,0.7,0.3,0.1,'k1')")
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES ('M0:interaction:g1','M0','InteractionMemory','i1',1,'{}')")
        conn.execute("INSERT INTO memory_scores (node_id, replay_priority) VALUES ('M0:interaction:g1', 0.5)")
        conn.execute("INSERT INTO memory_summary (key, value_json) VALUES ('total_interactions_seen', '100')")
        conn.commit()
    result = evaluate_h01_contingency_emergence(run_dir, tmp_path / "out_h01_compact", memory_dir=memory_dir)
    assert result["decision"] == "VALID"


def test_h01_direct_streaming_raw_cleanup_uses_compact_evidence(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h01_streamed"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1"}]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h01_streamed"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES ('M0:interaction:g1','M0','InteractionMemory','i1',1,'{}')")
        conn.execute("INSERT INTO stable_contingencies (contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error, mean_replay_priority, representative_example_count, prediction_attempt_count, prediction_success_count, prediction_accuracy, prediction_error_before, prediction_error_after, normalized_contingency_key) VALUES (1,'c1','g1','s1',1,1,'e1',25,1,10,0.9,0.1,0.5,1,10,8,0.8,0.3,0.1,'k1')")
        conn.commit()
    result = evaluate_h01_contingency_emergence(run_dir, tmp_path / "out_h01_streamed", memory_dir=memory_dir)
    assert result["evidence_source"] == "direct_streaming_manifest_and_compact_memory"
    assert result["decision"] != "INVALID"
    assert result["stable_contingency_count"] > 0


def test_h02_low_coverage_demotes_decision_and_txt_prints_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention

    run_dir = tmp_path / "run_h02_low_cov"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": f"g{i}", "sampler_name": "s", "mean_isf_prediction_error": 0.5, "memory_replay_candidate_count": 1, "high_priority_replay_count": 1, "context_contradiction_count": 1, "repeated_contradiction_count": 1, "context_expansion_suggested_count": 1} for i in range(10)]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h02_low_cov"
    ensure_memory_layout(memory_dir)
    manifest_path = memory_dir / "direct_streaming_fold_manifest.sqlite"
    with sqlite3.connect(manifest_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS folded_jobs (job_id TEXT PRIMARY KEY, status TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS job_metrics (job_id TEXT PRIMARY KEY, game TEXT, sampler TEXT, seed INTEGER, metrics_json TEXT NOT NULL)")
        for i in range(4):
            conn.execute("INSERT INTO job_metrics (job_id, game, sampler, seed, metrics_json) VALUES (?, ?, ?, ?, ?)", (f'j{i}', f'g{i}', 's', 0, '{}'))
        conn.commit()
    monkeypatch.setattr(
        "v6.hypothesis_h02_report._compute_prediction_violation_replay_lift_from_existing_db",
        lambda *args, **kwargs: {"direct_replay_lift_available": True, "prediction_violation_replay_lift": 2.0, "sqlite_db_count_inspected": 1},
    )
    result = evaluate_h02_prediction_violation_attention(run_dir, tmp_path / "out_h02_low_cov", memory_dir=memory_dir)
    assert result["h02a_replay_attention_decision"] == "PARTIALLY_VALID_WITH_LOW_COVERAGE"
    txt = (tmp_path / "out_h02_low_cov" / "h02_prediction_violation_attention_report.txt").read_text(encoding="utf-8")
    assert "total_jobs_expected:" in txt
    assert "jobs_represented_in_compact_or_manifest_evidence:" in txt
    assert "evidence_coverage_ratio:" in txt


def test_h10_txt_prints_subtests(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h10_txt"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        for idx in range(5):
            conn.execute(
                "INSERT INTO future_option_attention_links (event_id, motif_signature, owner_type, owner_key, option_delta_abs, replay_priority_score, memory_priority_score, contradiction_score, high_option_change, high_attention) VALUES (?, 'm', 'interaction', ?, 2.0, 0.9, 0.7, 0.1, 1, 1)",
                (f'h{idx}', f'hi{idx}'),
            )
            conn.execute(
                "INSERT INTO future_option_attention_links (event_id, motif_signature, owner_type, owner_key, option_delta_abs, replay_priority_score, memory_priority_score, contradiction_score, high_option_change, high_attention) VALUES (?, 'm', 'interaction', ?, 0.1, 0.1, 0.1, 0.0, 0, 0)",
                (f'l{idx}', f'lo{idx}'),
            )
        conn.commit()
    evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10_txt", already_derived=True)
    txt = (tmp_path / "h10_txt" / "h10_future_option_attention_report.txt").read_text(encoding="utf-8")
    assert "H10A:" in txt
    assert "H10B:" in txt
    assert "H10C:" in txt
    assert "H10D:" in txt


def test_h08_label_matches_world_model_coherence_implementation(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h08_label"
    ensure_memory_layout(memory_dir)
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08_label", already_derived=True)
    assert "World-model coherence" in str(result.get("hypothesis_name"))


def test_compact_memory_restore_retries_locked_query_reads() -> None:
    class _FakeConnection:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, query: str, params: tuple = ()) -> object:
            del query, params
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("database is locked")

            class _Cursor:
                def fetchall(self_inner) -> list[tuple[str]]:
                    del self_inner
                    return [("ok",)]

            return _Cursor()

    fake = _FakeConnection()
    rows = _fetchall_readonly_with_retry(fake, "SELECT 1", attempts=3, base_delay_seconds=0.0)
    assert rows == [("ok",)]
    assert fake.calls == 2


def test_memory_substrate_upsert_node_increments_support_count() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        substrate.upsert_node(
            MemoryNode(node_id="M0:interaction:1", memory_level="M0", node_type="InteractionMemory", attrs={"a": 1}),
            step=1,
        )
        substrate.upsert_node(
            MemoryNode(node_id="M0:interaction:1", memory_level="M0", node_type="InteractionMemory", attrs={"a": 2}),
            step=2,
            support_increment=2,
        )
        row = substrate.get_node("M0:interaction:1")
        assert row is not None
        assert row["support_count"] == 3
        assert row["attrs"]["a"] == 2
        assert row["first_seen_step"] == 1
        assert row["last_seen_step"] == 2
    finally:
        connection.close()


def test_memory_substrate_upsert_edge_merges_duplicates() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        substrate.upsert_edge(MemoryEdge("A", "B", "follows", weight=0.5, evidence={"x": 1}))
        substrate.upsert_edge(MemoryEdge("A", "B", "follows", weight=0.8, evidence={"y": 2}), support_increment=2)
        edges = substrate.edges_from("A", "follows")
        assert len(edges) == 1
        assert edges[0]["support_count"] == 3
        assert edges[0]["weight"] == 0.8
        assert edges[0]["evidence"]["y"] == 2
    finally:
        connection.close()


def test_v6_system_run_step_writes_m0_memory_nodes_and_scores(tmp_path) -> None:
    db_path = tmp_path / "memory_substrate.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=2,
            random_seed=0,
        ),
    )

    result = system.run_step()
    system.close()

    connection = sqlite3.connect(db_path)
    try:
        node_ids = {
            str(row[0])
            for row in connection.execute("SELECT node_id FROM memory_nodes").fetchall()
        }
        interaction_node = next(node for node in node_ids if node.startswith("M0:interaction:"))
        assert interaction_node in node_ids
        assert action_node_id(result.action) in node_ids
        assert delta_node_id(result.delta_id) in node_ids
        observation_nodes = [
            str(row[0])
            for row in connection.execute(
                "SELECT node_id FROM memory_nodes WHERE node_type = 'ObservationMemory' ORDER BY node_id ASC"
            ).fetchall()
        ]
        assert len(observation_nodes) >= 2
        score_row = connection.execute(
            """
            SELECT isf_total, replay_priority, retention_status
            FROM memory_scores
            WHERE node_id = ?
            """,
            (interaction_node,),
        ).fetchone()
        assert score_row is not None
        assert score_row[0] is not None
        edge_types = {
            str(row[0])
            for row in connection.execute(
                "SELECT edge_type FROM memory_edges WHERE source_node_id = ?",
                (interaction_node,),
            ).fetchall()
        }
        assert {"takes_action", "produces_delta", "observed_after"}.issubset(edge_types)
    finally:
        connection.close()


def test_compact_memory_fold_and_restore_preserves_memory_substrate(tmp_path) -> None:
    db_path = tmp_path / "live.sqlite"
    memory_dir = tmp_path / "compact_memory"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=2,
            random_seed=0,
        ),
    )
    result = system.run_step()
    fold_live_system_into_compact_memory(system, memory_dir)
    system.close()

    restored = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=":memory:",
            memory_input_dir=str(memory_dir),
            restore_compact_memory=True,
            random_seed=0,
        ),
    )
    try:
        interaction_node_id_value = restored._interaction_memory_node_id(result.interaction_id)
        interaction_node = restored.memory.get_node(interaction_node_id_value)
        assert interaction_node is not None
        assert interaction_node["memory_level"] == "M0"
        edges = restored.memory.edges_from(interaction_node_id_value)
        assert any(edge["edge_type"] == "takes_action" for edge in edges)
    finally:
        restored.close()


def test_compact_memory_summary_exports_memory_record_and_replay_counts(tmp_path) -> None:
    db_path = tmp_path / "live.sqlite"
    memory_dir = tmp_path / "compact_memory"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=2,
            random_seed=0,
        ),
    )
    system.run_step()
    system.memory.upsert_score(
        MemoryScore(
            node_id=interaction_node_id(1),
            replay_priority=0.8,
            retention_status="protected",
        ),
        step=1,
    )
    system.memory.upsert_edge(MemoryEdge("replay:q1", interaction_node_id(1), "selected_for_replay"))
    summary = fold_live_system_into_compact_memory(system, memory_dir)
    system.close()

    assert int(summary.get("memory_record_count") or 0) > 0
    assert int(summary.get("memory_replay_candidate_count") or 0) > 0
    assert int(summary.get("high_replay_priority_count") or 0) > 0
    assert int(summary.get("protected_memory_count") or 0) > 0

def test_v6_system_run_step_stores_efficiency_fields(tmp_path) -> None:
    db_path = tmp_path / "efficiency.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=2,
            random_seed=0,
        ),
    )

    system.run_step()
    system.close()

    connection = __import__("sqlite3").connect(db_path)
    try:
        interaction_row = connection.execute(
            """
            SELECT
                efficiency_action_cost,
                efficiency_cumulative_cost,
                efficiency_repeated_state,
                efficiency_repeated_context_action,
                efficiency_no_effect_action,
                efficiency_terminal_outcome,
                efficiency_normalized_solve_efficiency,
                efficiency_equivalent_outcome_cost_gap
            FROM interactions
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        prediction_row = connection.execute(
            """
            SELECT
                efficiency_action_cost,
                efficiency_cumulative_cost,
                efficiency_repeated_state,
                efficiency_repeated_context_action,
                efficiency_no_effect_action,
                efficiency_terminal_outcome,
                efficiency_normalized_solve_efficiency,
                efficiency_equivalent_outcome_cost_gap
            FROM prediction_results
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        assert interaction_row is not None
        assert prediction_row is not None
        assert interaction_row[0] is not None
        assert interaction_row[1] is not None
        assert interaction_row[2] in {0, 1}
        assert interaction_row[3] in {0, 1}
        assert interaction_row[4] in {0, 1}
        assert interaction_row[5] in {0, 1}
        assert prediction_row[0] is not None
        assert prediction_row[1] is not None
    finally:
        connection.close()


def test_phase2_stable_contingency_creates_m1_memory_node(tmp_path) -> None:
    from v6.contingency.contingency_learner import Contingency

    db_path = tmp_path / "phase2_contingency.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(database_path=str(db_path), context_length=2, random_seed=0),
    )
    fake_contingency = Contingency(
        id=11,
        context_level=2,
        context_signature=("ctxA",),
        action=1,
        transformation_family=3,
        support_count=5,
        confidence=0.9,
    )
    system.contingency_learner.update_multi_scale = lambda *_args, **_kwargs: fake_contingency
    system.contingency_learner.stable_contingencies = lambda: [fake_contingency]
    system.clusterer.family_for_delta = lambda _delta_id: 3
    system.run_step()
    node = system.memory.query_nodes(memory_level="M1", node_type="ContingencyMemory")
    assert node
    assert node[0]["attrs"]["context_level"] == 2
    system.close()


def test_phase2_actual_family_creates_m2_memory_node(tmp_path) -> None:
    db_path = tmp_path / "phase2_family.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(database_path=str(db_path), context_length=2, random_seed=0),
    )
    system.clusterer.family_for_delta = lambda _delta_id: 7
    system.clusterer.families = {7: object()}
    result = system.run_step()
    family_node = system.memory.get_node(family_node_id(7))
    assert family_node is not None
    edges = system.memory.edges_from(system._interaction_memory_node_id(result.interaction_id), "supports")
    assert any(edge["target_node_id"] == family_node_id(7) for edge in edges)
    system.close()


def test_phase2_carrier_candidate_creates_m3_memory_node(tmp_path) -> None:
    db_path = tmp_path / "phase2_carrier.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(
            database_path=str(db_path),
            context_length=2,
            carrier_min_support=1,
            carrier_min_distinct_contexts=1,
            carrier_min_prediction_lift=0.0,
            carrier_min_compression_gain=0.0,
            random_seed=0,
        ),
    )
    system.run_step()
    carriers = system.memory.query_nodes(memory_level="M3", node_type="CarrierMemory")
    assert carriers
    system.close()


def test_phase2_prediction_contradiction_creates_violation_memory_node(tmp_path) -> None:
    from v6.contingency.contingency_learner import Contingency

    db_path = tmp_path / "phase2_violation.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(database_path=str(db_path), context_length=2, max_context_depth=5, random_seed=0),
    )
    fake_contingency = Contingency(
        id=7,
        context_level=2,
        context_signature=("ctx1",),
        action=1,
        transformation_family=1,
        support_count=3,
        confidence=0.9,
    )
    system.predictor.predict_multi_scale = lambda *_args, **_kwargs: 1
    system._prediction_confidence = lambda **_kwargs: 0.9
    system.contingency_learner.best_stable_for_action = lambda *_args, **_kwargs: fake_contingency
    system.contingency_learner.update_multi_scale = lambda *_args, **_kwargs: fake_contingency
    system.contingency_learner.stable_contingencies = lambda: [fake_contingency]
    system.clusterer.family_for_delta = lambda _delta_id: 2
    system.clusterer.families = {2: object()}
    result = system.run_step()
    outgoing = system.memory.edges_from(system._interaction_memory_node_id(result.interaction_id), "violates_prediction")
    assert outgoing
    violation_node = system.memory.get_node(outgoing[0]["target_node_id"])
    assert violation_node is not None
    assert violation_node["node_type"] == "PredictionViolationMemory"
    system.close()


def test_phase2_replay_candidate_updates_memory_scores_and_replay_edge(tmp_path) -> None:
    db_path = tmp_path / "phase2_replay.sqlite"
    system = V6System(
        env=OutcomeEnv(outcome_state="WIN", outcome_polarity="positive"),
        config=V6Config(database_path=str(db_path), context_length=2, random_seed=0),
    )
    system.run_step()
    interaction_node = system._interaction_memory_node_id(1)
    score_row = system.connection.execute(
        "SELECT replay_priority, retention_status FROM memory_scores WHERE node_id = ?",
        (interaction_node,),
    ).fetchone()
    assert score_row is not None
    replay_edges = system.memory.edges_from(interaction_node, "selected_for_replay")
    assert replay_edges
    interaction_row = system.connection.execute(
        "SELECT id FROM interactions ORDER BY id LIMIT 1"
    ).fetchone()
    assert interaction_row is not None
    prediction_row = system.connection.execute(
        "SELECT interaction_id FROM prediction_results ORDER BY id LIMIT 1"
    ).fetchone()
    assert prediction_row is not None
    system.close()


def test_phase3_repeated_context_action_family_promotes_m1_node() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(substrate, MemoryPromotionConfig(min_contingency_support=3, min_contingency_confidence=0.6))
        substrate.upsert_node(
            MemoryNode(
                node_id="m1c",
                memory_level="M1",
                node_type="ContingencyMemory",
                attrs={"support_count": 4, "confidence": 0.8, "transformation_family": 9},
            ),
            step=1,
            support_increment=4,
        )
        summary = engine.promote_m0_to_m1(step=1)
        node = substrate.get_node("m1c")
        assert summary["count"] == 1
        assert node is not None
        assert node["attrs"]["promotion_status"] == "promoted"
    finally:
        connection.close()


def test_phase3_family_support_promotes_m2_node() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(substrate, MemoryPromotionConfig(min_family_support=3))
        for index in range(3):
            substrate.upsert_node(
                MemoryNode(
                    node_id=f"c{index}",
                    memory_level="M1",
                    node_type="ContingencyMemory",
                    attrs={"support_count": 1, "confidence": 0.9, "transformation_family": 5},
                ),
                step=index + 1,
            )
        summary = engine.promote_m1_to_m2(step=10)
        assert summary["count"] >= 1
        assert substrate.get_node(family_node_id(5)) is not None
    finally:
        connection.close()


def test_phase3_carrier_with_support_lift_and_compression_promotes_m3_carrier() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(substrate, MemoryPromotionConfig())
        substrate.upsert_node(
            MemoryNode(
                node_id="carrierA",
                memory_level="M3",
                node_type="CarrierMemory",
                attrs={
                    "carrier_source": "object",
                    "support_count": 4,
                    "prediction_lift": 0.2,
                    "compression_gain": 0.2,
                },
            ),
            step=1,
        )
        summary = engine.promote_m2_to_m3_carrier(step=1)
        node = substrate.get_node("carrierA")
        assert summary["count"] == 1
        assert node is not None
        assert node["attrs"]["promotion_status"] == "promoted"
    finally:
        connection.close()


def test_phase3_two_carriers_with_similar_graph_position_promote_same_role() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(substrate, MemoryPromotionConfig(min_role_support=2))
        for carrier in ("carrier1", "carrier2"):
            substrate.upsert_node(
                MemoryNode(
                    node_id=carrier,
                    memory_level="M3",
                    node_type="CarrierMemory",
                    attrs={"promotion_status": "promoted"},
                ),
                step=1,
            )
            substrate.upsert_edge(MemoryEdge(carrier, family_node_id(1), "associated_with_family"))
            substrate.upsert_edge(MemoryEdge(carrier, "ctx", "appears_in_context"))
        summary = engine.promote_m3_carrier_to_role(step=5)
        roles = substrate.query_nodes(memory_level="M3", node_type="FunctionalRoleMemory")
        assert summary["count"] >= 2
        assert len(roles) == 1
    finally:
        connection.close()


def test_phase3_role_transfer_evidence_can_promote_concept() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(
            substrate,
            MemoryPromotionConfig(min_concept_transfer_tests=2, min_concept_transfer_success_rate=0.5),
        )
        substrate.upsert_node(
            MemoryNode(
                node_id="roleA",
                memory_level="M3",
                node_type="FunctionalRoleMemory",
                attrs={"role_signature": "r1", "carrier_count": 3},
            ),
            step=1,
        )
        summary = engine.promote_role_to_concept(step=10)
        concepts = substrate.query_nodes(memory_level="M4", node_type="ConceptMemory")
        assert summary["count"] == 1
        assert concepts
    finally:
        connection.close()


def test_phase3_equivalent_outcome_lower_cost_promotes_strategy() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(substrate)
        node_id = strategy_node_id("same_outcome")
        substrate.upsert_node(
            MemoryNode(
                node_id=node_id,
                memory_level="M6",
                node_type="EfficientStrategyMemory",
                attrs={
                    "outcome_signature": "o1",
                    "best_known_cost": 2.0,
                    "current_cost": 5.0,
                    "normalized_solve_efficiency": 0.4,
                    "equivalent_outcome_cost_gap": 3.0,
                },
            ),
            step=1,
        )
        summary = engine.promote_strategy_memories(step=1)
        promotion_count = connection.execute("SELECT COUNT(*) FROM memory_promotions").fetchone()[0]
        assert summary["count"] == 1
        assert promotion_count >= 1
    finally:
        connection.close()


def test_phase4_future_option_estimator_creates_stable_option_sets() -> None:
    estimator = FutureOptionEstimator()
    observation = np.zeros((2, 2), dtype=int)
    left = estimator.estimate_option_set(observation, depth=1, available_actions=[2, 1])
    right = estimator.estimate_option_set(observation, depth=1, available_actions=[1, 2])
    assert left.option_set_id == right.option_set_id
    assert left.reachable_signatures == right.reachable_signatures


def test_phase4_positive_future_option_delta_creates_expands_edge(tmp_path) -> None:
    class ExpandingEnv(ToggleEnv):
        def available_actions(self) -> list[int]:
            return [1] if self.state == 0 else [1, 2]

    db_path = tmp_path / "future_expand.sqlite"
    system = V6System(env=ExpandingEnv(), config=V6Config(database_path=str(db_path), random_seed=0))
    result = system.run_step()
    edges = system.memory.edges_from(system._interaction_memory_node_id(result.interaction_id), "expands_future_options")
    assert edges
    score = system.connection.execute(
        "SELECT future_option_delta FROM memory_scores WHERE node_id = ?",
        (system._interaction_memory_node_id(result.interaction_id),),
    ).fetchone()
    assert score is not None
    assert float(score[0]) > 0.0
    system.close()


def test_phase4_negative_future_option_delta_creates_restricts_edge(tmp_path) -> None:
    class RestrictingEnv(ToggleEnv):
        def __init__(self) -> None:
            super().__init__()
            self.state = 1

        def available_actions(self) -> list[int]:
            return [1, 2] if self.state == 1 else [1]

    db_path = tmp_path / "future_restrict.sqlite"
    system = V6System(env=RestrictingEnv(), config=V6Config(database_path=str(db_path), random_seed=0))
    result = system.run_step()
    edges = system.memory.edges_from(system._interaction_memory_node_id(result.interaction_id), "restricts_future_options")
    assert edges
    system.close()


def test_phase4_isf_changes_when_future_option_delta_is_passed() -> None:
    without_delta = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id="d",
        context_signature="ctx",
        memory_counts={},
        graph_counts={},
        future_option_delta=None,
        outcome_state="NOT_FINISHED",
        outcome_polarity="neutral",
        level_completed_event=False,
    )
    with_negative_delta = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id="d",
        context_signature="ctx",
        memory_counts={},
        graph_counts={},
        future_option_delta=-0.8,
        outcome_state="NOT_FINISHED",
        outcome_polarity="neutral",
        level_completed_event=False,
    )
    with_delta = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id=None,
        delta_id="d",
        context_signature="ctx",
        memory_counts={},
        graph_counts={},
        future_option_delta=2.0,
        outcome_state="NOT_FINISHED",
        outcome_polarity="neutral",
        level_completed_event=False,
    )
    assert with_negative_delta.survival_impact >= 0.8
    assert with_delta.transfer_potential > without_delta.transfer_potential
    assert with_delta.survival_impact == without_delta.survival_impact


def test_phase4_compact_memory_preserves_future_option_nodes_and_edges(tmp_path) -> None:
    class ExpandingEnv(ToggleEnv):
        def available_actions(self) -> list[int]:
            return [1] if self.state == 0 else [1, 2]

    db_path = tmp_path / "future_compact.sqlite"
    memory_dir = tmp_path / "future_compact_memory"
    system = V6System(env=ExpandingEnv(), config=V6Config(database_path=str(db_path), random_seed=0))
    result = system.run_step()
    fold_live_system_into_compact_memory(system, memory_dir)
    system.close()
    restored = V6System(
        env=ToggleEnv(),
        config=V6Config(database_path=":memory:", memory_input_dir=str(memory_dir), restore_compact_memory=True, random_seed=0),
    )
    try:
        future_delta_edges = restored.memory.edges_from(restored._interaction_memory_node_id(result.interaction_id), "changes_future_options")
        assert future_delta_edges
    finally:
        restored.close()


def test_phase5_memory_query_disabled_keeps_old_predictor_path(tmp_path) -> None:
    db_path = tmp_path / "memory_query_disabled.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(database_path=str(db_path), random_seed=0, memory_query_enabled=False),
    )
    system.predictor.predict_multi_scale = lambda *_args, **_kwargs: 9
    result = system.run_step()
    assert result.predicted_family == 9
    system.close()


def test_phase5_exact_m1_contingency_predicts_family() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        context_signature = json.dumps(["ctx"])
        substrate.upsert_node(
            MemoryNode(
                node_id="m1exact",
                memory_level="M1",
                node_type="ContingencyMemory",
                attrs={"context_signature": context_signature, "action": 1, "transformation_family": 4, "confidence": 0.9},
            )
        )
        prediction = engine.predict_family({1: ("ctx",)}, 1, record_query=False)
        assert prediction.predicted_family == 4
        assert prediction.source == "memory_contingency"
    finally:
        connection.close()


def test_phase5_contingency_learner_fallback_still_works() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        learner = type("Learner", (), {"best_stable_for_action": lambda self, *_args, **_kwargs: type("Stable", (), {"transformation_family": 6, "confidence": 0.7})()})()
        engine = MemoryQueryEngine(substrate, contingency_learner=learner)
        prediction = engine.predict_family({1: ("ctx",)}, 1, record_query=False)
        assert prediction.predicted_family == 6
        assert prediction.source == "contingency_learner"
    finally:
        connection.close()


def test_phase5_similar_role_can_predict_when_exact_contingency_missing() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        context_signature = json.dumps(["ctx"])
        context_node = "M0:context:" + __import__("hashlib").sha1(str(context_signature).encode("utf-8")).hexdigest()[:20]
        substrate.upsert_node(MemoryNode(node_id="carrierX", memory_level="M3", node_type="CarrierMemory", attrs={}))
        substrate.upsert_node(MemoryNode(node_id="roleX", memory_level="M3", node_type="FunctionalRoleMemory", attrs={"transfer_score": 0.6}))
        substrate.upsert_node(MemoryNode(node_id=interaction_node_id(1), memory_level="M0", node_type="InteractionMemory"))
        substrate.upsert_edge(MemoryEdge("carrierX", "roleX", "plays_role"))
        substrate.upsert_edge(MemoryEdge("carrierX", family_node_id(7), "associated_with_family"))
        substrate.upsert_edge(MemoryEdge("carrierX", context_node, "appears_in_context"))
        substrate.upsert_edge(MemoryEdge("carrierX", interaction_node_id(1), "carried_by"))
        substrate.upsert_edge(MemoryEdge(interaction_node_id(1), action_node_id(1), "takes_action"))
        prediction = engine.predict_family({1: ("ctx",)}, 1, record_query=False)
        assert prediction.predicted_family == 7
        assert prediction.source == "role_match"
    finally:
        connection.close()


def test_phase5_concept_match_can_predict_when_role_exists() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        context_signature = json.dumps(["ctx"])
        context_node = "M0:context:" + __import__("hashlib").sha1(str(context_signature).encode("utf-8")).hexdigest()[:20]
        substrate.upsert_node(MemoryNode(node_id="carrierY", memory_level="M3", node_type="CarrierMemory", attrs={}))
        substrate.upsert_node(MemoryNode(node_id="roleY", memory_level="M3", node_type="FunctionalRoleMemory", attrs={}))
        substrate.upsert_node(MemoryNode(node_id="conceptY", memory_level="M4", node_type="ConceptMemory", attrs={"transfer_success_count": 3}))
        substrate.upsert_node(MemoryNode(node_id=interaction_node_id(1), memory_level="M0", node_type="InteractionMemory"))
        substrate.upsert_edge(MemoryEdge("roleY", "conceptY", "transfers_to"))
        substrate.upsert_edge(MemoryEdge("roleY", "carrierY", "abstracts_from"))
        substrate.upsert_edge(MemoryEdge("carrierY", "roleY", "plays_role"))
        substrate.upsert_edge(MemoryEdge("carrierY", family_node_id(8), "associated_with_family"))
        substrate.upsert_edge(MemoryEdge("carrierY", context_node, "appears_in_context"))
        substrate.upsert_edge(MemoryEdge("carrierY", interaction_node_id(1), "carried_by"))
        substrate.upsert_edge(MemoryEdge(interaction_node_id(1), action_node_id(1), "takes_action"))
        prediction = engine.predict_family({1: ("ctx",)}, 1, record_query=False)
        assert prediction.predicted_family == 8
        assert prediction.source in {"role_match", "concept_match"}
    finally:
        connection.close()


def test_phase5_action_ranking_prefers_positive_future_option_evidence() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        for action, delta in ((1, 1.0), (2, -1.0)):
            interaction_id = interaction_node_id(action)
            substrate.upsert_node(MemoryNode(node_id=interaction_id, memory_level="M0", node_type="InteractionMemory"))
            substrate.upsert_node(MemoryNode(node_id=action_node_id(action), memory_level="M0", node_type="ActionMemory"))
            substrate.upsert_edge(MemoryEdge(interaction_id, action_node_id(action), "takes_action"))
            substrate.upsert_score(MemoryScore(node_id=interaction_id, future_option_delta=delta))
        ranked = engine.rank_actions({1: {1: ("ctx",)}, 2: {1: ("ctx",)}}, [1, 2])
        assert ranked[0].action == 1
    finally:
        connection.close()


def test_memory_action_selection_uses_candidate_specific_context() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        substrate.upsert_node(
            MemoryNode(
                node_id="m1a1",
                memory_level="M1",
                node_type="ContingencyMemory",
                attrs={"context_signature": json.dumps([1]), "action": 1, "transformation_family": 10, "confidence": 0.9},
            )
        )
        substrate.upsert_node(
            MemoryNode(
                node_id="m1a2",
                memory_level="M1",
                node_type="ContingencyMemory",
                attrs={"context_signature": json.dumps([2]), "action": 2, "transformation_family": 20, "confidence": 0.9},
            )
        )
        score1 = engine.score_action({1: (1,)}, 1, [1, 2], record_query=False)
        score2 = engine.score_action({1: (2,)}, 2, [1, 2], record_query=False)
        assert score1.predicted_family == 10
        assert score2.predicted_family == 20
    finally:
        connection.close()


def test_phase5_query_scoring_has_no_selected_action_side_effects() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        engine.rank_actions({1: {1: ("ctx",)}, 2: {1: ("ctx",)}, 3: {1: ("ctx",)}}, [1, 2, 3])
        assert not substrate.query_nodes(memory_level="M5", node_type="MemoryQueryEvent")
        engine.record_selected_action_query(context_signatures={1: ("ctx",)}, action=2)
        query_nodes = substrate.query_nodes(memory_level="M5", node_type="MemoryQueryEvent")
        assert len(query_nodes) == 1
        selected_edges = substrate.edges_from(query_nodes[0]["node_id"], "selected_action")
        assert len(selected_edges) == 1
        assert selected_edges[0]["target_node_id"] == action_node_id(2)
    finally:
        connection.close()


def test_phase5_role_matching_is_action_sensitive() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        context_node = "M0:context:" + __import__("hashlib").sha1(str(json.dumps(["ctx"])).encode("utf-8")).hexdigest()[:20]
        substrate.upsert_node(MemoryNode(node_id="carrierR", memory_level="M3", node_type="CarrierMemory"))
        substrate.upsert_node(MemoryNode(node_id="roleR", memory_level="M3", node_type="FunctionalRoleMemory", attrs={"transfer_score": 0.5}))
        substrate.upsert_node(MemoryNode(node_id=interaction_node_id(1), memory_level="M0", node_type="InteractionMemory"))
        substrate.upsert_edge(MemoryEdge("carrierR", "roleR", "plays_role"))
        substrate.upsert_edge(MemoryEdge("carrierR", context_node, "appears_in_context"))
        substrate.upsert_edge(MemoryEdge("carrierR", interaction_node_id(1), "carried_by"))
        substrate.upsert_edge(MemoryEdge(interaction_node_id(1), action_node_id(1), "takes_action"))
        good = engine.find_similar_roles(json.dumps(["ctx"]), 1)
        bad = engine.find_similar_roles(json.dumps(["ctx"]), 2)
        assert good
        assert good[0]["score"] > 0.5
        assert not bad or good[0]["score"] > bad[0]["score"]
    finally:
        connection.close()


def test_phase5_concept_matching_is_action_sensitive() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        context_node = "M0:context:" + __import__("hashlib").sha1(str(json.dumps(["ctx"])).encode("utf-8")).hexdigest()[:20]
        substrate.upsert_node(MemoryNode(node_id="carrierC", memory_level="M3", node_type="CarrierMemory"))
        substrate.upsert_node(MemoryNode(node_id="roleC", memory_level="M3", node_type="FunctionalRoleMemory", attrs={"transfer_score": 0.8}))
        substrate.upsert_node(MemoryNode(node_id="conceptC", memory_level="M4", node_type="ConceptMemory", attrs={"transfer_success_count": 3}))
        substrate.upsert_node(MemoryNode(node_id=interaction_node_id(1), memory_level="M0", node_type="InteractionMemory"))
        substrate.upsert_edge(MemoryEdge("carrierC", "roleC", "plays_role"))
        substrate.upsert_edge(MemoryEdge("roleC", "conceptC", "transfers_to"))
        substrate.upsert_edge(MemoryEdge("roleC", "carrierC", "abstracts_from"))
        substrate.upsert_edge(MemoryEdge("carrierC", family_node_id(8), "associated_with_family"))
        substrate.upsert_edge(MemoryEdge("carrierC", context_node, "appears_in_context"))
        substrate.upsert_edge(MemoryEdge("carrierC", interaction_node_id(1), "carried_by"))
        substrate.upsert_edge(MemoryEdge(interaction_node_id(1), action_node_id(1), "takes_action"))
        assert engine.find_concept_matches(json.dumps(["ctx"]), 1)
        assert not engine.find_concept_matches(json.dumps(["ctx"]), 2)
    finally:
        connection.close()


def test_phase5_action_ranking_penalizes_failure_path_evidence() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        for action, delta in ((1, -1.0), (2, 0.5)):
            interaction_id = interaction_node_id(100 + action)
            substrate.upsert_node(MemoryNode(node_id=interaction_id, memory_level="M0", node_type="InteractionMemory"))
            substrate.upsert_node(MemoryNode(node_id=action_node_id(action), memory_level="M0", node_type="ActionMemory"))
            substrate.upsert_edge(MemoryEdge(interaction_id, action_node_id(action), "takes_action"))
            substrate.upsert_score(MemoryScore(node_id=interaction_id, future_option_delta=delta))
        ranked = engine.rank_actions({1: {1: ("ctx",)}, 2: {1: ("ctx",)}}, [1, 2])
        assert ranked[0].action == 2
    finally:
        connection.close()


def test_selected_action_query_recorded_once_per_executed_step(tmp_path) -> None:
    class MultiActionEnv(ToggleEnv):
        def available_actions(self) -> list[int]:
            return [1, 2, 3]

    db_path = tmp_path / "selected_action_once.sqlite"
    system = V6System(
        env=MultiActionEnv(),
        config=V6Config(
            database_path=str(db_path),
            random_seed=0,
            memory_query_enabled=True,
        ),
    )
    result = system.run_step()
    try:
        rows = system.connection.execute(
            "SELECT source_node_id, target_node_id FROM memory_edges WHERE edge_type = 'selected_action' ORDER BY source_node_id ASC"
        ).fetchall()
        assert len(rows) == 1
        assert str(rows[0][1]) == action_node_id(result.action)
    finally:
        system.close()


def test_m0_interaction_memory_stores_context_attrs(tmp_path) -> None:
    db_path = tmp_path / "interaction_context.sqlite"
    system = V6System(
        env=ToggleEnv(),
        config=V6Config(database_path=str(db_path), context_length=2, random_seed=0),
    )
    system.run_step()
    try:
        row = system.connection.execute(
            "SELECT attrs_json FROM memory_nodes WHERE node_type = 'InteractionMemory' ORDER BY node_id ASC LIMIT 1"
        ).fetchone()
        assert row is not None
        attrs = json.loads(str(row[0]))
        assert "context_signature" in attrs
        assert "context_level" in attrs
        assert attrs["context_signature"] not in {None, "**no_context**"}
    finally:
        system.close()


def test_phase5_memory_action_selection_chooses_best_ranked_action(tmp_path) -> None:
    class MultiActionEnv(ToggleEnv):
        def available_actions(self) -> list[int]:
            return [1, 2]

    db_path = tmp_path / "memory_action_selection.sqlite"
    system = V6System(
        env=MultiActionEnv(),
        config=V6Config(database_path=str(db_path), random_seed=0, memory_action_selection_enabled=True),
    )
    system.memory_query.score_action = lambda _ctx, action, _available, record_query=False: type(
        "Rank", (), {"action": action, "score": 0.9 if int(action) == 2 else 0.1}
    )()
    assert system.choose_action() == 2
    system.close()


def test_phase5_query_event_writes_evidence_edges() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        substrate.upsert_node(
            MemoryNode(
                node_id="m1exact",
                memory_level="M1",
                node_type="ContingencyMemory",
                attrs={"context_signature": json.dumps(["ctx"]), "action": 1, "transformation_family": 4, "confidence": 0.9},
            )
        )
        _prediction = engine.predict_family({1: ("ctx",)}, 1, record_query=True)
        query_nodes = substrate.query_nodes(memory_level="M5", node_type="MemoryQueryEvent")
        assert query_nodes
        used_edges = substrate.edges_from(query_nodes[0]["node_id"], "used_evidence")
        assert used_edges
    finally:
        connection.close()


def test_phase5_real_m0_to_m1_promotion_uses_m0_interactions() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(substrate, MemoryPromotionConfig(min_contingency_support=3, min_contingency_confidence=0.6))
        for idx in range(1, 4):
            interaction_id = interaction_node_id(idx)
            substrate.upsert_node(
                MemoryNode(
                    node_id=interaction_id,
                    memory_level="M0",
                    node_type="InteractionMemory",
                    attrs={"context_signature": "ctxA"},
                )
            )
            substrate.upsert_edge(MemoryEdge(interaction_id, action_node_id(1), "takes_action"))
            substrate.upsert_edge(MemoryEdge(interaction_id, family_node_id(9), "supports"))
        summary = engine.promote_m0_to_m1(step=1)
        contingencies = substrate.query_nodes(memory_level="M1", node_type="ContingencyMemory")
        promotions = connection.execute(
            "SELECT source_node_id, target_node_id FROM memory_promotions WHERE promotion_type = 'M0_M1'"
        ).fetchall()
        assert summary["count"] >= 1
        assert contingencies
        assert promotions
        assert str(promotions[0][0]).startswith("M0:interaction:")
        assert str(promotions[0][1]).startswith("M1:contingency:")
        assert promotions[0][0] != promotions[0][1]
    finally:
        connection.close()


def test_m1_validation_does_not_record_fake_m0_m1_self_promotion() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(substrate, MemoryPromotionConfig(min_contingency_support=3, min_contingency_confidence=0.6))
        substrate.upsert_node(
            MemoryNode(
                node_id="M1:contingency:test",
                memory_level="M1",
                node_type="ContingencyMemory",
                attrs={"support_count": 4, "confidence": 0.9},
            )
        )
        summary = engine.validate_existing_m1_contingencies(step=1)
        rows = connection.execute(
            "SELECT promotion_type, source_node_id, target_node_id FROM memory_promotions ORDER BY promotion_id ASC"
        ).fetchall()
        assert summary["count"] == 1
        assert rows
        assert all(str(row[0]) == "M1_VALIDATION" for row in rows)
        assert not connection.execute(
            "SELECT COUNT(*) FROM memory_promotions WHERE promotion_type = 'M0_M1' AND source_node_id = target_node_id"
        ).fetchone()[0]
    finally:
        connection.close()


def test_rank_actions_uses_candidate_specific_contexts() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryQueryEngine(substrate)
        substrate.upsert_node(
            MemoryNode(
                node_id="m1ctx1",
                memory_level="M1",
                node_type="ContingencyMemory",
                attrs={"context_signature": json.dumps(["ctx1"]), "action": 1, "transformation_family": 10, "confidence": 0.9},
            )
        )
        substrate.upsert_node(
            MemoryNode(
                node_id="m1ctx2",
                memory_level="M1",
                node_type="ContingencyMemory",
                attrs={"context_signature": json.dumps(["ctx2"]), "action": 2, "transformation_family": 20, "confidence": 0.9},
            )
        )
        ranked = engine.rank_actions({1: {1: ("ctx1",)}, 2: {1: ("ctx2",)}}, [1, 2])
        by_action = {int(item.action): item for item in ranked}
        assert by_action[1].predicted_family == 10
        assert by_action[2].predicted_family == 20
    finally:
        connection.close()


def test_choose_action_ranking_does_not_pollute_selected_action_edges_before_step(tmp_path) -> None:
    class MultiActionEnv(ToggleEnv):
        def available_actions(self) -> list[int]:
            return [1, 2]

    db_path = tmp_path / "no_query_pollution.sqlite"
    system = V6System(
        env=MultiActionEnv(),
        config=V6Config(
            database_path=str(db_path),
            random_seed=0,
            memory_action_selection_enabled=True,
        ),
    )
    try:
        _ = system.choose_action()
        assert not system.connection.execute(
            "SELECT COUNT(*) FROM memory_edges WHERE edge_type = 'selected_action'"
        ).fetchone()[0]
        result = system.run_step()
        rows = system.connection.execute(
            "SELECT target_node_id FROM memory_edges WHERE edge_type = 'selected_action'"
        ).fetchall()
        assert len(rows) == 1
        assert str(rows[0][0]) == action_node_id(result.action)
    finally:
        system.close()


def test_phase5_future_option_role_promotion_uses_carried_interactions() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        substrate = MemorySubstrate(connection)
        engine = MemoryPromotionEngine(substrate, MemoryPromotionConfig(min_role_support=1))
        substrate.upsert_node(
            MemoryNode(
                node_id="carrierZ",
                memory_level="M3",
                node_type="CarrierMemory",
                attrs={"promotion_status": "promoted"},
            )
        )
        substrate.upsert_node(MemoryNode(node_id=interaction_node_id(1), memory_level="M0", node_type="InteractionMemory"))
        substrate.upsert_edge(MemoryEdge("carrierZ", interaction_node_id(1), "carried_by"))
        substrate.upsert_edge(MemoryEdge("carrierZ", family_node_id(1), "associated_with_family"))
        substrate.upsert_edge(MemoryEdge(interaction_node_id(1), "outcome:1", "expands_future_options"))
        summary = engine.promote_m3_carrier_to_role(step=5)
        roles = substrate.query_nodes(memory_level="M3", node_type="FunctionalRoleMemory")
        assert summary["count"] >= 1
        assert roles
        assert roles[0]["attrs"]["future_option_effect"] == "positive"
    finally:
        connection.close()


def test_phase5_restore_complete_substrate(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    state_path = memory_dir / "current_state.sqlite"
    graph_path = memory_dir / "graph.sqlite"
    replay_path = memory_dir / "replay_queue.sqlite"
    summary_path = memory_dir / "memory_summary.json"
    connection = sqlite3.connect(state_path)
    try:
        substrate = MemorySubstrate(connection)
        connection.execute("CREATE TABLE IF NOT EXISTS family_identity_map (canonical_signature TEXT PRIMARY KEY, stable_family_id INTEGER)")
        connection.execute("CREATE TABLE IF NOT EXISTS stable_contingencies (canonical_key TEXT, context_level INTEGER, action INTEGER, effect_signature TEXT, support_count INTEGER, stability_score REAL)")
        connection.execute("CREATE TABLE IF NOT EXISTS family_members (family_signature TEXT, contingency_key TEXT, support_count INTEGER)")
        connection.execute("CREATE TABLE IF NOT EXISTS transformation_families (family_id INTEGER, canonical_signature TEXT, support_count INTEGER)")
        connection.execute("CREATE TABLE IF NOT EXISTS carrier_candidates (carrier_signature TEXT, carrier_source TEXT, support_count INTEGER, linked_family_count INTEGER, first_seen_global_step INTEGER, last_seen_global_step INTEGER, stability_score REAL, is_emergent INTEGER)")
        connection.execute("CREATE TABLE IF NOT EXISTS memory_summary (key TEXT, value_json TEXT)")
        substrate.upsert_node(MemoryNode(node_id="node1", memory_level="M0", node_type="InteractionMemory"))
        substrate.upsert_edge(MemoryEdge("node1", "node2", "follows"))
        substrate.add_evidence(__import__("v6.memory.substrate", fromlist=["MemoryEvidence"]).MemoryEvidence("ev1", "node1", 1, "test"))
        substrate.upsert_score(MemoryScore(node_id="node1", isf_total=0.5), step=1)
        substrate.record_promotion(__import__("v6.memory.substrate", fromlist=["MemoryPromotion"]).MemoryPromotion("pr1", "node1", "node2", "M0_M1", 1, 0.5, "promoted"))
        connection.commit()
    finally:
        connection.close()
    graph_conn = sqlite3.connect(graph_path)
    try:
        graph_conn.execute("CREATE TABLE IF NOT EXISTS graph_nodes (node_id TEXT, node_type TEXT, canonical_key TEXT, support_count INTEGER)")
        graph_conn.execute("CREATE TABLE IF NOT EXISTS graph_edges (source_node_id TEXT, target_node_id TEXT, edge_type TEXT, support_count INTEGER, weight REAL)")
        graph_conn.commit()
    finally:
        graph_conn.close()
    replay_conn = sqlite3.connect(replay_path)
    try:
        replay_conn.execute("CREATE TABLE IF NOT EXISTS replay_queue (replay_id TEXT, owner_type TEXT, owner_id TEXT, priority_score REAL, reason TEXT, first_seen_global_step INTEGER, last_seen_global_step INTEGER, compact_payload_json TEXT)")
        replay_conn.commit()
    finally:
        replay_conn.close()
    summary_path.write_text("{}", encoding="utf-8")
    system = V6System(env=ToggleEnv(), config=V6Config(database_path=":memory:", memory_input_dir=str(memory_dir), restore_compact_memory=True, random_seed=0))
    try:
        summary = system.compact_memory_restore_summary
        assert summary["memory_nodes_restored"] >= 1
        assert summary["memory_edges_restored"] >= 1
        assert summary["memory_evidence_restored"] >= 1
        assert summary["memory_scores_restored"] >= 1
        assert summary["memory_promotions_restored"] >= 1
        assert system.connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] >= 1
        assert system.connection.execute("SELECT COUNT(*) FROM memory_promotions").fetchone()[0] >= 1
    finally:
        system.close()


def test_phase5_observation_canonical_key_is_hashed_and_future_option_links_match(tmp_path) -> None:
    class ExpandingEnv(ToggleEnv):
        def available_actions(self) -> list[int]:
            return [1] if self.state == 0 else [1, 2]

    db_path = tmp_path / "obs_hash.sqlite"
    system = V6System(env=ExpandingEnv(), config=V6Config(database_path=str(db_path), future_options_enabled=True, random_seed=0))
    result = system.run_step()
    observation_nodes = system.memory.query_nodes(memory_level="M0", node_type="ObservationMemory")
    assert observation_nodes
    for node in observation_nodes:
        assert node["canonical_key"] is not None
        assert len(str(node["canonical_key"])) <= 40
        assert "observation_hash" in node["attrs"]
        assert "signature_len" in node["attrs"]
    has_future = system.connection.execute(
        "SELECT source_node_id FROM memory_edges WHERE edge_type = 'has_future_options' ORDER BY source_node_id ASC LIMIT 1"
    ).fetchone()
    assert has_future is not None
    source_node = system.memory.get_node(str(has_future[0]))
    assert source_node is not None
    assert source_node["node_type"] == "ObservationMemory"
    interaction_edges = system.memory.edges_from(system._interaction_memory_node_id(result.interaction_id), "changes_future_options")
    assert interaction_edges
    system.close()


def test_h07_requires_at_least_five_promoted_concepts(tmp_path) -> None:
    memory_dir = tmp_path / "memory_h07_small"
    ensure_memory_layout(memory_dir)
    conn = sqlite3.connect(memory_dir / "current_state.sqlite")
    try:
        for idx in range(2):
            conn.execute(
                """
                INSERT INTO concept_candidates (
                    concept_signature, concept_type, support_count, linked_role_count, linked_carrier_count,
                    linked_family_count, transfer_success_count, strong_transfer_success_count, cross_game_count,
                    cross_context_count, compression_gain, explanatory_reach, promotion_score,
                    first_seen_global_step, last_seen_global_step, is_promoted
                ) VALUES (?, 'x', 5, 2, 2, 2, 3, 3, 2, 3, 2.0, 4.0, 0.8, 1, 10, 1)
                """,
                (f"c{idx}",),
            )
        for idx in range(6):
            conn.execute(
                """
                INSERT INTO role_transfer_attempts (
                    attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key,
                    target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature,
                    observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason,
                    first_seen_global_step, last_seen_global_step, best_margin, source_carrier_count, candidate_role_count
                ) VALUES (?, ?, 'cross_game', 'not_game', 'g1', 'game', 'g2', ?, ?, ?, 0.9, 0.9, 1, 'success', 1, 10, 0.2, 2, 2)
                """,
                (f"a{idx}", f"r{idx%2}", f"car{idx}", f"r{idx%2}", f"r{idx%2}"),
            )
        conn.commit()
    finally:
        conn.close()
    result = evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07", already_derived=True)
    assert result["promoted_concept_count"] == 2
    assert result["decision"] == "PARTIALLY_VALID"


def test_h08_requires_at_least_five_coherent_components(tmp_path) -> None:
    memory_dir = tmp_path / "memory_h08_small"
    ensure_memory_layout(memory_dir)
    conn = sqlite3.connect(memory_dir / "current_state.sqlite")
    try:
        for idx in range(2):
            conn.execute(
                """
                INSERT INTO world_model_components (
                    component_signature, component_type, node_count, edge_count, linked_concept_count, linked_role_count,
                    linked_family_count, linked_carrier_count, cross_context_count, cross_game_count,
                    explanatory_coverage, prediction_support_count, contradiction_coverage_count, coherence_score,
                    candidate_only, predicted_outcome_count, predicted_outcome_count_is_proxy,
                    first_seen_global_step, last_seen_global_step, is_coherent
                ) VALUES (?, 'promoted', 8, 20, 2, 2, 2, 2, 3, 2, 1.0, 3, 1, 0.7, 0, 3, 1, 1, 10, 1)
                """,
                (f"wm{idx}",),
            )
            for linked_type, linked_key in (("concept", f"c{idx}"), ("role", f"r{idx}"), ("family", f"f{idx}"), ("family", f"f{idx+10}"), ("context", f"ctx{idx}"), ("context", f"ctx{idx+10}")):
                conn.execute(
                    "INSERT INTO world_model_links (component_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES (?, ?, ?, 1, 1, 10)",
                    (f"wm{idx}", linked_type, linked_key),
                )
        conn.execute("INSERT INTO concept_candidates (concept_signature, concept_type, support_count, linked_role_count, linked_carrier_count, linked_family_count, transfer_success_count, strong_transfer_success_count, cross_game_count, cross_context_count, compression_gain, explanatory_reach, promotion_score, first_seen_global_step, last_seen_global_step, is_promoted) VALUES ('c0','x',1,2,2,2,2,2,2,3,2.0,4.0,0.8,1,10,1)")
        conn.commit()
    finally:
        conn.close()
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08", already_derived=True)
    assert result["coherent_world_model_component_count"] == 2
    assert result["decision"] == "PARTIALLY_VALID"


def test_h09_and_h10_reports_include_live_future_option_diagnostics(tmp_path) -> None:
    memory_dir = tmp_path / "memory_h09_h10"
    ensure_memory_layout(memory_dir)
    conn = sqlite3.connect(memory_dir / "current_state.sqlite")
    try:
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES ('M0:interaction:1', 'M0', 'InteractionMemory', 'i1', 1, '{}')")
        conn.execute("INSERT INTO memory_scores (node_id, future_option_delta, replay_priority) VALUES ('M0:interaction:1', 2.0, 0.9)")
        conn.execute("INSERT INTO memory_edges (source_node_id, target_node_id, edge_type, weight, support_count, evidence_json) VALUES ('M0:interaction:1', 'delta:1', 'expands_future_options', 1.0, 1, '{}')")
        conn.execute("INSERT INTO memory_edges (source_node_id, target_node_id, edge_type, weight, support_count, evidence_json) VALUES ('M0:interaction:1', 'replay:1', 'selected_for_replay', 1.0, 1, '{}')")
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, effect_type, action_group, polarity, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1, 'fam1', 'positive_change', 'move', 'positive', 5, 2, 1, 10, 0.9)")
        conn.execute("INSERT INTO future_option_events (event_id, owner_type, owner_key, source_kind, motif_type, option_delta, option_delta_bucket, option_count_before, option_count_after, novelty_score, reversibility_score, branching_score, termination_score, contradiction_score, replay_priority_score, memory_priority_score, first_seen_global_step, last_seen_global_step, evidence_json) VALUES ('foe1', 'family', 'fam1', 'transformation_family', 'transform', 1.0, 'positive', 10.0, 11.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5, 1, 10, ?)", (json.dumps({"motif_type_source": "structured_effect"}),))
        conn.execute("INSERT INTO future_option_motifs (motif_signature, motif_type, support_count, linked_event_count, linked_family_count, linked_carrier_count, linked_role_count, linked_concept_count, cross_context_count, cross_game_count, mean_option_delta, mean_abs_option_delta, mean_novelty_score, mean_reversibility_score, mean_branching_score, mean_termination_score, mean_replay_priority_score, first_seen_global_step, last_seen_global_step, motif_stability_score, is_emergent) VALUES ('m1','transform',3,3,1,0,0,0,2,2,1.0,1.0,0.5,0.0,0.0,0.0,0.9,1,10,0.7,1)")
        conn.commit()
    finally:
        conn.close()
    h09 = evaluate_h09_future_option_motifs(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h09", already_derived=False)
    h10 = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10", already_derived=False)
    assert "motif_type_source_counts" in h09
    assert h10["high_option_change_source"] in {"live", "mixed"}

def test_validation_report_classifies_k0_sufficient(tmp_path) -> None:
    db_path = tmp_path / "k0.sqlite"
    connection = __import__("sqlite3").connect(db_path)
    _create_validation_fixture_schema(connection)
    _insert_fixture_contingency(connection, 1, 0, [1], 1, 10, 21, 1.0)
    for index in range(25):
        _insert_fixture_prediction(connection, index + 1, action=1, actual_family=10)
    connection.close()

    report = build_validation_report(str(db_path), game_id="toy", required_actions=(1,))

    assert report.classification == "A"
    assert report.minimum_context_level == 0
    assert not report.ambiguous_k0_actions


def test_validation_report_marks_k1_resolution_for_ambiguous_k0(tmp_path) -> None:
    db_path = tmp_path / "k1.sqlite"
    connection = __import__("sqlite3").connect(db_path)
    _create_validation_fixture_schema(connection)
    _insert_fixture_contingency(connection, 1, 1, [10, 1, 1], 1, 10, 21, 1.0)
    for index in range(25):
        _insert_fixture_prediction(connection, index + 1, action=1, actual_family=10)
    for index in range(25, 50):
        _insert_fixture_prediction(connection, index + 1, action=1, actual_family=11)
    connection.close()

    report = build_validation_report(str(db_path), game_id="toy", required_actions=(1,))

    assert report.classification == "B"
    assert report.minimum_context_level == 1
    assert len(report.ambiguous_k0_actions) == 1
    assert report.ambiguous_k0_actions[0].resolved_by_level == 1


def test_validation_report_supports_k5_classification(tmp_path) -> None:
    db_path = tmp_path / "k5.sqlite"
    connection = __import__("sqlite3").connect(db_path)
    _create_validation_fixture_schema(connection)
    _insert_fixture_contingency(connection, 1, 5, [10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 1], 1, 10, 21, 1.0)
    for index in range(25):
        _insert_fixture_prediction(connection, index + 1, action=1, actual_family=10)
    connection.close()

    report = build_validation_report(str(db_path), game_id="toy", required_actions=(1,), max_context_level=5)

    assert report.classification == "C"
    assert report.minimum_context_level == 5


def test_future_effect_calculates_fo_before_after_and_delta() -> None:
    events = [
        InteractionEvent(1, 0, 1, 1),
        InteractionEvent(2, 0, 1, 2),
        InteractionEvent(3, 0, 1, 2),
        InteractionEvent(4, 0, 1, 3),
        InteractionEvent(5, 0, 1, 4),
    ]

    assert future_effect_for_occurrence(events, index=2, horizon=2) == (2, 2, 0)


def test_future_effect_does_not_cross_reset_boundaries() -> None:
    events = [
        InteractionEvent(1, 0, 1, 1),
        InteractionEvent(2, 0, 1, 2),
        InteractionEvent(3, 1, 1, 3),
        InteractionEvent(4, 1, 1, 4),
        InteractionEvent(5, 1, 1, 5),
    ]

    assert future_effect_for_occurrence(events, index=2, horizon=10) == (0, 2, 2)


def test_future_effect_classification() -> None:
    assert classify_future_effect(mean_delta_fo=2.0, mean_fo_after=3.0, collapse_ratio=0.0) == "EXPAND"
    assert classify_future_effect(mean_delta_fo=-2.0, mean_fo_after=1.0, collapse_ratio=0.0) == "RESTRICT"
    assert classify_future_effect(mean_delta_fo=0.5, mean_fo_after=2.0, collapse_ratio=0.0) == "PRESERVE"
    assert classify_future_effect(mean_delta_fo=2.0, mean_fo_after=0.0, collapse_ratio=0.5) == "COLLAPSE"


def test_future_effect_sqlite_persistence(tmp_path) -> None:
    db_path = tmp_path / "future_effects.sqlite"
    connection = __import__("sqlite3").connect(db_path)
    ensure_future_effects_schema(connection)
    effect = FutureEffect(
        id=1,
        game="toy",
        seed=0,
        steps=10,
        horizon=2,
        contingency_id=3,
        context_level=1,
        action=4,
        transformation_family=7,
        occurrence_count=5,
        skipped_occurrence_count=1,
        mean_fo_before=1.0,
        mean_fo_after=3.0,
        mean_delta_fo=2.0,
        median_delta_fo=2.0,
        std_delta_fo=0.0,
        positive_delta_ratio=1.0,
        negative_delta_ratio=0.0,
        zero_delta_ratio=0.0,
        collapse_ratio=0.0,
        future_effect_class="EXPAND",
    )
    replace_future_effects(connection, [effect])
    loaded = load_future_effects(connection)
    connection.close()

    assert loaded == [effect]


def test_role_candidate_discovery_groups_stable_contingencies(tmp_path) -> None:
    db_path = tmp_path / "roles.sqlite"
    connection = __import__("sqlite3").connect(db_path)
    _create_validation_fixture_schema(connection)
    _insert_fixture_contingency(connection, 1, 0, [1], 1, 10, 25, 0.9)
    _insert_fixture_contingency(connection, 2, 0, [1], 2, 10, 24, 0.8)
    _insert_fixture_contingency(connection, 3, 1, [10, 1, 3], 3, 11, 23, 0.95)
    replace_future_effects(
        connection,
        [
            _future_effect(1, contingency_id=1, context_level=0, action=1, family=10, effect_class="EXPAND", mean_delta=2.0),
            _future_effect(2, contingency_id=2, context_level=0, action=2, family=10, effect_class="EXPAND", mean_delta=2.5),
            _future_effect(3, contingency_id=3, context_level=1, action=3, family=11, effect_class="PRESERVE", mean_delta=0.0),
        ],
    )
    connection.close()

    rows = analyze_role_candidates(db_path=str(db_path), game="toy", seed=0, steps=10, horizon=2)

    deterministic = next(row for row in rows if row["mode"] == "deterministic")
    assert deterministic["role_candidate_count"] == 2
    assert deterministic["expand_role_count"] == 1
    assert deterministic["preserve_role_count"] == 1
    assert deterministic["largest_role_size"] == 2


def test_role_candidate_sqlite_persistence(tmp_path) -> None:
    db_path = tmp_path / "role_candidates.sqlite"
    connection = __import__("sqlite3").connect(db_path)
    ensure_role_candidates_schema(connection)
    candidate = RoleCandidate(
        id=1,
        game="toy",
        seed=0,
        steps=10,
        horizon=2,
        mode="deterministic",
        role_id="R1",
        member_contingency_ids=(3, 4),
        support_count=2,
        dominant_future_effect_class="EXPAND",
        mean_delta_fo=2.0,
        mean_collapse_ratio=0.0,
        mean_confidence=0.9,
        mean_context_level=1.0,
        prototype_vector=(0.1, 0.2),
        stability_score=0.8,
    )
    replace_role_candidates(connection, [candidate])
    loaded = load_role_candidates(connection)
    connection.close()

    assert loaded == [candidate]


def test_role_candidate_seed_matching_uses_cosine_threshold() -> None:
    assert _matched_ratio([(1.0, 0.0), (0.0, 1.0)], [(0.9, 0.1), (1.0, 1.0)]) == 0.5


def test_role_validation_examples_keep_target_label_out_of_features(tmp_path) -> None:
    db_path = tmp_path / "role_validation_features.sqlite"
    connection = __import__("sqlite3").connect(db_path)
    _create_validation_fixture_schema(connection)
    _insert_fixture_contingency(connection, 1, 0, [1], 1, 10, 25, 0.9)
    replace_future_effects(
        connection,
        [_future_effect(1, contingency_id=1, context_level=0, action=1, family=10, effect_class="EXPAND", mean_delta=2.0)],
    )
    connection.close()

    examples = _load_examples(db_path)

    assert len(examples) == 1
    assert len(examples[0].raw_vector) == len(ALLOWED_ASSIGNMENT_FEATURE_NAMES) == 13
    assert examples[0].label == "EXPAND"


def test_role_validation_predictor_can_improve_without_label_leakage() -> None:
    preserve = _validation_example(1, context_level=0, family=10, label="PRESERVE", delta=0.0)
    expand = _validation_example(2, context_level=1, family=11, label="EXPAND", delta=2.0)
    train_examples = [preserve, preserve, expand, expand]
    test_examples = [
        _validation_example(3, context_level=0, family=10, label="PRESERVE", delta=0.0),
        _validation_example(4, context_level=1, family=11, label="EXPAND", delta=2.0),
    ]

    row = validate_role_predictor(
        game="toy",
        mode="deterministic",
        train_seeds=(0, 1),
        test_seed=2,
        steps=10,
        horizon=2,
        train_examples=train_examples,
        test_examples=test_examples,
    )

    assert row["label_leakage_prevented"] is True
    assert row["assignment_coverage"] == 1.0
    assert row["strict_role_accuracy"] > row["baseline_accuracy"]
    assert row["assigned_only_role_accuracy"] == row["strict_role_accuracy"]
    assert row["assigned_only_macro_f1"] == row["macro_f1"]
    assert row["per_class_recall"]["EXPAND"] == 1.0


def test_v04b_feature_sets_exclude_future_effect_class_id() -> None:
    all_feature_names = set(ALLOWED_ASSIGNMENT_FEATURE_NAMES)

    assert "future_effect_class_id" not in all_feature_names
    assert set(FEATURE_SETS["full_allowed"]) == all_feature_names
    assert "action_id" not in FEATURE_SETS["no_action"]
    assert "transformation_family_id" not in FEATURE_SETS["no_family"]
    assert set(FEATURE_SETS["future_effect_only_no_label"]) == {
        "mean_fo_before",
        "mean_fo_after",
        "mean_delta_fo",
        "std_delta_fo",
        "positive_delta_ratio",
        "negative_delta_ratio",
        "zero_delta_ratio",
        "collapse_ratio",
    }


def test_v04b_feature_projection_uses_selected_columns_only() -> None:
    example = _validation_example(1, context_level=2, family=7, label="EXPAND", delta=2.0)

    projected = project_feature_vectors([example], "future_effect_only_no_label")

    assert len(projected[0]) == len(FEATURE_SETS["future_effect_only_no_label"])
    assert projected[0] == tuple(example.raw_vector[index] for index in feature_indices("future_effect_only_no_label"))


def test_v04b_normalization_uses_train_statistics_only() -> None:
    train_vectors = [(0.0, 10.0), (2.0, 14.0)]
    test_vector = (100.0, 100.0)

    means, stds = _normalization_stats(train_vectors)
    normalized_test = _apply_normalization(test_vector, means, stds)

    assert means == (1.0, 12.0)
    assert stds == (1.0, 2.0)
    assert normalized_test == (99.0, 44.0)


def test_v04b_threshold_assignment_marks_low_similarity_unassigned() -> None:
    roles = [
        TrainRole(
            role_id="R1",
            mode="vector",
            member_count=3,
            prototype_vector=(1.0, 0.0),
            dominant_future_effect_class="EXPAND",
        )
    ]

    assigned_predictions, assigned_count = role_predictions_at_threshold(roles, [(0.9, 0.1)], similarity_threshold=0.65)
    unassigned_predictions, unassigned_count = role_predictions_at_threshold(roles, [(0.0, 1.0)], similarity_threshold=0.65)

    assert assigned_predictions == ["EXPAND"]
    assert assigned_count == 1
    assert unassigned_predictions == [None]
    assert unassigned_count == 0


def test_v04b_strict_and_assigned_only_metrics_treat_unassigned_differently() -> None:
    metrics = _classification_metrics(["EXPAND", "PRESERVE"], ["EXPAND", None])

    assert metrics["strict_accuracy"] == 0.5
    assert metrics["assigned_only_accuracy"] == 1.0
    assert metrics["per_class_recall"]["PRESERVE"] == 0.0
    assert metrics["assigned_only_per_class_recall"]["PRESERVE"] == 0.0


def test_v04c_prefuture_feature_sets_exclude_forbidden_features() -> None:
    assert all(forbidden_feature_check(name) for name in PREFUTURE_FEATURE_SETS)
    assert "mean_delta_fo" not in PREFUTURE_FEATURE_SETS["all_prefuture_no_ids"]
    assert "future_effect_class_id" not in PREFUTURE_FEATURE_SETS["all_prefuture_with_ids"]


def test_v04c_feature_set_construction() -> None:
    example = _prefuture_example("EXPAND")

    matrix = feature_matrix([example], "contingency_only")

    assert len(matrix[0]) == len(PREFUTURE_FEATURE_SETS["contingency_only"])
    assert matrix[0][0] == example.features["context_level"]


def test_v04c_train_only_normalization() -> None:
    means, stds = normalization_stats([(0.0, 10.0), (2.0, 14.0)])
    normalized = apply_normalization([(100.0, 100.0)], means, stds)

    assert means == (1.0, 12.0)
    assert stds == (1.0, 2.0)
    assert tuple(normalized[0]) == (99.0, 44.0)


def test_v04c_classifier_handles_missing_classes() -> None:
    train_x = np.asarray([[0.0], [1.0]])
    test_x = np.asarray([[0.2], [0.8]])

    predictions = nearest_centroid_predictions(train_x, ["PRESERVE", "PRESERVE"], test_x)

    assert predictions == ["PRESERVE", "PRESERVE"]


def test_v04c_macro_f1_and_confusion_matrix() -> None:
    metrics = classification_metrics(["EXPAND", "PRESERVE"], ["EXPAND", "EXPAND"])

    assert metrics["confusion_matrix"]["EXPAND"]["EXPAND"] == 1
    assert metrics["confusion_matrix"]["PRESERVE"]["EXPAND"] == 1
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_v04c_report_generation(tmp_path) -> None:
    train = [_prefuture_example("PRESERVE"), _prefuture_example("EXPAND", context_level=1, confidence=0.8)]
    test = [_prefuture_example("EXPAND", context_level=1, confidence=0.8)]
    row = evaluate_prefuture_classifier(
        game="toy",
        feature_set="contingency_only",
        classifier="nearest_centroid",
        train_seeds=(0, 1),
        test_seed=2,
        steps=10,
        horizon=2,
        train_examples=train,
        test_examples=test,
    )

    write_prefuture_reports([row], output_dir=tmp_path)

    assert (tmp_path / "prefuture_role_prediction_v04c_report.csv").exists()
    assert (tmp_path / "prefuture_role_prediction_v04c_report.json").exists()
    assert (tmp_path / "prefuture_role_prediction_v04c_report.txt").exists()
    assert (tmp_path / "prefuture_role_prediction_v04c_best.csv").exists()


def test_v04d_excludes_action_and_transformation_family_ids() -> None:
    for feature_set in ID_FREE_FEATURE_SETS.values():
        assert not (set(feature_set) & FORBIDDEN_ID_FEATURE_NAMES)
    assert all(forbidden_id_feature_check(name) for name in ID_FREE_FEATURE_SETS)


def test_v04d_excludes_future_effect_features() -> None:
    assert all(forbidden_future_feature_check(name) for name in ID_FREE_FEATURE_SETS)
    assert "mean_delta_fo" not in ID_FREE_FEATURE_SETS["all_prefuture_no_ids"]


def test_v04d_feature_set_construction() -> None:
    example = _prefuture_example("EXPAND")
    example.features["pagerank"] = 0.3

    matrix = feature_matrix_for_id_free([example], "graph_only_no_ids")

    assert len(matrix[0]) == len(ID_FREE_FEATURE_SETS["graph_only_no_ids"])
    assert matrix[0][-1] == 0.3


def test_v04d_classifier_and_metrics_with_missing_classes() -> None:
    train = [_prefuture_example("PRESERVE"), _prefuture_example("PRESERVE", confidence=0.8)]
    test = [_prefuture_example("PRESERVE")]

    row = evaluate_id_free_config(
        game="toy",
        feature_set="contingency_only_no_ids",
        classifier="nearest_centroid",
        train_seeds=(0, 1),
        test_seed=2,
        steps=10,
        horizon=2,
        train_examples=train,
        test_examples=test,
    )

    assert row["id_free_accuracy"] == 1.0
    assert row["forbidden_id_feature_check_passed"] is True
    assert row["forbidden_future_feature_check_passed"] is True


def test_v04d_report_generation(tmp_path) -> None:
    train = [_prefuture_example("PRESERVE"), _prefuture_example("EXPAND", context_level=1, confidence=0.8)]
    test = [_prefuture_example("EXPAND", context_level=1, confidence=0.8)]
    row = evaluate_id_free_config(
        game="toy",
        feature_set="contingency_only_no_ids",
        classifier="nearest_centroid",
        train_seeds=(0, 1),
        test_seed=2,
        steps=10,
        horizon=2,
        train_examples=train,
        test_examples=test,
    )

    write_id_free_reports([row], output_dir=tmp_path)

    assert (tmp_path / "id_free_prefuture_validation_v04d_report.csv").exists()
    assert (tmp_path / "id_free_prefuture_validation_v04d_report.json").exists()
    assert (tmp_path / "id_free_prefuture_validation_v04d_report.txt").exists()
    assert (tmp_path / "id_free_prefuture_validation_v04d_best.csv").exists()


def test_v05_broad_game_list_parsing_and_family_grouping() -> None:
    assert parse_game_selector("control") == ("ez01", "ez02", "ez03", "ez04")
    assert "va02" in parse_game_selector("original_primary")
    assert family_for_game("pb02") == "push_crate"
    assert family_for_game("not_real") == "unknown"


def test_v05_forbidden_feature_exclusions_in_default_sets() -> None:
    for name in ID_FREE_FEATURE_SETS:
        assert forbidden_future_feature_check(name)
        assert forbidden_id_feature_check(name)


def test_v05_best_config_selection_prefers_macro_accuracy_nonpreserve_and_simplicity() -> None:
    rows = [
        _broad_row("g1", "deterministic", "all_prefuture_no_ids", "knn3", macro=0.5, accuracy=0.6, nonp=0.5),
        _broad_row("g1", "deterministic", "contingency_only_no_ids", "knn3", macro=0.5, accuracy=0.6, nonp=0.5),
        _broad_row("g1", "vector", "", "", status="failed"),
    ]

    best = broad_best_configs(rows)

    assert next(row for row in best if row["mode"] == "deterministic")["feature_set"] == "contingency_only_no_ids"
    assert next(row for row in best if row["mode"] == "vector")["run_status"] == "failed"


def test_v05_family_summary_and_pass_fail_criteria() -> None:
    rows = [
        _broad_row("va02", "deterministic", "all_prefuture_no_ids", "knn3", macro=0.5, accuracy=0.6, majority=0.4, majority_macro=0.2, nonp=0.5),
        _broad_row("mo01", "deterministic", "all_prefuture_no_ids", "knn3", macro=0.1, accuracy=0.3, majority=0.4, majority_macro=0.2, nonp=0.0),
    ]
    best = broad_best_configs(rows)
    game_rows = summary_by_game_rows(rows, best)
    family_rows = summary_by_family_rows(game_rows)
    validation = broad_validation_summary(game_rows)

    assert game_passes(rows[0])
    assert any(row["family"] == "coverage_path" and row["games_passed"] == 1 for row in family_rows)
    assert validation["non_control_broad_games_passed"] == 1


def test_v05_per_game_failure_isolation_row() -> None:
    row = _failed_row("bad01", "deterministic", BroadValidationConfig(games=("bad01",)), "adapter failed")

    assert row["run_status"] == "failed"
    assert row["failure_reason"] == "adapter failed"


def test_v05_report_generation(tmp_path) -> None:
    rows = [
        _broad_row("va02", "deterministic", "all_prefuture_no_ids", "knn3", macro=0.5, accuracy=0.6, majority=0.4, majority_macro=0.2, nonp=0.5),
        _failed_row("va02", "vector", BroadValidationConfig(games=("va02",)), "vector unsupported"),
    ]

    write_broad_reports(rows, output_dir=tmp_path)

    assert (tmp_path / "broad_game_validation_v05_report.csv").exists()
    assert (tmp_path / "broad_game_validation_v05_report.json").exists()
    assert (tmp_path / "broad_game_validation_v05_report.txt").exists()
    assert (tmp_path / "broad_game_validation_v05_best.csv").exists()
    assert (tmp_path / "broad_game_validation_v05_summary_by_game.csv").exists()
    assert (tmp_path / "broad_game_validation_v05_summary_by_family.csv").exists()


def test_v05b_game_preset_parsing() -> None:
    assert "tt01" in parse_v05b_games("failed_families")
    assert "va02" in parse_v05b_games("passing_reference")
    assert parse_v05b_games("ez01,mo01") == ("ez01", "mo01")


def test_cli_no_longer_registers_future_effect_v02_command() -> None:
    parser = build_parser()
    subparsers_action = next(action for action in parser._actions if getattr(action, "choices", None))

    assert "future-effect-v02" not in subparsers_action.choices


def test_v05b_failure_reason_assignment() -> None:
    reasons = classify_failure_reasons(
        {
            "non_preserve_ratio": None,
            "non_preserve_count": None,
            "stable_contingency_count": 5,
            "prediction_accuracy": 0.4,
            "singleton_family_ratio": 0.6,
            "mean_family_support": 5,
        },
        {},
        [],
        [],
        "toy",
    )

    assert "legacy_future_effects_removed_use_h09_h10_h11" in reasons
    assert "WEAK_CONTINGENCY_DISCOVERY" in reasons


def test_v05b_step_and_context_sensitivity() -> None:
    config = FailureDiagnosticsConfig(games=("toy",), steps_list=(10000, 30000), context_depths=(0, 1, 3))
    prediction_rows = [
        {"game": "toy", "steps": 10000, "context_depth": 1, "id_free_accuracy": 0.4, "id_free_macro_f1": 0.2, "majority_baseline_accuracy": 0.5, "majority_baseline_macro_f1": 0.2, "non_preserve_recall_any": 0.0},
        {"game": "toy", "steps": 30000, "context_depth": 3, "id_free_accuracy": 0.6, "id_free_macro_f1": 0.3, "majority_baseline_accuracy": 0.5, "majority_baseline_macro_f1": 0.2, "non_preserve_recall_any": 1.0},
    ]
    diagnostic_rows = [
        {"game": "toy", "steps": 10000, "run_status": "ok", "non_preserve_count": 1},
        {"game": "toy", "steps": 30000, "run_status": "ok", "non_preserve_count": 30},
    ]

    step_rows = step_sensitivity_summary(config, prediction_rows, diagnostic_rows)
    context_rows = context_depth_sensitivity_summary(config, prediction_rows)

    assert step_rows[0]["diagnosis_random_policy_undersampling"] is True
    assert context_rows[0]["diagnosis_context_depth_too_shallow"] is True


def test_v05b_family_batches_fill_worker_target() -> None:
    config = FailureDiagnosticsConfig(
        games=("tt01", "pb01", "pb02", "pb03"),
        steps_list=(10000,),
        horizons=(10,),
        workers=15,
    )

    batches = _family_game_batches(config)

    assert batches[0][0] == "collection_survival+push_crate"
    assert batches[0][1] == ["tt01", "pb01", "pb02", "pb03"]


def test_v05b_family_and_failure_reason_summary() -> None:
    game_summary = [
        {"game": "va02", "family": "coverage_path", "original_v05_pass": True, "primary_failure_reason": "", "secondary_failure_reasons": [], "non_preserve_ratio": 0.1, "stable_contingency_count": 10, "prediction_accuracy": 0.8},
        {"game": "pb01", "family": "push_crate", "original_v05_pass": False, "primary_failure_reason": "PRESERVE_ONLY_OR_NEAR_PRESERVE_ONLY", "secondary_failure_reasons": ["INSUFFICIENT_NON_PRESERVE_SAMPLES"], "non_preserve_ratio": 0.0, "stable_contingency_count": 10, "prediction_accuracy": 0.8},
    ]

    family_rows = family_diagnostic_summary(game_summary)
    reason_rows = failure_reason_rows(game_summary)

    assert any(row["family"] == "push_crate" and row["dominant_failure_reason"] == "PRESERVE_ONLY_OR_NEAR_PRESERVE_ONLY" for row in family_rows)
    assert any(row["failure_reason"] == "INSUFFICIENT_NON_PRESERVE_SAMPLES" for row in reason_rows)


def test_v05b_report_generation(tmp_path) -> None:
    diagnostics = [{"game": "toy", "family": "unknown", "seed": 0, "steps": 10, "horizon": 2, "run_status": "ok"}]
    game_summary = [{"game": "toy", "family": "unknown", "original_v05_pass": False, "primary_failure_reason": "PRESERVE_ONLY_OR_NEAR_PRESERVE_ONLY", "secondary_failure_reasons": []}]
    family_summary = family_diagnostic_summary(game_summary)
    repair_rows = [{"game": "toy", "original_best_score": 0.0, "best_repair_feature_group": "v05_best_original", "repaired_score": 0.0, "repaired_macro_f1": 0.0, "repaired_non_preserve_recall": 0.0, "repair_delta": 0.0, "repair_recommendation": "no_clear_gain"}]

    write_failure_diagnostics_reports(
        diagnostics,
        game_summary,
        family_summary,
        [{"game": "toy"}],
        [{"game": "toy"}],
        [{"game": "toy"}],
        repair_rows,
        failure_reason_rows(game_summary),
        output_dir=tmp_path,
    )

    assert (tmp_path / "failure_diagnostics_v05b_report.csv").exists()
    assert (tmp_path / "failure_diagnostics_v05b_report.json").exists()
    assert (tmp_path / "failure_diagnostics_v05b_report.txt").exists()
    assert (tmp_path / "failure_diagnostics_v05b_by_family.csv").exists()
    assert (tmp_path / "failure_diagnostics_v05b_failure_reasons.csv").exists()
    assert (tmp_path / "failure_diagnostics_v05b_feature_repair.csv").exists()
    assert (tmp_path / "failure_diagnostics_v05b_recommended_next_steps.txt").exists()


def test_v05c_sampler_registry_and_aliases() -> None:
    registry = sampler_registry()

    assert "random_baseline" in registry
    assert make_sampler("mixed", seed=1).name == "mixed"
    assert make_sampler("memory_guided", seed=1).name == "memory_guided"
    assert make_sampler("memory_guided_explore", seed=1).name == "memory_guided_explore"
    assert parse_v05c_samplers("random_baseline,reset_aware_mixed") == ("random_baseline", "reset_aware_mixed")
    assert "tt01" in parse_v05c_games("failed_representatives")
    assert InteractionSamplingConfig().commit_steps == 5000


def test_v05c_games_all_expands_to_registered_games(monkeypatch) -> None:
    monkeypatch.setattr(interaction_sampling, "registered_game_ids", lambda env_root=None: ("pb02", "tt01", "ab01", "gc01"))

    assert resolve_game_ids("all") == ["ab01", "pb02", "tt01"]
    assert parse_v05c_games("all") == ("ab01", "pb02", "tt01")


def test_v05c_games_comma_separated_still_works(monkeypatch) -> None:
    monkeypatch.setattr(interaction_sampling, "registered_game_ids", lambda env_root=None: ("pb02", "tt01", "ab01"))

    assert resolve_game_ids("tt01,pb02") == ["tt01", "pb02"]
    assert parse_v05c_games("tt01,pb02") == ("tt01", "pb02")


def test_v05c_invalid_game_id_errors_clearly(monkeypatch) -> None:
    monkeypatch.setattr(interaction_sampling, "registered_game_ids", lambda env_root=None: ("pb02", "tt01"))

    try:
        resolve_game_ids("tt01,nope99")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ValueError")

    assert "invalid game id(s): nope99" in message
    assert "pb02" in message
    assert "tt01" in message


def test_v05c_action_balance_and_no_change_scores() -> None:
    balance = ActionBalanceSampler(seed=0)
    avoid = NoChangeAvoidanceSampler(seed=0)
    balance.action_counts[1] = 0
    balance.action_counts[2] = 8
    avoid.action_no_change[1].extend([1] * 50)
    avoid.action_no_change[2].extend([0] * 50)

    assert balance.action_balance_score(1) > balance.action_balance_score(2)
    assert avoid.no_change_avoidance_score(1) == 0.05
    assert avoid.no_change_avoidance_score(2) == 1.0


def test_v05c_low_confidence_and_novelty_sparse_history() -> None:
    system = V6System(env=ToggleEnv(), config=V6Config(database_path=":memory:", random_seed=0))
    low = LowConfidenceSampler(seed=0)
    novelty = NoveltyDeltaSampler(seed=0)

    assert low.low_confidence_score(system, 1) > 0.0
    assert novelty.novelty_delta_score(system, 1) > 0.0
    system.close()


def test_v05c_mixed_probability_and_epsilon_fallback() -> None:
    system = V6System(env=ToggleEnv(), config=V6Config(database_path=":memory:", random_seed=0))
    sampler = MixedExplorerSampler(seed=0, epsilon=1.0)
    actions = [1, 2, 3]

    assert sampler.choose_action(system, actions) in actions
    system.close()


def test_memory_guided_sampler_chooses_higher_scored_memory_action() -> None:
    class DummyContextBuilder:
        def multi_scale_signatures(self, action: int, max_level: int = 1) -> dict[int, tuple]:
            return {int(max_level): (f"ctx:{action}",)}

    class DummyScore:
        def __init__(self, action: int, score: float) -> None:
            self.action = int(action)
            self.score = float(score)
            self.predicted_family = None
            self.expected_future_option_delta = 0.0
            self.failure_risk = 0.0

    class DummyMemoryQuery:
        def __init__(self) -> None:
            self.selected: list[int] = []

        def rank_actions(self, contexts_by_action, actions):
            return [DummyScore(2, 0.9), DummyScore(1, 0.1)]

        def record_selected_action_query(self, *, context_signatures, action: int, prediction=None) -> None:
            del context_signatures, prediction
            self.selected.append(int(action))

    system = type(
        "System",
        (),
        {
            "context_builder": DummyContextBuilder(),
            "_context_depth_for_action": lambda self, action: 1,
            "memory_query": DummyMemoryQuery(),
        },
    )()
    sampler = MemoryGuidedSampler(seed=0, epsilon=0.0)

    chosen = sampler.choose_action(system, [1, 2])

    assert chosen == 2
    assert sampler.memory_guided_action_count == 1
    assert system.memory_query.selected == [2]


def test_memory_guided_sampler_falls_back_when_memory_scores_are_zero(monkeypatch) -> None:
    class DummyContextBuilder:
        def multi_scale_signatures(self, action: int, max_level: int = 1) -> dict[int, tuple]:
            return {int(max_level): (f"ctx:{action}",)}

    class DummyScore:
        def __init__(self, action: int) -> None:
            self.action = int(action)
            self.score = 0.0
            self.predicted_family = None
            self.expected_future_option_delta = 0.0
            self.failure_risk = 0.0

    class DummyMemoryQuery:
        def rank_actions(self, contexts_by_action, actions):
            return [DummyScore(action) for action in actions]

    system = type(
        "System",
        (),
        {
            "context_builder": DummyContextBuilder(),
            "_context_depth_for_action": lambda self, action: 1,
            "memory_query": DummyMemoryQuery(),
        },
    )()
    sampler = MemoryGuidedSampler(seed=0, epsilon=0.0)
    monkeypatch.setattr(sampler, "mixed_explorer_scores", lambda system, actions: [0.1, 0.9])

    chosen = sampler.choose_action(system, [1, 2])

    assert chosen == 2
    assert sampler.memory_guided_fallback_count == 1
    assert sampler.memory_guided_action_count == 0


def test_memory_guided_explore_sampler_can_use_exploration_path(monkeypatch) -> None:
    class DummyContextBuilder:
        def multi_scale_signatures(self, action: int, max_level: int = 1) -> dict[int, tuple]:
            return {int(max_level): (f"ctx:{action}",)}

    class DummyMemoryQuery:
        def rank_actions(self, contexts_by_action, actions):
            raise AssertionError("memory-guided path should not be used in this branch")

    system = type(
        "System",
        (),
        {
            "context_builder": DummyContextBuilder(),
            "_context_depth_for_action": lambda self, action: 1,
            "memory_query": DummyMemoryQuery(),
        },
    )()
    sampler = MemoryGuidedExploreSampler(seed=0, epsilon=0.0)
    monkeypatch.setattr(sampler.rng, "random", lambda: 0.95)
    monkeypatch.setattr(sampler, "mixed_explorer_scores", lambda system, actions: [0.2, 0.8])

    chosen = sampler.choose_action(system, [1, 2])

    assert chosen == 2
    assert sampler.memory_guided_action_count == 0


def test_v05c_reset_aware_logic() -> None:
    system = V6System(env=ToggleEnv(), config=V6Config(database_path=":memory:", random_seed=0))
    sampler = ResetAwareMixedExplorerSampler(seed=0)
    sampler.recent_no_change.extend([1] * 100)

    sampler.before_step(system)

    assert sampler.reset_unavailable is True
    system.close()


def test_v05c_db_path_does_not_overlap_v05(tmp_path) -> None:
    path = sampling_db_path(tmp_path / "sampling_v05c", "tt01", "mixed", 30000, 0)

    assert "sampling_v05c" in str(path)
    assert "future_effect_v02_dbs" not in str(path)


def test_v05c_report_generation(tmp_path) -> None:
    rows = [
        {
            "game": "tt01",
            "family": "collection_survival",
            "sampler_name": "random_baseline",
            "steps": 10,
            "horizon": 2,
            "context_depth": 1,
            "adaptive_context_expansion_enabled": True,
            "base_context_depth": 1,
            "max_context_depth": 3,
            "adaptive_context_expansion_count": 2,
            "adaptive_context_active_action_count": 1,
            "adaptive_context_max_depth_reached": 3,
            "adaptive_context_depth_used_count": 10,
            "adaptive_context_expansion_applied_count": 2,
            "run_status": "ok",
            "pass_status": False,
            "non_preserve_count": 1,
            "non_preserve_ratio": 0.1,
            "unique_transformation_families": 2,
            "stable_contingency_count": 3,
            "prediction_accuracy": 0.5,
            "id_free_accuracy": 0.4,
            "id_free_macro_f1": 0.2,
            "non_preserve_recall_any": 0.0,
            "no_change_ratio": 0.5,
            "forbidden_future_feature_check_passed": True,
            "forbidden_id_feature_check_passed": True,
        },
        {
            "game": "tt01",
            "family": "collection_survival",
            "sampler_name": "mixed",
            "steps": 10,
            "horizon": 2,
            "context_depth": 1,
            "adaptive_context_expansion_enabled": True,
            "base_context_depth": 1,
            "max_context_depth": 3,
            "adaptive_context_expansion_count": 3,
            "adaptive_context_active_action_count": 1,
            "adaptive_context_max_depth_reached": 3,
            "adaptive_context_depth_used_count": 10,
            "adaptive_context_expansion_applied_count": 3,
            "run_status": "ok",
            "pass_status": True,
            "non_preserve_count": 4,
            "non_preserve_ratio": 0.4,
            "unique_transformation_families": 4,
            "stable_contingency_count": 5,
            "prediction_accuracy": 0.6,
            "id_free_accuracy": 0.7,
            "id_free_macro_f1": 0.5,
            "non_preserve_recall_any": 1.0,
            "no_change_ratio": 0.2,
            "forbidden_future_feature_check_passed": True,
            "forbidden_id_feature_check_passed": True,
        },
    ]
    comparison = sampler_comparison_rows(rows)
    best = sampling_best_by_game(rows)
    payload = {
        "runs": rows,
        "sampler_comparison": comparison,
        "best_by_game": best,
        "summary_by_family": [],
        "validation": sampling_validation_summary(rows, comparison, best),
    }

    write_interaction_sampling_reports(payload, tmp_path)

    assert (tmp_path / "interaction_sampling_v05c_report.csv").exists()
    assert (tmp_path / "interaction_sampling_v05c_report.json").exists()
    assert (tmp_path / "interaction_sampling_v05c_report.txt").exists()
    assert (tmp_path / "interaction_sampling_v05c_sampler_comparison.csv").exists()
    assert (tmp_path / "interaction_sampling_v05c_best_by_game.csv").exists()
    assert (tmp_path / "interaction_sampling_v05c_summary_by_family.csv").exists()
    assert (tmp_path / "interaction_sampling_v05c_recommended_next_steps.txt").exists()
    report = json.loads((tmp_path / "interaction_sampling_v05c_report.json").read_text())
    assert "adaptive_context_expansion_enabled" in report["runs"][0]
    assert "base_context_depth" in report["runs"][0]
    assert "max_context_depth" in report["runs"][0]
    assert "adaptive_context_expansion_count" in report["runs"][0]
    assert "adaptive_context_active_action_count" in report["runs"][0]
    assert "adaptive_context_max_depth_reached" in report["runs"][0]
    assert "adaptive_context_depth_used_count" in report["runs"][0]
    assert "adaptive_context_expansion_applied_count" in report["runs"][0]


def test_v05c_validation_summary_reports_operational_failure() -> None:
    rows = [
        {
            "game": "tt01",
            "family": "collection_survival",
            "sampler_name": "random_baseline",
            "run_status": "failed",
            "failure_reason": "KeyError: pagerank",
        }
    ]

    summary = sampling_validation_summary(rows, [], [])

    assert summary["diagnostic_success"] is False
    assert summary["failed_run_count"] == 1
    assert summary["failure_reason_counts"]["KeyError: pagerank"] == 1
    assert summary["scientific_conclusion"] is None


def test_storage_backends_write_and_query_matching_rows(tmp_path) -> None:
    records = [
        {"game": "toy", "seed": 0, "step": 1, "action": 1, "terminal": 0, "reset": 0},
        {"game": "toy", "seed": 0, "step": 2, "action": 2, "terminal": 1, "reset": 1},
    ]
    sqlite_backend = SQLiteStorageBackend(tmp_path / "sqlite" / "run.sqlite", batch_size=10)
    sqlite_backend.write_interactions(records)
    sqlite_backend.finalize()
    parquet_backend = ParquetStorageBackend(root=tmp_path / "parquet", game="toy", sampler="mixed", seed=0, steps=2, batch_size=10)
    parquet_backend.write_interactions(records)
    parquet_backend.finalize()

    sqlite_connection = __import__("sqlite3").connect(tmp_path / "sqlite" / "run.sqlite")
    try:
        sqlite_rows = sqlite_connection.execute("SELECT game, seed, step, action, terminal, reset FROM interactions ORDER BY step").fetchall()
    finally:
        sqlite_connection.close()
    parquet_rows = load_run_table(tmp_path / "parquet", "interactions", game="toy", sampler="mixed", seed=0, steps=2)

    assert len(sqlite_rows) == len(parquet_rows)
    assert list(parquet_rows["action"]) == [1, 2]


def test_v05c_missing_pagerank_defaults_without_crash() -> None:
    example = PrefutureExample(
        contingency_id=1,
        contingency_key=(1, (1,), 1, 2),
        features={
            "context_level": 1.0,
            "confidence": 0.5,
            "support_count_log": 1.0,
            "prediction_error_rate": 0.0,
            "context_support": 1.0,
            "action_entropy_at_context": 0.0,
            "transformation_entropy_at_context": 0.0,
            "transformation_family_support_log": 1.0,
            "changed_cells": 1.0,
            "dx": 0.0,
            "dy": 0.0,
            "colors_added_count": 0.0,
            "colors_removed_count": 0.0,
            "contingency_in_degree": 0.0,
            "contingency_out_degree": 0.0,
            "follows_in_degree": 0.0,
            "follows_out_degree": 0.0,
            "cooccurrence_degree": 0.0,
            "clustering_coefficient": 0.0,
            "degree_centrality": 0.2,
        },
        label="PRESERVE",
    )

    rows = feature_matrix_for_id_free([example], "graph_only_no_ids")

    assert len(rows) == 1
    assert rows[0][-1] == 0.2


def test_parquet_backend_chunking_and_finalize_flush(tmp_path) -> None:
    backend = ParquetStorageBackend(root=tmp_path, game="toy", sampler="mixed", seed=0, steps=10, batch_size=2, compression="zstd")
    backend.write_deltas(
        [
            {"step": 1, "changed_cells": 1, "dx": 0.0, "dy": 1.0},
            {"step": 2, "changed_cells": 2, "dx": 1.0, "dy": 0.0},
            {"step": 3, "changed_cells": 3, "dx": 1.0, "dy": 1.0},
        ]
    )
    backend.finalize()

    parts = sorted((tmp_path / "game=toy" / "sampler=mixed" / "seed=0" / "steps=10").glob("deltas_part_*.parquet"))
    rows = load_run_table(tmp_path, "deltas", game="toy", sampler="mixed", seed=0, steps=10)

    assert len(parts) == 2
    assert len(rows) == 3


def test_sqlite_to_parquet_migration_tiny_db(tmp_path) -> None:
    sqlite_path = tmp_path / "source.sqlite"
    connection = __import__("sqlite3").connect(sqlite_path)
    try:
        connection.execute("CREATE TABLE interactions (game TEXT, seed INTEGER, step INTEGER, action INTEGER, terminal INTEGER, reset INTEGER)")
        connection.execute("CREATE TABLE deltas (step INTEGER, changed_cells INTEGER, dx REAL, dy REAL)")
        connection.execute("CREATE TABLE contingencies (context_level INTEGER, confidence REAL, support_count INTEGER)")
        connection.execute("CREATE TABLE future_effects (contingency_id INTEGER, future_effect_class TEXT)")
        connection.execute("INSERT INTO interactions VALUES ('toy', 0, 1, 4, 0, 0)")
        connection.execute("INSERT INTO deltas VALUES (1, 2, 1.0, 0.0)")
        connection.execute("INSERT INTO contingencies VALUES (1, 0.9, 5)")
        connection.execute("INSERT INTO future_effects VALUES (7, 'EXPAND')")
        connection.commit()
    finally:
        connection.close()

    migrate_sqlite_to_parquet(sqlite_path=sqlite_path, parquet_root=tmp_path / "parquet", game="toy", sampler="mixed", seed=0, steps=1)
    interactions = load_run_table(tmp_path / "parquet", "interactions", game="toy", sampler="mixed", seed=0, steps=1)
    effects = load_run_table(tmp_path / "parquet", "future_effects", game="toy", sampler="mixed", seed=0, steps=1)

    assert len(interactions) == 1
    assert list(effects["future_effect_class"]) == ["EXPAND"]


def test_cli_accepts_parquet_storage_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "interaction-sampling-v05c",
            "--games",
            "tt01",
            "--samplers",
            "random_baseline",
            "--storage-backend",
            "parquet",
            "--parquet-root",
            "runs/v6/storage_parquet",
            "--storage-batch-size",
            "500",
            "--compress",
            "zstd",
        ]
    )

    assert args.storage_backend == "parquet"
    assert args.storage_batch_size == 500


def _grid_at(y: int, x: int) -> np.ndarray:
    grid = np.zeros((4, 4), dtype=int)
    grid[y, x] = 3
    return grid


def _delta_with_features(delta_id: int, *, changed_cells: int, dx: float, dy: float) -> Delta:
    return Delta(
        id=delta_id,
        changed_cells=changed_cells,
        changed_positions=[],
        colors_added=[],
        colors_removed=[],
        centroid_before_x=0.0,
        centroid_before_y=0.0,
        centroid_after_x=dx,
        centroid_after_y=dy,
        dx=dx,
        dy=dy,
    )


def _create_validation_fixture_schema(connection) -> None:
    connection.executescript(
        """
        CREATE TABLE transformation_families (
            id INTEGER PRIMARY KEY,
            centroid_vector TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            member_delta_ids TEXT NOT NULL
        );
        CREATE TABLE contingencies (
            id INTEGER PRIMARY KEY,
            context_level INTEGER NOT NULL,
            context_signature TEXT NOT NULL,
            action INTEGER NOT NULL,
            transformation_family INTEGER NOT NULL,
            support_count INTEGER NOT NULL,
            confidence REAL NOT NULL
        );
        CREATE TABLE prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL,
            context_level INTEGER,
            context_signature TEXT NOT NULL,
            action INTEGER NOT NULL,
            predicted_family INTEGER,
            actual_family INTEGER,
            prediction_error INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO transformation_families (id, centroid_vector, support_count, member_delta_ids) VALUES (10, '[1,0,0,0,0]', 25, '[]')"
    )
    connection.execute(
        "INSERT INTO transformation_families (id, centroid_vector, support_count, member_delta_ids) VALUES (11, '[2,0,0,0,0]', 25, '[]')"
    )
    connection.commit()


def _insert_fixture_contingency(connection, row_id: int, level: int, context: list[int], action: int, family: int, support: int, confidence: float) -> None:
    import json

    connection.execute(
        """
        INSERT INTO contingencies (
            id, context_level, context_signature, action, transformation_family, support_count, confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (row_id, level, json.dumps(context), action, family, support, confidence),
    )
    connection.commit()


def _insert_fixture_prediction(connection, interaction_id: int, *, action: int, actual_family: int) -> None:
    connection.execute(
        """
        INSERT INTO prediction_results (
            interaction_id, context_level, context_signature, action, predicted_family, actual_family, prediction_error
        )
        VALUES (?, NULL, '[]', ?, NULL, ?, NULL)
        """,
        (interaction_id, action, actual_family),
    )
    connection.commit()


def _future_effect(
    row_id: int,
    *,
    contingency_id: int,
    context_level: int,
    action: int,
    family: int,
    effect_class: str,
    mean_delta: float,
) -> FutureEffect:
    return FutureEffect(
        id=row_id,
        game="toy",
        seed=0,
        steps=10,
        horizon=2,
        contingency_id=contingency_id,
        context_level=context_level,
        action=action,
        transformation_family=family,
        occurrence_count=5,
        skipped_occurrence_count=0,
        mean_fo_before=1.0,
        mean_fo_after=1.0 + mean_delta,
        mean_delta_fo=mean_delta,
        median_delta_fo=mean_delta,
        std_delta_fo=0.0,
        positive_delta_ratio=1.0 if mean_delta > 0 else 0.0,
        negative_delta_ratio=1.0 if mean_delta < 0 else 0.0,
        zero_delta_ratio=1.0 if mean_delta == 0 else 0.0,
        collapse_ratio=1.0 if effect_class == "COLLAPSE" else 0.0,
        future_effect_class=effect_class,
    )


def _validation_example(contingency_id: int, *, context_level: int, family: int, label: str, delta: float) -> ValidationExample:
    return ValidationExample(
        contingency_id=contingency_id,
        contingency_key=(context_level, (1,), 1, family),
        raw_vector=(
            float(context_level),
            1.0,
            float(family),
            0.9,
            3.0,
            1.0,
            1.0 + float(delta),
            float(delta),
            0.0,
            1.0 if delta > 0.0 else 0.0,
            1.0 if delta < 0.0 else 0.0,
            1.0 if delta == 0.0 else 0.0,
            0.0,
        ),
        label=label,
    )


def _prefuture_example(label: str, *, context_level: int = 0, confidence: float = 1.0) -> PrefutureExample:
    return PrefutureExample(
        contingency_id=1,
        contingency_key=(context_level, (1,), 1, 2),
        label=label,
        features={
            "context_level": float(context_level),
            "action_id": 1.0,
            "transformation_family_id": 2.0,
            "confidence": float(confidence),
            "support_count_log": 3.0,
            "prediction_error_rate": 0.0,
            "context_support": 10.0,
            "action_entropy_at_context": 0.5,
            "transformation_entropy_at_context": 0.5,
            "transformation_family_support_log": 4.0,
            "changed_cells": 2.0,
            "dx": 1.0,
            "dy": 0.0,
            "colors_added_count": 1.0,
            "colors_removed_count": 0.0,
            "contingency_in_degree": 1.0,
            "contingency_out_degree": 2.0,
            "follows_in_degree": 1.0,
            "follows_out_degree": 1.0,
            "cooccurrence_degree": 3.0,
            "clustering_coefficient": 0.5,
            "degree_centrality": 0.2,
        },
    )


def _broad_row(
    game: str,
    mode: str,
    feature_set: str,
    classifier: str,
    *,
    macro: float = 0.0,
    accuracy: float = 0.0,
    majority: float = 0.0,
    majority_macro: float = 0.0,
    nonp: float = 0.0,
    status: str = "ok",
) -> dict:
    return {
        "game": game,
        "mode": mode,
        "feature_set": feature_set,
        "classifier": classifier,
        "train_seeds": [0, 1],
        "test_seed": 2,
        "steps": 10,
        "horizon": 2,
        "train_sample_count": 1,
        "test_sample_count": 1,
        "class_distribution_train": {"PRESERVE": 1},
        "class_distribution_test": {"PRESERVE": 1},
        "majority_baseline_accuracy": majority,
        "majority_baseline_macro_f1": majority_macro,
        "contingency_baseline_accuracy": majority,
        "random_stratified_accuracy": majority,
        "id_free_accuracy": accuracy,
        "id_free_macro_f1": macro,
        "id_free_vs_majority_delta": accuracy - majority,
        "id_free_vs_contingency_delta": accuracy - majority,
        "preserve_precision": 1.0,
        "preserve_recall": 1.0,
        "expand_precision": 0.0,
        "expand_recall": nonp,
        "restrict_precision": 0.0,
        "restrict_recall": 0.0,
        "collapse_precision": 0.0,
        "collapse_recall": 0.0,
        "non_preserve_recall_any": nonp,
        "non_preserve_class_count_train": 1 if nonp else 0,
        "non_preserve_class_count_test": 1 if nonp else 0,
        "confusion_matrix_json": {},
        "forbidden_future_feature_check_passed": True,
        "forbidden_id_feature_check_passed": True,
        "run_status": status,
        "failure_reason": "",
    }


def test_v06_loads_small_synthetic_parquet_table(tmp_path) -> None:
    root = _build_v06_fixture(tmp_path / "parquet")
    runs = discover_parquet_runs(root, games=("va02",))

    assert len(runs) == 1
    assert runs[0].game == "va02"
    assert runs[0].sampler == "mixed"


def test_v06_builds_m0_episode_summaries(tmp_path) -> None:
    root = _build_v06_fixture(tmp_path / "parquet")
    run = discover_parquet_runs(root, games=("va02",))[0]

    loaded = load_partition_events(run, context_depth=1)

    assert len(loaded["episodes"]) >= 2
    assert loaded["episodes"][0].steps_total > 0
    assert loaded["episodes"][0].trajectory_cost == loaded["episodes"][0].steps_total
    assert loaded["episodes"][0].success_observed is True
    assert loaded["episodes"][0].terminal_observed is True
    assert loaded["episodes"][0].terminal_type == "WIN"


def test_v06_maps_levels_completed_and_terminal_outcomes() -> None:
    state = RunStreamingState(game_id="va02", sampler="mixed", seed=0, context_depth=1, exact_reconstruction=False)
    before = _obs(0)
    after = _obs(1)
    base_row = {
        "id": 1,
        "timestamp": 1,
        "observation_before": encode_array(before),
        "observation_after": encode_array(after),
        "action": 4,
        "delta_id": 1,
    }
    delta_map = {1: {"changed_cells": 1, "dx": 1.0, "dy": 0.0}}

    event_level, _, _ = process_interaction_row(
        {**base_row, "outcome_state": "NOT_FINISHED", "outcome_polarity": "positive", "level_completed_event": 1},
        state=state,
        delta_map=delta_map,
    )
    assert event_level.outcome_signature == "progress:levels_completed"
    assert event_level.terminal_observed is False

    for outcome_state in ("GAME_OVER", "WIN"):
        terminal_state = RunStreamingState(game_id="va02", sampler="mixed", seed=0, context_depth=1, exact_reconstruction=False)
        event_terminal, _, _ = process_interaction_row(
            {**base_row, "outcome_state": outcome_state, "outcome_polarity": "negative", "is_terminal_outcome": 1},
            state=terminal_state,
            delta_map=delta_map,
        )
        assert event_terminal.outcome_signature == f"terminal:{outcome_state}"


def test_v06_groups_by_context_and_action() -> None:
    context0 = build_context_signature(__import__("collections").deque([], maxlen=1), 1)
    context1 = ("a3|ochange",)

    events = [
        _trace_event(step=0, action=4, context=context0, outcome="change"),
        _trace_event(step=1, action=4, context=context0, outcome="change"),
        _trace_event(step=2, action=4, context=context1, outcome="blocked_no_change"),
    ]

    contingencies = build_m1_contingencies(events, min_support=1, prediction_threshold=0.5)

    assert len(contingencies) == 2
    assert sorted(item.total_count for item in contingencies) == [1, 2]


def test_v06_discovers_deterministic_contingency() -> None:
    events = [_trace_event(step=index, action=4, context=tuple(), outcome="change") for index in range(5)]

    contingencies = build_m1_contingencies(events, min_support=5, prediction_threshold=0.75)

    assert len(contingencies) == 1
    assert contingencies[0].discovered is True
    assert contingencies[0].prediction_accuracy == 1.0


def test_v06_rejects_low_support_contingency() -> None:
    events = [_trace_event(step=index, action=4, context=tuple(), outcome="change") for index in range(3)]

    contingencies = build_m1_contingencies(events, min_support=5, prediction_threshold=0.75)

    assert len(contingencies) == 1
    assert contingencies[0].discovered is False


def test_v06_computes_action_only_baseline_and_context_lift() -> None:
    events = [
        _trace_event(step=0, action=4, context=("a1|opreserve",), outcome="change"),
        _trace_event(step=1, action=4, context=("a1|opreserve",), outcome="change"),
        _trace_event(step=2, action=4, context=("a2|ochange",), outcome="blocked_no_change"),
        _trace_event(step=3, action=4, context=("a2|ochange",), outcome="blocked_no_change"),
    ]
    contingencies = build_m1_contingencies(events, min_support=1, prediction_threshold=0.5)

    baseline = action_only_accuracy(events)
    contextual = context_model_accuracy(contingencies)

    assert baseline == 0.5
    assert contextual == 1.0
    assert contextual - baseline > 0.0


def test_v06_does_not_require_forbidden_fields() -> None:
    before = np.zeros((2, 2), dtype=int)
    after = before.copy()

    outcome = classify_outcome(before, after, {"changed_cells": None, "dx": 0.0, "dy": 0.0})

    assert outcome == "blocked_no_change"


def test_v06_does_not_require_screen_delta_table() -> None:
    summary = build_episode_summary([_trace_event(step=0, action=4, context=tuple(), outcome="blocked_no_change")])

    assert summary.blocked_or_no_change_count == 1


def test_v06_terminal_transition_gets_terminate_candidate() -> None:
    events = [_trace_event(step=index, action=5, context=tuple(), outcome="terminal_transition") for index in range(5)]

    contingencies = build_m1_contingencies(events, min_support=2, prediction_threshold=0.75)

    assert contingencies[0].future_option_motif_candidate == "terminate_candidate"


def test_v06_writes_report_and_parquet_outputs(tmp_path) -> None:
    root = _build_v06_fixture(tmp_path / "parquet")
    output_dir = tmp_path / "out"

    payload = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("va02",),
            output_dir=str(output_dir),
            context_depth=1,
            min_support=2,
            prediction_threshold=0.75,
        )
    )

    assert (output_dir / "contingencies.json").exists()
    assert (output_dir / "contingencies.parquet").exists()
    assert (output_dir / "episode_summaries.parquet").exists()
    assert (output_dir / "v06_report.json").exists()
    assert (output_dir / "v06_report.txt").exists()
    assert payload["validation"]["milestone_classification"] in {
        "m1_not_established",
        "m1_weak",
        "m1_strong",
        "m1_very_strong",
    }


def test_cli_accepts_contingency_memory_v06_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "contingency-memory-v06",
            "--parquet-root",
            "runs/v6/p",
            "--games",
            "tt01,pb02",
            "--samplers",
            "random_baseline,novelty_delta",
            "--seeds",
            "0,1",
            "--output-dir",
            "runs/v6/v06",
            "--context-depth",
            "1",
            "--min-support",
            "5",
            "--prediction-threshold",
            "0.75",
            "--streaming",
            "--max-files",
            "100",
        ]
    )

    assert args.parquet_root == "runs/v6/p"
    assert args.min_support == 5
    assert args.max_files == 100


def test_v06_cli_manifest_out_default_is_none() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "contingency-memory-v06",
            "--parquet-root",
            "runs/v6/p",
        ]
    )

    assert args.manifest_out is None


def test_v06_determine_manifest_path_variants(tmp_path) -> None:
    output_dir = tmp_path / "out"
    explicit = ContingencyMemoryConfig(parquet_root="runs/v6/p", manifest_out=str(tmp_path / "explicit.json"))
    replayed = ContingencyMemoryConfig(parquet_root="runs/v6/p", manifest_in=str(tmp_path / "input.json"))
    defaulted = ContingencyMemoryConfig(parquet_root="runs/v6/p")

    assert determine_manifest_path(explicit, output_dir) == tmp_path / "explicit.json"
    assert determine_manifest_path(replayed, output_dir) == output_dir / "input_manifest.replayed.json"
    assert determine_manifest_path(defaulted, output_dir) == output_dir / "input_manifest.json"


def test_v06_manifest_creation_filters_game_sampler_seed(tmp_path) -> None:
    root = _build_v06_manifest_fixture(tmp_path / "parquet")
    discovered = list_interaction_files(__import__("pathlib").Path(root))
    manifest = build_input_manifest(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("gr01",),
            samplers=("no_change_avoidance",),
            seeds=(0,),
            streaming=True,
        ),
        discovered,
        warnings=[],
    )

    assert manifest["selected_file_count"] > 0
    assert all("game=gr01" in path for path in manifest["selected_files"])
    assert all("sampler=no_change_avoidance" in path for path in manifest["selected_files"])
    assert all("seed=0" in path for path in manifest["selected_files"])


def test_v06_streaming_schema_failure_marks_validation_invalid(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(_build_v06_fixture(tmp_path / "parquet"))
    interaction_path = next(root.glob("game=*/sampler=*/seed=*/steps=*/interactions*.parquet"))
    delta_path = next(root.glob("game=*/sampler=*/seed=*/steps=*/deltas*.parquet"))

    bad_interactions = pa.table(
        {
            "id": pa.array([1], type=pa.int64()),
            "timestamp": pa.array([1], type=pa.int64()),
            "observation_before": pa.array([encode_array(_obs(0))], type=pa.binary()),
            "observation_after": pa.array([encode_array(_obs(1))], type=pa.binary()),
            "delta_id": pa.array([1], type=pa.int64()),
        }
    )
    pq.write_table(bad_interactions, interaction_path, compression="zstd")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": 1,
                    "changed_cells": 1,
                    "changed_positions": json.dumps([]),
                    "colors_added": json.dumps([]),
                    "colors_removed": json.dumps([]),
                    "centroid_before_x": 0.0,
                    "centroid_before_y": 0.0,
                    "centroid_after_x": 1.0,
                    "centroid_after_y": 0.0,
                    "dx": 1.0,
                    "dy": 0.0,
                }
            ]
        ),
        delta_path,
        compression="zstd",
    )

    payload = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("va02",),
            output_dir=str(tmp_path / "out"),
            context_depth=1,
            min_support=2,
            prediction_threshold=0.75,
            streaming=True,
        )
    )

    assert payload["validation"]["schema_valid"] is False
    assert payload["validation"]["failed_load_count"] > 0


def test_v06_empty_outputs_use_readable_parquet_schemas(tmp_path) -> None:
    import pandas as pd

    root = tmp_path / "empty_parquet"
    root.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "out"

    payload = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("va02",),
            output_dir=str(output_dir),
            context_depth=1,
            min_support=2,
            prediction_threshold=0.75,
            streaming=True,
        )
    )

    contingencies = pd.read_parquet(output_dir / "contingencies.parquet")
    episodes = pd.read_parquet(output_dir / "episode_summaries.parquet")
    edges = pd.read_parquet(output_dir / "m1_graph_edges.parquet")

    assert payload["scan_summary"]["files_selected"] == 0
    assert "_empty" not in contingencies.columns
    assert "_empty" not in episodes.columns
    assert "_empty" not in edges.columns
    assert {"contingency_id", "game_id", "action", "discovered"}.issubset(contingencies.columns)
    assert {"game_id", "sampler", "seed", "episode_id", "success_observed"}.issubset(episodes.columns)
    assert {"edge_type", "game_id", "sampler", "seed", "source_interaction_id", "target_interaction_id"}.issubset(edges.columns)


def test_v06_manifest_respects_max_files(tmp_path) -> None:
    root = _build_v06_manifest_fixture(tmp_path / "parquet")
    discovered = list_interaction_files(__import__("pathlib").Path(root))
    manifest = build_input_manifest(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("gr01", "va02"),
            samplers=("no_change_avoidance", "novelty_delta"),
            seeds=(0, 1),
            max_files=2,
            streaming=True,
        ),
        discovered,
        warnings=[],
    )

    assert manifest["selected_file_count"] == 2


def test_v06_streaming_accumulator_matches_bounded_full_mode(tmp_path) -> None:
    root = _build_v06_manifest_fixture(tmp_path / "parquet")
    streaming = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("gr01",),
            samplers=("no_change_avoidance",),
            seeds=(0,),
            output_dir=str(tmp_path / "streaming"),
            context_depth=1,
            min_support=1,
            prediction_threshold=0.5,
            streaming=True,
        )
    )
    full = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("gr01",),
            samplers=("no_change_avoidance",),
            seeds=(0,),
            output_dir=str(tmp_path / "full"),
            context_depth=1,
            min_support=1,
            prediction_threshold=0.5,
            streaming=False,
        )
    )

    assert streaming["report"]["total_contingency_candidates"] == full["report"]["total_contingency_candidates"]
    assert streaming["report"]["total_discovered_contingencies"] == full["report"]["total_discovered_contingencies"]


def test_v06_manifest_replay_processes_same_files(tmp_path) -> None:
    root = _build_v06_manifest_fixture(tmp_path / "parquet")
    first = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("gr01",),
            samplers=("no_change_avoidance",),
            seeds=(0,),
            output_dir=str(tmp_path / "first"),
            context_depth=1,
            min_support=1,
            prediction_threshold=0.5,
            streaming=True,
            manifest_out=str(tmp_path / "first" / "manifest.json"),
        )
    )
    second = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("gr01",),
            samplers=("random_baseline",),
            seeds=(1,),
            output_dir=str(tmp_path / "second"),
            context_depth=1,
            min_support=1,
            prediction_threshold=0.5,
            streaming=True,
            manifest_in=str(tmp_path / "first" / "manifest.json"),
        )
    )

    assert first["manifest"]["selected_files"] == second["manifest"]["selected_files"]


def test_v06_empty_selection_fails_clearly(tmp_path) -> None:
    root = _build_v06_manifest_fixture(tmp_path / "parquet")
    payload = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("tt01",),
            samplers=("reset_aware_mixed",),
            seeds=(9,),
            output_dir=str(tmp_path / "empty"),
            streaming=True,
        )
    )

    assert payload["validation"]["diagnostic_success"] is False
    assert payload["scan_summary"]["files_selected"] == 0


def test_v06_progress_reporting_does_not_crash(capsys) -> None:
    print_progress(rows_processed=100, files_processed=2, contingencies_built=3, elapsed=1.5)

    captured = capsys.readouterr()
    assert "rows=100" in captured.out


def test_v06_max_rows_bounds_processing(tmp_path) -> None:
    root = _build_v06_manifest_fixture(tmp_path / "parquet")
    payload = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("gr01",),
            samplers=("no_change_avoidance",),
            seeds=(0,),
            output_dir=str(tmp_path / "bounded"),
            context_depth=1,
            min_support=1,
            prediction_threshold=0.5,
            streaming=True,
            max_rows=3,
        )
    )

    assert payload["scan_summary"]["rows_processed"] == 3


def test_v06_report_formatter_mentions_scan_summary(tmp_path) -> None:
    root = _build_v06_manifest_fixture(tmp_path / "parquet")
    payload = run_contingency_memory_v06(
        ContingencyMemoryConfig(
            parquet_root=str(root),
            games=("gr01",),
            samplers=("no_change_avoidance",),
            seeds=(0,),
            output_dir=str(tmp_path / "format"),
            context_depth=1,
            min_support=1,
            prediction_threshold=0.5,
            streaming=True,
        )
    )

    text = format_v06_report(payload)

    assert "files_selected=" in text
    assert "rows_processed=" in text


def test_v07_loads_synthetic_m1_contingencies(tmp_path) -> None:
    input_dir = _build_v07_fixture(tmp_path / "v06")

    rows = load_m1_contingencies(input_dir)

    assert len(rows) == 5
    assert rows[0].family_label_candidate


def test_v07_groups_identical_outcomes_into_one_family(tmp_path) -> None:
    input_dir = _build_v07_fixture(tmp_path / "v06")
    rows = load_m1_contingencies(input_dir)

    families = build_m2_families(rows[:2], min_family_support=1, similarity_threshold=0.70)

    assert len(families) == 1


def test_v07_keeps_low_coherence_contingencies_separate(tmp_path) -> None:
    input_dir = _build_v07_fixture(tmp_path / "v06")
    rows = load_m1_contingencies(input_dir)

    families = build_m2_families([rows[0], rows[2]], min_family_support=1, similarity_threshold=0.95)

    assert len(families) == 2


def test_v07_computes_compression_ratio(tmp_path) -> None:
    payload = run_transformation_families_v07(
        TransformationFamiliesV07Config(
            input_dir=str(_build_v07_fixture(tmp_path / "v06")),
            output_dir=str(tmp_path / "v07"),
            min_family_support=1,
            similarity_threshold=0.70,
        )
    )

    assert payload["report"]["compression_ratio"] > 1.0


def test_v07_computes_family_coherence(tmp_path) -> None:
    payload = run_transformation_families_v07(
        TransformationFamiliesV07Config(
            input_dir=str(_build_v07_fixture(tmp_path / "v06")),
            output_dir=str(tmp_path / "v07"),
            min_family_support=1,
            similarity_threshold=0.70,
        )
    )

    assert payload["report"]["mean_family_coherence"] >= 0.0


def test_v07_detects_cross_game_family_presence(tmp_path) -> None:
    payload = run_transformation_families_v07(
        TransformationFamiliesV07Config(
            input_dir=str(_build_v07_fixture(tmp_path / "v06")),
            output_dir=str(tmp_path / "v07"),
            min_family_support=1,
            similarity_threshold=0.70,
        )
    )

    assert any(family["cross_game_presence"] > 1 for family in payload["report"]["cross_game_families"])


def test_v07_does_not_require_semantic_fields(tmp_path) -> None:
    input_dir = _build_v07_fixture(tmp_path / "v06")
    rows = load_m1_contingencies(input_dir)

    assert all(not hasattr(row, "object_name") for row in rows)


def test_v07_writes_family_outputs(tmp_path) -> None:
    out = tmp_path / "v07"
    run_transformation_families_v07(
        TransformationFamiliesV07Config(
            input_dir=str(_build_v07_fixture(tmp_path / "v06")),
            output_dir=str(out),
            min_family_support=1,
            similarity_threshold=0.70,
        )
    )

    assert (out / "m2_families.json").exists()
    assert (out / "m2_families.parquet").exists()
    assert (out / "contingency_to_family.parquet").exists()
    assert (out / "v07_report.json").exists()
    assert (out / "v07_report.txt").exists()


def test_v07_validation_conclusion_generation() -> None:
    fake_families = []
    conclusion = v07_validation_summary(families=fake_families, contingencies=[], compression_ratio=0.0)

    assert conclusion["scientific_conclusion"] == "m2_not_established"
    assert conclusion["M3_status"] == "not tested"


def test_v07_report_explicitly_says_m3_m4_not_tested(tmp_path) -> None:
    payload = run_transformation_families_v07(
        TransformationFamiliesV07Config(
            input_dir=str(_build_v07_fixture(tmp_path / "v06")),
            output_dir=str(tmp_path / "v07"),
            min_family_support=1,
            similarity_threshold=0.70,
        )
    )

    text = format_v07_report(payload)

    assert "M3=not tested" in text
    assert "M4=not tested" in text


def test_cli_accepts_transformation_families_v07_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "transformation-families-v07",
            "--input-dir",
            "runs/v6/v06",
            "--output-dir",
            "runs/v6/v07",
            "--min-family-support",
            "5",
            "--similarity-threshold",
            "0.70",
        ]
    )

    assert args.input_dir == "runs/v6/v06"
    assert args.min_family_support == 5


def test_v07_context_depth_cd1_beats_cd0_and_cd2_fragments(tmp_path) -> None:
    runs = _build_context_compare_fixture(tmp_path)
    payload = run_context_depth_compare_v07(
        ContextDepthCompareConfig(
            runs=(str(runs["cd0_v07"]), str(runs["cd1_v07"]), str(runs["cd2_v07"]), str(runs["cd3_v07"])),
            labels=("cd0", "cd1", "cd2", "cd3"),
            output_dir=str(tmp_path / "compare"),
        )
    )

    assert payload["best_context_depth"] == 1
    assert payload["context_depth_0_status"] == "under_contextualized"
    assert payload["context_depth_1_status"] == "balanced"
    assert payload["context_depth_2_status"] == "over_fragmented"
    assert payload["cd3_status"] == "over_contextualized"


def test_v07_context_depth_tie_break_prefers_cd1_with_less_than_five_percent_gain(tmp_path) -> None:
    runs = _build_context_compare_fixture(tmp_path, tie_break=True)
    payload = run_context_depth_compare_v07(
        ContextDepthCompareConfig(
            runs=(str(runs["cd0_v07"]), str(runs["cd1_v07"]), str(runs["cd2_v07"]), str(runs["cd3_v07"])),
            labels=("cd0", "cd1", "cd2", "cd3"),
            output_dir=str(tmp_path / "compare_tie"),
        )
    )

    assert payload["best_context_depth"] == 2
    assert "less than 5%" in payload["reason"] or "keep cd2" in payload["reason"]


def test_v07_context_depth_cd3_useful_case_can_win_on_extended_fixture(tmp_path) -> None:
    runs = _build_context_compare_fixture(tmp_path, cd3_useful=True, extended=True)
    payload = run_context_depth_compare_v07(
        ContextDepthCompareConfig(
            runs=(str(runs["cd0_v07"]), str(runs["cd1_v07"]), str(runs["cd2_v07"]), str(runs["cd3_v07"])),
            labels=("cd0", "cd1", "cd2", "cd3"),
            output_dir=str(tmp_path / "compare_cd3_useful"),
        )
    )

    assert payload["best_context_depth"] == 3
    assert payload["cd3_status"] == "balanced"
    assert payload["best_context_depth_extended"] == 3


def test_v07_context_depth_report_includes_best_depth_and_reason(tmp_path) -> None:
    runs = _build_context_compare_fixture(tmp_path)
    payload = run_context_depth_compare_v07(
        ContextDepthCompareConfig(
            runs=(str(runs["cd0_v07"]), str(runs["cd1_v07"]), str(runs["cd2_v07"]), str(runs["cd3_v07"])),
            labels=("cd0", "cd1", "cd2", "cd3"),
            output_dir=str(tmp_path / "compare_report"),
        )
    )

    text = format_context_depth_comparison(payload)

    assert "best_context_depth=1" in text
    assert "reason=" in text


def test_v07_context_depth_no_oracle_inputs_required(tmp_path) -> None:
    runs = _build_context_compare_fixture(tmp_path)
    payload = run_context_depth_compare_v07(
        ContextDepthCompareConfig(
            runs=(str(runs["cd0_v07"]), str(runs["cd1_v07"]), str(runs["cd2_v07"]), str(runs["cd3_v07"])),
            labels=("cd0", "cd1", "cd2", "cd3"),
            output_dir=str(tmp_path / "compare_no_oracle"),
        )
    )

    assert payload["runs"][0]["run_dir"]
    assert "reason" in payload


def test_cli_accepts_compare_context_depth_v07_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-context-depth-v07",
            "--runs",
            "runs/v6/v07_cd0,runs/v6/v07_cd1,runs/v6/v07_cd2,runs/v6/v07_cd3",
            "--labels",
            "cd0,cd1,cd2,cd3",
            "--output-dir",
            "runs/v6/v07_context_compare",
        ]
    )

    assert args.output_dir == "runs/v6/v07_context_compare"
    assert args.labels == "cd0,cd1,cd2,cd3"


def test_v08_neighborhood_fingerprint_generation() -> None:
    families, m1_support = _v08_fixture_records()
    neighborhoods = build_neighborhoods(families, m1_support, {"g1": "movement", "g2": "toggle_switch", "g3": "push_pull"})

    record = neighborhoods["m2-a"]
    assert record.fingerprint["blocked_frequency"] > 0.0
    assert record.fingerprint["mean_prediction_accuracy"] > 0.8
    assert record.dominant_outcome_signature == "blocked_no_change"


def test_v08_similar_neighborhoods_high_similarity() -> None:
    families, m1_support = _v08_fixture_records()
    neighborhoods = build_neighborhoods(families, m1_support, {"g1": "movement", "g2": "toggle_switch", "g3": "push_pull"})
    pair = next(
        item
        for item in evaluate_pairwise_similarity(neighborhoods, threshold=0.7, workers=1)
        if {item.left_family_id, item.right_family_id} == {"m2-a", "m2-b"}
    )

    assert pair.similarity >= 0.7


def test_v08_different_neighborhoods_low_similarity() -> None:
    families, m1_support = _v08_fixture_records()
    neighborhoods = build_neighborhoods(families, m1_support, {"g1": "movement", "g2": "toggle_switch", "g3": "push_pull"})
    pair = next(
        item
        for item in evaluate_pairwise_similarity(neighborhoods, threshold=0.7, workers=1)
        if {item.left_family_id, item.right_family_id} == {"m2-a", "m2-c"}
    )

    assert pair.similarity < 0.7


def test_v08_stable_role_candidate_creation() -> None:
    families, m1_support = _v08_fixture_records()
    neighborhoods = build_neighborhoods(families, m1_support, {"g1": "movement", "g2": "toggle_switch", "g3": "push_pull"})
    pair_results = evaluate_pairwise_similarity(neighborhoods, threshold=0.7, workers=1)
    adjacency = build_similarity_adjacency(pair_results, 0.7)
    clusters, _rejected = cluster_role_candidates(adjacency, pair_results, 0.7)
    roles = build_role_candidates(clusters=clusters, neighborhoods=neighborhoods, min_role_support=2, role_similarity_threshold=0.7)

    stable = next(role for role in roles if role.status == "stable")
    assert stable.cross_game_support >= 2
    assert stable.cross_game_family_support >= 2
    assert stable.role_consistency_score >= 0.7


def test_v08_workers_1_equals_workers_25_on_synthetic_fixture(tmp_path) -> None:
    input_dir, m1_dir = _write_v08_fixture(tmp_path)
    payload_one = run_role_candidates_v08(
        RoleCandidatesV08Config(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "out1"),
            workers=1,
            game_set_manifest=str(tmp_path / "game_set.json"),
        )
    )
    payload_many = run_role_candidates_v08(
        RoleCandidatesV08Config(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "out25"),
            workers=25,
            game_set_manifest=str(tmp_path / "game_set.json"),
        )
    )

    assert payload_one["validation"]["scientific_conclusion"] == payload_many["validation"]["scientific_conclusion"]
    assert payload_one["report"]["stable_clusters"] == payload_many["report"]["stable_clusters"]
    assert payload_one["report"]["cross_family_clusters"] == payload_many["report"]["cross_family_clusters"]


def test_cli_accepts_role_candidates_v08_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "role-candidates-v08",
            "--input-dir",
            "runs/v6/v07_cd2",
            "--m1-input-dir",
            "runs/v6/v06_cd2",
            "--output-dir",
            "runs/v6/v08_cd2_core7",
            "--context-depth",
            "2",
            "--min-role-support",
            "3",
            "--role-similarity-threshold",
            "0.70",
            "--workers",
            "25",
            "--partition-by",
            "family_pair,neighborhood_shard",
            "--game-set-name",
            "core7",
        ]
    )

    assert args.command == "role-candidates-v08"
    assert args.game_set_name == "core7"
    assert args.workers == 25


def test_v08c_detects_and_splits_overcompressed_family(tmp_path) -> None:
    input_dir, m1_dir, manifest = _write_v08c_fixture(tmp_path)

    payload = run_m2_expand_v08c(
        M2ExpandV08cConfig(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "expanded"),
            game_set_manifest=str(manifest),
            min_family_support=1,
            max_family_share=0.40,
            min_expanded_families=3,
            target_expanded_families=6,
        )
    )

    assert payload["report"]["diagnostics_before"]["overcompressed_family_count"] >= 1
    assert payload["report"]["expanded_m2_family_count"] > payload["report"]["original_m2_family_count"]
    assert payload["report"]["diagnostics_after"]["largest_family_percent"] < payload["report"]["diagnostics_before"]["largest_family_percent"]


def test_v08c_expansion_map_is_complete_and_no_contingency_is_lost(tmp_path) -> None:
    input_dir, m1_dir, manifest = _write_v08c_fixture(tmp_path)
    out = tmp_path / "expanded"
    payload = run_m2_expand_v08c(
        M2ExpandV08cConfig(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(out),
            game_set_manifest=str(manifest),
            min_family_support=1,
            max_family_share=0.40,
            min_expanded_families=3,
            target_expanded_families=6,
        )
    )

    import pandas as pd

    mapping = pd.read_parquet(out / "contingency_to_family.parquet")
    expansion = pd.read_parquet(out / "m2_expansion_map.parquet")
    assert len(mapping) == 6
    assert set(mapping["contingency_id"]) == {
        "m1-g1-s1-0001",
        "m1-g2-s1-0002",
        "m1-g3-s1-0003",
        "m1-g4-s1-0004",
        "m1-g1-s1-0005",
        "m1-g4-s1-0006",
    }
    assert set(expansion["original_family_id"]) == {"m2-a", "m2-b"}
    assert payload["validation"]["diagnostic_success"] is True


def test_v08c_v08_can_consume_expanded_outputs(tmp_path) -> None:
    input_dir, m1_dir, manifest = _write_v08c_fixture(tmp_path)
    expanded = tmp_path / "expanded"
    run_m2_expand_v08c(
        M2ExpandV08cConfig(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(expanded),
            game_set_manifest=str(manifest),
            min_family_support=1,
            max_family_share=0.40,
            min_expanded_families=3,
            target_expanded_families=6,
        )
    )

    payload = run_role_candidates_v08(
        RoleCandidatesV08Config(
            input_dir=str(expanded),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "v08"),
            workers=1,
            game_set_manifest=str(manifest),
        )
    )

    assert (tmp_path / "v08" / "m3_role_candidates.json").exists()
    assert payload["validation"]["diagnostic_success"] is True


def test_cli_accepts_m2_expand_v08c_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "m2-expand-v08c",
            "--input-dir",
            "runs/v6/v07_cd2_extended32",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--output-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--max-family-share",
            "0.25",
            "--min-expanded-families",
            "40",
            "--target-expanded-families",
            "60",
        ]
    )

    assert args.command == "m2-expand-v08c"
    assert args.max_family_share == 0.25


def test_v08d_directional_and_future_option_profiles(tmp_path) -> None:
    input_dir, m1_dir, manifest = _write_v08c_fixture(tmp_path)
    expanded = tmp_path / "expanded"
    run_m2_expand_v08c(
        M2ExpandV08cConfig(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(expanded),
            game_set_manifest=str(manifest),
            min_family_support=1,
            max_family_share=0.40,
            min_expanded_families=3,
            target_expanded_families=6,
        )
    )
    families = load_m2_families_v08d(expanded)
    support = load_m1_support_v08d(m1_dir)
    neighborhoods, _ = build_discriminative_neighborhoods(families, support, {"g1": "collection", "g2": "push_crate", "g3": "switch_unlock", "g4": "teleport_warp"})

    record = next(iter(neighborhoods.values()))
    assert "predecessor_count" in record.directional_features
    assert "enable_score" in record.future_option_features


def test_v08d_reversible_loop_detection_and_label_evidence(tmp_path) -> None:
    input_dir, m1_dir = _write_v08_fixture(tmp_path)
    payload = run_role_candidates_v08d(
        RoleCandidatesV08dConfig(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "outd"),
            workers=1,
            game_set_manifest=str(tmp_path / "game_set.json"),
        )
    )

    rows = json.loads((tmp_path / "outd" / "m3_role_candidates.json").read_text(encoding="utf-8"))
    assert all("label_evidence" in row for row in rows)
    assert role_status("blocker_candidate", [M3RoleCandidate(**row) for row in rows]) in {"stable", "weak", "singleton", "absent"}


def test_v08d_workers_1_equals_workers_25_on_small_fixture(tmp_path) -> None:
    input_dir, m1_dir = _write_v08_fixture(tmp_path)
    payload_one = run_role_candidates_v08d(
        RoleCandidatesV08dConfig(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "out1"),
            workers=1,
            game_set_manifest=str(tmp_path / "game_set.json"),
        )
    )
    payload_many = run_role_candidates_v08d(
        RoleCandidatesV08dConfig(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "out25"),
            workers=25,
            game_set_manifest=str(tmp_path / "game_set.json"),
        )
    )

    assert payload_one["validation"]["scientific_conclusion"] == payload_many["validation"]["scientific_conclusion"]
    assert payload_one["report"]["stable_clusters"] == payload_many["report"]["stable_clusters"]
    assert json.loads((tmp_path / "out1" / "m3_role_candidates.json").read_text()) == json.loads((tmp_path / "out25" / "m3_role_candidates.json").read_text())


def test_cli_accepts_role_candidates_v08d_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "role-candidates-v08d",
            "--input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--output-dir",
            "runs/v6/v08_cd2_extended32_discriminative",
            "--fingerprint-mode",
            "discriminative",
            "--weight-coarse",
            "0.25",
            "--weight-directional",
            "0.20",
        ]
    )

    assert args.command == "role-candidates-v08d"
    assert args.fingerprint_mode == "discriminative"


def test_v09_leave_family_out_split_and_role_prototypes(tmp_path) -> None:
    m3_dir, m2_dir, m1_dir, manifest = _write_v09_fixture(tmp_path)
    neighborhoods = load_neighborhoods(m3_dir / "role_neighborhoods.parquet")
    roles = load_roles(m3_dir / "m3_role_candidates.json")
    source = {family_id: row for family_id, row in neighborhoods.items() if "g3" not in row.games_present and "g4" not in row.games_present}
    prototypes = build_source_role_prototypes(roles, source, min_source_role_support=1)

    assert prototypes
    assert all(proto.family_ids for proto in prototypes.values())


def test_v09_runs_and_computes_lift(tmp_path) -> None:
    m3_dir, m2_dir, m1_dir, manifest = _write_v09_fixture(tmp_path)
    payload = run_role_transfer_v09(
        RoleTransferV09Config(
            m3_input_dir=str(m3_dir),
            m2_input_dir=str(m2_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "v09"),
            game_set_manifest=str(manifest),
            workers=1,
            min_source_role_support=1,
            min_target_family_support=1,
        )
    )

    assert payload["report"]["heldout_family_count"] == 2
    assert "evaluable_heldout_family_count" in payload["report"]
    assert "role_transfer_lift_vs_random" in payload["report"]
    assert (tmp_path / "v09" / "v09_report.json").exists()
    assert "h2_interpretation=" in (tmp_path / "v09" / "v09_report.txt").read_text(encoding="utf-8")


def test_cli_accepts_role_transfer_v09_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "role-transfer-v09",
            "--m3-input-dir",
            "runs/v6/v08_cd2_extended32_discriminative",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--output-dir",
            "runs/v6/v09_role_transfer_extended32",
            "--split-mode",
            "leave_family_out",
            "--workers",
            "25",
        ]
    )

    assert args.command == "role-transfer-v09"
    assert args.split_mode == "leave_family_out"


def test_v09b_medoid_prototype_selection(tmp_path) -> None:
    m3_dir, _, _, _ = _write_v09_fixture(tmp_path)
    neighborhoods = load_neighborhoods(m3_dir / "role_neighborhoods.parquet")
    members = [neighborhoods["f1"], neighborhoods["f2"]]
    idx = medoid_index([all_features(member) for member in members])

    assert idx in {0, 1}


def test_v09b_family_balanced_prototype_calculation(tmp_path) -> None:
    m3_dir, _, _, _ = _write_v09_fixture(tmp_path)
    neighborhoods = load_neighborhoods(m3_dir / "role_neighborhoods.parquet")
    roles = load_roles(m3_dir / "m3_role_candidates.json")
    role = next(row for row in roles if row.role_id == "r1")
    entry = build_prototype_entry(StrategySpec("fb", "family_balanced_centroid"), role, [neighborhoods["f1"], neighborhoods["f2"]])

    assert entry is not None
    assert len(entry.vectors) == 1


def test_v09b_top_k_assignment_and_confidence(tmp_path) -> None:
    m3_dir, _, _, _ = _write_v09_fixture(tmp_path)
    neighborhoods = load_neighborhoods(m3_dir / "role_neighborhoods.parquet")
    roles = load_roles(m3_dir / "m3_role_candidates.json")
    role = next(row for row in roles if row.role_id == "r1")
    entry = build_prototype_entry(StrategySpec("topk", "top_k_neighbors", top_k=1), role, [neighborhoods["f1"], neighborhoods["f2"]])
    ranked = rank_roles_for_target(neighborhoods["f1"], {"r1": entry}, mode="weighted_structural", top_k=1)

    assert ranked[0][0] == "r1"
    assert ranked[0][1] > 0.9
    assert passes_confidence_gate(StrategySpec("gate", "top_k_neighbors", similarity_threshold=0.7, margin=0.05, confidence_mode="threshold_margin_gating"), ranked[0][1], 0.80) is True


def test_v09b_subtype_mode_does_not_create_new_roles(tmp_path) -> None:
    m3_dir, _, _, _ = _write_v09_fixture(tmp_path)
    neighborhoods = load_neighborhoods(m3_dir / "role_neighborhoods.parquet")
    roles = load_roles(m3_dir / "m3_role_candidates.json")
    role = next(row for row in roles if row.role_id == "r2")
    entry = build_prototype_entry(StrategySpec("subtype", "subtype_aware"), role, [neighborhoods["f3"], neighborhoods["f4"]])

    assert entry is not None
    assert entry.role_id == "r2"


def test_v09b_strategy_ranking(tmp_path) -> None:
    payload_a = {"strategy_metrics": {"strategy_name": "a", "positive_lift_families": 3, "lift_vs_raw_m2": 0.01, "mean_role_lift_over_best_baseline": 0.1, "transfer_accuracy_structural_role": 0.8, "coverage_rate": 1.0, "confident_assignment_precision": 0.9, "low_confidence_assignment_rate": 0.0}}
    payload_b = {"strategy_metrics": {"strategy_name": "b", "positive_lift_families": 6, "lift_vs_raw_m2": 0.02, "mean_role_lift_over_best_baseline": 0.08, "transfer_accuracy_structural_role": 0.79, "coverage_rate": 1.0, "confident_assignment_precision": 0.9, "low_confidence_assignment_rate": 0.0}}

    best = select_best_strategy_payload([payload_a, payload_b])

    assert best["strategy_metrics"]["strategy_name"] == "b"


def test_v09b_source_clean_split_and_deterministic_fixture(tmp_path, monkeypatch) -> None:
    manifest, prev_v09a, tracker = _install_v09b_sourceclean_fixture(tmp_path, monkeypatch)
    import v6.role_transfer_v09b as mod

    monkeypatch.setattr(
        mod,
        "build_strategy_specs",
        lambda: [
            StrategySpec("centroid_test", "centroid"),
            StrategySpec("family_balanced_test", "family_balanced_centroid"),
            StrategySpec("subtype_margin_test", "subtype_aware", confidence_mode="threshold_margin_gating", similarity_threshold=0.7, margin=0.03),
        ],
    )
    one = run_role_transfer_v09b(
        RoleTransferV09bConfig(
            m2_input_dir=str(tmp_path / "m2"),
            m1_input_dir=str(tmp_path / "m1"),
            previous_v09a_dir=str(prev_v09a),
            output_dir=str(tmp_path / "v09b1"),
            game_set_manifest=str(manifest),
            workers=1,
        )
    )
    many = run_role_transfer_v09b(
        RoleTransferV09bConfig(
            m2_input_dir=str(tmp_path / "m2"),
            m1_input_dir=str(tmp_path / "m1"),
            previous_v09a_dir=str(prev_v09a),
            output_dir=str(tmp_path / "v09b25"),
            game_set_manifest=str(manifest),
            workers=2,
        )
    )

    assert one["validation"]["scientific_conclusion"] == many["validation"]["scientific_conclusion"]
    assert one["report"]["best_strategy"]["strategy_name"] == many["report"]["best_strategy"]["strategy_name"]
    assert one["report"]["best_strategy"]["transfer_accuracy_structural_role"] == many["report"]["best_strategy"]["transfer_accuracy_structural_role"]
    assert all(heldout not in member_games for heldout, member_games in tracker["source_games_by_holdout"])
    import pandas as pd

    best_assignments = pd.read_parquet(tmp_path / "v09b1" / "v09b_best_strategy_assignments.parquet")
    assert "ground_truth_role_id" not in best_assignments.columns
    assert (tmp_path / "v09b1" / "v09b_regression_vs_v09a.parquet").exists()


def test_v09b_unknown_role_handling_modes(tmp_path) -> None:
    m3_dir, _, _, _ = _write_v09_fixture(tmp_path)
    neighborhoods = load_neighborhoods(m3_dir / "role_neighborhoods.parquet")
    roles = load_roles(m3_dir / "m3_role_candidates.json")
    role = next(row for row in roles if row.role_id == "r1")
    known = build_prototype_entry(StrategySpec("known", "centroid"), role, [neighborhoods["f1"], neighborhoods["f2"]])
    unknown = build_prototype_entry(StrategySpec("unknown", "centroid"), role, [neighborhoods["f1"], neighborhoods["f2"]])
    assert known is not None and unknown is not None
    unknown = unknown.__class__(**{**unknown.__dict__, "role_id": "u1", "role_label_candidate": "unknown_role_candidate", "unknown_role": True})

    excluded = rank_roles_for_target(
        neighborhoods["f1"],
        {"r1": known, "u1": unknown},
        mode="weighted_structural",
        top_k=0,
        strategy=StrategySpec("exclude", "centroid", unknown_mode="exclude_unknown_roles_from_assignment"),
    )
    included = rank_roles_for_target(
        neighborhoods["f1"],
        {"r1": known, "u1": unknown},
        mode="weighted_structural",
        top_k=0,
        strategy=StrategySpec("include", "centroid", unknown_mode="include_unknown_roles"),
    )

    assert [row[0] for row in excluded] == ["r1"]
    assert {row[0] for row in included} == {"r1", "u1"}


def test_cli_accepts_role_transfer_v09b_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "role-transfer-v09b",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--previous-v09a-dir",
            "runs/v6/v09a_role_transfer_sourceclean_extended32",
            "--output-dir",
            "runs/v6/v09b_role_transfer_refined_sourceclean_extended32",
            "--split-mode",
            "leave_family_out",
            "--workers",
            "25",
        ]
    )

    assert args.command == "role-transfer-v09b"
    assert args.split_mode == "leave_family_out"


def test_v09c_surface_bins_and_challenge_selectors() -> None:
    assert surface_similarity_bin(0.40) == "low_surface_similarity"
    assert surface_similarity_bin(0.70) == "medium_surface_similarity"
    assert surface_similarity_bin(0.90) == "high_surface_similarity"
    assert select_same_effect_different_role_case(
        {"challenge_core_score": 0.82, "effect_score": 0.92},
        {"challenge_core_score": 0.60, "effect_score": 0.93},
        0.90,
    ) is True
    assert select_same_role_different_effect_case(
        {"challenge_core_score": 0.75, "effect_score": 0.60},
        0.70,
        0.65,
    ) is True


def test_v09c_graph_position_excludes_effect_features(tmp_path) -> None:
    m3_dir, _, _, _ = _write_v09_fixture(tmp_path)
    neighborhoods = load_neighborhoods(m3_dir / "role_neighborhoods.parquet")
    left = neighborhoods["f1"]
    right = left.__class__(**{**left.__dict__, "temporal_effect_features": {"no_change_rate": 0.0, "position_change_rate": 1.0, "terminal_rate": 1.0}})

    assert graph_position_features(left) == graph_position_features(right)


def test_v09c_future_option_not_reducible_to_effect_labels(tmp_path) -> None:
    m3_dir, _, _, _ = _write_v09_fixture(tmp_path)
    neighborhoods = load_neighborhoods(m3_dir / "role_neighborhoods.parquet")
    left = neighborhoods["f1"]
    right = left.__class__(**{**left.__dict__, "future_option_features": {"reachable_before_rate": 0.1, "reachable_after_rate": 0.9, "enable_score": 0.8, "block_score": 0.1, "preserve_score": 0.2, "terminate_score": 0.0, "reversibility_score": 0.9}})

    assert future_option_behavior_features(left) != future_option_behavior_features(right)
    assert left.temporal_effect_features == right.temporal_effect_features


def test_v09c_runs_deterministically_on_hardened_fixture(tmp_path, monkeypatch) -> None:
    prev_v09b = _write_v09c_previous_report(tmp_path)
    contexts = _build_v09c_fixture_contexts()
    import v6.role_transfer_v09c as mod

    monkeypatch.setattr(mod, "prepare_family_contexts", lambda config: contexts)
    monkeypatch.setattr(mod, "load_best_v09b_strategy", lambda _path: StrategySpec("topk1", "top_k_neighbors", top_k=1, unknown_mode="include_unknown_roles_but_downweight"))

    one = run_role_transfer_v09c(
        RoleTransferV09cConfig(
            m2_input_dir=str(tmp_path / "m2"),
            m1_input_dir=str(tmp_path / "m1"),
            previous_v09b_dir=str(prev_v09b),
            output_dir=str(tmp_path / "v09c1"),
            workers=1,
        )
    )
    many = run_role_transfer_v09c(
        RoleTransferV09cConfig(
            m2_input_dir=str(tmp_path / "m2"),
            m1_input_dir=str(tmp_path / "m1"),
            previous_v09b_dir=str(prev_v09b),
            output_dir=str(tmp_path / "v09c2"),
            workers=2,
        )
    )

    assert one["validation"]["scientific_conclusion"] == many["validation"]["scientific_conclusion"]
    assert one["report"]["lift_vs_surface_effect_hardened"] == many["report"]["lift_vs_surface_effect_hardened"]
    import pandas as pd

    assignments = pd.read_parquet(tmp_path / "v09c1" / "v09c_hardened_assignments.parquet")
    assert "ground_truth_role_id" not in assignments.columns
    assert all(heldout not in source_games for heldout, source_games in [(ctx.heldout_games[0], {game for record in ctx.source_neighborhoods.values() for game in record.game_ids}) for ctx in contexts])


def test_cli_accepts_role_transfer_v09c_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "role-transfer-v09c",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--previous-v09b-dir",
            "runs/v6/v09b_role_transfer_refined_sourceclean_extended32",
            "--output-dir",
            "runs/v6/v09c_transfer_hardened_extended32",
            "--split-mode",
            "leave_family_out",
            "--workers",
            "25",
        ]
    )

    assert args.command == "role-transfer-v09c"
    assert args.split_mode == "leave_family_out"


def test_cli_accepts_concept_candidates_v10fix_c_previous_v09b_dir() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "concept-candidates-v10fix-c",
            "--previous-v09b-dir",
            "runs/custom_v09b",
        ]
    )

    assert args.command == "concept-candidates-v10fix-c"
    assert args.previous_v09b_dir == "runs/custom_v09b"


def test_v10fixc_passes_configured_previous_v09b_dir(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.concept_candidates_v10fixc as mod

    transfer_dir = tmp_path / "transfer"
    transfer_dir.mkdir()
    (transfer_dir / "v09c_report.json").write_text(json.dumps({"report": {}, "validation": {}}), encoding="utf-8")
    captured: dict[str, str] = {}

    monkeypatch.setattr(mod, "_load_optional_json", lambda path: {})
    monkeypatch.setattr(mod.pd, "read_parquet", lambda path: pd.DataFrame())
    monkeypatch.setattr(mod, "load_source_manifest_family_map", lambda path: {})
    monkeypatch.setattr(mod, "load_game_to_manifest_family", lambda config: {})

    def fake_prepare_family_context_stream(role_config):
        captured["previous_v09b_dir"] = role_config.previous_v09b_dir
        return iter(())

    monkeypatch.setattr(mod, "prepare_family_context_stream", fake_prepare_family_context_stream)
    monkeypatch.setattr(mod, "load_shard_group", lambda shards_dir, pattern: [])
    monkeypatch.setattr(mod, "build_collision_rows", lambda rows: [])
    monkeypatch.setattr(mod, "merge_exact_candidates", lambda rows, **kwargs: ([], {}))
    monkeypatch.setattr(mod, "fuzzy_group_candidates", lambda rows, **kwargs: ([], [], {}))
    monkeypatch.setattr(mod, "remap_concept_ids", lambda rows, mapping: rows)
    monkeypatch.setattr(mod, "apply_target_metrics", lambda rows, mapped_transfer_rows: [])
    monkeypatch.setattr(mod, "annotate_projection_outcomes", lambda rows, mapped_transfer_rows: [])
    monkeypatch.setattr(mod, "build_attrition_rows_fixc", lambda **kwargs: [])
    monkeypatch.setattr(mod, "build_label_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_role_composition_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_graph_edges", lambda rows: [])
    monkeypatch.setattr(mod, "build_surface_comparison_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_target_projection_mode_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_concept_by_family_rows", lambda concept_rows, mapped_transfer_rows: [])
    monkeypatch.setattr(
        mod,
        "build_report_payload_fixc",
        lambda **kwargs: {
            "config": {"previous_v09b_dir": kwargs["config"].previous_v09b_dir},
            "report": {},
            "validation": {},
        },
    )
    monkeypatch.setattr(mod, "_write_parquet", lambda path, rows: None)
    monkeypatch.setattr(mod, "format_report_fixc", lambda payload: "")

    payload = run_concept_candidates_v10fixc(
        ConceptCandidatesV10FixCConfig(
            transfer_input_dir=str(transfer_dir),
            previous_v09b_dir="runs/custom_v09b",
            output_dir=str(tmp_path / "out"),
        )
    )

    assert captured["previous_v09b_dir"] == "runs/custom_v09b"
    assert payload["config"]["previous_v09b_dir"] == "runs/custom_v09b"


def test_v10_runtime_sources_only_keep_v09b_path_as_dataclass_default() -> None:
    expected = "runs/v6/v09b_role_transfer_refined_sourceclean_extended32"
    paths = [
        Path("src/v6/m4_role_concepts_v10e.py"),
        Path("src/v6/concept_candidates_v10fixc.py"),
        Path("src/v6/concept_candidates_v10fixd.py"),
    ]

    for path in paths:
        assert path.read_text(encoding="utf-8").count(expected) == 1


def test_v10_concept_sequence_and_motif_extraction() -> None:
    motif = extract_role_graph_motif(
        [
            {"role_id": "r1", "predecessor_count": 0.0, "successor_count": 2.0, "future_delta": 0.8},
            {"role_id": "r2", "predecessor_count": 1.0, "successor_count": 1.0, "future_delta": 0.4},
            {"role_id": "r3", "predecessor_count": 2.0, "successor_count": 0.0, "future_delta": 0.1},
        ]
    )

    assert motif in {"chain", "source_to_sink"}
    assert sequence_similarity(("r1", "r2", "r3"), ("r1", "r2")) > 0.5


def test_v10_concept_generation_and_lift_over_role_baseline(tmp_path) -> None:
    roles, neighborhoods, transfer_rows, family_to_role, role_label_by_id, game_to_manifest = _build_v10_fixture_parts()
    concepts = discover_concept_candidates(
        source_neighborhoods=neighborhoods,
        source_assignments=transfer_rows,
        family_to_role=family_to_role,
        role_label_by_id=role_label_by_id,
        game_to_manifest_family=game_to_manifest,
        config=ConceptCandidatesV10Config(min_games=2, min_manifest_families=2, min_role_count=2, max_role_count=4),
    )

    assert concepts
    assert all(len(item.role_ids) >= 2 for item in concepts)
    row = next(iter(concepts))
    assert row.hardened_transfer_score > 0


def test_v10_source_clean_validation_and_determinism(tmp_path, monkeypatch) -> None:
    import v6.concept_candidates_v10 as mod

    m3_dir, transfer_dir, manifest = _write_v10_fixture(tmp_path)
    one = run_concept_candidates_v10(
        ConceptCandidatesV10Config(
            m3_input_dir=str(m3_dir),
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10a"),
            game_set_manifest=str(manifest),
            workers=1,
            min_games=2,
            min_manifest_families=2,
        )
    )
    many = run_concept_candidates_v10(
        ConceptCandidatesV10Config(
            m3_input_dir=str(m3_dir),
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10b"),
            game_set_manifest=str(manifest),
            workers=2,
            min_games=2,
            min_manifest_families=2,
        )
    )

    assert one["validation"]["scientific_conclusion"] == many["validation"]["scientific_conclusion"]
    assert one["report"]["concept_candidates_total"] == many["report"]["concept_candidates_total"]
    import pandas as pd

    scores = pd.read_parquet(tmp_path / "v10a" / "concept_transfer_scores.parquet")
    assert "ground_truth_role_id" not in scores.columns


def test_cli_accepts_concept_candidates_v10_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "concept-candidates-v10",
            "--m3-input-dir",
            "runs/v6/v08d_cd2_extended32_sourceclean",
            "--transfer-input-dir",
            "runs/v6/v09c_transfer_hardened_extended32",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--output-dir",
            "runs/v6/v10_m4_concepts_extended32",
            "--workers",
            "25",
        ]
    )

    assert args.command == "concept-candidates-v10"
    assert args.workers == 25


def test_v10fix_concept_ids_are_deterministic_and_merge_cleanly() -> None:
    structure = {
        "role_ids": ("r1", "r2"),
        "role_labels": ("blocker_candidate", "movement_controller_candidate"),
        "graph_ordered_role_pattern": ("r1", "r2"),
        "episode_ordered_role_sequence": (),
        "episode_order_available": False,
        "graph_order_fallback_used": True,
        "motif_type": "chain",
        "future_option_delta_profile": {"enable_score": 0.8, "reachable_delta_mean": 0.7},
        "graph_position_profile": {"source_like_score": 0.8, "bridge_like_score": 0.4},
        "local_motif_profile": {"local_branching_score": 0.2},
        "predecessor_successor_profile": {"predecessor_count": 0.5, "successor_count": 1.5},
        "temporal_profile": {"early_episode_frequency": 0.3},
        "effect_residual_profile": {"source_effect_complexity": 0.2},
        "games_present": ("g1", "g2", "g3"),
        "manifest_families_present": ("mfA", "mfB"),
        "source_role_support": 4,
        "source_concept_quality_score": 0.6,
        "required_roles_present": ["blocker_candidate", "movement_controller_candidate"],
    }
    same_a = stable_concept_signature(structure)
    same_b = stable_concept_signature(dict(structure))
    different = stable_concept_signature({**structure, "graph_ordered_role_pattern": ("r2", "r1")})

    assert same_a["concept_id"] == same_b["concept_id"]
    assert same_a["concept_fingerprint_hash"] == same_b["concept_fingerprint_hash"]
    assert same_a["concept_id"] != different["concept_id"]

    merged = merge_concept_rows(
        [
            [{"concept_id": same_a["concept_id"], "concept_signature_json": same_a["concept_signature_json"], "games_present": ["g1"], "manifest_families_present": ["mfA"], "support_count": 1, "source_role_support": 1, "source_concept_quality_score": 0.6}],
            [{"concept_id": same_b["concept_id"], "concept_signature_json": same_b["concept_signature_json"], "games_present": ["g2"], "manifest_families_present": ["mfB"], "support_count": 2, "source_role_support": 2, "source_concept_quality_score": 0.8}],
            [{"concept_id": different["concept_id"], "concept_signature_json": different["concept_signature_json"], "games_present": ["g3"], "manifest_families_present": ["mfC"], "support_count": 1, "source_role_support": 1, "source_concept_quality_score": 0.7}],
        ]
    )

    assert len(merged) == 2
    merged_same = next(row for row in merged if row["concept_id"] == same_a["concept_id"])
    assert set(merged_same["games_present"]) == {"g1", "g2"}


def test_v10fix_target_projection_excludes_role_overlap_from_main_score() -> None:
    contexts = _build_v09c_fixture_contexts()
    context = contexts[0]
    target_record = context.full_neighborhoods["tb1"]
    target_rows = [{"assigned_role_id": "role_block", "surface_hardened_score": 0.15, "effect_residual_score": 0.2, "future_option_role_score": 0.8, "graph_position_role_score": 0.7}]
    source_role_map = {family_id: {"role_id": "role_block", "role_label_candidate": "blocker_candidate", "all_features": role["all_features"]} for family_id, role in [("sb1", context.source_roles["role_block"])]}
    base_profiles = {
        "future_option_delta_profile": future_option_behavior_features(target_record),
        "graph_position_profile": graph_position_features(target_record),
        "local_motif_profile": target_record.local_motif_features,
        "predecessor_successor_profile": {
            "predecessor_count": float(target_record.directional_features.get("predecessor_count", 0.0)),
            "successor_count": float(target_record.directional_features.get("successor_count", 0.0)),
            "directional_asymmetry_score": float(target_record.directional_features.get("directional_asymmetry_score", 0.0)),
        },
        "temporal_profile": {
            "early_episode_frequency": float(target_record.temporal_effect_features.get("early_episode_frequency", 0.0)),
            "mid_episode_frequency": float(target_record.temporal_effect_features.get("mid_episode_frequency", 0.0)),
            "late_episode_frequency": float(target_record.temporal_effect_features.get("late_episode_frequency", 0.0)),
            "reversible_effect_rate": float(target_record.temporal_effect_features.get("reversible_effect_rate", 0.0)),
        },
        "effect_residual_profile": {"target_effect_residual": 0.2},
        "source_concept_quality_score": 0.6,
        "source_role_support": 2,
        "graph_ordered_role_pattern": ("role_block", "role_move"),
        "episode_ordered_role_sequence": (),
    }
    with_overlap = evaluate_target_projection({"role_ids": ("role_block", "role_move"), **base_profiles}, target_record, target_rows, source_role_map, context)
    no_overlap = evaluate_target_projection({"role_ids": ("role_x", "role_y"), **base_profiles}, target_record, target_rows, source_role_map, context)

    assert with_overlap["target_concept_prediction_score"] == no_overlap["target_concept_prediction_score"]
    assert with_overlap["role_id_overlap_diagnostic"] > no_overlap["role_id_overlap_diagnostic"]


def test_v10fix_access_control_requires_stricter_evidence() -> None:
    weak = {
        "role_labels": ("blocker_candidate", "unknown_role_candidate"),
        "future_option_delta_profile": {"reachable_delta_mean": 0.0, "enable_score": 0.0, "block_score": 0.0},
        "motif_type": "chain",
        "source_concept_quality_score": 0.7,
        "manifest_family_count": 3,
        "role_ids": ("r1", "r2"),
    }
    movement = {
        "role_labels": ("blocker_candidate", "movement_controller_candidate"),
        "future_option_delta_profile": {"reachable_delta_mean": 0.1, "enable_score": 0.0, "block_score": 0.0},
        "motif_type": "bridge",
        "source_concept_quality_score": 0.7,
        "manifest_family_count": 3,
        "role_ids": ("r1", "r2"),
    }

    weak_label, _ = strict_label_candidate(weak)
    movement_label, movement_evidence = strict_label_candidate(movement)

    assert weak_label != "access_control_concept"
    assert movement_label == "movement_constraint_concept"
    assert "access_control_concept" in movement_evidence["rejected_alternative_labels"]


def test_v10fix_runs_source_clean_and_deterministically(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.concept_candidates_v10fix as mod

    transfer_dir, manifest = _write_v10fix_fixture(tmp_path)
    contexts = _build_v10fix_fixture_contexts()
    monkeypatch.setattr(mod, "prepare_family_contexts", lambda config: contexts)

    one = run_concept_candidates_v10fix(
        ConceptCandidatesV10FixConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fix1"),
            game_set_manifest=str(manifest),
            workers=1,
            min_games=2,
            min_manifest_families=2,
        )
    )
    many = run_concept_candidates_v10fix(
        ConceptCandidatesV10FixConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fix2"),
            game_set_manifest=str(manifest),
            workers=2,
            min_games=2,
            min_manifest_families=2,
        )
    )

    assert one["validation"]["scientific_conclusion"] == many["validation"]["scientific_conclusion"]
    assert one["report"]["concept_id_collision_check_passed"] is True
    assert one["report"]["source_only_concept_discovery"] is True
    assert one["report"]["target_role_id_overlap_removed_from_main_score"] is True
    assert "target_mean_concept_lift_vs_surface_raw" in one["report"]
    assert "target_mean_concept_lift_vs_surface_hardened" in one["report"]
    assert one["report"]["graph_order_fallback_used"] is True
    assert all(ctx.heldout_games[0] not in {game for record in ctx.source_neighborhoods.values() for game in record.game_ids} for ctx in contexts)

    scores = pd.read_parquet(tmp_path / "v10fix1" / "concept_transfer_scores_fixed.parquet")
    assert "ground_truth_role_id" not in scores.columns
    assert "surface_effect_raw_score" in scores.columns
    assert "surface_effect_hardened_score" in scores.columns


def test_cli_accepts_concept_candidates_v10fix_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "concept-candidates-v10fix",
            "--m3-input-dir",
            "runs/v6/v08d_cd2_extended32_sourceclean",
            "--transfer-input-dir",
            "runs/v6/v09c_transfer_hardened_extended32",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--output-dir",
            "runs/v6/v10_m4_concepts_methodology_fixed_extended32",
            "--workers",
            "25",
        ]
    )

    assert args.command == "concept-candidates-v10fix"
    assert args.workers == 25


def test_v10fixb_subcomposition_generation_produces_pairs_triples_and_subchains() -> None:
    contexts = _build_v10fixb_fixture_contexts(include_four_role_manifest=True)
    context = contexts[0]
    source_role_map = build_source_role_map_fixb(context.source_roles)
    manifest_items = [
        {
            "role_id": role["role_id"],
            "role_label_candidate": role["role_label_candidate"],
        }
        for role in source_role_map.values()
    ]
    assert len(manifest_items) >= 4

    items = []
    for family_id, record in sorted(context.source_neighborhoods.items()):
        role_info = source_role_map[family_id]
        items.append(
            {
                "record": record,
                "role_id": role_info["role_id"],
                "role_label": role_info["role_label_candidate"],
                **canonical_role_fingerprint(role_info["role_label_candidate"], record),
            }
        )
    candidates = generate_subcomposition_candidates(
        source_fold=context.heldout_family,
        heldout_family=context.heldout_family,
        manifest_family="srcA",
        items=items,
        max_role_count=5,
    )
    generators = {row["generator_type"] for row in candidates}
    role_counts = {len(row["canonical_role_fingerprint_hashes"]) for row in candidates}

    assert "role_pairs" in generators
    assert "role_triples" in generators
    assert "subchains" in generators
    assert 2 in role_counts
    assert 3 in role_counts


def test_v10fixb_canonical_role_fingerprint_matches_across_fold_local_ids() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    left = contexts[0].source_neighborhoods["sa_block"]
    right = contexts[1].source_neighborhoods["sb_block"]

    left_fp = canonical_role_fingerprint("blocker_candidate", left)
    right_fp = canonical_role_fingerprint("blocker_candidate", right)

    assert left_fp["canonical_role_fingerprint_hash"] == right_fp["canonical_role_fingerprint_hash"]


def test_v10fixb_concept_identity_ignores_support_but_separates_structure() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]
    source_role_map = build_source_role_map_fixb(context.source_roles)
    raw_rows, _ = run_v10fixb_raw_rows_for_context(context, source_role_map)
    merged, _ = merge_exact_candidates(raw_rows, min_games=1, min_manifest_families=1, min_role_count=2)
    same_rows = [row for row in merged if row["motif_type"] == "source_to_sink"][:2]
    different = next(row for row in merged if row["motif_type"] != same_rows[0]["motif_type"])

    adjusted = dict(same_rows[0])
    adjusted["source_manifest_families_present"] = ["x", "y", "z"]
    adjusted["source_games_present"] = ["g1", "g2", "g3", "g4"]

    same_merged, _ = merge_exact_candidates([same_rows[0], adjusted], min_games=1, min_manifest_families=1, min_role_count=2)
    assert same_rows[0]["concept_id"] == same_merged[0]["concept_id"]
    assert same_rows[0]["concept_id"] != different["concept_id"]


def test_v10fixb_premerge_collision_diagnostics_detect_collisions() -> None:
    rows = [
        {"concept_id": "m4-forced", "concept_signature_json": '{"a":1}'},
        {"concept_id": "m4-forced", "concept_signature_json": '{"a":2}'},
    ]
    collisions = build_collision_rows_fixb(rows)
    assert collisions[0]["collision_detected"] is True
    assert collisions[0]["distinct_signature_count"] == 2


def test_v10fixb_target_projection_uses_local_target_matches_not_aggregate_washout() -> None:
    contexts = _build_v10fixb_fixture_contexts(two_target_families=True)
    context = contexts[0]
    source_role_map = build_source_role_map_fixb(context.source_roles)
    raw_rows, _ = run_v10fixb_raw_rows_for_context(context, source_role_map)
    merged, _ = merge_exact_candidates(raw_rows, min_games=1, min_manifest_families=1, min_role_count=2)
    concept = merged[0]
    transfer_rows = [
        {"heldout_family": context.heldout_family, "target_family_id": "ta", "assigned_role_id": "x1", "surface_hardened_score": 0.15, "effect_residual_score": 0.18, "future_option_role_score": 0.78, "graph_position_role_score": 0.74},
        {"heldout_family": context.heldout_family, "target_family_id": "ta2", "assigned_role_id": "x2", "surface_hardened_score": 0.15, "effect_residual_score": -0.18, "future_option_role_score": 0.10, "graph_position_role_score": 0.10},
    ]
    projection = evaluate_target_projection_by_family(concept, context, transfer_rows)

    assert projection["target_best_match_score"] >= projection["target_full_mean_score"]
    assert projection["target_top3_mean_score"] >= projection["target_full_mean_score"]


def test_v10fixb_residual_profiles_use_shared_keys() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    record = contexts[0].source_neighborhoods["sa_block"]
    source_profile = build_effect_residual_profile_from_record(record)
    target_profile = build_effect_residual_profile_from_target_rows([{"effect_residual_score": 0.2}])

    assert tuple(sorted(source_profile)) == tuple(sorted(RESIDUAL_PROFILE_KEYS))
    assert set(source_profile) == set(target_profile)


def test_v10fixb_main_lift_uses_raw_role_baseline_not_discounted() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]
    source_role_map = build_source_role_map_fixb(context.source_roles)
    raw_rows, _ = run_v10fixb_raw_rows_for_context(context, source_role_map)
    merged, _ = merge_exact_candidates(raw_rows, min_games=1, min_manifest_families=1, min_role_count=2)
    concept = merged[0]
    projection = evaluate_target_projection_by_family(
        concept,
        context,
        [{"heldout_family": context.heldout_family, "target_family_id": "ta", "assigned_role_id": "x1", "surface_hardened_score": 0.12, "effect_residual_score": 0.22, "future_option_role_score": 0.80, "graph_position_role_score": 0.72}],
    )

    assert projection["best_individual_role_baseline_raw"] >= projection["best_individual_role_baseline_discounted"]
    assert projection["target_mean_concept_lift_vs_role_raw"] != projection["target_mean_concept_lift_vs_role_discounted"]


def test_v10fixb_fuzzy_grouping_merges_structurally_similar_concepts() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    source_role_map = build_source_role_map_fixb(contexts[0].source_roles)
    raw_a, _ = run_v10fixb_raw_rows_for_context(contexts[0], source_role_map)
    raw_b, _ = run_v10fixb_raw_rows_for_context(contexts[1], source_role_map)
    merged, _ = merge_exact_candidates(raw_a + raw_b, min_games=1, min_manifest_families=1, min_role_count=2)

    fuzzy_rows, _, mapping = fuzzy_group_candidates(merged, role_threshold=0.75, concept_threshold=0.70)

    assert len(fuzzy_rows) <= len(merged)
    assert len(set(mapping.values())) == len(fuzzy_rows)


def test_v10fixb_unknown_labels_can_survive_transferability() -> None:
    row = {
        "manifest_family_count": 3,
        "game_count": 4,
        "concept_stability_score": 0.6,
        "target_projection_coverage": 1.0,
        "target_mean_concept_lift_vs_role_raw": 0.1,
        "target_mean_concept_lift_vs_surface_raw": 0.08,
        "transfer_stability_score": 0.08,
        "concept_label_candidate": "unknown_concept_candidate",
    }
    assert row["concept_label_candidate"] == "unknown_concept_candidate"
    assert row["target_mean_concept_lift_vs_role_raw"] > 0
    assert row["target_mean_concept_lift_vs_surface_raw"] > 0


def test_v10fixb_worker_cap_uses_largest_task_and_shared_state(monkeypatch) -> None:
    import v6.concept_candidates_v10fixb as mod

    tasks = [("t1",), ("t2",), ("t3",), ("t4",)]
    sizes = {
        tasks[0]: 64 * 1024 * 1024,
        tasks[1]: 64 * 1024 * 1024,
        tasks[2]: 64 * 1024 * 1024,
        tasks[3]: 1024 * 1024 * 1024,
    }
    monkeypatch.setattr(mod, "estimate_task_payload_bytes", lambda task: sizes[task])
    monkeypatch.setattr(mod, "detect_available_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)

    workers = choose_effective_worker_count(tasks, requested_workers=8, shared_state_bytes=1024 * 1024 * 1024)

    assert workers == 1


def test_v09c_parallel_worker_cap_respects_memory_budget(monkeypatch) -> None:
    import v6.role_transfer_v09c as mod

    monkeypatch.setattr(mod, "detect_available_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)

    workers = choose_parallel_worker_count(requested_workers=25, task_count=16)

    assert workers == 1


def test_v10fixb_invalid_role_overlap_is_diagnostic_only() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]
    source_role_map = build_source_role_map_fixb(context.source_roles)
    raw_rows, _ = run_v10fixb_raw_rows_for_context(context, source_role_map)
    merged, _ = merge_exact_candidates(raw_rows, min_games=1, min_manifest_families=1, min_role_count=2)
    concept = merged[0]
    with_overlap = evaluate_target_projection_by_family(
        concept,
        context,
        [{"heldout_family": context.heldout_family, "target_family_id": "ta", "assigned_role_id": "role_block", "surface_hardened_score": 0.15, "effect_residual_score": 0.2, "future_option_role_score": 0.8, "graph_position_role_score": 0.7}],
    )
    no_overlap = evaluate_target_projection_by_family(
        concept,
        context,
        [{"heldout_family": context.heldout_family, "target_family_id": "ta", "assigned_role_id": "different_role", "surface_hardened_score": 0.15, "effect_residual_score": 0.2, "future_option_role_score": 0.8, "graph_position_role_score": 0.7}],
    )

    assert with_overlap["target_concept_prediction_score"] == no_overlap["target_concept_prediction_score"]
    assert with_overlap["role_id_overlap_diagnostic"] != no_overlap["role_id_overlap_diagnostic"]


def test_v10fixb_runs_deterministically_and_cli_accepts_options(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.concept_candidates_v10fixb as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    contexts = _build_v10fixb_fixture_contexts()
    monkeypatch.setattr(mod, "prepare_family_contexts", lambda config: contexts)

    one = run_concept_candidates_v10fixb(
        ConceptCandidatesV10FixBConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fixb1"),
            game_set_manifest=str(manifest),
            workers=1,
            min_games=2,
            min_manifest_families=2,
        )
    )
    many = run_concept_candidates_v10fixb(
        ConceptCandidatesV10FixBConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fixb2"),
            game_set_manifest=str(manifest),
            workers=2,
            min_games=2,
            min_manifest_families=2,
        )
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "concept-candidates-v10fix-b",
            "--m3-input-dir",
            "runs/v6/v08d_cd2_extended32_sourceclean",
            "--transfer-input-dir",
            "runs/v6/v09c_transfer_hardened_extended32",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--output-dir",
            "runs/v6/v10_m4_concepts_fixb_extended32",
            "--workers",
            "25",
        ]
    )

    assert one["validation"]["scientific_conclusion"] == many["validation"]["scientific_conclusion"]
    assert one["report"]["concept_id_collision_check_passed"] is True
    assert one["report"]["target_role_id_overlap_removed_from_main_score"] is True
    assert args.command == "concept-candidates-v10fix-b"
    assert args.workers == 25
    scores = pd.read_parquet(tmp_path / "v10fixb1" / "concept_transfer_scores_fixb.parquet")
    assert "target_best_match_score" in scores.columns
    assert "target_top3_mean_score" in scores.columns


def test_game_set_manifest_auto_resolves_extended32_from_available_games(tmp_path, monkeypatch) -> None:
    root = tmp_path / "runs" / "v6" / "game_sets"
    root.mkdir(parents=True, exist_ok=True)
    (root / "extended32_v08.json").write_text(
        json.dumps(
            {
                "name": "extended32_v08",
                "games": ["tt01", "tt02", "pb02", "pb03"],
                "families": {
                    "collection": ["tt01", "tt02"],
                    "push_crate": ["pb02", "pb03"],
                },
                "purpose": "test fixture",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    manifest = load_game_set_manifest(
        fallback_games=("pb02", "pb03", "tt01", "tt02")
    )

    assert manifest.name == "extended32_v08"
    assert len(manifest.families) == 2
    assert manifest.families["collection"] == ("tt01", "tt02")


def test_v10fixc_source_role_map_diagnostics_no_source_roles_uses_fallback() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]

    raw_rows, attrition, source_diag, _ = discover_source_only_candidates_fixc(
        context=context,
        source_role_map={},
        source_manifest_family_map={},
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
        config=ConceptCandidatesV10FixCConfig(min_games=1, min_manifest_families=1),
    )

    assert source_diag["failure_mode"] == "no_source_roles"
    assert raw_rows
    assert attrition["fallback_candidate_count"] > 0
    assert {row["candidate_source"] for row in raw_rows} == {"fallback_neighborhood"}


def test_v10fixc_role_map_mismatch_still_generates_fallback_candidates() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]
    source_role_map = {
        "mismatch_family": {
            "role_id": "role_block",
            "role_label_candidate": "blocker_candidate",
            "all_features": {},
        }
    }

    raw_rows, _, source_diag, _ = discover_source_only_candidates_fixc(
        context=context,
        source_role_map=source_role_map,
        source_manifest_family_map={},
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
        config=ConceptCandidatesV10FixCConfig(min_games=1, min_manifest_families=1),
    )

    assert source_diag["failure_mode"] == "source_role_map_family_id_mismatch"
    assert raw_rows
    assert source_diag["fallback_raw_candidate_count"] > 0


def test_v10fixc_manifest_fallback_uses_record_and_unknown_bucket() -> None:
    from types import SimpleNamespace

    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]
    source_role_map = build_source_role_map_fixb(context.source_roles)
    record = context.source_neighborhoods["sa_block"]
    unknown_record = SimpleNamespace(**record.__dict__)
    unknown_record.game_family_ids = ()
    unknown_record.game_ids = ("ux1",)
    mixed_context = context.__class__(
        heldout_family=context.heldout_family,
        heldout_games=context.heldout_games,
        source_neighborhoods={**context.source_neighborhoods, "unknown_one": unknown_record},
        source_roles=context.source_roles,
        source_no_label_roles=context.source_no_label_roles,
        target_families=context.target_families,
        full_neighborhoods=context.full_neighborhoods,
        full_no_label_neighborhoods=context.full_no_label_neighborhoods,
        graph_source_used=context.graph_source_used,
        graph_edge_coverage=context.graph_edge_coverage,
    )

    _, _, source_diag, manifest_rows = discover_source_only_candidates_fixc(
        context=mixed_context,
        source_role_map=source_role_map,
        source_manifest_family_map={},
        game_to_manifest_family={},
        config=ConceptCandidatesV10FixCConfig(min_games=1, min_manifest_families=1),
    )

    assert source_diag["manifest_groups_created"] > 0
    assert any(row["final_manifest_resolution"] == ["unknown_manifest_family"] for row in manifest_rows)


def test_v10fixc_mixed_candidates_generated() -> None:
    contexts = _build_v10fixb_fixture_contexts(include_four_role_manifest=True)
    context = contexts[0]
    source_role_map = build_source_role_map_fixb(context.source_roles)
    source_role_map.pop("sa_link")

    raw_rows, attrition, _, _ = discover_source_only_candidates_fixc(
        context=context,
        source_role_map=source_role_map,
        source_manifest_family_map={},
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
        config=ConceptCandidatesV10FixCConfig(min_games=1, min_manifest_families=1),
    )

    assert raw_rows
    assert attrition["mixed_candidates_generated"] > 0
    assert "mixed" in {row["candidate_source"] for row in raw_rows}


def test_v10fixc_fallback_candidate_is_diagnostic_only() -> None:
    import v6.concept_candidates_v10fixc as mod

    candidate = {
        "candidate_source": "fallback",
        "manifest_resolution_status": "resolved",
    }

    assert mod.classify_candidate_evidence_lane(candidate) == "diagnostic_fallback"
    assert mod.is_candidate_concept_eligible(candidate) is False


def test_v10fixc_mixed_candidate_is_diagnostic_only() -> None:
    import v6.concept_candidates_v10fixc as mod

    candidate = {
        "candidate_source": "mixed",
        "manifest_resolution_status": "resolved",
    }

    assert mod.classify_candidate_evidence_lane(candidate) == "diagnostic_mixed"
    assert mod.is_candidate_concept_eligible(candidate) is False


def test_v10fixc_unknown_manifest_candidate_is_diagnostic_only() -> None:
    import v6.concept_candidates_v10fixc as mod

    candidate = {
        "candidate_source": "stable",
        "manifest_resolution_status": "resolved",
        "manifest_family_ids": ["unknown_manifest_family"],
    }

    assert mod.classify_candidate_evidence_lane(candidate) == "diagnostic_unknown_manifest"
    assert mod.is_candidate_concept_eligible(candidate) is False


def test_v10fixc_unresolved_manifest_candidate_is_diagnostic_only() -> None:
    import v6.concept_candidates_v10fixc as mod

    candidate = {
        "candidate_source": "stable",
        "manifest_resolution_status": "failed",
    }

    assert mod.classify_candidate_evidence_lane(candidate) == "diagnostic_unresolved_manifest"
    assert mod.is_candidate_concept_eligible(candidate) is False


def test_v10fixc_stable_resolved_candidate_is_eligible() -> None:
    import v6.concept_candidates_v10fixc as mod

    candidate = {
        "candidate_source": "stable",
        "manifest_resolution_status": "resolved",
        "manifest_family_ids": ["family_a", "family_b"],
    }

    assert mod.classify_candidate_evidence_lane(candidate) == "eligible_stable"
    assert mod.is_candidate_concept_eligible(candidate) is True


def test_v10fixc_policy_excludes_diagnostic_only_candidates_from_accepted_counts() -> None:
    import v6.concept_candidates_v10fixc as mod

    candidates = [
        {"concept_id": "stable", "candidate_source": "stable", "manifest_resolution_status": "resolved", "manifest_family_ids": ["family_a"]},
        {"concept_id": "fallback", "candidate_source": "fallback", "manifest_resolution_status": "resolved"},
        {"concept_id": "mixed", "candidate_source": "mixed", "manifest_resolution_status": "resolved"},
        {"concept_id": "unknown", "candidate_source": "stable", "manifest_resolution_status": "resolved", "manifest_family_ids": ["unknown_manifest_family"]},
    ]

    policy = mod.apply_candidate_evidence_policy(
        candidates,
        stable_predicate=lambda candidate: True,
        transferable_predicate=lambda candidate: True,
    )

    assert len(policy["stable_concepts"]) == 1
    assert len(policy["transferable_concepts"]) == 1
    assert len(policy["diagnostic_only_candidates"]) == 3
    assert len(policy["eligible_candidates"]) == 1


def test_v10fixc_projection_rows_do_not_embed_nested_target_rows() -> None:
    contexts = _build_v10fixb_fixture_contexts(two_target_families=True)
    context = contexts[0]
    source_role_map = build_source_role_map_fixb(context.source_roles)
    raw_rows, _, _, _ = discover_source_only_candidates_fixc(
        context=context,
        source_role_map=source_role_map,
        source_manifest_family_map={},
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
        config=ConceptCandidatesV10FixCConfig(min_games=1, min_manifest_families=1),
    )
    merged, _ = merge_exact_candidates(raw_rows, min_games=1, min_manifest_families=1, min_role_count=2)
    projection, detail_rows = evaluate_target_projection_by_family_fixc(
        merged[0],
        context,
        [{"heldout_family": context.heldout_family, "target_family_id": "ta", "assigned_role_id": "role_block", "surface_hardened_score": 0.15, "effect_residual_score": 0.2, "future_option_role_score": 0.8, "graph_position_role_score": 0.7}],
    )

    assert "target_family_rows" not in projection
    assert detail_rows


def test_v10fixc_runs_streaming_and_resumes_from_shards(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.concept_candidates_v10fixc as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    contexts = _build_v10fixb_fixture_contexts()
    monkeypatch.setattr(mod, "prepare_family_context_stream", lambda config: iter(contexts))

    first = run_concept_candidates_v10fixc(
        ConceptCandidatesV10FixCConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fixc1"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
            write_shards=True,
        )
    )
    shard_dir = tmp_path / "v10fixc1" / "shards"
    assert (shard_dir / "raw_concept_candidates_premerge__holdA.parquet").exists()

    monkeypatch.setattr(mod, "evaluate_family_fixc", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should resume from shards")))
    resumed = run_concept_candidates_v10fixc(
        ConceptCandidatesV10FixCConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fixc1"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
            write_shards=True,
            resume_from_shards=True,
        )
    )

    assert first["validation"]["scientific_conclusion"] == resumed["validation"]["scientific_conclusion"]
    scores = pd.read_parquet(tmp_path / "v10fixc1" / "concept_transfer_scores_fixc.parquet")
    details = pd.read_parquet(tmp_path / "v10fixc1" / "concept_target_family_scores_fixc.parquet")
    assert "target_family_rows" not in scores.columns
    assert "target_family_id" in details.columns


def test_v10fixc_completed_run_validator_accepts_fixture_run(tmp_path, monkeypatch) -> None:
    import v6.concept_candidates_v10fixc as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    contexts = _build_v10fixb_fixture_contexts()
    monkeypatch.setattr(mod, "prepare_family_context_stream", lambda config: iter(contexts))

    run_concept_candidates_v10fixc(
        ConceptCandidatesV10FixCConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fixc_validate"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
            write_shards=True,
        )
    )
    validation = validate_completed_fixc_run(tmp_path / "v10fixc_validate")

    assert validation["valid"] is True
    assert validation["checks"]["transfer_rows_no_nested_target_family_payloads"] is True


def test_v10fixd_single_family_context_builder_omits_full_neighborhoods(monkeypatch) -> None:
    from types import SimpleNamespace

    import v6.role_transfer_v09c as mod

    family = SimpleNamespace(family_id="srcA", games_present=("g1",), support_count=2)
    heldout = SimpleNamespace(family_id="holdA", games_present=("h1",), support_count=2)
    graph_diag = SimpleNamespace(graph_source_used="hybrid", graph_edge_coverage=1.0)

    monkeypatch.setattr(mod, "load_m2_families", lambda path: [family, heldout])
    monkeypatch.setattr(
        mod,
        "load_game_set_manifest",
        lambda **kwargs: SimpleNamespace(games=("g1", "h1"), families={"srcA": ("g1",), "holdA": ("h1",)}),
    )
    monkeypatch.setattr(mod, "load_m1_support", lambda path: {})
    monkeypatch.setattr(mod, "load_episode_summaries", lambda path: {})
    monkeypatch.setattr(mod, "load_m2_graph_edges", lambda path: {})
    monkeypatch.setattr(mod, "build_game_family_map", lambda game_set, games: {"g1": "srcA", "h1": "holdA"})
    monkeypatch.setattr(
        mod,
        "build_discriminative_neighborhoods",
        lambda families, *args, **kwargs: ({f"{families[0].family_id}_n": SimpleNamespace(family_id=f"{families[0].family_id}_n")}, graph_diag),
    )
    monkeypatch.setattr(mod, "build_source_only_roles", lambda families, neighborhoods: {"role": {"member_family_ids": tuple(neighborhoods)}})

    context = build_single_family_context(
        RoleTransferV09cConfig(m2_input_dir="unused", m1_input_dir="unused", game_set_manifest="unused"),
        "holdA",
    )

    assert isinstance(context, SingleFamilyContext)
    assert hasattr(context, "source_neighborhoods")
    assert hasattr(context, "target_neighborhoods")
    assert not hasattr(context, "full_neighborhoods")
    assert context.heldout_family == "holdA"


def test_v10fixd_diagnostics_shards_written_before_candidate_generation_exception(tmp_path, monkeypatch) -> None:
    import v6.concept_candidates_v10fixd as mod

    contexts = _build_v10fixb_fixture_contexts()
    fixture = contexts[0]
    single = SingleFamilyContext(
        heldout_family=fixture.heldout_family,
        heldout_games=fixture.heldout_games,
        source_neighborhoods=fixture.source_neighborhoods,
        source_roles=fixture.source_roles,
        target_families=fixture.target_families,
        target_neighborhoods=fixture.full_neighborhoods,
        graph_source_used=fixture.graph_source_used,
        graph_edge_coverage=fixture.graph_edge_coverage,
    )
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single)
    monkeypatch.setattr(mod, "generate_candidate_chunks_for_group", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    try:
        run_single_family_fixd(
            heldout_family="holdA",
            config=ConceptCandidatesV10FixDConfig(output_dir=str(tmp_path), game_set_manifest=str(tmp_path / "manifest.json")),
            source_manifest_family_map={},
            transfer_rows=[],
            game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
            shards_dir=shards_dir,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected candidate generation failure")

    paths = family_fixd_paths(shards_dir, "holdA")
    assert paths["source_diag"].exists()
    assert paths["manifest_diag"].exists()
    assert paths["memory_stage0"].exists()


def test_v10fixd_empty_source_roles_generates_fallback_manifest_groups() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    fixture = contexts[0]
    single = SingleFamilyContext(
        heldout_family=fixture.heldout_family,
        heldout_games=fixture.heldout_games,
        source_neighborhoods=fixture.source_neighborhoods,
        source_roles={},
        target_families=fixture.target_families,
        target_neighborhoods=fixture.full_neighborhoods,
        graph_source_used=fixture.graph_source_used,
        graph_edge_coverage=fixture.graph_edge_coverage,
    )

    _, source_diag, _, manifest_groups = collect_source_items_and_manifest_groups(
        context=single,
        source_role_map={},
        source_manifest_family_map={},
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
    )

    assert source_diag["failure_mode_if_zero_structures"] == "no_source_roles"
    assert source_diag["fallback_items_available"] > 0
    assert manifest_groups
    assert all(item["candidate_source"] == "fallback_neighborhood" for items in manifest_groups.values() for item in items)


def test_v10fixd_role_map_mismatch_reports_failure_mode_but_fallback_runs() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    fixture = contexts[0]
    single = SingleFamilyContext(
        heldout_family=fixture.heldout_family,
        heldout_games=fixture.heldout_games,
        source_neighborhoods=fixture.source_neighborhoods,
        source_roles=fixture.source_roles,
        target_families=fixture.target_families,
        target_neighborhoods=fixture.full_neighborhoods,
        graph_source_used=fixture.graph_source_used,
        graph_edge_coverage=fixture.graph_edge_coverage,
    )

    _, source_diag, _, manifest_groups = collect_source_items_and_manifest_groups(
        context=single,
        source_role_map={"mismatch": {"role_id": "r", "role_label_candidate": "x", "all_features": {}}},
        source_manifest_family_map={},
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
    )

    assert source_diag["failure_mode_if_zero_structures"] == "source_role_map_family_id_mismatch"
    assert manifest_groups
    assert any(item["candidate_source"] == "fallback_neighborhood" for items in manifest_groups.values() for item in items)


def test_v10fixd_unknown_manifest_records_are_grouped_not_dropped() -> None:
    from types import SimpleNamespace

    contexts = _build_v10fixb_fixture_contexts()
    fixture = contexts[0]
    unknown_record = SimpleNamespace(**fixture.source_neighborhoods["sa_block"].__dict__)
    unknown_record.family_id = "unknown_one"
    unknown_record.game_ids = ("ux1",)
    unknown_record.game_family_ids = ()
    source_neighborhoods = {**fixture.source_neighborhoods, unknown_record.family_id: unknown_record}
    source_roles = dict(fixture.source_roles)
    single = SingleFamilyContext(
        heldout_family=fixture.heldout_family,
        heldout_games=fixture.heldout_games,
        source_neighborhoods=source_neighborhoods,
        source_roles=source_roles,
        target_families=fixture.target_families,
        target_neighborhoods=fixture.full_neighborhoods,
        graph_source_used=fixture.graph_source_used,
        graph_edge_coverage=fixture.graph_edge_coverage,
    )

    _, _, manifest_rows, manifest_groups = collect_source_items_and_manifest_groups(
        context=single,
        source_role_map=build_source_role_map_fixb(source_roles),
        source_manifest_family_map={},
        game_to_manifest_family={},
    )

    assert "unknown_manifest_family" in manifest_groups
    assert any(row["family_id"] == "unknown_one" and row["final_manifest_resolution"] == ["unknown_manifest_family"] for row in manifest_rows)


def test_v10fixd_large_manifest_group_writes_multiple_raw_candidate_part_files(tmp_path, monkeypatch) -> None:
    import v6.concept_candidates_v10fixd as mod

    contexts = _build_v10fixb_fixture_contexts()
    fixture = contexts[0]
    single = SingleFamilyContext(
        heldout_family=fixture.heldout_family,
        heldout_games=fixture.heldout_games,
        source_neighborhoods=fixture.source_neighborhoods,
        source_roles=fixture.source_roles,
        target_families=fixture.target_families,
        target_neighborhoods=fixture.full_neighborhoods,
        graph_source_used=fixture.graph_source_used,
        graph_edge_coverage=fixture.graph_edge_coverage,
    )
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single)

    chunk1 = [{"concept_id": "m4-a", "candidate_source": "stable_role", "canonical_role_fingerprint_hashes": ["r1"], "source_fold": "holdA", "heldout_family": "holdA", "fold_local_role_ids": ["ra"]}]
    chunk2 = [{"concept_id": "m4-b", "candidate_source": "fallback_neighborhood", "canonical_role_fingerprint_hashes": ["r2"], "source_fold": "holdA", "heldout_family": "holdA", "fold_local_role_ids": ["rb"]}]
    monkeypatch.setattr(mod, "generate_candidate_chunks_for_group", lambda **kwargs: iter([chunk1, chunk2]))
    monkeypatch.setattr(
        mod,
        "score_candidate_chunk",
        lambda raw_chunk, context, target_rows: (
            [
                {
                    "heldout_family": "holdA",
                    "concept_id": raw_chunk[0]["concept_id"],
                    "projection_used": True,
                    "target_concept_prediction_score": 0.5,
                }
            ],
            [{"heldout_family": "holdA", "concept_id": raw_chunk[0]["concept_id"], "target_family_id": "ta", "target_family_score": 0.5}],
            [],
            [{"heldout_family": "holdA", "concept_id": raw_chunk[0]["concept_id"], "canonical_role_fingerprint_hash": raw_chunk[0]["canonical_role_fingerprint_hashes"][0]}],
        ),
    )

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    run_single_family_fixd(
        heldout_family="holdA",
        config=ConceptCandidatesV10FixDConfig(output_dir=str(tmp_path), game_set_manifest=str(tmp_path / "manifest.json")),
        source_manifest_family_map={},
        transfer_rows=[],
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
        shards_dir=shards_dir,
    )

    assert (shards_dir / "raw_concept_candidates_premerge__holdA__part-0001.parquet").exists()
    assert (shards_dir / "raw_concept_candidates_premerge__holdA__part-0002.parquet").exists()


def test_v10fixd_transfer_rows_do_not_embed_nested_target_payloads(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.concept_candidates_v10fixd as mod

    contexts = _build_v10fixb_fixture_contexts()
    fixture = contexts[0]
    single = SingleFamilyContext(
        heldout_family=fixture.heldout_family,
        heldout_games=fixture.heldout_games,
        source_neighborhoods=fixture.source_neighborhoods,
        source_roles=fixture.source_roles,
        target_families=fixture.target_families,
        target_neighborhoods=fixture.full_neighborhoods,
        graph_source_used=fixture.graph_source_used,
        graph_edge_coverage=fixture.graph_edge_coverage,
    )
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single)
    monkeypatch.setattr(
        mod,
        "generate_candidate_chunks_for_group",
        lambda **kwargs: iter(
            [[{"concept_id": "m4-a", "candidate_source": "stable_role", "canonical_role_fingerprint_hashes": ["r1"], "source_fold": "holdA", "heldout_family": "holdA", "fold_local_role_ids": ["ra"]}]]
        ),
    )
    monkeypatch.setattr(
        mod,
        "score_candidate_chunk",
        lambda raw_chunk, context, target_rows: (
            [{"heldout_family": "holdA", "concept_id": "m4-a", "projection_used": True, "target_concept_prediction_score": 0.5}],
            [{"heldout_family": "holdA", "concept_id": "m4-a", "target_family_id": "ta", "target_family_score": 0.5}],
            [],
            [{"heldout_family": "holdA", "concept_id": "m4-a", "canonical_role_fingerprint_hash": "r1"}],
        ),
    )

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    run_single_family_fixd(
        heldout_family="holdA",
        config=ConceptCandidatesV10FixDConfig(output_dir=str(tmp_path), game_set_manifest=str(tmp_path / "manifest.json")),
        source_manifest_family_map={},
        transfer_rows=[],
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
        shards_dir=shards_dir,
    )

    transfer = pd.read_parquet(shards_dir / "concept_transfer_scores__holdA__part-0001.parquet")
    assert "target_family_rows" not in transfer.columns
    assert "record" not in transfer.columns


def test_v10fixd_resume_from_shards_skips_completed_family(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.concept_candidates_v10fixd as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    output_dir = tmp_path / "v10fixd_resume"
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True)
    paths = family_fixd_paths(shards_dir, "holdA")
    paths["complete"].write_text(json.dumps({"heldout_family": "holdA"}), encoding="utf-8")
    for path, rows in (
        (paths["summary"], [{"heldout_family": "holdA", "positive_concept_lift": 0}]),
        (paths["source_diag"], [{"heldout_family": "holdA", "failure_mode_if_zero_structures": "no_source_roles"}]),
        (paths["manifest_diag"], [{"heldout_family": "holdA", "family_id": "x", "final_manifest_resolution": ["unknown_manifest_family"]}]),
        (paths["attrition"], [{"heldout_family": "holdA", "raw_candidate_count_premerge": 0}]),
        (paths["memory_all"], [{"heldout_family": "holdA", "stage": "end", "rss_mb": 1.0}]),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA"])
    monkeypatch.setattr(mod, "load_source_manifest_family_map", lambda path: {})
    monkeypatch.setattr(mod, "run_single_family_fixd", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should skip completed shard family")))
    monkeypatch.setattr(
        mod,
        "finalize_fixd_run",
        lambda **kwargs: {"report": {"corrected_concept_candidate_count": 0, "target_family_score_count": 0}, "validation": {"scientific_conclusion": "m4_concepts_fixd_pipeline_not_diagnostic"}},
    )

    payload = run_concept_candidates_v10fixd(
        ConceptCandidatesV10FixDConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(output_dir),
            game_set_manifest=str(manifest),
            resume_from_shards=True,
        )
    )

    assert payload["validation"]["scientific_conclusion"] == "m4_concepts_fixd_pipeline_not_diagnostic"


def test_v10fixd_memory_diagnostics_include_expected_stages(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.concept_candidates_v10fixd as mod

    contexts = _build_v10fixb_fixture_contexts()
    fixture = contexts[0]
    single = SingleFamilyContext(
        heldout_family=fixture.heldout_family,
        heldout_games=fixture.heldout_games,
        source_neighborhoods=fixture.source_neighborhoods,
        source_roles=fixture.source_roles,
        target_families=fixture.target_families,
        target_neighborhoods=fixture.full_neighborhoods,
        graph_source_used=fixture.graph_source_used,
        graph_edge_coverage=fixture.graph_edge_coverage,
    )
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single)
    monkeypatch.setattr(
        mod,
        "generate_candidate_chunks_for_group",
        lambda **kwargs: iter(
            [[{"concept_id": "m4-a", "candidate_source": "stable_role", "canonical_role_fingerprint_hashes": ["r1"], "source_fold": "holdA", "heldout_family": "holdA", "fold_local_role_ids": ["ra"]}]]
        ),
    )
    monkeypatch.setattr(
        mod,
        "score_candidate_chunk",
        lambda raw_chunk, context, target_rows: (
            [{"heldout_family": "holdA", "concept_id": "m4-a", "projection_used": True, "target_concept_prediction_score": 0.5}],
            [{"heldout_family": "holdA", "concept_id": "m4-a", "target_family_id": "ta", "target_family_score": 0.5}],
            [],
            [{"heldout_family": "holdA", "concept_id": "m4-a", "canonical_role_fingerprint_hash": "r1"}],
        ),
    )

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    run_single_family_fixd(
        heldout_family="holdA",
        config=ConceptCandidatesV10FixDConfig(output_dir=str(tmp_path), game_set_manifest=str(tmp_path / "manifest.json")),
        source_manifest_family_map={},
        transfer_rows=[],
        game_to_manifest_family={"g1": "srcA", "g2": "srcA", "g3": "srcB", "g4": "srcB"},
        shards_dir=shards_dir,
    )

    memory = pd.read_parquet(shards_dir / "memory_diagnostics__holdA.parquet")
    stages = set(memory["stage"])
    assert {"start", "after_load_single_family_context", "after_source_role_map", "after_manifest_groups", "after_diagnostics_write", "after_candidate_chunk", "after_projection_chunk", "after_gc", "end"} <= stages
    assert (memory["rss_mb"] >= 0).all()


def test_v10fixd_runs_deterministically_on_small_fixture(tmp_path, monkeypatch) -> None:
    import v6.concept_candidates_v10fixd as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    contexts = _build_v10fixb_fixture_contexts()
    single_contexts = {
        context.heldout_family: SingleFamilyContext(
            heldout_family=context.heldout_family,
            heldout_games=context.heldout_games,
            source_neighborhoods=context.source_neighborhoods,
            source_roles=context.source_roles,
            target_families=context.target_families,
            target_neighborhoods=context.full_neighborhoods,
            graph_source_used=context.graph_source_used,
            graph_edge_coverage=context.graph_edge_coverage,
        )
        for context in contexts
    }
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single_contexts[heldout_family])
    monkeypatch.setattr(mod, "load_source_manifest_family_map", lambda path: {})
    monkeypatch.setattr(mod, "choose_fixd_worker_count", lambda requested_workers, task_count: 1)

    one = run_concept_candidates_v10fixd(
        ConceptCandidatesV10FixDConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fixd1"),
            game_set_manifest=str(manifest),
            workers=1,
            min_games=1,
            min_manifest_families=1,
        )
    )
    many = run_concept_candidates_v10fixd(
        ConceptCandidatesV10FixDConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10fixd2"),
            game_set_manifest=str(manifest),
            workers=2,
            min_games=1,
            min_manifest_families=1,
        )
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "concept-candidates-v10fix-d",
            "--m3-input-dir",
            "runs/v6/v08d_cd2_extended32_sourceclean",
            "--transfer-input-dir",
            "runs/v6/v09c_transfer_hardened_extended32",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--previous-v09b-dir",
            "runs/custom_v09b",
            "--output-dir",
            "runs/v6/v10_m4_concepts_fixd_extended32",
            "--workers",
            "1",
            "--streaming",
            "true",
            "--memory-safe",
            "true",
        ]
    )

    assert one["validation"]["scientific_conclusion"] == many["validation"]["scientific_conclusion"]
    assert one["report"]["target_family_score_count"] == many["report"]["target_family_score_count"]
    assert args.command == "concept-candidates-v10fix-d"
    assert args.previous_v09b_dir == "runs/custom_v09b"


def _build_v10e_single_contexts(include_four_role_manifest: bool = False, two_target_families: bool = False):
    contexts = _build_v10fixb_fixture_contexts(
        include_four_role_manifest=include_four_role_manifest,
        two_target_families=two_target_families,
    )
    return {
        context.heldout_family: SingleFamilyContext(
            heldout_family=context.heldout_family,
            heldout_games=context.heldout_games,
            source_neighborhoods=context.source_neighborhoods,
            source_roles=context.source_roles,
            target_families=context.target_families,
            target_neighborhoods=context.full_neighborhoods,
            graph_source_used=context.graph_source_used,
            graph_edge_coverage=context.graph_edge_coverage,
        )
        for context in contexts
    }


def test_v10e_source_clean_split_excludes_heldout_games(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.m4_role_concepts_v10e as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    single_contexts = _build_v10e_single_contexts()
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single_contexts[heldout_family])

    run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e1"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
        )
    )
    concepts = pd.read_parquet(tmp_path / "v10e1" / "role_based_m4_concepts.parquet").to_dict(orient="records")

    assert concepts
    assert all("h1" not in row["source_games_present"] and "h2" not in row["source_games_present"] for row in concepts)


def test_v10e_manifest_labels_not_used_for_grouping() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]
    role_map = build_source_role_map_fixb(context.source_roles)
    item_a = {
        "source_fold": "holdA",
        "family_id": "sa_block",
        "record": context.source_neighborhoods["sa_block"],
        "role_id": "role_block",
        "role_label": "blocker_candidate",
        **canonical_role_fingerprint("blocker_candidate", context.source_neighborhoods["sa_block"]),
        "source_manifest_families": ["srcA"],
    }
    item_b = {
        "source_fold": "holdA",
        "family_id": "sb_move",
        "record": context.source_neighborhoods["sb_move"],
        "role_id": "role_move",
        "role_label": "movement_controller_candidate",
        **canonical_role_fingerprint("movement_controller_candidate", context.source_neighborhoods["sb_move"]),
        "source_manifest_families": ["srcB"],
    }
    left = build_role_based_candidate_row(source_fold="holdA", heldout_family="holdA", local_candidate_id="a", items=[item_a, item_b])
    item_a["source_manifest_families"] = ["altA"]
    item_b["source_manifest_families"] = ["altB"]
    right = build_role_based_candidate_row(source_fold="holdA", heldout_family="holdA", local_candidate_id="b", items=[item_a, item_b])

    assert role_map["sa_block"]["role_id"] == "role_block"
    assert left["concept_id"] == right["concept_id"]


def test_v10e_fallback_candidates_do_not_affect_conclusion(tmp_path, monkeypatch) -> None:
    import v6.m4_role_concepts_v10e as mod

    fallback_only = {}
    for heldout_family, context in _build_v10e_single_contexts().items():
        fallback_only[heldout_family] = SingleFamilyContext(
            heldout_family=context.heldout_family,
            heldout_games=context.heldout_games,
            source_neighborhoods=context.source_neighborhoods,
            source_roles={},
            target_families=context.target_families,
            target_neighborhoods=context.target_neighborhoods,
            graph_source_used=context.graph_source_used,
            graph_edge_coverage=context.graph_edge_coverage,
        )
    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: fallback_only[heldout_family])
    monkeypatch.setattr(
        mod,
        "score_fallback_rows",
        lambda context, fallback_rows, target_rows: [
            {
                "heldout_family": context.heldout_family,
                "candidate_source": "fallback_diagnostic_only",
                "concept_id": row["concept_id"],
                "target_family_id": context.target_families[0].family_id,
                "target_family_score": 0.9,
                "best_individual_role_baseline_raw": 0.1,
                "best_surface_raw_baseline": 0.1,
            }
            for row in fallback_rows
        ],
    )

    payload = run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e_fallback"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
        )
    )

    assert payload["report"]["fallback_diagnostic_candidate_count"] > 0
    assert payload["validation"]["scientific_conclusion"] == "m4_fallback_signal_only_m3_bottleneck"


def test_v10e_target_assigned_role_id_does_not_affect_score() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]
    items = []
    for role_id, role in sorted(context.source_roles.items()):
        record = context.source_neighborhoods["sa_block" if role_id == "role_block" else "sa_move"]
        items.append(
            {
                "source_fold": "holdA",
                "family_id": record.family_id,
                "record": record,
                "role_id": role_id,
                "role_label": role["role_label_candidate"],
                **canonical_role_fingerprint(role["role_label_candidate"], record),
                "source_manifest_families": ["srcA"],
            }
        )
    concept = build_role_based_candidate_row(source_fold="holdA", heldout_family="holdA", local_candidate_id="a", items=items)
    target = context.full_neighborhoods["ta"]
    with_overlap = score_role_based_concept_against_target_family(
        concept,
        context,
        "ta",
        target,
        [{"assigned_role_id": "role_block", "target_family_id": "ta", "surface_hardened_score": 0.2, "effect_residual_score": 0.2}],
    )
    no_overlap = score_role_based_concept_against_target_family(
        concept,
        context,
        "ta",
        target,
        [{"assigned_role_id": "different_role", "target_family_id": "ta", "surface_hardened_score": 0.2, "effect_residual_score": 0.2}],
    )

    assert with_overlap["target_family_score"] == no_overlap["target_family_score"]


def test_v10e_compression_gain_is_required() -> None:
    row = {
        "candidate_source": "stable_role",
        "canonical_role_fingerprint_hashes": ["a", "b"],
        "target_mean_concept_lift_vs_role_raw": 0.1,
        "target_mean_concept_lift_vs_role_bag": 0.1,
        "target_mean_concept_lift_vs_surface_raw": 0.1,
        "target_mean_future_option_prediction_lift": 0.1,
        "mean_compression_gain": 0.0,
        "positive_lift_family_count": 2,
        "explained_m2_family_count": 2,
    }

    assert is_transferable_role_based_concept(row) is False


def test_v10e_graph_no_label_baseline_is_source_clean() -> None:
    import v6.m4_role_concepts_v10e as mod

    contexts = _build_v10fixb_fixture_contexts()
    context = contexts[0]
    target = context.full_neighborhoods["ta"]
    source_only = score_role_based_concept_against_target_family(
        build_role_based_candidate_row(
            source_fold="holdA",
            heldout_family="holdA",
            local_candidate_id="a",
            items=[
                {
                    "source_fold": "holdA",
                    "family_id": "sa_block",
                    "record": context.source_neighborhoods["sa_block"],
                    "role_id": "role_block",
                    "role_label": "blocker_candidate",
                    **canonical_role_fingerprint("blocker_candidate", context.source_neighborhoods["sa_block"]),
                    "source_manifest_families": ["srcA"],
                },
                {
                    "source_fold": "holdA",
                    "family_id": "sb_move",
                    "record": context.source_neighborhoods["sb_move"],
                    "role_id": "role_move",
                    "role_label": "movement_controller_candidate",
                    **canonical_role_fingerprint("movement_controller_candidate", context.source_neighborhoods["sb_move"]),
                    "source_manifest_families": ["srcB"],
                },
            ],
        ),
        context,
        "ta",
        target,
        [],
    )
    expected = max(
        mod.cosine_similarity(graph_position_features(target), graph_position_features(record))
        for record in context.source_neighborhoods.values()
    )

    assert abs(source_only["graph_no_label_baseline"] - expected) < 1e-9


def test_v10e_weak_gate_requires_surface_compression_future_and_family_count() -> None:
    row = {
        "candidate_source": "stable_role",
        "canonical_role_fingerprint_hashes": ["a", "b"],
        "target_mean_concept_lift_vs_role_raw": 0.1,
        "target_mean_concept_lift_vs_role_bag": 0.1,
        "target_mean_concept_lift_vs_surface_raw": 0.0,
        "target_mean_future_option_prediction_lift": 0.1,
        "mean_compression_gain": 0.1,
        "positive_lift_family_count": 6,
        "explained_m2_family_count": 2,
    }

    assert is_transferable_role_based_concept(row) is False


def test_v10e_report_cannot_claim_weak_with_one_transferable(tmp_path, monkeypatch) -> None:
    import v6.m4_role_concepts_v10e as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    single_contexts = _build_v10e_single_contexts()
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single_contexts[heldout_family])
    original = mod.evaluate_role_based_projection_by_family

    def one_transferable(concept, context, target_rows):
        projection, rows = original(concept, context, target_rows)
        if concept["local_candidate_id"].endswith("0001"):
            projection["target_mean_concept_lift_vs_role_raw"] = 0.1
            projection["target_mean_concept_lift_vs_role_bag"] = 0.1
            projection["target_mean_concept_lift_vs_surface_raw"] = 0.1
            projection["target_mean_future_option_prediction_lift"] = 0.1
            projection["mean_compression_gain"] = 0.1
            projection["positive_lift_family_count"] = 6
            projection["explained_m2_family_count"] = 2
        else:
            projection["target_mean_concept_lift_vs_role_raw"] = -0.1
            projection["target_mean_concept_lift_vs_role_bag"] = -0.1
            projection["target_mean_concept_lift_vs_surface_raw"] = -0.1
            projection["target_mean_future_option_prediction_lift"] = -0.1
            projection["mean_compression_gain"] = -0.1
            projection["positive_lift_family_count"] = 0
            projection["explained_m2_family_count"] = 0
        projection["passes_role_based_gate"] = mod.is_transferable_role_based_concept({**concept, **projection})
        return projection, rows

    monkeypatch.setattr(mod, "evaluate_role_based_projection_by_family", one_transferable)

    payload = run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e_one_transferable"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
        )
    )

    assert payload["validation"]["scientific_conclusion"] == "m4_role_based_not_established"


def test_v10e_mixed_candidates_do_not_affect_role_based_proof() -> None:
    row = {
        "candidate_source": "mixed",
        "canonical_role_fingerprint_hashes": ["a", "b"],
        "target_mean_concept_lift_vs_role_raw": 0.1,
        "target_mean_concept_lift_vs_role_bag": 0.1,
        "target_mean_concept_lift_vs_surface_raw": 0.1,
        "target_mean_future_option_prediction_lift": 0.1,
        "mean_compression_gain": 0.1,
        "positive_lift_family_count": 6,
        "explained_m2_family_count": 2,
    }

    assert is_transferable_role_based_concept(row) is False


def test_v10e_required_output_files_are_produced(tmp_path, monkeypatch) -> None:
    import v6.m4_role_concepts_v10e as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    single_contexts = _build_v10e_single_contexts()
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single_contexts[heldout_family])

    run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e_outputs"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
        )
    )
    output_dir = tmp_path / "v10e_outputs"
    expected = {
        "v10e_report.json",
        "v10e_report.txt",
        "role_based_m4_concepts.parquet",
        "role_based_m4_concepts.json",
        "role_based_concept_transfer_scores.parquet",
        "role_based_concept_by_family.parquet",
        "role_based_concept_compression_scores.parquet",
        "role_based_future_option_prediction_scores.parquet",
        "role_based_baseline_comparison.parquet",
        "fallback_diagnostic_candidates.parquet",
        "fallback_diagnostic_scores.parquet",
        "rejected_candidate_diagnostics.parquet",
        "concept_identity_diagnostics.parquet",
        "candidate_generator_diagnostics.parquet",
        "multiprocessing_diagnostics.parquet",
        "candidate_cap_diagnostics.parquet",
        "v10e_failure_decomposition.parquet",
        "v10e_generator_failure_summary.parquet",
        "v10e_projection_audit.parquet",
        "v10e_baseline_dominance_audit.parquet",
        "v10e_representation_loss_audit.parquet",
        "v10e_closest_candidates.parquet",
        "m4_failure_diagnostics.json",
    }

    assert expected <= {path.name for path in output_dir.iterdir()}
    assert (output_dir / "shards").is_dir()
    assert (output_dir / "shards" / "role_based_candidates__holdA.parquet").exists()
    assert (output_dir / "shards" / "family_summary__holdB.parquet").exists()


def test_v10e_failure_decomposition_records_failed_gates(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.m4_role_concepts_v10e as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    single_contexts = _build_v10e_single_contexts()
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single_contexts[heldout_family])

    payload = run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e_diag"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
        )
    )
    failure = pd.read_parquet(tmp_path / "v10e_diag" / "v10e_failure_decomposition.parquet")

    assert payload["validation"]["scientific_conclusion"] == "m4_role_based_not_established"
    assert bool(failure["surface_effect_failed"].any()) is True
    assert bool(failure["future_option_failed"].any()) is True
    assert bool(failure["raw_m2_failed"].any()) is True
    assert bool(failure["graph_no_label_failed"].any()) is True
    assert bool(failure["family_coverage_failed"].any()) is True


def test_v10e_generator_summary_identifies_closest_generator(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.m4_role_concepts_v10e as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    single_contexts = _build_v10e_single_contexts(include_four_role_manifest=True)
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single_contexts[heldout_family])

    payload = run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e_gen"),
            game_set_manifest=str(manifest),
            min_games=1,
            min_manifest_families=1,
        )
    )
    summary = pd.read_parquet(tmp_path / "v10e_gen" / "v10e_generator_failure_summary.parquet")

    assert not summary.empty
    assert summary.iloc[0]["best_candidate_id"].startswith("m4-")
    assert payload["report"]["best_generator_by_role_lift"] in set(summary["generator_type"])


def test_v10e_projection_audit_detects_best_match_vs_mean_gap() -> None:
    rows = build_projection_audit_rows(
        [{"concept_id": "m4-a", "generator_type": "gen", "heldout_family": "holdA"}],
        [
            {"concept_id": "m4-a", "heldout_family": "holdA", "target_family_id": "t1", "target_family_score": 0.9},
            {"concept_id": "m4-a", "heldout_family": "holdA", "target_family_id": "t2", "target_family_score": 0.1},
        ],
    )

    assert rows[0]["best_target_family_id"] == "t1"
    assert rows[0]["local_match_lost_by_averaging"] == 0.4


def test_m4_failure_diagnostics_helpers() -> None:
    from v6.m4_failure_diagnostics import count_by_reason, ensure_failure_buckets

    counts = count_by_reason(
        [
            {"rejection_reason": "no_target_projection"},
            {"rejection_reason": "no_target_projection"},
            {"rejection_reason": "no_future_option_prediction_lift"},
        ]
    )
    bucketed = ensure_failure_buckets({"no_target_projection": 2})

    assert counts == {
        "no_target_projection": 2,
        "no_future_option_prediction_lift": 1,
    }
    assert bucketed["no_target_projection"] == 2
    assert bucketed["no_source_roles"] == 0


def test_v10e_zero_candidate_family_diagnostics() -> None:
    import v6.m4_role_concepts_v10e as mod

    class _Context:
        heldout_family = "holdA"
        source_neighborhoods = {"fam1": object()}
        source_roles = {}

    row = mod._build_v10e_family_failure_row(
        heldout_family="holdA",
        context=_Context(),
        source_role_map={},
        stable_items=[],
        raw_candidate_rows=[],
        projection_rows=[],
        target_family_rows=[],
        rejected_rows=[],
        fallback_rows=[],
        min_role_count=2,
    )

    assert row["source_neighborhoods_available"] is True
    assert row["source_roles_available"] is False
    assert row["stable_role_items_count"] == 0
    assert row["raw_candidates_count"] == 0
    assert row["failure_reason_counts"]["no_source_roles"] >= 1
    assert row["failure_reason_counts"]["insufficient_stable_role_items"] >= 1


def test_v10fixc_failure_mode_aggregation_from_synthetic_rows() -> None:
    import v6.concept_candidates_v10fixc as mod

    diagnostics = mod.build_fixc_failure_diagnostics(
        by_family_rows=[{"heldout_family": "holdA", "zero_candidate_reason": "subcomposition_generation_failure"}],
        source_role_map_diag_rows=[
            {
                "heldout_family": "holdA",
                "source_neighborhood_count": 1,
                "source_role_count": 1,
                "source_role_map_overlap_count": 1,
                "stable_role_candidate_count": 0,
                "failure_mode": "manifest_resolution_failure",
            }
        ],
        attrition_rows=[{"heldout_family": "holdA", "raw_candidate_count_premerge": 0}],
        raw_candidate_rows=[
            {"heldout_family": "holdA", "candidate_source": "fallback"},
            {"heldout_family": "holdA", "candidate_source": "mixed"},
        ],
        stable_concepts=[],
        transferable_concepts=[],
        mapped_transfer_rows=[],
    )
    row = diagnostics["per_family"][0]

    assert row["failure_reason_counts"]["manifest_resolution_failure"] == 1
    assert row["failure_reason_counts"]["subcomposition_generation_failure"] == 1
    assert row["fallback_candidate_count"] == 1
    assert row["mixed_candidate_count"] == 1


def test_v10e_baseline_dominance_audit_identifies_raw_m2_and_graph_no_label() -> None:
    rows = build_baseline_dominance_rows(
        [
            {
                "concept_id": "m4-a",
                "target_family_id": "t1",
                "lift_vs_raw_m2": -0.9,
                "lift_vs_graph_no_label": -0.2,
                "lift_vs_surface_effect": -0.1,
                "future_option_prediction_lift": -0.05,
                "lift_vs_best_role": 0.1,
            },
            {
                "concept_id": "m4-b",
                "target_family_id": "t2",
                "lift_vs_raw_m2": -0.1,
                "lift_vs_graph_no_label": -0.8,
                "lift_vs_surface_effect": -0.2,
                "future_option_prediction_lift": -0.05,
                "lift_vs_best_role": 0.1,
            },
        ]
    )

    assert rows[0]["raw_m2_dominates"] is True
    assert rows[1]["graph_no_label_dominates"] is True


def _build_v10e_report_fixture(
    *,
    generator_failure_rows: list[dict[str, Any]] | None = None,
    failure_rows: list[dict[str, Any]] | None = None,
    projection_rows: list[dict[str, Any]] | None = None,
    baseline_rows: list[dict[str, Any]] | None = None,
    representation_rows: list[dict[str, Any]] | None = None,
    closest_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return build_v10e_report(
        config=M4RoleConceptsV10eConfig(output_dir="unused", workers=2),
        transfer_report={},
        concept_rows=[],
        transferable_rows=[],
        transfer_rows=[],
        fallback_rows=[],
        fallback_score_rows=[],
        by_family_rows=[],
        merge_diag={},
        fallback_only_signal_detected=False,
        requested_workers=2,
        effective_workers=2,
        multiprocessing_used=True,
        generator_rows=[],
        cap_rows=[],
        failure_decomposition_rows=failure_rows or [],
        generator_failure_rows=generator_failure_rows or [],
        projection_audit_rows=projection_rows or [],
        baseline_dominance_rows=baseline_rows or [],
        closest_candidate_rows=closest_rows or [],
        representation_loss_rows=representation_rows or [],
        failure_diagnostics={"per_family": [], "total_failure_reason_counts": {}, "attrition_totals": {}},
    )


def test_v10e_report_best_generator_by_role_lift_not_empty_when_rows_exist() -> None:
    payload = _build_v10e_report_fixture(
        generator_failure_rows=[
            {"generator_type": "g1", "max_lift_vs_best_role": 0.1},
            {"generator_type": "g2", "max_lift_vs_best_role": 0.3},
        ]
    )

    assert payload["report"]["best_generator_by_role_lift"] == "g2"


def test_v10e_report_distinguishes_raw_m2_failure_from_raw_m2_dominance() -> None:
    payload = _build_v10e_report_fixture(
        failure_rows=[
            {"failed_gates": ["raw_m2"], "future_option_failed": False, "surface_effect_failed": False, "family_coverage_failed": False},
            {"failed_gates": ["raw_m2"], "future_option_failed": False, "surface_effect_failed": False, "family_coverage_failed": False},
        ],
        baseline_rows=[
            {"raw_m2_dominates": False, "graph_no_label_dominates": True},
            {"raw_m2_dominates": False, "graph_no_label_dominates": True},
        ],
    )

    assert payload["report"]["raw_m2_failure_rate"] == 1.0
    assert payload["report"]["raw_m2_dominance_rate"] == 0.0


def test_v10e_family_coverage_failure_has_priority_in_diagnostic_conclusion() -> None:
    payload = _build_v10e_report_fixture(
        failure_rows=[
            {"failed_gates": ["family_coverage", "raw_m2", "graph_no_label"], "future_option_failed": False, "surface_effect_failed": False, "family_coverage_failed": True},
            {"failed_gates": ["family_coverage", "raw_m2", "graph_no_label"], "future_option_failed": False, "surface_effect_failed": False, "family_coverage_failed": True},
        ],
        baseline_rows=[
            {"raw_m2_dominates": False, "graph_no_label_dominates": True},
            {"raw_m2_dominates": False, "graph_no_label_dominates": True},
        ],
    )

    assert payload["report"]["diagnostic_conclusion"] == "m4_failure_due_to_family_coverage"


def test_v10e_report_includes_projection_averaging_summary() -> None:
    payload = _build_v10e_report_fixture(
        projection_rows=[
            {"concept_id": "c1", "target_best_match_score": 0.9, "target_mean_score": 0.3, "best_target_family_id": "t1", "local_match_lost_by_averaging": 0.6},
            {"concept_id": "c2", "target_best_match_score": 0.8, "target_mean_score": 0.5, "best_target_family_id": "t2", "local_match_lost_by_averaging": 0.3},
        ]
    )

    assert abs(payload["report"]["mean_projection_averaging_loss"] - 0.45) < 1e-9
    assert abs(payload["report"]["max_projection_averaging_loss"] - 0.6) < 1e-9
    assert payload["report"]["worst_projection_averaging_candidate"] == "c1"
    assert payload["report"]["best_local_match_candidate"] == "c1"
    assert payload["report"]["best_local_match_score"] == 0.9
    assert payload["report"]["best_local_match_target_family"] == "t1"


def test_v10e_report_includes_diagnostic_row_counts() -> None:
    payload = _build_v10e_report_fixture(
        failure_rows=[{"failed_gates": []}] * 3,
        generator_failure_rows=[{"generator_type": "g", "max_lift_vs_best_role": 0.1}] * 2,
        projection_rows=[{"concept_id": "c", "target_best_match_score": 0.2, "target_mean_score": 0.1, "best_target_family_id": "t", "local_match_lost_by_averaging": 0.1}] * 4,
        baseline_rows=[{"raw_m2_dominates": False, "graph_no_label_dominates": False}] * 5,
        representation_rows=[{"concept_id": "x"}] * 6,
        closest_rows=[{"concept_id": "y"}] * 7,
    )

    assert payload["report"]["failure_decomposition_rows"] == 3
    assert payload["report"]["generator_failure_summary_rows"] == 2
    assert payload["report"]["projection_audit_rows"] == 4
    assert payload["report"]["baseline_dominance_rows"] == 5
    assert payload["report"]["representation_loss_rows"] == 6
    assert payload["report"]["closest_candidate_rows"] == 7


def test_v10e_deterministic_concept_ids_and_cli_accepts_options(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.m4_role_concepts_v10e as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    single_contexts = _build_v10e_single_contexts()
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "build_single_family_context", lambda config, heldout_family: single_contexts[heldout_family])
    monkeypatch.setattr(mod, "choose_v10e_worker_count", lambda requested_workers, heldout_family_count, worker_memory_gib: min(requested_workers, heldout_family_count))

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _ImmediateFuture(fn(*args, **kwargs))

    monkeypatch.setattr(mod, "PROCESS_POOL_CLASS", _FakeExecutor)

    one = run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e_one"),
            game_set_manifest=str(manifest),
            workers=1,
            min_games=1,
            min_manifest_families=1,
        )
    )
    many = run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e_many"),
            game_set_manifest=str(manifest),
            workers=2,
            min_games=1,
            min_manifest_families=1,
        )
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "m4-role-concepts-v10e",
            "--m3-input-dir",
            "runs/v6/v08d_cd2_extended32_sourceclean",
            "--transfer-input-dir",
            "runs/v6/v09c_transfer_hardened_extended32",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--previous-v09b-dir",
            "runs/custom_v09b",
            "--output-dir",
            "runs/v6/v10e_role_based_m4_extended32",
            "--workers",
            "2",
        ]
    )
    ids_one = sorted(pd.read_parquet(tmp_path / "v10e_one" / "role_based_m4_concepts.parquet")["concept_id"].tolist())
    ids_many = sorted(pd.read_parquet(tmp_path / "v10e_many" / "role_based_m4_concepts.parquet")["concept_id"].tolist())

    assert one["validation"]["scientific_conclusion"] == many["validation"]["scientific_conclusion"]
    assert ids_one == ids_many
    assert many["report"]["effective_workers"] == 2
    assert one["report"]["diagnostic_conclusion"] == many["report"]["diagnostic_conclusion"]
    assert args.command == "m4-role-concepts-v10e"
    assert args.previous_v09b_dir == "runs/custom_v09b"


def test_v10e_passes_configured_previous_v09b_dir(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import v6.m4_role_concepts_v10e as mod

    transfer_dir = tmp_path / "transfer"
    transfer_dir.mkdir()
    (transfer_dir / "v09c_report.json").write_text(json.dumps({"report": {}, "validation": {}}), encoding="utf-8")
    captured: dict[str, str] = {}

    monkeypatch.setattr(mod.pd, "read_parquet", lambda path: pd.DataFrame())

    def fake_list_heldout_families(role_config):
        captured["previous_v09b_dir"] = role_config.previous_v09b_dir
        return []

    monkeypatch.setattr(mod, "list_heldout_families", fake_list_heldout_families)
    monkeypatch.setattr(mod, "load_v10e_shards", lambda shards_dir, prefix: [])
    monkeypatch.setattr(mod, "merge_exact_candidates", lambda rows, **kwargs: ([], {}))
    monkeypatch.setattr(mod, "apply_target_metrics", lambda rows, transfer_rows: [])
    monkeypatch.setattr(mod, "annotate_projection_outcomes", lambda rows, transfer_rows: [])
    monkeypatch.setattr(mod, "apply_role_based_gates", lambda rows, transfer_rows: [])
    monkeypatch.setattr(mod, "build_rejected_candidate_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_concept_identity_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_compression_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_future_option_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_baseline_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_failure_decomposition_rows", lambda concept_rows, target_family_rows: [])
    monkeypatch.setattr(mod, "build_generator_failure_summary_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_projection_audit_rows", lambda concept_rows, target_family_rows: [])
    monkeypatch.setattr(mod, "build_baseline_dominance_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_representation_loss_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_closest_candidate_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_candidate_generator_rows", lambda rows: [])
    monkeypatch.setattr(mod, "build_multiprocessing_rows", lambda family_results, **kwargs: [])
    monkeypatch.setattr(mod, "build_candidate_cap_rows", lambda rows: [])
    monkeypatch.setattr(
        mod,
        "build_v10e_report",
        lambda **kwargs: {
            "config": {"previous_v09b_dir": kwargs["config"].previous_v09b_dir},
            "report": {},
            "validation": {},
        },
    )
    monkeypatch.setattr(mod, "_write_parquet", lambda path, rows: None)
    monkeypatch.setattr(mod, "format_v10e_report", lambda payload: "")

    payload = run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            previous_v09b_dir="runs/custom_v09b",
            output_dir=str(tmp_path / "out"),
        )
    )

    assert captured["previous_v09b_dir"] == "runs/custom_v09b"
    assert payload["config"]["previous_v09b_dir"] == "runs/custom_v09b"


def test_v10e_workers_2_uses_executor_and_builds_context_inside_worker(tmp_path, monkeypatch) -> None:
    import v6.m4_role_concepts_v10e as mod

    transfer_dir, manifest = _write_v10fixb_fixture(tmp_path)
    single_contexts = _build_v10e_single_contexts()
    built = []
    submitted = []
    monkeypatch.setattr(mod, "list_heldout_families", lambda config: ["holdA", "holdB"])
    monkeypatch.setattr(mod, "choose_v10e_worker_count", lambda requested_workers, heldout_family_count, worker_memory_gib: 2)
    monkeypatch.setattr(
        mod,
        "build_single_family_context",
        lambda config, heldout_family: built.append(heldout_family) or single_contexts[heldout_family],
    )

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            submitted.append((fn, args, kwargs))
            return _ImmediateFuture(fn(*args, **kwargs))

    monkeypatch.setattr(mod, "PROCESS_POOL_CLASS", _FakeExecutor)

    payload = run_m4_role_concepts_v10e(
        M4RoleConceptsV10eConfig(
            transfer_input_dir=str(transfer_dir),
            output_dir=str(tmp_path / "v10e_mp"),
            game_set_manifest=str(manifest),
            workers=2,
            min_games=1,
            min_manifest_families=1,
        )
    )

    assert payload["report"]["multiprocessing_used"] is True
    assert payload["report"]["effective_workers"] == 2
    assert built == ["holdA", "holdB"]
    assert len(submitted) == 2
    assert all(isinstance(args[0], str) for _, args, _ in submitted)
    assert all(not any(isinstance(arg, SingleFamilyContext) for arg in args) for _, args, _ in submitted)


def test_v10e_generators_produce_graph_future_and_motif_candidates() -> None:
    import v6.m4_role_concepts_v10e as mod

    context = _build_v10e_single_contexts(include_four_role_manifest=True)["holdA"]
    stable_items = mod.build_stable_role_items(context, max_role_items=128)
    rows, counts, _, _ = mod.generate_role_based_candidates(context, stable_items, max_role_count=3, max_candidates=250000)

    assert rows
    assert counts["adjacent_graph_pair"] > 0
    assert counts["future_option_pair"] > 0
    assert counts["graph_motif_bridge"] > 0


def test_v08d_no_label_ablation_removes_label_features(tmp_path) -> None:
    import pandas as pd

    input_dir, m1_dir = _write_v08_fixture(tmp_path)
    payload = run_role_candidates_v08d(
        RoleCandidatesV08dConfig(
            input_dir=str(input_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "out_no_label"),
            workers=1,
            game_set_manifest=str(tmp_path / "game_set.json"),
            ablation="no_m2_labels",
            graph_source="hybrid",
        )
    )

    rows = pd.read_parquet(tmp_path / "out_no_label" / "role_neighborhoods.parquet").to_dict(orient="records")
    assert all(not any(str(key).startswith("label::") for key in row["coarse_features"]) for row in rows)
    assert payload["report"]["ablation"] == "no_m2_labels"


def test_v09a_runs_sourceclean_and_uses_structural_metric(tmp_path) -> None:
    m2_dir, m1_dir = _write_v08_fixture(tmp_path)
    payload = run_role_transfer_v09a(
        RoleTransferV09aConfig(
            m2_input_dir=str(m2_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "v09a"),
            game_set_manifest=str(tmp_path / "game_set.json"),
            workers=1,
        )
    )

    assert "transfer_accuracy_structural_role" in payload["report"]
    assert payload["validation"]["scientific_conclusion"] in {
        "transfer_methodology_invalid",
        "role_transfer_sourceclean_not_established",
        "role_transfer_sourceclean_partial",
        "role_transfer_sourceclean_weak",
        "role_transfer_sourceclean_strong",
    }


def test_v09a_deterministic_on_fixture(tmp_path) -> None:
    m2_dir, m1_dir = _write_v08_fixture(tmp_path)
    one = run_role_transfer_v09a(
        RoleTransferV09aConfig(
            m2_input_dir=str(m2_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "v09a1"),
            game_set_manifest=str(tmp_path / "game_set.json"),
            workers=1,
        )
    )
    many = run_role_transfer_v09a(
        RoleTransferV09aConfig(
            m2_input_dir=str(m2_dir),
            m1_input_dir=str(m1_dir),
            output_dir=str(tmp_path / "v09a25"),
            game_set_manifest=str(tmp_path / "game_set.json"),
            workers=25,
        )
    )

    assert one["validation"]["scientific_conclusion"] == many["validation"]["scientific_conclusion"]
    assert one["report"]["transfer_accuracy_structural_role"] == many["report"]["transfer_accuracy_structural_role"]


def test_cli_accepts_role_transfer_v09a_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "role-transfer-v09a",
            "--m2-input-dir",
            "runs/v6/v07_cd2_extended32_expanded",
            "--m1-input-dir",
            "runs/v6/v06_cd2_extended32",
            "--output-dir",
            "runs/v6/v09a_role_transfer_sourceclean_extended32",
            "--split-mode",
            "leave_family_out",
            "--workers",
            "25",
            "--graph-source",
            "hybrid",
        ]
    )

    assert args.command == "role-transfer-v09a"
    assert args.graph_source == "hybrid"


def test_v05c_resolve_scope_can_select_only_missing_manifest_games(tmp_path) -> None:
    root = tmp_path / "parquet"
    (root / "game=tt01").mkdir(parents=True)
    (root / "game=pb02").mkdir(parents=True)
    manifest = tmp_path / "extended.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "extended32_v08",
                "games": ["tt01", "pb02", "fs02", "tp02"],
                "families": {"a": ["tt01", "pb02"], "b": ["fs02", "tp02"]},
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_interaction_sampling_scope(
        InteractionSamplingConfig(
            games=("x",),
            parquet_root=str(root),
            game_set_manifest=str(manifest),
            only_missing_from_parquet_root=True,
        )
    )

    assert resolved.games == ("fs02", "tp02")


def test_cli_accepts_interaction_sampling_manifest_gap_fix_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "interaction-sampling-v05c",
            "--games",
            "broad",
            "--game-set-manifest",
            "runs/v6/game_sets/extended32_v08.json",
            "--game-set-name",
            "extended32_v08",
            "--only-missing-from-parquet-root",
        ]
    )

    assert args.command == "interaction-sampling-v05c"
    assert args.game_set_name == "extended32_v08"
    assert args.only_missing_from_parquet_root is True


def test_cli_accepts_interaction_sampling_collect_only_option() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "interaction-sampling-v05c",
            "--games",
            "broad",
            "--storage-backend",
            "parquet",
            "--collect-only",
        ]
    )

    assert args.command == "interaction-sampling-v05c"
    assert args.collect_only is True


def test_interaction_sampling_preset_fills_defaults() -> None:
    parser = build_parser()
    argv = [
        "interaction-sampling-v05c",
        "--experiment-preset",
        "broad_hypothesis_probe",
        "--output-dir",
        "runs/out",
    ]
    args = _apply_interaction_sampling_experiment_preset(parser.parse_args(argv), argv)

    assert args.games == "all"
    assert args.samplers == "random_baseline,low_confidence,novelty_delta,mixed,reset_aware_mixed"
    assert args.seeds == "0"
    assert args.steps == 5000
    assert args.horizon == 10
    assert args.context_depth == 1


def test_interaction_sampling_explicit_args_override_preset_defaults() -> None:
    parser = build_parser()
    argv = [
        "interaction-sampling-v05c",
        "--experiment-preset",
        "broad_hypothesis_probe",
        "--games",
        "tt01,pb02",
        "--steps",
        "7000",
        "--context-depth",
        "3",
    ]
    args = _apply_interaction_sampling_experiment_preset(parser.parse_args(argv), argv)

    assert args.games == "tt01,pb02"
    assert args.steps == 7000
    assert args.context_depth == 3


def test_cli_accepts_interaction_sampling_adaptive_context_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "interaction-sampling-v05c",
            "--adaptive-context-expansion",
            "true",
            "--max-context-depth",
            "4",
        ]
    )

    assert args.command == "interaction-sampling-v05c"
    assert args.adaptive_context_expansion is True
    assert args.max_context_depth == 4


def test_cli_accepts_fast_postprocessing_options() -> None:
    parser = build_parser()
    default_continuous_args = parser.parse_args(
        [
            "continuous-research-run",
            "--experiment-name",
            "exp",
            "--output-dir",
            "runs/out",
        ]
    )
    sampling_args = parser.parse_args(
        [
            "interaction-sampling-v05c",
            "--fast-postprocessing",
            "true",
        ]
    )
    continuous_args = parser.parse_args(
        [
            "continuous-research-run",
            "--experiment-name",
            "exp",
            "--output-dir",
            "runs/out",
            "--fast-postprocessing",
            "true",
        ]
    )

    assert sampling_args.fast_postprocessing is True
    assert default_continuous_args.fast_postprocessing is True
    assert continuous_args.fast_postprocessing is True


def test_cli_accepts_initial_workers_for_continuous_run() -> None:
    parser = build_parser()
    default_args = parser.parse_args(
        [
            "continuous-research-run",
            "--experiment-name",
            "exp",
            "--output-dir",
            "runs/out",
        ]
    )
    explicit_args = parser.parse_args(
        [
            "continuous-research-run",
            "--experiment-name",
            "exp",
            "--output-dir",
            "runs/out",
            "--initial-workers",
            "3",
        ]
    )

    assert default_args.initial_workers is None
    assert explicit_args.initial_workers == 3


def test_v05c_generate_sampling_dbs_passes_context_depth_into_jobs(tmp_path, monkeypatch) -> None:
    captured: list[dict] = []

    def fake_run_sampling_jobs(jobs: list[dict], *, workers: int) -> None:
        captured.extend(jobs)

    monkeypatch.setattr(interaction_sampling, "_run_sampling_jobs", fake_run_sampling_jobs)

    interaction_sampling._generate_sampling_dbs(
        InteractionSamplingConfig(
            context_depth=2,
            adaptive_context_expansion=True,
            max_context_depth=4,
            games=("tt01",),
            samplers=("random_baseline",),
            seeds=(0,),
            steps=10,
            workers=1,
        ),
        tmp_path,
    )

    assert len(captured) == 1
    assert captured[0]["context_depth"] == 2
    assert captured[0]["adaptive_context_expansion"] is True
    assert captured[0]["max_context_depth"] == 4


def test_v05c_run_sampling_job_uses_context_depth_for_v6_config(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummySampler:
        reset_count = 0
        reset_unavailable = False

    class DummyEnv:
        reset_count = 0
        skipped_terminal_steps = 0

        def __init__(self, game_id: str, seed: int, env_root: str | None = None) -> None:
            captured["env_init"] = {"game_id": game_id, "seed": seed, "env_root": env_root}

    class DummySystem:
        def __init__(self, env, config, action_sampler) -> None:
            captured["env"] = env
            captured["config"] = config
            captured["action_sampler"] = action_sampler

        def run(self, steps: int) -> None:
            captured["steps"] = steps

        def close(self) -> None:
            captured["closed"] = True

        def adaptive_context_summary(self) -> dict:
            return {
                "adaptive_context_expansion_enabled": bool(captured["config"].adaptive_context_expansion),
                "base_context_depth": int(captured["config"].context_length),
                "max_context_depth": int(captured["config"].max_context_depth or captured["config"].context_length),
                "adaptive_context_expansion_count": 1,
                "adaptive_context_active_action_count": 1,
                "adaptive_context_max_depth_reached": int(captured["config"].max_context_depth or captured["config"].context_length),
                "adaptive_context_action_depths": {1: int(captured["config"].max_context_depth or captured["config"].context_length)},
            }

    def fake_make_sampler(name: str, seed: int) -> DummySampler:
        captured["sampler_init"] = {"name": name, "seed": seed}
        return DummySampler()

    def fake_write_sampling_metadata(path, **values) -> None:
        captured["metadata_path"] = path
        captured["metadata"] = values

    monkeypatch.setattr(interaction_sampling, "make_sampler", fake_make_sampler)
    monkeypatch.setattr(interaction_sampling, "ArcGridEnvironment", DummyEnv)
    monkeypatch.setattr(interaction_sampling, "V6System", DummySystem)
    monkeypatch.setattr(interaction_sampling, "_write_sampling_metadata", fake_write_sampling_metadata)

    interaction_sampling._run_sampling_job(
        {
            "game": "tt01",
            "sampler_name": "random_baseline",
            "seed": 0,
            "steps": 10,
            "horizon": 3,
            "context_depth": 4,
            "adaptive_context_expansion": True,
            "max_context_depth": 6,
            "commit_steps": 5,
            "db_path": str(tmp_path / "seed_0.sqlite"),
            "env_root": None,
        }
    )

    assert captured["config"].context_length == 4
    assert captured["config"].adaptive_context_expansion is True
    assert captured["config"].max_context_depth == 6
    assert captured["metadata"]["context_depth"] == 4
    assert captured["metadata"]["context_length"] == 4
    assert captured["metadata"]["adaptive_context_expansion_enabled"] is True
    assert captured["metadata"]["base_context_depth"] == 4
    assert captured["metadata"]["max_context_depth"] == 6
    assert "adaptive_context_expansion_count" in captured["metadata"]
    assert "adaptive_context_active_action_count" in captured["metadata"]
    assert "adaptive_context_max_depth_reached" in captured["metadata"]
    assert "carrier_candidate_count" in captured["metadata"]
    assert "emergent_carrier_count" in captured["metadata"]
    assert "carrier_spatial_candidate_count" in captured["metadata"]
    assert "carrier_object_candidate_count" in captured["metadata"]
    assert "carrier_cell_candidate_count" in captured["metadata"]
    assert "carrier_context_action_fallback_candidate_count" in captured["metadata"]
    assert "emergent_spatial_carrier_count" in captured["metadata"]
    assert "emergent_object_carrier_count" in captured["metadata"]
    assert "emergent_cell_carrier_count" in captured["metadata"]
    assert "emergent_context_action_fallback_count" in captured["metadata"]
    assert "carrier_event_count" in captured["metadata"]
    assert "carrier_max_support" in captured["metadata"]
    assert "memory_record_count" in captured["metadata"]
    assert "memory_active_count" in captured["metadata"]
    assert "memory_protected_count" in captured["metadata"]
    assert "memory_compressed_count" in captured["metadata"]
    assert "memory_forgotten_count" in captured["metadata"]
    assert "memory_replay_candidate_count" in captured["metadata"]
    assert "efficiency_event_count" in captured["metadata"]
    assert "efficiency_total_action_cost" in captured["metadata"]
    assert "efficiency_mean_action_cost" in captured["metadata"]
    assert "efficiency_no_effect_action_count" in captured["metadata"]


def test_v05c_run_sampling_job_fast_postprocessing_skips_expensive_outputs(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {"delta_called": False, "apply_called": False}

    class DummySampler:
        reset_count = 0
        reset_unavailable = False

    class DummyEnv:
        def __init__(self, game_id: str, seed: int, env_root=None) -> None:
            self.game_id = game_id
            self.seed = seed
            self.env_root = env_root
            self.reset_count = 0
            self.skipped_terminal_steps = 0

    class DummyGraph:
        def edge_type_counts(self) -> dict[str, int]:
            return {"supports": 1}

        def export_compact_rows(self) -> dict[str, list[dict[str, object]]]:
            return {"nodes": [], "edges": []}

    class DummyMemoryLifecycle:
        def summary(self) -> dict[str, object]:
            return {"memory_record_count": 1, "memory_replay_candidate_count": 1}

        def get_replay_batch(self, limit: int = 1000) -> list[dict[str, object]]:
            return [{"replay_id": "r1", "priority_score": 0.9}]

    class DummySystem:
        def __init__(self, env, config, action_sampler=None) -> None:
            self.env = env
            self.config = config
            self.action_sampler = action_sampler
            self.graph = DummyGraph()
            self.context_contradictions = type("Tracker", (), {"summary": lambda self: {"context_contradiction_count": 0}})()
            self.carrier_tracker = type("Carrier", (), {"build_candidates": lambda self: []})()
            self.memory_lifecycle = DummyMemoryLifecycle()
            self.efficiency_tracker = type("Efficiency", (), {"summary": lambda self: {"efficiency_event_count": 2}})()
            self.compact_memory_restore_summary = {}

        def run(self, steps: int) -> None:
            pass

        def close(self) -> None:
            pass

        def adaptive_context_summary(self) -> dict[str, object]:
            return {"adaptive_context_expansion_enabled": False, "base_context_depth": 1, "max_context_depth": 1}

    def fake_make_sampler(name: str, seed: int) -> DummySampler:
        return DummySampler()

    def fake_future_option_deltas(db_path, *, horizon: int) -> dict[str, float]:
        captured["delta_called"] = True
        return {}

    def fake_apply_future_option_efficiency_diagnostics(db_path, deltas_by_interaction_id: dict[str, float]) -> None:
        captured["apply_called"] = True

    def fake_write_sampling_metadata(path, **values) -> None:
        captured["metadata"] = values

    monkeypatch.setattr(interaction_sampling, "make_sampler", fake_make_sampler)
    monkeypatch.setattr(interaction_sampling, "ArcGridEnvironment", DummyEnv)
    monkeypatch.setattr(interaction_sampling, "V6System", DummySystem)
    monkeypatch.setattr(interaction_sampling, "_future_option_deltas_by_interaction_id", fake_future_option_deltas)
    monkeypatch.setattr(interaction_sampling, "_apply_future_option_efficiency_diagnostics", fake_apply_future_option_efficiency_diagnostics)
    monkeypatch.setattr(interaction_sampling, "_write_sampling_metadata", fake_write_sampling_metadata)

    db_path = tmp_path / "seed_0.sqlite"
    result = interaction_sampling._run_sampling_job(
        {
            "game": "tt01",
            "sampler_name": "random_baseline",
            "seed": 0,
            "steps": 10,
            "horizon": 3,
            "context_depth": 1,
            "commit_steps": 5,
            "db_path": str(db_path),
            "env_root": None,
            "fast_postprocessing": True,
        }
    )

    assert result == {"legacy_future_effects_removed": True}
    assert captured["delta_called"] is False
    assert captured["apply_called"] is False
    assert captured["metadata"]["fast_postprocessing_enabled"] is True
    assert captured["metadata"]["future_effects_postprocessing_skipped"] is True
    assert captured["metadata"]["future_effects_legacy_removed"] is True
    assert captured["metadata"]["future_option_memory_source"] == "memory_substrate"
    assert db_path.with_name("live_graph_compact.json").exists()
    assert db_path.with_name("carrier_candidates.json").exists()
    assert db_path.with_name("memory_lifecycle_summary.json").exists()
    assert not db_path.with_name("memory_replay_candidates.json").exists()
    assert not db_path.with_name("efficiency_summary.json").exists()


def test_v05c_run_sampling_job_passes_memory_flags_into_v6config(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummySampler:
        reset_count = 0
        reset_unavailable = False

        def memory_guided_summary(self) -> dict[str, float | int]:
            return {
                "memory_guided_action_count": 3,
                "memory_guided_fallback_count": 1,
                "mean_memory_action_score": 0.6,
                "selected_action_memory_score_mean": 0.7,
                "selected_action_failure_risk_mean": 0.2,
                "selected_action_future_option_gain_mean": 0.3,
            }

    class DummyEnv:
        def __init__(self, game_id: str, seed: int, env_root=None) -> None:
            self.game_id = game_id
            self.seed = seed
            self.env_root = env_root
            self.reset_count = 0
            self.skipped_terminal_steps = 0

    class DummyGraph:
        def edge_type_counts(self) -> dict[str, int]:
            return {}

        def export_compact_rows(self) -> dict[str, list[dict[str, object]]]:
            return {"nodes": [], "edges": []}

    class DummySystem:
        def __init__(self, env, config, action_sampler=None, live_memory_queue=None, live_memory_cache=None) -> None:
            del live_memory_queue, live_memory_cache
            captured["config"] = config
            self.env = env
            self.config = config
            self.action_sampler = action_sampler
            self.graph = DummyGraph()
            self.context_contradictions = type("Tracker", (), {"summary": lambda self: {}})()
            self.carrier_tracker = type("Carrier", (), {"build_candidates": lambda self: []})()
            self.memory_lifecycle = type("Lifecycle", (), {"summary": lambda self: {}, "get_replay_batch": lambda self, limit=1000: []})()
            self.efficiency_tracker = type("Efficiency", (), {"summary": lambda self: {}})()
            self.compact_memory_restore_summary = {}

        def run(self, steps: int) -> None:
            del steps

        def close(self) -> None:
            pass

        def adaptive_context_summary(self) -> dict[str, object]:
            return {"adaptive_context_expansion_enabled": False, "base_context_depth": 1, "max_context_depth": 1}

    def fake_write_sampling_metadata(path, **values) -> None:
        del path
        captured["metadata"] = values

    monkeypatch.setattr(interaction_sampling, "make_sampler", lambda name, seed: DummySampler())
    monkeypatch.setattr(interaction_sampling, "ArcGridEnvironment", DummyEnv)
    monkeypatch.setattr(interaction_sampling, "V6System", DummySystem)
    monkeypatch.setattr(interaction_sampling, "_write_sampling_metadata", fake_write_sampling_metadata)

    interaction_sampling._run_sampling_job(
        {
            "game": "tt01",
            "sampler_name": "memory_guided",
            "seed": 0,
            "steps": 10,
            "horizon": 3,
            "context_depth": 1,
            "commit_steps": 5,
            "db_path": str(tmp_path / "seed_0.sqlite"),
            "env_root": None,
            "memory_query_enabled": True,
            "memory_action_selection_enabled": True,
            "restore_compact_graph": True,
            "restore_compact_substrate": True,
        }
    )

    config = captured["config"]
    metadata = captured["metadata"]
    assert config.memory_query_enabled is True
    assert config.memory_action_selection_enabled is True
    assert config.restore_compact_graph is True
    assert config.restore_compact_substrate is True
    assert metadata["memory_query_enabled"] is True
    assert metadata["memory_action_selection_enabled"] is True
    assert metadata["restore_compact_graph"] is True
    assert metadata["restore_compact_substrate"] is True
    assert metadata["memory_guided_action_count"] == 3
    assert metadata["memory_guided_fallback_count"] == 1


def test_continuous_research_passes_memory_flags_into_interaction_sampling(tmp_path, monkeypatch) -> None:
    import v6.continuous_research as continuous_research

    captured: dict[str, object] = {}

    def _capture_sampling_config(config):
        captured["sampling_config"] = config
        return []

    monkeypatch.setattr(continuous_research, "run_interaction_sampling_v05c", _capture_sampling_config)
    monkeypatch.setattr(
        continuous_research,
        "run_hypothesis_suite_report",
        lambda **kwargs: {"H01 decision": "INSUFFICIENT_EVIDENCE", "H03 decision": "INSUFFICIENT_EVIDENCE", "H07 decision": "INSUFFICIENT_EVIDENCE", "H10 decision": "INSUFFICIENT_EVIDENCE"},
    )
    monkeypatch.setattr(continuous_research, "run_selective_forgetting_pass", lambda **kwargs: {})
    monkeypatch.setattr(continuous_research, "evaluate_h10b_selective_forgetting", lambda **kwargs: {"decision": "INSUFFICIENT_EVIDENCE"})
    monkeypatch.setattr(continuous_research, "build_memory_summary", lambda *args, **kwargs: {"stable_contingency_count": 0, "transformation_family_count": 0, "memory_node_count": 0})
    monkeypatch.setattr(continuous_research, "load_memory_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(continuous_research, "_write_memory_continuity_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(continuous_research, "validate_cleanup_safe", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(continuous_research, "cleanup_epoch_artifacts", lambda *args, **kwargs: {"disk_before_cleanup_bytes": 0, "disk_after_cleanup_bytes": 0})

    continuous_research.run_continuous_research(
        continuous_research.ContinuousResearchConfig(
            experiment_name="memory_flags",
            games="tt01",
            samplers="memory_guided",
            seeds="0",
            steps_per_epoch=1,
            max_epochs=1,
            horizon=2,
            context_depth=1,
            output_dir=str(tmp_path / "continuous"),
            cleanup=True,
            memory_query_enabled=True,
            memory_action_selection_enabled=True,
            restore_compact_graph=True,
            restore_compact_substrate=True,
        )
    )

    sampling_config = captured["sampling_config"]
    assert sampling_config.memory_query_enabled is True
    assert sampling_config.memory_action_selection_enabled is True
    assert sampling_config.restore_compact_graph is True
    assert sampling_config.restore_compact_substrate is True


def test_direct_streaming_fold_job_id_is_epoch_qualified() -> None:
    job = {
        "game": "tt01",
        "sampler_name": "mixed",
        "seed": 0,
        "steps": 100,
        "global_step_offset": 200,
        "db_path": "runs/v6/continuous/x/epochs/epoch_0003/raw/sampling_v05c/tt01/mixed/steps_100/seed_0.sqlite",
    }
    job_id = interaction_sampling._direct_streaming_fold_job_id(job)
    assert job_id.startswith("epoch_0003:tt01:mixed:seed0:steps100")
    assert job_id.endswith(":g201-300")


def test_resume_rejects_incomplete_next_epoch(tmp_path) -> None:
    import v6.continuous_research as continuous_research

    root = tmp_path / "continuous"
    memory_dir = root / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    for name in ("current_state.sqlite", "graph.sqlite", "replay_queue.sqlite", "memory_summary.json"):
        (memory_dir / name).write_text("{}", encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "current_epoch": 2,
                "memory_paths": {
                    "current_state": str(memory_dir / "current_state.sqlite"),
                    "graph": str(memory_dir / "graph.sqlite"),
                    "replay_queue": str(memory_dir / "replay_queue.sqlite"),
                    "memory_summary": str(memory_dir / "memory_summary.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    status_dir = root / "epochs" / "epoch_0003" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "epoch_start.json").write_text(json.dumps({"epoch_id": "epoch_0003"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete next epoch"):
        continuous_research._load_or_initialize_manifest(
            continuous_research.ContinuousResearchConfig(
                experiment_name="resume_guard",
                games="tt01",
                samplers="mixed",
                seeds="0",
                steps_per_epoch=100,
                max_epochs=4,
                horizon=2,
                context_depth=1,
                output_dir=str(root),
                resume=True,
            ),
            manifest_path,
        )


def test_v05c_run_sampling_job_non_fast_mode_does_not_call_legacy_future_effects(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {"delta_called": False, "apply_called": False}

    class DummySampler:
        reset_count = 0
        reset_unavailable = False

    class DummyEnv:
        def __init__(self, game_id: str, seed: int, env_root=None) -> None:
            self.game_id = game_id
            self.seed = seed
            self.env_root = env_root
            self.reset_count = 0
            self.skipped_terminal_steps = 0

    class DummyGraph:
        def edge_type_counts(self) -> dict[str, int]:
            return {}

        def export_compact_rows(self) -> dict[str, list[dict[str, object]]]:
            return {"nodes": [], "edges": []}

    class DummySystem:
        def __init__(self, env, config, action_sampler=None) -> None:
            self.env = env
            self.config = config
            self.graph = DummyGraph()
            self.context_contradictions = type("Tracker", (), {"summary": lambda self: {}})()
            self.carrier_tracker = type("Carrier", (), {"build_candidates": lambda self: []})()
            self.memory_lifecycle = type("Lifecycle", (), {"summary": lambda self: {}, "get_replay_batch": lambda self, limit=1000: []})()
            self.efficiency_tracker = type("Efficiency", (), {"summary": lambda self: {}})()
            self.compact_memory_restore_summary = {}

        def run(self, steps: int) -> None:
            pass

        def close(self) -> None:
            pass

        def adaptive_context_summary(self) -> dict[str, object]:
            return {"adaptive_context_expansion_enabled": False, "base_context_depth": 1, "max_context_depth": 1}

    def fake_future_option_deltas(db_path, *, horizon: int) -> dict[str, float]:
        captured["delta_called"] = True
        return {}

    def fake_apply_future_option_efficiency_diagnostics(db_path, deltas_by_interaction_id: dict[str, float]) -> None:
        captured["apply_called"] = True

    monkeypatch.setattr(interaction_sampling, "make_sampler", lambda name, seed: DummySampler())
    monkeypatch.setattr(interaction_sampling, "ArcGridEnvironment", DummyEnv)
    monkeypatch.setattr(interaction_sampling, "V6System", DummySystem)
    monkeypatch.setattr(interaction_sampling, "_future_option_deltas_by_interaction_id", fake_future_option_deltas)
    monkeypatch.setattr(interaction_sampling, "_apply_future_option_efficiency_diagnostics", fake_apply_future_option_efficiency_diagnostics)

    result = interaction_sampling._run_sampling_job(
        {
            "game": "tt01",
            "sampler_name": "random_baseline",
            "seed": 0,
            "steps": 10,
            "horizon": 3,
            "context_depth": 1,
            "commit_steps": 5,
            "db_path": str(tmp_path / "seed_0.sqlite"),
            "env_root": None,
            "fast_postprocessing": False,
        }
    )

    assert result == {"legacy_future_effects_removed": True}
    assert captured["delta_called"] is False
    assert captured["apply_called"] is False


def test_live_memory_writer_projects_events(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    queue = make_live_memory_queue(100)
    writer = start_live_memory_writer(
        LiveMemoryWriterConfig(memory_dir=str(memory_dir), queue_maxsize=100, batch_size=2, flush_seconds=0.1),
        queue,
    )
    queue.put(
        {
            "event_type": "stable_contingency",
            "event_id": "stable_contingency:c1",
            "global_step": 1,
            "worker_id": "w1",
            "priority": 0.9,
            "payload": {
                "key": "c1",
                "action": 1,
                "context_signature": "[1,2]",
                "context_level": 0,
                "transformation_family": 7,
                "support_count": 3,
                "confidence": 0.9,
            },
        }
    )
    queue.put(
        {
            "event_type": "family_update",
            "event_id": "family_update:f1",
            "global_step": 1,
            "worker_id": "w1",
            "priority": 0.5,
            "payload": {
                "family_signature": "f1",
                "family_id": 7,
                "support_count": 5,
            },
        }
    )
    queue.put(
        {
            "event_type": "high_priority_replay",
            "event_id": "high_priority_replay:1",
            "global_step": 1,
            "worker_id": "w1",
            "priority": 0.95,
            "payload": {
                "interaction_id": "1",
                "replay_priority": 0.95,
                "reason": "test",
                "family_id": "7",
                "context_signature": "[1,2]",
                "action_signature": "1",
            },
        }
    )
    queue.put(
        {
            "event_type": "contradiction_cluster",
            "event_id": "contradiction_cluster:x1",
            "global_step": 1,
            "worker_id": "w1",
            "priority": 0.8,
            "payload": {
                "contradiction_key": "x1",
                "context_signature": "[1,2]",
                "action_signature": "1",
                "count": 2,
            },
        }
    )
    queue.put(
        {
            "event_type": "carrier_candidate",
            "event_id": "carrier_candidate:k1",
            "global_step": 1,
            "worker_id": "w1",
            "priority": 0.7,
            "payload": {
                "carrier_signature": "k1",
                "carrier_source": "object",
                "support_count": 4,
                "linked_family_count": 1,
            },
        }
    )
    queue.put(
        {
            "event_type": "future_option_event",
            "event_id": "future_option_event:1",
            "global_step": 1,
            "worker_id": "w1",
            "priority": 0.6,
            "payload": {
                "event_id": "fo1",
                "interaction_id": "1",
                "option_delta": 1.0,
                "motif_type": "enable",
                "motif_type_source": "live_delta_rule",
            },
        }
    )
    stop_summary = stop_live_memory_writer(queue, writer)

    assert stop_summary["writer_exitcode"] == 0
    sqlite_path = memory_dir / "live_memory.sqlite"
    assert sqlite_path.exists()
    with sqlite3.connect(sqlite_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM live_memory_events").fetchone()[0] == 6
        assert connection.execute("SELECT COUNT(*) FROM live_stable_contingencies").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM live_replay_candidates").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM live_contradiction_clusters").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM live_carrier_candidates").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM live_future_option_events").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM live_family_updates").fetchone()[0] == 1

    summary = json.loads((memory_dir / "live_memory_summary.json").read_text(encoding="utf-8"))
    assert summary["events_received"] == 7
    assert summary["events_written"] == 6
    assert summary["queue_stop_received"] is True
    assert summary["event_type_counts"]["stable_contingency"] == 1


def test_live_memory_emit_queue_full_does_not_block(tmp_path: Path) -> None:
    queue = make_live_memory_queue(1)
    queue.put({"event_type": "occupied"})
    system = V6System(
        ToggleEnv(),
        V6Config(
            database_path=str(tmp_path / "queue_full.sqlite"),
            shared_live_memory_mode="write",
        ),
        live_memory_queue=queue,
    )
    try:
        system._emit_live_memory_event(
            "stable_contingency",
            "stable_contingency:test",
            1,
            0.9,
            {
                "key": "test",
                "action": 1,
                "context_signature": "[1]",
                "context_level": 0,
                "transformation_family": 1,
                "support_count": 3,
                "confidence": 0.9,
            },
        )
        assert system.live_memory_events_dropped_queue_full == 1
    finally:
        system.close()


def test_live_memory_read_cache_missing_db(tmp_path: Path) -> None:
    cache = LiveMemoryReadCache(memory_dir=tmp_path / "missing", refresh_steps=5)
    assert cache.refresh(force=True) is False
    assert cache.refresh_failed_count == 1


def test_live_memory_read_cache_loads_rows(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    queue = make_live_memory_queue(10)
    writer = start_live_memory_writer(
        LiveMemoryWriterConfig(memory_dir=str(memory_dir), queue_maxsize=10, batch_size=1, flush_seconds=0.1),
        queue,
    )
    queue.put(
        {
            "event_type": "stable_contingency",
            "event_id": "stable_contingency:cache",
            "global_step": 5,
            "worker_id": "w1",
            "priority": 0.9,
            "payload": {
                "key": "cache",
                "action": 1,
                "context_signature": "[1]",
                "context_level": 0,
                "transformation_family": 2,
                "support_count": 4,
                "confidence": 0.95,
            },
        }
    )
    queue.put(
        {
            "event_type": "high_priority_replay",
            "event_id": "high_priority_replay:cache",
            "global_step": 5,
            "worker_id": "w1",
            "priority": 0.85,
            "payload": {
                "interaction_id": "42",
                "replay_priority": 0.85,
                "reason": "shared",
                "family_id": "2",
                "context_signature": "[1]",
                "action_signature": "1",
            },
        }
    )
    stop_live_memory_writer(queue, writer)

    cache = LiveMemoryReadCache(memory_dir=memory_dir, refresh_steps=5)
    assert cache.refresh(force=True) is True
    assert len(cache.stable_contingencies) == 1
    assert len(cache.replay_candidates) == 1


def test_v6system_readwrite_imports_live_memory_once(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    queue = make_live_memory_queue(10)
    writer = start_live_memory_writer(
        LiveMemoryWriterConfig(memory_dir=str(memory_dir), queue_maxsize=10, batch_size=2, flush_seconds=0.1),
        queue,
    )
    queue.put(
        {
            "event_type": "stable_contingency",
            "event_id": "stable_contingency:imported",
            "global_step": 7,
            "worker_id": "w1",
            "priority": 0.95,
            "payload": {
                "key": "imported",
                "action": 1,
                "context_signature": "[1]",
                "context_level": 0,
                "transformation_family": 3,
                "support_count": 5,
                "confidence": 0.95,
            },
        }
    )
    queue.put(
        {
            "event_type": "high_priority_replay",
            "event_id": "high_priority_replay:imported",
            "global_step": 7,
            "worker_id": "w1",
            "priority": 0.9,
            "payload": {
                "interaction_id": "77",
                "replay_priority": 0.9,
                "reason": "shared_live_memory",
                "family_id": "3",
                "context_signature": "[1]",
                "action_signature": "1",
            },
        }
    )
    queue.put(
        {
            "event_type": "carrier_candidate",
            "event_id": "carrier_candidate:imported",
            "global_step": 7,
            "worker_id": "w1",
            "priority": 0.7,
            "payload": {
                "carrier_signature": "carrier:imported",
                "carrier_source": "object",
                "support_count": 4,
                "linked_family_count": 1,
            },
        }
    )
    stop_live_memory_writer(queue, writer)

    cache = LiveMemoryReadCache(memory_dir=memory_dir, refresh_steps=1)
    assert cache.refresh(force=True) is True
    system = V6System(
        ToggleEnv(),
        V6Config(
            database_path=str(tmp_path / "live_import.sqlite"),
            shared_live_memory_mode="readwrite",
        ),
        live_memory_cache=cache,
    )
    try:
        system._apply_live_memory_cache()
        stable_after_first = system.contingency_learner.stable_contingencies()
        assert len(stable_after_first) == 1
        assert system.live_memory_stable_contingencies_imported == 1
        assert "77" in system.memory_lifecycle.replay_candidates
        assert system.live_memory_replay_candidates_imported == 1
        assert any(candidate.carrier_signature == "carrier:imported" for candidate in system.carrier_tracker.build_candidates())
        assert system.live_memory_carrier_candidates_imported == 1

        system._apply_live_memory_cache()
        stable_after_second = system.contingency_learner.stable_contingencies()
        assert len(stable_after_second) == 1
        assert system.live_memory_stable_contingencies_imported == 1
        assert system.live_memory_replay_candidates_imported == 1
        assert system.live_memory_carrier_candidates_imported == 1
    finally:
        system.close()


def test_interaction_sampling_shared_live_memory_default_is_none() -> None:
    assert InteractionSamplingConfig().shared_live_memory == "none"


def test_v05c_run_sampling_job_shared_live_memory_write_emits_summary(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    memory_dir = tmp_path / "memory"
    db_path = tmp_path / "raw" / "seed_0.sqlite"
    queue = make_live_memory_queue(100)
    writer = start_live_memory_writer(
        LiveMemoryWriterConfig(memory_dir=str(memory_dir), queue_maxsize=100, batch_size=1, flush_seconds=0.1),
        queue,
    )

    class DummySampler:
        reset_count = 0
        reset_unavailable = False

    class DummyEnv:
        def __init__(self, game_id: str, seed: int, env_root=None) -> None:
            self.game_id = game_id
            self.seed = seed
            self.env_root = env_root
            self.reset_count = 0
            self.skipped_terminal_steps = 0

    class DummyGraph:
        def edge_type_counts(self) -> dict[str, int]:
            return {}

        def export_compact_rows(self) -> dict[str, list[dict[str, object]]]:
            return {"nodes": [], "edges": []}

    class DummySystem:
        def __init__(self, env, config, action_sampler=None, live_memory_queue=None, live_memory_cache=None) -> None:
            self.env = env
            self.config = config
            self.graph = DummyGraph()
            self.context_contradictions = type("Tracker", (), {"summary": lambda self: {}})()
            self.carrier_tracker = type("Carrier", (), {"build_candidates": lambda self: []})()
            self.memory_lifecycle = type("Lifecycle", (), {"summary": lambda self: {}, "get_replay_batch": lambda self, limit=1000: []})()
            self.efficiency_tracker = type("Efficiency", (), {"summary": lambda self: {}})()
            self.compact_memory_restore_summary = {}
            self.live_memory_queue = live_memory_queue
            self.live_memory_events_emitted = 0
            self.live_memory_events_dropped_queue_full = 0
            self.live_memory_events_dropped_error = 0
            self.live_memory_refresh_count = 0
            self.live_memory_refresh_failed_count = 0
            self.live_memory_stable_contingencies_imported = 0
            self.live_memory_replay_candidates_imported = 0
            self.live_memory_carrier_candidates_imported = 0
            self.live_memory_family_updates_imported = 0
            self.live_memory_contradiction_clusters_loaded = 0
            self.live_memory_future_option_events_loaded = 0

        def run(self, steps: int) -> None:
            if self.live_memory_queue is not None:
                self.live_memory_queue.put_nowait(
                    {
                        "event_type": "stable_contingency",
                        "event_id": "stable_contingency:dummy",
                        "global_step": 1,
                        "worker_id": str(self.config.live_memory_worker_id or "worker"),
                        "priority": 0.95,
                        "payload": {
                            "key": "dummy",
                            "action": 1,
                            "context_signature": "[1]",
                            "context_level": 0,
                            "transformation_family": 1,
                            "support_count": 3,
                            "confidence": 0.95,
                        },
                    }
                )
                self.live_memory_events_emitted = 1
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db_path) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS interactions (id INTEGER PRIMARY KEY)")
                connection.commit()

        def close(self) -> None:
            pass

        def adaptive_context_summary(self) -> dict[str, object]:
            return {"adaptive_context_expansion_enabled": False, "base_context_depth": 1, "max_context_depth": 1}

    def fake_write_sampling_metadata(path, **values) -> None:
        captured["metadata"] = values

    monkeypatch.setattr(interaction_sampling, "make_sampler", lambda name, seed: DummySampler())
    monkeypatch.setattr(interaction_sampling, "ArcGridEnvironment", DummyEnv)
    monkeypatch.setattr(interaction_sampling, "V6System", DummySystem)
    monkeypatch.setattr(interaction_sampling, "_write_sampling_metadata", fake_write_sampling_metadata)

    try:
        result = interaction_sampling._run_sampling_job(
            {
                "game": "tt01",
                "sampler_name": "random_baseline",
                "seed": 0,
                "steps": 10,
                "horizon": 3,
                "context_depth": 1,
                "commit_steps": 5,
                "db_path": str(db_path),
                "env_root": None,
                "memory_output_dir": str(memory_dir),
                "shared_live_memory": "write",
                "live_memory_queue": queue,
            }
        )
    finally:
        stop_summary = stop_live_memory_writer(queue, writer)

    assert result == {"legacy_future_effects_removed": True}
    assert stop_summary["writer_exitcode"] == 0
    assert db_path.exists()
    assert (memory_dir / "live_memory_summary.json").exists()
    summary = json.loads((memory_dir / "live_memory_summary.json").read_text(encoding="utf-8"))
    assert summary["events_written"] >= 1
    assert captured["metadata"]["shared_live_memory_mode"] == "write"
    assert captured["metadata"]["live_memory_events_emitted"] == 1


def test_v05c_sampling_db_ready_accepts_fast_postprocessing_without_future_effects(tmp_path) -> None:
    db_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE interactions (id INTEGER PRIMARY KEY);
            CREATE TABLE deltas (id INTEGER PRIMARY KEY);
            CREATE TABLE contingencies (id INTEGER PRIMARY KEY);
            CREATE TABLE prediction_results (id INTEGER PRIMARY KEY);
            CREATE TABLE sampling_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO sampling_metadata (key, value) VALUES (?, ?)",
            [
                ("fast_postprocessing_enabled", "true"),
                ("future_effects_postprocessing_skipped", "true"),
            ],
        )
        connection.commit()

    assert interaction_sampling._sampling_db_ready(db_path) is True


def test_compute_run_diagnostics_accepts_fast_postprocessing_without_future_effects(tmp_path) -> None:
    db_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY,
                action INTEGER,
                delta_id INTEGER
            );
            CREATE TABLE deltas (
                id INTEGER PRIMARY KEY,
                changed_cells INTEGER,
                colors_added TEXT,
                colors_removed TEXT,
                dx REAL,
                dy REAL,
                changed_positions TEXT
            );
            CREATE TABLE transformation_families (
                id INTEGER PRIMARY KEY,
                centroid_vector TEXT,
                support_count INTEGER
            );
            CREATE TABLE contingencies (
                id INTEGER PRIMARY KEY,
                context_level INTEGER,
                context_signature TEXT,
                action INTEGER,
                transformation_family INTEGER,
                support_count INTEGER,
                confidence REAL
            );
            CREATE TABLE prediction_results (
                interaction_id INTEGER PRIMARY KEY,
                context_level INTEGER,
                action INTEGER,
                actual_family INTEGER,
                prediction_error INTEGER,
                episode_id INTEGER
            );
            CREATE TABLE sampling_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.execute("INSERT INTO interactions (id, action, delta_id) VALUES (1, 2, 1)")
        connection.execute(
            "INSERT INTO deltas (id, changed_cells, colors_added, colors_removed, dx, dy, changed_positions) VALUES (1, 1, '[]', '[]', 0.0, 0.0, '[]')"
        )
        connection.execute(
            "INSERT INTO transformation_families (id, centroid_vector, support_count) VALUES (7, '[1,0,0]', 3)"
        )
        connection.execute(
            """
            INSERT INTO contingencies (
                id, context_level, context_signature, action, transformation_family, support_count, confidence
            ) VALUES (1, 0, '[2]', 2, 7, 3, 0.9)
            """
        )
        connection.execute(
            "INSERT INTO prediction_results (interaction_id, context_level, action, actual_family, prediction_error, episode_id) VALUES (1, 0, 2, 7, 0, 0)"
        )
        connection.executemany(
            "INSERT INTO sampling_metadata (key, value) VALUES (?, ?)",
            [
                ("fast_postprocessing_enabled", "true"),
                ("future_effects_postprocessing_skipped", "true"),
            ],
        )
        connection.commit()

    result = compute_run_diagnostics(db_path, game="tt01", seed=0, steps=1, horizon=2)

    assert result["run_status"] == "ok"
    assert result["failure_reason"] == ""
    assert result["legacy_future_effects_removed"] is True
    assert result["future_effect_metrics_interpretable"] is False
    assert result["future_effect_count"] is None
    assert result["non_preserve_count"] is None


def test_incremental_sidecar_fold_deletes_large_json_sidecars(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "sampling_v05c" / "tt01" / "mixed" / "steps_10"
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_path = raw_dir / "seed_0.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE transformation_families (id INTEGER PRIMARY KEY, centroid_vector TEXT, support_count INTEGER);
            INSERT INTO transformation_families (id, centroid_vector, support_count) VALUES (1, '[1,0,0]', 3);
            """
        )
        connection.commit()
    db_path.with_name("live_graph_compact.json").write_text(
        json.dumps(
            {
                "nodes": [{"node_id": "carrier:c1", "node_type": "carrier", "canonical_key": "c1", "support_count": 1}],
                "edges": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    db_path.with_name("carrier_candidates.json").write_text(
        json.dumps(
            [
                {
                    "carrier_id": "c1",
                    "carrier_signature": "c1",
                    "carrier_source": "object",
                    "support_count": 3,
                    "distinct_family_count": 1,
                    "family_id": 1,
                    "context_signature": "[1,2]",
                    "status": "emergent_carrier",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    db_path.with_name("context_contradictions.json").write_text("{}", encoding="utf-8")
    db_path.with_name("memory_lifecycle_summary.json").write_text("{}", encoding="utf-8")

    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    summary = fold_sampling_job_sidecars_into_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
        delete_after_merge=True,
    )

    assert summary["graph_live_exports_ingested"] == 1
    assert summary["carrier_candidates_added"] == 1
    assert not db_path.with_name("live_graph_compact.json").exists()
    assert not db_path.with_name("carrier_candidates.json").exists()
    assert not db_path.with_name("context_contradictions.json").exists()
    assert not db_path.with_name("memory_lifecycle_summary.json").exists()
    assert db_path.exists()
    memory_summary = load_memory_summary(memory_dir / "memory_summary.json")
    assert memory_summary["incremental_sidecar_fold"]["db_path"] == str(db_path)


def test_parallel_sidecar_fold_shards_then_merges_into_main_memory(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "sampling_v05c" / "tt01" / "mixed" / "steps_10"
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_path = raw_dir / "seed_0.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE transformation_families (id INTEGER PRIMARY KEY, centroid_vector TEXT, support_count INTEGER);
            INSERT INTO transformation_families (id, centroid_vector, support_count) VALUES (1, '[1,0,0]', 3);
            """
        )
        connection.commit()
    db_path.with_name("live_graph_compact.json").write_text(
        json.dumps(
            {
                "nodes": [{"node_id": "carrier:c1", "node_type": "carrier", "canonical_key": "c1", "support_count": 1}],
                "edges": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    db_path.with_name("carrier_candidates.json").write_text(
        json.dumps(
            [
                {
                    "carrier_id": "c1",
                    "carrier_signature": "c1",
                    "carrier_source": "object",
                    "support_count": 3,
                    "distinct_family_count": 1,
                    "family_id": 1,
                    "context_signature": "[1,2]",
                    "status": "emergent_carrier",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    db_path.with_name("context_contradictions.json").write_text("{}", encoding="utf-8")
    db_path.with_name("memory_lifecycle_summary.json").write_text("{}", encoding="utf-8")

    shard_dir = tmp_path / "memory_shard"
    shard_summary = fold_sampling_job_sidecars_into_compact_memory_shard(
        db_path=db_path,
        shard_memory_dir=shard_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
        delete_after_merge=True,
    )
    assert shard_summary["graph_live_exports_ingested"] == 1
    assert shard_summary["carrier_candidates_added"] == 1
    assert not db_path.with_name("live_graph_compact.json").exists()
    main_memory_dir = tmp_path / "memory_main"
    ensure_memory_layout(main_memory_dir)
    merged = merge_compact_memory_shards_into_main(
        memory_dir=main_memory_dir,
        shard_dirs=[shard_dir],
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
    )
    assert merged["carrier_candidate_count"] >= 1
    memory_summary = load_memory_summary(main_memory_dir / "memory_summary.json")
    assert memory_summary["parallel_sidecar_shard_merge"]["merged_shard_count"] == 1


def test_parallel_compact_memory_shard_merge_reduces_multiple_shards(tmp_path) -> None:
    shard_a = tmp_path / "memory_shard_a"
    shard_b = tmp_path / "memory_shard_b"
    ensure_memory_layout(shard_a)
    ensure_memory_layout(shard_b)

    with sqlite3.connect(shard_a / "current_state.sqlite") as connection:
        connection.execute(
            """
            INSERT INTO carrier_candidates (
                carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                first_seen_global_step, last_seen_global_step, stability_score, is_emergent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ca", "carrier:a", "object", 3, 1, 1, 10, 0.5, 1),
        )
        connection.commit()
    with sqlite3.connect(shard_b / "current_state.sqlite") as connection:
        connection.execute(
            """
            INSERT INTO carrier_candidates (
                carrier_id, carrier_signature, carrier_source, support_count, linked_family_count,
                first_seen_global_step, last_seen_global_step, stability_score, is_emergent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("cb", "carrier:b", "object", 4, 1, 1, 10, 0.6, 1),
        )
        connection.commit()

    main_memory_dir = tmp_path / "memory_main_parallel"
    ensure_memory_layout(main_memory_dir)
    merged = merge_compact_memory_shards_into_main(
        memory_dir=main_memory_dir,
        shard_dirs=[shard_a, shard_b],
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
        parallel_workers=2,
    )

    assert merged["carrier_candidate_count"] >= 2
    memory_summary = load_memory_summary(main_memory_dir / "memory_summary.json")
    assert memory_summary["parallel_sidecar_shard_merge"]["merged_shard_count"] == 1
    assert memory_summary["parallel_sidecar_shard_merge"]["parallel_workers"] == 2


def test_v05c_resolve_scope_clamps_train_and_test_seeds_to_available_set() -> None:
    config = interaction_sampling.resolve_interaction_sampling_scope(
        InteractionSamplingConfig(
            games=("tt01",),
            samplers=("random_baseline",),
            seeds=(0,),
            train_seeds=(0, 1),
            test_seed=2,
            steps=10,
            horizon=3,
            context_depth=2,
            workers=1,
        )
    )

    assert config.train_seeds == (0,)
    assert config.test_seed == 0


def _build_v06_fixture(root) -> str:
    backend = ParquetStorageBackend(root=root, game="va02", sampler="mixed", seed=0, steps=8, batch_size=50)
    interactions = []
    deltas = []
    rows = [
        (_obs(0), _obs(1), 4, "NOT_FINISHED", "neutral"),
        (_obs(1), _obs(2), 4, "NOT_FINISHED", "neutral"),
        (_obs(2), _obs(2), 4, "WIN", "positive"),
        (_obs(0), _obs(1), 4, "NOT_FINISHED", "neutral"),
        (_obs(1), _obs(2), 4, "NOT_FINISHED", "neutral"),
        (_obs(2), _obs(3), 4, "GAME_OVER", "negative"),
    ]
    for index, (before, after, action, outcome_state, outcome_polarity) in enumerate(rows, start=1):
        interactions.append(
            {
                "id": index,
                "timestamp": index,
                "observation_before": encode_array(before),
                "action": action,
                "observation_after": encode_array(after),
                "delta_id": index,
                "outcome_state": outcome_state,
                "outcome_polarity": outcome_polarity,
                "is_terminal_outcome": int(outcome_state in {"WIN", "GAME_OVER"}),
                "is_success_outcome": int(outcome_state == "WIN"),
                "is_failure_outcome": int(outcome_state == "GAME_OVER"),
            }
        )
        deltas.append(
            {
                "id": index,
                "changed_cells": int(np.count_nonzero(before != after)),
                "changed_positions": json.dumps([]),
                "colors_added": json.dumps([]),
                "colors_removed": json.dumps([]),
                "centroid_before_x": 0.0,
                "centroid_before_y": 0.0,
                "centroid_after_x": 1.0 if not np.array_equal(before, after) else 0.0,
                "centroid_after_y": 0.0,
                "dx": 1.0 if not np.array_equal(before, after) else 0.0,
                "dy": 0.0,
            }
        )
    backend.write_interactions(interactions)
    backend.write_deltas(deltas)
    backend.write_run_summary({"game": "va02", "sampler": "mixed", "seed": 0, "steps": 8})
    backend.finalize()
    return str(root)


def _build_v06_manifest_fixture(root) -> str:
    for game, sampler, seed in (
        ("gr01", "no_change_avoidance", 0),
        ("gr01", "random_baseline", 1),
        ("va02", "novelty_delta", 0),
    ):
        backend = ParquetStorageBackend(root=root, game=game, sampler=sampler, seed=seed, steps=6, batch_size=2)
        rows = [
            (_obs(0), _obs(1), 4),
            (_obs(1), _obs(2), 4),
            (_obs(2), _obs(2), 4),
            (_obs(0), _obs(1), 4),
        ]
        interactions = []
        deltas = []
        for index, (before, after, action) in enumerate(rows, start=1):
            interactions.append(
                {
                    "id": index,
                    "timestamp": index,
                    "observation_before": encode_array(before),
                    "action": action,
                    "observation_after": encode_array(after),
                    "delta_id": index,
                }
            )
            deltas.append(
                {
                    "id": index,
                    "changed_cells": int(np.count_nonzero(before != after)),
                    "changed_positions": json.dumps([]),
                    "colors_added": json.dumps([]),
                    "colors_removed": json.dumps([]),
                    "centroid_before_x": 0.0,
                    "centroid_before_y": 0.0,
                    "centroid_after_x": 1.0 if not np.array_equal(before, after) else 0.0,
                    "centroid_after_y": 0.0,
                    "dx": 1.0 if not np.array_equal(before, after) else 0.0,
                    "dy": 0.0,
                }
            )
        backend.write_interactions(interactions)
        backend.write_deltas(deltas)
        backend.write_run_summary({"game": game, "sampler": sampler, "seed": seed, "steps": 6})
        backend.finalize()
    return str(root)


def _build_v07_fixture(root) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "contingency_id": "m1-gr01-random-0001",
            "game_id": "gr01",
            "sampler_scope": "random_baseline",
            "context_signature": [],
            "action": 1,
            "outcome_signature": "blocked_no_change",
            "support_count": 10,
            "total_count": 10,
            "prediction_accuracy": 1.0,
            "prediction_error_rate": 0.0,
            "entropy": 0.0,
            "confidence": 1.0,
            "first_seen_step": 0,
            "last_seen_step": 10,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "block_candidate",
            "discovered": True,
            "notes": {"seed_count": 1, "outcome_counts": {"blocked_no_change": 10}},
        },
        {
            "contingency_id": "m1-fs02-random-0002",
            "game_id": "fs02",
            "sampler_scope": "random_baseline",
            "context_signature": ["a1|opreserve"],
            "action": 1,
            "outcome_signature": "blocked_no_change",
            "support_count": 9,
            "total_count": 9,
            "prediction_accuracy": 1.0,
            "prediction_error_rate": 0.0,
            "entropy": 0.0,
            "confidence": 1.0,
            "first_seen_step": 0,
            "last_seen_step": 10,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "block_candidate",
            "discovered": True,
            "notes": {"seed_count": 1, "outcome_counts": {"blocked_no_change": 9}},
        },
        {
            "contingency_id": "m1-va02-random-0003",
            "game_id": "va02",
            "sampler_scope": "random_baseline",
            "context_signature": [],
            "action": 3,
            "outcome_signature": "change",
            "support_count": 8,
            "total_count": 10,
            "prediction_accuracy": 0.8,
            "prediction_error_rate": 0.2,
            "entropy": 0.5,
            "confidence": 0.8,
            "first_seen_step": 0,
            "last_seen_step": 10,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "change_candidate",
            "discovered": True,
            "notes": {"seed_count": 1, "outcome_counts": {"change": 8, "large_change": 2}},
        },
        {
            "contingency_id": "m1-mo01-random-0004",
            "game_id": "mo01",
            "sampler_scope": "random_baseline",
            "context_signature": ["a4|ochange"],
            "action": 4,
            "outcome_signature": "position_like_change",
            "support_count": 12,
            "total_count": 12,
            "prediction_accuracy": 1.0,
            "prediction_error_rate": 0.0,
            "entropy": 0.0,
            "confidence": 1.0,
            "first_seen_step": 0,
            "last_seen_step": 10,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "change_candidate",
            "discovered": True,
            "notes": {"seed_count": 1, "outcome_counts": {"position_like_change": 12}},
        },
        {
            "contingency_id": "m1-mo01-random-0005",
            "game_id": "mo01",
            "sampler_scope": "random_baseline",
            "context_signature": ["a5|oterminal"],
            "action": 5,
            "outcome_signature": "terminal_transition",
            "support_count": 6,
            "total_count": 6,
            "prediction_accuracy": 1.0,
            "prediction_error_rate": 0.0,
            "entropy": 0.0,
            "confidence": 1.0,
            "first_seen_step": 0,
            "last_seen_step": 10,
            "example_episode_ids": [1],
            "terminal_effect_candidate": True,
            "future_option_motif_candidate": "terminate_candidate",
            "discovered": True,
            "notes": {"seed_count": 1, "outcome_counts": {"terminal_transition": 6}},
        },
    ]
    report = {"validation": {"scientific_conclusion": "m1_strong"}, "report": {"tested_theory_components": {"M0": "yes", "M1": "yes"}}}
    (root / "contingencies.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (root / "v06_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return root


def _v08_fixture_records():
    from v6.role_candidates_v08 import M1SupportRecord, M2FamilyRecord

    families = [
        M2FamilyRecord(
            family_id="m2-a",
            family_label_candidate="blocked_no_change_family_candidate",
            games_present=("g1", "g2"),
            samplers_present=("s1",),
            contingency_ids=("m1-a1", "m1-a2"),
            support_count=20,
            mean_prediction_accuracy=0.92,
            mean_context_lift=0.12,
            dominant_outcome_signature="blocked_no_change",
            outcome_signature_distribution={"blocked_no_change": 18, "preserve_no_change": 2},
            motif_candidate_distribution={"block_candidate": 18, "preserve_candidate": 2},
            family_coherence=0.91,
            compression_ratio=2.0,
            cross_game_presence=2,
            stable=True,
            examples=(),
            notes={},
        ),
        M2FamilyRecord(
            family_id="m2-b",
            family_label_candidate="blocked_no_change_family_candidate",
            games_present=("g2", "g3"),
            samplers_present=("s1",),
            contingency_ids=("m1-b1", "m1-b2"),
            support_count=22,
            mean_prediction_accuracy=0.90,
            mean_context_lift=0.11,
            dominant_outcome_signature="blocked_no_change",
            outcome_signature_distribution={"blocked_no_change": 19, "preserve_no_change": 3},
            motif_candidate_distribution={"block_candidate": 19, "preserve_candidate": 3},
            family_coherence=0.89,
            compression_ratio=2.0,
            cross_game_presence=2,
            stable=True,
            examples=(),
            notes={},
        ),
        M2FamilyRecord(
            family_id="m2-c",
            family_label_candidate="terminal_family_candidate",
            games_present=("g1",),
            samplers_present=("s1",),
            contingency_ids=("m1-c1",),
            support_count=9,
            mean_prediction_accuracy=0.84,
            mean_context_lift=0.05,
            dominant_outcome_signature="terminal_transition",
            outcome_signature_distribution={"terminal_transition": 9},
            motif_candidate_distribution={"terminate_candidate": 9},
            family_coherence=0.95,
            compression_ratio=1.0,
            cross_game_presence=1,
            stable=True,
            examples=(),
            notes={},
        ),
    ]
    m1_support = {
        "m1-a1": M1SupportRecord("m1-a1", "g1", "s1", 1, "blocked_no_change", 0.95, 0.0, 10, 10, "block_candidate", ("a1|ob", "a2|ob")),
        "m1-a2": M1SupportRecord("m1-a2", "g2", "s1", 2, "blocked_no_change", 0.89, 0.1, 10, 11, "block_candidate", ("a1|ob", "a2|ob")),
        "m1-b1": M1SupportRecord("m1-b1", "g2", "s1", 1, "blocked_no_change", 0.91, 0.0, 11, 11, "block_candidate", ("a1|ob", "a2|ob")),
        "m1-b2": M1SupportRecord("m1-b2", "g3", "s1", 2, "preserve_no_change", 0.87, 0.2, 11, 12, "preserve_candidate", ("a1|ob", "a2|ob")),
        "m1-c1": M1SupportRecord("m1-c1", "g1", "s1", 5, "terminal_transition", 0.84, 0.0, 9, 9, "terminate_candidate", ("a5|ot",)),
    }
    return families, m1_support


def _write_v08_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_dir = tmp_path / "v07"
    m1_dir = tmp_path / "v06"
    input_dir.mkdir()
    m1_dir.mkdir()
    families, m1_support = _v08_fixture_records()
    family_rows = [
        {
            **family.__dict__,
            "games_present": list(family.games_present),
            "samplers_present": list(family.samplers_present),
            "contingency_ids": list(family.contingency_ids),
            "examples": list(family.examples),
        }
        for family in families
    ]
    contingency_rows = [
        {
            "contingency_id": record.contingency_id,
            "game_id": record.game_id,
            "sampler_scope": record.sampler_scope,
            "action": record.action,
            "outcome_signature": record.outcome_signature,
            "support_count": record.support_count,
            "total_count": record.total_count,
            "prediction_accuracy": record.prediction_accuracy,
            "prediction_error_rate": 1.0 - record.prediction_accuracy,
            "entropy": record.entropy,
            "confidence": record.prediction_accuracy,
            "first_seen_step": 0,
            "last_seen_step": 1,
            "example_episode_ids": [1],
            "terminal_effect_candidate": record.outcome_signature == "terminal_transition",
            "future_option_motif_candidate": record.future_option_motif_candidate,
            "discovered": True,
            "context_signature": list(record.context_signature),
            "notes": {},
        }
        for record in m1_support.values()
    ]
    (input_dir / "m2_families.json").write_text(json.dumps(family_rows), encoding="utf-8")
    (input_dir / "v07_report.json").write_text(json.dumps({"report": {"families_by_game": {"g1": {}, "g2": {}, "g3": {}}}}), encoding="utf-8")
    (m1_dir / "contingencies.json").write_text(json.dumps(contingency_rows), encoding="utf-8")
    (m1_dir / "v06_report.json").write_text(json.dumps({"report": {"games": ["g1", "g2", "g3"]}}), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist(
            [{"contingency_id": "m1-a1", "family_id": "m2-a", "family_label_candidate": "blocked_no_change_family_candidate", "stable": True}]
        ),
        input_dir / "contingency_to_family.parquet",
        compression="zstd",
    )
    pq.write_table(pa.Table.from_pylist([{"node_id": "m2-a", "node_type": "m2_family"}]), input_dir / "m2_graph_nodes.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist([{"edge_type": "member_of", "source_id": "m1-a1", "target_id": "m2-a"}]), input_dir / "m2_graph_edges.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist([{"game_id": "g1", "sampler": "s1", "seed": 0, "episode_id": 1}]), m1_dir / "episode_summaries.parquet", compression="zstd")
    (tmp_path / "game_set.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "games": ["g1", "g2", "g3"],
                "families": {"movement": ["g1"], "toggle_switch": ["g2"], "push_pull": ["g3"]},
                "purpose": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    return input_dir, m1_dir


def _write_v08c_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_dir = tmp_path / "v07"
    m1_dir = tmp_path / "v06"
    input_dir.mkdir()
    m1_dir.mkdir()

    contingency_rows = [
        {
            "contingency_id": "m1-g1-s1-0001",
            "game_id": "g1",
            "sampler_scope": "s1",
            "context_signature": ["a1|ox", "a2|oy"],
            "action": 1,
            "outcome_signature": "position_like_change",
            "support_count": 10,
            "total_count": 10,
            "prediction_accuracy": 0.95,
            "prediction_error_rate": 0.05,
            "entropy": 0.0,
            "confidence": 0.95,
            "first_seen_step": 0,
            "last_seen_step": 1,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "change_candidate",
            "discovered": True,
            "notes": {},
        },
        {
            "contingency_id": "m1-g2-s1-0002",
            "game_id": "g2",
            "sampler_scope": "s1",
            "context_signature": ["a1|ox", "a2|oy"],
            "action": 2,
            "outcome_signature": "position_like_change",
            "support_count": 9,
            "total_count": 10,
            "prediction_accuracy": 0.90,
            "prediction_error_rate": 0.10,
            "entropy": 0.1,
            "confidence": 0.90,
            "first_seen_step": 0,
            "last_seen_step": 1,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "change_candidate",
            "discovered": True,
            "notes": {},
        },
        {
            "contingency_id": "m1-g3-s1-0003",
            "game_id": "g3",
            "sampler_scope": "s1",
            "context_signature": ["a1|ox", "a2|oy"],
            "action": 5,
            "outcome_signature": "position_like_change",
            "support_count": 8,
            "total_count": 10,
            "prediction_accuracy": 0.80,
            "prediction_error_rate": 0.20,
            "entropy": 0.2,
            "confidence": 0.80,
            "first_seen_step": 0,
            "last_seen_step": 1,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "change_candidate",
            "discovered": True,
            "notes": {},
        },
        {
            "contingency_id": "m1-g4-s1-0004",
            "game_id": "g4",
            "sampler_scope": "s1",
            "context_signature": ["a1|ox", "a2|oy"],
            "action": 6,
            "outcome_signature": "position_like_change",
            "support_count": 8,
            "total_count": 10,
            "prediction_accuracy": 0.82,
            "prediction_error_rate": 0.18,
            "entropy": 0.2,
            "confidence": 0.82,
            "first_seen_step": 0,
            "last_seen_step": 1,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "change_candidate",
            "discovered": True,
            "notes": {},
        },
        {
            "contingency_id": "m1-g1-s1-0005",
            "game_id": "g1",
            "sampler_scope": "s1",
            "context_signature": ["a1|ob", "a2|ob"],
            "action": 1,
            "outcome_signature": "blocked_no_change",
            "support_count": 7,
            "total_count": 8,
            "prediction_accuracy": 0.875,
            "prediction_error_rate": 0.125,
            "entropy": 0.2,
            "confidence": 0.875,
            "first_seen_step": 0,
            "last_seen_step": 1,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "block_candidate",
            "discovered": True,
            "notes": {},
        },
        {
            "contingency_id": "m1-g4-s1-0006",
            "game_id": "g4",
            "sampler_scope": "s1",
            "context_signature": ["a1|ob", "a2|ob"],
            "action": 2,
            "outcome_signature": "blocked_no_change",
            "support_count": 7,
            "total_count": 8,
            "prediction_accuracy": 0.875,
            "prediction_error_rate": 0.125,
            "entropy": 0.2,
            "confidence": 0.875,
            "first_seen_step": 0,
            "last_seen_step": 1,
            "example_episode_ids": [1],
            "terminal_effect_candidate": False,
            "future_option_motif_candidate": "block_candidate",
            "discovered": True,
            "notes": {},
        },
    ]
    (m1_dir / "contingencies.json").write_text(json.dumps(contingency_rows), encoding="utf-8")
    (m1_dir / "v06_report.json").write_text(json.dumps({"report": {"games": ["g1", "g2", "g3", "g4"]}}), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{"game_id": "g1", "sampler": "s1", "seed": 0, "episode_id": 1}]), m1_dir / "episode_summaries.parquet", compression="zstd")

    family_rows = [
        {
            "family_id": "m2-a",
            "family_label_candidate": "position_like_change_family_candidate",
            "games_present": ["g1", "g2", "g3", "g4"],
            "samplers_present": ["s1"],
            "contingency_ids": ["m1-g1-s1-0001", "m1-g2-s1-0002", "m1-g3-s1-0003", "m1-g4-s1-0004"],
            "support_count": 35,
            "mean_prediction_accuracy": 0.8675,
            "mean_context_lift": 0.10,
            "dominant_outcome_signature": "position_like_change",
            "outcome_signature_distribution": {"position_like_change": 4},
            "motif_candidate_distribution": {"change_candidate": 4},
            "family_coherence": 0.85,
            "compression_ratio": 2.0,
            "cross_game_presence": 4,
            "stable": True,
            "examples": [],
            "notes": {"mean_pairwise_similarity": 0.82},
        },
        {
            "family_id": "m2-b",
            "family_label_candidate": "blocked_no_change_family_candidate",
            "games_present": ["g1", "g4"],
            "samplers_present": ["s1"],
            "contingency_ids": ["m1-g1-s1-0005", "m1-g4-s1-0006"],
            "support_count": 14,
            "mean_prediction_accuracy": 0.875,
            "mean_context_lift": 0.08,
            "dominant_outcome_signature": "blocked_no_change",
            "outcome_signature_distribution": {"blocked_no_change": 2},
            "motif_candidate_distribution": {"block_candidate": 2},
            "family_coherence": 0.92,
            "compression_ratio": 1.0,
            "cross_game_presence": 2,
            "stable": True,
            "examples": [],
            "notes": {"mean_pairwise_similarity": 0.90},
        },
    ]
    (input_dir / "m2_families.json").write_text(json.dumps(family_rows), encoding="utf-8")
    (input_dir / "v07_report.json").write_text(json.dumps({"report": {"families_by_game": {"g1": {}, "g2": {}, "g3": {}, "g4": {}}}}), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"contingency_id": "m1-g1-s1-0001", "family_id": "m2-a", "family_label_candidate": "position_like_change_family_candidate", "stable": True},
                {"contingency_id": "m1-g2-s1-0002", "family_id": "m2-a", "family_label_candidate": "position_like_change_family_candidate", "stable": True},
                {"contingency_id": "m1-g3-s1-0003", "family_id": "m2-a", "family_label_candidate": "position_like_change_family_candidate", "stable": True},
                {"contingency_id": "m1-g4-s1-0004", "family_id": "m2-a", "family_label_candidate": "position_like_change_family_candidate", "stable": True},
                {"contingency_id": "m1-g1-s1-0005", "family_id": "m2-b", "family_label_candidate": "blocked_no_change_family_candidate", "stable": True},
                {"contingency_id": "m1-g4-s1-0006", "family_id": "m2-b", "family_label_candidate": "blocked_no_change_family_candidate", "stable": True},
            ]
        ),
        input_dir / "contingency_to_family.parquet",
        compression="zstd",
    )
    pq.write_table(pa.Table.from_pylist([{"node_id": "m2-a", "node_type": "m2_family"}, {"node_id": "m2-b", "node_type": "m2_family"}]), input_dir / "m2_graph_nodes.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist([{"edge_type": "similar_to", "source_id": "m2-a", "target_id": "m2-b", "similarity": 0.5}]), input_dir / "m2_graph_edges.parquet", compression="zstd")

    manifest = tmp_path / "game_set_v08c.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "fixture_v08c",
                "games": ["g1", "g2", "g3", "g4"],
                "families": {
                    "collection": ["g1"],
                    "push_crate": ["g2"],
                    "switch_unlock": ["g3"],
                    "teleport_warp": ["g4"],
                },
                "purpose": "synthetic v08c expansion",
            }
        ),
        encoding="utf-8",
    )
    return input_dir, m1_dir, manifest


def _write_v09_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    m3_dir = tmp_path / "m3"
    m2_dir = tmp_path / "m2"
    m1_dir = tmp_path / "m1"
    m3_dir.mkdir()
    m2_dir.mkdir()
    m1_dir.mkdir()

    neighborhoods = [
        {
            "family_id": "f1",
            "family_label_candidate": "blocked_no_change_family_candidate",
            "games_present": ["g1"],
            "game_families_present": ["famA"],
            "support_count": 10,
            "family_coherence": 0.9,
            "mean_prediction_accuracy": 0.9,
            "mean_context_lift": 0.1,
            "dominant_outcome_signature": "blocked_no_change",
            "dominant_motif_candidate": "block_candidate",
            "coarse_features": {"c1": 1.0},
            "directional_features": {"d1": 0.9},
            "future_option_features": {"f1": 0.9},
            "local_motif_features": {"l1": 0.8},
            "temporal_effect_features": {"no_change_rate": 1.0, "position_change_rate": 0.0},
            "incoming_edge_profile": {"blocked_no_change_family_candidate": 2},
            "outgoing_edge_profile": {"blocked_no_change_family_candidate": 2},
            "examples": [],
        },
        {
            "family_id": "f2",
            "family_label_candidate": "blocked_no_change_family_candidate",
            "games_present": ["g2"],
            "game_families_present": ["famA"],
            "support_count": 10,
            "family_coherence": 0.9,
            "mean_prediction_accuracy": 0.9,
            "mean_context_lift": 0.1,
            "dominant_outcome_signature": "blocked_no_change",
            "dominant_motif_candidate": "block_candidate",
            "coarse_features": {"c1": 0.95},
            "directional_features": {"d1": 0.88},
            "future_option_features": {"f1": 0.91},
            "local_motif_features": {"l1": 0.81},
            "temporal_effect_features": {"no_change_rate": 0.98, "position_change_rate": 0.0},
            "incoming_edge_profile": {"blocked_no_change_family_candidate": 2},
            "outgoing_edge_profile": {"blocked_no_change_family_candidate": 2},
            "examples": [],
        },
        {
            "family_id": "f3",
            "family_label_candidate": "position_like_change_family_candidate",
            "games_present": ["g3"],
            "game_families_present": ["famB"],
            "support_count": 10,
            "family_coherence": 0.92,
            "mean_prediction_accuracy": 0.92,
            "mean_context_lift": 0.2,
            "dominant_outcome_signature": "position_like_change",
            "dominant_motif_candidate": "change_candidate",
            "coarse_features": {"c1": -1.0},
            "directional_features": {"d1": -0.8},
            "future_option_features": {"f1": -0.9},
            "local_motif_features": {"l1": -0.7},
            "temporal_effect_features": {"no_change_rate": 0.0, "position_change_rate": 1.0},
            "incoming_edge_profile": {"position_like_change_family_candidate": 2},
            "outgoing_edge_profile": {"position_like_change_family_candidate": 2},
            "examples": [],
        },
        {
            "family_id": "f4",
            "family_label_candidate": "position_like_change_family_candidate",
            "games_present": ["g4"],
            "game_families_present": ["famB"],
            "support_count": 10,
            "family_coherence": 0.92,
            "mean_prediction_accuracy": 0.92,
            "mean_context_lift": 0.2,
            "dominant_outcome_signature": "position_like_change",
            "dominant_motif_candidate": "change_candidate",
            "coarse_features": {"c1": -0.95},
            "directional_features": {"d1": -0.78},
            "future_option_features": {"f1": -0.88},
            "local_motif_features": {"l1": -0.68},
            "temporal_effect_features": {"no_change_rate": 0.0, "position_change_rate": 0.98},
            "incoming_edge_profile": {"position_like_change_family_candidate": 2},
            "outgoing_edge_profile": {"position_like_change_family_candidate": 2},
            "examples": [],
        },
    ]
    pq.write_table(pa.Table.from_pylist(neighborhoods), m3_dir / "role_neighborhoods.parquet", compression="zstd")

    roles = [
        {
            "role_id": "r1",
            "role_label_candidate": "blocker_candidate",
            "member_family_ids": ["f1", "f2"],
            "games_present": ["g1", "g2"],
            "game_families_present": ["famA"],
            "support_count": 20,
            "cross_game_support": 2,
            "cross_game_family_support": 1,
            "role_consistency_score": 0.9,
            "mean_neighborhood_similarity": 0.9,
            "mean_family_coherence": 0.9,
            "dominant_motif_profile": {"block_candidate": 1.0},
            "incoming_edge_profile": {},
            "outgoing_edge_profile": {},
            "future_option_effect_profile": {"blocked_no_change": 1.0},
            "transfer_readiness_score": 0.9,
            "label_evidence": {},
            "examples": [],
            "status": "stable",
            "notes": {},
        },
        {
            "role_id": "r2",
            "role_label_candidate": "movement_controller_candidate",
            "member_family_ids": ["f3", "f4"],
            "games_present": ["g3", "g4"],
            "game_families_present": ["famB"],
            "support_count": 20,
            "cross_game_support": 2,
            "cross_game_family_support": 1,
            "role_consistency_score": 0.9,
            "mean_neighborhood_similarity": 0.9,
            "mean_family_coherence": 0.9,
            "dominant_motif_profile": {"change_candidate": 1.0},
            "incoming_edge_profile": {},
            "outgoing_edge_profile": {},
            "future_option_effect_profile": {"position_like_change": 1.0},
            "transfer_readiness_score": 0.9,
            "label_evidence": {},
            "examples": [],
            "status": "stable",
            "notes": {},
        },
    ]
    (m3_dir / "m3_role_candidates.json").write_text(json.dumps(roles), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{"role_id": "r1", "family_id": "f1", "role_label_candidate": "blocker_candidate", "status": "stable"}, {"role_id": "r1", "family_id": "f2", "role_label_candidate": "blocker_candidate", "status": "stable"}, {"role_id": "r2", "family_id": "f3", "role_label_candidate": "movement_controller_candidate", "status": "stable"}, {"role_id": "r2", "family_id": "f4", "role_label_candidate": "movement_controller_candidate", "status": "stable"}]), m3_dir / "role_candidate_membership.parquet", compression="zstd")
    (m3_dir / "v08d_report.json").write_text(json.dumps({"report": {}, "validation": {"scientific_conclusion": "m3_discriminative_weak_role_candidates"}}), encoding="utf-8")
    manifest = tmp_path / "v09_manifest.json"
    manifest.write_text(json.dumps({"name": "fixture_v09", "games": ["g1", "g2", "g3", "g4"], "families": {"famA": ["g1", "g2"], "famB": ["g3", "g4"]}, "purpose": "synthetic v09"}), encoding="utf-8")
    return m3_dir, m2_dir, m1_dir, manifest


def _install_v09b_sourceclean_fixture(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.parquet as pq
    from v6.role_candidates_v08d import DiscNeighborhood, GraphBuildDiagnostics
    import v6.role_transfer_v09b as mod

    manifest = tmp_path / "v09b_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "fixture_v09b",
                "games": ["a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4"],
                "families": {
                    "mf1": ["a1", "b1"],
                    "mf2": ["a2", "b2"],
                    "mf3": ["a3", "b3"],
                    "mf4": ["a4", "b4"],
                },
            }
        ),
        encoding="utf-8",
    )
    prev = tmp_path / "prev_v09a"
    prev.mkdir()
    (prev / "v09a_report.json").write_text(
        json.dumps(
            {
                "report": {
                    "scientific_conclusion": "role_transfer_sourceclean_weak",
                    "supports_H2": True,
                    "transfer_accuracy_structural_role": 0.5224,
                    "transfer_accuracy_raw_m2": 0.5052,
                    "transfer_accuracy_surface_effect": 0.3915,
                    "transfer_accuracy_graph_role_no_label": 0.4911,
                    "lift_vs_raw_m2": 0.0172,
                    "lift_vs_surface_effect": 0.1309,
                    "lift_vs_no_label_graph": 0.0313,
                    "positive_lift_families": 8,
                }
            }
        ),
        encoding="utf-8",
    )
    prev_rows = [
        {"heldout_family": family_id, "transfer_accuracy_structural_role": 0.50, "role_lift_over_best_baseline": 0.01}
        for family_id in ("mf1", "mf2", "mf3", "mf4")
    ]
    pq.write_table(pa.Table.from_pylist(prev_rows), prev / "role_transfer_by_family.parquet", compression="zstd")

    def _family_record(family_id: str, game_id: str, label: str):
        return SimpleNamespace(family_id=family_id, games_present=(game_id,), support_count=8, family_label_candidate=label)

    families = [
        _family_record("fa1", "a1", "blocked_no_change_family_candidate"),
        _family_record("fa2", "a2", "blocked_no_change_family_candidate"),
        _family_record("fa3", "a3", "blocked_no_change_family_candidate"),
        _family_record("fa4", "a4", "blocked_no_change_family_candidate"),
        _family_record("fb1", "b1", "position_like_change_family_candidate"),
        _family_record("fb2", "b2", "position_like_change_family_candidate"),
        _family_record("fb3", "b3", "position_like_change_family_candidate"),
        _family_record("fb4", "b4", "position_like_change_family_candidate"),
    ]

    def _disc(fid: str, gid: str, fam: str, sign: float, label: str, outcome: str, motif: str):
        return DiscNeighborhood(
            family_id=fid,
            family_label_candidate=label,
            game_ids=(gid,),
            game_family_ids=(fam,),
            support_count=8,
            family_coherence=0.9,
            mean_prediction_accuracy=0.9,
            mean_context_lift=0.2,
            dominant_outcome_signature=outcome,
            dominant_motif_candidate=motif,
            coarse_features={"c1": sign, "c2": sign * 0.9},
            directional_features={"d1": sign, "directional_asymmetry_score": sign * 0.8},
            future_option_features={"f1": sign, "enable_score": max(sign, 0.0), "block_score": max(-sign, 0.0), "terminate_score": max(-sign, 0.0) * 0.2},
            local_motif_features={"l1": sign, "cross_game_family_presence": 0.75},
            temporal_effect_features={"position_change_rate": max(-sign, 0.0), "no_change_rate": max(sign, 0.0), "terminal_rate": max(-sign, 0.0) * 0.1},
            incoming_edge_profile={label: 2},
            outgoing_edge_profile={label: 2},
            examples=(),
        )

    neighborhoods = {
        "fa1": _disc("fa1", "a1", "mf1", 1.0, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
        "fa2": _disc("fa2", "a2", "mf2", 0.98, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
        "fa3": _disc("fa3", "a3", "mf3", 0.96, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
        "fa4": _disc("fa4", "a4", "mf4", 0.94, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
        "fb1": _disc("fb1", "b1", "mf1", -1.0, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
        "fb2": _disc("fb2", "b2", "mf2", -0.98, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
        "fb3": _disc("fb3", "b3", "mf3", -0.96, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
        "fb4": _disc("fb4", "b4", "mf4", -0.94, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
    }
    tracker = {"source_games_by_holdout": []}

    def fake_load_m2_families(_path):
        return list(families)

    def fake_load_m1_support(_path):
        return {f"m1-{family.games_present[0]}": SimpleNamespace(game_id=family.games_present[0]) for family in families}

    def fake_load_episode_summaries(_path):
        return []

    def fake_load_m2_graph_edges(_path):
        return []

    def fake_build_game_family_map(_game_set, _selected_games):
        return {game: family_id for family_id, games in json.loads(manifest.read_text())["families"].items() for game in games}

    def fake_build_discriminative_neighborhoods(source_families, _source_support, _game_family_map, **_kwargs):
        subset_ids = {family.family_id for family in source_families}
        subset = {family_id: neighborhoods[family_id] for family_id in subset_ids}
        diag = GraphBuildDiagnostics("hybrid", 0, 0, 0, 0, 0, 0.0, 1.0)
        return subset, diag

    def fake_build_source_only_roles(_source_families, source_neighborhoods):
        source_ids = sorted(source_neighborhoods)
        source_games = {game for record in source_neighborhoods.values() for game in record.game_ids}
        heldout = next(iter({"mf1", "mf2", "mf3", "mf4"} - {next(iter(record.game_family_ids)) for record in source_neighborhoods.values()}), "none")
        tracker["source_games_by_holdout"].append((heldout, source_games))

        def _entry(role_id: str, label: str, member_ids: list[str]):
            members = [source_neighborhoods[item] for item in member_ids if item in source_neighborhoods]
            return {
                "role_id": role_id,
                "role_label_candidate": label,
                "member_family_ids": tuple(sorted(item.family_id for item in members)),
                "all_features": {
                **{f"coarse:{k}": float(v) for k, v in mod.mean_vector([item.coarse_features for item in members]).items()},
                **{f"directional:{k}": float(v) for k, v in mod.mean_vector([item.directional_features for item in members]).items()},
                **{f"future:{k}": float(v) for k, v in mod.mean_vector([item.future_option_features for item in members]).items()},
                **{f"motif:{k}": float(v) for k, v in mod.mean_vector([item.local_motif_features for item in members]).items()},
                **{f"effect:{k}": float(v) for k, v in mod.mean_vector([item.temporal_effect_features for item in members]).items()},
            },
            "coarse_features": mod.mean_vector([item.coarse_features for item in members]),
            "appearance_features": mod.mean_vector([mod.appearance_features(item) for item in members]),
        }

        role_a = [family_id for family_id in source_ids if family_id.startswith("fa")]
        role_b = [family_id for family_id in source_ids if family_id.startswith("fb")]
        return {
            "role_block": _entry("role_block", "blocker_candidate", role_a),
            "role_move": _entry("role_move", "movement_controller_candidate", role_b),
            "role_unknown": _entry("role_unknown", "unknown_role_candidate", role_a),
        }

    monkeypatch.setattr(mod, "load_m2_families", fake_load_m2_families)
    monkeypatch.setattr(mod, "load_m1_support", fake_load_m1_support)
    monkeypatch.setattr(mod, "load_episode_summaries", fake_load_episode_summaries)
    monkeypatch.setattr(mod, "load_m2_graph_edges", fake_load_m2_graph_edges)
    monkeypatch.setattr(mod, "build_game_family_map", fake_build_game_family_map)
    monkeypatch.setattr(mod, "build_discriminative_neighborhoods", fake_build_discriminative_neighborhoods)
    monkeypatch.setattr(mod, "build_source_only_roles", fake_build_source_only_roles)
    return manifest, prev, tracker


def _write_v09c_previous_report(tmp_path):
    prev = tmp_path / "prev_v09b"
    prev.mkdir()
    (prev / "v09b_report.json").write_text(
        json.dumps(
            {
                "report": {
                    "best_strategy": {
                        "strategy_name": "top_k_neighbors__k1__weights_default__include_unknown_roles_but_downweight__no_gating",
                        "prototype_mode": "top_k_neighbors",
                        "weight_profile": "default",
                        "top_k": 1,
                        "unknown_mode": "include_unknown_roles_but_downweight",
                        "confidence_mode": "no_gating",
                        "similarity_threshold": 0.0,
                        "margin": 0.0,
                        "transfer_accuracy_structural_role": 0.95,
                        "transfer_accuracy_surface_effect": 0.91,
                        "lift_vs_surface_effect": 0.04,
                        "scientific_conclusion": "role_transfer_refined_weak",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return prev


def _build_v09c_fixture_contexts():
    from types import SimpleNamespace
    from v6.role_transfer_v09b import FamilyContext
    from v6.role_transfer_v09 import appearance_features, mean_vector

    def disc(fid, gid, fam, sign, label, outcome, motif, effect_scale=1.0):
        from v6.role_candidates_v08d import DiscNeighborhood

        return DiscNeighborhood(
            family_id=fid,
            family_label_candidate=label,
            game_ids=(gid,),
            game_family_ids=(fam,),
            support_count=8,
            family_coherence=0.9,
            mean_prediction_accuracy=0.88,
            mean_context_lift=0.2,
            dominant_outcome_signature=outcome,
            dominant_motif_candidate=motif,
            coarse_features={"c1": sign, "support_density": abs(sign)},
            directional_features={
                "predecessor_count": max(sign, 0.0),
                "successor_count": max(-sign, 0.0),
                "source_like_score": max(sign, 0.0),
                "sink_like_score": max(-sign, 0.0),
                "bridge_like_score": abs(sign) * 0.3,
                "bottleneck_like_score": abs(sign) * 0.2,
                "branch_in_score": max(-sign, 0.0) * 0.5,
                "branch_out_score": max(sign, 0.0) * 0.5,
                "loop_score": abs(sign) * 0.1,
                "directional_asymmetry_score": sign,
            },
            future_option_features={
                "reachable_before_rate": 0.2 if sign > 0 else 0.7,
                "reachable_after_rate": 0.8 if sign > 0 else 0.3,
                "enable_score": max(sign, 0.0),
                "block_score": max(-sign, 0.0),
                "preserve_score": 0.4,
                "terminate_score": max(-sign, 0.0) * 0.2,
                "reversibility_score": 0.7 if sign > 0 else 0.2,
            },
            local_motif_features={"cross_game_family_presence": 0.75, "motif_entropy": 0.2, "local_branching_score": abs(sign) * 0.4, "local_loop_score": abs(sign) * 0.1},
            temporal_effect_features={"no_change_rate": max(sign, 0.0) * effect_scale, "position_change_rate": max(-sign, 0.0) * effect_scale, "terminal_rate": max(-sign, 0.0) * 0.1},
            incoming_edge_profile={label: 1},
            outgoing_edge_profile={label: 2},
            examples=(),
        )

    block_members = [
        disc("sb1", "g1", "srcA", 1.0, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
        disc("sb2", "g2", "srcB", 0.95, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
        disc("sb3", "g3", "srcC", 0.90, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
    ]
    move_members = [
        disc("sm1", "g4", "srcD", -1.0, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
        disc("sm2", "g5", "srcE", -0.95, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
        disc("sm3", "g6", "srcF", -0.90, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
    ]
    target_block = disc("tb1", "h1", "holdA", 0.97, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate", effect_scale=0.8)
    target_move = disc("tm1", "h2", "holdB", -0.97, "position_like_change_family_candidate", "position_like_change", "change_candidate", effect_scale=0.8)

    def entry(role_id, label, members):
        return {
            "role_id": role_id,
            "role_label_candidate": label,
            "member_family_ids": tuple(item.family_id for item in members),
            "all_features": {
                **{f"coarse:{k}": float(v) for k, v in mean_vector([item.coarse_features for item in members]).items()},
                **{f"directional:{k}": float(v) for k, v in mean_vector([item.directional_features for item in members]).items()},
                **{f"future:{k}": float(v) for k, v in mean_vector([item.future_option_features for item in members]).items()},
                **{f"motif:{k}": float(v) for k, v in mean_vector([item.local_motif_features for item in members]).items()},
                **{f"effect:{k}": float(v) for k, v in mean_vector([item.temporal_effect_features for item in members]).items()},
            },
            "coarse_features": mean_vector([item.coarse_features for item in members]),
            "appearance_features": mean_vector([appearance_features(item) for item in members]),
        }

    source_roles = {
        "role_block": entry("role_block", "blocker_candidate", block_members),
        "role_move": entry("role_move", "movement_controller_candidate", move_members),
        "role_unknown": entry("role_unknown", "unknown_role_candidate", block_members),
    }
    source_neighborhoods = {item.family_id: item for item in [*block_members, *move_members]}
    context_a = FamilyContext(
        heldout_family="holdA",
        heldout_games=("h1",),
        source_neighborhoods=source_neighborhoods,
        source_roles=source_roles,
        source_no_label_roles=source_roles,
        target_families=(SimpleNamespace(family_id="tb1", games_present=("h1",), support_count=8),),
        full_neighborhoods={"tb1": target_block},
        full_no_label_neighborhoods={"tb1": target_block},
        graph_source_used="hybrid",
        graph_edge_coverage=1.0,
    )
    context_b = FamilyContext(
        heldout_family="holdB",
        heldout_games=("h2",),
        source_neighborhoods=source_neighborhoods,
        source_roles=source_roles,
        source_no_label_roles=source_roles,
        target_families=(SimpleNamespace(family_id="tm1", games_present=("h2",), support_count=8),),
        full_neighborhoods={"tm1": target_move},
        full_no_label_neighborhoods={"tm1": target_move},
        graph_source_used="hybrid",
        graph_edge_coverage=1.0,
    )
    return [context_a, context_b]


def _build_v10_fixture_parts():
    from v6.role_transfer_v09 import Neighborhood, RoleRecord

    neighborhoods = {
        "f1": Neighborhood("f1", "blocked_no_change_family_candidate", ("g1",), ("famA",), 8, 0.9, 0.9, 0.2, "blocked_no_change", "block_candidate", {"c1": 1.0}, {"predecessor_count": 0.0, "successor_count": 2.0}, {"reachable_before_mean": 0.1, "reachable_after_mean": 0.9, "reachable_delta_mean": 0.8}, {"loop_2cycle_count": 0.0}, {"early_episode_frequency": 0.2, "mid_episode_frequency": 0.4, "late_episode_frequency": 0.6, "reversible_effect_rate": 0.2}, {"x": 1}, {"y": 2}),
        "f2": Neighborhood("f2", "position_like_change_family_candidate", ("g2",), ("famA",), 8, 0.9, 0.9, 0.2, "change", "change_candidate", {"c1": -1.0}, {"predecessor_count": 2.0, "successor_count": 0.0}, {"reachable_before_mean": 0.8, "reachable_after_mean": 0.2, "reachable_delta_mean": -0.6}, {"loop_2cycle_count": 0.0}, {"early_episode_frequency": 0.1, "mid_episode_frequency": 0.3, "late_episode_frequency": 0.7, "reversible_effect_rate": 0.1}, {"x": 1}, {"y": 2}),
        "f3": Neighborhood("f3", "blocked_no_change_family_candidate", ("g3",), ("famB",), 8, 0.9, 0.9, 0.2, "blocked_no_change", "block_candidate", {"c1": 0.9}, {"predecessor_count": 0.5, "successor_count": 1.8}, {"reachable_before_mean": 0.2, "reachable_after_mean": 0.8, "reachable_delta_mean": 0.6}, {"loop_2cycle_count": 0.0}, {"early_episode_frequency": 0.3, "mid_episode_frequency": 0.4, "late_episode_frequency": 0.5, "reversible_effect_rate": 0.2}, {"x": 1}, {"y": 2}),
        "f4": Neighborhood("f4", "position_like_change_family_candidate", ("g4",), ("famB",), 8, 0.9, 0.9, 0.2, "change", "change_candidate", {"c1": -0.9}, {"predecessor_count": 2.2, "successor_count": 0.1}, {"reachable_before_mean": 0.7, "reachable_after_mean": 0.1, "reachable_delta_mean": -0.6}, {"loop_2cycle_count": 0.0}, {"early_episode_frequency": 0.1, "mid_episode_frequency": 0.2, "late_episode_frequency": 0.8, "reversible_effect_rate": 0.1}, {"x": 1}, {"y": 2}),
        "f5": Neighborhood("f5", "blocked_no_change_family_candidate", ("g5",), ("famC",), 8, 0.9, 0.9, 0.2, "blocked_no_change", "block_candidate", {"c1": 0.95}, {"predecessor_count": 0.2, "successor_count": 1.9}, {"reachable_before_mean": 0.15, "reachable_after_mean": 0.85, "reachable_delta_mean": 0.7}, {"loop_2cycle_count": 0.0}, {"early_episode_frequency": 0.25, "mid_episode_frequency": 0.35, "late_episode_frequency": 0.55, "reversible_effect_rate": 0.2}, {"x": 1}, {"y": 2}),
        "f6": Neighborhood("f6", "position_like_change_family_candidate", ("g6",), ("famC",), 8, 0.9, 0.9, 0.2, "change", "change_candidate", {"c1": -0.95}, {"predecessor_count": 2.1, "successor_count": 0.2}, {"reachable_before_mean": 0.75, "reachable_after_mean": 0.15, "reachable_delta_mean": -0.6}, {"loop_2cycle_count": 0.0}, {"early_episode_frequency": 0.15, "mid_episode_frequency": 0.25, "late_episode_frequency": 0.75, "reversible_effect_rate": 0.1}, {"x": 1}, {"y": 2}),
    }
    roles = [
        RoleRecord("r1", "blocker_candidate", ("f1", "f3", "f5"), ("g1", "g3", "g5"), ("famA", "famB", "famC"), 24, 3, 3, 0.9, "stable"),
        RoleRecord("r2", "movement_controller_candidate", ("f2", "f4", "f6"), ("g2", "g4", "g6"), ("famA", "famB", "famC"), 24, 3, 3, 0.9, "stable"),
    ]
    family_to_role = {fid: roles[0] for fid in ("f1", "f3", "f5")} | {fid: roles[1] for fid in ("f2", "f4", "f6")}
    role_label_by_id = {"r1": "blocker_candidate", "r2": "movement_controller_candidate"}
    game_to_manifest = {"g1": "mf1", "g2": "mf1", "g3": "mf2", "g4": "mf2", "g5": "mf3", "g6": "mf3"}
    transfer_rows = [
        {"heldout_family": "mf1", "target_family_id": "f1", "assigned_role_id": "r1", "role_hardened_score": 0.82, "surface_hardened_score": 0.12, "raw_m2_hardened_score": 0.60, "effect_residual_score": 0.20, "future_option_role_score": 0.8, "graph_position_role_score": 0.7},
        {"heldout_family": "mf1", "target_family_id": "f2", "assigned_role_id": "r2", "role_hardened_score": 0.80, "surface_hardened_score": 0.10, "raw_m2_hardened_score": 0.62, "effect_residual_score": 0.18, "future_option_role_score": 0.75, "graph_position_role_score": 0.65},
        {"heldout_family": "mf2", "target_family_id": "f3", "assigned_role_id": "r1", "role_hardened_score": 0.84, "surface_hardened_score": 0.14, "raw_m2_hardened_score": 0.64, "effect_residual_score": 0.22, "future_option_role_score": 0.82, "graph_position_role_score": 0.72},
        {"heldout_family": "mf2", "target_family_id": "f4", "assigned_role_id": "r2", "role_hardened_score": 0.83, "surface_hardened_score": 0.13, "raw_m2_hardened_score": 0.61, "effect_residual_score": 0.21, "future_option_role_score": 0.79, "graph_position_role_score": 0.70},
        {"heldout_family": "mf3", "target_family_id": "f5", "assigned_role_id": "r1", "role_hardened_score": 0.81, "surface_hardened_score": 0.11, "raw_m2_hardened_score": 0.63, "effect_residual_score": 0.19, "future_option_role_score": 0.78, "graph_position_role_score": 0.68},
        {"heldout_family": "mf3", "target_family_id": "f6", "assigned_role_id": "r2", "role_hardened_score": 0.82, "surface_hardened_score": 0.12, "raw_m2_hardened_score": 0.62, "effect_residual_score": 0.20, "future_option_role_score": 0.80, "graph_position_role_score": 0.69},
    ]
    return roles, neighborhoods, transfer_rows, family_to_role, role_label_by_id, game_to_manifest


def _write_v10_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    roles, neighborhoods, transfer_rows, _, _, _ = _build_v10_fixture_parts()
    m3_dir = tmp_path / "m3v10"
    transfer_dir = tmp_path / "v09c"
    m3_dir.mkdir()
    transfer_dir.mkdir()
    rows = []
    for record in neighborhoods.values():
        rows.append(
            {
                "family_id": record.family_id,
                "family_label_candidate": record.family_label_candidate,
                "games_present": list(record.games_present),
                "game_families_present": list(record.game_families_present),
                "support_count": record.support_count,
                "family_coherence": record.family_coherence,
                "mean_prediction_accuracy": record.mean_prediction_accuracy,
                "mean_context_lift": record.mean_context_lift,
                "dominant_outcome_signature": record.dominant_outcome_signature,
                "dominant_motif_candidate": record.dominant_motif_candidate,
                "coarse_features": json.dumps(record.coarse_features),
                "directional_features": json.dumps(record.directional_features),
                "future_option_features": json.dumps(record.future_option_features),
                "local_motif_features": json.dumps(record.local_motif_features),
                "temporal_effect_features": json.dumps(record.temporal_effect_features),
                "incoming_edge_profile": json.dumps(record.incoming_edge_profile),
                "outgoing_edge_profile": json.dumps(record.outgoing_edge_profile),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), m3_dir / "role_neighborhoods.parquet", compression="zstd")
    (m3_dir / "m3_role_candidates.json").write_text(json.dumps([role.__dict__ for role in roles]), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{"role_id": "r1", "family_id": "f1", "role_label_candidate": "blocker_candidate", "status": "stable"}, {"role_id": "r2", "family_id": "f2", "role_label_candidate": "movement_controller_candidate", "status": "stable"}, {"role_id": "r1", "family_id": "f3", "role_label_candidate": "blocker_candidate", "status": "stable"}, {"role_id": "r2", "family_id": "f4", "role_label_candidate": "movement_controller_candidate", "status": "stable"}, {"role_id": "r1", "family_id": "f5", "role_label_candidate": "blocker_candidate", "status": "stable"}, {"role_id": "r2", "family_id": "f6", "role_label_candidate": "movement_controller_candidate", "status": "stable"}]), m3_dir / "role_candidate_membership.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(transfer_rows), transfer_dir / "v09c_hardened_assignments.parquet", compression="zstd")
    (transfer_dir / "v09c_report.json").write_text(json.dumps({"report": {"scientific_conclusion": "hardened_transfer_strong", "supports_H2": True, "v10_gate_cleared": True, "transfer_accuracy_role_hardened": 0.7, "lift_vs_surface_effect_hardened": 0.6, "lift_vs_no_label_graph_hardened": 0.5, "positive_lift_families_hardened": 4}}, indent=2), encoding="utf-8")
    manifest = tmp_path / "v10_manifest.json"
    manifest.write_text(json.dumps({"name": "v10_fixture", "games": ["g1", "g2", "g3", "g4", "g5", "g6"], "families": {"mf1": ["g1", "g2"], "mf2": ["g3", "g4"], "mf3": ["g5", "g6"]}}), encoding="utf-8")
    return m3_dir, transfer_dir, manifest


def _build_v10fix_fixture_contexts():
    from types import SimpleNamespace

    from v6.role_transfer_v09c import FamilyContext
    from v6.role_candidates_v08d import DiscNeighborhood
    from v6.role_transfer_v09 import appearance_features, mean_vector

    def disc(fid, gid, fam, sign, label, outcome, motif, support=8):
        return DiscNeighborhood(
            family_id=fid,
            family_label_candidate=label,
            game_ids=(gid,),
            game_family_ids=(fam,),
            support_count=support,
            family_coherence=0.9,
            mean_prediction_accuracy=0.88,
            mean_context_lift=0.2,
            dominant_outcome_signature=outcome,
            dominant_motif_candidate=motif,
            coarse_features={"c1": sign, "support_density": abs(sign)},
            directional_features={
                "predecessor_count": 0.2 if sign > 0 else 1.8,
                "successor_count": 1.8 if sign > 0 else 0.2,
                "source_like_score": max(sign, 0.0),
                "sink_like_score": max(-sign, 0.0),
                "bridge_like_score": abs(sign) * 0.4,
                "bottleneck_like_score": abs(sign) * 0.3,
                "branch_in_score": max(-sign, 0.0) * 0.4,
                "branch_out_score": max(sign, 0.0) * 0.4,
                "loop_score": 0.1,
                "directional_asymmetry_score": sign,
            },
            future_option_features={
                "reachable_before_rate": 0.2 if sign > 0 else 0.7,
                "reachable_after_rate": 0.8 if sign > 0 else 0.3,
                "reachable_delta_mean": 0.6 if sign > 0 else -0.6,
                "enable_score": max(sign, 0.0),
                "block_score": max(-sign, 0.0),
                "preserve_score": 0.4,
                "terminate_score": max(-sign, 0.0) * 0.2,
                "reversibility_score": 0.7 if sign > 0 else 0.2,
            },
            local_motif_features={"cross_game_family_presence": 0.75, "motif_entropy": 0.2, "local_branching_score": abs(sign) * 0.4, "local_loop_score": 0.1},
            temporal_effect_features={"early_episode_frequency": 0.3, "mid_episode_frequency": 0.4, "late_episode_frequency": 0.3, "reversible_effect_rate": 0.2, "no_change_rate": max(sign, 0.0), "position_change_rate": max(-sign, 0.0)},
            incoming_edge_profile={label: 1},
            outgoing_edge_profile={label: 2},
            examples=(),
        )

    source_members = [
        disc("sa_block", "g1", "srcA", 1.0, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
        disc("sa_move", "g2", "srcA", -1.0, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
        disc("sb_block", "g3", "srcB", 0.95, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate"),
        disc("sb_move", "g4", "srcB", -0.95, "position_like_change_family_candidate", "position_like_change", "change_candidate"),
    ]
    source_neighborhoods = {item.family_id: item for item in source_members}

    def entry(role_id, label, members):
        return {
            "role_id": role_id,
            "role_label_candidate": label,
            "member_family_ids": tuple(item.family_id for item in members),
            "all_features": {
                **{f"coarse:{k}": float(v) for k, v in mean_vector([item.coarse_features for item in members]).items()},
                **{f"directional:{k}": float(v) for k, v in mean_vector([item.directional_features for item in members]).items()},
                **{f"future:{k}": float(v) for k, v in mean_vector([item.future_option_features for item in members]).items()},
                **{f"motif:{k}": float(v) for k, v in mean_vector([item.local_motif_features for item in members]).items()},
                **{f"effect:{k}": float(v) for k, v in mean_vector([item.temporal_effect_features for item in members]).items()},
            },
            "coarse_features": mean_vector([item.coarse_features for item in members]),
            "appearance_features": mean_vector([appearance_features(item) for item in members]),
        }

    source_roles = {
        "role_block": entry("role_block", "blocker_candidate", [source_neighborhoods["sa_block"], source_neighborhoods["sb_block"]]),
        "role_move": entry("role_move", "movement_controller_candidate", [source_neighborhoods["sa_move"], source_neighborhoods["sb_move"]]),
    }
    target_a = disc("ta", "h1", "holdA", 0.98, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate")
    target_b = disc("tb", "h2", "holdB", -0.98, "position_like_change_family_candidate", "position_like_change", "change_candidate")

    return [
        FamilyContext(
            heldout_family="holdA",
            heldout_games=("h1",),
            source_neighborhoods=source_neighborhoods,
            source_roles=source_roles,
            source_no_label_roles=source_roles,
            target_families=(SimpleNamespace(family_id="ta", games_present=("h1",), support_count=8),),
            full_neighborhoods={"ta": target_a},
            full_no_label_neighborhoods={"ta": target_a},
            graph_source_used="hybrid",
            graph_edge_coverage=1.0,
        ),
        FamilyContext(
            heldout_family="holdB",
            heldout_games=("h2",),
            source_neighborhoods=source_neighborhoods,
            source_roles=source_roles,
            source_no_label_roles=source_roles,
            target_families=(SimpleNamespace(family_id="tb", games_present=("h2",), support_count=8),),
            full_neighborhoods={"tb": target_b},
            full_no_label_neighborhoods={"tb": target_b},
            graph_source_used="hybrid",
            graph_edge_coverage=1.0,
        ),
    ]


def _write_v10fix_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    transfer_dir = tmp_path / "v09c_fix"
    transfer_dir.mkdir()
    rows = [
        {"heldout_family": "holdA", "target_family_id": "ta", "assigned_role_id": "role_block", "surface_hardened_score": 0.18, "effect_residual_score": 0.22, "future_option_role_score": 0.80, "graph_position_role_score": 0.72},
        {"heldout_family": "holdA", "target_family_id": "ta", "assigned_role_id": "role_move", "surface_hardened_score": 0.18, "effect_residual_score": 0.20, "future_option_role_score": 0.78, "graph_position_role_score": 0.70},
        {"heldout_family": "holdB", "target_family_id": "tb", "assigned_role_id": "role_block", "surface_hardened_score": 0.16, "effect_residual_score": 0.21, "future_option_role_score": 0.81, "graph_position_role_score": 0.73},
        {"heldout_family": "holdB", "target_family_id": "tb", "assigned_role_id": "role_move", "surface_hardened_score": 0.16, "effect_residual_score": 0.19, "future_option_role_score": 0.79, "graph_position_role_score": 0.71},
    ]
    pq.write_table(pa.Table.from_pylist(rows), transfer_dir / "v09c_hardened_assignments.parquet", compression="zstd")
    (transfer_dir / "v09c_report.json").write_text(
        json.dumps({"report": {"scientific_conclusion": "hardened_transfer_strong", "supports_H2": True, "v10_gate_cleared": True}}, indent=2),
        encoding="utf-8",
    )
    manifest = tmp_path / "v10fix_manifest.json"
    manifest.write_text(json.dumps({"name": "v10fix_fixture", "games": ["g1", "g2", "g3", "g4", "h1", "h2"], "families": {"srcA": ["g1", "g2"], "srcB": ["g3", "g4"], "holdA": ["h1"], "holdB": ["h2"]}}), encoding="utf-8")
    return transfer_dir, manifest


def _build_v10fixb_fixture_contexts(include_four_role_manifest: bool = False, two_target_families: bool = False):
    from types import SimpleNamespace

    from v6.role_transfer_v09c import FamilyContext
    from v6.role_candidates_v08d import DiscNeighborhood
    from v6.role_transfer_v09 import appearance_features, mean_vector

    def disc(fid, gid, fam, sign, label, outcome, motif, support=8, block=0.0, enable=0.0, preserve=0.3):
        return DiscNeighborhood(
            family_id=fid,
            family_label_candidate=label,
            game_ids=(gid,),
            game_family_ids=(fam,),
            support_count=support,
            family_coherence=0.9,
            mean_prediction_accuracy=0.88,
            mean_context_lift=0.2,
            dominant_outcome_signature=outcome,
            dominant_motif_candidate=motif,
            coarse_features={"c1": sign, "support_density": abs(sign)},
            directional_features={
                "predecessor_count": 0.2 if sign > 0 else 1.8,
                "successor_count": 1.8 if sign > 0 else 0.2,
                "source_like_score": max(sign, 0.0),
                "sink_like_score": max(-sign, 0.0),
                "bridge_like_score": abs(sign) * 0.4,
                "bottleneck_like_score": abs(sign) * 0.3,
                "branch_in_score": max(-sign, 0.0) * 0.4,
                "branch_out_score": max(sign, 0.0) * 0.4,
                "loop_score": 0.1 if preserve < 0.4 else 0.6,
                "directional_asymmetry_score": sign,
            },
            future_option_features={
                "reachable_before_rate": 0.2 if sign > 0 else 0.7,
                "reachable_after_rate": 0.8 if sign > 0 else 0.3,
                "reachable_delta_mean": 0.6 if sign > 0 else -0.6,
                "enable_score": enable,
                "block_score": block,
                "preserve_score": preserve,
                "terminate_score": max(-sign, 0.0) * 0.2,
                "reversibility_score": 0.7 if preserve > 0.4 else 0.2,
            },
            local_motif_features={"cross_game_family_presence": 0.75, "motif_entropy": 0.2, "local_branching_score": abs(sign) * 0.4, "local_loop_score": 0.1 if preserve < 0.4 else 0.6},
            temporal_effect_features={"early_episode_frequency": 0.3, "mid_episode_frequency": 0.4, "late_episode_frequency": 0.3, "reversible_effect_rate": 0.2 if preserve < 0.4 else 0.7, "no_change_rate": max(sign, 0.0), "position_change_rate": max(-sign, 0.0)},
            incoming_edge_profile={label: 1},
            outgoing_edge_profile={label: 2},
            examples=(),
        )

    source_members = [
        disc("sa_block", "g1", "srcA", 1.0, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate", block=0.7),
        disc("sa_move", "g2", "srcA", -1.0, "position_like_change_family_candidate", "position_like_change", "change_candidate", enable=0.6),
        disc("sb_block", "g3", "srcB", 1.0, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate", block=0.7),
        disc("sb_move", "g4", "srcB", -1.0, "position_like_change_family_candidate", "position_like_change", "change_candidate", enable=0.6),
    ]
    if include_four_role_manifest:
        source_members.extend(
            [
                disc("sa_cover", "g5", "srcA", 0.6, "coverage_like_family_candidate", "coverage_gain", "cover_candidate", enable=0.5),
                disc("sa_link", "g6", "srcA", 0.4, "connector_family_candidate", "link_path", "link_candidate", preserve=0.6),
            ]
        )
    source_neighborhoods = {item.family_id: item for item in source_members}

    def entry(role_id, label, members):
        return {
            "role_id": role_id,
            "role_label_candidate": label,
            "member_family_ids": tuple(item.family_id for item in members),
            "all_features": {
                **{f"coarse:{k}": float(v) for k, v in mean_vector([item.coarse_features for item in members]).items()},
                **{f"directional:{k}": float(v) for k, v in mean_vector([item.directional_features for item in members]).items()},
                **{f"future:{k}": float(v) for k, v in mean_vector([item.future_option_features for item in members]).items()},
                **{f"motif:{k}": float(v) for k, v in mean_vector([item.local_motif_features for item in members]).items()},
                **{f"effect:{k}": float(v) for k, v in mean_vector([item.temporal_effect_features for item in members]).items()},
            },
            "coarse_features": mean_vector([item.coarse_features for item in members]),
            "appearance_features": mean_vector([appearance_features(item) for item in members]),
        }

    role_defs = {
        "role_block": ("blocker_candidate", [source_neighborhoods["sa_block"], source_neighborhoods["sb_block"]]),
        "role_move": ("movement_controller_candidate", [source_neighborhoods["sa_move"], source_neighborhoods["sb_move"]]),
    }
    if include_four_role_manifest:
        role_defs["role_cover"] = ("coverage_expander_candidate", [source_neighborhoods["sa_cover"]])
        role_defs["role_link"] = ("connector_candidate", [source_neighborhoods["sa_link"]])
    source_roles = {role_id: entry(role_id, label, members) for role_id, (label, members) in role_defs.items()}

    target_a = disc("ta", "h1", "holdA", 0.98, "blocked_no_change_family_candidate", "blocked_no_change", "block_candidate", block=0.7)
    target_b = disc("tb", "h2", "holdB", -0.98, "position_like_change_family_candidate", "position_like_change", "change_candidate", enable=0.6)
    target_a2 = disc("ta2", "h3", "holdA", -0.2, "misc_family_candidate", "misc", "misc_candidate", preserve=0.2)

    first_targets = (SimpleNamespace(family_id="ta", games_present=("h1",), support_count=8),)
    full_neighborhoods = {"ta": target_a}
    full_no_label = {"ta": target_a}
    if two_target_families:
        first_targets = (
            SimpleNamespace(family_id="ta", games_present=("h1",), support_count=8),
            SimpleNamespace(family_id="ta2", games_present=("h3",), support_count=8),
        )
        full_neighborhoods = {"ta": target_a, "ta2": target_a2}
        full_no_label = {"ta": target_a, "ta2": target_a2}

    return [
        FamilyContext(
            heldout_family="holdA",
            heldout_games=("h1",),
            source_neighborhoods=source_neighborhoods,
            source_roles=source_roles,
            source_no_label_roles=source_roles,
            target_families=first_targets,
            full_neighborhoods=full_neighborhoods,
            full_no_label_neighborhoods=full_no_label,
            graph_source_used="hybrid",
            graph_edge_coverage=1.0,
        ),
        FamilyContext(
            heldout_family="holdB",
            heldout_games=("h2",),
            source_neighborhoods=source_neighborhoods,
            source_roles=source_roles,
            source_no_label_roles=source_roles,
            target_families=(SimpleNamespace(family_id="tb", games_present=("h2",), support_count=8),),
            full_neighborhoods={"tb": target_b},
            full_no_label_neighborhoods={"tb": target_b},
            graph_source_used="hybrid",
            graph_edge_coverage=1.0,
        ),
    ]


def run_v10fixb_raw_rows_for_context(context, source_role_map):
    import v6.concept_candidates_v10fixb as mod

    return mod.discover_source_only_candidates(
        context,
        source_role_map,
        {},
        ConceptCandidatesV10FixBConfig(workers=1, min_games=1, min_manifest_families=1),
    )


def _write_v10fixb_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    transfer_dir = tmp_path / "v09c_fixb"
    transfer_dir.mkdir()
    rows = [
        {"heldout_family": "holdA", "target_family_id": "ta", "assigned_role_id": "role_block", "surface_hardened_score": 0.18, "effect_residual_score": 0.22, "future_option_role_score": 0.80, "graph_position_role_score": 0.72},
        {"heldout_family": "holdA", "target_family_id": "ta", "assigned_role_id": "role_move", "surface_hardened_score": 0.18, "effect_residual_score": 0.20, "future_option_role_score": 0.78, "graph_position_role_score": 0.70},
        {"heldout_family": "holdB", "target_family_id": "tb", "assigned_role_id": "role_block", "surface_hardened_score": 0.16, "effect_residual_score": 0.21, "future_option_role_score": 0.81, "graph_position_role_score": 0.73},
        {"heldout_family": "holdB", "target_family_id": "tb", "assigned_role_id": "role_move", "surface_hardened_score": 0.16, "effect_residual_score": 0.19, "future_option_role_score": 0.79, "graph_position_role_score": 0.71},
    ]
    pq.write_table(pa.Table.from_pylist(rows), transfer_dir / "v09c_hardened_assignments.parquet", compression="zstd")
    (transfer_dir / "v09c_report.json").write_text(
        json.dumps({"report": {"scientific_conclusion": "hardened_transfer_strong", "supports_H2": True, "v10_gate_cleared": True}}, indent=2),
        encoding="utf-8",
    )
    manifest = tmp_path / "v10fixb_manifest.json"
    manifest.write_text(json.dumps({"name": "v10fixb_fixture", "games": ["g1", "g2", "g3", "g4", "h1", "h2"], "families": {"srcA": ["g1", "g2"], "srcB": ["g3", "g4"], "holdA": ["h1"], "holdB": ["h2"]}}), encoding="utf-8")
    return transfer_dir, manifest


def test_v10fixb_loads_manifest_family_map_from_m3_neighborhoods(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    m3_dir = tmp_path / "m3"
    m3_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"family_id": "m2x-0001", "game_families_present": ["srcA"]},
                {"family_id": "m2x-0002", "game_families_present": ["srcB", "srcA"]},
            ]
        ),
        m3_dir / "role_neighborhoods.parquet",
        compression="zstd",
    )

    mapping = load_source_manifest_family_map(m3_dir)

    assert mapping["m2x-0001"] == ("srcA",)
    assert mapping["m2x-0002"] == ("srcA", "srcB")


def test_v10fixb_manifest_family_resolution_prefers_m3_mapping() -> None:
    contexts = _build_v10fixb_fixture_contexts()
    record = contexts[0].source_neighborhoods["sa_block"]

    resolved = resolve_manifest_families_for_record(
        family_id="sa_block",
        record=record,
        source_manifest_family_map={"sa_block": ("mapped_src",)},
    )

    assert resolved == ("mapped_src",)


def _build_context_compare_fixture(tmp_path, tie_break: bool = False, cd3_useful: bool = False, extended: bool = False) -> dict[str, Path]:
    runs = {}
    game_list = ["tt01", "pb02", "fs02"] if not extended else ["tt01", "pb02", "fs02", "tp02", "gr01", "va02", "mo01", "ex01"]
    specs = {
        "cd0": {
            "depth": 0,
            "m1_acc": 0.62,
            "lift": 0.01,
            "m2_coherence": 0.72,
            "compression": 6.0,
            "cross_game": 2,
            "tiny_ratio": 0.10,
            "unknown_ratio": 0.10,
            "stable": 5,
            "total": 8,
        },
        "cd1": {
            "depth": 1,
            "m1_acc": 0.83,
            "lift": 0.11,
            "m2_coherence": 0.91,
            "compression": 12.0,
            "cross_game": 5,
            "tiny_ratio": 0.08,
            "unknown_ratio": 0.05,
            "stable": 7,
            "total": 8,
        },
        "cd2": {
            "depth": 2,
            "m1_acc": 0.89 if tie_break else 0.78,
            "lift": 0.14 if tie_break else 0.09,
            "m2_coherence": 0.93 if tie_break else 0.79,
            "compression": 12.2 if tie_break else 4.0,
            "cross_game": 6 if tie_break else 1,
            "tiny_ratio": 0.08 if tie_break else 0.45,
            "unknown_ratio": 0.05 if tie_break else 0.40,
            "stable": 8 if tie_break else 6,
            "total": 9 if tie_break else 15,
        },
        "cd3": {
            "depth": 3,
            "m1_acc": 0.90 if tie_break else (0.95 if cd3_useful else 0.84),
            "lift": 0.145 if tie_break else (0.16 if cd3_useful else 0.13),
            "m2_coherence": 0.931 if tie_break else (0.95 if cd3_useful else 0.74),
            "compression": 12.4 if tie_break else (13.5 if cd3_useful else 3.6),
            "cross_game": 6 if tie_break else (8 if cd3_useful and extended else 2),
            "tiny_ratio": 0.08 if tie_break else (0.08 if cd3_useful else 0.50),
            "unknown_ratio": 0.05 if tie_break else (0.05 if cd3_useful else 0.45),
            "stable": 8 if tie_break else (9 if cd3_useful else 5),
            "total": 9 if tie_break else (9 if cd3_useful else 18),
        },
    }
    for label, spec in specs.items():
        v06_dir = tmp_path / f"v06_{label}"
        v07_dir = tmp_path / f"v07_{label}"
        v06_dir.mkdir(parents=True, exist_ok=True)
        v07_dir.mkdir(parents=True, exist_ok=True)
        v06_report = {
            "config": {"context_depth": spec["depth"], "games": game_list, "min_support": 5},
            "report": {
                "total_contingency_candidates": 100 + spec["depth"] * 20,
                "total_discovered_contingencies": 80 + spec["depth"] * 15,
                "discovered_contingencies_by_game": {game: 20 for game in game_list},
                "game_summary": [
                    {"game": game, "discovered_contingency_count": 20, "mean_prediction_accuracy": spec["m1_acc"], "context_lift": spec["lift"]}
                    for game in game_list
                ],
            },
        }
        contingency_rows = [
            {
                "contingency_id": f"m1-{label}-{index}",
                "support_count": 10 if index % 4 else 3,
            }
            for index in range(12)
        ]
        families = []
        for index in range(spec["total"]):
            tiny = index < int(round(spec["tiny_ratio"] * spec["total"]))
            unknown = index < int(round(spec["unknown_ratio"] * spec["total"]))
            families.append(
                {
                    "family_id": f"m2-{label}-{index}",
                    "family_label_candidate": "unknown_change_family_candidate" if unknown else "position_like_change_family_candidate",
                    "games_present": ["tt01", "pb02", "fs02"] if index < spec["cross_game"] else ["tt01"],
                    "samplers_present": ["random_baseline"],
                    "contingency_ids": [f"m1-{label}-{index}"] if tiny else [f"m1-{label}-{index}", f"m1-{label}-{index}-b", f"m1-{label}-{index}-c"],
                    "support_count": 10,
                    "mean_prediction_accuracy": spec["m1_acc"],
                    "mean_context_lift": spec["lift"],
                    "dominant_outcome_signature": "position_like_change",
                    "outcome_signature_distribution": {"position_like_change": 3},
                    "motif_candidate_distribution": {"change_candidate": 3},
                    "family_coherence": spec["m2_coherence"],
                    "compression_ratio": spec["compression"] / spec["total"],
                    "cross_game_presence": 3 if index < spec["cross_game"] else 1,
                    "stable": index < spec["stable"],
                    "examples": [],
                    "notes": {"outcome_entropy": 0.0},
                }
            )
        v07_report = {
            "config": {"input_dir": str(v06_dir)},
            "validation": {"scientific_conclusion": "m2_weak"},
            "report": {
                "total_m1_contingencies_loaded": len(contingency_rows),
                "total_m2_family_candidates": spec["total"],
                "stable_m2_families": spec["stable"],
                "compression_ratio": spec["compression"],
                "mean_family_coherence": spec["m2_coherence"],
                "cross_game_families": [family for family in families if family["cross_game_presence"] > 1],
                "families_by_game": {
                    game: {
                        "total_families": spec["total"] if game == game_list[0] else max(1, spec["cross_game"]),
                        "stable_families": min(spec["stable"], max(1, spec["cross_game"])),
                        "game_compression_ratio": spec["compression"] if game == game_list[0] else spec["compression"] / 2,
                    }
                    for game in game_list
                },
            },
        }
        (v06_dir / "v06_report.json").write_text(json.dumps(v06_report, indent=2), encoding="utf-8")
        (v06_dir / "contingencies.json").write_text(json.dumps(contingency_rows, indent=2), encoding="utf-8")
        (v07_dir / "v07_report.json").write_text(json.dumps(v07_report, indent=2), encoding="utf-8")
        (v07_dir / "m2_families.json").write_text(json.dumps(families, indent=2), encoding="utf-8")
        runs[f"{label}_v07"] = v07_dir
    return runs


def _obs(value: int) -> np.ndarray:
    return np.full((2, 2), value, dtype=int)


def _trace_event(*, step: int, action: int, context: tuple[str, ...], outcome: str, outcome_state: str | None = None, outcome_polarity: str | None = None):
    from v6.contingency_memory import TraceEvent

    non_preserve = outcome in {"position_like_change", "large_change", "change", "terminal_transition"}
    blocked = outcome in {"blocked_no_change", "preserve_no_change"}
    return TraceEvent(
        game_id="va02",
        sampler="mixed",
        seed=0,
        step=step,
        timestamp=step,
        interaction_id=step + 1,
        episode_id=1,
        action=action,
        context_signature=context,
        outcome_signature=outcome,
        state_before_signature=f"s{step}",
        state_after_signature=f"s{step+1}",
        blocked_or_no_change=blocked,
        non_preserve=non_preserve,
        terminal_observed=outcome == "terminal_transition" or outcome_state in {"WIN", "GAME_OVER"},
        outcome_state=outcome_state,
        outcome_polarity=outcome_polarity,
        delta_summary={},
    )


def test_compact_fold_scopes_legacy_local_interaction_ids_across_raw_dbs(tmp_path: Path) -> None:
    from v6.memory.compact_memory import CompactMemoryFoldConfig, ensure_memory_layout, fold_epoch_raw_into_compact_memory

    raw_root = tmp_path / "epochs" / "epoch_0001" / "raw" / "sampling_v05c"
    db1 = raw_root / "ga01" / "sampler_a" / "steps_500" / "seed_0.sqlite"
    db2 = raw_root / "ga02" / "sampler_b" / "steps_500" / "seed_0.sqlite"
    db1.parent.mkdir(parents=True, exist_ok=True)
    db2.parent.mkdir(parents=True, exist_ok=True)
    for db_path in (db1, db2):
        with sqlite3.connect(db_path) as conn:
            substrate = MemorySubstrate(conn)
            substrate.upsert_node(MemoryNode(node_id="M0:interaction:1", memory_level="M0", node_type="InteractionMemory", canonical_key="1"))
            substrate.upsert_score(MemoryScore(node_id="M0:interaction:1", replay_priority=0.7))
            substrate.add_evidence(
                MemoryEvidence(
                    evidence_id="e1",
                    target_node_id="M0:interaction:1",
                    source_interaction_id=1,
                    evidence_type="test",
                    payload={},
                )
            )
            substrate.record_promotion(
                MemoryPromotion(
                    promotion_id="p1",
                    source_node_id="M0:interaction:1",
                    target_node_id="M0:interaction:1",
                    promotion_type="TEST",
                    evidence_count=1,
                    promotion_score=1.0,
                    status="ok",
                )
            )
            conn.commit()
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    fold_epoch_raw_into_compact_memory(
        epoch_raw_dir=raw_root,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        nodes = [row[0] for row in conn.execute("SELECT node_id FROM memory_nodes WHERE node_type = 'InteractionMemory' ORDER BY node_id ASC").fetchall()]
        canonical_keys = [row[0] for row in conn.execute("SELECT canonical_key FROM memory_nodes WHERE node_type = 'InteractionMemory' ORDER BY canonical_key ASC").fetchall()]
        scores = [row[0] for row in conn.execute("SELECT node_id FROM memory_scores WHERE node_id LIKE 'M0:interaction:%' ORDER BY node_id ASC").fetchall()]
        evidence_ids = [row[0] for row in conn.execute("SELECT evidence_id FROM memory_evidence ORDER BY evidence_id ASC").fetchall()]
        promotion_ids = [row[0] for row in conn.execute("SELECT promotion_id FROM memory_promotions ORDER BY promotion_id ASC").fetchall()]
    assert len(nodes) == 2
    assert len(scores) == 2
    assert len(evidence_ids) == 2
    assert len(promotion_ids) == 2
    assert len(set(nodes)) == 2
    assert len(set(canonical_keys)) == 2
    assert len(set(evidence_ids)) == 2
    assert len(set(promotion_ids)) == 2
    assert all(":local:" in node for node in nodes)
    assert all(str(value).startswith("canonical:db:") for value in canonical_keys)


def test_h02_compact_high_priority_threshold_uses_0_8_cutoff(tmp_path: Path) -> None:
    from v6.hypothesis_h02_report import _extract_h02_compact_metrics

    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        for idx, priority in enumerate((0.1, 0.79, 0.8, 1.0), start=1):
            conn.execute(
                "INSERT INTO memory_scores (node_id, replay_priority) VALUES (?, ?)",
                (f"M0:interaction:g{idx}", priority),
            )
        conn.commit()
    metrics = _extract_h02_compact_metrics(memory_dir)
    assert metrics["high_priority_replay_count"] == 2


def test_h02b_generic_carriers_do_not_count_as_no_carriers() -> None:
    from v6.hypothesis_h02_report import _evaluate_h02b_pre_carrier_timing

    payload = _evaluate_h02b_pre_carrier_timing(
        {
            "emergent_carrier_count": 10,
            "emergent_object_carrier_count": 0,
            "emergent_context_action_fallback_count": 0,
        },
        [],
    )
    assert payload["decision"] == "PARTIALLY_VALID"
    assert "Generic emergent carriers" in payload["note"]


def test_h02b_zero_emergent_carriers_is_valid() -> None:
    from v6.hypothesis_h02_report import _evaluate_h02b_pre_carrier_timing

    payload = _evaluate_h02b_pre_carrier_timing(
        {
            "emergent_carrier_count": 0,
            "emergent_object_carrier_count": 0,
            "emergent_context_action_fallback_count": 0,
        },
        [],
    )
    assert payload["decision"] == "VALID"


def test_h02b_object_carriers_without_temporal_rows_is_inconclusive() -> None:
    from v6.hypothesis_h02_report import _evaluate_h02b_pre_carrier_timing

    payload = _evaluate_h02b_pre_carrier_timing(
        {
            "emergent_carrier_count": 3,
            "emergent_object_carrier_count": 2,
            "emergent_context_action_fallback_count": 0,
        },
        [],
    )
    assert payload["decision"] == "INCONCLUSIVE"


def test_h01_uses_compact_memory_record_count_when_raw_zero(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1", "total_interactions": 10, "memory_record_count": 0}]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES ('M0:interaction:g1', 'M0', 'InteractionMemory', 'g1', 1, '{}')")
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES ('M0:interaction:g2', 'M0', 'InteractionMemory', 'g2', 1, '{}')")
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, action, effect_signature, support_count) VALUES ('c1', 'g1', 's1', 1, 'f1', 20)")
        conn.commit()
    result = evaluate_h01_contingency_emergence(run_dir, tmp_path / "out_h01", memory_dir=memory_dir)
    assert result["memory_record_count"] == 2
    assert result["compact_interaction_memory_count"] == 2
    assert result["manifest_interaction_count"] == 10


def test_h01_reports_coverage_attribution_missing_when_stable_exists_without_game_sampler_counts(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import _finalize_h01_result

    result = {
        "hypothesis_id": "H01",
        "hypothesis_statement": "x",
        "decision": "PARTIALLY_VALID",
        "stable_contingency_count": 3,
        "stable_contingencies_count": 3,
        "per_game_contingency_counts": {},
        "per_sampler_contingency_counts": {},
        "missing_evidence": [],
        "evidence_diagnostics": {},
    }
    _finalize_h01_result(result, tmp_path / "h01_attr")
    assert result["coverage_attribution_missing"] is True
    assert "Contingency attribution by game/sampler is missing from current artifacts." in result["missing_evidence"]


def test_h01_does_not_report_missing_attribution_when_compact_cross_counts_exist(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import _finalize_h01_result

    result = {
        "hypothesis_id": "H01",
        "hypothesis_statement": "x",
        "decision": "PARTIALLY_VALID",
        "stable_contingency_count": 3,
        "cross_game_contingency_presence": None,
        "cross_game_contingency_count": 2,
        "cross_sampler_contingency_count": 0,
        "per_game_contingency_counts": {},
        "per_sampler_contingency_counts": {},
        "missing_evidence": [],
        "evidence_diagnostics": {},
    }
    _finalize_h01_result(result, tmp_path / "h01_attr_ok")
    assert result["coverage_attribution_missing"] is False
    assert "Contingency attribution by game/sampler is missing from current artifacts." not in result["missing_evidence"]


def test_h02_preserves_compact_direct_linkage(tmp_path: Path) -> None:
    from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h02_compact"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps(
            {
                "runs": [{"game": "g1", "sampler_name": "s1", "total_interactions": 2}],
                "validation": {
                    "memory_record_count": 2,
                    "memory_replay_candidate_count": 2,
                    "high_priority_replay_count": 1,
                    "context_contradiction_count": 1,
                    "repeated_contradiction_count": 1,
                    "prediction_error_positive_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h02_compact"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO memory_scores (node_id, replay_priority) VALUES ('M0:interaction:1', 0.9)")
        conn.execute("INSERT INTO memory_scores (node_id, replay_priority) VALUES ('M0:interaction:2', 0.4)")
        conn.execute(
            "INSERT INTO memory_edges (source_node_id, target_node_id, edge_type, weight, support_count) VALUES ('M0:interaction:1', 'contradiction:c1', 'violates_prediction', 1.0, 1)"
        )
        conn.commit()
    result = evaluate_h02_prediction_violation_attention(run_dir, tmp_path / "out_h02_compact", memory_dir=memory_dir)
    assert result["direct_replay_lift_available"] is True
    assert float(result["prediction_violation_replay_lift"]) > 1.25
    assert "Direct per-interaction prediction-error to replay-priority linkage unavailable" not in " ".join(result["missing_evidence"])


def test_h01_derives_prediction_accuracy_and_context_lift_from_prediction_results(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence

    run_dir = tmp_path / "run_h01_pred"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1", "total_interactions": 8, "memory_record_count": 8}]}),
        encoding="utf-8",
    )
    db_path = run_dir / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        store = ContingencyStore(conn)
        rows = [
            ((1, 1), 1, 10, 0),
            ((1, 1), 1, 10, 1),
            ((1, 1), 1, 10, 2),
            ((1, 1), 1, 10, 3),
            ((2, 2), 1, 20, 4),
            ((2, 2), 1, 20, 5),
            ((3, 3), 2, 30, 6),
            ((3, 3), 2, 40, 7),
        ]
        for context_signature, action, actual_family, interaction_id in rows:
            store.add_prediction_result(
                interaction_id=interaction_id,
                context_level=1,
                context_signature=context_signature,
                action=action,
                predicted_family=10,
                actual_family=actual_family,
            )
        conn.commit()
    result = evaluate_h01_contingency_emergence(run_dir, tmp_path / "out_h01_pred")
    assert result["mean_prediction_accuracy"] is not None
    assert result["mean_prediction_accuracy"] > 0.0
    assert result["context_lift_available"] is True
    assert result["mean_context_lift"] is not None
    assert result["mean_context_lift"] > 0.0
    assert result["positive_context_lift_count"] > 0


def test_h01_handles_empty_prediction_results_without_context_lift_keyerror(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence

    run_dir = tmp_path / "run_h01_empty_pred"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1", "total_interactions": 2, "memory_record_count": 2}]}),
        encoding="utf-8",
    )
    db_path = run_dir / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        store = ContingencyStore(conn)
        conn.commit()
    result = evaluate_h01_contingency_emergence(run_dir, tmp_path / "out_h01_empty_pred")
    assert result["decision"] in {"INCONCLUSIVE", "PARTIALLY_VALID", "INVALID", "VALID"}
    assert result["context_lift_available"] is False
    assert result["positive_context_lift_count"] == 0


def test_h02_uses_compact_replay_counter_when_raw_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1", "memory_replay_candidate_count": 0, "high_priority_replay_count": 0}]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "replay_queue.sqlite") as conn:
        conn.execute(
            "INSERT INTO replay_queue (replay_id, owner_type, owner_id, priority_score, reason, first_seen_global_step, last_seen_global_step, compact_payload_json) VALUES ('1', 'interaction', '1', 0.9, 'r', 1, 1, '{}')"
        )
        conn.commit()
    monkeypatch.setattr(
        "v6.hypothesis_h02_report._compute_prediction_violation_replay_lift_from_existing_db",
        lambda *args, **kwargs: {"direct_replay_lift_available": False},
    )
    result = evaluate_h02_prediction_violation_attention(run_dir, tmp_path / "out_h02", memory_dir=memory_dir)
    assert result["memory_replay_candidate_count"] == 1
    assert result["acceptance_checks"]["replay_candidates_present"] is True
    assert result["compact_counter_fallback_used"] is True


def test_h02_raw_cleanup_without_direct_linkage_is_not_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h02_streamed"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1"}]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h02_streamed"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO contradiction_clusters (cluster_id, canonical_key, support_count, first_seen_global_step, last_seen_global_step, max_prediction_error, mean_replay_priority) VALUES ('c1', 'k1', 2, 1, 2, 1.0, 0.9)")
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES ('M0:interaction:g1', 'M0', 'InteractionMemory', 'g1', 1, '{}')")
        conn.execute("INSERT INTO memory_scores (node_id, replay_priority) VALUES ('M0:interaction:g1', 0.9)")
        conn.commit()
    monkeypatch.setattr(
        "v6.hypothesis_h02_report._compute_prediction_violation_replay_lift_from_existing_db",
        lambda *args, **kwargs: {"direct_replay_lift_available": False},
    )
    result = evaluate_h02_prediction_violation_attention(run_dir, tmp_path / "out_h02_streamed", memory_dir=memory_dir)
    assert result["raw_cleanup_prevents_direct_linkage"] is True
    assert result["decision"] != "INVALID"


def test_h09_unknown_event_ratio_blocks_valid(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO future_option_motifs (motif_signature, motif_type, support_count, linked_event_count, cross_context_count, cross_game_count, mean_option_delta, mean_abs_option_delta, motif_stability_score, is_emergent) VALUES ('m1', 'enable', 3, 3, 2, 1, 1.0, 1.0, 0.8, 1)"
        )
        for idx, source in enumerate(["unknown", "unknown", "unknown", "structured_effect"], start=1):
            evidence = json.dumps({"motif_type_source": source})
            conn.execute(
                "INSERT INTO future_option_events (event_id, owner_type, owner_key, source_kind, motif_type, option_delta, option_delta_bucket, first_seen_global_step, last_seen_global_step, evidence_json) VALUES (?, 'interaction', ?, 'stable_contingency', ?, 1.0, 'positive', 1, 1, ?)",
                (f"e{idx}", f"o{idx}", "unknown" if idx <= 3 else "enable", evidence),
            )
        conn.commit()
    result = evaluate_h09_future_option_motifs(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h09", already_derived=True)
    assert result["decision"] != "VALID"
    assert result["unknown_motif_event_ratio"] > 0.20


def test_h09_live_option_delta_is_used_exactly() -> None:
    from v6.future_options import _build_future_option_event

    row = _build_future_option_event(
        owner_type="interaction",
        owner_key="i1",
        source_kind="stable_contingency",
        game="g1",
        sampler="s1",
        context_key="ctx",
        action_key="a1",
        text_fragments=["irrelevant"],
        support_count=50,
        polarity="positive",
        first_seen=1,
        last_seen=1,
        mean_prediction_error=0.0,
        mean_replay_priority=0.0,
        stability_score=0.0,
        event_id_seed="seed",
        evidence_json={},
        live_option_delta=1.7,
    )
    assert row["option_delta"] == pytest.approx(1.7)
    zero_row = _build_future_option_event(
        owner_type="interaction",
        owner_key="i2",
        source_kind="stable_contingency",
        game="g1",
        sampler="s1",
        context_key="ctx",
        action_key="a1",
        text_fragments=["irrelevant"],
        support_count=50,
        polarity="positive",
        first_seen=1,
        last_seen=1,
        mean_prediction_error=0.0,
        mean_replay_priority=0.0,
        stability_score=0.0,
        event_id_seed="seed2",
        evidence_json={},
        live_option_delta=0.0,
    )
    assert zero_row["option_delta"] == pytest.approx(0.0)
    assert zero_row["motif_type"] == "neutral"
    zero_evidence = zero_row["evidence_json"]
    if isinstance(zero_evidence, str):
        zero_evidence = json.loads(zero_evidence)
    assert zero_evidence["motif_type_source"] == "live_delta_rule"


def test_h10_attention_saturation_is_partial(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        for idx in range(5):
            conn.execute(
                "INSERT INTO future_option_attention_links (event_id, motif_signature, owner_type, owner_key, option_delta_abs, replay_priority_score, memory_priority_score, contradiction_score, high_option_change, high_attention) VALUES (?, 'm', 'interaction', ?, 2.0, 0.9, 0.0, 0.0, 1, 1)",
                (f"h{idx}", f"hi{idx}"),
            )
        for idx in range(5):
            conn.execute(
                "INSERT INTO future_option_attention_links (event_id, motif_signature, owner_type, owner_key, option_delta_abs, replay_priority_score, memory_priority_score, contradiction_score, high_option_change, high_attention) VALUES (?, 'm', 'interaction', ?, 0.1, 0.9, 0.0, 0.0, 0, 1)",
                (f"l{idx}", f"lo{idx}"),
            )
        conn.commit()
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["attention_all_high_saturation"] is True
    assert result["attention_saturation"] is True


def test_h10_all_low_attention_saturation_is_partial(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        for idx in range(5):
            conn.execute(
                "INSERT INTO future_option_events (event_id, owner_type, owner_key, source_kind, motif_type, option_delta, option_delta_bucket, first_seen_global_step, last_seen_global_step, replay_priority_score, memory_priority_score, contradiction_score, evidence_json) VALUES (?, 'interaction', ?, 'stable_contingency', 'enable', 2.0, 'large_positive', 1, 1, 0.0, 0.0, 0.0, '{}')",
                (f"h{idx}", f"hi{idx}"),
            )
            conn.execute(
                "INSERT INTO future_option_attention_links (event_id, motif_signature, owner_type, owner_key, option_delta_abs, replay_priority_score, memory_priority_score, contradiction_score, high_option_change, high_attention) VALUES (?, 'm', 'interaction', ?, 2.0, 0.0, 0.0, 0.0, 1, 0)",
                (f"h{idx}", f"hi{idx}"),
            )
        for idx in range(5):
            conn.execute(
                "INSERT INTO future_option_events (event_id, owner_type, owner_key, source_kind, motif_type, option_delta, option_delta_bucket, first_seen_global_step, last_seen_global_step, replay_priority_score, memory_priority_score, contradiction_score, evidence_json) VALUES (?, 'interaction', ?, 'stable_contingency', 'enable', 0.1, 'small_positive', 1, 1, 0.0, 0.0, 0.0, '{}')",
                (f"l{idx}", f"lo{idx}"),
            )
            conn.execute(
                "INSERT INTO future_option_attention_links (event_id, motif_signature, owner_type, owner_key, option_delta_abs, replay_priority_score, memory_priority_score, contradiction_score, high_option_change, high_attention) VALUES (?, 'm', 'interaction', ?, 0.1, 0.0, 0.0, 0.0, 0, 0)",
                (f"l{idx}", f"lo{idx}"),
            )
        conn.commit()
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10_low", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["attention_all_low_saturation"] is True
    assert result["attention_saturation"] is True
    assert "Attention signal is saturated all-low; selective attention is not demonstrated." in result["missing_evidence"]


def test_h10_degenerate_calibration_keeps_raw_high_attention_not_all_low(tmp_path: Path) -> None:
    from v6.future_options import derive_future_option_attention_links

    memory_dir = tmp_path / "memory_h10_degenerate"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES ('M0:interaction:1', 'M0', 'InteractionMemory', '1', 1, '{\"global_step\":1}')")
        conn.execute("INSERT INTO memory_scores (node_id, future_option_delta, replay_priority) VALUES ('M0:interaction:1', 0.0, 0.9)")
        conn.commit()
        derive_future_option_attention_links(conn)
        row = conn.execute("SELECT high_attention, raw_high_attention, calibrated_high_attention, attention_calibration_degenerate FROM future_option_attention_links").fetchone()
        assert row == (1, 1, 0, 1)
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10_deg", already_derived=True)
    assert result["attention_all_low_saturation"] is False
    assert result["attention_calibration_degenerate"] is True


def test_h10_falls_back_to_heuristic_events_when_live_deltas_are_zero(tmp_path: Path) -> None:
    from v6.future_options import derive_future_option_attention_links

    memory_dir = tmp_path / "memory_h10_fallback"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, attrs_json) VALUES ('M0:interaction:1', 'M0', 'InteractionMemory', '1', 1, '{\"global_step\":1}')")
        conn.execute("INSERT INTO memory_scores (node_id, future_option_delta, replay_priority) VALUES ('M0:interaction:1', 0.0, 0.0)")
        conn.execute(
            "INSERT INTO future_option_events (event_id, owner_type, owner_key, source_kind, motif_type, option_delta, option_delta_bucket, first_seen_global_step, last_seen_global_step, replay_priority_score, memory_priority_score, contradiction_score, evidence_json) VALUES ('e1', 'interaction', 'i1', 'stable_contingency', 'enable', 2.0, 'large_positive', 1, 1, 0.9, 0.0, 0.0, '{}')"
        )
        conn.commit()
        summary = derive_future_option_attention_links(conn)
    assert summary["h10_heuristic_rows_used"] > 0
    assert summary["h10_fallback_reason"] == "all_live_option_deltas_zero"


def test_h03_normalized_family_signature_reduces_numeric_singletons(tmp_path: Path) -> None:
    from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation

    run_dir = tmp_path / "run_h03_norm"
    out_dir = tmp_path / "out_h03_norm"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps(
            {
                "validation": {
                    "memory_record_count": 10,
                    "carrier_candidate_count": 0,
                    "emergent_carrier_count": 0,
                    "carrier_spatial_candidate_count": 0,
                    "carrier_object_candidate_count": 0,
                    "emergent_object_carrier_count": 0,
                    "carrier_context_action_fallback_candidate_count": 0,
                    "emergent_context_action_fallback_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    db_path = run_dir / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE transformation_families (
                family_id TEXT,
                effect_signature TEXT,
                member_count INTEGER,
                support_count INTEGER,
                action INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO transformation_families VALUES (?, ?, ?, ?, ?)",
            [
                ("f1", "[1.2,0.2,0.1,0,0]", 3, 3, 1),
                ("f2", "[3.8,0.3,0.2,0,0]", 3, 3, 1),
            ],
        )
        conn.commit()
    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)
    assert result["transformation_family_count"] == 1
    assert result["singleton_family_ratio"] == 0.0
    assert result["over_specific_singleton_count"] == 0


def test_h03_compact_derives_family_counts_from_stable_contingencies(tmp_path: Path) -> None:
    from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h03_streamed"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1"}]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h03_streamed"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count) VALUES ('c1', 'g1', 's1', 1, 1, 'e1', 10)")
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count) VALUES ('c2', 'g1', 's1', 1, 1, 'e1', 12)")
        conn.commit()
    result = evaluate_h03_transformation_family_formation(run_dir, tmp_path / "out_h03_streamed", memory_dir=memory_dir)
    assert result["evidence_source"] == "direct_streaming_manifest_and_compact_memory"
    assert result["stable_contingencies_count"] == 2
    assert result["transformation_families_count"] >= 1


def test_fold_single_sampling_db_persists_raw_contingencies_into_stable_contingencies(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_fold_contingencies"
    db_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE contingencies (
                id INTEGER PRIMARY KEY,
                context_level INTEGER,
                context_signature TEXT,
                action INTEGER,
                transformation_family INTEGER,
                support_count INTEGER,
                confidence REAL,
                prediction_attempt_count INTEGER,
                prediction_success_count INTEGER,
                prediction_accuracy REAL
            );
            CREATE TABLE prediction_results (
                interaction_id INTEGER PRIMARY KEY,
                action INTEGER,
                actual_family INTEGER
            );
            CREATE TABLE transformation_families (
                id INTEGER PRIMARY KEY,
                centroid_vector TEXT,
                support_count INTEGER
            );
            """
        )
        conn.execute("INSERT INTO transformation_families (id, centroid_vector, support_count) VALUES (7, '[1,0,0]', 3)")
        conn.execute(
            """
            INSERT INTO contingencies (
                id, context_level, context_signature, action, transformation_family, support_count, confidence,
                prediction_attempt_count, prediction_success_count, prediction_accuracy
            ) VALUES (1, 1, '[1,2,3]', 2, 7, 3, 0.9, 4, 3, 0.75)
            """
        )
        conn.execute("INSERT INTO prediction_results (interaction_id, action, actual_family) VALUES (1, 2, 7)")
        conn.commit()
    totals = fold_single_sampling_db_into_main_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
        finalize_after_fold=True,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        stable_count = conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]
        family_count = conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]
    assert totals["raw_contingency_rows_seen"] == 1
    assert totals["stable_contingencies_inserted"] == 1
    assert stable_count > 0
    assert family_count > 0


def test_fold_single_sampling_db_empty_contingency_table_uses_prediction_result_fallback(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_fold_no_m1"
    db_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE contingencies (
                id INTEGER PRIMARY KEY,
                context_level INTEGER,
                context_signature TEXT,
                action INTEGER,
                transformation_family INTEGER,
                support_count INTEGER,
                confidence REAL
            );
            CREATE TABLE prediction_results (
                interaction_id INTEGER PRIMARY KEY,
                action INTEGER,
                actual_family INTEGER
            );
            """
        )
        conn.execute("INSERT INTO prediction_results (interaction_id, action, actual_family) VALUES (1, 2, 7)")
        conn.commit()
    fold_single_sampling_db_into_main_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
        finalize_after_fold=True,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        stable_count = conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]
        family_count = conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]
    assert stable_count > 0
    assert family_count > 0


def test_compact_family_repair_creates_transformation_families_and_members(tmp_path: Path) -> None:
    from v6.memory.compact_memory import derive_missing_transformation_families_from_stable_contingencies

    memory_dir = tmp_path / "memory_h03_repair"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1', 'g1', 's1', 1, 1, 'e1', 10, 1, 2, 0.8)")
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c2', 'g1', 's1', 1, 1, 'e1', 12, 2, 3, 0.9)")
        conn.commit()
    summary = derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    assert summary["compact_family_repair_used"] is True
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        family_count = conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]
        member_count = conn.execute("SELECT COUNT(*) FROM family_members").fetchone()[0]
    assert family_count >= 1
    assert member_count == 2


def test_compact_family_repair_is_idempotent(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h03_repair_idempotent"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1', 'g1', 's1', 1, 1, 'e1', 10, 1, 2, 0.8)")
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c2', 'g1', 's1', 1, 1, 'e1', 12, 2, 3, 0.9)")
        conn.commit()
    first = derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    second = derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        family_count = conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]
        member_count = conn.execute("SELECT COUNT(*) FROM family_members").fetchone()[0]
    assert first["compact_family_repair_used"] is True
    assert second["compact_family_repair_used"] is False
    assert second["compact_family_repair_reason"] == "family_substrate_already_present"
    assert family_count >= 1
    assert member_count == 2


def test_compact_family_repair_reports_missing_m1_substrate_when_only_families_exist(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h03_missing_m1"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO transformation_families (
                family_id, canonical_signature, relaxed_signature, effect_type, action_group, polarity,
                support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score
            ) VALUES (1, 'fam:a', 'fam:a', 'raw', '1', 'unknown', 3, 1, 1, 2, 0.5)
            """
        )
        conn.execute(
            "INSERT INTO family_members (family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('fam:a', 'c1', 1, 1, 2)"
        )
        conn.commit()
    summary = derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    assert summary["stable_contingencies_count"] == 0
    assert summary["transformation_families_count"] == 1
    assert summary["family_members_count"] == 1
    assert summary["compact_m1_substrate_missing"] is True
    assert summary["compact_family_repair_reason"] == "missing_m1_substrate_with_existing_m2"


def test_compact_family_repair_enables_role_links_with_family(tmp_path: Path) -> None:
    from v6.higher_order_substrate import derive_higher_order_memory

    memory_dir = tmp_path / "memory_h03_role_repair"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1', 'g1', 's1', 1, 1, 'e1', 10, 1, 2, 0.8)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, stability_score, is_emergent) VALUES ('id1', 'carrierA', 'object', 4, 0, 1, 2, 0.8, 1)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrierA', 'contingency', 'c1', 1, 1, 2)")
        conn.commit()
    derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    derive_higher_order_memory(memory_dir=memory_dir, run_dir=None)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        family_role_links = conn.execute("SELECT COUNT(*) FROM role_links WHERE linked_type = 'family'").fetchone()[0]
    assert family_role_links > 0


def test_hypothesis_suite_report_imports_successfully() -> None:
    import v6.hypothesis_suite_report as hypothesis_suite_report

    assert callable(hypothesis_suite_report.run_hypothesis_suite_report)


def test_hypothesis_suite_writes_phase_log(tmp_path: Path, monkeypatch) -> None:
    import v6.hypothesis_suite_report as hypothesis_suite_report

    run_dir = tmp_path / "run"
    output_dir = tmp_path / "reports"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps({"runs": []}), encoding="utf-8")

    stub = {"decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": []}
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h01_contingency_emergence", lambda **kwargs: dict(stub, hypothesis_id="H01"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h02_prediction_violation_attention", lambda **kwargs: dict(stub, hypothesis_id="H02"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h03_transformation_family_formation", lambda **kwargs: dict(stub, hypothesis_id="H03"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h04_carrier_emergence", lambda **kwargs: dict(stub, hypothesis_id="H04"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h12_efficiency_emergence", lambda **kwargs: dict(stub, hypothesis_id="H12"))

    summary = hypothesis_suite_report.run_hypothesis_suite_report(
        run_dir=run_dir,
        memory_dir=None,
        output_dir=output_dir,
        scan_all_dbs=False,
        max_db_files=0,
        max_rows=10,
        epoch_id="epoch_0001",
        hypothesis_progress=False,
        hypothesis_progress_log_every=1,
    )
    assert summary["suite_total_seconds"] >= 0.0
    log_path = output_dir / "hypothesis_phase_log.jsonl"
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(line["phase"] == "H01" and line["status"] == "starting" for line in lines)
    assert any(line["phase"] == "summary_write" and line["status"] == "done" for line in lines)


def test_hypothesis_suite_progress_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    import v6.hypothesis_suite_report as hypothesis_suite_report

    run_dir = tmp_path / "run_disable"
    output_dir = tmp_path / "reports_disable"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps({"runs": []}), encoding="utf-8")
    stub = {"decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": []}
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h01_contingency_emergence", lambda **kwargs: dict(stub, hypothesis_id="H01"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h02_prediction_violation_attention", lambda **kwargs: dict(stub, hypothesis_id="H02"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h03_transformation_family_formation", lambda **kwargs: dict(stub, hypothesis_id="H03"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h04_carrier_emergence", lambda **kwargs: dict(stub, hypothesis_id="H04"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h12_efficiency_emergence", lambda **kwargs: dict(stub, hypothesis_id="H12"))

    seen: dict[str, object] = {}

    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            seen.setdefault("calls", []).append(kwargs)
        def set_postfix_str(self, *args, **kwargs):
            return None
        def update(self, *args, **kwargs):
            return None
        def close(self):
            return None

    monkeypatch.setattr(hypothesis_suite_report, "tqdm", DummyTqdm)
    hypothesis_suite_report.run_hypothesis_suite_report(
        run_dir=run_dir,
        memory_dir=None,
        output_dir=output_dir,
        scan_all_dbs=False,
        max_db_files=0,
        max_rows=10,
        hypothesis_progress=False,
    )
    assert seen["calls"][0]["disable"] is True


def test_higher_order_transfer_progress_updates_from_parent(tmp_path: Path, monkeypatch) -> None:
    from v6.higher_order_substrate import derive_role_transfer_attempts_only

    memory_dir = tmp_path / "memory_transfer_progress"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO role_neighborhood_signatures (carrier_signature, role_signature, role_type, token_json, diagnostic_token_json, linked_family_count, linked_context_count, linked_game_count, in_edge_count, out_edge_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1','r1','t','[\"a\"]','[]',1,1,1,0,0,1,1,0.7)")
        conn.execute("INSERT INTO role_neighborhood_signatures (carrier_signature, role_signature, role_type, token_json, diagnostic_token_json, linked_family_count, linked_context_count, linked_game_count, in_edge_count, out_edge_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c2','r2','t','[\"b\"]','[]',1,1,1,0,0,1,1,0.7)")
        conn.commit()
    progress_events: list[tuple[str, int | None]] = []

    class Tracker:
        def __init__(self, phase: str, total: int | None) -> None:
            self.phase = phase
            self.total = total
        def update(self, n=1, current=None, extra=None):
            progress_events.append((self.phase, current if current is not None else n))
        def close(self, extra=None):
            progress_events.append((self.phase, self.total))

    derive_role_transfer_attempts_only(
        memory_dir=memory_dir,
        workers=1,
        chunk_size=1,
        progress_factory=lambda phase, total, unit, leave=False: Tracker(phase, total),
    )
    assert any(phase == "derive_role_transfer chunks" for phase, _ in progress_events)


def test_h03_family_prediction_lift_available_and_non_negative(tmp_path: Path) -> None:
    from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation

    run_dir = tmp_path / "run_h03_lift"
    out_dir = tmp_path / "out_h03_lift"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps(
            {
                "validation": {
                    "memory_record_count": 10,
                    "carrier_candidate_count": 0,
                    "emergent_carrier_count": 0,
                    "carrier_spatial_candidate_count": 0,
                    "carrier_object_candidate_count": 0,
                    "emergent_object_carrier_count": 0,
                    "carrier_context_action_fallback_candidate_count": 0,
                    "emergent_context_action_fallback_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    db_path = run_dir / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE transformation_families (
                family_id TEXT,
                effect_signature TEXT,
                member_count INTEGER,
                support_count INTEGER,
                action INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO transformation_families VALUES (?, ?, ?, ?, ?)",
            [
                ("f1", "[1.2,0.2,0.1,0,0]", 4, 4, 1),
                ("f2", "[3.8,0.3,0.2,0,0]", 4, 4, 1),
            ],
        )
        conn.execute(
            """
            CREATE TABLE prediction_results (
                interaction_id INTEGER,
                context_signature TEXT,
                action INTEGER,
                predicted_family TEXT,
                actual_family TEXT
            )
            """
        )
        rows = []
        for idx in range(1, 3):
            rows.append((idx, "ctx", 1, "f1", "f1"))
        for idx in range(3, 5):
            rows.append((idx, "ctx", 1, "f2", "f1"))
        for idx in range(5, 7):
            rows.append((idx, "ctx", 1, "f2", "f2"))
        for idx in range(7, 9):
            rows.append((idx, "ctx", 1, "f1", "f2"))
        conn.executemany("INSERT INTO prediction_results VALUES (?, ?, ?, ?, ?)", rows)
        conn.commit()
    result = evaluate_h03_transformation_family_formation(run_dir, out_dir)
    assert result["family_prediction_lift_available"] is True
    assert result["family_prediction_lift_mean"] is not None
    assert result["family_prediction_lift_mean"] >= 0.0
    assert result["family_prediction_lift_mean"] > 0.0


def test_h03_missing_evidence_flags_are_exposed() -> None:
    from v6.hypothesis_h03_report import _populate_h03_evidence_lists

    result = {
        "missing_evidence": [],
        "discovered_contingency_count": 10,
        "transformation_family_count": 5,
        "family_max_member_count": 1,
        "compression_ratio": 1.2,
        "compression_gain": 0.1,
        "pre_object_condition_satisfied": True,
        "families_merged_across_shards": 0,
        "family_prediction_lift_mean": None,
        "max_rows_applied": True,
        "singleton_family_ratio": 0.75,
        "singleton_family_count": 10,
        "singleton_families_by_action": {"unknown": 3},
        "singleton_family_diagnostics": {"over_specific_context_count": 3},
    }
    _populate_h03_evidence_lists(result)
    assert result["family_prediction_lift_available"] is False
    assert result["h03_row_cap_applied"] is True
    assert result["singleton_action_metadata_incomplete"] is True
    assert result["singleton_context_overspecific"] is True
    assert "H03 family prediction-lift evidence is unavailable." in result["missing_evidence"]
    assert "H03 direct family evidence was row-capped; inspect more rows or use scan-all/full max-rows." in result["missing_evidence"]
    assert "H03 singleton action metadata contains unknown actions." in result["missing_evidence"]
    assert "H03 singleton families include over-specific context signatures." in result["missing_evidence"]


def test_h03_missing_prediction_lift_is_not_valid(tmp_path: Path) -> None:
    from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h03_missing_lift"
    out_dir = tmp_path / "out_h03_missing_lift"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"validation": {"stable_contingency_count": 2, "unique_transformation_families": 1, "emergent_object_carrier_count": 0}, "runs": [{"game": "g1", "sampler_name": "s1"}]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h03_missing_lift"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1','g1','s1',1,1,'e1',5,1,2,0.5)")
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c2','g1','s1',1,1,'e1',5,1,2,0.5)")
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam',10,2,1,2,0.8)")
        conn.execute("INSERT INTO family_members (family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('fam','c1',5,1,2)")
        conn.execute("INSERT INTO family_members (family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('fam','c2',5,1,2)")
        conn.commit()
    result = evaluate_h03_transformation_family_formation(run_dir, out_dir, memory_dir=memory_dir)
    assert result["decision"] == "PARTIALLY_VALID"
    assert "H03 family prediction-lift evidence is unavailable." in result["missing_evidence"]


def test_h03_negative_prediction_lift_is_invalid(tmp_path: Path) -> None:
    import v6.hypothesis_h03_report as h03_report

    run_dir = tmp_path / "run_h03_negative_lift"
    out_dir = tmp_path / "out_h03_negative_lift"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"validation": {"stable_contingency_count": 2, "unique_transformation_families": 1, "emergent_object_carrier_count": 0}, "runs": []}),
        encoding="utf-8",
    )
    original = h03_report.compute_h03_family_metrics_from_existing_artifacts
    h03_report.compute_h03_family_metrics_from_existing_artifacts = lambda *args, **kwargs: {
        "usable_direct_family_evidence": True,
        "discovered_contingency_count": 4,
        "transformation_family_count": 2,
        "family_member_count_total": 6,
        "family_max_member_count": 3,
        "compression_ratio": 1.5,
        "compression_gain": 1.0,
        "singleton_family_ratio": 0.0,
        "family_prediction_lift_mean": -0.1,
        "emergent_object_carrier_count": 0,
        "emergent_context_action_fallback_count": 0,
        "singleton_family_count": 0,
        "max_rows_applied": False,
    }
    try:
        result = h03_report.evaluate_h03_transformation_family_formation(run_dir, out_dir)
    finally:
        h03_report.compute_h03_family_metrics_from_existing_artifacts = original
    assert result["decision"] == "INVALID"
    assert "H03 family prediction-lift is negative." in result["missing_evidence"]


def test_h03_backfills_prediction_lift_for_existing_families(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h03_backfill"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO stable_contingencies (
                canonical_key, game, sampler, context_level, action, effect_signature,
                support_count, first_seen_global_step, last_seen_global_step, stability_score,
                prediction_accuracy, prediction_error_before, prediction_error_after
            ) VALUES
            ('c1','g','s',1,1,'fam',5,1,2,0.5,0.8,0.4,0.1),
            ('c2','g','s',1,1,'fam',5,1,2,0.5,0.6,0.5,0.2)
            """
        )
        conn.execute(
            "INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam',10,2,1,2,0.5)"
        )
        conn.execute(
            "INSERT INTO family_members (family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('fam','c1',5,1,2)"
        )
        conn.execute(
            "INSERT INTO family_members (family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('fam','c2',5,1,2)"
        )
        conn.commit()
    summary = derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        row = conn.execute(
            "SELECT prediction_lift, prediction_accuracy_mean, prediction_error_before_mean, prediction_error_after_mean FROM transformation_families WHERE canonical_signature = 'fam'"
        ).fetchone()
    assert summary["compact_family_prediction_lift_backfill_used"] is True
    assert summary["compact_family_prediction_lift_backfill_count"] == 1
    assert row is not None
    assert row[0] is not None
    assert row[1] is not None
    assert row[2] is not None
    assert row[3] is not None


def test_h04_uses_compact_temporal_fallback(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_fallback"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,5,6,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, stability_score, is_emergent) VALUES ('c1','carrier1','object',5,2,7,8,0.8,1)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier1','family','fam1',1,7,8)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier1','context','ctx1',1,7,8)")
        conn.commit()
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e1','carrier:carrier1','b','explains',1,1,1,1.0)")
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e2','carrier:carrier1','b','anchors',1,1,1,1.0)")
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04")
    assert result["core_metrics"]["h04_temporal_source"] == "compact_table_fallback"
    assert result["core_metrics"]["h03_before_h04"] is True
    assert result["core_metrics"]["h03_before_h04_usable"] is True
    assert result["core_metrics"]["first_usable_emergent_carrier_step"] == 7


def test_h04_blocker_reads_temporal_order_from_core_metrics() -> None:
    from v6.hypothesis_suite_report import _blocker_flags_for_result

    flags = _blocker_flags_for_result(
        "H04",
        {
            "decision": "PARTIALLY_VALID",
            "core_metrics": {"h03_before_h04": True},
            "missing_evidence": [],
        },
    )
    assert flags["h04_missing_temporal_order"] is False
    assert flags["h04_temporal_order_failed"] is False


def test_h04_temporal_failure_sets_failed_not_missing() -> None:
    from v6.hypothesis_suite_report import _blocker_flags_for_result

    flags = _blocker_flags_for_result(
        "H04",
        {
            "decision": "PARTIALLY_VALID",
            "core_metrics": {"h03_before_h04": False},
            "missing_evidence": [],
        },
    )
    assert flags["h04_missing_temporal_order"] is False
    assert flags["h04_temporal_order_failed"] is True


def test_h05_uses_compact_temporal_fallback_without_artificial_step_one(tmp_path: Path) -> None:
    from v6.hypothesis_h05_report import evaluate_h05_role_emergence

    memory_dir = tmp_path / "memory_h05_fallback"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, stability_score, is_emergent) VALUES ('c1','carrier1','object',5,2,7,8,0.8,1)")
        conn.execute("INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, linked_context_count, cross_game_count, cross_context_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r1','role',5,2,1,2,1,2,11,12,0.8,1)")
        conn.commit()
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05", already_derived=True)
    assert result["h05_temporal_source"] == "compact_table_fallback"
    assert result["first_emergent_role_step"] == 11
    assert result["first_emergent_role_step"] != 1
    assert result["h04_before_h05"] is True


def test_h05_valid_is_demoted_when_h04_invalid() -> None:
    from v6.hypothesis_suite_report import _apply_higher_order_dependency_gates

    h04 = {"decision": "INVALID"}
    h05 = {"decision": "VALID", "missing_evidence": []}
    h06 = {"decision": "VALID", "missing_evidence": []}
    h07 = {"decision": "PARTIALLY_VALID"}
    h08 = {"decision": "PARTIALLY_VALID"}
    h09 = {"decision": "PARTIALLY_VALID"}
    h10 = {"decision": "PARTIALLY_VALID"}
    h11 = {"decision": "PARTIALLY_VALID"}
    h05, h06, *_rest = _apply_higher_order_dependency_gates(h04, h05, h06, h07, h08, h09, h10, h11)
    assert h05["decision"] == "PARTIALLY_VALID"
    assert h05["h05_depends_on_invalid_h04"] is True
    assert "H05 depends on invalid H04 carrier emergence." in h05["missing_evidence"]
    assert h06["decision"] == "PARTIALLY_VALID"


def test_skipped_fast_mode_has_no_scientific_blockers() -> None:
    from v6.hypothesis_suite_report import _blocker_flags_for_result

    flags = _blocker_flags_for_result("H07", {"decision": "SKIPPED_FAST_MODE"})
    assert flags["skipped_fast_mode"] is True
    assert flags["h07_no_promoted_concepts"] is False
    assert flags["h09_no_future_option_events"] is False


def test_h02_missing_linkage_sets_blocker() -> None:
    from v6.hypothesis_suite_report import _blocker_flags_for_result

    flags = _blocker_flags_for_result(
        "H02",
        {
            "decision": "INSUFFICIENT_EVIDENCE",
            "direct_replay_lift_available": False,
            "raw_cleanup_prevents_direct_linkage": True,
            "missing_evidence": ["direct linkage unavailable after raw cleanup"],
        },
    )
    assert flags["h02_missing_direct_linkage"] is True


def test_h04_overconnected_carrier_is_not_usable(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_overconnected"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam0',4,2,1,2,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, stability_score, is_emergent) VALUES ('c1','carrier1','object',5,51,2,3,0.8,1)")
        for idx in range(51):
            conn.execute(
                "INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier1','family',?,1,2,3)",
                (f"fam{idx}",),
            )
        conn.commit()
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e1','carrier:carrier1','b','explains',1,1,1,1.0)")
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e2','carrier:carrier1','b','anchors',1,1,1,1.0)")
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_over")
    assert result["core_metrics"]["overconnected_carrier_count"] == 1
    assert result["core_metrics"]["usable_carrier_count"] == 0
    assert result["decision"] != "VALID"


def test_h04_uses_usable_emergent_carrier_timing_for_decision(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_usable"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,10,12,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('bad','carrier_bad','object',5,200,7,8,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('good','carrier_good','object',5,2,14,15,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('ok1','carrier_ok1','object',5,1,14,15,'real_evidence',0.8,0)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('ok2','carrier_ok2','object',5,1,14,15,'real_evidence',0.8,0)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('ok3','carrier_ok3','object',5,1,14,15,'real_evidence',0.8,0)")
        for idx in range(200):
            conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_bad','family',?,1,7,8)", (f'fam{idx}',))
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','family','fam1',1,14,15)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','context','ctx1',1,14,15)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','context','ctx2',1,14,15)")
        conn.commit()
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('g1','carrier:carrier_good','b','explains',1,1,1,1.0)")
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('g2','carrier:carrier_good','b','anchors',1,1,1,1.0)")
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_usable")
    assert result["h03_before_h04_all"] is False
    assert result["h03_before_h04_usable"] is True
    assert result["decision"] == "VALID"


def test_h04_valid_is_downgraded_by_graph_explosion(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_graph_explosion"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,10,12,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('good','carrier_good','object',5,2,14,15,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','family','fam1',1,14,15)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','context','ctx1',1,14,15)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','context','ctx2',1,14,15)")
        conn.commit()
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('g1','carrier:carrier_good','b','explains',1,1,1,1.0)")
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('g2','carrier:carrier_good','b','anchors',1,1,1,1.0)")
        for idx in range(30):
            conn.execute(
                "INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES (?, 'other', 'b', 'explains',1,1,1,1.0)",
                (f"extra{idx}",),
            )
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_graph_explosion")
    assert result["h04_graph_quality_pass"] is False
    assert result["decision"] == "PARTIALLY_VALID"
    assert "H04 carrier graph is overconnected; remapping/edge explosion prevents robust VALID classification." in result["missing_evidence"]


def test_h04_usable_edge_counts_do_not_fallback_to_total_graph_edges(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_no_usable_edges"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,10,12,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('good','carrier_good','object',5,2,14,15,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','family','fam1',1,14,15)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','context','ctx1',1,14,15)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','context','ctx2',1,14,15)")
        conn.commit()
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('g1','carrier:other','b','explains',1,1,1,1.0)")
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('g2','carrier:other','b','anchors',1,1,1,1.0)")
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_no_usable_edges")
    assert result["usable_carrier_explains_edge_count"] == 0
    assert result["usable_carrier_anchors_edge_count"] == 0
    assert "Usable carriers have no usable explains/anchors graph edges; carrier graph IDs may be mismatched or graph projection failed." in result["missing_evidence"]


def test_h04_graph_quality_warning_is_reported_even_when_already_partial(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_partial_quality"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,10,12,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('bad','carrier_bad','object',5,100,14,15,'real_evidence',0.8,1)")
        for idx in range(60):
            conn.execute(
                "INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_bad','family',?,1,14,15)",
                (f"fam{idx}",),
            )
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_partial_quality")
    assert result["decision"] == "PARTIALLY_VALID"
    assert result["h04_graph_quality_pass"] is False
    assert "H04 carrier graph is overconnected; remapping/edge explosion prevents robust VALID classification." in result["missing_evidence"]


def test_h08_candidate_proxy_only_is_insufficient_evidence(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h08_proxy_only"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO concept_candidates (concept_signature, concept_type, support_count, linked_role_count, linked_carrier_count, linked_family_count, transfer_success_count, strong_transfer_success_count, cross_game_count, cross_context_count, compression_gain, explanatory_reach, promotion_score, first_seen_global_step, last_seen_global_step, is_promoted) VALUES ('c1','x',2,1,1,1,1,0,1,1,1.0,1.0,0.1,1,2,0)")
        conn.execute("INSERT INTO world_model_components (component_signature, component_type, node_count, edge_count, linked_concept_count, linked_role_count, linked_family_count, linked_carrier_count, cross_context_count, cross_game_count, explanatory_coverage, prediction_support_count, contradiction_coverage_count, coherence_score, candidate_only, predicted_outcome_count, predicted_outcome_count_is_proxy, first_seen_global_step, last_seen_global_step, is_coherent) VALUES ('w1','proxy',1,1,1,1,1,1,1,1,0.1,1,0,0.1,1,1,1,1,2,0)")
        conn.commit()
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08_proxy", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["candidate_proxy_only"] is True


def test_h08_no_world_model_components_is_insufficient_evidence(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h08_none"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO concept_candidates (concept_signature, concept_type, support_count, linked_role_count, linked_carrier_count, linked_family_count, transfer_success_count, strong_transfer_success_count, cross_game_count, cross_context_count, compression_gain, explanatory_reach, promotion_score, first_seen_global_step, last_seen_global_step, is_promoted) VALUES ('c1','x',2,1,1,1,1,1,1,1,1.0,1.0,0.8,1,2,1)")
        conn.commit()
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08_none", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "no world-model components available" in result["missing_evidence"]


def test_h02_compact_replay_lift_can_be_infinite(tmp_path: Path) -> None:
    from v6.hypothesis_h02_report import _extract_h02_compact_metrics

    memory_dir = tmp_path / "memory_h02_inf"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO memory_scores (node_id, replay_priority) VALUES ('M0:interaction:1', 1.0)")
        conn.execute("INSERT INTO memory_scores (node_id, replay_priority) VALUES ('M0:interaction:2', 0.0)")
        conn.execute("INSERT INTO memory_edges (source_node_id, target_node_id, edge_type, support_count) VALUES ('M0:interaction:1', 'p', 'violates_prediction', 1)")
        conn.commit()
    metrics = _extract_h02_compact_metrics(memory_dir)
    assert metrics["direct_replay_lift_available"] is True
    assert math.isinf(metrics["prediction_violation_replay_lift"])


def test_h06_too_few_transfer_attempts_is_insufficient_evidence(tmp_path: Path) -> None:
    from v6.hypothesis_h06_report import evaluate_h06_role_transfer

    memory_dir = tmp_path / "memory_h06_too_few"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r1','x',2,1,1,1,2,0.8,1)")
        for idx in range(3):
            conn.execute(
                "INSERT INTO role_transfer_attempts (attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key, target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature, observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason, best_margin, source_carrier_count, candidate_role_count, first_seen_global_step, last_seen_global_step) VALUES (?, 'r1', 'cross_game', 'game', 'g1', 'game', 'g2', 'c1', 'r1', 'r1', 0.9, 0.9, 1, NULL, 0.2, 2, 2, 1, 1)",
                (f"a{idx}",),
            )
        conn.commit()
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h06_too_few", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "too few role transfer attempts for H06 evaluation" in result["missing_evidence"]


def test_h07_no_successful_transfers_and_no_concepts_is_insufficient_evidence(tmp_path: Path) -> None:
    from v6.hypothesis_h07_report import evaluate_h07_concept_emergence

    memory_dir = tmp_path / "memory_h07_none"
    ensure_memory_layout(memory_dir)
    result = evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07_none", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "no successful role transfers and no concept candidates available" in result["missing_evidence"]


def test_h05_valid_is_demoted_when_h04_not_valid_or_graph_quality_failed() -> None:
    from v6.hypothesis_suite_report import _apply_higher_order_dependency_gates

    h04 = {"decision": "PARTIALLY_VALID", "h04_graph_quality_pass": False, "usable_emergent_carrier_count": 1}
    h05 = {"decision": "VALID", "missing_evidence": []}
    h06 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h07 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h08 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h09 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h10 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h11 = {"decision": "PARTIALLY_VALID", "missing_evidence": []}
    h05, *_ = _apply_higher_order_dependency_gates(h04, h05, h06, h07, h08, h09, h10, h11)
    assert h05["decision"] == "PARTIALLY_VALID"
    assert "H05 cannot be fully VALID until H04 has VALID usable carrier emergence with graph-quality pass." in h05["missing_evidence"]


def test_h09_h10_h11_missing_tables_return_insufficient_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import v6.hypothesis_h09_report as h09_report
    import v6.hypothesis_h10_report as h10_report
    import v6.hypothesis_h11_report as h11_report

    memory_dir = tmp_path / "memory_missing_tables"
    memory_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("CREATE TABLE placeholder (id INTEGER)")
        conn.commit()
    h09 = h09_report.evaluate_h09_future_option_motifs(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h09_missing", already_derived=True)
    h10 = h10_report.evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10_missing", already_derived=True)
    h11 = h11_report.evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h11_missing", already_derived=True)
    assert h09["decision"] == "INSUFFICIENT_EVIDENCE"
    assert h10["decision"] == "INSUFFICIENT_EVIDENCE"
    assert h11["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "Missing expected compact-memory table(s):" in h09["missing_evidence"][0]
    assert "Missing expected compact-memory table(s):" in h10["missing_evidence"][0]
    assert "Missing expected compact-memory table(s):" in h11["missing_evidence"][0]


def test_h_reports_do_not_create_current_state_when_memory_missing(tmp_path: Path) -> None:
    from v6.hypothesis_h07_report import evaluate_h07_concept_emergence
    from v6.hypothesis_h08_report import evaluate_h08_world_model_coherence
    from v6.hypothesis_h09_report import evaluate_h09_future_option_motifs
    from v6.hypothesis_h10_report import evaluate_h10_future_option_attention
    from v6.hypothesis_h11_report import evaluate_h11_future_option_transfer_concepts

    memory_dir = tmp_path / "missing_memory"
    assert not (memory_dir / "current_state.sqlite").exists()
    evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07_missing", already_derived=True)
    evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h08_missing", already_derived=True)
    evaluate_h09_future_option_motifs(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h09_missing", already_derived=True)
    evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10_missing", already_derived=True)
    evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h11_missing", already_derived=True)
    assert not (memory_dir / "current_state.sqlite").exists()


def test_suite_does_not_run_family_repair_unless_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import v6.hypothesis_suite_report as suite_report

    run_dir = tmp_path / "run_suite_no_repair"
    output_dir = tmp_path / "out_suite_no_repair"
    memory_dir = tmp_path / "memory_suite_no_repair"
    run_dir.mkdir()
    ensure_memory_layout(memory_dir)
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps({"runs": []}), encoding="utf-8")
    called = {"count": 0}
    monkeypatch.setattr(suite_report, "derive_missing_transformation_families_from_stable_contingencies", lambda memory_dir: called.__setitem__("count", called["count"] + 1) or {"compact_family_repair_used": True})
    stub = {"decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": []}
    monkeypatch.setattr(suite_report, "evaluate_h01_contingency_emergence", lambda **kwargs: dict(stub, hypothesis_id="H01"))
    monkeypatch.setattr(suite_report, "evaluate_h02_prediction_violation_attention", lambda **kwargs: dict(stub, hypothesis_id="H02"))
    monkeypatch.setattr(suite_report, "evaluate_h03_transformation_family_formation", lambda **kwargs: dict(stub, hypothesis_id="H03"))
    monkeypatch.setattr(suite_report, "evaluate_h04_carrier_emergence", lambda **kwargs: dict(stub, hypothesis_id="H04"))
    monkeypatch.setattr(suite_report, "evaluate_h05_role_emergence", lambda **kwargs: dict(stub, hypothesis_id="H05"))
    monkeypatch.setattr(suite_report, "evaluate_h12_efficiency_emergence", lambda **kwargs: dict(stub, hypothesis_id="H12"))
    summary = suite_report.run_hypothesis_suite_report(
        run_dir=run_dir,
        memory_dir=memory_dir,
        output_dir=output_dir,
        scan_all_dbs=False,
        max_db_files=0,
        max_rows=0,
        allow_memory_repair=False,
        hypothesis_progress=False,
        suite_mode="fast",
    )
    assert called["count"] == 0
    assert summary["memory_repair_ran_during_report"] is False


def test_h04_graph_quality_false_without_usable_emergent_carriers(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_no_usable_emergent"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,10,12,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('bad','carrier_bad','object',1,100,14,15,'real_evidence',0.8,1)")
        for idx in range(60):
            conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_bad','family',?,1,14,15)", (f'fam{idx}',))
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_no_usable_emergent")
    assert result["usable_emergent_carrier_count"] == 0
    assert result["h04_graph_quality_pass"] is False


def test_h04_graph_quality_false_without_usable_graph_edges(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_no_edges"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,10,12,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('good','carrier_good','object',5,1,14,15,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','family','fam1',1,14,15)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','context','ctx1',1,14,15)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier_good','context','ctx2',1,14,15)")
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_no_edges")
    assert result["usable_carrier_explains_edge_count"] == 0
    assert result["usable_carrier_anchors_edge_count"] == 0
    assert result["h04_graph_quality_pass"] is False


def test_h10_unbounded_lift_can_be_valid(tmp_path: Path) -> None:
    from v6.hypothesis_h10_report import evaluate_h10_future_option_attention

    memory_dir = tmp_path / "memory_h10_unbounded"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        for idx in range(5):
            conn.execute(
                """
                INSERT INTO future_option_events (
                    event_id, owner_type, owner_key, game, sampler, context_key, action_key,
                    source_kind, motif_type, option_delta, first_seen_global_step, last_seen_global_step
                ) VALUES (?, 'family', 'k', 'g', 's', 'ctx', 'a', 'x', 'm', 1.0, 1, 1)
                """,
                (f"e{idx}",),
            )
            conn.execute(
                """
                INSERT INTO future_option_attention_links (
                    event_id, motif_signature, owner_type, owner_key, option_delta_abs,
                    replay_priority_score, memory_priority_score, contradiction_score,
                    high_option_change, high_attention, raw_high_attention, calibrated_high_attention,
                    source_label, attention_signal_source, attention_score, attention_score_percentile,
                    attention_threshold_method, attention_calibration_degenerate, first_seen_global_step, last_seen_global_step
                ) VALUES (?, 'm', 'family', 'k', 1.0, 1.0, 0.2, 0.0, 1, 1, 1, 1, 'live', 'raw', 1.0, 1.0, 'fixed', 0, 1, 1)
                """,
                (f"e{idx}",),
            )
        for idx in range(5, 10):
            conn.execute(
                """
                INSERT INTO future_option_events (
                    event_id, owner_type, owner_key, game, sampler, context_key, action_key,
                    source_kind, motif_type, option_delta, first_seen_global_step, last_seen_global_step
                ) VALUES (?, 'family', 'k', 'g', 's', 'ctx', 'a', 'x', 'm', 0.1, 1, 1)
                """,
                (f"e{idx}",),
            )
            conn.execute(
                """
                INSERT INTO future_option_attention_links (
                    event_id, motif_signature, owner_type, owner_key, option_delta_abs,
                    replay_priority_score, memory_priority_score, contradiction_score,
                    high_option_change, high_attention, raw_high_attention, calibrated_high_attention,
                    source_label, attention_signal_source, attention_score, attention_score_percentile,
                    attention_threshold_method, attention_calibration_degenerate, first_seen_global_step, last_seen_global_step
                ) VALUES (?, 'm', 'family', 'k', 0.1, 0.0, 0.1, 0.0, 0, 0, 0, 0, 'live', 'raw', 0.0, 0.0, 'fixed', 0, 1, 1)
                """,
                (f"e{idx}",),
            )
        conn.commit()
    result = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h10_unbounded", already_derived=True)
    assert result["option_attention_lift_unbounded"] is True
    assert result["decision"] == "VALID"


def test_h09_zero_events_with_stable_substrate_reports_derivation_failure(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h09_zero_events"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1','g1','s1',1,1,'e1',5,1,2,0.5)")
        conn.execute("INSERT INTO memory_summary (key, value_json) VALUES ('future_option_derivation_summary', ?)", (json.dumps({"stable_contingency_rows_seen": 1, "stable_contingency_events_inserted": 0, "future_option_events_inserted_total": 0}),))
        conn.commit()
    result = evaluate_h09_future_option_motifs(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h09_zero", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "Future-option derivation produced zero events despite available substrate." in result["missing_evidence"]


def test_h10_blocked_by_h09_when_future_option_events_absent(tmp_path: Path) -> None:
    result = evaluate_h10_future_option_attention(memory_dir=tmp_path / "memory_h10_blocked", run_dir=None, output_dir=tmp_path / "h10_blocked", already_derived=True)
    assert result["h10_blocked_by_h09"] is True
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"


def test_h11_reports_blockers_when_motifs_or_promoted_concepts_absent(tmp_path: Path) -> None:
    from v6.hypothesis_h11_report import evaluate_h11_future_option_transfer_concepts

    memory_dir = tmp_path / "memory_h11_blockers"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO role_transfer_attempts (attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key, target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature, observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason, best_margin, source_carrier_count, candidate_role_count, first_seen_global_step, last_seen_global_step) VALUES ('1','r1','cross_game','game','g1','game','g2','c1','r2','r2',0.9,0.9,1,NULL,0.2,2,2,1,1)")
        conn.commit()
    result = evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h11_blockers", already_derived=True)
    assert result["h11_blocked_by_no_motifs"] is True
    assert result["h11_blocked_by_no_promoted_concepts"] is True


def test_hypothesis_suite_fast_mode_skips_expensive_hypotheses(tmp_path: Path, monkeypatch) -> None:
    import v6.hypothesis_suite_report as hypothesis_suite_report

    run_dir = tmp_path / "run_fast_suite"
    output_dir = tmp_path / "out_fast_suite"
    memory_dir = tmp_path / "memory_fast_suite"
    run_dir.mkdir()
    ensure_memory_layout(memory_dir)
    (run_dir / "interaction_sampling_v05c_report.json").write_text(json.dumps({"runs": []}), encoding="utf-8")
    called = {"transfer": 0}
    stub = {"decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": []}
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h01_contingency_emergence", lambda **kwargs: dict(stub, hypothesis_id="H01"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h02_prediction_violation_attention", lambda **kwargs: dict(stub, hypothesis_id="H02"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h03_transformation_family_formation", lambda **kwargs: dict(stub, hypothesis_id="H03"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h04_carrier_emergence", lambda **kwargs: dict(stub, hypothesis_id="H04"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h05_role_emergence", lambda **kwargs: dict(stub, hypothesis_id="H05"))
    monkeypatch.setattr(hypothesis_suite_report, "evaluate_h12_efficiency_emergence", lambda **kwargs: dict(stub, hypothesis_id="H12"))
    monkeypatch.setattr(hypothesis_suite_report, "derive_role_candidates_only", lambda **kwargs: {})
    monkeypatch.setattr(hypothesis_suite_report, "derive_role_transfer_attempts_only", lambda **kwargs: called.__setitem__("transfer", called["transfer"] + 1))
    summary = hypothesis_suite_report.run_hypothesis_suite_report(
        run_dir=run_dir,
        memory_dir=memory_dir,
        output_dir=output_dir,
        scan_all_dbs=False,
        max_db_files=0,
        max_rows=100,
        suite_mode="fast",
        hypothesis_progress=False,
    )
    assert called["transfer"] == 0
    assert summary["H06 decision"] == "SKIPPED_FAST_MODE"


def test_compact_memory_creates_requested_indexes(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_indexes"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        graph_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    assert "idx_carrier_links_type_key" in names
    assert "idx_role_links_signature_type_key" in names
    assert "idx_role_transfer_kind_scope" in names
    assert "idx_future_option_links_motif_type_key" in names
    assert "idx_trajectory_efficiency_scope" in names
    assert "idx_carrier_candidates_timing_source" in names
    assert "idx_graph_edges_type_source" in graph_names


def test_carrier_candidates_schema_has_timing_source_after_migration(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_carrier_schema"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(carrier_candidates)").fetchall()}
    assert "carrier_timing_source" in columns


def test_h07_reports_role_skip_reasons_when_concepts_zero(tmp_path: Path) -> None:
    from v6.hypothesis_h07_report import evaluate_h07_concept_emergence

    memory_dir = tmp_path / "memory_h07"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, linked_context_count, cross_game_count, cross_context_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r1', 'role', 5, 1, 0, 1, 0, 1, 1, 2, 0.8, 1)"
        )
        conn.execute(
            "INSERT INTO role_links (role_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('r1', 'carrier', 'c1', 1, 1, 2)"
        )
        conn.execute(
            "INSERT INTO role_transfer_attempts (attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key, target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature, observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason, best_margin, source_carrier_count, candidate_role_count, first_seen_global_step, last_seen_global_step) VALUES ('1', 'r1', 'k', 'game', 'g1', 'context', 'ctx', 'c2', 'r2', 'r2', 0.8, 0.8, 1, NULL, 0.2, 2, 2, 1, 2)"
        )
        conn.commit()
    result = evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07", already_derived=True)
    assert result["concept_candidate_count"] == 0
    assert result["roles_skipped_missing_family_links"] > 0
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    txt = (tmp_path / "h07" / "h07_concept_emergence_report.txt").read_text(encoding="utf-8")
    assert "roles skipped missing family links:" in txt


def test_h07_after_compact_family_repair_reports_remaining_blocker_not_missing_family(tmp_path: Path) -> None:
    from v6.hypothesis_h07_report import evaluate_h07_concept_emergence
    from v6.memory.compact_memory import derive_missing_transformation_families_from_stable_contingencies

    memory_dir = tmp_path / "memory_h07_repaired"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1', 'g1', 's1', 1, 1, 'e1', 10, 1, 2, 0.8)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, stability_score, is_emergent) VALUES ('id1', 'carrierA', 'object', 4, 0, 1, 2, 0.8, 1)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrierA', 'contingency', 'c1', 1, 1, 2)")
        conn.execute("INSERT INTO role_transfer_attempts (attempt_id, role_signature, transfer_kind, source_scope_type, source_scope_key, target_scope_type, target_scope_key, target_carrier_signature, predicted_role_signature, observed_role_signature, similarity_score, transfer_score, reuse_success, failure_reason, best_margin, source_carrier_count, candidate_role_count, first_seen_global_step, last_seen_global_step) VALUES ('1', 'placeholder', 'k', 'game', 'g1', 'context', 'ctx', 'carrierA', 'r2', 'r2', 0.8, 0.8, 1, NULL, 0.2, 2, 2, 1, 2)")
        conn.commit()
    derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    result = evaluate_h07_concept_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h07_repaired", already_derived=False)
    assert result["roles_skipped_missing_family_links"] == 0


def test_h03_uses_repaired_compact_evidence_without_raw_dbs(tmp_path: Path) -> None:
    from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation
    from v6.memory.compact_memory import derive_missing_transformation_families_from_stable_contingencies
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h03_repaired_only"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1"}]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h03_repaired_only"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1', 'g1', 's1', 1, 1, 'e1', 10, 1, 2, 0.8)")
        conn.commit()
    derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    result = evaluate_h03_transformation_family_formation(run_dir, tmp_path / "out_h03_repaired_only", memory_dir=memory_dir)
    assert result["decision"] != "INSUFFICIENT_EVIDENCE"
    assert result["transformation_family_count"] > 0


def test_h01_compact_memory_contingencies_do_not_force_invalid(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h01_compact"
    run_dir.mkdir()
    memory_dir = tmp_path / "memory_h01_compact"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, prediction_accuracy, normalized_contingency_key) VALUES ('c1', 'g1', 's1', 1, 1, 'e1', 3, 0.75, 'k1')"
        )
        conn.execute(
            "INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, prediction_accuracy, normalized_contingency_key) VALUES ('c2', 'g2', 's2', 1, 1, 'e1', 4, 0.80, 'k1')"
        )
        conn.execute(
            "INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json) VALUES ('M0:interaction:g1', 'M0', 'InteractionMemory', 'g1', 1, 1, 1, '{}')"
        )
        conn.execute("INSERT INTO memory_summary (key, value_json) VALUES ('total_interactions_seen', '2')")
        conn.commit()
    result = evaluate_h01_contingency_emergence(run_dir, tmp_path / "out_h01_compact", memory_dir=memory_dir)
    assert result["decision"] != "INVALID"
    assert int(result["discovered_contingency_count"] or 0) > 0


def test_h01_direct_streaming_missing_m1_is_insufficient_evidence_not_invalid(tmp_path: Path) -> None:
    from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h01_missing_m1"
    run_dir.mkdir()
    memory_dir = tmp_path / "memory_h01_missing_m1"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json) VALUES ('M0:interaction:g1', 'M0', 'InteractionMemory', 'g1', 1, 1, 1, '{}')"
        )
        conn.execute("INSERT INTO memory_summary (key, value_json) VALUES ('total_interactions_seen', '1')")
        conn.execute(
            "INSERT INTO memory_summary (key, value_json) VALUES ('fold_summary', ?)",
            (json.dumps({"raw_contingency_rows_seen": 0, "raw_dbs_without_contingency_table": 0}),),
        )
        conn.commit()
    result = evaluate_h01_contingency_emergence(run_dir, tmp_path / "out_h01_missing_m1", memory_dir=memory_dir)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["compact_m1_substrate_missing"] is True
    assert "compact M1 contingency substrate missing after direct-streaming raw cleanup" in result["missing_evidence"]
    assert result["raw_contingency_rows_seen"] == 0


def test_fold_single_sampling_db_prediction_results_without_contingencies_creates_stable_contingencies(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_fold_prediction_only"
    db_path = tmp_path / "seed_prediction_only.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE prediction_results (
                interaction_id INTEGER,
                global_step INTEGER,
                context_level INTEGER,
                context_signature TEXT,
                action INTEGER,
                predicted_family INTEGER,
                actual_family INTEGER
            );
            """
        )
        conn.executemany(
            "INSERT INTO prediction_results VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 1, '[1,2,3]', 2, 7, 7),
                (2, 2, 1, '[1,2,3]', 2, 8, 7),
                (3, 3, 1, '[3,2,1]', 5, 9, 9),
            ],
        )
        conn.commit()
    totals = fold_single_sampling_db_into_main_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
        finalize_after_fold=True,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        stable_count = conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]
        family_count = conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]
    assert totals["raw_contingency_rows_seen"] > 0
    assert totals["stable_contingencies_inserted"] > 0
    assert totals["transformation_families_inserted"] > 0
    assert stable_count >= 2
    assert family_count >= 1


def test_fold_single_sampling_db_prediction_results_low_support_still_persist_m1(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_fold_low_support"
    db_path = tmp_path / "seed_low_support.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE prediction_results (
                interaction_id INTEGER,
                global_step INTEGER,
                context_level INTEGER,
                context_signature TEXT,
                action INTEGER,
                predicted_family INTEGER,
                actual_family INTEGER
            );
            """
        )
        conn.execute("INSERT INTO prediction_results VALUES (1, 1, 1, '[1]', 2, 7, 7)")
        conn.commit()
    fold_single_sampling_db_into_main_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
        finalize_after_fold=True,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        stable_row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(support_count), 0) FROM stable_contingencies"
        ).fetchone()
    assert stable_row[0] == 1
    assert stable_row[1] == 1


def test_finalize_main_compact_memory_preserves_fold_summary_counters(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_finalize_preserve"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO memory_summary (key, value_json) VALUES ('fold_summary', ?)",
            (json.dumps({"raw_contingency_rows_seen": 3, "stable_contingencies_inserted": 2}),),
        )
        conn.commit()
    summary = finalize_main_compact_memory(
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=11, global_step_end=20),
    )
    fold_summary = summary["fold_summary"]
    assert fold_summary["raw_contingency_rows_seen"] == 3
    assert fold_summary["stable_contingencies_inserted"] == 2
    assert fold_summary["global_step_start"] == 11
    assert fold_summary["global_step_end"] == 20
    persisted = load_memory_summary(memory_dir / "memory_summary.json")
    assert persisted["fold_summary"]["raw_contingency_rows_seen"] == 3
    assert persisted["fold_summary"]["stable_contingencies_inserted"] == 2


def test_merge_compact_memory_shards_adds_stable_contingency_support_counts(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_main_merge"
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    ensure_memory_layout(memory_dir)
    ensure_memory_layout(shard_a)
    ensure_memory_layout(shard_b)
    for shard_dir in (shard_a, shard_b):
        with sqlite3.connect(shard_dir / "current_state.sqlite") as conn:
            conn.execute(
                """
                INSERT INTO stable_contingencies (
                    contingency_id, canonical_key, game, sampler, context_level, action, effect_signature, support_count,
                    first_seen_global_step, last_seen_global_step, stability_score, mean_prediction_error,
                    mean_replay_priority, representative_example_count, prediction_attempt_count,
                    prediction_success_count, prediction_accuracy, prediction_error_before,
                    prediction_error_after, normalized_contingency_key
                ) VALUES (1, 'same_key', 'g1', 's1', 1, 1, 'e1', 12, 1, 10, 0.6, 0.0, 0.0, 0, 12, 9, 0.75, NULL, NULL, 'nk1')
                """
            )
            conn.commit()
    merge_compact_memory_shards_into_main(
        memory_dir=memory_dir,
        shard_dirs=[shard_a, shard_b],
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=10),
        parallel_workers=1,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        row = conn.execute(
            "SELECT support_count, stability_score, prediction_attempt_count, prediction_success_count, prediction_accuracy FROM stable_contingencies WHERE canonical_key = 'same_key'"
        ).fetchone()
        stable_count = conn.execute(
            "SELECT COUNT(*) FROM stable_contingencies WHERE support_count >= 20"
        ).fetchone()[0]
    assert row[0] == 24
    assert float(row[1]) >= 1.2
    assert row[2] == 24
    assert row[3] == 18
    assert abs(float(row[4]) - 0.75) < 1e-9
    assert stable_count == 1


def test_graph_edge_fold_cap_stops_writes_after_limit(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_graph_cap"
    db_path = tmp_path / "seed_0.sqlite"
    ensure_memory_layout(memory_dir)
    sqlite3.connect(db_path).close()
    (tmp_path / "live_graph_compact.json").write_text(
        json.dumps(
            {
                "nodes": [{"node_id": f"n{i}", "node_type": "x", "canonical_key": f"k{i}"} for i in range(4)],
                "edges": [
                    {"source_node_id": "n0", "target_node_id": "n1", "edge_type": "related_to"},
                    {"source_node_id": "n0", "target_node_id": "n2", "edge_type": "related_to"},
                    {"source_node_id": "n0", "target_node_id": "n3", "edge_type": "related_to"},
                ],
            }
        ),
        encoding="utf-8",
    )
    fold_single_sampling_db_into_main_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=2, max_graph_edges_per_fold=2),
        finalize_after_fold=True,
    )
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        edge_count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
    summary = load_memory_summary(memory_dir / "memory_summary.json")
    assert edge_count == 2
    assert summary["fold_summary"]["graph_edges_attempted"] == 3
    assert summary["fold_summary"]["graph_edges_written"] == 2
    assert summary["fold_summary"]["graph_edges_skipped_by_fold_cap"] == 1


def test_graph_edge_source_and_carrier_caps_limit_writes(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_graph_source_cap"
    db_path = tmp_path / "seed_0.sqlite"
    ensure_memory_layout(memory_dir)
    sqlite3.connect(db_path).close()
    (tmp_path / "live_graph_compact.json").write_text(
        json.dumps(
            {
                "nodes": [{"node_id": "carrier:c1", "node_type": "carrier", "canonical_key": "c1"}],
                "edges": [
                    {"source_node_id": "carrier:c1", "target_node_id": "family:f1", "edge_type": "explains"},
                    {"source_node_id": "carrier:c1", "target_node_id": "context:c1", "edge_type": "appears_in"},
                    {"source_node_id": "carrier:c1", "target_node_id": "contingency:k1", "edge_type": "anchors"},
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = fold_single_sampling_db_into_main_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(
            global_step_start=1,
            global_step_end=2,
            max_edges_per_source_node=5,
            max_edges_per_carrier=2,
        ),
        finalize_after_fold=True,
    )
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        edge_count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
    assert edge_count == 2
    assert summary["graph_edges_skipped_by_carrier_cap"] == 1


def test_graph_edge_source_cap_stops_excessive_single_source_edges(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_graph_source_only_cap"
    db_path = tmp_path / "seed_0.sqlite"
    ensure_memory_layout(memory_dir)
    sqlite3.connect(db_path).close()
    (tmp_path / "live_graph_compact.json").write_text(
        json.dumps(
            {
                "edges": [
                    {"source_node_id": "context:a", "target_node_id": "n1", "edge_type": "related_to"},
                    {"source_node_id": "context:a", "target_node_id": "n2", "edge_type": "related_to"},
                    {"source_node_id": "context:a", "target_node_id": "n3", "edge_type": "related_to"},
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = fold_single_sampling_db_into_main_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=2, max_edges_per_source_node=2),
    )
    assert summary["graph_edges_skipped_by_source_cap"] == 1


def test_set_based_merge_matches_row_loop_counts_on_small_fixture(tmp_path: Path) -> None:
    shard = tmp_path / "merge_fixture_shard"
    ensure_memory_layout(shard)
    with sqlite3.connect(shard / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO family_members (family_signature, contingency_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('fam', 'cont', 2, 1, 4)")
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json) VALUES ('n1', 'M1', 't', 'k', 1, 1, 1, '{}')")
        conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json) VALUES ('n2', 'M1', 't', 'k2', 1, 1, 1, '{}')")
        conn.execute("INSERT INTO memory_edges (source_node_id, target_node_id, edge_type, weight, support_count, evidence_json) VALUES ('n1', 'n2', 'rel', 0.5, 2, '{}')")
        conn.commit()
    with sqlite3.connect(shard / "graph.sqlite") as conn:
        conn.execute("INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count) VALUES ('g1', 't', 'k', 1, 1, 1)")
        conn.execute("INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count) VALUES ('g2', 't', 'k2', 1, 1, 1)")
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e', 'g1', 'g2', 'rel', 1, 1, 2, 0.7)")
        conn.commit()
    with sqlite3.connect(shard / "replay_queue.sqlite") as conn:
        conn.execute("INSERT INTO replay_queue (replay_id, owner_type, owner_id, priority_score, reason, first_seen_global_step, last_seen_global_step, compact_payload_json) VALUES ('r1', 'interaction', '1', 0.8, 'x', 1, 2, '{}')")
        conn.commit()
    main_set = tmp_path / "main_set"
    main_row = tmp_path / "main_row"
    ensure_memory_layout(main_set)
    ensure_memory_layout(main_row)
    merge_compact_memory_shards_into_main(
        memory_dir=main_set,
        shard_dirs=[shard],
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=4, use_set_based_merge=True),
        parallel_workers=1,
    )
    merge_compact_memory_shards_into_main(
        memory_dir=main_row,
        shard_dirs=[shard],
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=4, use_set_based_merge=False),
        parallel_workers=1,
    )
    with sqlite3.connect(main_set / "current_state.sqlite") as conn:
        set_counts = (
            conn.execute("SELECT COUNT(*) FROM family_members").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0],
        )
    with sqlite3.connect(main_row / "current_state.sqlite") as conn:
        row_counts = (
            conn.execute("SELECT COUNT(*) FROM family_members").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0],
        )
    assert set_counts == row_counts == (1, 1)


def test_set_based_merge_accumulates_memory_edge_support_and_replay_priority(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_merge_set_based"
    shard_a = tmp_path / "shard_set_a"
    shard_b = tmp_path / "shard_set_b"
    ensure_memory_layout(memory_dir)
    ensure_memory_layout(shard_a)
    ensure_memory_layout(shard_b)
    for shard_dir, support, priority in ((shard_a, 2, 0.4), (shard_b, 3, 0.9)):
        with sqlite3.connect(shard_dir / "current_state.sqlite") as conn:
            conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json) VALUES ('n1', 'M1', 't', 'k1', 1, 1, 1, '{}')")
            conn.execute("INSERT INTO memory_nodes (node_id, memory_level, node_type, canonical_key, support_count, first_seen_step, last_seen_step, attrs_json) VALUES ('n2', 'M1', 't', 'k2', 1, 1, 1, '{}')")
            conn.execute("INSERT INTO memory_edges (source_node_id, target_node_id, edge_type, weight, support_count, evidence_json) VALUES ('n1', 'n2', 'rel', 1.0, ?, '{}')", (support,))
            conn.execute("INSERT INTO memory_scores (node_id, replay_priority) VALUES ('n1', ?)", (priority,))
            conn.commit()
        with sqlite3.connect(shard_dir / "replay_queue.sqlite") as conn:
            conn.execute(
                "INSERT INTO replay_queue (replay_id, owner_type, owner_id, priority_score, reason, first_seen_global_step, last_seen_global_step, compact_payload_json) VALUES ('r1', 'interaction', '1', ?, 'x', 1, 3, '{}')",
                (priority,),
            )
            conn.commit()
    merge_compact_memory_shards_into_main(
        memory_dir=memory_dir,
        shard_dirs=[shard_a, shard_b],
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=3, use_set_based_merge=True),
        parallel_workers=1,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        support_count = conn.execute("SELECT support_count FROM memory_edges WHERE source_node_id = 'n1' AND target_node_id = 'n2' AND edge_type = 'rel'").fetchone()[0]
        replay_priority = conn.execute("SELECT replay_priority FROM memory_scores WHERE node_id = 'n1'").fetchone()[0]
    with sqlite3.connect(memory_dir / "replay_queue.sqlite") as conn:
        queue_priority = conn.execute("SELECT priority_score FROM replay_queue WHERE replay_id = 'r1'").fetchone()[0]
    assert support_count == 5
    assert replay_priority == pytest.approx(0.9)
    assert queue_priority == pytest.approx(0.9)


def test_set_based_merge_keeps_graph_edge_update_semantics(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_graph_merge_semantics"
    shard_a = tmp_path / "graph_shard_a"
    shard_b = tmp_path / "graph_shard_b"
    ensure_memory_layout(memory_dir)
    ensure_memory_layout(shard_a)
    ensure_memory_layout(shard_b)
    for shard_dir, support, weight in ((shard_a, 2, 0.5), (shard_b, 5, 0.9)):
        with sqlite3.connect(shard_dir / "graph.sqlite") as conn:
            conn.execute("INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count) VALUES ('a', 't', 'a', 1, 1, 1)")
            conn.execute("INSERT INTO graph_nodes (node_id, node_type, canonical_key, first_seen_global_step, last_seen_global_step, support_count) VALUES ('b', 't', 'b', 1, 1, 1)")
            conn.execute(
                "INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('a->rel->b', 'a', 'b', 'rel', 1, 2, ?, ?)",
                (support, weight),
            )
            conn.commit()
    merge_compact_memory_shards_into_main(
        memory_dir=memory_dir,
        shard_dirs=[shard_a, shard_b],
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=2, use_set_based_merge=True),
        parallel_workers=1,
    )
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        row = conn.execute("SELECT support_count, weight FROM graph_edges WHERE edge_id = 'a->rel->b'").fetchone()
    assert row == (5, 0.9)


def test_raw_db_fold_cache_helpers_match_uncached_behavior(tmp_path: Path) -> None:
    db_path = tmp_path / "raw_cache.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE transformation_families (id INTEGER PRIMARY KEY, centroid_vector TEXT, support_count INTEGER)")
        conn.execute(
            "CREATE TABLE prediction_results (context_signature TEXT, context_action TEXT, action INTEGER, actual_family TEXT, predicted_family TEXT, context_level INTEGER)"
        )
        conn.execute("INSERT INTO transformation_families (id, centroid_vector, support_count) VALUES (7, '[1,2,3]', 4)")
        conn.execute(
            "INSERT INTO prediction_results (context_signature, context_action, action, actual_family, predicted_family, context_level) VALUES ('ctx', NULL, 2, '7', '7', 3)"
        )
        conn.commit()
        payload = {"effect_signature": None, "context_signature": "ctx", "action": 2}
        uncached_signature = canonical_family_signature_from_raw_db(conn, 7, payload)
        uncached_level = _context_level_from_raw(conn, payload=payload, family_id=7)
        caches = _build_raw_db_fold_caches(conn, {"transformation_families", "prediction_results"})
        cached_signature = canonical_family_signature_from_raw_db(conn, 7, payload, caches=caches)
        cached_level = _context_level_from_raw(conn, payload=payload, family_id=7, caches=caches)
    assert cached_signature == uncached_signature
    assert cached_level == uncached_level == 3
    assert normalized_contingency_identity_cached(context_level=3, action=2, effect_signature="fam", caches=caches) == normalized_contingency_identity(context_level=3, action=2, effect_signature="fam")


def test_raw_db_fold_cache_avoids_repeated_family_lookup(tmp_path: Path) -> None:
    class CountingConnection:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn
            self.select_count = 0

        def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
            if "SELECT centroid_vector, support_count FROM transformation_families" in sql:
                self.select_count += 1
            return self._conn.execute(sql, params)

    db_path = tmp_path / "raw_cache_count.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE transformation_families (id INTEGER PRIMARY KEY, centroid_vector TEXT, support_count INTEGER)")
    conn.execute("INSERT INTO transformation_families (id, centroid_vector, support_count) VALUES (3, '[9,9]', 2)")
    conn.commit()
    wrapper = CountingConnection(conn)
    payload = {"effect_signature": None}
    assert canonical_family_signature_from_raw_db(wrapper, 3, payload) == canonical_family_signature_from_raw_db(wrapper, 3, payload)
    assert wrapper.select_count == 2
    caches = _build_raw_db_fold_caches(conn, {"transformation_families"})
    assert canonical_family_signature_from_raw_db(wrapper, 3, payload, caches=caches) == canonical_family_signature_from_raw_db(wrapper, 3, payload, caches=caches)
    assert wrapper.select_count == 2
    conn.close()


def test_ensure_memory_layout_drops_duplicate_indexes(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_indexes"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        current_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()}
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        graph_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()}
    assert "idx_carrier_links_carrier_type_key" not in current_names
    assert "idx_role_neighborhood_carrier" not in current_names
    assert "idx_graph_edges_source" not in graph_names
    assert "idx_graph_edges_target" not in graph_names


def test_h03_compact_fold_evidence_exposes_stable_contingencies_and_families(tmp_path: Path) -> None:
    from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    run_dir = tmp_path / "run_h03_compact_after_fold"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": "g1", "sampler_name": "s1"}]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h03_compact_after_fold"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c1', 'g1', 's1', 1, 1, 'e1', 5, 1, 1, 0.5)")
        conn.execute("INSERT INTO stable_contingencies (canonical_key, game, sampler, context_level, action, effect_signature, support_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES ('c2', 'g1', 's1', 1, 1, 'e1', 6, 1, 2, 0.6)")
        conn.commit()
    derive_missing_transformation_families_from_stable_contingencies(memory_dir)
    result = evaluate_h03_transformation_family_formation(run_dir, tmp_path / "out_h03_compact_after_fold", memory_dir=memory_dir)
    assert int(result["stable_contingencies_count"] or 0) == 2
    assert int(result["transformation_families_count"] or 0) >= 1
    assert int(result["family_members_count"] or 0) >= 2


def test_suite_summary_exposes_individual_vs_gated_decisions() -> None:
    from v6.hypothesis_suite_report import _apply_higher_order_dependency_gates, _format_text, _suite_gate_status

    h04 = {"decision": "VALID"}
    h05 = {"decision": "VALID"}
    h06 = {"decision": "PARTIALLY_VALID"}
    h07 = {"decision": "VALID", "missing_evidence": []}
    h08 = {"decision": "VALID", "missing_evidence": []}
    h09 = {"decision": "PARTIALLY_VALID"}
    h10 = {"decision": "VALID", "missing_evidence": []}
    h11 = {"decision": "VALID", "missing_evidence": []}
    _, _, h07, h08, _, h10, h11, _ = _apply_higher_order_dependency_gates(h04, h05, h06, h07, h08, h09, h10, h11)
    assert h11["decision"] == "PARTIALLY_VALID"
    assert h11["individual_decision_before_suite_gates"] == "VALID"
    text = _format_text(
        {
            "source_run_dir": "run",
            "H01 decision": "VALID",
            "H02 decision": "VALID",
            "H03 decision": "VALID",
            "H04 decision": "VALID",
            "H05 decision": "VALID",
            "H06 decision": "PARTIALLY_VALID",
            "H07 decision": "PARTIALLY_VALID",
            "H08 decision": "PARTIALLY_VALID",
            "H09 decision": "PARTIALLY_VALID",
            "H10 decision": "PARTIALLY_VALID",
            "H11 decision": "PARTIALLY_VALID",
            "game_count": 1,
            "sampler_count": 1,
            "seed_count": 1,
            "total_interactions": 1000,
            "levels_successfully_completed_per_epoch": 0,
            "games_solved_per_epoch": 0,
            "solved_games": [],
            "next_recommended_action": "x",
            "H11 suite gating": _suite_gate_status(h11),
        }
    )
    assert "H11 individual: VALID" in text
    assert "H11 suite-gated: PARTIALLY_VALID" in text


def test_h06_reports_insufficient_evidence_with_explicit_missing_table(tmp_path: Path) -> None:
    from v6.hypothesis_h06_report import evaluate_h06_role_transfer

    memory_dir = tmp_path / "memory_h06_missing"
    memory_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=tmp_path / "run", output_dir=tmp_path / "h06", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "current_state.sqlite" in json.dumps(result.get("evidence_diagnostics", {}))


def test_h08_reports_insufficient_evidence_with_explicit_missing_table(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_h08_missing"
    memory_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=tmp_path / "run", output_dir=tmp_path / "h08", already_derived=True)
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "current_state.sqlite" in json.dumps(result.get("evidence_diagnostics", {}))


def test_h02_reports_manifest_compact_coverage_ratio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention

    run_dir = tmp_path / "run_h02_cov"
    run_dir.mkdir()
    (run_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"runs": [{"game": f"g{i}", "sampler_name": "s", "run_status": "ok"} for i in range(10)]}),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "memory_h02_cov"
    ensure_memory_layout(memory_dir)
    manifest_path = memory_dir / "direct_streaming_fold_manifest.sqlite"
    with sqlite3.connect(manifest_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS folded_jobs (job_id TEXT PRIMARY KEY, status TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS job_metrics (job_id TEXT PRIMARY KEY, game TEXT, sampler TEXT, seed INTEGER, metrics_json TEXT NOT NULL)")
        for i in range(4):
            conn.execute(
                "INSERT INTO job_metrics (job_id, game, sampler, seed, metrics_json) VALUES (?, ?, ?, ?, ?)",
                (f'j{i}', f'g{i}', 's', 0, '{}'),
            )
        conn.commit()
    monkeypatch.setattr(
        "v6.hypothesis_h02_report._compute_prediction_violation_replay_lift_from_existing_db",
        lambda *args, **kwargs: {"direct_replay_lift_available": False, "sqlite_db_count_inspected": 0},
    )
    result = evaluate_h02_prediction_violation_attention(run_dir, tmp_path / "out_cov", memory_dir=memory_dir)
    assert result["jobs_represented_in_compact_or_manifest_evidence"] == 4
    assert result["total_jobs_expected"] == 10
    assert result["evidence_coverage_ratio"] == 0.4


def test_h12_stays_insufficient_evidence_without_successful_comparable_trajectories(tmp_path: Path) -> None:
    from v6.evaluation.h12_efficiency_emergence import evaluate_h12_efficiency_emergence

    memory_dir = tmp_path / "memory_h12"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class, comparable_outcome_group_id,
                efficiency_active, success, terminal, trajectory_length
            ) VALUES ('t1', 'g', 'l', 's', 0, 1, 'non_success', '', 0, 0, 0, 5)
            """
        )
        conn.commit()
    result = evaluate_h12_efficiency_emergence(run_dir=tmp_path / "run_h12", memory_dir=memory_dir, output_dir=tmp_path / "h12")
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert any("No success events" in item for item in result["missing_evidence"])


def test_carrier_sidecar_with_real_timing_preserves_real_timing(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_carrier_timing_real"
    db_path = tmp_path / "seed_0.sqlite"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY)")
        conn.commit()
    (tmp_path / "carrier_candidates.json").write_text(
        json.dumps(
            [
                {
                    "carrier_id": "c1",
                    "carrier_signature": "carrier1",
                    "carrier_source": "object",
                    "context_signature": "ctx1",
                    "support_count": 4,
                    "distinct_family_count": 1,
                    "status": "emergent_carrier",
                    "first_seen_global_step": 11,
                    "last_seen_global_step": 19,
                    "first_emergent_global_step": 13,
                }
            ]
        ),
        encoding="utf-8",
    )
    fold_sampling_job_sidecars_into_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=2),
        delete_after_merge=False,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        row = conn.execute(
            "SELECT first_seen_global_step, last_seen_global_step, carrier_timing_source FROM carrier_candidates WHERE carrier_signature = 'carrier1'"
        ).fetchone()
        timing_count = json.loads(
            conn.execute("SELECT value_json FROM memory_summary WHERE key = 'carrier_sidecar_real_timing_count'").fetchone()[0]
        )
    assert row == (13, 19, "real_evidence")
    assert int(timing_count) == 1


def test_carrier_sidecar_without_timing_uses_fallback_source(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_carrier_timing_fallback_source"
    db_path = tmp_path / "seed_0.sqlite"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY)")
        conn.commit()
    (tmp_path / "carrier_candidates.json").write_text(
        json.dumps(
            [
                {
                    "carrier_id": "c1",
                    "carrier_signature": "carrier1",
                    "carrier_source": "object",
                    "context_signature": "ctx1",
                    "support_count": 4,
                    "distinct_family_count": 1,
                    "status": "emergent_carrier",
                }
            ]
        ),
        encoding="utf-8",
    )
    fold_sampling_job_sidecars_into_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=21, global_step_end=22),
        delete_after_merge=False,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        row = conn.execute(
            "SELECT first_seen_global_step, last_seen_global_step, carrier_timing_source FROM carrier_candidates WHERE carrier_signature = 'carrier1'"
        ).fetchone()
    assert row == (21, 22, "fold_start_fallback")


def test_carrier_timing_source_merge_preserves_and_upgrades(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_carrier_merge"
    shard_a = tmp_path / "shard_a_carrier"
    shard_b = tmp_path / "shard_b_carrier"
    shard_c = tmp_path / "shard_c_carrier"
    ensure_memory_layout(memory_dir)
    ensure_memory_layout(shard_a)
    ensure_memory_layout(shard_b)
    ensure_memory_layout(shard_c)
    with sqlite3.connect(shard_a / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('a','carrier_real','object',1,1,1,2,'fold_start_fallback',0.1,1)"
        )
        conn.execute(
            "INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('b','carrier_unknown','object',1,1,1,2,'unknown',0.1,1)"
        )
        conn.execute(
            "INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('c','carrier_mixed','object',1,1,1,2,'fold_start_fallback',0.1,1)"
        )
        conn.commit()
    with sqlite3.connect(shard_b / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('a2','carrier_real','object',1,1,3,4,'real_evidence',0.2,1)"
        )
        conn.execute(
            "INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('b2','carrier_unknown','object',1,1,3,4,'fold_start_fallback',0.2,1)"
        )
        conn.commit()
    with sqlite3.connect(shard_c / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('c2','carrier_mixed','object',1,1,3,4,'mixed',0.2,1)"
        )
        conn.commit()
    merge_compact_memory_shards_into_main(
        memory_dir=memory_dir,
        shard_dirs=[shard_a, shard_b, shard_c],
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=4),
        parallel_workers=1,
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        rows = dict(conn.execute("SELECT carrier_signature, carrier_timing_source FROM carrier_candidates").fetchall())
    assert rows["carrier_real"] == "real_evidence"
    assert rows["carrier_unknown"] == "fold_start_fallback"
    assert rows["carrier_mixed"] == "mixed"


def test_fold_live_system_preserves_candidate_timing(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory_live_timing"
    system = SimpleNamespace(
        step_count=50,
        config=SimpleNamespace(memory_output_dir=str(memory_dir)),
        contingency_learner=SimpleNamespace(stable_contingencies=lambda: []),
        clusterer=SimpleNamespace(families={}),
        memory_lifecycle=SimpleNamespace(replay_candidates={}),
        carrier_tracker=SimpleNamespace(
            build_candidates=lambda: [
                SimpleNamespace(
                    carrier_id="c1",
                    carrier_signature="carrier1",
                    carrier_source="object",
                    support_count=3,
                    distinct_family_count=1,
                    prediction_lift=0.2,
                    status="emergent_carrier",
                    first_seen_global_step=31,
                    last_seen_global_step=39,
                    first_emergent_global_step=33,
                    family_id=None,
                    context_signature="ctx1",
                )
            ]
        ),
        memory=None,
    )
    summary = fold_live_system_into_compact_memory(system, memory_dir)
    assert summary is not None
    with sqlite3.connect(Path(memory_dir) / "current_state.sqlite") as conn:
        row = conn.execute(
            "SELECT first_seen_global_step, last_seen_global_step, carrier_timing_source FROM carrier_candidates WHERE carrier_signature = 'carrier1'"
        ).fetchone()
        link_row = conn.execute(
            "SELECT first_seen_global_step, last_seen_global_step FROM carrier_links WHERE carrier_signature = 'carrier1' AND linked_type = 'context'"
        ).fetchone()
    assert row == (33, 39, "real_evidence")
    assert link_row == (33, 39)


def test_import_candidate_preserves_imported_timing_in_synthetic_events() -> None:
    tracker = CarrierEmergenceTracker()
    tracker.import_candidate(
        carrier_signature="carrier1",
        carrier_source="object",
        support_count=3,
        linked_family_count=1,
        first_seen_global_step=10,
        last_seen_global_step=20,
        stability_score=0.5,
        is_emergent=True,
    )
    steps = [event.global_step for event in tracker.by_carrier["carrier1"]]
    assert steps[0] == 10
    assert steps[-1] == 20
    assert all(step is not None for step in steps)


def test_fallback_carrier_timing_does_not_hard_invalidate_h04(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_carrier_timing_fallback"
    db_path = tmp_path / "seed_0.sqlite"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY)")
        conn.commit()
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            "INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,10,12,0.7)"
        )
        conn.commit()
    (tmp_path / "carrier_candidates.json").write_text(
        json.dumps(
            [
                {
                    "carrier_id": "c1",
                    "carrier_signature": "carrier1",
                    "carrier_source": "object",
                    "context_signature": "ctx1",
                    "support_count": 5,
                    "distinct_family_count": 1,
                    "status": "emergent_carrier",
                },
                {
                    "carrier_id": "c1",
                    "carrier_signature": "carrier1",
                    "carrier_source": "object",
                    "context_signature": "ctx2",
                    "support_count": 5,
                    "distinct_family_count": 1,
                    "status": "emergent_carrier",
                },
            ]
        ),
        encoding="utf-8",
    )
    fold_sampling_job_sidecars_into_compact_memory(
        db_path=db_path,
        memory_dir=memory_dir,
        fold_config=CompactMemoryFoldConfig(global_step_start=1, global_step_end=2),
        delete_after_merge=False,
    )
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e1','carrier:carrier1','b','explains',1,1,1,1.0)")
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e2','carrier:carrier1','b','anchors',1,1,1,1.0)")
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_fallback")
    assert result["carrier_timing_source"] == "fold_start_fallback"
    assert result["decision"] == "PARTIALLY_VALID"
    assert any("without fully real carrier timing provenance" in item for item in result["missing_evidence"])


def test_h04_false_temporal_order_with_real_timing_is_invalid(tmp_path: Path) -> None:
    from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence

    memory_dir = tmp_path / "memory_h04_real_invalid"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO transformation_families (family_id, canonical_signature, support_count, member_count, first_seen_global_step, last_seen_global_step, stability_score) VALUES (1,'fam1',4,2,10,12,0.7)")
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('c1','carrier1','object',5,2,7,8,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier1','family','fam1',1,7,8)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier1','context','ctx1',1,7,8)")
        conn.execute("INSERT INTO carrier_links (carrier_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('carrier1','context','ctx2',1,7,8)")
        conn.commit()
    with sqlite3.connect(memory_dir / "graph.sqlite") as conn:
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e1','carrier:carrier1','b','explains',1,1,1,1.0)")
        conn.execute("INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, first_seen_global_step, last_seen_global_step, support_count, weight) VALUES ('e2','carrier:carrier1','b','anchors',1,1,1,1.0)")
        conn.commit()
    result = evaluate_h04_carrier_emergence(run_dir=None, memory_dir=memory_dir, output_dir=tmp_path / "h04_real_invalid")
    assert result["carrier_timing_source"] == "real_evidence"
    assert result["h03_before_h04"] is False
    assert result["decision"] == "INVALID"


def test_role_timing_fallback_cannot_make_h05_valid(tmp_path: Path) -> None:
    from v6.hypothesis_h05_report import evaluate_h05_role_emergence

    memory_dir = tmp_path / "memory_h05_fallback_role"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('c1','carrier1','object',5,2,1,2,'fold_start_fallback',0.8,1)")
        conn.execute("INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, linked_context_count, cross_game_count, cross_context_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r1','role',5,2,1,2,1,2,11,12,0.8,1)")
        conn.execute("INSERT INTO role_links (role_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('r1', 'carrier', 'carrier1', 1, 11, 12)")
        conn.commit()
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05_fallback_role", already_derived=True)
    assert result["role_timing_source"] == "fold_start_fallback"
    assert result["decision"] != "VALID"
    assert "Role timing is not fully grounded in real carrier evidence timing." in result["missing_evidence"]


def test_h05_overconnected_roles_do_not_make_validity(tmp_path: Path) -> None:
    from v6.hypothesis_h05_report import evaluate_h05_role_emergence

    memory_dir = tmp_path / "memory_h05_overconnected"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('c1','carrier1','object',5,2,7,8,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO temporal_milestones (game, sampler, seed, first_emergent_carrier_step) VALUES ('g','s',0,7)")
        conn.execute("INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, linked_context_count, cross_game_count, cross_context_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r1','role',5,101,1,2,1,2,11,12,0.8,1)")
        conn.execute("INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key) VALUES ('first_role_candidate_step', 11, 'r1')")
        conn.execute("INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key) VALUES ('first_emergent_role_step', 11, 'r1')")
        conn.execute("INSERT INTO role_links (role_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('r1', 'carrier', 'carrier1', 1, 11, 12)")
        conn.commit()
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05_overconnected", already_derived=True)
    assert result["overconnected_role_count"] == 1
    assert result["usable_emergent_role_count"] == 0
    assert result["decision"] != "VALID"


def test_h05_singleton_role_ratio_downgrades_validity(tmp_path: Path) -> None:
    from v6.hypothesis_h05_report import evaluate_h05_role_emergence

    memory_dir = tmp_path / "memory_h05_noisy"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('c1','carrier1','object',5,2,7,8,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO temporal_milestones (game, sampler, seed, first_emergent_carrier_step) VALUES ('g','s',0,7)")
        conn.execute("INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, linked_context_count, cross_game_count, cross_context_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r1','role',5,2,1,2,1,2,11,12,0.8,1)")
        conn.execute("INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, linked_context_count, cross_game_count, cross_context_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r2','role',3,1,1,1,0,1,11,12,0.6,0)")
        conn.execute("INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, linked_context_count, cross_game_count, cross_context_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r3','role',3,1,1,1,0,1,11,12,0.6,0)")
        conn.execute("INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key) VALUES ('first_role_candidate_step', 11, 'r1')")
        conn.execute("INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key) VALUES ('first_emergent_role_step', 11, 'r1')")
        conn.execute("INSERT INTO role_links (role_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('r1', 'carrier', 'carrier1', 1, 11, 12)")
        conn.commit()
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05_noisy", already_derived=True)
    assert float(result["singleton_role_ratio"]) > 0.50
    assert result["h05_role_quality_pass"] is False
    assert result["decision"] == "PARTIALLY_VALID"
    assert "H05 role graph is noisy or overconnected; remapping quality prevents robust VALID classification." in result["missing_evidence"]


def test_h05_can_be_valid_only_with_real_role_timing_and_true_temporal_order(tmp_path: Path) -> None:
    from v6.hypothesis_h05_report import evaluate_h05_role_emergence

    memory_dir = tmp_path / "memory_h05_real_valid"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute("INSERT INTO carrier_candidates (carrier_id, carrier_signature, carrier_source, support_count, linked_family_count, first_seen_global_step, last_seen_global_step, carrier_timing_source, stability_score, is_emergent) VALUES ('c1','carrier1','object',5,2,7,8,'real_evidence',0.8,1)")
        conn.execute("INSERT INTO temporal_milestones (game, sampler, seed, first_emergent_carrier_step) VALUES ('g','s',0,7)")
        conn.execute("INSERT INTO role_candidates (role_signature, role_type, support_count, linked_carrier_count, linked_family_count, linked_context_count, cross_game_count, cross_context_count, first_seen_global_step, last_seen_global_step, role_stability_score, is_emergent) VALUES ('r1','role',5,2,1,2,1,2,11,12,0.8,1)")
        conn.execute("INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key) VALUES ('first_role_candidate_step', 11, 'r1')")
        conn.execute("INSERT INTO higher_order_milestones (milestone_name, first_global_step, evidence_key) VALUES ('first_emergent_role_step', 11, 'r1')")
        conn.execute("INSERT INTO role_links (role_signature, linked_type, linked_key, support_count, first_seen_global_step, last_seen_global_step) VALUES ('r1', 'carrier', 'carrier1', 1, 11, 12)")
        conn.commit()
    result = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=None, output_dir=tmp_path / "h05_real_valid", already_derived=True)
    assert result["role_timing_source"] == "real_evidence"
    assert result["h04_before_h05"] is True
    assert result["decision"] == "VALID"


def test_h12_reconstructs_trajectory_rows_from_compact_events_after_raw_cleanup(tmp_path: Path) -> None:
    from v6.evaluation.h12_efficiency_emergence import evaluate_h12_efficiency_emergence
    from v6.memory.direct_streaming_fold import ensure_direct_streaming_fold_manifest

    memory_dir = tmp_path / "memory_h12_compact_reconstruct"
    ensure_memory_layout(memory_dir)
    ensure_direct_streaming_fold_manifest(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.executemany(
            """
            INSERT INTO compact_interaction_trajectory_events (
                event_id, interaction_id, game_id, level_id, sampler, seed, epoch, episode_id,
                global_step, outcome_state, level_completed_event, state_hash_before, state_hash_after,
                action, no_effect_action, future_option_gain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("e1", "i1", "g1", "l1", "mixed", 0, 1, 1, 1, "RUNNING", 0, "a", "b", 0, 0, 0.1),
                ("e2", "i2", "g1", "l1", "mixed", 0, 1, 1, 2, "WIN", 1, "b", "c", 1, 0, 0.2),
                ("e3", "i3", "g1", "l1", "mixed", 0, 1, 2, 3, "RUNNING", 0, "d", "e", 0, 0, 0.1),
                ("e4", "i4", "g1", "l1", "mixed", 0, 1, 2, 4, "WIN", 1, "e", "f", 1, 0, 0.2),
            ],
        )
        conn.commit()
    result = evaluate_h12_efficiency_emergence(
        run_dir=tmp_path / "run_h12_compact_reconstruct",
        memory_dir=memory_dir,
        output_dir=tmp_path / "h12_compact_reconstruct",
    )
    assert result["compact_trajectory_rows"] == 4
    assert result["reconstructed_trajectory_rows"] == 2
    assert result["blocked_by_missing_trajectory_evidence"] is False


def test_h12_sets_not_improving_flag_and_caps_decision(tmp_path: Path) -> None:
    from v6.evaluation.h12_efficiency_emergence import evaluate_h12_efficiency_emergence

    memory_dir = tmp_path / "memory_h12_not_improving"
    ensure_memory_layout(memory_dir)
    efficiency_root = tmp_path / "efficiency"
    efficiency_root.mkdir(parents=True, exist_ok=True)
    (efficiency_root / "trajectory_efficiency_state.json").write_text(
        json.dumps({"mean_normalized_solve_efficiency": 0.9}),
        encoding="utf-8",
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class, comparable_outcome_group_id,
                efficiency_active, success, terminal, trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, future_option_gain, future_option_gain_per_action, equivalent_outcome_cost_gap,
                loop_count, loop_ratio, repeated_state_count, repeated_state_ratio, blocked_action_count, blocked_action_ratio,
                wasted_action_count, wasted_action_ratio, unique_state_count, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus, efficiency_retention_bonus, efficiency_promotion_bonus
            ) VALUES
            ('t1','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.8,0.0,0.0,2.0,0,0.0,0,0.0,0,0.0,0,0.0,10,0.8,0.1,0.1,0.0,0.0),
            ('t2','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.8,0.0,0.0,2.0,0,0.0,0,0.0,0,0.0,0,0.0,10,0.8,0.1,0.1,0.0,0.0)
            """
        )
        conn.commit()
    result = evaluate_h12_efficiency_emergence(run_dir=tmp_path / "run_h12_not_improving", memory_dir=memory_dir, output_dir=tmp_path / "h12_not_improving")
    assert result["h12_efficiency_not_improving"] is True
    assert result["decision"] == "PARTIALLY_VALID"


def test_h12_negative_cost_gap_correlation_is_not_flagged_bad(tmp_path: Path, monkeypatch) -> None:
    import v6.evaluation.h12_efficiency_emergence as h12_report

    memory_dir = tmp_path / "memory_h12_good_gap"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class, comparable_outcome_group_id,
                efficiency_active, success, terminal, trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, future_option_gain, future_option_gain_per_action, equivalent_outcome_cost_gap,
                loop_count, loop_ratio, repeated_state_count, repeated_state_ratio, blocked_action_count, blocked_action_ratio,
                wasted_action_count, wasted_action_ratio, unique_state_count, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus, efficiency_retention_bonus, efficiency_promotion_bonus
            ) VALUES
            ('t1','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.8,0.0,0.0,1.0,0,0.0,0,0.0,0,0.0,0,0.0,10,0.8,0.1,0.4,0.0,0.0),
            ('t2','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.9,0.0,0.0,3.0,0,0.0,0,0.0,0,0.0,0,0.0,10,0.9,0.1,0.1,0.0,0.0)
            """
        )
        conn.commit()
    original = h12_report._correlation
    def fake_correlation(rows, x, y):
        if x == "equivalent_outcome_cost_gap" and y == "efficiency_replay_bonus":
            return -0.303
        return original(rows, x, y)
    monkeypatch.setattr(h12_report, "_correlation", fake_correlation)
    result = h12_report.evaluate_h12_efficiency_emergence(
        run_dir=tmp_path / "run_h12_good_gap",
        memory_dir=memory_dir,
        output_dir=tmp_path / "h12_good_gap",
    )
    assert result["cost_gap_replay_selection_bad"] is False
    assert result["negative_cost_gap_replay_selection"] is False
    assert not any("positively correlated with equivalent-outcome cost gap" in item for item in result["missing_evidence"])


def test_h12_positive_cost_gap_correlation_is_flagged_bad(tmp_path: Path, monkeypatch) -> None:
    import v6.evaluation.h12_efficiency_emergence as h12_report

    memory_dir = tmp_path / "memory_h12_bad_gap"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class, comparable_outcome_group_id,
                efficiency_active, success, terminal, trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, future_option_gain, future_option_gain_per_action, equivalent_outcome_cost_gap,
                loop_count, loop_ratio, repeated_state_count, repeated_state_ratio, blocked_action_count, blocked_action_ratio,
                wasted_action_count, wasted_action_ratio, unique_state_count, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus, efficiency_retention_bonus, efficiency_promotion_bonus
            ) VALUES
            ('t1','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.8,0.0,0.0,1.0,0,0.0,0,0.0,0,0.0,0,0.0,10,0.8,0.1,0.4,0.0,0.0),
            ('t2','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.9,0.0,0.0,3.0,0,0.0,0,0.0,0,0.0,0,0.0,10,0.9,0.1,0.1,0.0,0.0)
            """
        )
        conn.commit()
    original = h12_report._correlation
    def fake_correlation(rows, x, y):
        if x == "equivalent_outcome_cost_gap" and y == "efficiency_replay_bonus":
            return 0.303
        return original(rows, x, y)
    monkeypatch.setattr(h12_report, "_correlation", fake_correlation)
    result = h12_report.evaluate_h12_efficiency_emergence(
        run_dir=tmp_path / "run_h12_bad_gap",
        memory_dir=memory_dir,
        output_dir=tmp_path / "h12_bad_gap",
    )
    assert result["cost_gap_replay_selection_bad"] is True
    assert "Replay preference is positively correlated with equivalent-outcome cost gap; inefficient trajectories are being preferentially selected." in result["missing_evidence"]


def test_h12_efficiency_score_populates_correlations(tmp_path: Path) -> None:
    from v6.evaluation.h12_efficiency_emergence import evaluate_h12_efficiency_emergence

    memory_dir = tmp_path / "memory_h12_efficiency_score"
    ensure_memory_layout(memory_dir)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class, comparable_outcome_group_id,
                efficiency_active, success, terminal, trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, equivalent_outcome_cost_gap, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus
            ) VALUES
            ('t1','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.8,1.0,0.2,0.1,0.2),
            ('t2','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.9,2.0,0.4,0.2,0.4)
            """
        )
        conn.commit()
    result = evaluate_h12_efficiency_emergence(run_dir=tmp_path / "run_h12_efficiency_score", memory_dir=memory_dir, output_dir=tmp_path / "out_h12_efficiency_score")
    assert result["efficiency_memory_fitness_correlation"] is not None
    assert result["efficiency_replay_priority_correlation"] is not None


def test_h12_cannot_be_valid_when_efficiency_correlations_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import v6.evaluation.h12_efficiency_emergence as h12_report

    memory_dir = tmp_path / "memory_h12_missing_corr"
    ensure_memory_layout(memory_dir)
    efficiency_root = tmp_path / "efficiency"
    efficiency_root.mkdir(parents=True, exist_ok=True)
    (efficiency_root / "trajectory_efficiency_state.json").write_text(
        json.dumps({"epoch": 0, "mean_normalized_solve_efficiency": 0.1}),
        encoding="utf-8",
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class, comparable_outcome_group_id,
                efficiency_active, success, terminal, trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, equivalent_outcome_cost_gap, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus
            ) VALUES
            ('t1','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.8,1.0,0.2,0.1,0.2),
            ('t2','g','l','s',0,1,'WIN','grp',1,1,1,10,10,8,0.9,2.0,0.4,0.2,0.4)
            """
        )
        conn.commit()
    original = h12_report._correlation
    def fake_correlation(rows, x, y):
        if (x, y) in {
            ("efficiency_score", "efficiency_memory_bonus"),
            ("efficiency_score", "efficiency_replay_bonus"),
        }:
            return None
        return original(rows, x, y)
    monkeypatch.setattr(h12_report, "_correlation", fake_correlation)
    result = h12_report.evaluate_h12_efficiency_emergence(
        run_dir=tmp_path / "epochs" / "epoch_0001" / "raw",
        memory_dir=memory_dir,
        output_dir=tmp_path / "out_h12_missing_corr",
    )
    assert result["decision"] == "PARTIALLY_VALID"
    assert "Trajectory efficiency improved, but memory/replay correlation evidence is unavailable." in result["missing_evidence"]


def test_h12_rerunning_same_epoch_does_not_compare_against_itself(tmp_path: Path) -> None:
    from v6.evaluation.h12_efficiency_emergence import evaluate_h12_efficiency_emergence

    memory_dir = tmp_path / "memory_h12_same_epoch"
    ensure_memory_layout(memory_dir)
    efficiency_root = tmp_path / "epochs" / "epoch_0002" / "efficiency"
    efficiency_root.mkdir(parents=True, exist_ok=True)
    (efficiency_root / "trajectory_efficiency_state.json").write_text(
        json.dumps({"epoch": 2, "mean_normalized_solve_efficiency": 0.95}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "epochs" / "epoch_0002" / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class, comparable_outcome_group_id,
                efficiency_active, success, terminal, trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, equivalent_outcome_cost_gap, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus
            ) VALUES
            ('t1','g','l','s',0,2,'WIN','grp',1,1,1,10,10,8,0.8,1.0,0.2,0.1,0.2),
            ('t2','g','l','s',0,2,'WIN','grp',1,1,1,10,10,8,0.9,2.0,0.4,0.2,0.4)
            """
        )
        conn.commit()
    result = evaluate_h12_efficiency_emergence(run_dir=run_dir, memory_dir=memory_dir, output_dir=tmp_path / "out_h12_same_epoch")
    assert result["efficiency_improved_vs_previous_epoch"] is None
    assert result["h12_efficiency_not_improving"] is False


def test_h12_preserves_best_known_solution_lengths_not_in_current_rows(tmp_path: Path) -> None:
    from v6.evaluation.h12_efficiency_emergence import evaluate_h12_efficiency_emergence

    memory_dir = tmp_path / "memory_h12_best_known"
    ensure_memory_layout(memory_dir)
    efficiency_root = tmp_path / "efficiency"
    efficiency_root.mkdir(parents=True, exist_ok=True)
    (efficiency_root / "best_known_solution_lengths.json").write_text(
        json.dumps({"old_game|__none__": 7}),
        encoding="utf-8",
    )
    with sqlite3.connect(memory_dir / "current_state.sqlite") as conn:
        conn.execute(
            """
            INSERT INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class, comparable_outcome_group_id,
                efficiency_active, success, terminal, trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, equivalent_outcome_cost_gap, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus
            ) VALUES
            ('t1','new_game','l','s',0,1,'WIN','grp',1,1,1,10,10,9,0.9,1.0,0.2,0.1,0.2),
            ('t2','new_game','l','s',0,1,'WIN','grp',1,1,1,9,9,9,1.0,0.0,0.3,0.2,0.3)
            """
        )
        conn.commit()
    evaluate_h12_efficiency_emergence(run_dir=tmp_path / "run_h12_best_known", memory_dir=memory_dir, output_dir=tmp_path / "out_h12_best_known")
    saved = json.loads((efficiency_root / "best_known_solution_lengths.json").read_text(encoding="utf-8"))
    assert saved["old_game|__none__"] == 7
    assert "new_game|l" in saved


def test_is_retryable_fold_error_matches_locked_busy_variants() -> None:
    from v6.memory.direct_streaming_fold import is_retryable_fold_error

    assert is_retryable_fold_error(sqlite3.OperationalError("database is locked")) is True
    assert is_retryable_fold_error(sqlite3.DatabaseError("database table is locked")) is True
    assert is_retryable_fold_error(sqlite3.DatabaseError("database disk image is malformed")) is False


def test_sampling_read_db_applies_busy_timeout(tmp_path: Path) -> None:
    from v6.evaluation.sampling_job_metrics import _connect_sampling_read_db

    db_path = tmp_path / "busy.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
    with _connect_sampling_read_db(db_path, busy_timeout_ms=54321) as conn:
        timeout_ms = int(conn.execute("PRAGMA busy_timeout").fetchone()[0] or 0)
    assert timeout_ms == 54321


def test_direct_fold_passes_busy_timeout_to_metrics_and_shard_connections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from v6.memory.direct_streaming_fold import DirectStreamingFoldConfig, DirectStreamingFoldJob, fold_one_completed_job_to_shard

    db_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY)")
        conn.commit()
    memory_dir = tmp_path / "memory_busy_fold"
    shard_dir = tmp_path / "shard_busy_fold"
    shard_dir.mkdir()
    seen: dict[str, Any] = {"metric_timeouts": [], "compact_busy_timeouts": []}

    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_metrics",
        lambda *args, **kwargs: seen["metric_timeouts"].append(kwargs.get("busy_timeout_ms")) or {},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_temporal_milestones",
        lambda *args, **kwargs: seen["metric_timeouts"].append(kwargs.get("busy_timeout_ms")) or {},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.compute_sampling_job_validation_payload",
        lambda *args, **kwargs: seen["metric_timeouts"].append(kwargs.get("busy_timeout_ms")) or {"examples": []},
    )
    monkeypatch.setattr(
        "v6.memory.direct_streaming_fold.fold_single_sampling_db_into_main_compact_memory",
        lambda *args, **kwargs: seen["compact_busy_timeouts"].append(kwargs.get("busy_timeout_ms")) or {},
    )
    config = DirectStreamingFoldConfig(memory_dir=str(memory_dir), busy_timeout_ms=54321)
    job = DirectStreamingFoldJob(
        job_id="job1",
        db_path=str(db_path),
        game="g",
        sampler="s",
        seed=0,
        steps=1,
        horizon=1,
        context_depth=1,
        global_step_start=1,
        global_step_end=1,
        memory_dir=str(memory_dir),
        delete_raw_after_fold=False,
    )
    result = fold_one_completed_job_to_shard(
        job=job,
        config=config,
        sampling_config=SimpleNamespace(steps=1, horizon=1),
        shard_dir=shard_dir,
    )
    assert result.status == "folded"
    assert seen["metric_timeouts"] == [54321, 54321, 54321]
    assert seen["compact_busy_timeouts"] == [54321]


def test_failed_fold_manifest_records_retry_history(tmp_path: Path) -> None:
    from v6.memory.direct_streaming_fold import DirectStreamingFoldConfig, DirectStreamingFoldJob, fold_one_completed_job_to_shard

    db_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY)")
        conn.commit()
    memory_dir = tmp_path / "memory_retry_fail"
    shard_dir = tmp_path / "shard_retry_fail"
    shard_dir.mkdir()
    config = DirectStreamingFoldConfig(memory_dir=str(memory_dir), retry_attempts=2, retry_initial_delay_seconds=0.0)
    job = DirectStreamingFoldJob(
        job_id="job_fail",
        db_path=str(db_path),
        game="g",
        sampler="s",
        seed=0,
        steps=1,
        horizon=1,
        context_depth=1,
        global_step_start=1,
        global_step_end=1,
        memory_dir=str(memory_dir),
        delete_raw_after_fold=False,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "v6.memory.direct_streaming_fold._fold_one_completed_job_to_shard_once",
            lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
        )
        result = fold_one_completed_job_to_shard(
            job=job,
            config=config,
            sampling_config=SimpleNamespace(steps=1, horizon=1),
            shard_dir=shard_dir,
        )
    assert result.status == "failed"
    manifest = memory_dir / "direct_streaming_fold_manifest.sqlite"
    with sqlite3.connect(manifest) as conn:
        row = conn.execute(
            "SELECT retry_attempt_count, retryable_error_count, last_retry_error, retry_error_history_json FROM folded_jobs WHERE job_id = 'job_fail'"
        ).fetchone()
    assert int(row[0]) == 2
    assert int(row[1]) == 2
    assert "database is locked" in str(row[2]).lower()
    assert len(json.loads(row[3])) == 2


def test_successful_fold_with_one_retry_records_retry_attempt_count(tmp_path: Path) -> None:
    from v6.memory.direct_streaming_fold import DirectStreamingFoldConfig, DirectStreamingFoldJob, DirectStreamingFoldResult, fold_one_completed_job_to_shard

    db_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE interactions (id INTEGER PRIMARY KEY)")
        conn.commit()
    memory_dir = tmp_path / "memory_retry_success"
    shard_dir = tmp_path / "shard_retry_success"
    shard_dir.mkdir()
    config = DirectStreamingFoldConfig(memory_dir=str(memory_dir), retry_attempts=3, retry_initial_delay_seconds=0.0)
    job = DirectStreamingFoldJob(
        job_id="job_success",
        db_path=str(db_path),
        game="g",
        sampler="s",
        seed=0,
        steps=1,
        horizon=1,
        context_depth=1,
        global_step_start=1,
        global_step_end=1,
        memory_dir=str(memory_dir),
        delete_raw_after_fold=False,
    )
    state = {"calls": 0}

    def _fake_once(**kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return DirectStreamingFoldResult(
            job_id=job.job_id,
            db_path=str(job.db_path),
            status="folded",
            fold_started_at=1.0,
            fold_finished_at=2.0,
            deleted_raw=False,
            metrics={"m": 1},
            milestones={"t": 1},
            validation_payload={"examples": []},
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("v6.memory.direct_streaming_fold._fold_one_completed_job_to_shard_once", _fake_once)
        result = fold_one_completed_job_to_shard(
            job=job,
            config=config,
            sampling_config=SimpleNamespace(steps=1, horizon=1),
            shard_dir=shard_dir,
        )
    assert result.status == "folded"
    assert result.retry_attempt_count == 2
    assert result.retryable_error_count == 1
    manifest = memory_dir / "direct_streaming_fold_manifest.sqlite"
    with sqlite3.connect(manifest) as conn:
        row = conn.execute(
            "SELECT retry_attempt_count, retryable_error_count, retry_error_history_json FROM folded_jobs WHERE job_id = 'job_success'"
        ).fetchone()
    assert int(row[0]) == 2
    assert int(row[1]) == 1
    assert len(json.loads(row[2])) == 1


def test_direct_streaming_fold_writer_uses_configured_max_tasks_per_child(tmp_path: Path, monkeypatch) -> None:
    from v6.memory.direct_streaming_fold import DirectStreamingFoldConfig, DirectStreamingFoldWriter

    memory_dir = tmp_path / "memory_fold_executor"
    seen: dict[str, Any] = {}

    class _FakeExecutor:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            return None

    monkeypatch.setattr("v6.memory.direct_streaming_fold.ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr("v6.memory.direct_streaming_fold._make_shard_dirs", lambda config, workers: [tmp_path / "shard0"])
    monkeypatch.setattr("v6.memory.direct_streaming_fold._cleanup_stale_existing_shard_root", lambda config: False)
    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=3, max_tasks_per_child=17),
        sampling_config=SimpleNamespace(steps=1, horizon=1),
    )
    writer.start()
    assert seen["max_workers"] == 3
    assert seen["max_tasks_per_child"] == 17


def test_direct_streaming_fold_writer_disables_child_recycling_when_zero(tmp_path: Path, monkeypatch) -> None:
    from v6.memory.direct_streaming_fold import DirectStreamingFoldConfig, DirectStreamingFoldWriter

    memory_dir = tmp_path / "memory_fold_executor_zero"
    seen: dict[str, Any] = {}

    class _FakeExecutor:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            return None

    monkeypatch.setattr("v6.memory.direct_streaming_fold.ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr("v6.memory.direct_streaming_fold._make_shard_dirs", lambda config, workers: [tmp_path / "shard0"])
    monkeypatch.setattr("v6.memory.direct_streaming_fold._cleanup_stale_existing_shard_root", lambda config: False)
    writer = DirectStreamingFoldWriter(
        DirectStreamingFoldConfig(memory_dir=str(memory_dir), fold_workers=2, max_tasks_per_child=0),
        sampling_config=SimpleNamespace(steps=1, horizon=1),
    )
    writer.start()
    assert seen["max_workers"] == 2
    assert "max_tasks_per_child" not in seen


def test_sampling_executor_preserves_zero_max_tasks_per_child(monkeypatch, tmp_path: Path) -> None:
    from v6.evaluation.interaction_sampling import _run_sampling_jobs

    seen: dict[str, Any] = {}

    class _FakeFuture:
        def result(self):
            return {"ok": True}

    class _FakeExecutor:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, job):
            return _FakeFuture()

    monkeypatch.setattr("v6.evaluation.interaction_sampling.ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr("v6.evaluation.interaction_sampling.wait", lambda futures, timeout, return_when: (set(futures.keys()), set()))
    jobs = [
        {
            "game": "tt01",
            "sampler_name": "mixed",
            "seed": 0,
            "steps": 1,
            "horizon": 1,
            "context_depth": 1,
            "global_step_offset": 0,
            "db_path": str(tmp_path / "seed_0.sqlite"),
            "max_tasks_per_child": 0,
            "memory_output_dir": None,
            "direct_streaming_fold_enabled": False,
            "shared_live_memory": "none",
        }
    ]
    stats = _run_sampling_jobs(jobs, workers=1)
    assert seen["max_workers"] == 1
    assert "max_tasks_per_child" not in seen
    assert stats["max_tasks_per_child"] == 0


def test_sampling_executor_passes_positive_max_tasks_per_child(monkeypatch, tmp_path: Path) -> None:
    from v6.evaluation.interaction_sampling import _run_sampling_jobs

    seen: dict[str, Any] = {}

    class _FakeFuture:
        def result(self):
            return {"ok": True}

    class _FakeExecutor:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, job):
            return _FakeFuture()

    monkeypatch.setattr("v6.evaluation.interaction_sampling.ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr("v6.evaluation.interaction_sampling.wait", lambda futures, timeout, return_when: (set(futures.keys()), set()))
    jobs = [
        {
            "game": "tt01",
            "sampler_name": "mixed",
            "seed": 0,
            "steps": 1,
            "horizon": 1,
            "context_depth": 1,
            "global_step_offset": 0,
            "db_path": str(tmp_path / "seed_0.sqlite"),
            "max_tasks_per_child": 50,
            "memory_output_dir": None,
            "direct_streaming_fold_enabled": False,
            "shared_live_memory": "none",
        }
    ]
    stats = _run_sampling_jobs(jobs, workers=1)
    assert seen["max_workers"] == 1
    assert seen["max_tasks_per_child"] == 50
    assert stats["max_tasks_per_child"] == 50


def test_retry_direct_streaming_fold_preserves_zero_and_positive_max_tasks_per_child(tmp_path: Path, monkeypatch) -> None:
    from v6.memory import direct_streaming_fold as dsf

    manifest_path = tmp_path / "manifest.sqlite"
    memory_dir = tmp_path / "memory_retry_cfg"
    memory_dir.mkdir()
    seen: list[int] = []

    class _Sentinel(Exception):
        pass

    class _FakeConn:
        def execute(self, sql, *args, **kwargs):
            sql_text = str(sql)
            if "FROM folded_jobs" in sql_text:
                return SimpleNamespace(fetchall=lambda: [])
            return SimpleNamespace(fetchone=lambda: (0,))

        def commit(self) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_config(*args, **kwargs):
        seen.append(int(kwargs.get("max_tasks_per_child", -1)))
        raise _Sentinel()

    monkeypatch.setattr(dsf, "_connect_manifest", lambda *args, **kwargs: _FakeConn())
    monkeypatch.setattr(dsf, "DirectStreamingFoldConfig", _fake_config)
    try:
        dsf.retry_direct_streaming_fold_failures(
            manifest_path=manifest_path,
            memory_dir=memory_dir,
            max_tasks_per_child=0,
        )
    except _Sentinel:
        pass
    try:
        dsf.retry_direct_streaming_fold_failures(
            manifest_path=manifest_path,
            memory_dir=memory_dir,
            max_tasks_per_child=1000,
        )
    except _Sentinel:
        pass
    assert seen == [0, 1000]


def test_cli_and_config_propagation_preserve_zero_max_tasks_per_child(monkeypatch, tmp_path: Path) -> None:
    from v6.cli import build_parser
    from v6.continuous_research import ContinuousResearchConfig
    from v6.evaluation.interaction_sampling import InteractionSamplingConfig, _generate_sampling_dbs

    parser = build_parser()
    args = parser.parse_args(
        [
            "continuous-research-run",
            "--experiment-name",
            "exp",
            "--games",
            "tt01",
            "--samplers",
            "mixed",
            "--seeds",
            "0",
            "--steps-per-epoch",
            "10",
            "--max-epochs",
            "1",
            "--horizon",
            "1",
            "--context-depth",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--max-tasks-per-child",
            "0",
        ]
    )
    assert int(args.max_tasks_per_child) == 0
    continuous_config = ContinuousResearchConfig(
        experiment_name=str(args.experiment_name),
        games=str(args.games),
        samplers=str(args.samplers),
        seeds=str(args.seeds),
        steps_per_epoch=int(args.steps_per_epoch),
        max_epochs=int(args.max_epochs),
        horizon=int(args.horizon),
        context_depth=int(args.context_depth),
        output_dir=str(args.output_dir),
        max_tasks_per_child=int(args.max_tasks_per_child),
    )
    assert continuous_config.max_tasks_per_child == 0

    seen_jobs: list[dict[str, Any]] = []

    def _fake_invoke(jobs, **kwargs):
        seen_jobs.extend(jobs)
        return {"requested_workers": 1}

    monkeypatch.setattr("v6.evaluation.interaction_sampling._invoke_run_sampling_jobs", _fake_invoke)
    sampling_config = InteractionSamplingConfig(
        games=("tt01",),
        samplers=("mixed",),
        seeds=(0,),
        steps=10,
        horizon=1,
        context_depth=1,
        output_dir=str(tmp_path / "sampling"),
        max_tasks_per_child=int(continuous_config.max_tasks_per_child),
    )
    _generate_sampling_dbs(sampling_config, Path(sampling_config.output_dir) / "sampling_v05c")
    assert seen_jobs
    assert seen_jobs[0]["max_tasks_per_child"] == 0
