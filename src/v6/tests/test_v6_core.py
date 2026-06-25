from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import numpy as np
import v6.evaluation.interaction_sampling as interaction_sampling

from v6.cli import build_parser
from v6.cli import _apply_interaction_sampling_experiment_preset
from v6.carrier_emergence import CarrierEmergenceTracker, extract_carrier_signature
from v6.game_sets import load_game_set_manifest
from v6.efficiency_metrics import EfficiencyTracker, is_no_effect_delta
from v6.contingency_memory import (
    ContingencyMemoryConfig,
    action_only_accuracy,
    build_input_manifest,
    build_m1_contingencies,
    build_episode_summary,
    build_context_signature,
    classify_outcome,
    context_model_accuracy,
    discover_parquet_runs,
    format_v06_report,
    list_interaction_files,
    load_partition_events,
    print_progress,
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
from v6.interaction_significance import compute_interaction_significance
from v6.main import V6Config, V6System
from v6.memory.contingency_store import ContingencyStore
from v6.memory.interaction_store import Interaction, InteractionStore, encode_array
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
        reward=0,
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


def test_v05c_post_run_future_option_efficiency_enrichment(tmp_path) -> None:
    db_path = tmp_path / "future_option.sqlite"
    with sqlite3.connect(db_path) as connection:
        interaction_store = InteractionStore(connection)
        contingency_store = ContingencyStore(connection)
        for interaction_id, action, actual_family in (
            (1, 1, 10),
            (2, 2, 20),
            (3, 1, 10),
            (4, 3, 30),
            (5, 1, 20),
        ):
            interaction_store.add(
                Interaction(
                    id=interaction_id,
                    timestamp=interaction_id,
                    observation_before=np.zeros((2, 2), dtype=int),
                    action=action,
                    observation_after=np.full((2, 2), interaction_id, dtype=int),
                    delta_id=interaction_id,
                    efficiency_action_cost=2.0,
                )
            )
            contingency_store.add_prediction_result(
                interaction_id=interaction_id,
                context_level=0,
                context_signature=(action,),
                action=action,
                predicted_family=actual_family,
                actual_family=actual_family,
                episode_id=0,
                efficiency_action_cost=2.0,
            )
        contingency_store.upsert_contingency(
            Contingency(
                id=1,
                context_level=0,
                context_signature=(1,),
                action=1,
                transformation_family=10,
                support_count=5,
                confidence=0.95,
            )
        )
        connection.commit()

    analyze_future_effects(
        db_path=str(db_path),
        game="tt01",
        seed=0,
        steps=5,
        horizon=2,
    )
    deltas = _future_option_deltas_by_interaction_id(db_path, horizon=2)
    _apply_future_option_efficiency_diagnostics(db_path, deltas)

    assert len(deltas) > 0
    with sqlite3.connect(db_path) as connection:
        interaction_count = connection.execute(
            "SELECT COUNT(*) FROM interactions WHERE efficiency_future_option_gain_per_cost IS NOT NULL"
        ).fetchone()[0]
        prediction_count = connection.execute(
            "SELECT COUNT(*) FROM prediction_results WHERE efficiency_future_option_gain_per_cost IS NOT NULL"
        ).fetchone()[0]
    assert interaction_count > 0
    assert prediction_count > 0


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
        reward=0,
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
        reward=0,
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
        reward=0,
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

    assert score.survival_impact >= 0.75


def test_interaction_significance_graph_proxy_boosts_explanatory_potential() -> None:
    score = compute_interaction_significance(
        reward=0,
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
        assert row[0] == "isf_v01"
        assert all(value is not None for value in row[1:])
    finally:
        connection.close()


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


def test_v05b_failure_reason_assignment() -> None:
    reasons = classify_failure_reasons(
        {
            "non_preserve_ratio": 0.0,
            "non_preserve_count": 0,
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

    assert "PRESERVE_ONLY_OR_NEAR_PRESERVE_ONLY" in reasons
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
    assert parse_v05c_samplers("random_baseline,reset_aware_mixed") == ("random_baseline", "reset_aware_mixed")
    assert "tt01" in parse_v05c_games("failed_representatives")
    assert InteractionSamplingConfig().commit_steps == 1000


def test_v05c_games_all_expands_to_registered_games(monkeypatch) -> None:
    monkeypatch.setattr(interaction_sampling, "registered_game_ids", lambda env_root=None: ("pb02", "tt01", "ab01"))

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


def test_game_set_manifest_auto_resolves_extended32_from_available_games() -> None:
    from pathlib import Path

    m2_families = load_m2_families_v08d(Path("runs/v6/v07_cd2_extended32_expanded"))
    manifest = load_game_set_manifest(
        fallback_games=tuple(sorted({game for family in m2_families for game in family.games_present}))
    )

    assert manifest.name == "extended32_v08"
    assert len(manifest.families) == 16
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

    def fake_analyze_future_effects(**kwargs) -> list[dict]:
        captured["effects_call"] = kwargs
        return []

    def fake_write_sampling_metadata(path, **values) -> None:
        captured["metadata_path"] = path
        captured["metadata"] = values

    monkeypatch.setattr(interaction_sampling, "make_sampler", fake_make_sampler)
    monkeypatch.setattr(interaction_sampling, "ArcGridEnvironment", DummyEnv)
    monkeypatch.setattr(interaction_sampling, "V6System", DummySystem)
    monkeypatch.setattr(interaction_sampling, "analyze_future_effects", fake_analyze_future_effects)
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
        (_obs(0), _obs(1), 4),
        (_obs(1), _obs(2), 4),
        (_obs(2), _obs(2), 4),
        (_obs(0), _obs(1), 4),
        (_obs(1), _obs(2), 4),
        (_obs(2), _obs(3), 4),
    ]
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


def _trace_event(*, step: int, action: int, context: tuple[str, ...], outcome: str):
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
        terminal_observed=outcome == "terminal_transition",
        delta_summary={},
    )
