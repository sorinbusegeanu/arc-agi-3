from __future__ import annotations

import json
import numpy as np

from v6.cli import build_parser
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
from v6.context_depth_compare_v07 import (
    ContextDepthCompareConfig,
    format_context_depth_comparison,
    run_context_depth_compare_v07,
)
from v6.contingency.contingency_learner import ContingencyLearner
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
    best_by_game as sampling_best_by_game,
    parse_v05c_games,
    parse_v05c_samplers,
    resolve_interaction_sampling_scope,
    sampler_comparison_rows,
    sampling_db_path,
    validation_summary as sampling_validation_summary,
    write_interaction_sampling_reports,
)
from v6.evaluation.future_effects import (
    FutureEffect,
    InteractionEvent,
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
from v6.main import V6Config, V6System
from v6.memory_types import M3RoleCandidate
from v6.m2_expand_v08c import M2ExpandV08cConfig, run_m2_expand_v08c
from v6.memory.interaction_store import encode_array
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
from v6.role_transfer_v09 import RoleTransferV09Config, build_source_role_prototypes, load_neighborhoods, load_roles, run_role_transfer_v09
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
    neighborhoods = build_discriminative_neighborhoods(families, support, {"g1": "collection", "g2": "push_crate", "g3": "switch_unlock", "g4": "teleport_warp"})

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
