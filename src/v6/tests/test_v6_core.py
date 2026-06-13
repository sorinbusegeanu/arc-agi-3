from __future__ import annotations

import numpy as np

from v6.cli import build_parser
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
