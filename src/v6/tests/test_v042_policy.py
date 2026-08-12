from __future__ import annotations

import sqlite3

import numpy as np

from v6.contingency.contingency_learner import ContingencyLearner
from v6.delta.delta_extractor import extract_delta
from v6.interaction_significance import compute_interaction_significance
from v6.prediction.predictor import Predictor
from v6.v042_policy import RobustISFNormalizer
from v6 import higher_order_substrate as higher_order


def _score(*, correct: bool | None, confidence: float | None):
    return compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=correct,
        prediction_confidence=confidence,
        actual_family_id="1",
        delta_id="d1",
        context_signature="ctx",
        memory_counts={},
        graph_counts={},
        weights={
            "survival_impact": 0.2,
            "prediction_error": 0.25,
            "learning_value": 0.25,
            "transfer_potential": 0.15,
            "explanatory_potential": 0.15,
        },
    )


def test_prediction_error_requires_stable_expectation() -> None:
    learner = ContingencyLearner(support_threshold=3, confidence_threshold=0.8)
    predictor = Predictor(learner)
    context = ("x",)

    learner.update(context, 1, 7)
    assert predictor.predict_multi_scale({0: context}, 1) == 7
    speculative = _score(correct=False, confidence=1.0)
    assert speculative.prediction_error == 0.0
    assert speculative.component_active["prediction_error"] is False
    assert speculative.expectation_evidence_status == "speculative"

    learner.update(context, 1, 7)
    learner.update(context, 1, 7)
    assert predictor.predict_multi_scale({0: context}, 1) == 7
    supported = _score(correct=False, confidence=1.0)
    assert supported.prediction_error == 1.0
    assert supported.component_active["prediction_error"] is True
    assert supported.expectation_supported is True
    assert supported.expectation_support_count >= 3


def test_robust_isf_normalizer_winsorizes_and_ranks_after_cold_start() -> None:
    normalizer = RobustISFNormalizer(window=16, min_samples=2)
    weights = {
        "survival_impact": 1.0,
        "prediction_error": 0.0,
        "learning_value": 0.0,
        "transfer_potential": 0.0,
        "explanatory_potential": 0.0,
    }
    active = {key: key == "survival_impact" for key in weights}
    base = {key: 0.0 for key in weights}
    base["survival_impact"] = 0.1
    assert normalizer.normalize(base, weights=weights, active=active)["survival_impact"] == 0.1
    base["survival_impact"] = 0.2
    assert normalizer.normalize(base, weights=weights, active=active)["survival_impact"] == 0.2
    base["survival_impact"] = 1.0
    ranked = normalizer.normalize(base, weights=weights, active=active)["survival_impact"]
    assert 0.0 <= ranked <= 1.0
    assert ranked != 1.0


def test_arc_delta_operator_uses_categorical_identity_not_color_order() -> None:
    before = np.array([[7, 0, 0], [0, 0, 0]])
    after = np.array([[0, 7, 0], [0, 0, 0]])
    delta = extract_delta(before, after, delta_id=4)
    assert delta.dx == 1.0
    assert delta.dy == 0.0
    assert delta.delta_operator_version == "arc_d_o_v042"
    assert delta.change_tuples == ((0, 0, 7, 0), (0, 1, 0, 7))
    assert delta.connected_component_count == 1


def _validation_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE promotion_validation_state (
            candidate_type TEXT NOT NULL,
            candidate_signature TEXT NOT NULL,
            failure_count INTEGER NOT NULL DEFAULT 0,
            promotion_status TEXT,
            last_validation_scope TEXT,
            last_validation_prediction_lift REAL,
            last_validation_action_selection_lift REAL,
            last_validation_transfer_lift REAL,
            last_validation_epoch TEXT,
            last_validation_global_step INTEGER,
            last_validation_result TEXT,
            updated_global_step INTEGER,
            PRIMARY KEY(candidate_type, candidate_signature)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE concept_candidates (
            concept_signature TEXT PRIMARY KEY,
            support_count INTEGER,
            validation_evidence_count INTEGER,
            cross_context_count INTEGER,
            cross_game_count INTEGER
        )
        """
    )
    return conn


def test_demotion_enters_probation_when_context_is_unresolved() -> None:
    conn = _validation_connection()
    conn.execute(
        "INSERT INTO concept_candidates VALUES ('c1', 10, 10, 0, 0)"
    )
    status, failures, demoted = higher_order._update_promotion_validation_state(
        conn,
        candidate_type="concept",
        candidate_signature="c1",
        passed=False,
        demotion_failure_limit=2,
        validation_scope="relevant_later_global_step",
        validation_prediction_lift=-0.1,
        validation_action_selection_lift=-0.1,
        validation_transfer_lift=-0.1,
        updated_global_step=100,
        validation_epoch="e1",
        count_failure=True,
        retain_previous_promotion=False,
        previously_promoted=True,
        validation_result="insufficient_context",
    )
    assert demoted is False
    assert failures == 0
    row = conn.execute(
        "SELECT lifecycle_status, demotion_suppressed_reason FROM promotion_hysteresis_v042 "
        "WHERE candidate_type='concept' AND candidate_signature='c1'"
    ).fetchone()
    assert row["lifecycle_status"] == "probation"
    assert row["demotion_suppressed_reason"] == "unresolved_context"


def test_demotion_requires_repeated_comparable_failure_and_can_reactivate() -> None:
    conn = _validation_connection()
    conn.execute(
        "INSERT INTO concept_candidates VALUES ('c2', 20, 20, 3, 2)"
    )
    common = dict(
        state_conn=conn,
        candidate_type="concept",
        candidate_signature="c2",
        demotion_failure_limit=2,
        validation_scope="relevant_later_global_step",
        validation_prediction_lift=-0.1,
        validation_action_selection_lift=-0.1,
        validation_transfer_lift=-0.1,
        updated_global_step=100,
        count_failure=True,
        retain_previous_promotion=False,
        previously_promoted=True,
        validation_result="failed",
    )
    first = higher_order._update_promotion_validation_state(
        **common, passed=False, validation_epoch="e1"
    )
    assert first[2] is False
    second = higher_order._update_promotion_validation_state(
        **{**common, "updated_global_step": 200}, passed=False, validation_epoch="e2"
    )
    assert second[2] is True
    revived = higher_order._update_promotion_validation_state(
        **{**common, "updated_global_step": 300, "validation_result": "passed"},
        passed=True,
        validation_epoch="e3",
    )
    assert revived[2] is False
    row = conn.execute(
        "SELECT lifecycle_status, reactivation_count FROM promotion_hysteresis_v042 "
        "WHERE candidate_type='concept' AND candidate_signature='c2'"
    ).fetchone()
    assert row["lifecycle_status"] == "reactivated"
    assert row["reactivation_count"] == 1
