from __future__ import annotations

import sqlite3

from v6.carrier_emergence import CarrierEmergenceTracker
from v6 import hypothesis_h06_report as h06
from v6 import hypothesis_h08_report as h08
from v6.memory.compact_memory import ensure_memory_layout
from v6.v63_higher_order_semantics import (
    _build_relational_world_models,
    install_v63_higher_order_semantics,
)


def test_carrier_emergence_uses_threshold_crossing_step() -> None:
    install_v63_higher_order_semantics()
    tracker = CarrierEmergenceTracker(
        min_support=2,
        min_distinct_contexts=2,
        min_prediction_lift=-1.0,
        min_compression_gain=0.0,
    )
    tracker.record_interaction(
        interaction_id="1",
        carrier_signature="carrier-a",
        context_signature="c1",
        action_signature="a1",
        family_id="f1",
        delta_signature="d1",
        prediction_correct=True,
        carrier_source="object",
        global_step=1,
    )
    first = tracker._build_candidate("carrier-a")
    assert first.status != "emergent_carrier"
    assert first.first_emergent_global_step is None

    tracker.record_interaction(
        interaction_id="2",
        carrier_signature="carrier-a",
        context_signature="c2",
        action_signature="a1",
        family_id="f1",
        delta_signature="d2",
        prediction_correct=True,
        carrier_source="object",
        global_step=5,
    )
    emerged = tracker._build_candidate("carrier-a")
    assert emerged.status == "emergent_carrier"
    assert emerged.first_seen_global_step == 1
    assert emerged.first_emergent_global_step == 5


def test_h06_surrogate_game_cannot_validate_transfer() -> None:
    install_v63_higher_order_semantics()
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        """
        SELECT 'cross_game' AS transfer_kind,
               1 AS source_game_is_surrogate,
               0 AS target_game_is_surrogate
        """
    ).fetchone()
    assert h06._transfer_provenance_error(row) == "surrogate_game_provenance"


def test_h08_requires_multiple_concepts() -> None:
    install_v63_higher_order_semantics()
    record = {
        "effective_currently_coherent": True,
        "effective_validation_status": "passed",
        "has_positive_heldout_gain": True,
        "cross_context_count": 3,
        "cross_game_count": 2,
        "supported_context_count": 3,
        "concept_link_count": 1,
        "role_link_count": 1,
        "family_link_count": 2,
        "verified_predicted_outcome_count": 1,
        "coherence_score": 0.8,
        "explanatory_coverage": 0.5,
        "candidate_only": False,
    }
    assert not h08._component_passes_h08_validity(record)
    record["concept_link_count"] = 2
    assert h08._component_passes_h08_validity(record)


def test_relational_world_model_uses_two_concepts_and_later_prediction(tmp_path) -> None:
    install_v63_higher_order_semantics()
    paths = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(paths.current_state) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            INSERT INTO concept_candidates (
                concept_signature, concept_type, support_count,
                linked_role_count, linked_carrier_count, linked_family_count,
                transfer_success_count, strong_transfer_success_count,
                cross_game_count, cross_context_count, compression_gain,
                explanatory_reach, promotion_score, first_seen_global_step,
                last_seen_global_step, is_promoted, promotion_status
            ) VALUES
                ('concept:a','relational',10,1,2,2,4,3,2,3,2.0,8.0,0.9,3,10,1,'promoted'),
                ('concept:b','relational',10,1,2,2,4,3,2,3,2.0,8.0,0.85,4,10,1,'promoted')
            """
        )
        for concept in ("concept:a", "concept:b"):
            for kind, values in {
                "role": ("r1",),
                "carrier": ("ca", "cb"),
                "family": ("f1", "f2"),
                "context": ("c1", "c2", "c3"),
                "game": ("g1", "g2"),
            }.items():
                for value in values:
                    connection.execute(
                        """
                        INSERT INTO concept_links (
                            concept_signature, linked_type, linked_key,
                            support_count, first_seen_global_step,
                            last_seen_global_step
                        ) VALUES (?, ?, ?, 1, 3, 10)
                        """,
                        (concept, kind, value),
                    )
        connection.execute(
            """
            INSERT INTO role_links (
                role_signature, linked_type, linked_key, support_count,
                first_seen_global_step, last_seen_global_step
            ) VALUES
                ('r1','family','f1',1,2,10),
                ('r1','family','f2',1,2,10)
            """
        )
        connection.execute(
            "INSERT INTO family_members (family_signature, contingency_key, support_count) VALUES ('f1','x1',5)"
        )
        connection.execute(
            "INSERT INTO family_members (family_signature, contingency_key, support_count) VALUES ('f2','x2',5)"
        )
        connection.execute(
            """
            INSERT INTO future_option_events (
                event_id, owner_type, owner_key, game, context_key,
                option_delta, first_seen_global_step, last_seen_global_step
            ) VALUES
                ('e1','family','f1','g1','c1',1.0,7,7),
                ('e2','family','f1','g1','c1',1.0,8,8),
                ('e3','family','f2','g1','c1',1.0,9,9)
            """
        )
        first = _build_relational_world_models(connection)
        assert first["world_model_component_count"] == 1
        row = connection.execute(
            "SELECT component_signature, linked_concept_count, observed_outcome_count "
            "FROM world_model_components"
        ).fetchone()
        assert int(row["linked_concept_count"]) == 2
        assert int(row["observed_outcome_count"]) == 0
        component_signature = str(row["component_signature"])
        assert connection.execute(
            "SELECT COUNT(*) FROM world_model_links "
            "WHERE component_signature=? AND linked_type='concept'",
            (component_signature,),
        ).fetchone()[0] == 2

        connection.execute(
            """
            INSERT INTO future_option_events (
                event_id, owner_type, owner_key, game, context_key,
                option_delta, first_seen_global_step, last_seen_global_step
            ) VALUES ('e4','family','f1','g1','c1',1.0,11,11)
            """
        )
        second = _build_relational_world_models(connection)
        assert second["world_model_component_count"] == 1
        row = connection.execute(
            "SELECT linked_concept_count, observed_outcome_count, "
            "       correct_prediction_count, prediction_evidence_status "
            "FROM world_model_components"
        ).fetchone()
        assert int(row["linked_concept_count"]) == 2
        assert int(row["observed_outcome_count"]) >= 1
        assert int(row["correct_prediction_count"]) >= 1
        assert row["prediction_evidence_status"] == "verified"
