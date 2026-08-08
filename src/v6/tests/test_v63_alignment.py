from __future__ import annotations

import sqlite3
import time

import pytest

from v6.interaction_significance import compute_interaction_significance
from v6.memory.promotion_engine import MemoryPromotionEngine
from v6.memory.substrate import MemoryNode, MemoryScore, MemorySubstrate
from v6.memory.v621_compact import merge_v621_state_connections
from v6.memory.v621_runtime import V621AbstractionEngine
from v6.memory.v63_policy import (
    ABSTRACTION_VERSION,
    V63CandidateBudget,
    _bounded_pairs,
    resolve_transfer_evidence,
)
from v6.memory.v63_transfer import (
    QUALIFIED_EVIDENCE_PREFIX,
    _initial_retention_score_v63,
)
from v6.memory_lifecycle import MemoryLifecycleManager, compute_memory_fitness


def test_v63_prediction_error_is_inactive_without_supported_expectation() -> None:
    score = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=0.0,
        actual_family_id=None,
        delta_id="d1",
        context_signature="c1",
        memory_counts={},
        graph_counts={"new_contingency": 0, "new_graph_edge": 0},
        weights={
            "survival_impact": 0.65,
            "prediction_error": 0.0,
            "learning_value": 0.35,
            "transfer_potential": 0.0,
            "explanatory_potential": 0.0,
        },
    )
    assert score.prediction_error == 0.0
    assert score.component_active is not None
    assert score.component_active["prediction_error"] is False
    assert score.version == "isf_v02"
    assert score.policy_version == "isf_v63"


def test_v63_transfer_prior_requires_cross_context_reuse() -> None:
    local_only = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id="7",
        delta_id="d1",
        context_signature="ctx",
        memory_counts={
            "actual_family_id:7": 4,
            "context_family:ctx|7": 4,
        },
        graph_counts={},
        future_option_delta=10.0,
    )
    assert local_only.transfer_prior == 0.0

    cross_context = compute_interaction_significance(
        terminated=False,
        truncated=False,
        prediction_correct=None,
        prediction_confidence=None,
        actual_family_id="7",
        delta_id="d1",
        context_signature="ctx",
        memory_counts={
            "actual_family_id:7": 4,
            "context_family:ctx|7": 1,
        },
        graph_counts={},
        future_option_delta=0.0,
    )
    assert cross_context.transfer_prior == pytest.approx(0.75)


def test_v63_memory_fitness_does_not_double_count_prediction_error() -> None:
    low_pe = compute_memory_fitness(
        isf_total=0.6,
        prediction_error=0.0,
        learning_value=0.1,
        transfer_potential=0.9,
        explanatory_potential=0.3,
        context_contradiction=False,
        replay_count=0,
    )
    high_pe = compute_memory_fitness(
        isf_total=0.6,
        prediction_error=1.0,
        learning_value=1.0,
        transfer_potential=0.9,
        explanatory_potential=0.3,
        context_contradiction=True,
        replay_count=99,
    )
    assert low_pe == pytest.approx(0.6)
    assert high_pe == pytest.approx(low_pe)


def test_v63_retention_does_not_feed_replay_or_future_option_back_into_fitness() -> None:
    low = _initial_retention_score_v63(
        None,
        isf_total=0.6,
        replay_priority=0.0,
        explanatory_reach=0.0,
        transfer_potential=0.0,
        recurrence=0.0,
        future_option_impact=0.0,
    )
    high_legacy_inputs = _initial_retention_score_v63(
        None,
        isf_total=0.6,
        replay_priority=1.0,
        explanatory_reach=0.0,
        transfer_potential=0.0,
        recurrence=0.0,
        future_option_impact=1.0,
    )
    assert low == pytest.approx(0.6)
    assert high_legacy_inputs == pytest.approx(low)


def test_v63_empirical_transfer_overrides_prior() -> None:
    prior, empirical, effective, status = resolve_transfer_evidence(
        {
            "transfer_prior": 0.9,
            "transfer_tests": 4,
            "transfer_success_count": 1,
        }
    )
    assert prior == pytest.approx(0.9)
    assert empirical == pytest.approx(0.25)
    assert effective == pytest.approx(0.25)
    assert status == "empirical"


def test_v63_concept_empirical_counts_only_qualified_unseen_reuse() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    MemoryPromotionEngine(memory)
    abstraction = V621AbstractionEngine(memory)

    role_ids = {"M3:role:r1", "M3:role:r2"}
    memory.upsert_node(
        MemoryNode(
            node_id="M4:concept:v63test",
            memory_level="M4",
            node_type="ConceptMemory",
            canonical_key="v63test",
            attrs={
                "source_roles": sorted(role_ids),
                "concept_version": ABSTRACTION_VERSION,
                "source_games": ["source_game"],
            },
        )
    )
    now = time.time()
    connection.execute(
        """
        INSERT INTO concept_transfer_attempts_v621(
            attempt_id, concept_id, game, context_key,
            action, predicted_family, actual_family,
            success, evidence_source, global_step, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "qualified",
            "M4:concept:v63test",
            "heldout_game",
            "ctx",
            1,
            "7",
            "7",
            1,
            QUALIFIED_EVIDENCE_PREFIX + "test",
            1,
            now,
        ),
    )
    connection.execute(
        """
        INSERT INTO concept_transfer_attempts_v621(
            attempt_id, concept_id, game, context_key,
            action, predicted_family, actual_family,
            success, evidence_source, global_step, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "unqualified",
            "M4:concept:v63test",
            "source_game",
            "ctx",
            1,
            "7",
            "7",
            1,
            "v63_unqualified_transfer:test",
            2,
            now,
        ),
    )
    connection.commit()

    counts = abstraction._direct_concept_transfer_for_roles(role_ids)
    assert counts == {"tests": 1, "successes": 1}


def test_v63_migration_and_legacy_world_model_path_disabled() -> None:
    connection = sqlite3.connect(":memory:")
    memory = MemorySubstrate(connection)
    engine = MemoryPromotionEngine(memory)

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(memory_scores)").fetchall()
    }
    assert "transfer_prior" in columns
    assert "transfer_empirical_rate" in columns
    assert "memory_fitness" in columns
    version = connection.execute(
        "SELECT value FROM memory_versions WHERE key='memory_substrate_schema'"
    ).fetchone()
    assert version is not None and version[0] == "v6.3"

    memory.upsert_node(
        MemoryNode(
            node_id="M4:concept:test",
            memory_level="M4",
            node_type="ConceptMemory",
            canonical_key="test",
            attrs={"explanatory_reach": 10},
        )
    )
    summary = engine.promote_concept_to_world_model_fragment()
    assert summary["count"] == 0
    assert summary["status"] == "delegated_to_v63_relational_world_model"
    assert memory.query_nodes(memory_level="M5") == []


def test_v63_retrospective_evidence_is_separate_from_prospective_values() -> None:
    lifecycle = MemoryLifecycleManager()
    record = lifecycle.register_interaction(
        interaction_id="1",
        family_id="f",
        context_signature="c",
        action_signature="a",
        carrier_signature=None,
        isf_total=0.4,
        prediction_error=0.0,
        learning_value=0.2,
        transfer_potential=0.3,
        explanatory_potential=0.1,
        context_contradiction=False,
        timestamp_step=1,
    )
    assert record.learning_value_realized is None
    lifecycle.apply_retrospective_evidence(
        "1",
        learning_value=0.8,
        explanatory_value=0.6,
        transfer_empirical_rate=0.5,
        reason="later_validation",
    )
    revised = lifecycle.records["1"]
    assert revised.learning_value == pytest.approx(0.2)
    assert revised.learning_value_realized == pytest.approx(0.8)
    assert revised.explanatory_value_realized == pytest.approx(0.6)
    assert revised.transfer_prior == pytest.approx(0.3)
    assert revised.transfer_empirical_rate == pytest.approx(0.5)
    assert revised.evidence_revision_count == 1


def test_v63_pair_generation_respects_budget() -> None:
    ids = [str(index) for index in range(100)]
    pairs = list(_bounded_pairs(ids, 17))
    assert len(pairs) == 17
    assert V63CandidateBudget().max_role_pair_comparisons > 17


def test_v63_compact_merge_preserves_score_extensions_and_evidence_tables() -> None:
    source = sqlite3.connect(":memory:")
    target = sqlite3.connect(":memory:")
    source_memory = MemorySubstrate(source)
    target_memory = MemorySubstrate(target)
    MemoryPromotionEngine(source_memory)
    MemoryPromotionEngine(target_memory)

    for memory in (source_memory, target_memory):
        memory.upsert_node(
            MemoryNode(
                node_id="M3:role:compact",
                memory_level="M3",
                node_type="FunctionalRoleMemory",
                canonical_key="compact",
                attrs={},
            )
        )
        memory.upsert_score(
            MemoryScore(
                node_id="M3:role:compact",
                isf_total=0.5,
            )
        )

    source.execute(
        """
        UPDATE memory_scores
        SET transfer_prior=0.7,
            transfer_empirical_rate=0.4,
            transfer_evidence_status='empirical',
            memory_fitness=0.6,
            score_policy_version='v63_unified_memory_fitness_v1'
        WHERE node_id='M3:role:compact'
        """
    )
    source.execute(
        """
        INSERT INTO memory_evidence_revisions_v63(
            revision_id, node_id, evidence_kind,
            prospective_value, realized_value,
            evidence_status, evidence_source, global_step, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "revision-1",
            "M3:role:compact",
            "transfer",
            0.7,
            0.4,
            "empirical",
            "test",
            10,
            time.time(),
        ),
    )
    source.commit()

    merge_v621_state_connections(source, target)
    score = target.execute(
        """
        SELECT transfer_prior, transfer_empirical_rate,
               transfer_evidence_status, memory_fitness, score_policy_version
        FROM memory_scores
        WHERE node_id='M3:role:compact'
        """
    ).fetchone()
    revision_count = target.execute(
        "SELECT COUNT(*) FROM memory_evidence_revisions_v63 WHERE revision_id='revision-1'"
    ).fetchone()[0]
    assert score is not None
    assert score[0] == pytest.approx(0.7)
    assert score[1] == pytest.approx(0.4)
    assert score[2] == "empirical"
    assert score[3] == pytest.approx(0.6)
    assert score[4] == "v63_unified_memory_fitness_v1"
    assert revision_count == 1
