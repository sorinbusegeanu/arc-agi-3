from __future__ import annotations

from types import SimpleNamespace

import v7.hypotheses as hypotheses
from v7.context_evidence import ContextEpisodeEvidence
from v7.derivation.online_runtime import (
    _continuous_episode_rows,
    _rotating_budget,
)
from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.planning_runtime import (
    Phase1PlanningBuilder,
    _procedure_signature,
)
from v7.derivation.scientific import (
    TYPE_CONCEPT,
    TYPE_CONTINGENCY,
    TYPE_ROLE,
    TYPE_WORLD_MODEL,
)
from v7.environment.cognition import (
    ContextualActionScorer,
    DecisionContext,
    LocalCognitionOverlay,
)
from v7.environment.phase1_policy import _is_transfer_frontier
from v7.hypotheses import _h06, _h08, _h10, _h12
from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.developmental_policy import DevelopmentStage, infer_development_stage
from v7.memory.evidence_lifecycle import (
    EvidenceLifecycleStore,
    ProvenanceRecord,
    TransferTrialRecord,
)
from v7.memory.evidence_store import EvidenceStore
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import (
    ContingencyIndexMutation,
    RoleConceptIndexMutation,
    RoleIndexMutation,
)
from v7.memory.models import MemoryNode, NodeMutation, ScoreMutation
from v7.memory.planning import TYPE_EXECUTABLE_PROCEDURE
from v7.memory.status import MemoryStatus
from v7.memory.writer import CanonicalMemoryWriter
from v7.runtime import V7Runtime, V7RuntimeConfig


def test_failure_triggered_reset_clears_failure_streak() -> None:
    overlay = LocalCognitionOverlay()
    contexts = (10, 11, 12, 13, 14)
    for index in range(3):
        overlay.record_step(
            contexts=contexts,
            next_contexts=(),
            action_id=1,
            outcome_signature=100 + index,
            terminal_polarity=-1,
            prediction_error=0.0,
            future_option_delta=0.0,
            changed=True,
        )
    assert overlay.should_reset() is True
    overlay.reset_episode_history(
        keep_statistics=True,
        clear_failure_streak=True,
    )
    assert overlay.should_reset() is False


def test_rotating_budget_eventually_visits_entire_frontier() -> None:
    values = tuple(range(20))
    seen = set()
    for generation in range(5):
        chunk = _rotating_budget(values, 4, generation)
        assert len(chunk) == 4
        seen.update(chunk)
    assert seen == set(values)


def test_transition_continuity_rejects_reset_or_parallel_segment_stitching() -> None:
    prior = {
        "trajectory_segment_id": "job_1/segment_0",
        "source_global_step": 10,
        "terminal_polarity": 0,
        "next_context_signatures": (1, 2, 3, 13, 15),
    }
    current = {
        "trajectory_segment_id": "job_1/segment_0",
        "source_global_step": 11,
        "context_signatures": (1, 2, 3, 13, 16),
    }
    assert _continuous_episode_rows(prior, current) is True
    assert _continuous_episode_rows(
        prior,
        {**current, "trajectory_segment_id": "job_2/segment_0"},
    ) is False
    assert _continuous_episode_rows(
        prior,
        {**current, "source_global_step": 12},
    ) is False


def test_terminal_episode_uses_c3_as_canonical_planning_memory() -> None:
    writer = CanonicalMemoryWriter()
    pipeline = MemoryLearningPipeline(writer)
    evidence = ContextEpisodeEvidence(
        context_signature=14,
        action_id=2,
        outcome_signature=99,
        success=True,
        terminal_polarity=1,
        prediction_error=1.0,
        context_signatures=(10, 11, 12, 13, 14),
    )
    memory_id = pipeline.observe_episode(evidence)
    key = getattr(writer, "_canonical_registry").key_for(memory_id)
    assert key is not None
    assert key.parts[0] == 13
    assert writer.canonical_memory_id(
        CanonicalMemoryKey(MemoryLevel.M1, TYPE_CONTINGENCY, (14, 2, 99))
    ) is not None


def test_canonical_candidate_scores_are_support_weighted_not_last_write() -> None:
    writer = CanonicalMemoryWriter()
    key = CanonicalMemoryKey(MemoryLevel.M1, TYPE_CONTINGENCY, (1, 2, 3))
    resolved = writer.apply_canonical_candidate_batch(
        (
            CanonicalCandidateMutation(key, significance=0.0),
            CanonicalCandidateMutation(key, significance=1.0),
        )
    )
    memory_id = resolved[key]
    assert writer._scores[memory_id].significance == 0.5
    writer.apply_canonical_candidate_batch(
        (CanonicalCandidateMutation(key, significance=1.0),)
    )
    assert abs(writer._scores[memory_id].significance - (2.0 / 3.0)) < 1e-12


def test_active_ranked_retrieval_excludes_demoted_and_rejected_before_limits() -> None:
    writer = CanonicalMemoryWriter()
    context = 50
    action = 3
    roles = tuple(MemoryId(value) for value in range(1, 71))
    writer.apply_mutation_batch(
        NodeMutation(
            role_id,
            MemoryLevel.M3,
            TYPE_ROLE,
            support_delta=10 if role_id == roles[-1] else 1,
            status_flags=int(MemoryStatus.DEMOTED) if role_id == roles[0] else None,
        )
        for role_id in roles
    )
    writer.apply_score_batch(
        (ScoreMutation(roles[-1], significance=1.0, learning_value=1.0),)
    )
    writer.apply_role_index_batch(
        RoleIndexMutation(context, action, role_id) for role_id in roles
    )

    rejected = tuple(MemoryId(value) for value in range(1000, 1130))
    valid = MemoryId(2000)
    writer.apply_mutation_batch(
        (
            *(
                NodeMutation(
                    concept_id,
                    MemoryLevel.M4,
                    TYPE_CONCEPT,
                    support_delta=5,
                    status_flags=int(ConceptValidationStatus.TRANSFER_REJECTED),
                )
                for concept_id in rejected
            ),
            NodeMutation(valid, MemoryLevel.M4, TYPE_CONCEPT, support_delta=5),
        )
    )
    writer.apply_score_batch(
        (ScoreMutation(valid, significance=1.0, transfer_prior=1.0),)
    )
    writer.apply_role_concept_index_batch(
        (
            *(
                RoleConceptIndexMutation(roles[-1], concept_id)
                for concept_id in rejected
            ),
            RoleConceptIndexMutation(roles[-1], valid),
        )
    )
    _, view, _ = writer.commit_generation()
    row = view.score_inputs(
        context_signature=context,
        action_ids=(action,),
        role_limit=2,
        concept_limit=1,
    )[0]
    assert roles[0] not in row.role_ids
    assert roles[-1] in row.role_ids
    assert row.concept_ids == (valid,)


def test_demoted_higher_memory_does_not_advance_development_stage() -> None:
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        (
            NodeMutation(MemoryId(1), MemoryLevel.M1, TYPE_CONTINGENCY, support_delta=2),
            NodeMutation(MemoryId(2), MemoryLevel.M2, 200, support_delta=2),
            NodeMutation(MemoryId(3), MemoryLevel.M3, TYPE_ROLE, support_delta=2),
            NodeMutation(
                MemoryId(4),
                MemoryLevel.M4,
                TYPE_CONCEPT,
                support_delta=4,
                status_flags=int(ConceptValidationStatus.TRANSFER_VALIDATED),
            ),
            NodeMutation(
                MemoryId(5),
                MemoryLevel.M6,
                TYPE_EXECUTABLE_PROCEDURE,
                support_delta=3,
                status_flags=int(MemoryStatus.DEMOTED),
            ),
        )
    )
    _, view, _ = writer.commit_generation()
    assert infer_development_stage(view) == DevelopmentStage.PLANNING


def test_context_specializes_without_two_x_support_and_does_not_double_count_local() -> None:
    writer = CanonicalMemoryWriter()
    c0 = MemoryId(1)
    c3 = MemoryId(2)
    writer.apply_mutation_batch(
        (
            NodeMutation(c0, MemoryLevel.M1, TYPE_CONTINGENCY, support_delta=10),
            NodeMutation(c3, MemoryLevel.M1, TYPE_CONTINGENCY, support_delta=2),
        )
    )
    writer.apply_score_batch(
        (
            ScoreMutation(c0, significance=0.5),
            ScoreMutation(c3, significance=0.5),
        )
    )
    writer.apply_contingency_index_batch(
        (
            ContingencyIndexMutation(10, 1, c0),
            ContingencyIndexMutation(13, 1, c3),
        )
    )
    _, view, _ = writer.commit_generation()
    overlay = LocalCognitionOverlay()
    overlay.record_step(
        contexts=(13,),
        next_contexts=(),
        action_id=1,
        outcome_signature=7,
        terminal_polarity=0,
        prediction_error=0.0,
        future_option_delta=0.0,
        changed=True,
    )
    contexts = DecisionContext((10, 11, 12, 13, 14), 12, 14)
    decision = ContextualActionScorer().score_actions(
        view=view,
        contexts=contexts,
        actions=(1,),
        overlay=overlay,
    )[0]
    assert decision.support.context_signature == 13
    assert decision.support.contextual_support == 2
    assert decision.support.local_support == 1


def test_transfer_frontier_needs_two_local_confirmations() -> None:
    contexts = SimpleNamespace(signatures=(1, 2, 3, 4, 5))
    one = SimpleNamespace(
        support=SimpleNamespace(context_rank=3, local_support=1)
    )
    two = SimpleNamespace(
        support=SimpleNamespace(context_rank=3, local_support=2)
    )
    assert _is_transfer_frontier(contexts, one) is True
    assert _is_transfer_frontier(contexts, two) is False


def test_heldout_transfer_summary_deduplicates_terminal_and_trajectory(tmp_path) -> None:
    ledger = EvidenceLifecycleStore(tmp_path / "ledger.sqlite")
    memory_id = MemoryId(1)
    try:
        ledger.append_provenance(
            (ProvenanceRecord(memory_id, 1, source_game="source"),)
        )
        ledger.append_transfer_trials(
            (
                TransferTrialRecord(memory_id, 2, "source", "target-a", True, 1.0, {"source_global_step": 10, "attribution": "terminal_action"}),
                TransferTrialRecord(memory_id, 2, "source", "target-a", True, 1.0, {"source_global_step": 10, "attribution": "trajectory_usage"}),
                TransferTrialRecord(memory_id, 2, "source", "target-b", False, 0.0, {"source_global_step": 20, "attribution": "trajectory_usage"}),
            )
        )
        assert ledger.heldout_transfer_summary((memory_id,))[memory_id][:2] == (2, 1)
        assert ledger.transfer_trial_exists(
            memory_id,
            target_game="target-a",
            source_global_step=10,
            attribution="trajectory_usage",
        )
    finally:
        ledger.close()


def test_h06_requires_success_across_two_distinct_game_pairs() -> None:
    class Stub:
        transfer_trials = [
            {"memory_id": 1, "source_game": "a", "target_game": "b", "source_game_count": 1, "success": True, "attribution": "trajectory_usage"},
            {"memory_id": 2, "source_game": "a", "target_game": "b", "source_game_count": 1, "success": True, "attribution": "trajectory_usage"},
            {"memory_id": 1, "source_game": "c", "target_game": "d", "source_game_count": 1, "success": False, "attribution": "trajectory_usage"},
            {"memory_id": 2, "source_game": "c", "target_game": "d", "source_game_count": 1, "success": False, "attribution": "trajectory_usage"},
        ]

        def nodes_at(self, level, type_id=None):
            if level == MemoryLevel.M3:
                return [
                    (1, SimpleNamespace(status_flags=0)),
                    (2, SimpleNamespace(status_flags=0)),
                ]
            return []

    result = _h06(Stub())
    assert result["raw_decision"] != "VALID"
    assert result["evidence"]["measurement"]["distinct_game_pair_count"] == 1


def test_h08_requires_post_creation_and_cross_game_on_same_model() -> None:
    sig_cross = hypotheses.world_transition_signature((1,), 1, (1, 2))
    sig_late = hypotheses.world_transition_signature((1,), 2, (1, 2))

    def pair(game, segment, start, generation, action):
        return [
            {
                "source_game": game,
                "trajectory_segment_id": segment,
                "source_global_step": start,
                "generation_id": generation,
                "terminal_polarity": 0,
                "action_id": action,
                "decision_concept_ids": [1],
            },
            {
                "source_game": game,
                "trajectory_segment_id": segment,
                "source_global_step": start + 1,
                "generation_id": generation,
                "terminal_polarity": 0,
                "action_id": 9,
                "decision_concept_ids": [1, 2],
            },
        ]

    model_cross = SimpleNamespace(created_generation=10, status_flags=0)
    model_late = SimpleNamespace(created_generation=1, status_flags=0)
    concepts = [
        (1, SimpleNamespace(status_flags=int(ConceptValidationStatus.TRANSFER_VALIDATED))),
        (2, SimpleNamespace(status_flags=int(ConceptValidationStatus.TRANSFER_VALIDATED))),
    ]

    class Registry:
        @staticmethod
        def key_for(memory_id):
            signature = sig_cross if int(memory_id) == 10 else sig_late
            return CanonicalMemoryKey(
                MemoryLevel.M5,
                TYPE_WORLD_MODEL,
                (signature,),
            )

    class Stub:
        registry = Registry()
        episodes = (
            pair("g1", "s1", 1, 1, 1)
            + pair("g2", "s2", 10, 1, 1)
            + pair("g3", "s3", 20, 5, 2)
        )

        def nodes_at(self, level, type_id=None):
            if level == MemoryLevel.M4:
                return concepts
            if level == MemoryLevel.M5:
                return [(10, model_cross), (11, model_late)]
            return []

    result = _h08(Stub())
    measurement = result["evidence"]["measurement"]
    assert measurement["models_with_post_creation_recurrence"] == 1
    assert measurement["cross_game_recurrent_model_count"] == 1
    assert measurement["robust_recurrent_model_count"] == 0
    assert result["raw_decision"] != "VALID"


def test_h10_large_score_effect_without_action_change_is_not_valid() -> None:
    class Stub:
        episodes = []

    for index in range(20):
        high = index < 10
        Stub.episodes.append(
            {
                "raw_action_option_delta": 2.0 if high else 0.0,
                "future_option_ablation_available": True,
                "future_option_observable": True,
                "future_option_ablation_score_delta": 0.3 if high else 0.01,
                "future_option_ablation_rank_lift": 0,
                "future_option_ablation_choice_changed": False,
            }
        )
    result = _h10(Stub())
    assert result["raw_decision"] == "INVALID"
    assert result["evidence"]["measurement"]["high_option_choice_change_count"] == 0


def test_h12_counts_actual_executable_procedure_use() -> None:
    strategy = SimpleNamespace(
        type_id=TYPE_EXECUTABLE_PROCEDURE,
        status_flags=0,
        created_generation=1,
    )

    class Stub:
        trajectories = [
            {"success": True, "source_game": "g", "level_key": "l", "source_global_step": 1, "steps_to_success": 5},
            {"success": True, "source_game": "g", "level_key": "l", "source_global_step": 2, "steps_to_success": 3},
        ]
        episodes = [
            {"generation_id": 2, "decision_strategy_ids": [10]}
        ]
        replay_ids = set()
        promotion_ids = set()

        def nodes_at(self, level, type_id=None):
            return [(10, strategy)] if level == MemoryLevel.M6 else []

    result = _h12(Stub())
    assert result["raw_decision"] == "VALID"
    assert result["evidence"]["measurement"]["executable_strategy_post_creation_use_count"] == 1


def test_planning_rebuild_does_not_overwrite_observed_strategy_failure_score(tmp_path) -> None:
    writer = CanonicalMemoryWriter()
    step_id = MemoryId(1)
    writer.apply_mutation_batch(
        (NodeMutation(step_id, MemoryLevel.M1, TYPE_CONTINGENCY, support_delta=1),)
    )
    signature = _procedure_signature((2,), (13,))
    key = CanonicalMemoryKey(
        MemoryLevel.M6,
        TYPE_EXECUTABLE_PROCEDURE,
        (signature,),
    )
    strategy_id = writer.apply_canonical_candidate_batch(
        (CanonicalCandidateMutation(key, significance=1.0, learning_value=1.0),)
    )[key]
    writer.apply_score_batch(
        (
            ScoreMutation(
                strategy_id,
                significance=0.2,
                learning_value=0.2,
                future_option_delta=-0.6,
            ),
        )
    )
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    try:
        builder = Phase1PlanningBuilder(writer, evidence)
        episodes = [
            {
                "memory_id": int(step_id),
                "source_game": "g",
                "trajectory_segment_id": "s",
                "source_global_step": 1,
                "terminal_polarity": 1,
                "context_signatures": (10, 11, 12, 13, 14),
            }
        ]
        trajectories = [
            {
                "success": True,
                "source_game": "g",
                "trajectory_segment_id": "s",
                "source_global_step": 1,
                "action_sequence": [2],
                "context_sequence": [13],
                "steps_to_success": 1,
                "future_option_per_action": 1.0,
            }
        ]
        builder._derive_procedures(
            episodes=episodes,
            trajectories=trajectories,
        )
        score = writer._scores[strategy_id]
        assert score.significance == 0.2
        assert score.learning_value == 0.2
        assert score.future_option_delta == -0.6
    finally:
        evidence.close()


def test_partially_valid_dependency_blocks_downstream_valid_report(tmp_path, monkeypatch) -> None:
    runtime = V7Runtime(V7RuntimeConfig.from_path(tmp_path, restore=False))
    try:
        monkeypatch.setattr(
            hypotheses,
            "_h05",
            lambda _snapshot: hypotheses._base(
                "PARTIALLY_VALID",
                1,
                {"usable_role_count": 0, "carrier_precedes_role": True},
            ),
        )
        monkeypatch.setattr(
            hypotheses,
            "_h06",
            lambda _snapshot: hypotheses._base(
                "VALID",
                4,
                {
                    "verified_single_source_cross_game_trials": 4,
                    "successful_verified_trials": 2,
                    "successful_role_count": 2,
                    "distinct_game_pair_count": 2,
                    "transfer_success_rate": 0.5,
                },
            ),
        )
        reports = hypotheses.evaluate_hypothesis_suite(
            runtime,
            epoch=0,
            output_root=tmp_path,
        )
        assert reports["H06"]["raw_decision"] == "VALID"
        assert reports["H06"]["dependency_gate"] == "FAIL"
        assert reports["H06"]["final_decision"] == "INSUFFICIENT_EVIDENCE"
    finally:
        runtime.close()
