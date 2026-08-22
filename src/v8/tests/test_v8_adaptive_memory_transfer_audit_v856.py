from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8  # noqa: F401 - installs the production runtime stack
from v8 import adaptive_memory_transfer_audit_v856 as audit
from v8 import adaptive_memory_transfer_candidate_pool_v856 as pool
from v8 import adaptive_memory_transfer_experiment_v856 as experiment
from v8 import learning_fixes_v088 as v088
from v8 import learning_transfer_correctness_v854 as v854
from v8.arena import EdgeRecord, NodeRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    stable_u64,
)
from v8.runtime_v82 import V82ContinuousMemoryRuntime
from v8.similarity import BoundedNeighborhoodSimilarity
from v8.structural_correspondence import StructuralCorrespondenceEstimator
from v8.transfer import TransferValidator


def _node(level, memory_type, key, *, support=3):
    uid = MemoryUid.from_key(level, memory_type, key)
    return NodeRecord(
        uid=uid,
        fingerprint=1,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=tuple(int(v) for v in key),
        support_count=int(support),
        significance_sum=1.0,
        prediction_error_sum=0.0,
        learning_value_sum=1.0,
        transfer_prior_sum=0.0,
        explanatory_sum=1.0,
        future_option_sum=0.0,
        score_weight=1.0,
        updated_watermark=1,
        game_mask=0,
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.STRUCTURAL),
    )


def _edge(source, relation, target, *, support=1, score=0.0):
    return EdgeRecord(
        source,
        int(relation),
        target,
        int(support),
        1,
        score_sum=float(score),
        score_weight=1.0 if score else 0.0,
    )


class _ReadView:
    def __init__(self, games):
        self.games = games

    def source_games(self, uid):
        return frozenset(self.games.get(uid, ()))


class AdaptiveMemoryTransferAuditV856Tests(unittest.TestCase):
    def test_final_runtime_authority_filters_before_async_feedback(self) -> None:
        self.assertIs(V82ContinuousMemoryRuntime.record_actor_results, audit._record_actor_results_v856)

    def test_unknown_provenance_is_not_allowed_to_rewrite_intrinsic_strategy_quality(self) -> None:
        runtime = SimpleNamespace(read_view=_ReadView({}))
        uid = MemoryUid(7, 101)
        row = SimpleNamespace(
            game_id="target",
            strategy_stats=(SimpleNamespace(strategy_uid=uid),),
            preference_probes=(),
            replanning_trials=(),
            replans=0,
            pending_learning=None,
        )
        filtered = audit._filter_target_scoped_learning_v856(runtime, row)
        self.assertEqual(filtered.strategy_stats, ())

    def test_foreign_preference_probe_is_filtered_but_target_local_probe_survives(self) -> None:
        target = int(stable_u64("target", person=b"v8-game"))
        source = int(stable_u64("source", person=b"v8-game"))
        local_a, local_b = MemoryUid(6, 1), MemoryUid(6, 2)
        foreign = MemoryUid(6, 3)
        runtime = SimpleNamespace(
            read_view=_ReadView(
                {
                    local_a: (target,),
                    local_b: (target,),
                    foreign: (source,),
                }
            )
        )
        local_probe = SimpleNamespace(outcome_a=local_a, outcome_b=local_b)
        foreign_probe = SimpleNamespace(outcome_a=foreign, outcome_b=local_b)
        row = SimpleNamespace(
            game_id="target",
            strategy_stats=(),
            preference_probes=(local_probe, foreign_probe),
            replanning_trials=(),
            replans=0,
            pending_learning=None,
        )
        filtered = audit._filter_target_scoped_learning_v856(runtime, row)
        self.assertEqual(filtered.preference_probes, (local_probe,))

    def test_missing_formation_provenance_cannot_validate_transfer(self) -> None:
        validator = TransferValidator(effect_threshold=0.0)
        trial = validator.record_trial(
            MemoryUid(4, 1),
            target_game_hash=123,
            metric_on=2.0,
            metric_off=1.0,
            formation_games=(),
        )
        self.assertFalse(trial.passed)

    def test_supersedes_does_not_manufacture_cross_game_provenance(self) -> None:
        source = MemoryUid(4, 1)
        unrelated = MemoryUid(4, 2)
        game_hash = 999
        edges = (
            _edge(source, RelationType.SUPERSEDES, unrelated),
            _edge(unrelated, RelationType.GAME_PROVENANCE, MemoryUid(0, game_hash)),
        )
        result = TransferValidator._provenance_from_edges((source,), edges)
        self.assertEqual(result[source], ())

    def test_transfer_evidence_nodes_are_excluded_from_similarity_and_transfer_candidates(self) -> None:
        concept = _node(MemoryLevel.M4, MemoryType.CONCEPT, (11,))
        evidence = _node(MemoryLevel.M4, MemoryType.TRANSFER_EVIDENCE, (12, 13, 14, 1))
        descriptors = BoundedNeighborhoodSimilarity.descriptors((concept, evidence), ())
        self.assertIn(concept.uid, descriptors)
        self.assertNotIn(evidence.uid, descriptors)
        candidates = TransferValidator().candidates((evidence,), ())
        self.assertEqual(candidates, ())

    def test_structural_correspondence_ignores_edge_exposure_frequency(self) -> None:
        left = _node(MemoryLevel.M3, MemoryType.ROLE, (1, 1))
        right = _node(MemoryLevel.M3, MemoryType.ROLE, (2, 1))
        lower_left = _node(MemoryLevel.M2, MemoryType.FAMILY, (10,))
        lower_right = _node(MemoryLevel.M2, MemoryType.FAMILY, (20,))
        graph = (
            _edge(left.uid, RelationType.EXPLAINS, lower_left.uid, support=100),
            _edge(right.uid, RelationType.EXPLAINS, lower_right.uid, support=1),
            _edge(left.uid, RelationType.SIMILAR_TO, right.uid, score=0.9),
        )
        result = StructuralCorrespondenceEstimator(theta_struct=0.5).evaluate(
            (left, right, lower_left, lower_right),
            graph,
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].admissible)
        self.assertEqual(result[0].epsilon_struct, 0.0)

    def test_explicit_transfer_composite_can_execute_before_fallback_admission(self) -> None:
        strategy_uid = MemoryUid(7, 201)
        outcome_uid = MemoryUid(6, 201)
        strategy = SimpleNamespace(
            strategy_uid=strategy_uid,
            outcome_uid=outcome_uid,
        )
        sequence = SimpleNamespace(score=3.5, strategy_uid=strategy_uid)
        view = SimpleNamespace(
            _strategy_by_context={123: [strategy]},
            _strategy_fallback=(),
            _strategy_version=(1,),
            _v854_transfer_active={"target": (sequence, 1)},
            _refresh_strategy_cache=lambda: None,
        )
        prior = os.environ.get("ARC_AGI3_V8_SAMPLING_MODE")
        os.environ["ARC_AGI3_V8_SAMPLING_MODE"] = "TRANSFER"
        try:
            with patch("v8.environment_neutrality_v837._current_game_id", return_value="target"), patch.object(
                v854,
                "_ordered_action",
                return_value=(2, "M7_SEQUENCE_CORRESPONDENCE", strategy_uid),
            ):
                plans = pool._plan_chain_v856(view, 123, (1, 2, 3))
        finally:
            if prior is None:
                os.environ.pop("ARC_AGI3_V8_SAMPLING_MODE", None)
            else:
                os.environ["ARC_AGI3_V8_SAMPLING_MODE"] = prior
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].strategy_uid, strategy_uid)
        self.assertEqual(plans[0].action_id, 2)

    def test_held_out_on_policy_uses_target_grounded_action_not_source_raw_action(self) -> None:
        ancestor = MemoryUid(4, 301)
        strategy_uid = MemoryUid(7, 301)
        view = SimpleNamespace(
            strategy_has_ancestor=lambda strategy, required: strategy == strategy_uid and required == ancestor,
        )
        grounded = {
            9: ((4.0, strategy_uid, "M7_CORRESPONDENCE"),),
        }
        with patch.object(v854, "_ordered_sequences", return_value=()), patch(
            "v8.environment_neutrality_v837._grounded_transfer_index",
            return_value=(grounded, {}),
        ):
            action, state = experiment._grounded_candidate_action(
                read_view=view,
                game_id="target",
                context=123,
                actions=(2, 9),
                required_ancestor=ancestor,
                active_sequence=None,
            )
        self.assertEqual(action, 9)
        self.assertIsNone(state)

    def test_grounded_experiment_policy_is_installed_over_v088_raw_planning_probe(self) -> None:
        from v8 import experiments

        self.assertIs(v088._probe_policy_v088, experiment._probe_policy_grounded_v856)
        self.assertIs(experiments._probe_policy, experiment._probe_policy_grounded_v856)

    def test_candidate_pool_and_audit_are_installed_in_final_stack(self) -> None:
        from v8 import environment_neutrality_v837 as v837
        from v8 import sampling_portfolio_v831 as portfolio

        self.assertIs(portfolio._BASE_PLAN_CHAIN, pool._plan_chain_v856)
        self.assertIs(v854._ordered_sequences, pool._ordered_transfer_candidates_v856)
        self.assertIs(V82ContinuousMemoryRuntime.record_actor_results, audit._record_actor_results_v856)
        self.assertTrue(callable(v837._grounded_transfer_index))


if __name__ == "__main__":
    unittest.main()
