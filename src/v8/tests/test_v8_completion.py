from __future__ import annotations

import unittest

from v8.arena import NodeRecord
from v8.context_refinement import ContextRefiner
from v8.dirty import DirtyKeyTracker
from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.future_options import FutureOptionEstimator
from v8.lifecycle import LifecycleController
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState
from v8.planning import Planner
from v8.prediction import PredictionEstimator
from v8.preference import PreferenceEstimator
from v8.roles import FunctionalRoleEstimator
from v8.strategies import StrategyEvidence
from v8.transfer import TransferValidator


def node(
    level: MemoryLevel,
    memory_type: MemoryType,
    key: tuple[int, ...],
    *,
    support: int = 2,
    game_mask: int = 0,
    significance: float = 1.0,
    learning: float = 1.0,
    future: float = 0.0,
    cognitive_state: int = 0,
    validation_state: int = 0,
) -> NodeRecord:
    uid = MemoryUid.from_key(level, memory_type, key)
    return NodeRecord(
        uid=uid,
        fingerprint=1,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
        support_count=support,
        significance_sum=significance,
        prediction_error_sum=0.0,
        learning_value_sum=learning,
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=future,
        score_weight=1.0,
        updated_watermark=10,
        game_mask=game_mask,
        cognitive_state=cognitive_state,
        validation_state=validation_state,
    )


class DirtyTests(unittest.TestCase):
    def test_repeated_invalidation_coalesces_until_completed(self) -> None:
        dirty = DirtyKeyTracker()
        self.assertTrue(dirty.invalidate("x", 1))
        self.assertFalse(dirty.invalidate("x", 2))
        self.assertEqual(dirty.begin("x"), 2)
        self.assertFalse(dirty.complete("x", 2))
        self.assertFalse(dirty.state("x").queued)


class PredictionAndContextTests(unittest.TestCase):
    def test_prediction_activates_only_after_supported_stable_contingency(self) -> None:
        estimator = PredictionEstimator(min_support=3, stability_threshold=0.6)
        sparse = (node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4), support=1),)
        self.assertEqual(estimator.evaluate(sparse), ())
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4), support=8),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 9, 5), support=2),
        )
        evidence = estimator.evaluate(rows)
        self.assertEqual(len(evidence), 2)
        self.assertTrue(all(item.error >= 0 for item in evidence))

    def test_contradiction_proposes_context_refinement(self) -> None:
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4), support=5),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 9, 5), support=3),
        )
        proposals = ContextRefiner(min_support=4, contradiction_threshold=0.2).propose(rows)
        self.assertGreaterEqual(len(proposals), 2)


class RoleAndFutureOptionTests(unittest.TestCase):
    def test_distinct_carriers_form_one_functional_role(self) -> None:
        rows = (
            node(MemoryLevel.M3, MemoryType.CARRIER, (100, 200, 1), game_mask=1),
            node(MemoryLevel.M3, MemoryType.CARRIER, (100, 201, 1), game_mask=2),
        )
        roles = FunctionalRoleEstimator().propose(rows)
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0].game_evidence_count, 2)

    def test_bounded_future_option_uses_context_transition_graph(self) -> None:
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 1, 100, 20)),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (20, 1, 101, 30)),
        )
        evidence = FutureOptionEstimator(horizon=3).evaluate(rows)
        self.assertEqual(len(evidence), 2)
        self.assertTrue(any(item.delta != 0 for item in evidence))


class TransferPlanningPreferenceTests(unittest.TestCase):
    def test_transfer_structural_candidate_is_not_empirical_validation(self) -> None:
        row = node(MemoryLevel.M4, MemoryType.CONCEPT, (7, 1), support=5, game_mask=3)
        validator = TransferValidator(effect_threshold=0.1)
        self.assertEqual(len(validator.candidates((row,))), 1)
        self.assertFalse(validator.empirically_validated(row.uid))
        validator.record_trial(row.uid, target_game_hash=99, metric_on=0.8, metric_off=0.5)
        self.assertTrue(validator.empirically_validated(row.uid))

    def test_replanning_preserves_outcome_identity(self) -> None:
        outcome = MemoryUid(11, 22)
        s1 = StrategyEvidence(MemoryUid(1, 1), outcome, 1, 55, 8, 1.0, 1.0)
        s2 = StrategyEvidence(MemoryUid(2, 2), outcome, 2, 55, 5, 1.0, 1.0)
        planner = Planner()
        # Planner hashes raw context, so use its generated bucket for test evidence.
        from v8.model import stable_u64
        bucket = stable_u64(7, person=b"v8-context")
        s1 = StrategyEvidence(s1.uid, outcome, 1, bucket, 8, 1.0, 1.0)
        s2 = StrategyEvidence(s2.uid, outcome, 2, bucket, 5, 1.0, 1.0)
        current = planner.select(context_signature=7, available_actions=(1, 2), strategies=(s1, s2))
        self.assertIsNotNone(current)
        replanned = planner.replan(current, context_signature=7, available_actions=(1, 2), strategies=(s1, s2))
        self.assertIsNotNone(replanned)
        self.assertEqual(replanned.outcome_uid, current.outcome_uid)
        self.assertNotEqual(replanned.strategy_uid, current.strategy_uid)

    def test_preference_requires_comparable_repeated_outcomes(self) -> None:
        a, b = MemoryUid(10, 10), MemoryUid(20, 20)
        strategies = (
            StrategyEvidence(MemoryUid(1, 1), a, 1, 77, 10, 1.0, 1.0),
            StrategyEvidence(MemoryUid(2, 2), b, 2, 77, 2, 0.5, 1.0),
        )
        evidence = PreferenceEstimator(support_threshold=6, stable_margin=0.3).evaluate(strategies)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].state, "STABLE")
        self.assertEqual(evidence[0].preferred, a)


class LifecycleAndReportingTests(unittest.TestCase):
    def test_high_fitness_candidate_promotes_with_hysteresis_model(self) -> None:
        row = node(
            MemoryLevel.M4,
            MemoryType.CONCEPT,
            (7, 1),
            support=20,
            significance=1.0,
            learning=1.0,
            future=4.0,
            cognitive_state=int(CognitiveState.CANDIDATE),
            validation_state=int(ValidationState.STRUCTURAL),
        )
        decision = LifecycleController(promotion_threshold=0.3).decide(row)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.cognitive_state, int(CognitiveState.ACTIVE))

    def test_scientific_contract_does_not_validate_from_partial_proxy(self) -> None:
        evaluator = ScientificHypothesisEvaluator()
        uid = MemoryUid(1, 2)
        partial = EvidenceRecord.for_uid(
            "x",
            uid,
            evidence_kind="transfer_structural",
            watermark=10,
            raw_value=1.0,
            normalized_value=1.0,
            developmental_stage=4,
            validation_state=int(ValidationState.STRUCTURAL),
        )
        statuses = evaluator.status_map(evaluator.evaluate((partial,)))
        self.assertEqual(statuses["H06"], "PARTIALLY_VALID")
        self.assertNotEqual(statuses["H06"], "VALID")


if __name__ == "__main__":
    unittest.main()
