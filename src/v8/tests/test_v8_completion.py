from __future__ import annotations

import unittest
import threading
from dataclasses import replace
from types import SimpleNamespace

from v8.arena import NodeRecord
from v8.context_refinement import ContextRefiner
from v8.dirty import DirtyKeyTracker
from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.future_options import FutureOptionEstimator
from v8.lifecycle import LifecycleController
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState
from v8.actor import _select_replanning_ablation
from v8.planning import PlanSelection, Planner
from v8.peers import DevelopmentalPeerSupervisor, _transfer_freshness_kind
from v8.prediction import PredictionEstimator
from v8.preference import PreferenceEstimator
from v8.replanning import ReplanningController
from v8.runtime import ContinuousMemoryRuntime
from v8.roles import FunctionalRoleEstimator
from v8.strategies import StrategyEstimator, StrategyEvidence
from v8.transfer import TransferCandidate, TransferValidator
from v8.isf import infer_developmental_stage


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
    prediction_error: float = 0.0,
    updated_watermark: int = 10,
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
        prediction_error_sum=prediction_error,
        learning_value_sum=learning,
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=future,
        score_weight=1.0,
        updated_watermark=updated_watermark,
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
            node(
                MemoryLevel.M1,
                MemoryType.CONTINGENCY,
                (1, 2, 9, 5),
                support=2,
                prediction_error=1.0,
                updated_watermark=11,
            ),
        )
        evidence = estimator.evaluate(rows)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].expectation_uid, rows[0].uid)
        self.assertEqual(evidence[0].observation_uid, rows[1].uid)
        self.assertTrue(evidence[0].violated)

    def test_prediction_violation_rejects_unsupported_expectation(self) -> None:
        estimator = PredictionEstimator(min_support=3, stability_threshold=0.6)
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4), support=2),
            node(
                MemoryLevel.M1,
                MemoryType.CONTINGENCY,
                (1, 2, 9, 5),
                support=1,
                prediction_error=1.0,
            ),
        )
        self.assertEqual(estimator.evaluate(rows), ())
        self.assertIn(
            "insufficient_dominant_expectation_support", estimator.last_rejections
        )

    def test_contradiction_proposes_context_refinement(self) -> None:
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4), support=5),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 9, 5), support=3),
        )
        proposals = ContextRefiner(min_support=4, contradiction_threshold=0.2).propose(rows)
        self.assertGreaterEqual(len(proposals), 2)
        self.assertTrue(
            all(
                proposal.broad_prediction_error > proposal.refined_prediction_error
                and proposal.matched_holdout_count == 8
                and proposal.matched_prediction_error_gain > 0
                for proposal in proposals
            )
        )

    def test_context_gain_requires_supported_separating_partitions(self) -> None:
        rows = (
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 3, 4), support=3),
            node(MemoryLevel.M1, MemoryType.CONTINGENCY, (1, 2, 9, 5), support=1),
        )
        self.assertEqual(
            ContextRefiner(min_support=4, contradiction_threshold=0.2).propose(rows),
            (),
        )


class StageAndLiveTransferTests(unittest.TestCase):
    def test_stage_seven_is_reachable_only_with_matched_empirical_strategies(self) -> None:
        outcome = MemoryUid(70, 71)
        first = replace(
            node(
                MemoryLevel.M7,
                MemoryType.STRATEGY,
                (1, outcome.hi, outcome.lo, 9),
                cognitive_state=int(CognitiveState.ACTIVE),
            ),
            attempt_weight=2.0,
            success_sum=1.0,
            cost_sum=3.0,
        )
        second = replace(
            node(
                MemoryLevel.M7,
                MemoryType.STRATEGY,
                (2, outcome.hi, outcome.lo, 9),
                cognitive_state=int(CognitiveState.ACTIVE),
            ),
            attempt_weight=2.0,
            success_sum=2.0,
            cost_sum=2.0,
        )
        self.assertEqual(infer_developmental_stage((first,)), 6)
        self.assertEqual(infer_developmental_stage((first, second)), 7)

    def test_published_correspondence_remains_schedulable_after_restart(self) -> None:
        supervisor = object.__new__(DevelopmentalPeerSupervisor)
        supervisor._live_transfer_candidates = {}
        supervisor._live_transfer_signal_revision = 0
        supervisor._live_transfer_consumed_revision = 0
        candidate = TransferCandidate(
            MemoryUid(1, 2),
            2,
            0.9,
            (11,),
            MemoryUid(3, 4),
            (22,),
            15,
        )
        self.assertTrue(supervisor._register_live_transfer_candidate(candidate))
        self.assertTrue(supervisor.consume_live_transfer_signal())
        self.assertFalse(supervisor._register_live_transfer_candidate(candidate))
        self.assertFalse(supervisor.consume_live_transfer_signal())

        # Persisted freshness state does not make a valid correspondence stale:
        # a restarted supervisor has an empty live queue and schedules it again.
        supervisor._live_transfer_candidates = {}
        self.assertTrue(supervisor._register_live_transfer_candidate(candidate))
        self.assertTrue(supervisor.consume_live_transfer_signal())


class DemonstratedBehaviorEvidenceTests(unittest.TestCase):
    def test_replanning_observed_requires_the_current_trial_to_substitute(self) -> None:
        appended = []
        outcome = MemoryUid(30, 31)
        primary = MemoryUid(32, 33)
        alternative = MemoryUid(34, 35)

        class Peers:
            ledger = SimpleNamespace(append=appended.append)

            def record_replanning_trial(self, **kwargs):
                return SimpleNamespace(
                    valid_recovery=bool(kwargs["recovery_succeeded"])
                )

            def _append_evidence(self, *_args, **_kwargs):
                return None

        runtime = SimpleNamespace(
            peers=Peers(),
            read_view=SimpleNamespace(node_records=lambda **_kwargs: ()),
            watermark=20,
            generation=2,
        )

        def result(succeeded: bool):
            trial = SimpleNamespace(
                primary_strategy_uid=primary,
                alternative_strategy_uid=alternative,
                outcome_uid=outcome,
                recovery_succeeded=succeeded,
            )
            return SimpleNamespace(
                actor_id=1,
                game_id="g",
                strategy_stats=(),
                preference_probes=(),
                replanning_trials=(trial,),
            )

        ContinuousMemoryRuntime.record_actor_results(runtime, (result(False),))
        self.assertEqual(appended, [])
        ContinuousMemoryRuntime.record_actor_results(runtime, (result(True),))
        self.assertEqual([row.evidence_kind for row in appended], ["replanning_observed"])

    def test_preference_evidence_requires_repeated_clean_comparative_choices(self) -> None:
        first = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3))
        second = node(MemoryLevel.M6, MemoryType.OUTCOME, (4, 5, 6))
        emitted = []
        ledger_keys = set()

        class Ledger:
            def contains(self, key):
                return key in ledger_keys

            def append(self, row):
                ledger_keys.add(row.evidence_id)

        supervisor = SimpleNamespace(
            preference=PreferenceEstimator(support_threshold=6, stable_margin=0.3),
            read_view=SimpleNamespace(
                node_records=lambda **_kwargs: (first, second),
                source_games=lambda _row: frozenset((1,)),
            ),
            ledger=Ledger(),
            _append_evidence=lambda kind, *_args, **_kwargs: emitted.append(kind),
            _submit=lambda _proposal: None,
            _existing_proposal=lambda row, **_kwargs: row,
            current_watermark=lambda: 10,
            current_generation=lambda: 1,
            _v845_state_lock=threading.RLock(),
        )
        for _ in range(5):
            self.assertTrue(
                DevelopmentalPeerSupervisor.record_preference_probe(
                    supervisor,
                    outcome_a=first.uid,
                    outcome_b=second.uid,
                    context_bucket=7,
                    chosen_outcome=first.uid,
                    both_reachable=True,
                    preference_influenced=False,
                )
            )
        self.assertNotIn("preference_probe", emitted)
        DevelopmentalPeerSupervisor.record_preference_probe(
            supervisor,
            outcome_a=first.uid,
            outcome_b=second.uid,
            context_bucket=7,
            chosen_outcome=first.uid,
            both_reachable=True,
            preference_influenced=False,
        )
        self.assertEqual(emitted.count("preference_probe"), 1)


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
    def test_probationary_grounded_strategies_are_visible_as_alternatives(self) -> None:
        outcome = MemoryUid(40, 41)
        row = node(
            MemoryLevel.M7,
            MemoryType.STRATEGY,
            (1, outcome.hi, outcome.lo, 99),
            support=3,
            cognitive_state=int(CognitiveState.PROBATION),
        )
        evidence = StrategyEstimator().evaluate((row,))
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].outcome_uid, outcome)

    def test_transfer_freshness_is_scoped_to_correspondence_target(self) -> None:
        source = MemoryUid(1, 2)
        first = TransferCandidate(
            source,
            2,
            0.8,
            (10,),
            MemoryUid(3, 4),
            (20,),
        )
        second = TransferCandidate(
            source,
            2,
            0.8,
            (10,),
            MemoryUid(5, 6),
            (30,),
        )
        self.assertNotEqual(
            _transfer_freshness_kind(first),
            _transfer_freshness_kind(second),
        )

    def test_transfer_structural_candidate_is_not_empirical_validation(self) -> None:
        row = node(MemoryLevel.M4, MemoryType.CONCEPT, (7, 1), support=5, game_mask=3)
        validator = TransferValidator(effect_threshold=0.1)
        self.assertEqual(len(validator.candidates((row,))), 1)
        self.assertFalse(validator.empirically_validated(row.uid))
        validator.record_trial(row.uid, target_game_hash=99, metric_on=0.8, metric_off=0.5)
        self.assertTrue(validator.empirically_validated(row.uid))

    def test_replanning_preserves_outcome_identity(self) -> None:
        outcome = MemoryUid(11, 22)
        planner = Planner()
        from v8.model import stable_u64
        bucket = stable_u64(7, person=b"v8-context")
        s1 = StrategyEvidence(MemoryUid(1, 1), outcome, 1, bucket, 8, 1.0, 1.0)
        s2 = StrategyEvidence(MemoryUid(2, 2), outcome, 2, bucket, 5, 1.0, 1.0)
        current = planner.select(context_signature=7, available_actions=(1, 2), strategies=(s1, s2))
        self.assertIsNotNone(current)
        replanned = planner.replan(current, context_signature=7, available_actions=(1, 2), strategies=(s1, s2))
        self.assertIsNotNone(replanned)
        self.assertEqual(replanned.outcome_uid, current.outcome_uid)
        self.assertNotEqual(replanned.strategy_uid, current.strategy_uid)

    def test_h14_trial_requires_explicit_invalidation_and_recovery(self) -> None:
        outcome = MemoryUid(11, 22)
        controller = ReplanningController()
        invalid = controller.record_trial(
            primary_strategy_uid=MemoryUid(1, 1),
            alternative_strategy_uid=MemoryUid(2, 2),
            outcome_uid=outcome,
            primary_invalidated=False,
            alternative_selected=True,
            outcome_preserved=True,
            recovery_succeeded=True,
        )
        self.assertFalse(invalid.valid_recovery)
        valid = controller.record_trial(
            primary_strategy_uid=MemoryUid(1, 1),
            alternative_strategy_uid=MemoryUid(2, 2),
            outcome_uid=outcome,
            primary_invalidated=True,
            alternative_selected=True,
            outcome_preserved=True,
            recovery_succeeded=True,
        )
        self.assertTrue(valid.valid_recovery)

    def test_h14_ablation_excludes_primary_and_preserves_outcome(self) -> None:
        outcome = MemoryUid(30, 31)
        primary = PlanSelection(outcome, MemoryUid(32, 33), 1, 1.0)
        alternative = PlanSelection(outcome, MemoryUid(34, 35), 2, 0.8)

        class View:
            def __init__(self):
                self.kwargs = None

            def plan_candidates(self, _context, _actions, **kwargs):
                self.kwargs = kwargs
                return (alternative,)

        view = View()
        selected = _select_replanning_ablation(view, primary, 7, (1, 2))
        self.assertIs(selected, alternative)
        self.assertEqual(view.kwargs["outcome_uid"], outcome)
        self.assertEqual(
            view.kwargs["excluded_strategies"],
            frozenset({primary.strategy_uid}),
        )
        self.assertTrue(view.kwargs["ignore_preference"])

    def test_preference_ignores_preference_influenced_choices(self) -> None:
        a, b = MemoryUid(10, 10), MemoryUid(20, 20)
        estimator = PreferenceEstimator(support_threshold=3, stable_margin=0.3)
        for _ in range(10):
            self.assertFalse(
                estimator.record_probe(
                    outcome_a=a,
                    outcome_b=b,
                    context_bucket=77,
                    chosen_outcome=a,
                    both_reachable=True,
                    preference_influenced=True,
                )
            )
        self.assertEqual(estimator.evaluate(), ())

    def test_clean_preference_probes_can_become_stable(self) -> None:
        a, b = MemoryUid(10, 10), MemoryUid(20, 20)
        estimator = PreferenceEstimator(support_threshold=6, stable_margin=0.3)
        for chosen in (a, a, a, a, a, b):
            self.assertTrue(
                estimator.record_probe(
                    outcome_a=a,
                    outcome_b=b,
                    context_bucket=77,
                    chosen_outcome=chosen,
                    both_reachable=True,
                    preference_influenced=False,
                )
            )
        evidence = estimator.evaluate()
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].state, "STABLE")
        self.assertEqual(evidence[0].preferred, a)
        self.assertEqual(evidence[0].clean_probe_count, 6)


class LifecycleAndReportingTests(unittest.TestCase):
    def _h13_evidence(self) -> EvidenceRecord:
        return EvidenceRecord.for_uid(
            "h13-merge",
            MemoryUid(9, 9),
            evidence_kind="outcome_merge",
            watermark=9,
            raw_value=1.0,
            normalized_value=1.0,
            developmental_stage=6,
            validation_state=int(ValidationState.STRUCTURAL),
        )

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

    def test_h14_observed_switch_does_not_validate_without_recovery_trial(self) -> None:
        evaluator = ScientificHypothesisEvaluator()
        observed = EvidenceRecord.for_uid(
            "h14-observed",
            MemoryUid(1, 2),
            evidence_kind="replanning_observed",
            watermark=10,
            raw_value=1.0,
            normalized_value=1.0,
            developmental_stage=7,
            validation_state=int(ValidationState.STRUCTURAL),
        )
        statuses = evaluator.status_map(evaluator.evaluate((observed, self._h13_evidence())))
        self.assertEqual(statuses["H14"], "PARTIALLY_VALID")

    def test_h14_explicit_recovery_trial_can_validate(self) -> None:
        evaluator = ScientificHypothesisEvaluator()
        trial = EvidenceRecord.for_uid(
            "h14-trial",
            MemoryUid(1, 2),
            evidence_kind="replanning_recovery_trial",
            watermark=10,
            raw_value=1.0,
            normalized_value=1.0,
            developmental_stage=7,
            validation_state=int(ValidationState.VALIDATED),
            causal_intervention="strategy_ablation_recovery",
            effect_direction=1,
        )
        statuses = evaluator.status_map(evaluator.evaluate((self._h13_evidence(), trial)))
        self.assertEqual(statuses["H14"], "VALID")

    def test_h15_requires_stable_clean_probe(self) -> None:
        evaluator = ScientificHypothesisEvaluator()
        probe = EvidenceRecord.for_uid(
            "h15-probe",
            MemoryUid(1, 2),
            evidence_kind="preference_probe",
            watermark=10,
            raw_value=1.0,
            normalized_value=1.0,
            developmental_stage=6,
            validation_state=int(ValidationState.STRUCTURAL),
            causal_intervention="clean_choice_probe",
        )
        statuses = evaluator.status_map(evaluator.evaluate((self._h13_evidence(), probe)))
        self.assertEqual(statuses["H15"], "PARTIALLY_VALID")
        stable = EvidenceRecord.for_uid(
            "h15-stable",
            MemoryUid(1, 2),
            evidence_kind="stable_preference_probe",
            watermark=11,
            raw_value=1.0,
            normalized_value=1.0,
            developmental_stage=6,
            validation_state=int(ValidationState.VALIDATED),
            causal_intervention="clean_choice_probe",
            effect_direction=1,
        )
        statuses = evaluator.status_map(evaluator.evaluate((self._h13_evidence(), probe, stable)))
        self.assertEqual(statuses["H15"], "VALID")


if __name__ == "__main__":
    unittest.main()
