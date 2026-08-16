from __future__ import annotations

import unittest
from random import Random
from types import SimpleNamespace

from v8.arena import EdgeRecord, NodeRecord
from v8.behavior_recovery import (
    CausalEvidenceGatedPromotionEngine,
    _plan_candidates,
    canonical_outcome_key,
    observed_outcome_uids,
    strategy_can_control,
)
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    stable_u64,
)
from v8.publication import _StrategyRow


def node(
    level: MemoryLevel,
    memory_type: MemoryType,
    key: tuple[int, ...],
    *,
    support: int = 4,
    future: float = 1.0,
    cognitive_state: int = int(CognitiveState.ACTIVE),
    success_sum: float = 0.0,
    cost_sum: float = 0.0,
    attempt_weight: float = 0.0,
) -> NodeRecord:
    return NodeRecord(
        uid=MemoryUid.from_key(level, memory_type, key),
        fingerprint=1,
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
        support_count=int(support),
        significance_sum=2.0,
        prediction_error_sum=0.0,
        learning_value_sum=2.0,
        transfer_prior_sum=0.0,
        explanatory_sum=2.0,
        future_option_sum=float(future) * max(1, support),
        score_weight=float(max(1, support)),
        updated_watermark=10,
        game_mask=1,
        cognitive_state=int(cognitive_state),
        validation_state=int(ValidationState.STRUCTURAL),
        success_sum=float(success_sum),
        cost_sum=float(cost_sum),
        attempt_weight=float(attempt_weight),
    )


def edge(source: MemoryUid, relation: RelationType, target: MemoryUid) -> EdgeRecord:
    return EdgeRecord(source, int(relation), target, 1, 10)


class CanonicalOutcomeTests(unittest.TestCase):
    def test_m6_descriptor_has_one_terminal_free_authority(self) -> None:
        consequence = node(
            MemoryLevel.M5,
            MemoryType.CONSEQUENCE,
            (101, 202, 303, 1),
        )
        expected_variant = stable_u64(101, 202, person=b"v8.5-outcome-context")
        self.assertEqual(canonical_outcome_key(consequence), (1, 303, expected_variant))

    def test_actor_observation_resolves_existing_canonical_m6_uid(self) -> None:
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 303, 4))
        fake = SimpleNamespace(
            _refresh_strategy_cache=lambda: None,
            _behavior_observed_outcomes={(10, 2, 99): {outcome}},
        )
        self.assertEqual(
            observed_outcome_uids(
                fake,
                context_signature=10,
                action_id=2,
                outcome_signature=99,
            ),
            (outcome,),
        )


class CausalStrategyFormationTests(unittest.TestCase):
    def test_m7_is_formed_only_from_m1_in_outcome_lineage(self) -> None:
        linked = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 2, 77, 11), support=5)
        unrelated = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (20, 3, 88, 21), support=5)
        outcome = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 900, 1), support=5)
        rows = (linked, unrelated, outcome)
        graph = (edge(outcome.uid, RelationType.EXPLAINS, linked.uid),)

        candidates = CausalEvidenceGatedPromotionEngine().propose(rows, graph, budget=64)
        strategies = [item for item in candidates if item.level == MemoryLevel.M7]

        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0].parents, (outcome.uid, linked.uid))
        self.assertEqual(strategies[0].key_parts[0], 2)
        self.assertFalse(any(item.key_parts[0] == 3 for item in strategies))


class PlannerAdmissionTests(unittest.TestCase):
    def _fixture(self, *, causal: bool = True):
        outcome = node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 900, 1), support=8)
        m1 = node(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 2, 77, 11), support=8)
        context_bucket = stable_u64(10, person=b"v8-context")
        strategy = node(
            MemoryLevel.M7,
            MemoryType.STRATEGY,
            (2, outcome.uid.hi, outcome.uid.lo, context_bucket),
            support=8,
            success_sum=2.0,
            cost_sum=3.0,
            attempt_weight=3.0,
        )
        dependency = m1.uid if causal else MemoryUid(999, 999)
        view = SimpleNamespace(
            _node_by_uid={strategy.uid: strategy, outcome.uid: outcome, m1.uid: m1},
            _behavior_strategy_dependencies={strategy.uid: {dependency}},
            _behavior_m1_by_outcome={outcome.uid: {m1.uid}},
        )
        strategy_row = _StrategyRow(
            2,
            outcome.uid,
            strategy.uid,
            8,
            strategy.strategy_reliability,
            strategy.strategy_mean_cost,
            context_bucket,
            False,
            False,
        )
        return view, strategy, strategy_row, outcome

    def test_false_legacy_strategy_cannot_control_actor(self) -> None:
        view, strategy, _strategy_row, outcome = self._fixture(causal=False)
        self.assertFalse(strategy_can_control(view, strategy.uid, outcome.uid))

    def test_causal_empirically_reliable_strategy_can_control_actor(self) -> None:
        view, strategy, _strategy_row, outcome = self._fixture(causal=True)
        self.assertTrue(strategy_can_control(view, strategy.uid, outcome.uid))

    def test_epsilon_exploration_still_overrides_admitted_plan(self) -> None:
        view, strategy, strategy_row, outcome = self._fixture(causal=True)
        context_bucket = strategy_row.context_bucket
        view._refresh_strategy_cache = lambda: None
        view._strategy_by_context = {context_bucket: [strategy_row]}
        view._strategy_fallback = []
        view._preferred_outcomes = set()
        view._behavior_actor_mode = True
        view._behavior_epsilon = 1.0
        view._behavior_rng = Random(7)
        view._behavior_force_random = False
        view._behavior_last_plans = ()
        view.strategy_has_ancestor = lambda *_args, **_kwargs: True

        plans = _plan_candidates(view, 10, (2, 3))

        self.assertEqual(plans, ())
        self.assertTrue(view._behavior_force_random)

    def test_without_exploration_admitted_plan_is_used(self) -> None:
        view, strategy, strategy_row, outcome = self._fixture(causal=True)
        context_bucket = strategy_row.context_bucket
        view._refresh_strategy_cache = lambda: None
        view._strategy_by_context = {context_bucket: [strategy_row]}
        view._strategy_fallback = []
        view._preferred_outcomes = set()
        view._behavior_actor_mode = True
        view._behavior_epsilon = 0.0
        view._behavior_rng = Random(7)
        view._behavior_force_random = False
        view._behavior_last_plans = ()
        view.strategy_has_ancestor = lambda *_args, **_kwargs: True

        plans = _plan_candidates(view, 10, (2, 3))

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].strategy_uid, strategy.uid)
        self.assertEqual(plans[0].outcome_uid, outcome.uid)


if __name__ == "__main__":
    unittest.main()
