from __future__ import annotations

import pickle
from dataclasses import replace

from v9.cognition.outcome_selection import select_target_outcome, target_outcome_stability
from v9.cognition.planning import choose_strategy, replan
from v9.memory import DerivationProvenance, M4Concept, M5ConsequenceStructure, M6Outcome, M7Strategy, MemoryUid
from v9.runtime.actor_policy import ActorOutcomePolicy, ActorPolicySnapshot, ActorStrategyPolicy


def _uid(label: str, n: int = 1) -> MemoryUid:
    return MemoryUid.derive(label, n)


def _consequence(label: str, descriptor: tuple[int, ...] = (5,)) -> M5ConsequenceStructure:
    evidence = _uid(label + "-e")
    role = _uid(label + "-r")
    concept = M4Concept(_uid(label + "-c"), (11,), DerivationProvenance((role,), (evidence,), (7,)), 1.0, 2, 0.5, (9,), True)
    return M5ConsequenceStructure.form((concept,), descriptor)


def _outcome(*consequences: M5ConsequenceStructure) -> M6Outcome:
    return M6Outcome.form(tuple(consequences), diameter_bound=0)


def test_m6_starts_without_equivalence_evidence() -> None:
    outcome = _outcome(_consequence("a"))
    assert outcome.equivalence_trials == 0
    assert outcome.equivalence_successes == 0
    assert outcome.equivalence_confidence == 0.0


def test_m6_can_represent_multiple_distinct_m5_consequences() -> None:
    a, b = _consequence("a"), _consequence("b")
    outcome = _outcome(a, b)
    assert set(outcome.members) == {a.uid, b.uid}
    assert len(outcome.members) == 2


def test_m6_mean_valence_requires_preference_trials() -> None:
    outcome = replace(_outcome(_consequence("a")), primary_valence_sum=99, preference_trials=0)
    assert outcome.mean_primary_valence == 0.0


def test_m6_mean_valence_is_empirical_average() -> None:
    outcome = replace(_outcome(_consequence("a")), primary_valence_sum=6, preference_trials=3)
    assert outcome.mean_primary_valence == 2.0


def test_m6_equivalence_confidence_is_empirical_ratio() -> None:
    outcome = replace(_outcome(_consequence("a"), _consequence("b")), equivalence_trials=4, equivalence_successes=3)
    assert outcome.equivalence_confidence == 0.75


def test_outcome_selector_accepts_memory_outcomes() -> None:
    low = replace(_outcome(_consequence("a")), primary_valence_sum=1, preference_trials=1)
    high = replace(_outcome(_consequence("b")), primary_valence_sum=3, preference_trials=1)
    assert select_target_outcome((low, high)).uid == high.uid


def test_outcome_selector_accepts_actor_outcome_policy() -> None:
    low = ActorOutcomePolicy(_uid("low"), 7, 1.0, 1.0)
    high = ActorOutcomePolicy(_uid("high"), 7, 1.0, 2.0)
    assert select_target_outcome((low, high)).outcome_uid == high.outcome_uid


def test_outcome_selector_prefers_valence_before_equivalence() -> None:
    a = ActorOutcomePolicy(_uid("a"), 7, 1.0, 1.0)
    b = ActorOutcomePolicy(_uid("b"), 7, 0.1, 2.0)
    assert select_target_outcome((a, b)) == b


def test_outcome_selector_uses_equivalence_as_secondary_rank() -> None:
    a = ActorOutcomePolicy(_uid("a"), 7, 0.4, 1.0)
    b = ActorOutcomePolicy(_uid("b"), 7, 0.9, 1.0)
    assert select_target_outcome((a, b)) == b


def test_outcome_selector_honors_reachable_set() -> None:
    a = ActorOutcomePolicy(_uid("a"), 7, 0.4, 1.0)
    b = ActorOutcomePolicy(_uid("b"), 7, 0.9, 9.0)
    assert select_target_outcome((a, b), reachable={a.outcome_uid}) == a


def test_target_outcome_stability_empty_is_zero() -> None:
    assert target_outcome_stability(()) == 0.0


def test_target_outcome_stability_measures_modal_fraction() -> None:
    a, b = _uid("a"), _uid("b")
    assert target_outcome_stability((a, a, b, a)) == 0.75


def _strategy(uid_label: str, outcome: MemoryUid, actions: tuple[int, ...], reliability: float, valence: float, efficiency: float | None) -> ActorStrategyPolicy:
    trials = 100
    successes = int(reliability * trials)
    return ActorStrategyPolicy(_uid(uid_label), outcome, 7, actions, reliability, None, efficiency, valence)


def test_planner_requires_matching_environment() -> None:
    s = replace(_strategy("s", _uid("o"), (1,), 1.0, 1.0, 1.0), environment_id=8)
    assert choose_strategy((s,), target_environment_id=7) is None


def test_planner_requires_nonempty_actions() -> None:
    s = _strategy("s", _uid("o"), (), 1.0, 1.0, 1.0)
    assert choose_strategy((s,), target_environment_id=7) is None


def test_planner_requires_positive_reliability() -> None:
    s = _strategy("s", _uid("o"), (1,), 0.0, 1.0, 1.0)
    assert choose_strategy((s,), target_environment_id=7) is None


def test_planner_respects_available_first_action() -> None:
    bad = _strategy("bad", _uid("o"), (2,), 1.0, 5.0, 1.0)
    good = _strategy("good", _uid("o"), (1,), 0.5, 1.0, 1.0)
    assert choose_strategy((bad, good), target_environment_id=7, available_actions=(1,)) == good


def test_planner_ranks_reliability_first() -> None:
    o = _uid("o")
    reliable = _strategy("r", o, (1,), 0.9, 0.0, 0.1)
    valuable = _strategy("v", o, (2,), 0.8, 100.0, 1.0)
    assert choose_strategy((valuable, reliable), target_environment_id=7) == reliable


def test_planner_ranks_valence_second() -> None:
    o = _uid("o")
    a = _strategy("a", o, (1,), 0.9, 1.0, 1.0)
    b = _strategy("b", o, (2,), 0.9, 2.0, 0.1)
    assert choose_strategy((a, b), target_environment_id=7) == b


def test_planner_ranks_efficiency_third() -> None:
    o = _uid("o")
    a = _strategy("a", o, (1,), 0.9, 1.0, 0.4)
    b = _strategy("b", o, (2,), 0.9, 1.0, 0.8)
    assert choose_strategy((a, b), target_environment_id=7) == b


def test_replan_preserves_target_outcome() -> None:
    o1, o2 = _uid("o1"), _uid("o2")
    current = _strategy("cur", o1, (1,), 1.0, 1.0, 1.0)
    same = _strategy("same", o1, (2,), 0.8, 1.0, 1.0)
    other = _strategy("other", o2, (3,), 1.0, 9.0, 1.0)
    assert replan(current, (same, other), target_environment_id=7) == same


def test_replan_excludes_current_strategy() -> None:
    current = _strategy("cur", _uid("o"), (1,), 1.0, 1.0, 1.0)
    assert replan(current, (current,), target_environment_id=7) is None


def test_m7_observe_updates_trials_and_successes() -> None:
    outcome = _outcome(_consequence("a"))
    strategy = M7Strategy.form(outcome, target_environment_id=7, native_actions=(1, 2), successes=1, trials=2, primary_valence_sum=0, realized_cost_sum=0)
    updated = strategy.observe(success=True, realized_cost=2)
    assert updated.reliability_trials == 3
    assert updated.reliability_successes == 2


def test_m7_failed_observation_does_not_add_realized_success_cost() -> None:
    outcome = _outcome(_consequence("a"))
    strategy = M7Strategy.form(outcome, target_environment_id=7, native_actions=(1,), successes=0, trials=1, primary_valence_sum=0, realized_cost_sum=0)
    updated = strategy.observe(success=False, realized_cost=9)
    assert updated.realized_cost_sum == 0
    assert updated.reliability_trials == 1


def test_m7_success_observation_adds_realized_cost() -> None:
    outcome = _outcome(_consequence("a"))
    strategy = M7Strategy.form(outcome, target_environment_id=7, native_actions=(1,), successes=0, trials=1, primary_valence_sum=0, realized_cost_sum=0)
    updated = strategy.observe(success=True, realized_cost=3)
    assert updated.realized_cost_sum == 3


def test_actor_policy_snapshot_pickles_outcomes_and_strategies() -> None:
    outcome = ActorOutcomePolicy(_uid("o"), 7, 0.8, 2.0)
    strategy = _strategy("s", outcome.outcome_uid, (1, 2), 0.9, 2.0, 1.0)
    snapshot = ActorPolicySnapshot.build(generation=4, outcomes_by_environment={7: (outcome,)}, strategies_by_environment={7: (strategy,)})
    restored = pickle.loads(pickle.dumps(snapshot))
    assert restored.outcomes(7) == (outcome,)
    assert restored.strategies(7) == (strategy,)


def test_actor_policy_empty_environment_returns_empty_outcomes() -> None:
    assert ActorPolicySnapshot.build(generation=0, normalized_action_supports={}, hgt_action_scores={}, model_version='test').outcomes(99) == ()


def test_actor_policy_empty_environment_returns_empty_strategies() -> None:
    assert ActorPolicySnapshot.build(generation=0, normalized_action_supports={}, hgt_action_scores={}, model_version='test').strategies(99) == ()
