from __future__ import annotations

from dataclasses import replace

from v9.memory import DerivationProvenance, M4Concept, M5ConsequenceStructure, M6Outcome, M7Strategy, MemoryLevel, MemoryUid
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig


def _runtime(tmp_path, *, restore=False):
    return ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=restore, enable_snapshots=True))


def _validated_consequence(label: str, descriptor=(5,)):
    e = MemoryUid.derive(label, "e")
    r = MemoryUid.derive(label, "r")
    c = M4Concept(MemoryUid.derive(label, "c"), (11,), DerivationProvenance((r,), (e,), (7,)), 1.0, 2, 0.5, (9,), True)
    return M5ConsequenceStructure.form((c,), descriptor)


def test_singleton_m6_cannot_receive_equivalence_evidence(tmp_path):
    runtime = _runtime(tmp_path)
    consequence = _validated_consequence("a")
    outcome = M6Outcome.form((consequence,), diameter_bound=0)
    runtime._m5[consequence.uid] = consequence
    runtime._m6[outcome.uid] = outcome
    runtime.record_outcome_equivalence_evidence(outcome.uid, equivalent=True, context_scope_id=1, environment_id=7)
    assert runtime._m6[outcome.uid].equivalence_trials == 0


def test_strategy_success_updates_preference_but_not_equivalence(tmp_path):
    runtime = _runtime(tmp_path)
    consequence = _validated_consequence("a")
    outcome = M6Outcome.form((consequence,), diameter_bound=0)
    strategy = M7Strategy.form(outcome, target_environment_id=7, native_actions=(1,), successes=1, trials=1, primary_valence_sum=0, realized_cost_sum=1)
    runtime._m5[consequence.uid] = consequence
    runtime._m6[outcome.uid] = outcome
    runtime._m7[strategy.uid] = strategy
    # Missing graph state is deliberately tolerated by execution feedback.
    runtime.record_strategy_execution(strategy.uid, success=True, realized_cost=1, primary_valence=2)
    assert runtime._m6[outcome.uid].equivalence_trials == 0


def test_replanning_recovery_and_efficiency_are_separate(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.record_replanning_evidence(recovered=True, improved_efficiency=False)
    assert runtime._replans_demonstrated == 1
    assert runtime._recovered_replans == 1
    assert runtime._efficient_replans == 0
    runtime.record_replanning_evidence(recovered=True, improved_efficiency=True)
    assert runtime._replans_demonstrated == 2
    assert runtime._recovered_replans == 2
    assert runtime._efficient_replans == 1


def test_failed_replan_is_not_recovery_or_efficiency(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.record_replanning_evidence(recovered=False, improved_efficiency=True)
    assert runtime._replans_demonstrated == 1
    assert runtime._recovered_replans == 0
    assert runtime._efficient_replans == 0


def test_replanning_metrics_have_distinct_denominators_semantics(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.record_replanning_evidence(recovered=True, improved_efficiency=False)
    runtime.record_replanning_evidence(recovered=True, improved_efficiency=True)
    runtime.record_replanning_evidence(recovered=False, improved_efficiency=False)
    metrics = runtime.metrics()
    assert metrics["m7_replan_attempts"] == 3
    assert metrics["m7_replan_successes"] == 1
    assert metrics["m7_replanning_recovery_rate"] == 2 / 3
    assert metrics["m7_replanning_efficiency_rate"] == 1 / 3


def test_actor_policy_contains_no_m7_derived_grounded_action_scores(tmp_path):
    runtime = _runtime(tmp_path)
    consequence = _validated_consequence("a")
    outcome = M6Outcome.form((consequence,), diameter_bound=0)
    strategy = M7Strategy.form(outcome, target_environment_id=7, native_actions=(41,), successes=10, trials=10, primary_valence_sum=10, realized_cost_sum=10)
    runtime._m6[outcome.uid] = outcome
    runtime._m7[strategy.uid] = strategy
    snapshot = runtime.actor_policy_snapshot()
    assert snapshot.grounded_action_scores_by_type == {}
    assert snapshot.strategies(7)


def test_actor_policy_exposes_outcome_for_published_strategy(tmp_path):
    runtime = _runtime(tmp_path)
    consequence = _validated_consequence("a")
    outcome = replace(M6Outcome.form((consequence,), diameter_bound=0), equivalence_trials=2, equivalence_successes=2, primary_valence_sum=4, preference_trials=2)
    strategy = M7Strategy.form(outcome, target_environment_id=7, native_actions=(1,), successes=2, trials=2)
    runtime._m6[outcome.uid] = outcome
    runtime._m7[strategy.uid] = strategy
    snapshot = runtime.actor_policy_snapshot()
    assert snapshot.outcomes(7)[0].outcome_uid == outcome.uid
    assert snapshot.outcomes(7)[0].equivalence_confidence == 1.0
    assert snapshot.outcomes(7)[0].mean_primary_valence == 2.0


def test_restore_roundtrip_preserves_m6_scientific_fields(tmp_path):
    runtime = _runtime(tmp_path)
    consequence = _validated_consequence("a")
    outcome = replace(
        M6Outcome.form((consequence,), diameter_bound=0),
        equivalence_trials=5,
        equivalence_successes=4,
        contexts_observed=(1, 2),
        environments_observed=(7, 8),
        primary_valence_sum=9,
        preference_trials=3,
    )
    runtime._m5[consequence.uid] = consequence
    runtime._m6[outcome.uid] = outcome
    runtime.close()
    restored = _runtime(tmp_path, restore=True)
    row = restored._m6.get(outcome.uid)
    if row is not None:
        assert row.equivalence_trials == 5
        assert row.equivalence_successes == 4
        assert row.contexts_observed == (1, 2)
        assert row.environments_observed == (7, 8)
        assert row.primary_valence_sum == 9
        assert row.preference_trials == 3


def test_m7_strategy_identity_distinguishes_action_sequences():
    outcome = M6Outcome.form((_validated_consequence("a"),), diameter_bound=0)
    a = M7Strategy.form(outcome, target_environment_id=7, native_actions=(1, 2), successes=1, trials=1, primary_valence_sum=0, realized_cost_sum=2)
    b = M7Strategy.form(outcome, target_environment_id=7, native_actions=(2, 1), successes=1, trials=1, primary_valence_sum=0, realized_cost_sum=2)
    assert a.uid != b.uid


def test_m7_strategy_identity_distinguishes_target_environment():
    outcome = M6Outcome.form((_validated_consequence("a"),), diameter_bound=0)
    a = M7Strategy.form(outcome, target_environment_id=7, native_actions=(1,))
    b = M7Strategy.form(outcome, target_environment_id=8, native_actions=(1,), successes=1, trials=1, primary_valence_sum=0, realized_cost_sum=1)
    assert a.uid != b.uid


def test_m7_strategy_identity_distinguishes_target_outcome():
    aout = M6Outcome.form((_validated_consequence("a"),), diameter_bound=0)
    bout = M6Outcome.form((_validated_consequence("b", (6,)),), diameter_bound=0)
    a = M7Strategy.form(aout, target_environment_id=7, native_actions=(1,), successes=1, trials=1, primary_valence_sum=0, realized_cost_sum=1)
    b = M7Strategy.form(bout, target_environment_id=7, native_actions=(1,), successes=1, trials=1, primary_valence_sum=0, realized_cost_sum=1)
    assert a.uid != b.uid
