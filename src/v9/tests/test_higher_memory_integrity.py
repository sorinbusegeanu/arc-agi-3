from __future__ import annotations

from types import SimpleNamespace

from v9.memory import (
    DerivationProvenance,
    M4Concept,
    M5ConsequenceStructure,
    M6Outcome,
    M7Strategy,
    MemoryUid,
)
from v9.runtime.higher_memory_integrity import install


def _validated_concept(label: str) -> M4Concept:
    evidence = MemoryUid.derive(label, "evidence")
    role = MemoryUid.derive(label, "role")
    return M4Concept(
        MemoryUid.derive(label, "concept"),
        (11,),
        DerivationProvenance((role,), (evidence,), (7,)),
        1.0,
        2,
        0.5,
        (9,),
        True,
    )


def test_provenance_deduplicates_parents_evidence_and_scope() -> None:
    parent = MemoryUid.derive("parent", 1)
    evidence = MemoryUid.derive("evidence", 1)
    provenance = DerivationProvenance(
        (parent, parent),
        (evidence, evidence, evidence),
        (7, 7, 8, 7),
    )
    assert provenance.parents == (parent,)
    assert provenance.evidence == (evidence,)
    assert provenance.formation_scope == (7, 8)


def test_distinct_validated_concepts_form_distinct_m5_members_of_one_m6_class() -> None:
    first = M5ConsequenceStructure.form((_validated_concept("a"),), (5,))
    second = M5ConsequenceStructure.form((_validated_concept("b"),), (5,))
    assert first.uid != second.uid
    outcome = M6Outcome.form((first, second), diameter_bound=0)
    assert len(outcome.members) == 2
    assert set(outcome.members) == {first.uid, second.uid}


def test_duplicate_m5_input_cannot_fake_m6_equivalence_membership() -> None:
    consequence = M5ConsequenceStructure.form((_validated_concept("a"),), (5,))
    outcome = M6Outcome.form((consequence, consequence), diameter_bound=0)
    assert outcome.members == (consequence.uid,)


def test_reformation_preserves_m6_and_m7_empirical_state() -> None:
    first = M5ConsequenceStructure.form((_validated_concept("a"),), (5,))
    second = M5ConsequenceStructure.form((_validated_concept("b"),), (5,))
    old_outcome = M6Outcome.form((first,), diameter_bound=0)
    old_outcome = __import__("dataclasses").replace(
        old_outcome,
        equivalence_trials=4,
        equivalence_successes=3,
        contexts_observed=(1,),
        environments_observed=(7,),
        primary_valence_sum=6,
        preference_trials=3,
    )
    old_strategy = M7Strategy.form(
        old_outcome,
        target_environment_id=7,
        native_actions=(1,),
        successes=4,
        trials=5,
        primary_valence_sum=8,
        realized_cost_sum=12,
    )
    new_outcome = M6Outcome.form((first, second), diameter_bound=0)
    new_strategy = M7Strategy.form(
        new_outcome,
        target_environment_id=7,
        native_actions=(1,),
        successes=1,
        trials=1,
        primary_valence_sum=1,
        realized_cost_sum=1,
    )

    class FakeRuntime:
        def __init__(self):
            self._m6 = {old_outcome.uid: old_outcome}
            self._m7 = {old_strategy.uid: old_strategy}
            self.graph = SimpleNamespace(
                nodes={old_outcome.uid: object(), old_strategy.uid: object()},
                payloads={old_outcome.uid: {}, old_strategy.uid: {}},
            )
            self.gauges = {}

        def record_transfer_validation(self, concept_uid, **kwargs):
            self._m6[new_outcome.uid] = new_outcome
            self._m7[new_strategy.uid] = new_strategy

        def _publish(self, node, payload, evidence, **kwargs):
            return True

        def set_telemetry_gauge(self, key, value):
            self.gauges[key] = value

    install(FakeRuntime)
    runtime = FakeRuntime()
    runtime.record_transfer_validation(MemoryUid.derive("concept", 9))

    outcome = runtime._m6[old_outcome.uid]
    strategy = runtime._m7[old_strategy.uid]
    assert set(outcome.members) == {first.uid, second.uid}
    assert outcome.equivalence_trials == 4
    assert outcome.equivalence_successes == 3
    assert outcome.primary_valence_sum == 6
    assert outcome.preference_trials == 3
    assert outcome.class_version == 2
    assert strategy.reliability_trials == 5
    assert strategy.reliability_successes == 4
    assert strategy.primary_valence_sum == 8
    assert strategy.realized_cost_sum == 12
    assert runtime.gauges["m6_class_expansions"] == 1
    assert runtime.gauges["m7_state_preservations"] == 1
