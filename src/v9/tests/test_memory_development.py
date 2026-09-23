from __future__ import annotations

from pathlib import Path

import pytest

from v9.cognition.strategies import relative_efficiency
from v9.memory import (
    DerivationProvenance, M4Concept, M5ConsequenceStructure, M6Outcome, M7Strategy,
    MemoryLevel, MemoryUid,
)
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.memory_pipeline import DerivationTask, derive_memory


def _runtime(tmp_path: Path) -> ContinuousMemoryRuntime:
    return ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))


def test_raw_experience_stops_at_m1_before_recurrence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    event = runtime.make_experience(producer_id=1, producer_sequence=1, environment_instance_id=7, global_step=0, context_signature=3, action_id=2, outcome_signature=4, family_signature=5)
    runtime.submit(event)
    counts = runtime.metrics()["memory_levels"]
    assert counts["M0"] == 1
    assert counts["M1"] == 2
    assert all(counts[f"M{level}"] == 0 for level in range(2, 8))


def test_recurrence_forms_m2_m3_and_candidate_m4(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for index in range(2):
        runtime.submit(runtime.make_experience(producer_id=1, producer_sequence=index + 1, environment_instance_id=7, global_step=index, context_signature=3, action_id=2, outcome_signature=4, family_signature=5))
    counts = runtime.metrics()["memory_levels"]
    assert counts["M2"] == counts["M3"] == counts["M4"] == 1
    assert counts["M5"] == 0


def test_held_out_transfer_unlocks_m5_through_m7(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for index in range(2):
        runtime.submit(runtime.make_experience(producer_id=1, producer_sequence=index + 1, environment_instance_id=7, global_step=index, context_signature=3, action_id=2, outcome_signature=4, family_signature=5))
    concept = next(uid for uid, node in runtime.graph.nodes.items() if node.level is MemoryLevel.M4)
    runtime.record_transfer_validation(concept, target_environment_id=9, target_native_action=41, enabled_metric=1.0, ablated_metric=0.0)
    assert runtime.metrics()["memory_levels"]["M5"] == 0
    runtime.record_transfer_validation(concept, target_environment_id=10, target_native_action=41, enabled_metric=1.0, ablated_metric=0.0)
    counts = runtime.metrics()["memory_levels"]
    assert counts["M5"] == counts["M6"] == counts["M7"] == 1
    strategy_payload = next(runtime.graph.payloads[uid] for uid, node in runtime.graph.nodes.items() if node.level is MemoryLevel.M7)
    assert strategy_payload["native_actions"] == [41]
    assert strategy_payload["target_environment_id"] == 10


def test_m5_maturity_requires_validated_concept() -> None:
    parent = MemoryUid.derive("evidence", 1)
    role = MemoryUid.derive("role", 1)
    concept = M4Concept(MemoryUid.derive("concept", 1), (11,), DerivationProvenance((role,), (parent,), (7,)), 1.0, 2, 0.5)
    consequence = M5ConsequenceStructure.form((concept,), (3,))
    assert not consequence.mature
    assert M5ConsequenceStructure.form((concept.with_validation((9,)),), (3,)).mature


def test_outcome_has_no_terminal_label_and_efficiency_is_same_outcome_only() -> None:
    parent = MemoryUid.derive("evidence", 1)
    concept = M4Concept(MemoryUid.derive("concept", 1), (11,), DerivationProvenance((parent,), (parent,), (7,)), 1.0, 2, 0.5, (9,), True)
    consequence = M5ConsequenceStructure.form((concept,), (5,))
    outcome = M6Outcome.form((consequence,), diameter_bound=0)
    assert not hasattr(outcome, "terminal_label")
    fast = M7Strategy.form(outcome, target_environment_id=9, native_actions=(1,), successes=2, trials=2, primary_valence_sum=2, realized_cost_sum=4)
    slow = M7Strategy.form(outcome, target_environment_id=9, native_actions=(2,), successes=2, trials=2, primary_valence_sum=2, realized_cost_sum=8)
    scores = relative_efficiency((fast, slow))
    assert scores[fast.uid.lo] == 1.0
    assert scores[slow.uid.lo] == 0.5


def test_parallel_derivation_forms_m4_from_recurrent_single_source_scope(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for index in range(2):
        runtime.submit(
            runtime.make_experience(
                producer_id=1,
                producer_sequence=index + 1,
                environment_instance_id=7,
                global_step=index,
                context_signature=3,
                action_id=2,
                outcome_signature=4,
                family_signature=5,
            )
        )
    signature = next(iter(runtime._m1n_occurrences))
    rows = tuple(runtime._m1n_occurrences[signature])
    result = derive_memory(
        DerivationTask(
            task_id=1,
            structural_signature=signature,
            rows=rows,
            support=2,
            formation_scope=(7,),
            causal_watermark=runtime.watermark,
        )
    )
    assert result.roles
    assert result.concepts
    assert result.concepts[0].provenance.formation_scope == (7,)


def test_missing_canonical_m4_is_republished_by_batch_derivation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    for index in range(2):
        runtime.submit(
            runtime.make_experience(
                producer_id=1,
                producer_sequence=index + 1,
                environment_instance_id=7,
                global_step=index,
                context_signature=3,
                action_id=2,
                outcome_signature=4,
                family_signature=5,
            )
        )
    signature = next(iter(runtime._m1n_occurrences))
    rows = tuple(runtime._m1n_occurrences[signature])
    result = derive_memory(
        DerivationTask(
            task_id=1,
            structural_signature=signature,
            rows=rows,
            support=4,
            formation_scope=(7,),
            causal_watermark=runtime.watermark,
        )
    )
    assert result.concepts
    concept = result.concepts[0]
    runtime._m4[concept.uid] = concept
    runtime.graph.nodes.pop(concept.uid, None)
    runtime.graph.payloads.pop(concept.uid, None)
    runtime.graph._uids_by_level[MemoryLevel.M4].discard(concept.uid)

    runtime.apply_derivation_results_batch((result,))

    assert concept.uid in runtime.graph.nodes
    assert runtime.graph.nodes[concept.uid].level is MemoryLevel.M4
