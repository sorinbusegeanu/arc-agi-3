from __future__ import annotations

from types import SimpleNamespace

import pytest

from v9.runtime.developmental_cut import DevelopmentalMutationGate
from v9.runtime.parallel_memory_coordinator import MemoryPipelineService
from v9.runtime.runtime import ContinuousMemoryRuntime
from v9.runtime.scientific_modes import ScientificVisibilityMode


def test_matched_mode_rejects_background_developmental_publication() -> None:
    gate = DevelopmentalMutationGate(ScientificVisibilityMode.MATCHED_REASONING)
    with pytest.raises(RuntimeError, match="outside DevelopmentalCut"):
        gate.assert_publication_allowed()
    with gate.allow_cut("cut-1"):
        gate.assert_publication_allowed()


def test_async_mode_requires_declared_async_origin() -> None:
    gate = DevelopmentalMutationGate(ScientificVisibilityMode.ASYNC_DEVELOPMENT)
    gate.assert_publication_allowed(async_origin=True)


def test_matched_runtime_defers_worker_derivation_until_developmental_cut() -> None:
    runtime = object.__new__(ContinuousMemoryRuntime)
    runtime.config = SimpleNamespace(
        scientific=SimpleNamespace(
            scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING,
            proposal_queue_depth=2,
        )
    )
    runtime.developmental_mutation_gate = DevelopmentalMutationGate(
        ScientificVisibilityMode.MATCHED_REASONING
    )
    runtime._deferred_derivation_results = {}
    applied = []
    gauges = {}
    runtime.apply_derivation_result = applied.append
    runtime.set_telemetry_gauge = gauges.__setitem__
    later = SimpleNamespace(structural_signature=7, support=3)
    newer = SimpleNamespace(structural_signature=7, support=4)
    first = SimpleNamespace(structural_signature=2, support=1)

    runtime.apply_derivation_results_batch((later, first, newer))

    assert applied == []
    assert runtime._deferred_derivation_results == {7: newer, 2: first}
    assert gauges["deferred_developmental_derivations"] == 2
    with runtime.developmental_mutation_gate.allow_cut("cut-1"):
        assert runtime.flush_deferred_derivation_results() == 2
    assert applied == [first, newer]
    assert runtime._deferred_derivation_results == {}


def test_matched_deferred_derivation_state_has_a_hard_signature_ceiling() -> None:
    runtime = object.__new__(ContinuousMemoryRuntime)
    runtime.config = SimpleNamespace(
        scientific=SimpleNamespace(
            scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING,
            proposal_queue_depth=1,
        )
    )
    runtime.developmental_mutation_gate = DevelopmentalMutationGate(
        ScientificVisibilityMode.MATCHED_REASONING
    )
    runtime._deferred_derivation_results = {}
    runtime.set_telemetry_gauge = lambda *_args: None
    runtime.apply_derivation_result = lambda _row: None
    runtime.apply_derivation_results_batch((SimpleNamespace(structural_signature=1),))

    with pytest.raises(OverflowError, match="proposal ceiling"):
        runtime.apply_derivation_results_batch((SimpleNamespace(structural_signature=2),))


def test_matched_pipeline_does_not_dispatch_sampling_time_derivation() -> None:
    service = object.__new__(MemoryPipelineService)
    gauges = {}
    service.runtime = SimpleNamespace(
        config=SimpleNamespace(
            scientific=SimpleNamespace(
                scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING
            )
        ),
        set_telemetry_gauge=gauges.__setitem__,
    )
    service.last_support = {}
    service._derivation_candidates = {}
    candidate = SimpleNamespace(structural_signature=9, support=2)

    MemoryPipelineService._consider_candidate(service, candidate)

    assert service._derivation_candidates == {}
    assert gauges["matched_derivation_candidates_deferred_to_cut"] == 1
