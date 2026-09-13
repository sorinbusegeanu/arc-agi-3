from __future__ import annotations

import math

import pytest

from v9.cognition.correspondence import propose_correspondence
from v9.cognition.grounding import GroundingEvidence, GroundingMaturity, GroundingRegistry
from v9.cognition.similarity import (
    NormalizationState, ProgressiveSimilarity, ScaleStatistics, StructuralDescriptor,
    entropy, stable_distribution,
)
from v9.cognition.transfer import TransferTrial, validate_transfer


def _descriptor(uid: int, radius: int, values: tuple[float, ...], estimator: int = 1, generation: int = 1) -> StructuralDescriptor:
    return StructuralDescriptor(uid, generation, 1, radius, 1, estimator, values)


def _engine() -> ProgressiveSimilarity:
    statistics = ScaleStatistics(sample_threshold=2, contingency_threshold=1, reservoir_limit=2)
    return ProgressiveSimilarity(beta_by_radius={1: 1.0, 2: 1.0, 4: 1.0}, candidate_limit=4, equivalence_limit=2, maximum_radius=4, ambiguity_threshold=0.1, margin_threshold=0.01, information_threshold=0.001, symmetry_patience=1, statistics=statistics)


def test_log_softmax_is_stable_for_extreme_scores() -> None:
    values = stable_distribution((10_000.0, 9_999.0), 100.0)
    assert all(math.isfinite(value) for value in values)
    assert abs(sum(values) - 1.0) < 1e-12
    assert entropy(values) >= 0


def test_persistent_symmetry_returns_equivalence_not_arbitrary_winner() -> None:
    engine = _engine()
    query = {1: _descriptor(0, 1, (1.0,)), 2: _descriptor(0, 2, (1.0,))}
    candidates = {1: {1: _descriptor(1, 1, (1.0,)), 2: _descriptor(1, 2, (1.0,))}, 2: {1: _descriptor(2, 1, (1.0,)), 2: _descriptor(2, 2, (1.0,))}}
    outcome = engine.search(query, candidates, compute_budget=20)
    assert outcome.winner_uid is None
    assert outcome.equivalence_set is not None
    assert outcome.equivalence_set.candidate_uids == (1, 2)


def test_scale_statistics_exclude_non_authoritative_evidence() -> None:
    statistics = ScaleStatistics(sample_threshold=1, contingency_threshold=1)
    descriptor = _descriptor(1, 1, (2.0,))
    statistics.observe(descriptor, stable_contingency_uid=1, authoritative_evidence=False)
    assert statistics.state(1) is NormalizationState.EMPTY
    statistics.observe(descriptor, stable_contingency_uid=1)
    assert statistics.state(1) is NormalizationState.PROVISIONAL
    statistics.observe(_descriptor(2, 1, (4.0,), generation=2), stable_contingency_uid=1)
    assert statistics.state(1) is NormalizationState.AUTHORITATIVE


def test_radius_statistics_are_isolated_bounded_and_version_stale() -> None:
    statistics = ScaleStatistics(sample_threshold=1, contingency_threshold=1, reservoir_limit=2)
    for uid in range(10):
        statistics.observe(_descriptor(uid, 1, (float(uid),), generation=uid + 1), stable_contingency_uid=uid)
    statistics.observe(_descriptor(20, 2, (3.0,), generation=1), stable_contingency_uid=20)
    assert len(statistics._nodes[1]) == 2
    assert len(statistics._stable_contingencies[1]) == 2
    assert statistics.state(1) is NormalizationState.AUTHORITATIVE
    assert statistics.state(2) is NormalizationState.PROVISIONAL
    descriptor = _descriptor(1, 1, (1.0,))
    assert ProgressiveSimilarity.stale(descriptor, object_version=2, estimator_generation=1)
    assert ProgressiveSimilarity.stale(descriptor, object_version=1, estimator_generation=2)


def test_temporal_association_does_not_create_active_grounding() -> None:
    registry = GroundingRegistry()
    state = registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 1, recurrent_symbol=True, cross_modal_association=True))
    assert state.maturity is GroundingMaturity.G2
    assert not state.active
    state = registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 2, recurrent_symbol=True, cross_modal_association=True, prospective_prediction=True, causal_intervention=True))
    assert state.maturity is GroundingMaturity.G4
    assert state.active
    state = registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 3, causal_intervention=True, positive=False))
    assert state.suspended
    assert state.historical_peak is GroundingMaturity.G4


def test_grounding_g5_and_negative_evidence_are_target_context_scoped() -> None:
    registry = GroundingRegistry()
    g5 = registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 1, causal_intervention=True, unexperienced_interaction=True))
    assert g5.maturity is GroundingMaturity.G5
    registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 2, causal_intervention=True, positive=False))
    assert registry.states[(1, 2, 3, 4, 0)].suspended
    other = registry.observe(GroundingEvidence(1, 2, 3, 5, 0, 3, causal_intervention=True))
    assert other.active
    restored = GroundingRegistry.from_state_dict(registry.state_dict())
    assert restored.states[(1, 2, 3, 4, 0)].historical_peak is GroundingMaturity.G5


def test_similarity_rejects_unbounded_candidate_discovery() -> None:
    engine = _engine()
    query = {1: _descriptor(0, 1, (1.0,))}
    candidates = {uid: {1: _descriptor(uid, 1, (1.0,))} for uid in range(5)}
    with pytest.raises(ValueError, match="bounded"):
        engine.search(query, candidates, compute_budget=100)


def test_transfer_requires_matched_held_out_trials_and_target_local_action() -> None:
    outcome = _engine().search({1: _descriptor(0, 1, (1.0,))}, {8: {1: _descriptor(8, 1, (1.1,))}}, compute_budget=10)
    correspondence = propose_correspondence(1, 10, 20, outcome)
    assert correspondence is not None
    invalid = TransferTrial(correspondence, 20, 77, 1.0, 0.0, held_out=True, matched=False)
    assert not validate_transfer((invalid,), minimum_trials=1, effect_threshold=0).validated
    first = TransferTrial(correspondence, 20, 77, 1.0, 0.0, held_out=True, matched=True)
    second = TransferTrial(correspondence, 20, 77, 2.0, 0.0, held_out=True, matched=True)
    decision = validate_transfer((first, second), minimum_trials=2, effect_threshold=0)
    assert decision.validated
    assert decision.target_native_action == 77
