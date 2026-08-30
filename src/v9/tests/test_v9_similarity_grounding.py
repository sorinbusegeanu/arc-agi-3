from __future__ import annotations

import math
import unittest

from v9.correspondence import correspondence_from_search
from v9.grounding import GroundingEvidence, GroundingMaturity, GroundingRegistry
from v9.modalities.symbols import ModalityId
from v9.prediction_planning import ShadowPredictionEvaluator, SymbolActionGate
from v9.progressive_similarity import BootstrapCriteria, NormalizationState, ProgressiveSimilarityEngine, ScaleDescriptor, ScaleStatistics, entropy, stable_distribution


def _descriptor(uid: int, radius: int, values: tuple[float, ...], *, generation: int = 1, version: int = 1, estimator: int = 1):
    return ScaleDescriptor(uid, generation, version, radius, 1, estimator, values)


class ProgressiveSimilarityTests(unittest.TestCase):
    def test_provisional_memory_is_bounded_and_bootstraps(self) -> None:
        stats = ScaleStatistics(reservoir_limit=2, criteria=BootstrapCriteria(sample_count=2, stable_contingency_count=2, descriptor_coverage=2, minimum_observation_span=1))
        stats.update(_descriptor(1, 1, (1.0, 2.0), generation=1), stable_contingency=True); stats.update(_descriptor(2, 1, (2.0, 3.0), generation=2), stable_contingency=True)
        self.assertEqual(stats.state_for_radius(1), NormalizationState.AUTHORITATIVE)
        for row in stats.stats.values(): self.assertLessEqual(len(row.reservoir), 2)

    def test_radius_statistics_are_isolated(self) -> None:
        stats = ScaleStatistics(); stats.update(_descriptor(1, 1, (1.0,))); stats.update(_descriptor(1, 2, (100.0,)))
        self.assertNotEqual(stats.stats[(1, 0)].mean, stats.stats[(2, 0)].mean)

    def test_stable_entropy_contract(self) -> None:
        self.assertEqual(stable_distribution((5.0,), 1.0), (1.0,)); self.assertEqual(entropy((1.0,)), 0.0)
        equal = stable_distribution((1.0, 1.0), 2.0); self.assertAlmostEqual(equal[0], 0.5); self.assertAlmostEqual(equal[1], 0.5)
        with self.assertRaises(ValueError): stable_distribution((1.0, math.nan), 1.0)

    def test_progressive_search_finds_unique_candidate(self) -> None:
        engine = ProgressiveSimilarityEngine(beta_by_radius={1: 4.0, 2: 4.0}); query = {1: _descriptor(1, 1, (0.0,)), 2: _descriptor(1, 2, (0.0, 0.0))}
        candidates = {10: {1: _descriptor(10, 1, (0.0,)), 2: _descriptor(10, 2, (0.0, 0.0))}, 20: {1: _descriptor(20, 1, (3.0,)), 2: _descriptor(20, 2, (3.0, 3.0))}}
        self.assertEqual(engine.search(query, candidates).winner_uid, 10)

    def test_symmetry_produces_equivalence_set_not_order_winner(self) -> None:
        engine = ProgressiveSimilarityEngine(beta_by_radius={1: 1.0, 2: 1.0}, symmetry_patience=1); query = {1: _descriptor(1, 1, (0.0,)), 2: _descriptor(1, 2, (0.0,))}
        candidates = {10: {1: _descriptor(10, 1, (1.0,)), 2: _descriptor(10, 2, (1.0,))}, 20: {1: _descriptor(20, 1, (1.0,)), 2: _descriptor(20, 2, (1.0,))}}
        outcome = engine.search(query, candidates); self.assertIsNone(outcome.winner_uid); self.assertEqual(outcome.equivalence_set.candidate_uids, (10, 20))

    def test_descriptor_staleness_uses_object_and_estimator_versions(self) -> None:
        d = _descriptor(1, 1, (0.0,), generation=5, version=2, estimator=3)
        self.assertFalse(ProgressiveSimilarityEngine.descriptor_is_stale(d, graph_generation=5, object_version=2, estimator_generation=3)); self.assertTrue(ProgressiveSimilarityEngine.descriptor_is_stale(d, graph_generation=5, object_version=3, estimator_generation=3))

    def test_restart_preserves_beta_and_equivalence_sets(self) -> None:
        engine = ProgressiveSimilarityEngine(beta_by_radius={1: 2.5}, r_max=1); query = {1: _descriptor(1, 1, (0.0,))}; candidates = {10: {1: _descriptor(10, 1, (1.0,))}, 20: {1: _descriptor(20, 1, (1.0,))}}
        engine.search(query, candidates); self.assertEqual(ProgressiveSimilarityEngine.from_state_dict(engine.state_dict()).state_dict(), engine.state_dict())


class CorrespondenceAndGroundingTests(unittest.TestCase):
    def test_cross_modal_correspondence_has_no_grounding_authority(self) -> None:
        engine = ProgressiveSimilarityEngine(beta_by_radius={1: 3.0}, r_max=1); outcome = engine.search({1: _descriptor(1, 1, (0.0,))}, {8: {1: _descriptor(8, 1, (0.0,))}})
        candidate = correspondence_from_search(7, ModalityId.WORLD, ModalityId.SYMBOL, 1, outcome); self.assertIsNotNone(candidate); self.assertFalse(candidate.grounding_authority); self.assertFalse(candidate.transfer_authority); self.assertIsNone(correspondence_from_search(7, ModalityId.WORLD, ModalityId.SYMBOL, 0, outcome))

    def test_cooccurrence_stops_below_causal_grounding(self) -> None:
        state = GroundingRegistry().observe(GroundingEvidence(1, 2, 3, 4, 0, 10, recurrent_symbolic=True, cross_modal_temporal=True)); self.assertEqual(state.maturity, GroundingMaturity.G2); self.assertFalse(state.grounds_relation_active)

    def test_causal_and_heldout_evidence_promote_g4_g5(self) -> None:
        registry = GroundingRegistry(); g4 = registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 10, recurrent_symbolic=True, cross_modal_temporal=True, predictive=True, causal_intervention=True)); self.assertEqual(g4.maturity, GroundingMaturity.G4)
        g5 = registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 11, recurrent_symbolic=True, cross_modal_temporal=True, predictive=True, causal_intervention=True, held_out=True)); self.assertEqual(g5.maturity, GroundingMaturity.G5)

    def test_negative_grounding_is_context_scoped_and_preserves_history(self) -> None:
        registry = GroundingRegistry(); registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 10, predictive=True, causal_intervention=True, held_out=True)); bad = registry.observe(GroundingEvidence(1, 2, 3, 4, 0, 12, predictive=True, causal_intervention=True, positive=False)); other = registry.resolve(1, 2, 3, 5, 0)
        self.assertTrue(bad.suspended); self.assertEqual(bad.historical_peak, GroundingMaturity.G5); self.assertEqual(other.maturity, GroundingMaturity.G0)


class PredictionPlanningGateTests(unittest.TestCase):
    def _state(self, maturity: GroundingMaturity):
        return GroundingRegistry().observe(GroundingEvidence(1, 2, 3, 4, 0, 10, recurrent_symbolic=maturity >= GroundingMaturity.G1, cross_modal_temporal=maturity >= GroundingMaturity.G2, predictive=maturity >= GroundingMaturity.G3, causal_intervention=maturity >= GroundingMaturity.G4, held_out=maturity >= GroundingMaturity.G5))

    def test_shadow_prediction_can_use_g3_without_action_authority(self) -> None:
        evaluator = ShadowPredictionEvaluator(); row = evaluator.record(False, True, grounding=self._state(GroundingMaturity.G3)); self.assertTrue(row.symbol_used); self.assertGreater(evaluator.metrics()["improvement"], 0.0)

    def test_g3_cannot_change_action_ranking(self) -> None:
        ranked = SymbolActionGate().rank({0: 1.0, 1: 0.0}, {1: 5.0}, grounding=self._state(GroundingMaturity.G3), available_actions=(0, 1), higher_memory_validated=True); self.assertEqual(ranked[0].action, 0)

    def test_g4_changes_only_target_local_available_action_ranking(self) -> None:
        ranked = SymbolActionGate().rank({0: 1.0, 1: 0.0}, {1: 5.0, 9: 100.0}, grounding=self._state(GroundingMaturity.G4), available_actions=(0, 1), higher_memory_validated=True); self.assertEqual(ranked[0].action, 1); self.assertNotIn(9, {row.action for row in ranked})

    def test_cross_environment_requires_g5(self) -> None:
        gate = SymbolActionGate(); g4 = gate.rank({0: 1.0, 1: 0.0}, {1: 5.0}, grounding=self._state(GroundingMaturity.G4), available_actions=(0, 1), higher_memory_validated=True, cross_environment=True); g5 = gate.rank({0: 1.0, 1: 0.0}, {1: 5.0}, grounding=self._state(GroundingMaturity.G5), available_actions=(0, 1), higher_memory_validated=True, cross_environment=True)
        self.assertEqual(g4[0].action, 0); self.assertEqual(g5[0].action, 1)


if __name__ == "__main__": unittest.main()
