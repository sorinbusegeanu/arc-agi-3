import unittest

from v5_0.contracts.avatar_types import CrossResetPOIEvidence, POICandidate
from v5_0.poi.service import (
    _deduplicate_redundant_candidates,
    _hud_color_correlation_score,
    _rank_cross_reset_candidates,
)


class TestPoiHudCorrelation(unittest.TestCase):
    def _evidence(self, poi_id, hist):
        return CrossResetPOIEvidence(
            canonical_poi_id=poi_id,
            episode_indices=(0,),
            per_episode_poi_ids={0: poi_id},
            bbox_sequence=((1, 1, 2, 2),),
            center_sequence=((1.5, 1.5),),
            value_histogram_aggregate=dict(hist),
            mean_confidence=0.7,
            position_consistency_across_resets=1.0,
            support_episode_count=1,
        )

    def test_dominant_color_overlap_ranks_higher(self):
        evidence = (
            self._evidence("p_overlap", {2: 10}),
            self._evidence("p_other", {3: 10}),
        )
        ranked = _rank_cross_reset_candidates(evidence, hud_histograms=({2: 1.0},))
        self.assertEqual(ranked[0].poi_id, "p_overlap")

    def test_background_color_ignored_in_hud_correlation(self):
        score_with_bg = _hud_color_correlation_score({0: 100, 2: 1}, ({2: 1.0},))
        score_no_bg = _hud_color_correlation_score({2: 1}, ({2: 1.0},))
        self.assertAlmostEqual(score_with_bg, score_no_bg, places=6)

    def test_mixed_color_below_pure_dominant_when_similar(self):
        pure = _hud_color_correlation_score({2: 10}, ({2: 1.0},))
        mixed = _hud_color_correlation_score({2: 5, 5: 5}, ({2: 1.0},))
        self.assertGreater(pure, mixed)

    def test_redundant_overlapping_pois_merge_deterministically(self):
        c1 = POICandidate("a", (1, 1, 3, 3), (2.0, 2.0), 9, {2: 10}, (), (0,), "cross_reset", (), 9.0, 0.8, ())
        c2 = POICandidate("b", (1, 1, 3, 3), (2.0, 2.0), 9, {2: 8}, (), (0,), "cross_reset", (), 9.0, 0.7, ())
        kept, dropped = _deduplicate_redundant_candidates((c1, c2))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].poi_id, "a")
        self.assertIn("b", dropped)


if __name__ == "__main__":
    unittest.main()
