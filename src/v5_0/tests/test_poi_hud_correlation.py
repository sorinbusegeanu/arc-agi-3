import unittest
from types import SimpleNamespace

from v5_0.contracts.avatar_types import CrossResetPOIEvidence, POICandidate
from v5_0.poi.service import (
    _deduplicate_redundant_candidates,
    _hud_color_correlation_score,
    _rank_cross_reset_candidates,
)
from v5_0.poi.static_inventory import build_static_object_inventory


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

    def _evidence_with_bbox(self, poi_id, bbox, hist):
        return CrossResetPOIEvidence(
            canonical_poi_id=poi_id,
            episode_indices=(0,),
            per_episode_poi_ids={0: poi_id},
            bbox_sequence=(tuple(bbox),),
            center_sequence=(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),),
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

    def test_frame_spanning_background_candidate_is_not_ranked_as_poi(self):
        evidence = (
            self._evidence_with_bbox("background", (0, 0, 63, 63), {4: 1000, 9: 10}),
            self._evidence_with_bbox("object", (8, 8, 13, 13), {11: 36}),
        )

        ranked = _rank_cross_reset_candidates(evidence, frame_width=64, frame_height=64)

        self.assertEqual(tuple(item.poi_id for item in ranked), ("object",))

    def test_avatar_colored_cross_reset_candidate_is_not_ranked_as_poi(self):
        evidence = (
            self._evidence_with_bbox("avatar", (8, 2, 13, 7), {9: 72}),
            self._evidence_with_bbox("object", (14, 8, 19, 13), {11: 36}),
        )

        ranked = _rank_cross_reset_candidates(evidence, avatar_histograms=({9: 1.0},))

        self.assertEqual(tuple(item.poi_id for item in ranked), ("object",))

    def test_avatar_primary_color_candidate_is_filtered_even_with_secondary_noise(self):
        evidence = (
            self._evidence_with_bbox("avatar_like", (8, 2, 13, 7), {9: 60, 3: 12}),
            self._evidence_with_bbox("key", (14, 8, 19, 13), {11: 36}),
            self._evidence_with_bbox("door", (22, 8, 27, 13), {3: 36}),
        )

        ranked = _rank_cross_reset_candidates(evidence, avatar_histograms=({9: 0.7, 3: 0.3},))

        self.assertEqual(tuple(item.poi_id for item in ranked), ("door", "key"))

    def test_static_inventory_splits_adjacent_different_colored_objects(self):
        frame = (
            (5, 5, 5, 5, 5),
            (5, 2, 14, 5, 5),
            (5, 5, 5, 5, 5),
        )
        transition = SimpleNamespace(pre_frame=frame, post_frame=frame)

        items, dropped = build_static_object_inventory(
            (transition,),
            SimpleNamespace(selected_bbox=(4, 1, 4, 1)),
        )

        self.assertEqual(dropped, {})
        self.assertEqual({tuple(item["bbox"]) for item in items}, {(1, 1, 1, 1), (2, 1, 2, 1)})
        self.assertEqual({tuple(item["value_histogram"].keys()) for item in items}, {(2,), (14,)})

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
