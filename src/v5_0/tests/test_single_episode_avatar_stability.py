from __future__ import annotations

import unittest
from unittest.mock import patch

from v5_0.avatar.service import (
    _select_final_cross_reset_result,
    identify_avatar_candidates_multi_reset,
)
from v5_0.contracts.avatar_types import (
    AvatarCandidate,
    AvatarDiagnostics,
    AvatarIdentificationReport,
    AvatarSelectedResult,
    CrossResetAvatarEvidence,
)
from v5_0.runtime.run_avatar_bootstrap import _campaign_stable_avatar_found


class TestSingleEpisodeAvatarStability(unittest.TestCase):
    def _candidate(self, cid: str, score: float = 0.9) -> AvatarCandidate:
        return AvatarCandidate(
            candidate_id=cid,
            bbox=(1, 1, 2, 2),
            center=(1.5, 1.5),
            score=float(score),
            support_step_indices=(0, 1),
            support_actions=("RIGHT", "DOWN"),
            observed_motion_vectors=((1.0, 0.0), (0.0, 1.0)),
            direction_agreement_score=0.9,
            shape_consistency_score=0.9,
            track_consistency_score=0.9,
            value_histogram_pre={1: 2},
            value_histogram_post={1: 2},
            failure_flags=(),
        )

    def _report(self, *, candidate_id: str | None, failure_reason: str | None, score: float = 0.9) -> AvatarIdentificationReport:
        candidates = tuple() if candidate_id is None else (self._candidate(candidate_id, score),)
        selected_bbox = None if candidate_id is None else (1, 1, 2, 2)
        selected_center = None if candidate_id is None else (1.5, 1.5)
        return AvatarIdentificationReport(
            candidates=candidates,
            selected=AvatarSelectedResult(
                selected_candidate_id=candidate_id if failure_reason is None else None,
                selected_bbox=selected_bbox if failure_reason is None else None,
                selected_center=selected_center if failure_reason is None else None,
                confidence=score if failure_reason is None else 0.0,
                failure_reason=failure_reason,
                ranking_margin_to_second=1.0,
            ),
            diagnostics=AvatarDiagnostics(
                per_step_candidate_counts={0: 1},
                per_step_top_scores={0: (score,)},
                total_candidate_count=(1 if candidate_id is not None else 0),
                total_track_count=(1 if candidate_id is not None else 0),
                dropped_candidate_reasons={},
                ambiguous_ranking=False,
                no_motion=(candidate_id is None),
                all_blocked=False,
            ),
        )

    def test_one_successful_episode_clear_avatar_is_stable(self):
        with patch(
            "v5_0.avatar.service.identify_avatar_candidates",
            return_value=self._report(candidate_id="a0", failure_reason=None),
        ):
            out = identify_avatar_candidates_multi_reset((tuple(),))
        self.assertTrue(out.diagnostics.stable_avatar_found)
        self.assertIsNone(out.selected.failure_reason)

    def test_one_successful_episode_not_insufficient_support(self):
        with patch(
            "v5_0.avatar.service.identify_avatar_candidates",
            return_value=self._report(candidate_id="a0", failure_reason=None),
        ):
            out = identify_avatar_candidates_multi_reset((tuple(),))
        self.assertNotEqual(out.selected.failure_reason, "insufficient_support")

    def test_campaign_stable_gate_true_for_single_episode(self):
        avatar_multi = type("M", (), {})()
        avatar_multi.diagnostics = type("D", (), {"episode_count": 1, "successful_episode_count": 1, "cross_reset_ambiguous": False})()
        avatar_multi.selected = type("S", (), {"failure_reason": None})()
        self.assertTrue(_campaign_stable_avatar_found(avatar_multi))

    def test_multi_episode_mode_still_requires_two_successful(self):
        side_effect = [
            self._report(candidate_id="a0", failure_reason=None),
            self._report(candidate_id=None, failure_reason="no_moving_candidate"),
        ]
        with patch("v5_0.avatar.service.identify_avatar_candidates", side_effect=side_effect):
            out = identify_avatar_candidates_multi_reset((tuple(), tuple()))
        self.assertFalse(out.diagnostics.stable_avatar_found)
        self.assertEqual(out.selected.failure_reason, "insufficient_support")

    def test_genuine_ambiguity_still_returns_ambiguous_failure(self):
        evidence = (
            CrossResetAvatarEvidence(
                canonical_candidate_id="c0",
                episode_indices=(0,),
                per_episode_candidate_ids={0: "a0"},
                bbox_sequence=((1, 1, 2, 2),),
                center_sequence=((1.5, 1.5),),
                value_histogram_pre_aggregate={1: 2},
                value_histogram_post_aggregate={1: 2},
                mean_score=0.80,
                score_stddev=0.01,
                shape_consistency_across_resets=1.0,
                position_consistency_across_resets=1.0,
                support_episode_count=1,
            ),
            CrossResetAvatarEvidence(
                canonical_candidate_id="c1",
                episode_indices=(0,),
                per_episode_candidate_ids={0: "a1"},
                bbox_sequence=((4, 4, 5, 5),),
                center_sequence=((4.5, 4.5),),
                value_histogram_pre_aggregate={2: 2},
                value_histogram_post_aggregate={2: 2},
                mean_score=0.79,
                score_stddev=0.02,
                shape_consistency_across_resets=1.0,
                position_consistency_across_resets=1.0,
                support_episode_count=1,
            ),
        )
        selected, ambiguous = _select_final_cross_reset_result(evidence)
        self.assertTrue(ambiguous)
        self.assertEqual(selected.failure_reason, "ambiguous_avatar")


if __name__ == "__main__":
    unittest.main()
