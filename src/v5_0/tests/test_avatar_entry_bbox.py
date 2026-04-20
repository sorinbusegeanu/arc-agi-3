from __future__ import annotations

from unittest.mock import patch

from v5_0.contracts.avatar_types import (
    AvatarSelectedResult,
    CandidateComponent,
    CrossResetAvatarEvidence,
    MultiResetDiagnostics,
    ScoredStepCandidate,
)
from v5_0.avatar.track_builder import build_tracks
from v5_0.runtime.run_avatar_bootstrap import _build_multi_reset_avatar_report


def _scored(component: CandidateComponent, score: float = 0.8) -> ScoredStepCandidate:
    return ScoredStepCandidate(
        component=component,
        score=score,
        direction_agreement_score=1.0,
        movement_consistency_score=1.0,
        shape_consistency_score=1.0,
        compactness_score=1.0,
    )


def test_track_entry_bbox_is_inferred_from_first_post_only_component():
    first = CandidateComponent(
        step_index=0,
        action="UP",
        blocked_action=False,
        frame_width=64,
        frame_height=64,
        bbox=(0, 16, 7, 23),
        area=64,
        pre_center=(3.5, 16.0),
        post_center=(3.5, 19.5),
        observed_dx=0.0,
        observed_dy=0.0,
        pre_non_background_cells=tuple(),
        post_non_background_cells=tuple((x, y) for y in range(16, 24) for x in range(0, 8)),
        value_histogram_pre={5: 64},
        value_histogram_post={9: 64},
    )
    second = CandidateComponent(
        step_index=2,
        action="DOWN",
        blocked_action=False,
        frame_width=64,
        frame_height=64,
        bbox=(0, 24, 7, 31),
        area=64,
        pre_center=(3.5, 19.5),
        post_center=(3.5, 27.5),
        observed_dx=0.0,
        observed_dy=8.0,
        pre_non_background_cells=tuple((x, y) for y in range(16, 24) for x in range(0, 8)),
        post_non_background_cells=tuple((x, y) for y in range(24, 32) for x in range(0, 8)),
        value_histogram_pre={5: 64, 9: 64},
        value_histogram_post={5: 64, 9: 64},
    )
    tracks = build_tracks(((_scored(first),), tuple(), (_scored(second),)))
    assert tracks
    assert tracks[0].entry_bbox == (0, 24, 7, 31)


def test_track_entry_bbox_keeps_post_only_component_when_clamped_to_top_edge():
    top_edge = CandidateComponent(
        step_index=0,
        action="UP",
        blocked_action=False,
        frame_width=64,
        frame_height=64,
        bbox=(24, 0, 31, 7),
        area=64,
        pre_center=(27.5, 3.5),
        post_center=(27.5, 3.5),
        observed_dx=0.0,
        observed_dy=0.0,
        pre_non_background_cells=tuple(),
        post_non_background_cells=tuple((x, y) for y in range(0, 8) for x in range(24, 32)),
        value_histogram_pre={5: 64},
        value_histogram_post={9: 64},
    )
    follow = CandidateComponent(
        step_index=1,
        action="LEFT",
        blocked_action=False,
        frame_width=64,
        frame_height=64,
        bbox=(16, 0, 23, 7),
        area=64,
        pre_center=(27.5, 3.5),
        post_center=(19.5, 3.5),
        observed_dx=-8.0,
        observed_dy=0.0,
        pre_non_background_cells=tuple((x, y) for y in range(0, 8) for x in range(24, 32)),
        post_non_background_cells=tuple((x, y) for y in range(0, 8) for x in range(16, 24)),
        value_histogram_pre={5: 64, 9: 64},
        value_histogram_post={5: 64, 9: 64},
    )
    tracks = build_tracks(((_scored(top_edge),), (_scored(follow),)))
    assert tracks
    assert tracks[0].entry_bbox == (24, 0, 31, 7)


def test_track_entry_bbox_does_not_backshift_zero_motion_probe():
    first = CandidateComponent(
        step_index=0,
        action="UP",
        blocked_action=True,
        frame_width=64,
        frame_height=64,
        bbox=(8, 8, 15, 15),
        area=64,
        pre_center=(11.5, 11.5),
        post_center=(11.5, 11.5),
        observed_dx=0.0,
        observed_dy=0.0,
        pre_non_background_cells=tuple((x, y) for y in range(8, 16) for x in range(8, 16)),
        post_non_background_cells=tuple((x, y) for y in range(8, 16) for x in range(8, 16)),
        value_histogram_pre={5: 64, 9: 64},
        value_histogram_post={5: 64, 9: 64},
    )
    follow = CandidateComponent(
        step_index=1,
        action="LEFT",
        blocked_action=False,
        frame_width=64,
        frame_height=64,
        bbox=(0, 8, 7, 15),
        area=64,
        pre_center=(11.5, 11.5),
        post_center=(3.5, 11.5),
        observed_dx=-8.0,
        observed_dy=0.0,
        pre_non_background_cells=tuple((x, y) for y in range(8, 16) for x in range(8, 16)),
        post_non_background_cells=tuple((x, y) for y in range(8, 16) for x in range(0, 8)),
        value_histogram_pre={5: 64, 9: 64},
        value_histogram_post={5: 64, 9: 64},
    )
    tracks = build_tracks(((_scored(first),), (_scored(follow),)))
    assert tracks
    assert tracks[0].entry_bbox == (8, 8, 15, 15)


def test_multi_reset_selected_bbox_keeps_edge_clamped_entry_for_cross_reset_cluster_selection():
    candidate = type(
        "Candidate",
        (),
        {
            "candidate_id": "candidate_000",
            "entry_bbox": (24, 0, 31, 7),
            "bbox": (24, 8, 31, 15),
            "support_actions": ("UP", "DOWN", "LEFT", "RIGHT"),
            "observed_motion_vectors": ((0.0, 0.0), (-8.0, 0.0), (0.0, 8.0), (8.0, 0.0)),
        },
    )()
    episode_report = type(
        "EpisodeReport",
        (),
        {
            "selected": AvatarSelectedResult(
                selected_candidate_id="candidate_000",
                selected_bbox=(24, 8, 31, 15),
                selected_center=(27.5, 11.5),
                confidence=1.0,
                failure_reason=None,
                ranking_margin_to_second=1.0,
            ),
            "candidates": (candidate,),
        },
    )()
    multi_report = type(
        "MultiReport",
        (),
        {
            "episodes": (
                type(
                    "Episode",
                    (),
                    {
                        "episode_index": 0,
                        "seed": 0,
                        "plan": None,
                        "transitions": tuple(),
                        "report": episode_report,
                    },
                )(),
            ),
            "cross_reset_evidence": (
                CrossResetAvatarEvidence(
                    canonical_candidate_id="cross_reset_cluster_000",
                    per_episode_candidate_ids={0: "candidate_000"},
                    episode_indices=(0,),
                    bbox_sequence=((24, 8, 31, 15),),
                    center_sequence=((27.5, 11.5),),
                    value_histogram_pre_aggregate={},
                    value_histogram_post_aggregate={},
                    mean_score=1.0,
                    score_stddev=0.0,
                    shape_consistency_across_resets=1.0,
                    position_consistency_across_resets=1.0,
                    support_episode_count=1,
                ),
            ),
            "selected": AvatarSelectedResult(
                selected_candidate_id="cross_reset_cluster_000",
                selected_bbox=(24, 8, 31, 15),
                selected_center=(27.5, 11.5),
                confidence=1.0,
                failure_reason=None,
                ranking_margin_to_second=1.0,
            ),
            "diagnostics": MultiResetDiagnostics(
                episode_count=1,
                successful_episode_count=1,
                failed_episode_count=0,
                failure_reason_counts={},
                cross_reset_ambiguous=False,
                stable_avatar_found=True,
                confidence_accumulated=1.0,
                dropped_episode_indices=tuple(),
            ),
        },
    )()

    with patch("v5_0.runtime.run_avatar_bootstrap.identify_avatar_candidates_multi_reset", return_value=multi_report):
        out = _build_multi_reset_avatar_report(
            plan=type("Plan", (), {})(),
            episode_transitions=(tuple(),),
            seed=123,
        )

    assert out.selected.selected_bbox == (24, 0, 31, 7)
    assert out.selected.selected_center == (27.5, 3.5)


def test_multi_reset_selected_bbox_uses_observed_bbox_for_interior_cross_reset_cluster_selection():
    candidate = type(
        "Candidate",
        (),
        {
            "candidate_id": "candidate_000",
            "entry_bbox": (32, 40, 39, 47),
            "bbox": (32, 48, 39, 55),
            "support_actions": ("UP", "DOWN", "LEFT", "RIGHT"),
            "observed_motion_vectors": ((0.0, 0.0), (-8.0, 0.0), (0.0, 8.0), (8.0, 0.0)),
        },
    )()
    episode_report = type(
        "EpisodeReport",
        (),
        {
            "selected": AvatarSelectedResult(
                selected_candidate_id="candidate_000",
                selected_bbox=(32, 48, 39, 55),
                selected_center=(35.5, 51.5),
                confidence=1.0,
                failure_reason=None,
                ranking_margin_to_second=1.0,
            ),
            "candidates": (candidate,),
        },
    )()
    multi_report = type(
        "MultiReport",
        (),
        {
            "episodes": (
                type(
                    "Episode",
                    (),
                    {
                        "episode_index": 0,
                        "seed": 0,
                        "plan": None,
                        "transitions": tuple(),
                        "report": episode_report,
                    },
                )(),
            ),
            "cross_reset_evidence": (
                CrossResetAvatarEvidence(
                    canonical_candidate_id="cross_reset_cluster_000",
                    per_episode_candidate_ids={0: "candidate_000"},
                    episode_indices=(0,),
                    bbox_sequence=((32, 48, 39, 55),),
                    center_sequence=((35.5, 51.5),),
                    value_histogram_pre_aggregate={},
                    value_histogram_post_aggregate={},
                    mean_score=1.0,
                    score_stddev=0.0,
                    shape_consistency_across_resets=1.0,
                    position_consistency_across_resets=1.0,
                    support_episode_count=1,
                ),
            ),
            "selected": AvatarSelectedResult(
                selected_candidate_id="cross_reset_cluster_000",
                selected_bbox=(32, 48, 39, 55),
                selected_center=(35.5, 51.5),
                confidence=1.0,
                failure_reason=None,
                ranking_margin_to_second=1.0,
            ),
            "diagnostics": MultiResetDiagnostics(
                episode_count=1,
                successful_episode_count=1,
                failed_episode_count=0,
                failure_reason_counts={},
                cross_reset_ambiguous=False,
                stable_avatar_found=True,
                confidence_accumulated=1.0,
                dropped_episode_indices=tuple(),
            ),
        },
    )()

    with patch("v5_0.runtime.run_avatar_bootstrap.identify_avatar_candidates_multi_reset", return_value=multi_report):
        out = _build_multi_reset_avatar_report(
            plan=type("Plan", (), {})(),
            episode_transitions=(tuple(),),
            seed=123,
        )

    assert out.selected.selected_bbox == (32, 48, 39, 55)
    assert out.selected.selected_center == (35.5, 51.5)
