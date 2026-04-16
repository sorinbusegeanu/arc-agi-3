from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v5_0.cli import build_parser
from v5_0.contact.service import run_controlled_contact_multi_reset
from v5_0.contracts.avatar_types import ContactExperimentEpisode, SavedLevelTrace
from v5_0.runtime.campaign_state import CampaignLevelState
from v5_0.runtime.run_avatar_bootstrap import (
    run_frontier_avatar_bootstrap_from_frontier,
    run_full_campaign_analysis,
)


def test_cli_default_episode_count_is_one():
    parser = build_parser()
    args = parser.parse_args(["--game-id", "ez01"])
    assert int(args.episode_count) == 1


@patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.write_campaign_artifacts", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.update_campaign_state_after_level", side_effect=lambda **kwargs: kwargs["state"])
@patch("v5_0.runtime.run_avatar_bootstrap.run_frontier_level_from_live_session")
@patch(
    "v5_0.runtime.run_avatar_bootstrap.replay_prefix_traces_to_frontier",
    return_value={"session": object(), "frontier_reached": True, "frontier_level_id": "L0", "divergence": False, "executed_action_count": 0},
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=["L0", None, None])
@patch(
    "v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state",
    return_value={
        "L0": CampaignLevelState(game_id="ez01", level_id="L0", status="pending", solved=False, solution_trace_path=None, best_step_count=None, attempt_count=0)
    },
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0",))
@patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store", return_value="/tmp/trace_store.sqlite")
def test_campaign_default_passes_episode_count_one_unchanged(
    _init,
    _seq,
    _db,
    _load,
    _frontier,
    _current,
    _verified,
    _replay,
    run_frontier,
    _update,
    _write_campaign,
    _write_index,
):
    run_frontier.return_value = {
        "solved": False,
        "solution": {"action_trace": tuple(), "step_count": 0},
        "saved_trace": None,
        "diagnostics": {},
        "failure_reason": "failed",
        "artifact_paths": {},
    }

    run_full_campaign_analysis(game_id="ez01", output_dir="runs_v5_0_test")

    assert run_frontier.call_count == 1
    assert int(run_frontier.call_args.kwargs["episode_count"]) == 1


@patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.write_campaign_artifacts", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.update_campaign_state_after_level", side_effect=lambda **kwargs: kwargs["state"])
@patch(
    "v5_0.runtime.run_avatar_bootstrap.replay_prefix_traces_to_frontier",
    return_value={"session": object(), "frontier_reached": True, "frontier_level_id": "L0", "divergence": False, "executed_action_count": 0},
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=["L0", None, None])
@patch(
    "v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state",
    return_value={
        "L0": CampaignLevelState(game_id="ez01", level_id="L0", status="pending", solved=False, solution_trace_path=None, best_step_count=None, attempt_count=0)
    },
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0",))
@patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store", return_value="/tmp/trace_store.sqlite")
def test_campaign_default_creates_only_episode_000(
    _init,
    _seq,
    _db,
    _load,
    _frontier,
    _current,
    _verified,
    _replay,
    _update,
    _write_campaign,
    _write_index,
    tmp_path,
):
    output_dir = tmp_path / "runs"

    def _fake_frontier(**kwargs):
        level_id = str(kwargs["frontier_level_id"])
        game_id = str(kwargs["game_id"])
        episode_count = int(kwargs["episode_count"])
        base = Path(kwargs["output_dir"]) / game_id / level_id / "multi_reset"
        base.mkdir(parents=True, exist_ok=True)
        for idx in range(episode_count):
            (base / f"episode_{idx:03d}").mkdir(parents=True, exist_ok=True)
        return {
            "solved": False,
            "solution": {"action_trace": tuple(), "step_count": 0},
            "saved_trace": None,
            "diagnostics": {},
            "failure_reason": "failed",
            "artifact_paths": {},
        }

    with patch("v5_0.runtime.run_avatar_bootstrap.run_frontier_level_from_live_session", side_effect=_fake_frontier):
        run_full_campaign_analysis(game_id="ez01", output_dir=str(output_dir))

    multi_reset = output_dir / "ez01" / "L0" / "multi_reset"
    assert (multi_reset / "episode_000").exists()
    assert not (multi_reset / "episode_001").exists()
    assert not (multi_reset / "episode_002").exists()


def test_contact_experiments_do_not_repeat_same_poi_across_episodes_when_one_episode():
    avatar_multi = SimpleNamespace(
        episodes=(SimpleNamespace(episode_index=0, report=SimpleNamespace(selected=SimpleNamespace(failure_reason=None))),),
    )
    poi_candidate = SimpleNamespace(poi_id="p1", confidence=0.9, bbox=(1, 1, 2, 2), area=4, ambiguity_flags=())
    poi_multi = {"episodes": (SimpleNamespace(episode_index=0, poi_report=SimpleNamespace(candidates=(poi_candidate,))),)}

    with patch("v5_0.contact.service.run_controlled_contact_for_episode", return_value=(SimpleNamespace(outcome=SimpleNamespace(outcome_type="no_effect", contact_step_index=None, level_transition=False, terminal=False, hud_change_only=False), steps=tuple()),)) as run_one:
        report = run_controlled_contact_multi_reset(
            avatar_multi_report=avatar_multi,
            poi_multi_bundle=poi_multi,
            plan=SimpleNamespace(),
            base_seed=0,
            render_terminal=False,
            env_factory=None,
        )

    assert run_one.call_count == 1
    assert len(report.episodes) == 1
    assert isinstance(report.episodes[0], ContactExperimentEpisode)


def test_frontier_bootstrap_uses_passed_episode_count_unchanged():
    with patch("v5_0.runtime.run_avatar_bootstrap.run_probe_episodes_at_frontier", return_value=(tuple(),) * 7) as run_probe:
        run_frontier_avatar_bootstrap_from_frontier(
            game_id="ez01",
            frontier_level_id="L1",
            prefix_traces=tuple(),
            episode_count=7,
            seed=123,
            render_terminal=False,
            env_factory=None,
        )

    assert int(run_probe.call_args.kwargs["episode_count"]) == 7


@patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.update_campaign_state_after_level", side_effect=lambda **kwargs: kwargs["state"])
@patch(
    "v5_0.runtime.run_avatar_bootstrap.replay_prefix_traces_to_frontier",
    return_value={"session": object(), "frontier_reached": True, "frontier_level_id": "L0", "divergence": False, "executed_action_count": 0},
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=["L0", None, None])
@patch(
    "v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state",
    return_value={
        "L0": CampaignLevelState(game_id="ez01", level_id="L0", status="pending", solved=False, solution_trace_path=None, best_step_count=None, attempt_count=0)
    },
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0",))
@patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store", return_value="/tmp/trace_store.sqlite")
def test_campaign_step_trace_and_unsolved_stats_written(
    _init,
    _seq,
    _db,
    _load,
    _frontier,
    _current,
    _verified,
    _replay,
    _update,
    _rebuild,
    _index,
    tmp_path,
):
    output_dir = tmp_path / "runs"

    frontier_result = {
        "solved": False,
        "solution": {
            "game_id": "ez01",
            "level_id": "L0",
            "solved": False,
            "step_count": 1,
            "terminal": False,
            "level_transition": False,
            "failure_reason": "frontier_unsolved",
            "action_trace": (
                {
                    "step_index": 0,
                    "action": "RIGHT",
                    "source": "frontier_solve",
                    "pre_level_index": 0,
                    "post_level_index": 0,
                    "pre_frame": ((1,),),
                    "post_frame": ((2,),),
                },
            ),
        },
        "saved_trace": None,
        "diagnostics": {
            "trajectory_stats": {
                "generated_trajectory_count": 2,
                "attempted_trajectory_count": 1,
                "completed_trajectory_count": 0,
                "min_steps_per_attempted_trajectory": 1,
                "max_steps_per_attempted_trajectory": 1,
                "mean_steps_per_attempted_trajectory": 1.0,
                "total_executed_steps_across_attempted_trajectories": 1,
            }
        },
        "failure_reason": "frontier_unsolved",
        "artifact_paths": {},
    }
    with patch("v5_0.runtime.run_avatar_bootstrap.run_frontier_level_from_live_session", return_value=frontier_result):
        run_full_campaign_analysis(game_id="ez01", output_dir=str(output_dir))

    campaign_step = output_dir / "ez01" / "campaign" / "campaign_step_trace.json"
    unsolved_stats = output_dir / "ez01" / "campaign" / "unsolved_level_trajectory_stats.json"
    assert campaign_step.exists()
    assert unsolved_stats.exists()
    step_rows = json.loads(campaign_step.read_text(encoding="utf-8"))
    stats_rows = json.loads(unsolved_stats.read_text(encoding="utf-8"))
    assert len(step_rows) == 1
    assert step_rows[0]["action"] == "RIGHT"
    assert len(stats_rows) == 1
    assert stats_rows[0]["level_id"] == "L0"


@patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.update_campaign_state_after_level", side_effect=lambda **kwargs: kwargs["state"])
@patch(
    "v5_0.runtime.run_avatar_bootstrap.replay_prefix_traces_to_frontier",
    return_value={"session": object(), "frontier_reached": True, "frontier_level_id": "L0", "divergence": False, "executed_action_count": 0},
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=["L0", None, None])
@patch(
    "v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state",
    return_value={
        "L0": CampaignLevelState(game_id="ez01", level_id="L0", status="pending", solved=False, solution_trace_path=None, best_step_count=None, attempt_count=0)
    },
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0",))
@patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store", return_value="/tmp/trace_store.sqlite")
@patch(
    "v5_0.runtime.run_avatar_bootstrap.finalize_solved_level_trace",
    return_value={
        "saved_trace": SavedLevelTrace(
            game_id="ez01",
            level_id="L0",
            solved=True,
            action_trace=("RIGHT", "DOWN"),
            step_count=2,
            source_run_id=None,
            trace_version=1,
            replay_verified=True,
            action_sources=("bootstrap_replay", "frontier_solve"),
            trace_id="trace_l0",
        ),
        "replay_verified": True,
        "failure_reason": None,
        "trace_id": "trace_l0",
    },
)
def test_solved_level_writes_saved_level_trace_steps(
    _finalize,
    _init,
    _seq,
    _db,
    _load,
    _frontier,
    _current,
    _verified,
    _replay,
    _update,
    _rebuild,
    _index,
    tmp_path,
):
    output_dir = tmp_path / "runs"
    frontier_result = {
        "solved": True,
        "solution": {
            "game_id": "ez01",
            "level_id": "L0",
            "solved": True,
            "step_count": 2,
            "terminal": False,
            "level_transition": True,
            "failure_reason": None,
            "action_trace": (
                {
                    "step_index": 0,
                    "action": "RIGHT",
                    "source": "bootstrap_replay",
                    "pre_level_index": 0,
                    "post_level_index": 0,
                    "pre_frame": ((1,),),
                    "post_frame": ((1,),),
                },
                {
                    "step_index": 1,
                    "action": "DOWN",
                    "source": "frontier_solve",
                    "pre_level_index": 0,
                    "post_level_index": 1,
                    "pre_frame": ((1,),),
                    "post_frame": ((2,),),
                },
            ),
        },
        "saved_trace": None,
        "diagnostics": {},
        "failure_reason": None,
        "artifact_paths": {},
    }
    with patch("v5_0.runtime.run_avatar_bootstrap.run_frontier_level_from_live_session", return_value=frontier_result):
        run_full_campaign_analysis(game_id="ez01", output_dir=str(output_dir))

    saved_steps = output_dir / "ez01" / "L0" / "saved_level_trace_steps.json"
    assert saved_steps.exists()
    rows = json.loads(saved_steps.read_text(encoding="utf-8"))
    assert len(rows) == 2


@patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
@patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store", return_value="/tmp/trace_store.sqlite")
@patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0", "L1"))
@patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=tuple())
@patch(
    "v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state",
    return_value={
        "L0": CampaignLevelState("ez01", "L0", "pending", False, None, None, 0),
        "L1": CampaignLevelState("ez01", "L1", "pending", False, None, None, 0),
    },
)
@patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=["L0", "L1", None, None])
@patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=tuple())
@patch("v5_0.runtime.run_avatar_bootstrap.replay_prefix_traces_to_frontier")
@patch("v5_0.runtime.run_avatar_bootstrap.run_frontier_continuation_from_live_session")
@patch("v5_0.runtime.run_avatar_bootstrap.run_frontier_level_from_live_session")
@patch("v5_0.runtime.run_avatar_bootstrap.update_campaign_state_after_level", side_effect=lambda **kwargs: kwargs["state"])
@patch("v5_0.runtime.run_avatar_bootstrap.finalize_solved_level_trace")
def test_campaign_no_reset_handoff_after_solved_level(
    _finalize,
    _update,
    run_frontier,
    run_continue,
    replay_prefix,
    _prefix,
    _current,
    _frontier,
    _load,
    _db,
    _seq,
    _init,
    _rebuild,
    _index,
):
    sess = object()
    replay_prefix.return_value = {"session": sess, "frontier_reached": True, "frontier_level_id": "L0", "divergence": False}
    run_frontier.return_value = {
        "solved": True,
        "solution": {
            "game_id": "ez01",
            "level_id": "L0",
            "solved": True,
            "step_count": 1,
            "action_trace": (
                {"step_index": 0, "action": "RIGHT", "source": "frontier_solve", "pre_level_index": 0, "post_level_index": 1},
            ),
            "terminal": False,
            "level_transition": True,
            "failure_reason": None,
        },
        "saved_trace": None,
        "diagnostics": {},
        "failure_reason": None,
        "artifact_paths": {},
        "_selected_avatar_obj": SimpleNamespace(selected_bbox=(1, 1, 1, 1), value_histogram={1: 1}),
    }
    run_continue.return_value = {
        "solved": False,
        "solution": {"game_id": "ez01", "level_id": "L1", "solved": False, "step_count": 0, "action_trace": (), "terminal": False, "level_transition": False, "failure_reason": "frontier_unsolved"},
        "saved_trace": None,
        "diagnostics": {},
        "failure_reason": "frontier_unsolved",
        "artifact_paths": {},
    }
    _finalize.return_value = {
        "saved_trace": SavedLevelTrace("ez01", "L0", True, ("RIGHT",), 1, None, 1, True, action_sources=("frontier_solve",), trace_id="t1"),
        "replay_verified": True,
        "failure_reason": None,
        "trace_id": "t1",
    }
    with patch("v5_0.runtime.run_avatar_bootstrap.SessionAdapter") as sa:
        adapter = sa.return_value
        adapter.get_current_observation.return_value = SimpleNamespace(levels_completed=1)
        run_full_campaign_analysis(game_id="ez01", output_dir="runs_v5_0_test")
    assert replay_prefix.call_count == 1
    assert run_frontier.call_count == 1
    assert run_continue.call_count == 1
