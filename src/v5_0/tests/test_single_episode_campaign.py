from __future__ import annotations

import io
import json
from pathlib import Path
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from v5_0 import cli as v5_cli
from v5_0.cli import build_parser
from v5_0.contact.service import run_controlled_contact_multi_reset
from v5_0.contracts.avatar_types import ContactExperimentEpisode, SavedLevelTrace
from v5_0.runtime.campaign_state import CampaignLevelState
from v5_0.runtime.run_avatar_bootstrap import (
    run_frontier_avatar_bootstrap_from_frontier,
    run_frontier_level_from_live_session,
    run_full_campaign_analysis,
)
from v5_0.memory.trace_store import get_best_trace_for_level, initialize_trace_store, save_level_trace
from v5_0.replay.player import (
    is_bootstrap_replay_source,
    is_frontier_prefix_replay_source,
    is_solved_prefix_replay_source,
    trace_includes_bootstrap_prefix,
)


def test_cli_default_episode_count_is_one():
    parser = build_parser()
    args = parser.parse_args(["--game-id", "ez01"])
    assert int(args.episode_count) == 1


def test_cli_comma_separated_game_ids_run_campaigns_in_sequence(monkeypatch):
    calls: list[str] = []

    def _fake_campaign(**kwargs):
        calls.append(str(kwargs["game_id"]))
        return {"game_id": kwargs["game_id"], "solved": True, "artifact_paths": {}}

    monkeypatch.setattr("sys.argv", ["v5_0", "--game-id", "ez01,ez02,ez03", "--campaign-solve"])
    monkeypatch.setattr(v5_cli, "_clear_game_output_dir", lambda **_kwargs: None)
    monkeypatch.setattr(v5_cli, "get_global_trace_store_path", lambda: "/tmp/trace_store.sqlite")
    monkeypatch.setattr(v5_cli, "run_full_campaign_analysis", _fake_campaign)

    assert v5_cli.main() == 0
    assert calls == ["ez01", "ez02", "ez03"]


def test_cli_multi_game_campaign_prints_aggregate_solved_levels(monkeypatch, capsys):
    payloads = {
        "ez01": {
            "game_id": "ez01",
            "solved": True,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 5,
            "diagnostics": {"solved_level_count": 5, "requested_level_count": 5},
        },
        "ez02": {
            "game_id": "ez02",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 4 + [{"solved": False}],
            "diagnostics": {"solved_level_count": 4, "requested_level_count": 5},
        },
        "ez03": {
            "game_id": "ez03",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 3 + [{"solved": False}] * 2,
            "diagnostics": {"solved_level_count": 3, "requested_level_count": 5},
        },
        "ez04": {
            "game_id": "ez04",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 3 + [{"solved": False}] * 2,
            "diagnostics": {"solved_level_count": 3, "requested_level_count": 5},
        },
    }

    def _fake_campaign(**kwargs):
        return dict(payloads[str(kwargs["game_id"])])

    monkeypatch.setattr("sys.argv", ["v5_0", "--game-id", "ez01,ez02,ez03,ez04", "--campaign-solve"])
    monkeypatch.setattr(v5_cli, "_clear_game_output_dir", lambda **_kwargs: None)
    monkeypatch.setattr(v5_cli, "get_global_trace_store_path", lambda: "/tmp/trace_store.sqlite")
    monkeypatch.setattr(v5_cli, "run_full_campaign_analysis", _fake_campaign)

    assert v5_cli.main() == 2
    out = capsys.readouterr().out
    assert "15 levels out of 20 for 4 games" in out


def test_cli_multi_game_campaign_uses_levels_not_stale_diagnostics(monkeypatch, capsys):
    payloads = {
        "ez01": {
            "game_id": "ez01",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 5,
            "diagnostics": {"solved_level_count": 2, "requested_level_count": 5},
        },
        "ez02": {
            "game_id": "ez02",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 4 + [{"solved": False}],
            "diagnostics": {"solved_level_count": 1, "requested_level_count": 5},
        },
        "ez03": {
            "game_id": "ez03",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 3 + [{"solved": False}] * 2,
            "diagnostics": {"solved_level_count": 0, "requested_level_count": 5},
        },
        "ez04": {
            "game_id": "ez04",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 3 + [{"solved": False}] * 2,
            "diagnostics": {"solved_level_count": 0, "requested_level_count": 5},
        },
    }

    def _fake_campaign(**kwargs):
        return dict(payloads[str(kwargs["game_id"])])

    monkeypatch.setattr("sys.argv", ["v5_0", "--game-id", "ez01,ez02,ez03,ez04", "--campaign-solve"])
    monkeypatch.setattr(v5_cli, "_clear_game_output_dir", lambda **_kwargs: None)
    monkeypatch.setattr(v5_cli, "get_global_trace_store_path", lambda: "/tmp/trace_store.sqlite")
    monkeypatch.setattr(v5_cli, "run_full_campaign_analysis", _fake_campaign)

    assert v5_cli.main() == 2
    out = capsys.readouterr().out
    assert "15 levels out of 20 for 4 games" in out


def test_cli_multi_game_campaign_uses_requested_level_count_for_total(monkeypatch, capsys):
    payloads = {
        "ez01": {
            "game_id": "ez01",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 4,
            "diagnostics": {"solved_level_count": 4, "requested_level_count": 5},
        },
        "ez02": {
            "game_id": "ez02",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 4,
            "diagnostics": {"solved_level_count": 4, "requested_level_count": 5},
        },
        "ez03": {
            "game_id": "ez03",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 4,
            "diagnostics": {"solved_level_count": 4, "requested_level_count": 5},
        },
        "ez04": {
            "game_id": "ez04",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 4,
            "diagnostics": {"solved_level_count": 4, "requested_level_count": 5},
        },
        "ez05": {
            "game_id": "ez05",
            "solved": False,
            "artifact_paths": {},
            "levels": [{"solved": True}] * 4,
            "diagnostics": {"solved_level_count": 4, "requested_level_count": 5},
        },
    }

    def _fake_campaign(**kwargs):
        return dict(payloads[str(kwargs["game_id"])])

    monkeypatch.setattr("sys.argv", ["v5_0", "--game-id", "ez01,ez02,ez03,ez04,ez05", "--campaign-solve"])
    monkeypatch.setattr(v5_cli, "_clear_game_output_dir", lambda **_kwargs: None)
    monkeypatch.setattr(v5_cli, "get_global_trace_store_path", lambda: "/tmp/trace_store.sqlite")
    monkeypatch.setattr(v5_cli, "run_full_campaign_analysis", _fake_campaign)

    assert v5_cli.main() == 2
    out = capsys.readouterr().out
    assert "20 levels out of 25 for 5 games" in out


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
def test_campaign_summary_hides_unsolved_level_attempt_actions(
    _init,
    _seq,
    _db,
    _load,
    _frontier,
    _current,
    _verified,
    _replay,
    _update,
    _write_index,
    tmp_path,
):
    output_dir = tmp_path / "runs"

    def _write_campaign_side_effect(**kwargs):
        campaign_dir = Path(kwargs["run_dir"])
        campaign_dir.mkdir(parents=True, exist_ok=True)
        trace_path = campaign_dir / "campaign_action_trace.json"
        trace_path.write_text(
            json.dumps(
                [
                    {"level_id": "L0", "action": "RIGHT"},
                    {"level_id": "L0", "action": "RIGHT"},
                ]
            ),
            encoding="utf-8",
        )
        return {}

    def _fake_frontier(**_kwargs):
        return {
            "solved": False,
            "solution": {
                "game_id": "ez01",
                "level_id": "L0",
                "solved": False,
                "action_trace": (
                    {"action": "RIGHT", "source": "frontier_solve"},
                    {"action": "RIGHT", "source": "frontier_solve"},
                ),
                "step_count": 2,
                "terminal": False,
                "level_transition": False,
                "failure_reason": "route_exhausted_without_progress",
            },
            "saved_trace": None,
            "diagnostics": {},
            "failure_reason": "route_exhausted_without_progress",
            "artifact_paths": {},
        }

    buffer = io.StringIO()
    with patch("v5_0.runtime.run_avatar_bootstrap.write_campaign_artifacts", side_effect=_write_campaign_side_effect), patch(
        "v5_0.runtime.run_avatar_bootstrap.run_frontier_level_from_live_session", side_effect=_fake_frontier
    ), redirect_stdout(buffer):
        run_full_campaign_analysis(game_id="ez01", output_dir=str(output_dir))

    assert "L0: [no actions]" in buffer.getvalue()


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

    assert run_probe.call_count == 2
    assert int(run_probe.call_args.kwargs["episode_count"]) == 7
    assert bool(run_probe.call_args_list[0].kwargs["render_terminal"]) is False
    assert bool(run_probe.call_args_list[1].kwargs["render_terminal"]) is False


def test_frontier_bootstrap_discards_first_probe_then_captures_from_fresh_replay():
    with patch(
        "v5_0.runtime.run_avatar_bootstrap.run_probe_episodes_at_frontier",
        side_effect=[(("warmup",),), (("capture",),)],
    ) as run_probe:
        out = run_frontier_avatar_bootstrap_from_frontier(
            game_id="ez01",
            frontier_level_id="L3",
            prefix_traces=tuple(),
            episode_count=1,
            seed=123,
            render_terminal=True,
            env_factory=None,
        )

    assert out == (("capture",),)
    assert run_probe.call_count == 2
    assert bool(run_probe.call_args_list[0].kwargs["render_terminal"]) is False
    assert bool(run_probe.call_args_list[1].kwargs["render_terminal"]) is True


def test_frontier_bootstrap_writes_bootstrap_and_reset_markers_to_debug_log(tmp_path):
    debug_log = tmp_path / "debug.log"
    with patch(
        "v5_0.runtime.run_avatar_bootstrap.run_probe_episodes_at_frontier",
        side_effect=[(("warmup",),), (("capture",),)],
    ):
        run_frontier_avatar_bootstrap_from_frontier(
            game_id="ul01",
            frontier_level_id="L3",
            prefix_traces=(
                SavedLevelTrace("ul01", "L0", True, ("RIGHT", "DOWN"), 2, None, 1, True),
                SavedLevelTrace("ul01", "L1", True, ("UP",), 1, None, 1, True),
            ),
            episode_count=1,
            seed=0,
            render_terminal=False,
            env_factory=None,
            debug_log_path=str(debug_log),
        )

    assert debug_log.read_text(encoding="utf-8").splitlines() == [
        "ul01|L3|BOOTSTRAP",
        "ul01|L3|RESET",
        "ul01|L3|PLAY:RDU",
    ]


def test_frontier_level_l0_writes_reset_and_bootstrap_markers_to_debug_log(tmp_path):
    debug_log = tmp_path / "debug.log"
    with patch(
        "v5_0.runtime.run_avatar_bootstrap.run_probe_episodes_at_frontier",
        return_value=(tuple(),),
    ), patch(
        "v5_0.runtime.run_avatar_bootstrap._build_multi_reset_avatar_report",
        return_value=SimpleNamespace(
            selected=SimpleNamespace(failure_reason="no_stable_avatar"),
            diagnostics=SimpleNamespace(),
            episodes=tuple(),
            cross_reset_evidence=tuple(),
        ),
    ), patch(
        "v5_0.runtime.run_avatar_bootstrap.write_multi_reset_artifacts",
        return_value={},
    ), patch(
        "v5_0.runtime.run_avatar_bootstrap._campaign_stable_avatar_found",
        return_value=False,
    ):
        run_frontier_level_from_live_session(
            game_id="ez03",
            frontier_level_id="L0",
            session=object(),
            prefix_traces=(SavedLevelTrace("ez03", "L0", True, ("LEFT",), 1, None, 1, True),),
            output_dir=str(tmp_path),
            seed=0,
            episode_count=1,
            probe_montage=False,
            render_terminal=False,
            max_steps=40,
            env_factory=None,
            debug_log_path=str(debug_log),
        )

    assert debug_log.read_text(encoding="utf-8").splitlines() == [
        "ez03|L0|BOOTSTRAP",
        "ez03|L0|RESET",
        "ez03|L0|PLAY:L",
    ]


def test_replayed_solved_level_trace_preserves_action_sources_round_trip(tmp_path):
    db_path = tmp_path / "trace.sqlite"
    initialize_trace_store(db_path)
    trace = SavedLevelTrace(
        game_id="ez01",
        level_id="L0",
        solved=True,
        action_trace=("RIGHT",),
        step_count=1,
        source_run_id=None,
        trace_version=1,
        replay_verified=True,
        action_sources=("solved_prefix_replay",),
        trace_id="trace-1",
    )
    save_level_trace(db_path=db_path, trace=trace, trace_id="trace-1")
    restored = get_best_trace_for_level(db_path=db_path, game_id="ez01", level_id="L0")
    assert restored is not None
    assert restored.action_sources == ("solved_prefix_replay",)


def test_replay_classification_distinguishes_corrected_sources():
    assert is_bootstrap_replay_source("bootstrap_replay")
    assert is_solved_prefix_replay_source("solved_prefix_replay")
    assert is_frontier_prefix_replay_source("frontier_prefix_replay")
    assert not trace_includes_bootstrap_prefix({"action_sources": ["solved_prefix_replay", "frontier_prefix_replay"]})
    assert trace_includes_bootstrap_prefix({"action_sources": ["bootstrap_replay"]})


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
