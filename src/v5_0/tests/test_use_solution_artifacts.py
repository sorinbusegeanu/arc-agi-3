import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from v5_0 import cli
from v5_0.contracts.avatar_types import CampaignLevelState, SavedLevelTrace
from v5_0.io.final_video_builder import build_final_game_video, collect_complete_run_frames
from v5_0.runtime.run_avatar_bootstrap import run_full_campaign_analysis


class TestUseSolutionArtifacts(unittest.TestCase):
    def _state(self):
        return {
            "L0": CampaignLevelState("ez01", "L0", "solved", True, None, 1, 1),
            "L1": CampaignLevelState("ez01", "L1", "pending", False, None, None, 0),
        }

    @patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=("L0",))
    @patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=(SavedLevelTrace("ez01", "L0", True, ("RIGHT", "DOWN"), 2, None, 1, True, action_sources=("db_solution", "db_solution"), trace_id="l0"),))
    @patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=[None, None])
    @patch("v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state")
    @patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0",))
    @patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store")
    def test_use_solution_populates_campaign_action_trace_when_frontier_never_runs(
        self,
        init_store,
        _seq,
        load_state,
        _frontier,
        _prefix,
        _db,
        _rebuild,
        _index,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            init_store.return_value = str(Path(tmp) / "trace.sqlite")
            load_state.return_value = {
                "L0": CampaignLevelState("ez01", "L0", "solved", True, None, 2, 1),
            }
            run_full_campaign_analysis(game_id="ez01", output_dir=tmp, use_solutions=True)
            trace_path = Path(tmp) / "ez01" / "campaign" / "campaign_action_trace.json"
            self.assertTrue(trace_path.exists())
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual([item["level_id"] for item in payload], ["L0", "L0"])
            self.assertEqual([item["action"] for item in payload], ["RIGHT", "DOWN"])
            self.assertEqual([item["source"] for item in payload], ["db_solution", "db_solution"])

    @patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=("L0", "L1"))
    @patch(
        "v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces",
        return_value=(
            SavedLevelTrace("ez01", "L0", True, ("RIGHT", "DOWN"), 2, None, 1, True, action_sources=("db_solution", "db_solution"), trace_id="l0"),
            SavedLevelTrace("ez01", "L1", True, ("UP",), 1, None, 2, True, action_sources=("db_solution",), trace_id="l1"),
        ),
    )
    @patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=[None, None])
    @patch("v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state")
    @patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0", "L1"))
    @patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store")
    def test_use_solution_prints_saved_level_actions_for_pre_solved_levels(
        self,
        init_store,
        _seq,
        load_state,
        _frontier,
        _prefix,
        _db,
        _rebuild,
        _index,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            init_store.return_value = str(Path(tmp) / "trace.sqlite")
            load_state.return_value = {
                "L0": CampaignLevelState("ez01", "L0", "solved", True, None, 2, 1),
                "L1": CampaignLevelState("ez01", "L1", "solved", True, None, 1, 2),
            }
            buf = StringIO()
            with redirect_stdout(buf):
                run_full_campaign_analysis(game_id="ez01", output_dir=tmp, use_solutions=True)
            text = buf.getvalue()
            self.assertIn("L0: RD", text)
            self.assertIn("L1: U", text)

    @patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=("L0",))
    @patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
    @patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=(SavedLevelTrace("ez01", "L0", True, ("RIGHT",), 1, None, 1, True, trace_id="l0"),))
    @patch("v5_0.runtime.run_avatar_bootstrap.replay_prefix_traces_to_frontier", return_value={"frontier_reached": False, "divergence": True, "session": None})
    @patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=["L1", None, None])
    @patch("v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state")
    @patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0", "L1"))
    @patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store")
    def test_prefix_replay_failure_materializes_level_folder_and_failure_artifact(
        self,
        init_store,
        _seq,
        load_state,
        _frontier,
        _replay,
        _prefix,
        _current,
        _db,
        _rebuild,
        _index,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            init_store.return_value = str(Path(tmp) / "trace.sqlite")
            load_state.return_value = self._state()
            out = run_full_campaign_analysis(game_id="ez01", output_dir=tmp, use_solutions=True)
            level_dir = Path(tmp) / "ez01" / "L1"
            self.assertTrue(level_dir.exists())
            failure_path = level_dir / "prefix_replay_failure.json"
            self.assertTrue(failure_path.exists())
            payload = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("failure_reason"), "prefix_replay_failed")
            self.assertIn("L1/prefix_replay_failure.json", out.get("artifact_paths", {}))

    @patch("v5_0.runtime.run_avatar_bootstrap.write_trace_store_index_artifacts", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.rebuild_trace_store_index", return_value={})
    @patch("v5_0.runtime.run_avatar_bootstrap.get_db_solved_levels_for_game", return_value=("L0",))
    @patch("v5_0.runtime.run_avatar_bootstrap.get_current_run_prefix_traces", return_value=tuple())
    @patch("v5_0.runtime.run_avatar_bootstrap.get_verified_prefix_traces", return_value=(SavedLevelTrace("ez01", "L0", True, ("RIGHT",), 1, None, 1, True, trace_id="l0"),))
    @patch("v5_0.runtime.run_avatar_bootstrap.replay_prefix_traces_to_frontier", return_value={"frontier_reached": False, "divergence": False, "session": None})
    @patch("v5_0.runtime.run_avatar_bootstrap.get_frontier_level_id", side_effect=["L1", None, None])
    @patch("v5_0.runtime.run_avatar_bootstrap.load_or_initialize_campaign_state")
    @patch("v5_0.runtime.run_avatar_bootstrap.get_level_sequence_for_game", return_value=("L0", "L1"))
    @patch("v5_0.runtime.run_avatar_bootstrap.initialize_trace_store")
    def test_prefix_traces_used_written_before_replay(
        self,
        init_store,
        _seq,
        load_state,
        _frontier,
        _replay,
        _prefix,
        _current,
        _db,
        _rebuild,
        _index,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            init_store.return_value = str(Path(tmp) / "trace.sqlite")
            load_state.return_value = self._state()
            run_full_campaign_analysis(game_id="ez01", output_dir=tmp, use_solutions=True)
            path = Path(tmp) / "ez01" / "L1" / "prefix_traces_used.json"
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("trace_source"), "global_db")
            self.assertEqual(payload.get("total_prefix_action_count"), 1)

    def test_final_video_builder_no_renderable_frames_for_action_only_use_solution_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ez01"
            level_dir = root / "L1"
            level_dir.mkdir(parents=True, exist_ok=True)
            (level_dir / "frontier_attempt.json").write_text(json.dumps({"level_id": "L1"}), encoding="utf-8")
            (level_dir / "prefix_traces_used.json").write_text(json.dumps({"prefix_level_ids": ["L0"]}), encoding="utf-8")
            (level_dir / "saved_level_trace_actions.json").write_text(json.dumps(["RIGHT", "DOWN"]), encoding="utf-8")
            out = build_final_game_video(root, fps=2)
            self.assertEqual(out.get("failure_reason"), "no_renderable_frames")
            self.assertEqual(int(out.get("frame_count", -1)), 0)

    def test_saved_level_trace_steps_is_frame_source_and_actions_file_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ez01"
            level_dir = root / "L1"
            level_dir.mkdir(parents=True, exist_ok=True)
            (level_dir / "saved_level_trace_actions.json").write_text(json.dumps(["RIGHT", "DOWN"]), encoding="utf-8")
            (level_dir / "saved_level_trace_steps.json").write_text(
                json.dumps(
                    [
                        {"action": "RIGHT", "pre_frame": [[1, 0], [0, 0]], "post_frame": [[1, 0], [0, 2]]},
                        {"action": "DOWN", "pre_frame": [[1, 0], [0, 2]], "post_frame": [[1, 0], [2, 2]]},
                    ]
                ),
                encoding="utf-8",
            )
            frames = collect_complete_run_frames(root)
            self.assertGreaterEqual(len(frames), 3)
            self.assertEqual(frames[0], ((1, 0), (0, 0)))
            self.assertEqual(frames[-1], ((1, 0), (2, 2)))

    def test_duplicate_visible_frames_are_not_collapsed_for_executed_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ez01"
            campaign_dir = root / "campaign"
            campaign_dir.mkdir(parents=True, exist_ok=True)
            # Two steps with identical post frames must still produce per-step frames.
            (campaign_dir / "campaign_step_trace.json").write_text(
                json.dumps(
                    [
                        {"action": "RIGHT", "pre_frame": [[1]], "post_frame": [[1]]},
                        {"action": "RIGHT", "pre_frame": [[1]], "post_frame": [[1]]},
                        {"action": "DOWN", "pre_frame": [[1]], "post_frame": [[2]]},
                    ]
                ),
                encoding="utf-8",
            )
            frames = collect_complete_run_frames(root)
            self.assertEqual(len(frames), 4)
            self.assertEqual(frames[0], ((1,),))
            self.assertEqual(frames[1], ((1,),))
            self.assertEqual(frames[2], ((1,),))
            self.assertEqual(frames[3], ((2,),))

    @patch("v5_0.cli.build_final_game_video", return_value={"video_path": "/tmp/x.mp4", "frames_dir": "/tmp/frames", "frame_count": 0, "failure_reason": "no_renderable_frames"})
    def test_cli_surfaces_no_frame_bearing_explanation(self, _build):
        out = cli._maybe_build_video_artifacts(
            enabled=True,
            debug_all=False,
            game_id="ez01",
            output_dir="/tmp/runs_custom",
            artifact_paths=None,
        )
        self.assertEqual(out.get("final_video_failure_reason"), "no_renderable_frames")
        self.assertEqual(out.get("final_video_explanation"), "no_frame_bearing_artifacts_found_under_game_root")


if __name__ == "__main__":
    unittest.main()
