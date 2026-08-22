from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import v8
from v7.environment.arc_adapter import ArcGridEnvironment
from v8 import action_learning_report_v849 as report
from v8 import action_learning_report_v849_fixups as fixups
from v8 import action_learning_report_v849_integrity as integrity
from v8 import click_exploration_v848 as click
from v8 import plateau_progress_v846 as progress
from v8 import runtime_repair_v822 as v822
from v8 import solved_game_recovery_v821 as recovery
from v8.adaptive_learning_allocation_v819 import AdaptiveLearningCoordinator
from v8.learning_blockers_v055 import pack_action_choice


class _Raw:
    def __init__(self, frame, *, actions=(1, 6), levels=0, state="NOT_FINISHED"):
        self.frame = np.asarray(frame, dtype=np.int64)
        self.available_actions = list(actions)
        self.levels_completed = int(levels)
        self.state = state


class _Engine:
    def __init__(self):
        self.grid = np.zeros((3, 3), dtype=np.int64)
        self.levels = 0
        self.calls = []

    def reset(self):
        return _Raw(self.grid.copy(), levels=self.levels)

    @staticmethod
    def _action_id(action):
        try:
            return int(action)
        except (TypeError, ValueError):
            value = getattr(action, "value", None)
            if value is not None:
                return int(value)
            digits = "".join(ch for ch in str(action) if ch.isdigit())
            if digits:
                return int(digits)
            raise

    def step(self, action, data=None):
        action_id = self._action_id(action)
        self.calls.append((action_id, data))
        if action_id == 6 and data == {"x": 1, "y": 1}:
            self.grid[1, 1] = 7
        return _Raw(self.grid.copy(), levels=self.levels)


class ActionLearningAuthorityTests(unittest.TestCase):
    def test_historical_environment_authorities_are_preserved(self):
        self.assertIs(ArcGridEnvironment.step, v822._runtime_env_step)
        self.assertIs(ArcGridEnvironment.reset, v822._runtime_env_reset)
        self.assertIs(v822._BASE_ENV_STEP, recovery._tracked_env_step)
        self.assertIs(v822._BASE_ENV_RESET, recovery._tracked_env_reset)

    def test_v849_is_installed_after_v848_without_environment_probe(self):
        from v8 import runtime_stack_v88

        self.assertEqual(runtime_stack_v88._LAYERS[-1], "click_exploration_v848")
        self.assertIn("action_learning_report_v849", runtime_stack_v88._POST_LAYERS)
        self.assertTrue(report._INSTALLED)
        self.assertTrue(fixups._INSTALLED)
        self.assertTrue(integrity._INSTALLED)
        self.assertIs(click._probe_game_action_space, report._probe_game_action_space_v849)


class ActionLearningCollectionTests(unittest.TestCase):
    def setUp(self):
        self.prior_root = os.environ.get(report._TRAJECTORY_ROOT_ENV)
        report._reset_action_learning_state_v849()

    def tearDown(self):
        if self.prior_root is None:
            os.environ.pop(report._TRAJECTORY_ROOT_ENV, None)
        else:
            os.environ[report._TRAJECTORY_ROOT_ENV] = self.prior_root
        report._reset_action_learning_state_v849()

    def test_real_adapter_records_click_noop_productive_target_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[report._TRAJECTORY_ROOT_ENV] = str(Path(tmp) / "trajectory_optimizer")
            report._reset_action_learning_state_v849()
            engine = _Engine()
            env = ArcGridEnvironment(
                game_id="click-fixture",
                env_factory=lambda **_: engine,
            )
            actions = set(env.available_actions())
            noop = pack_action_choice(6, 0, 0)
            productive = pack_action_choice(6, 1, 1)
            self.assertIn(noop, actions)
            self.assertIn(productive, actions)

            env.step(noop)
            env.step(productive)
            env.reset()
            report._refresh_events(force=True)

            row = report._RUN["click-fixture"]
            self.assertEqual(int(row["click_actions_executed"]), 2)
            self.assertEqual(int(row["click_noops"]), 1)
            self.assertEqual(int(row["click_productive"]), 1)
            self.assertEqual(len(row["exact_click_targets_tested"]), 2)
            self.assertEqual(len(row["productive_click_targets"]), 1)
            self.assertEqual(int(row["grid_coordinate_capacity"]), 9)
            self.assertIn(6, row["native_types"])

    def test_mixed_episode_retains_earlier_productive_click_evidence(self):
        env = type("Metrics", (), {})()
        report._ensure_env_metrics(env)
        env._v849_last_action_token = pack_action_choice(6, 1, 1)
        report._record_episode_kind(
            env,
            "click",
            productive=True,
            level_advanced=False,
        )
        report._record_episode_kind(
            env,
            "movement",
            productive=False,
            level_advanced=False,
        )
        self.assertEqual(int(env._v849_metrics["mixed_sequences_observed"]), 1)
        self.assertEqual(int(env._v849_metrics["mixed_sequences_productive"]), 1)


class ActionLearningAggregationTests(unittest.TestCase):
    def setUp(self):
        report._reset_action_learning_state_v849()
        progress._reset_progress_depth_v846()

    def tearDown(self):
        report._reset_action_learning_state_v849()
        progress._reset_progress_depth_v846()

    @staticmethod
    def _space(*, native, click_available=(), movement=(), branching=0):
        row = report._empty_aggregate()
        row["native_types"].update(native)
        row["click_targets_available"].update(click_available)
        row["movement_actions_available"].update(movement)
        row["max_branching"] = int(branching)
        return row

    def test_click_branching_multiplier_uses_observed_nonclick_reference(self):
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(("move", "click"))
        coordinator._v848_action_spaces["move"] = (False, 4)
        coordinator._v848_action_spaces["click"] = (True, 100)
        self.assertAlmostEqual(
            report._click_complexity_multiplier_v849(coordinator, "click"),
            5.0,
        )

    def test_sampling_complexity_uses_cached_space_without_event_refresh(self):
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(("move", "click"))
        coordinator._v848_action_spaces["move"] = (False, 4)
        coordinator._v848_action_spaces["click"] = (True, 100)

        with patch.object(report, "_refresh_events") as refresh_events:
            multiplier = click._click_complexity_multiplier(coordinator, "click")

        self.assertAlmostEqual(multiplier, 5.0)
        refresh_events.assert_not_called()

    def test_summary_exposes_click_and_mixed_level_metrics(self):
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(("move", "click", "mixed"))
        report._SPACE["move"] = self._space(native={1, 2, 3, 4}, movement={1, 2, 3, 4}, branching=4)
        report._SPACE["click"] = self._space(native={6}, click_available={10, 11}, branching=64)
        report._SPACE["mixed"] = self._space(native={1, 2, 6}, click_available={20, 21}, movement={1, 2}, branching=66)
        report._RUN["move"] = self._space(native={1, 2, 3, 4}, movement={1, 2, 3, 4}, branching=4)
        report._RUN["click"] = self._space(native={6}, click_available={10, 11}, branching=64)
        report._RUN["mixed"] = self._space(native={1, 2, 6}, click_available={20, 21}, movement={1, 2}, branching=66)
        report._RUN["click"]["grid_coordinate_capacity"] = 10
        report._RUN["click"]["exact_click_targets_tested"].update({100, 101})
        report._RUN["click"]["click_targets_tested"].update({100, 101, 102})
        report._RUN["mixed"]["grid_coordinate_capacity"] = 10
        report._RUN["mixed"]["exact_click_targets_tested"].update({200, 201, 202})
        report._RUN["mixed"]["click_targets_tested"].update({200, 201, 202, 203})
        report._RUN["mixed"]["productive_click_targets"].update({200, 202})
        progress._MAX_LEVEL_REACHED.update({"click": 3, "mixed": 2})
        coordinator._game_won["click"] = False
        coordinator._game_won["mixed"] = True

        payload = report.action_learning_snapshot_v849(coordinator)
        summary = payload["summary"]
        self.assertEqual(summary["click_capable_games"], 2)
        self.assertEqual(summary["click_games"], 1)
        self.assertEqual(summary["mixed_games"], 1)
        self.assertEqual(summary["click_levels_solved"], 8)
        self.assertEqual(summary["click_levels_total"], 10)
        self.assertEqual(summary["mixed_levels_solved"], 5)
        self.assertEqual(summary["mixed_levels_total"], 5)
        self.assertEqual(summary["click_games_solved"], 1)
        self.assertAlmostEqual(summary["click_coverage_pct"], 25.0)
        mixed = next(row for row in payload["games"] if row["game_id"] == "mixed")
        self.assertEqual(mixed["exact_click_targets_tested"], 3)
        self.assertEqual(mixed["unique_productive_click_targets"], 2)

    def test_snapshot_refreshes_actor_events_once_for_all_games(self):
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(tuple(f"game-{index}" for index in range(36)))

        with patch.object(report, "_refresh_events") as refresh_events:
            payload = report.action_learning_snapshot_v849(coordinator)

        self.assertEqual(len(payload["games"]), 36)
        refresh_events.assert_called_once_with(force=True)

    def test_action_learning_log_contains_per_game_blocker_metrics(self):
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(("click",))
        report._SPACE["click"] = self._space(native={6}, click_available={10}, branching=64)
        report._RUN["click"] = self._space(native={6}, click_available={10}, branching=64)
        runtime = type("Runtime", (), {"generation": 7})()
        with tempfile.TemporaryDirectory() as tmp:
            runtime.root = tmp
            report._write_action_learning_log(runtime, coordinator)
            path = Path(tmp) / "action_learning.log"
            payload = json.loads(path.read_text(encoding="utf-8").strip())
        game = payload["games"][0]
        required = {
            "action_space_type",
            "click_target_coverage_pct",
            "click_actions_executed",
            "click_noops",
            "click_productive",
            "click_level_advances",
            "click_wins",
            "productive_click_rate",
            "click_revisit_rate",
            "click_frontier_nodes",
            "click_frontier_expandable",
            "suppressed_click_noop_frontiers",
            "mixed_sequences_observed",
            "mixed_sequences_productive",
            "mixed_sequences_level_advancing",
            "allocation_steps",
            "allocation_share",
            "click_complexity_multiplier",
        }
        self.assertTrue(required.issubset(game))
        self.assertIn("click_levels_solved", payload["summary"])


class ActionEventRefreshSchedulingTests(unittest.TestCase):
    def setUp(self):
        report._reset_action_learning_state_v849()

    def tearDown(self):
        report._reset_action_learning_state_v849()

    @staticmethod
    def _event(game: str = "g") -> dict[str, object]:
        return {
            "schema": 1,
            "time": time.time() + 1.0,
            "game_id": game,
            "steps": 1,
        }

    def test_unchanged_files_are_not_reopened(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_root = Path(tmp) / "trajectory_optimizer"
            event_root = Path(tmp) / report._EVENT_DIR
            event_root.mkdir()
            path = event_root / "actor-1.jsonl"
            path.write_text(json.dumps(self._event()) + "\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {report._TRAJECTORY_ROOT_ENV: str(trajectory_root)},
            ):
                report._refresh_events(force=True)
                with patch.object(
                    Path,
                    "open",
                    side_effect=AssertionError("unchanged file reopened"),
                ):
                    report._refresh_events(force=True)

    def test_bounded_refresh_resumes_from_saved_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_root = Path(tmp) / "trajectory_optimizer"
            event_root = Path(tmp) / report._EVENT_DIR
            event_root.mkdir()
            path = event_root / "actor-1.jsonl"
            path.write_text(
                "".join(json.dumps(self._event()) + "\n" for _ in range(3)),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {report._TRAJECTORY_ROOT_ENV: str(trajectory_root)},
                ),
                patch.object(report, "_REFRESH_MAX_EVENTS", 1),
                patch.object(report, "_REFRESH_MAX_SECONDS", 60.0),
            ):
                report._refresh_events(force=True)
                self.assertEqual(report._RUN["g"]["steps"], 1)
                report._refresh_events(force=True)
                self.assertEqual(report._RUN["g"]["steps"], 2)
                report._refresh_events(force=True)
                self.assertEqual(report._RUN["g"]["steps"], 3)

    def test_busy_actor_file_does_not_starve_other_actor_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_root = Path(tmp) / "trajectory_optimizer"
            event_root = Path(tmp) / report._EVENT_DIR
            event_root.mkdir()
            (event_root / "actor-1.jsonl").write_text(
                "".join(json.dumps(self._event("busy")) + "\n" for _ in range(4)),
                encoding="utf-8",
            )
            click_event = {
                **self._event("click"),
                "native_types": [6],
            }
            (event_root / "actor-2.jsonl").write_text(
                json.dumps(click_event) + "\n",
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {report._TRAJECTORY_ROOT_ENV: str(trajectory_root)},
                ),
                patch.object(report, "_REFRESH_MAX_EVENTS_PER_FILE", 1),
                patch.object(report, "_REFRESH_MAX_EVENTS", 10),
                patch.object(report, "_REFRESH_MAX_SECONDS", 60.0),
            ):
                report._refresh_events(force=True)
            self.assertEqual(report._RUN["busy"]["steps"], 1)
            self.assertEqual(report._RUN["click"]["steps"], 1)
            self.assertEqual(report._SPACE["click"]["native_types"], {6})

    def test_refresh_interval_begins_when_scan_finishes(self):
        with (
            patch.object(report, "_event_root", return_value=None),
            patch.object(report.time, "monotonic", side_effect=(10.0, 20.0)),
        ):
            report._refresh_events(force=True)
        self.assertEqual(report._LAST_REFRESH, 20.0)


class ActionFrontierReportSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.prior_root = os.environ.get(report._TRAJECTORY_ROOT_ENV)
        report._reset_action_learning_state_v849()

    def tearDown(self):
        if self.prior_root is None:
            os.environ.pop(report._TRAJECTORY_ROOT_ENV, None)
        else:
            os.environ[report._TRAJECTORY_ROOT_ENV] = self.prior_root
        report._reset_action_learning_state_v849()

    def test_movement_game_does_not_scan_click_frontier(self):
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(("movement",))
        report._SPACE["movement"] = {
            **report._empty_aggregate(),
            "native_types": {1, 2, 3, 4},
            "movement_actions_available": {1, 2, 3, 4},
        }

        with patch.object(
            report,
            "_frontier_metrics",
            side_effect=AssertionError("movement frontier scanned"),
        ):
            row = report._game_row(coordinator, "movement", refresh_events=False)

        self.assertEqual(row["click_frontier_nodes"], 0)
        self.assertEqual(row["click_frontier_expandable"], 0)

    def test_unchanged_frontier_file_is_not_reparsed(self):
        from v8 import sampling_evidence_frontier_v847 as frontier

        with tempfile.TemporaryDirectory() as tmp:
            trajectory_root = Path(tmp) / "trajectory_optimizer"
            frontier_root = trajectory_root / frontier._STATE_DIR
            frontier_root.mkdir(parents=True)
            path = frontier_root / f"{frontier._game_token('click')}-1.json"
            path.write_text(
                json.dumps(
                    {
                        "game_id": "click",
                        "nodes": [
                            {
                                "available_actions": [6],
                                "tried_actions": [],
                                "anchor": [6],
                                "latent": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            os.environ[report._TRAJECTORY_ROOT_ENV] = str(trajectory_root)

            first = report._frontier_metrics("click")
            with patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("unchanged frontier reparsed"),
            ):
                second = report._frontier_metrics("click")

        self.assertEqual(first, second)
        self.assertEqual(first["click_frontier_nodes"], 1)
        self.assertEqual(first["click_frontier_expandable"], 1)
        self.assertEqual(first["suppressed_click_noop_frontiers"], 1)

    def test_frontier_metrics_sidecar_avoids_large_node_payload(self):
        from v8 import sampling_evidence_frontier_v847 as frontier

        with tempfile.TemporaryDirectory() as tmp:
            trajectory_root = Path(tmp) / "trajectory_optimizer"
            frontier_root = trajectory_root / frontier._STATE_DIR
            frontier_root.mkdir(parents=True)
            path = frontier_root / f"{frontier._game_token('click')}-1.json"
            path.write_text("large payload must not be parsed", encoding="utf-8")
            path.with_suffix(".metrics").write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "game_id": "click",
                        "click_frontier_nodes": 11,
                        "click_frontier_expandable": 7,
                        "suppressed_click_noop_frontiers": 3,
                    }
                ),
                encoding="utf-8",
            )
            os.environ[report._TRAJECTORY_ROOT_ENV] = str(trajectory_root)

            metrics = report._frontier_metrics("click")

        self.assertEqual(metrics["click_frontier_nodes"], 11)
        self.assertEqual(metrics["click_frontier_expandable"], 7)
        self.assertEqual(metrics["suppressed_click_noop_frontiers"], 3)

    def test_snapshot_indexes_frontier_directory_once(self):
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(("click-a", "click-b"))
        for game in ("click-a", "click-b"):
            report._SPACE[game] = {
                **report._empty_aggregate(),
                "native_types": {6},
            }
        frontier_index = {}
        with (
            patch.object(
                report,
                "_frontier_file_index",
                return_value=frontier_index,
            ) as build_index,
            patch.object(
                report,
                "_frontier_metrics",
                return_value={
                    "click_frontier_nodes": 0,
                    "click_frontier_expandable": 0,
                    "suppressed_click_noop_frontiers": 0,
                },
            ) as metrics,
        ):
            report.action_learning_snapshot_v849(coordinator)

        build_index.assert_called_once_with()
        self.assertEqual(metrics.call_count, 2)
        for call in metrics.call_args_list:
            self.assertIs(call.kwargs["frontier_index"], frontier_index)


if __name__ == "__main__":
    unittest.main()
