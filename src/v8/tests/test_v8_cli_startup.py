from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from v8.cli import _graph_load_line, _restored_solved_games, run_continuous


class StartupGraphLineTests(unittest.TestCase):
    def test_snapshot_source_and_node_count_are_shown(self) -> None:
        line = _graph_load_line(
            snapshot_path=Path("runs/v8/continuous/snapshots/snapshot-00000000000000000042"),
            restore_enabled=True,
            nodes=24903,
        )
        self.assertEqual(
            line,
            "graph source=runs/v8/continuous/snapshots/snapshot-00000000000000000042 nodes=24903",
        )
        self.assertNotIn("\n", line)

    def test_empty_graph_without_snapshot_is_explicit(self) -> None:
        self.assertEqual(
            _graph_load_line(snapshot_path=None, restore_enabled=True, nodes=0),
            "graph source=empty(no-snapshot) nodes=0",
        )

    def test_no_restore_is_explicit(self) -> None:
        self.assertEqual(
            _graph_load_line(snapshot_path=None, restore_enabled=False, nodes=0),
            "graph source=empty(--no-restore) nodes=0",
        )

    def test_restored_solved_games_are_selected_for_startup_message(self) -> None:
        states = {
            "ez01": SimpleNamespace(value="SOLVED_OPTIMIZING"),
            "ls20": SimpleNamespace(value="SOLVED_STABLE"),
            "tt01": SimpleNamespace(value="UNSOLVED"),
        }
        runtime = SimpleNamespace(
            _v819_adaptive_learning=SimpleNamespace(
                game_state=lambda game: states[game]
            )
        )

        self.assertEqual(
            _restored_solved_games(runtime, ("ez01", "tt01", "ls20")),
            ("ez01", "ls20"),
        )

    def test_restored_solved_games_are_empty_without_coordinator(self) -> None:
        self.assertEqual(
            _restored_solved_games(SimpleNamespace(), ("ez01",)),
            (),
        )


class HypothesisStartupDelayTests(unittest.TestCase):
    def test_reporting_is_delegated_without_startup_hypothesis_evaluation(self) -> None:
        args = SimpleNamespace(
            games="tt01",
            actors=1,
            steps_per_game=1,
            seed=0,
            env_root=None,
            epsilon=0.1,
            no_restore=True,
            root="unused",
            shards=1,
            stage_workers=1,
            stage_ring_capacity=16,
            shard_ring_capacity=16,
            node_capacity_per_shard=16,
            edge_capacity_per_shard=16,
            action_capacity_per_shard=16,
            snapshot_interval_seconds=60.0,
            no_snapshots=True,
            no_peers=True,
            peer_interval_seconds=0.5,
            wait=0.0,
            actor_timeout=None,
            progress_interval_seconds=60.0,
            drain_timeout=1.0,
            final_save_timeout=1.0,
            no_automatic_experiments=True,
            max_transfer_experiments=0,
            transfer_experiment_steps=1,
        )
        runtime = Mock()
        runtime.read_view.memory_count = 0
        runtime.peers = None
        runtime.metrics.return_value = {}
        runtime.scientific_statuses.return_value = {
            f"H{index:02d}": "INSUFFICIENT_EVIDENCE"
            for index in range(1, 16)
        }
        runtime.close.return_value = None
        reporter = Mock()
        reporter.progress_queue = object()
        lifecycle: list[str] = []
        reporter.close.side_effect = lambda: lifecycle.append("reporter stopped")
        runtime.wait_quiescent.side_effect = lambda **_kwargs: lifecycle.append("runtime drained")
        runtime.metrics.side_effect = lambda: lifecycle.append("metrics") or {}

        def run_jobs(_runtime, _jobs, **kwargs):
            self.assertEqual(runtime.scientific_statuses.call_count, 0)
            kwargs["progress_callback"](())
            self.assertEqual(runtime.scientific_statuses.call_count, 0)
            self.assertIs(kwargs["reporting_queue"], reporter.progress_queue)
            return ()

        with (
            patch("v7.game_sets.resolve_game_selector", return_value=("tt01",)),
            patch("v8.cli._runtime_config"),
            patch("v8.cli.ContinuousMemoryRuntime", return_value=runtime),
            patch("v8.cli.DedicatedReporter", return_value=reporter) as reporter_type,
            patch("v8.cli.run_actor_jobs", side_effect=run_jobs),
            patch("v8.cli._log") as log,
            patch("v8.cli.Path.write_text"),
        ):
            self.assertEqual(run_continuous(args), 0)

        reporter_type.assert_called_once()
        self.assertEqual(reporter_type.call_args.kwargs["interval_seconds"], 60.0)
        self.assertEqual(reporter_type.call_args.kwargs["total_steps"], 1)
        reporter.start.assert_called_once_with()
        reporter.close.assert_called_once_with()
        self.assertEqual(
            lifecycle,
            ["reporter stopped", "runtime drained", "metrics"],
        )
        log.assert_any_call("sampling done")
        self.assertEqual(runtime.scientific_statuses.call_count, 1)


if __name__ == "__main__":
    unittest.main()
