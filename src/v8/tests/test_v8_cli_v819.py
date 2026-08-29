from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import v8
from v8 import cli_v819


class AdaptiveAllocationCliV819Tests(unittest.TestCase):
    def test_allocation_options_are_applied_only_during_delegated_cli_call(self) -> None:
        env_name = "ARC_AGI3_V8_ALLOCATION_LEASE_STEPS"
        prior = os.environ.pop(env_name, None)
        observed = {}
        try:
            def delegated(argv):
                observed["argv"] = list(argv)
                observed["lease"] = os.environ.get(env_name)
                observed["plateau"] = os.environ.get(
                    "ARC_AGI3_V8_PLATEAU_PRIORITY_ENABLED"
                )
                return 0

            with patch("v8.cli.main", delegated):
                result = cli_v819.main(
                    [
                        "continuous-run",
                        "--games",
                        "learning",
                        "--allocation-lease-steps",
                        "512",
                        "--allocation-plateau-priority",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                observed["argv"],
                ["continuous-run", "--games", "learning"],
            )
            self.assertEqual(observed["lease"], "512")
            self.assertEqual(observed["plateau"], "1")
            self.assertNotIn(env_name, os.environ)
            self.assertNotIn("ARC_AGI3_V8_PLATEAU_PRIORITY_ENABLED", os.environ)
        finally:
            if prior is not None:
                os.environ[env_name] = prior

    def test_invalid_allocation_weight_is_rejected_before_base_cli(self) -> None:
        with patch("v8.cli.main") as delegated:
            with self.assertRaises(ValueError):
                cli_v819.main(
                    [
                        "continuous-run",
                        "--games",
                        "learning",
                        "--allocation-stable-weight",
                        "0",
                    ]
                )
            delegated.assert_not_called()

    def test_single_game_thirty_actor_budget_preserves_all_ten_thousand_steps(self) -> None:
        observed = {}

        def delegated(_argv):
            from v8 import cli as base_cli

            jobs = base_cli._actor_jobs(
                ("gp03",),
                actors=30,
                steps_per_game=10000,
                seed=0,
                env_root=None,
                epsilon=0.10,
                graph_check_steps=1000,
            )
            observed["jobs"] = tuple(jobs)
            observed["reported_len"] = len(jobs)
            observed["total"] = sum(int(job.steps) for job in jobs)
            return 0

        stdout = io.StringIO()
        with redirect_stdout(stdout), patch("v8.cli.main", delegated):
            result = cli_v819.main(
                [
                    "continuous-run",
                    "--games",
                    "gp03",
                    "--steps-per-game",
                    "10000",
                    "--actors",
                    "30",
                    "--graph-check",
                    "1000",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(observed["reported_len"], 30)
        self.assertEqual(len(observed["jobs"]), 30)
        self.assertEqual(observed["total"], 10000)
        self.assertIn("steps_per_game=10000", stdout.getvalue())
        self.assertIn("graph_check_interval=1000steps", stdout.getvalue())

    def test_requested_run_budget_rejects_non_positive_steps(self) -> None:
        with self.assertRaises(ValueError):
            cli_v819._requested_run_budget(
                ["continuous-run", "--steps-per-game", "0"]
            )


if __name__ == "__main__":
    unittest.main()
