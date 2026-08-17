from __future__ import annotations

import os
import unittest

import v8
from v8 import actor as actor_module
from v8 import cli_v819
from v8 import decision_point_sampling_v821 as sampling
from v8 import progress_runtime_fix_v822 as progress
from v8 import sampling_baseline_recovery_v828 as baseline
from v8.sampling_baseline_recovery_v828 import _requested_actor_pool_v828


class SamplingBaselineRecoveryV828Tests(unittest.TestCase):
    def test_production_discovery_bypasses_v821_beneath_final_v822_wrapper(self) -> None:
        prior_mode = os.environ.get(sampling._SAMPLING_MODE_ENV)
        prior_base = sampling._BASE_ACTOR_WORKER
        calls = []
        sampling._BASE_ACTOR_WORKER = lambda **kwargs: calls.append(kwargs) or "baseline"
        try:
            os.environ[sampling._SAMPLING_MODE_ENV] = "DISCOVERY"
            self.assertTrue(sampling._decision_mode_enabled())
            self.assertIs(actor_module.actor_worker, progress._actor_worker_with_solve_metrics_v822)
            self.assertIs(progress._BASE_ACTOR_WORKER, baseline._actor_delegate_v828)
            result = progress._BASE_ACTOR_WORKER(job="job", marker=7)
        finally:
            sampling._BASE_ACTOR_WORKER = prior_base
            if prior_mode is None:
                os.environ.pop(sampling._SAMPLING_MODE_ENV, None)
            else:
                os.environ[sampling._SAMPLING_MODE_ENV] = prior_mode
        self.assertEqual(result, "baseline")
        self.assertEqual(calls, [{"job": "job", "marker": 7}])

    def test_non_discovery_keeps_prior_composed_actor_delegate(self) -> None:
        prior_mode = os.environ.get(sampling._SAMPLING_MODE_ENV)
        prior_base = baseline._BASE_ACTOR_DELEGATE
        calls = []
        baseline._BASE_ACTOR_DELEGATE = lambda **kwargs: calls.append(kwargs) or "composed"
        try:
            os.environ[sampling._SAMPLING_MODE_ENV] = "VERIFY"
            result = baseline._actor_delegate_v828(job="job", marker=9)
        finally:
            baseline._BASE_ACTOR_DELEGATE = prior_base
            if prior_mode is None:
                os.environ.pop(sampling._SAMPLING_MODE_ENV, None)
            else:
                os.environ[sampling._SAMPLING_MODE_ENV] = prior_mode
        self.assertEqual(result, "composed")
        self.assertEqual(calls, [{"job": "job", "marker": 9}])

    def test_learning_set_restores_one_worker_per_game_by_default(self) -> None:
        self.assertIs(cli_v819._requested_actor_pool, _requested_actor_pool_v828)
        self.assertEqual(
            cli_v819._requested_actor_pool(["continuous-run", "--games", "learning"]),
            36,
        )
        self.assertEqual(
            cli_v819._requested_actor_pool(
                ["continuous-run", "--games", "learning", "--actors", "8"]
            ),
            36,
        )

    def test_explicit_actor_count_above_game_count_still_expands_lanes(self) -> None:
        self.assertEqual(
            cli_v819._requested_actor_pool(
                ["continuous-run", "--games", "diverse", "--actors", "20"]
            ),
            20,
        )

    def test_actor_batch_reports_all_learning_workers_after_restored_pool_resolution(self) -> None:
        pool = cli_v819._requested_actor_pool(["continuous-run", "--games", "learning"])
        batch = cli_v819._ActorJobBatch(tuple(range(36)), pool)
        self.assertEqual(len(batch), 36)
        self.assertEqual(tuple(batch), tuple(range(36)))


if __name__ == "__main__":
    unittest.main()
