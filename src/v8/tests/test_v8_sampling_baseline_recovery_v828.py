from __future__ import annotations

import os
import unittest

import v8
from v8 import cli_v819
from v8 import decision_point_sampling_v821 as sampling
from v8.sampling_baseline_recovery_v828 import (
    _decision_mode_enabled_v828,
    _requested_actor_pool_v828,
)


class SamplingBaselineRecoveryV828Tests(unittest.TestCase):
    def test_production_discovery_bypasses_v821_decision_sampler(self) -> None:
        prior = os.environ.get(sampling._SAMPLING_MODE_ENV)
        try:
            os.environ[sampling._SAMPLING_MODE_ENV] = "DISCOVERY"
            self.assertIs(sampling._decision_mode_enabled, _decision_mode_enabled_v828)
            self.assertFalse(sampling._decision_mode_enabled())
        finally:
            if prior is None:
                os.environ.pop(sampling._SAMPLING_MODE_ENV, None)
            else:
                os.environ[sampling._SAMPLING_MODE_ENV] = prior

    def test_v821_actor_wrapper_falls_through_to_pre_v821_worker(self) -> None:
        calls = []
        prior_base = sampling._BASE_ACTOR_WORKER
        sampling._BASE_ACTOR_WORKER = lambda **kwargs: calls.append(kwargs) or "baseline"
        try:
            result = sampling._actor_worker_v821(job="job", marker=7)
        finally:
            sampling._BASE_ACTOR_WORKER = prior_base
        self.assertEqual(result, "baseline")
        self.assertEqual(calls, [{"job": "job", "marker": 7}])

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
