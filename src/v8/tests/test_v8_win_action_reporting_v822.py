from __future__ import annotations

import os
import unittest

import v8
from v8 import actor as actor_module
from v8 import behavior_recovery as behavior
from v8 import decision_point_sampling_v821 as sampling
from v8 import learning_fixes_v088 as learning
from v8 import progress_runtime_fix_v822 as progress


class WinActionReportingV822Tests(unittest.TestCase):
    def test_non_discovery_lease_enables_and_resets_solve_metrics(self) -> None:
        prior_worker = progress._BASE_ACTOR_WORKER
        prior_sampling = os.environ.get(sampling._SAMPLING_MODE_ENV)
        prior_actor_mode = os.environ.get(behavior._ACTOR_MODE_ENV)
        prior_learning_mode = learning._ACTOR_MODE_ENV
        prior_metric_env = os.environ.get(progress._SOLVE_METRICS_ENV)
        prior_metrics = (
            learning._EPISODE_STEPS,
            learning._FIRST_WIN_STEPS,
            learning._BEST_WIN_STEPS,
            learning._LAST_WIN_STEPS,
        )
        observed: dict[str, object] = {}

        def fake_worker(*, job, **kwargs):
            observed["actor_mode"] = os.environ.get(behavior._ACTOR_MODE_ENV)
            observed["metric_env_name"] = learning._ACTOR_MODE_ENV
            observed["metric_enabled"] = os.environ.get(learning._ACTOR_MODE_ENV)
            observed["best_before"] = learning._BEST_WIN_STEPS
            learning._BEST_WIN_STEPS = 9
            return job.game_id

        progress._BASE_ACTOR_WORKER = fake_worker
        os.environ[sampling._SAMPLING_MODE_ENV] = "VERIFY"
        os.environ.pop(behavior._ACTOR_MODE_ENV, None)
        os.environ.pop(progress._SOLVE_METRICS_ENV, None)
        learning._EPISODE_STEPS = 11
        learning._FIRST_WIN_STEPS = 12
        learning._BEST_WIN_STEPS = 13
        learning._LAST_WIN_STEPS = 14
        try:
            result = progress._actor_worker_with_solve_metrics_v822(
                job=actor_module.ActorJob(3, "ic02", 100, 7)
            )
            self.assertEqual(result, "ic02")
            self.assertIsNone(observed["actor_mode"])
            self.assertEqual(observed["metric_env_name"], progress._SOLVE_METRICS_ENV)
            self.assertEqual(observed["metric_enabled"], "1")
            self.assertEqual(observed["best_before"], 0)
            self.assertEqual(learning._ACTOR_MODE_ENV, prior_learning_mode)
            self.assertNotIn(progress._SOLVE_METRICS_ENV, os.environ)
        finally:
            progress._BASE_ACTOR_WORKER = prior_worker
            learning._ACTOR_MODE_ENV = prior_learning_mode
            if prior_sampling is None:
                os.environ.pop(sampling._SAMPLING_MODE_ENV, None)
            else:
                os.environ[sampling._SAMPLING_MODE_ENV] = prior_sampling
            if prior_actor_mode is None:
                os.environ.pop(behavior._ACTOR_MODE_ENV, None)
            else:
                os.environ[behavior._ACTOR_MODE_ENV] = prior_actor_mode
            if prior_metric_env is None:
                os.environ.pop(progress._SOLVE_METRICS_ENV, None)
            else:
                os.environ[progress._SOLVE_METRICS_ENV] = prior_metric_env
            (
                learning._EPISODE_STEPS,
                learning._FIRST_WIN_STEPS,
                learning._BEST_WIN_STEPS,
                learning._LAST_WIN_STEPS,
            ) = prior_metrics

    def test_solve_metric_wrapper_is_final_actor_authority(self) -> None:
        from v8 import memory_efficiency_v851 as memory_efficiency

        self.assertIs(actor_module.actor_worker, memory_efficiency._actor_worker_v851)
        self.assertIs(memory_efficiency._BASE_ACTOR_WORKER, progress._actor_worker_with_solve_metrics_v822)


if __name__ == "__main__":
    unittest.main()
