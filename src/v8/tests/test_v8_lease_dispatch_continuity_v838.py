from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v8 import actor as actor_module
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import lease_dispatch_continuity_v838 as v838
from v8 import trajectory_optimizer_v818 as v818
from v8.model import MemoryUid
from v8.runtime_v82 import V82ContinuousMemoryRuntime


class LeaseDispatchContinuityV838Tests(unittest.TestCase):
    def test_final_runtime_hooks_are_installed(self) -> None:
        self.assertIs(actor_module.run_actor_jobs, v838._run_actor_jobs_v838)
        self.assertIs(V82ContinuousMemoryRuntime.record_actor_results, v838._record_actor_results_v838)
        self.assertIs(v819._retry_deferred_sources, v838._retry_deferred_v838)

    def test_actor_feedback_submit_does_not_wait_for_consumer(self) -> None:
        started = threading.Event()
        release = threading.Event()
        consumed = []
        runtime = SimpleNamespace(_error_queue=queue.Queue())

        def consume(_runtime, rows) -> None:
            started.set()
            release.wait(1.0)
            consumed.append(tuple(rows))

        service = v838._ActorFeedbackService(runtime, consume)
        try:
            service.submit(("batch",))
            self.assertTrue(started.wait(1.0))
            self.assertFalse(release.is_set())
            submitted, completed, pending, _max_pending = service.metrics()
            self.assertEqual(submitted, 1)
            self.assertEqual(completed, 0)
            self.assertGreaterEqual(pending, 1)
            release.set()
            service.flush(timeout=1.0)
            self.assertEqual(consumed, [("batch",)])
        finally:
            release.set()
            service.close(timeout=1.0)

    def test_progress_callback_submit_does_not_run_inline(self) -> None:
        started = threading.Event()
        release = threading.Event()
        consumed = []

        def callback(rows) -> None:
            started.set()
            release.wait(1.0)
            consumed.append(tuple(rows))

        service = v838._ProgressCallbackService(callback)
        try:
            service.submit(("progress",))
            self.assertTrue(started.wait(1.0))
            self.assertFalse(release.is_set())
            release.set()
            service.flush(timeout=1.0)
            self.assertEqual(consumed, [("progress",)])
        finally:
            release.set()
            service.close(timeout=1.0)

    def test_deferred_retry_is_bounded_per_pass(self) -> None:
        pending = [(object(), object()) for _ in range(6)]
        runtime = SimpleNamespace(_v819_deferred_sources=pending)
        zero = MemoryUid.zero()

        with patch.object(v818, "_resolve_target_outcome", return_value=zero) as resolve:
            with patch.object(v819, "_publish_validated_source") as publish:
                examined, resolved = v838._retry_deferred_batch(runtime, limit=4)

        self.assertEqual(examined, 4)
        self.assertEqual(resolved, 0)
        self.assertEqual(resolve.call_count, 4)
        publish.assert_not_called()
        self.assertEqual(len(runtime._v819_deferred_sources), 6)


if __name__ == "__main__":
    unittest.main()
