from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from v8 import actor as actor_module
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import lease_dispatch_continuity_v839 as v839
from v8 import trajectory_optimizer_v818 as v818
from v8.model import MemoryUid
from v8.runtime_v82 import V82ContinuousMemoryRuntime


class LeaseDispatchContinuityV839Tests(unittest.TestCase):
    def test_final_runtime_hooks_are_installed(self) -> None:
        self.assertIs(actor_module.run_actor_jobs, v839._run_actor_jobs_v839)
        self.assertIs(
            V82ContinuousMemoryRuntime.record_actor_results,
            v839._record_actor_results_v839,
        )
        self.assertIs(v819._retry_deferred_sources, v839._retry_deferred_v839)

    def test_async_worker_submit_does_not_wait_for_callback(self) -> None:
        started = threading.Event()
        release = threading.Event()
        consumed = []

        def callback(value) -> None:
            started.set()
            release.wait(1.0)
            consumed.append(value)

        worker = v839._AsyncQueueWorker(callback, name="test-maintenance")
        try:
            worker.submit("batch")
            self.assertTrue(started.wait(1.0))
            self.assertFalse(release.is_set())
            submitted, completed, pending, _max_pending = worker.metrics()
            self.assertEqual(submitted, 1)
            self.assertEqual(completed, 0)
            self.assertGreaterEqual(pending, 1)
            release.set()
            worker.flush(timeout=1.0)
            self.assertEqual(consumed, ["batch"])
        finally:
            release.set()
            worker.close(timeout=1.0)

    def test_progress_worker_coalesces_stale_pending_callbacks(self) -> None:
        started = threading.Event()
        release = threading.Event()
        consumed = []

        def callback(value) -> None:
            started.set()
            if value == "first":
                release.wait(1.0)
            consumed.append(value)

        worker = v839._AsyncQueueWorker(
            callback,
            name="test-coalesced-maintenance",
            coalesce_pending=True,
        )
        try:
            worker.submit("first")
            self.assertTrue(started.wait(1.0))
            worker.submit("stale-1")
            worker.submit("stale-2")
            worker.submit("latest")
            release.set()
            worker.flush(timeout=1.0)
            self.assertEqual(consumed, ["first", "latest"])
        finally:
            release.set()
            worker.close(timeout=1.0)

    def test_feedback_path_queues_during_sampling(self) -> None:
        consumed = []
        error_queue = queue.Queue()
        runtime = SimpleNamespace(
            _v839_sampling_active=True,
            _error_queue=error_queue,
        )

        def base_record(_runtime, rows) -> None:
            consumed.append(tuple(rows))

        with patch.object(v839, "_BASE_RECORD_ACTOR_RESULTS", base_record):
            v839._record_actor_results_v839(runtime, ("learning",))
            worker = runtime._v839_actor_feedback
            worker.flush(timeout=1.0)
            worker.close(timeout=1.0)

        self.assertEqual(consumed, [("learning",)])

    def test_sampling_completion_is_reported_before_maintenance_cleanup(self) -> None:
        reporting = queue.Queue()
        runtime = SimpleNamespace(_v839_sampling_active=False)

        with patch.object(v839, "_BASE_RUN_ACTOR_JOBS", return_value=("done",)), patch.object(
            v839, "_BASE_RETRY_DEFERRED"
        ):
            result = v839._run_actor_jobs_v839(
                runtime,
                (),
                reporting_queue=reporting,
            )

        from v8.reporter import SAMPLING_COMPLETE

        self.assertEqual(result, ("done",))
        self.assertEqual(reporting.get_nowait(), SAMPLING_COMPLETE)
        self.assertIs(runtime._v839_sampling_done_reported, True)

    def test_feedback_drain_pauses_peer_cycles_and_waits_for_idle(self) -> None:
        events = []
        reporting = queue.Queue()
        feedback = SimpleNamespace(flush=lambda: events.append("feedback-flush"))

        class Peers:
            _pause = threading.Event()

            def pause(self):
                events.append("peers-pause")
                self._pause.set()

            def wait_idle(self, timeout):
                self.assert_timeout = timeout
                events.append("peers-idle")
                return True

            def resume(self):
                events.append("peers-resume")
                self._pause.clear()

        peers = Peers()
        runtime = SimpleNamespace(
            _v839_sampling_active=False,
            _v839_actor_feedback=feedback,
            peers=peers,
            _v814_trajectory_optimizer=SimpleNamespace(),
        )

        with patch.object(v839, "_BASE_RUN_ACTOR_JOBS", return_value=("done",)), patch.object(
            v839, "_BASE_RETRY_DEFERRED"
        ):
            result = v839._run_actor_jobs_v839(
                runtime,
                (),
                reporting_queue=reporting,
            )

        self.assertEqual(result, ("done",))
        self.assertEqual(
            events,
            ["peers-pause", "peers-idle", "feedback-flush"],
        )
        self.assertEqual(peers.assert_timeout, v839._DRAIN_TIMEOUT_SECONDS)
        self.assertTrue(peers._pause.is_set())
        self.assertTrue(runtime._sampling_complete)
        self.assertTrue(
            runtime._v814_trajectory_optimizer._v841_preserve_inbox_on_shutdown
        )

    def test_abnormal_runtime_close_aborts_feedback_without_second_drain(self) -> None:
        feedback = SimpleNamespace(abort=Mock(), close=Mock())
        runtime = SimpleNamespace(
            _v839_actor_feedback=feedback,
            _v839_deferred_retry=None,
        )

        with patch.object(v839, "_BASE_RUNTIME_CLOSE", return_value="closed") as close:
            result = v839._runtime_close_v839(runtime, normal=False)

        self.assertEqual(result, "closed")
        feedback.abort.assert_called_once_with()
        feedback.close.assert_not_called()
        close.assert_called_once_with(runtime, normal=False)
        self.assertIsNone(runtime._v839_actor_feedback)

    def test_deferred_retry_is_bounded_per_pass(self) -> None:
        pending = [(object(), object()) for _ in range(6)]
        runtime = SimpleNamespace(_v819_deferred_sources=pending)
        zero = MemoryUid.zero()

        with patch.object(v818, "_resolve_target_outcome", return_value=zero) as resolve:
            with patch.object(v819, "_publish_validated_source") as publish:
                examined, resolved = v839._retry_deferred_batch(runtime, limit=4)

        self.assertEqual(examined, 4)
        self.assertEqual(resolved, 0)
        self.assertEqual(resolve.call_count, 4)
        publish.assert_not_called()
        self.assertEqual(len(runtime._v819_deferred_sources), 6)


if __name__ == "__main__":
    unittest.main()
