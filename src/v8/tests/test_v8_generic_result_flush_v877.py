from __future__ import annotations

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v8 import generic_result_flush_v877 as v877
from v8 import mixed_environment_v859 as mixed
from v8 import mixed_research_runtime_integrity_v875 as v875


class _FlushQueue:
    def __init__(self):
        self.events = []

    def put(self, value):
        self.events.append(("put", value))

    def close(self):
        self.events.append(("close", None))

    def join_thread(self):
        self.events.append(("join_thread", None))


class _DelayedQueue:
    def __init__(self, message):
        self._message = message
        self.calls = 0

    def get(self, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise queue.Empty
        return self._message


class GenericResultFlushV877Tests(unittest.TestCase):
    def test_child_flushes_result_before_returning(self):
        result_queue = _FlushQueue()

        def base_worker(**kwargs):
            kwargs["result_queue"].put(("result", "ok"))
            return "done"

        with patch.object(v877, "_BASE_GENERIC_PROCESS_WORKER", side_effect=base_worker):
            result = v877._generic_process_worker_v877(result_queue=result_queue)

        self.assertEqual(result, "done")
        self.assertEqual(
            result_queue.events,
            [
                ("put", ("result", "ok")),
                ("close", None),
                ("join_thread", None),
            ],
        )

    def test_child_flush_preserves_worker_failure(self):
        result_queue = _FlushQueue()

        def base_worker(**kwargs):
            kwargs["result_queue"].put(("error", 34, "RuntimeError: boom"))
            raise RuntimeError("boom")

        with patch.object(v877, "_BASE_GENERIC_PROCESS_WORKER", side_effect=base_worker):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                v877._generic_process_worker_v877(result_queue=result_queue)

        self.assertEqual(result_queue.events[-2:], [("close", None), ("join_thread", None)])

    def test_terminal_drain_accepts_result_visible_after_initial_empty(self):
        row = SimpleNamespace(actor_id=34)
        results = _DelayedQueue(("result", row))
        by_actor = {}
        errors = []

        v877._terminal_result_drain_v877(results, by_actor, errors, 1)

        self.assertEqual(by_actor, {34: row})
        self.assertEqual(errors, [])
        self.assertEqual(results.calls, 2)

    def test_v877_is_final_generic_process_collection_authority(self):
        self.assertIs(v875._generic_process_worker_v875, v877._generic_process_worker_v877)
        self.assertIs(v875._run_generic_jobs_v875, v877._run_generic_jobs_v877)
        self.assertIs(mixed._run_generic_jobs, v875._run_generic_jobs_v875)


if __name__ == "__main__":
    unittest.main()
