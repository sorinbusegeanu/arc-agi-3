from __future__ import annotations

import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from v8.runtime import ContinuousMemoryRuntime
from v8.shutdown_semantics_v089 import _mark_sampling_complete


class ShutdownSemanticsV089Tests(unittest.TestCase):
    def _runtime_stub(self, *, sampling_complete: bool):
        runtime = object.__new__(ContinuousMemoryRuntime)
        runtime._sampling_complete = bool(sampling_complete)
        runtime._snapshot_freeze = threading.Event()
        runtime._generation = SimpleNamespace(value=7, get_lock=nullcontext)
        runtime.peers = Mock()
        runtime.peers.wait_idle.return_value = True
        runtime.raise_worker_errors = Mock()
        runtime._is_quiescent = Mock(return_value=True)
        runtime.metrics = Mock(return_value={"generation": 7})
        return runtime

    def test_sampling_completion_immediately_pauses_peer_scheduler(self) -> None:
        runtime = self._runtime_stub(sampling_complete=False)

        _mark_sampling_complete(runtime)

        self.assertTrue(runtime._sampling_complete)
        runtime.peers.pause.assert_called_once_with()

    def test_post_sampling_drain_uses_bounded_stabilization(self) -> None:
        runtime = self._runtime_stub(sampling_complete=True)

        def stabilize(*, max_cycles, commit_proposals, timeout):
            self.assertEqual(max_cycles, 8)
            commit_proposals()
            return "stable"

        runtime.peers.run_until_stable.side_effect = stabilize

        runtime.wait_quiescent(timeout=0.2, stable_checks=2)

        self.assertTrue(runtime.peers.pause.called)
        runtime.peers.wait_idle.assert_called_once()
        runtime.peers.run_until_stable.assert_called_once()
        runtime.peers.run_once.assert_not_called()
        runtime.peers.resume.assert_not_called()
        self.assertGreaterEqual(runtime._is_quiescent.call_count, 2)

    def test_midrun_semantic_settle_uses_graph_generation_not_proposal_counter(self) -> None:
        runtime = self._runtime_stub(sampling_complete=False)
        runtime.peers.metrics.side_effect = AssertionError(
            "proposal counters must not define convergence"
        )

        runtime.wait_quiescent(timeout=0.2, stable_checks=2)

        self.assertEqual(runtime.peers.run_once.call_count, 2)
        runtime.peers.metrics.assert_not_called()
        runtime.peers.resume.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
