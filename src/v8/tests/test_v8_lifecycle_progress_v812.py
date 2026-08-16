from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from v8 import lifecycle_progress_v812 as progress
from v8.model import CognitiveState, MemoryUid


class LifecycleProgressTests(unittest.TestCase):
    @staticmethod
    def _row(index: int, state: CognitiveState):
        return SimpleNamespace(
            uid=MemoryUid(index + 1, (index + 1) * 17),
            cognitive_state=int(state),
        )

    def test_prints_once_when_generation_lifecycle_completes(self) -> None:
        lifecycle = SimpleNamespace(
            _v812_active_window=3,
            _v812_last_completed_window=2,
            _v812_next_bucket=0,
        )
        supervisor = SimpleNamespace(
            lifecycle=lifecycle,
            current_generation=lambda: 192,
        )
        nodes = (
            self._row(0, CognitiveState.QUARANTINED),
            self._row(1, CognitiveState.RETIRE_PENDING),
            self._row(2, CognitiveState.RETIRED),
            self._row(3, CognitiveState.REACTIVATED),
            self._row(4, CognitiveState.ACTIVE),
        )

        original = progress._BASE_RUN_GENERATION_LIFECYCLE

        def complete(_supervisor, _nodes):
            lifecycle._v812_active_window = -1
            lifecycle._v812_last_completed_window = 3
            lifecycle._v812_next_bucket = 0
            return len(_nodes)

        progress._BASE_RUN_GENERATION_LIFECYCLE = complete
        try:
            output = io.StringIO()
            with redirect_stdout(output):
                evaluated = progress._run_generation_lifecycle_with_progress(
                    supervisor, nodes
                )
        finally:
            progress._BASE_RUN_GENERATION_LIFECYCLE = original

        self.assertEqual(evaluated, len(nodes))
        line = output.getvalue().strip()
        self.assertIn("lifecycle window=3 complete", line)
        self.assertIn("generation=192", line)
        self.assertIn("evaluated=5", line)
        self.assertIn("quarantined=1", line)
        self.assertIn("retire_pending=1", line)
        self.assertIn("retired=1", line)
        self.assertIn("reactivated=1", line)
        self.assertEqual(lifecycle._v812_progress_window, -1)
        self.assertEqual(lifecycle._v812_progress_evaluated, 0)

    def test_does_not_print_before_all_buckets_complete(self) -> None:
        lifecycle = SimpleNamespace(
            _v812_active_window=4,
            _v812_last_completed_window=3,
            _v812_next_bucket=8,
        )
        supervisor = SimpleNamespace(
            lifecycle=lifecycle,
            current_generation=lambda: 256,
        )
        nodes = (self._row(0, CognitiveState.ACTIVE),)

        original = progress._BASE_RUN_GENERATION_LIFECYCLE

        def incomplete(_supervisor, _nodes):
            lifecycle._v812_active_window = 4
            lifecycle._v812_last_completed_window = 3
            lifecycle._v812_next_bucket = 16
            return 1

        progress._BASE_RUN_GENERATION_LIFECYCLE = incomplete
        try:
            output = io.StringIO()
            with redirect_stdout(output):
                progress._run_generation_lifecycle_with_progress(supervisor, nodes)
        finally:
            progress._BASE_RUN_GENERATION_LIFECYCLE = original

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(lifecycle._v812_progress_window, 4)
        self.assertGreaterEqual(lifecycle._v812_progress_evaluated, 1)


if __name__ == "__main__":
    unittest.main()
