from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import v8
from v8 import snapshot_resilience_v833 as repair


class SnapshotResilienceV833Tests(unittest.TestCase):
    def test_periodic_peer_pause_timeout_is_skipped_not_fatal(self):
        runtime = SimpleNamespace(
            request_consistent_snapshot=Mock(side_effect=TimeoutError("peer busy")),
            _snapshot_error=None,
        )
        status = repair._background_snapshot_attempt_v833(runtime, timeout=10.0)
        self.assertEqual(status, "skipped")
        self.assertIsNone(runtime._snapshot_error)
        self.assertEqual(runtime._v833_background_snapshot_skips, 1)

    def test_real_snapshot_failure_remains_fatal(self):
        runtime = SimpleNamespace(
            request_consistent_snapshot=Mock(side_effect=RuntimeError("snapshotter died")),
            _snapshot_error=None,
        )
        status = repair._background_snapshot_attempt_v833(runtime, timeout=10.0)
        self.assertEqual(status, "fatal")
        self.assertIn("snapshotter died", runtime._snapshot_error)


if __name__ == "__main__":
    unittest.main()
