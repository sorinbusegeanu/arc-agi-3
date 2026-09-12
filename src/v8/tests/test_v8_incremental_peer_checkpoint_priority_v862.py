from __future__ import annotations

import types
import unittest

from v8 import incremental_peer_drain_v862 as v862
from v8.model import MemoryUid


class _Row:
    def __init__(self, value, level=1):
        self.value = value
        self.level = level
        self.uid = MemoryUid(1, int(value) + 1)
        self.source_uid = self.uid
        self.target_uid = MemoryUid.zero()


class _Arena:
    def __init__(self, rows):
        self._rows = tuple(rows)
        self.sequence = 0

    @property
    def count(self):
        return len(self._rows)

    def read(self, index):
        return self._rows[index]


class _View:
    def __init__(self):
        self._nodes = (_Arena(_Row(i) for i in range(v862._NODE_SLICE * 2)),)
        self._edges = (_Arena(_Row(i) for i in range(v862._EDGE_SLICE * 2)),)

    def node_records(self, *, level=None):
        rows = tuple(row for arena in self._nodes for row in arena._rows)
        if level is None:
            return rows
        return tuple(row for row in rows if int(row.level) == int(level))

    def edge_records(self):
        return tuple(row for arena in self._edges for row in arena._rows)


class IncrementalPeerCheckpointPriorityV862Tests(unittest.TestCase):
    def test_first_large_graph_checkpoint_anchors_to_restored_watermark(self):
        supervisor = types.SimpleNamespace(_v862_run_origin_watermark=60_000)
        self.assertFalse(
            v862._coherent_checkpoint_due(
                supervisor,
                watermark=60_000,
                now=10.0,
            )
        )
        self.assertEqual(supervisor._v862_last_coherent_checkpoint_watermark, 60_000)
        self.assertFalse(
            v862._coherent_checkpoint_due(
                supervisor,
                watermark=61_999,
                now=40.0,
            )
        )
        self.assertTrue(
            v862._coherent_checkpoint_due(
                supervisor,
                watermark=62_000,
                now=40.0,
            )
        )

    def test_due_checkpoint_runs_before_bounded_slice(self):
        view = _View()
        supervisor = types.SimpleNamespace(
            read_view=view,
            _seen={},
            _cycles=4,
            _last_developmental_cut="before",
            _v862_edge_wrapped_since_cycle=False,
            _v82_stabilizing=False,
            current_watermark=lambda: v862._COHERENT_MIN_WATERMARK_PROGRESS,
            _v862_last_coherent_checkpoint_time=0.0,
            _v862_last_coherent_checkpoint_watermark=0,
        )
        original_base = v862._BASE_PEER_RUN_ONCE
        original_time = v862.time.monotonic
        calls = []

        def base(self):
            calls.append(
                (
                    bool(getattr(self, "_v82_stabilizing", False)),
                    len(self.read_view.node_records()),
                    len(self.read_view.edge_records()),
                )
            )
            self._cycles += 1
            self._last_developmental_cut = "after"

        try:
            v862._BASE_PEER_RUN_ONCE = base
            v862.time.monotonic = lambda: 20.0
            v862._peer_run_once_v862(supervisor)
        finally:
            v862._BASE_PEER_RUN_ONCE = original_base
            v862.time.monotonic = original_time

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0],
            (True, v862._NODE_SLICE * 2, v862._NODE_SLICE * 2),
        )
        self.assertEqual(
            v862._saved_offset(supervisor, v862._NODE_OFFSET_KEY),
            0,
        )
        self.assertEqual(
            supervisor._v862_last_coherent_checkpoint_watermark,
            v862._COHERENT_MIN_WATERMARK_PROGRESS,
        )


if __name__ == "__main__":
    unittest.main()
