from __future__ import annotations

import types
import unittest

import v8
from v8 import incremental_peer_drain_v862 as v862
from v8 import runtime_scaling_v841 as v841
from v8.runtime import ContinuousMemoryRuntime


class _Arena:
    def __init__(self, rows):
        self._rows = tuple(rows)
        self.sequence = 0

    @property
    def count(self):
        return len(self._rows)

    def read(self, index):
        return self._rows[index]


class _Row:
    def __init__(self, value, level=1):
        self.value = value
        self.level = level


class IncrementalPeerDrainV862Tests(unittest.TestCase):
    def test_bounded_slice_never_materializes_more_than_limit(self):
        arenas = (_Arena(_Row(i) for i in range(10)), _Arena(_Row(i) for i in range(10, 20)))
        rows, offset, wrapped = v862._bounded_arena_slice(arenas, 0, 7)
        self.assertEqual([row.value for row in rows], list(range(7)))
        self.assertEqual(offset, 7)
        self.assertFalse(wrapped)

    def test_absolute_cursor_wraps_and_continues_without_reset(self):
        arenas = (_Arena(_Row(i) for i in range(5)),)
        rows, offset, wrapped = v862._bounded_arena_slice(arenas, 4, 3)
        self.assertEqual([row.value for row in rows], [4, 0, 1])
        self.assertEqual(offset, 7)
        self.assertTrue(wrapped)

    def test_offsets_are_persisted_through_existing_seen_state(self):
        supervisor = types.SimpleNamespace(_seen={})
        v862._save_offset(supervisor, v862._NODE_OFFSET_KEY, 123)
        v862._save_offset(supervisor, v862._NODE_OFFSET_KEY, 5)
        self.assertEqual(v862._saved_offset(supervisor, v862._NODE_OFFSET_KEY), 123)

    def test_incomplete_slice_does_not_mark_v841_input_complete(self):
        class View:
            def __init__(self):
                self._nodes = (_Arena(_Row(i) for i in range(v862._NODE_SLICE + 5)),)
                self._edges = (_Arena(_Row(i) for i in range(v862._EDGE_SLICE + 5)),)

            def node_records(self, *, level=None):
                raise AssertionError("full node scan must be replaced during peer slice")

            def edge_records(self):
                raise AssertionError("full edge scan must be replaced during peer slice")

        supervisor = types.SimpleNamespace(
            read_view=View(),
            _seen={},
            _cycles=10,
            _last_developmental_cut=object(),
            _v862_edge_wrapped_since_cycle=False,
        )
        original_base = v862._BASE_PEER_RUN_ONCE
        captured = {}

        def base(self):
            captured["nodes"] = len(self.read_view.node_records())
            captured["edges"] = len(self.read_view.edge_records())
            self._cycles += 1
            self._last_developmental_cut = object()

        try:
            v862._BASE_PEER_RUN_ONCE = base
            v862._peer_run_once_v862(supervisor)
        finally:
            v862._BASE_PEER_RUN_ONCE = original_base

        self.assertEqual(captured["nodes"], v862._NODE_SLICE)
        self.assertEqual(captured["edges"], v862._EDGE_SLICE)
        self.assertEqual(supervisor._cycles, 10)

    def test_small_graph_keeps_historical_single_pass_semantics(self):
        class View:
            def __init__(self):
                self._nodes = (_Arena((_Row(1), _Row(2))),)
                self._edges = (_Arena((_Row(3),)),)

            def node_records(self, *, level=None):
                return ()

            def edge_records(self):
                return ()

        supervisor = types.SimpleNamespace(
            read_view=View(),
            _seen={},
            _cycles=2,
            _last_developmental_cut=None,
            _v862_edge_wrapped_since_cycle=False,
        )
        original_base = v862._BASE_PEER_RUN_ONCE

        def base(self):
            self._cycles += 1

        try:
            v862._BASE_PEER_RUN_ONCE = base
            v862._peer_run_once_v862(supervisor)
        finally:
            v862._BASE_PEER_RUN_ONCE = original_base
        self.assertEqual(supervisor._cycles, 3)

    def test_post_sampling_wait_disables_peer_fixed_point_settle(self):
        original_base = v862._BASE_RUNTIME_WAIT
        captured = {}

        def base(self, **kwargs):
            captured.update(kwargs)

        runtime = types.SimpleNamespace(_sampling_complete=True, _accepting=True)
        try:
            v862._BASE_RUNTIME_WAIT = base
            v862._runtime_wait_quiescent_v862(runtime, timeout=9.0, settle_peers=True)
        finally:
            v862._BASE_RUNTIME_WAIT = original_base
        self.assertFalse(captured["settle_peers"])
        self.assertEqual(captured["timeout"], 9.0)

    def test_normal_active_wait_can_still_settle_peers(self):
        original_base = v862._BASE_RUNTIME_WAIT
        captured = {}

        def base(self, **kwargs):
            captured.update(kwargs)

        runtime = types.SimpleNamespace(_sampling_complete=False, _accepting=True)
        try:
            v862._BASE_RUNTIME_WAIT = base
            v862._runtime_wait_quiescent_v862(runtime, settle_peers=True)
        finally:
            v862._BASE_RUNTIME_WAIT = original_base
        self.assertTrue(captured["settle_peers"])

    def test_runtime_stack_installs_below_v841_without_replacing_peer_authority(self):
        self.assertIs(v841._BASE_PEER_RUN_ONCE, v862._peer_run_once_v862)
        self.assertIs(ContinuousMemoryRuntime.wait_quiescent, v862._runtime_wait_quiescent_v862)


if __name__ == "__main__":
    unittest.main()
