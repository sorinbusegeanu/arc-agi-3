from __future__ import annotations

import types
import unittest

import v8
from v8 import incremental_peer_drain_v862 as v862
from v8 import runtime_scaling_v841 as v841
from v8.developmental_cut import capture_developmental_cut
from v8.model import MemoryUid
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
        self.uid = MemoryUid(1, int(value) + 1)
        self.updated_watermark = max(0, int(value))
        self.cognitive_state = 0
        self.validation_state = 0
        self.source_uid = self.uid
        self.target_uid = MemoryUid.zero()
        self.relation_type = 1


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
            current_watermark=lambda: 100,
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

    def test_slice_targets_the_read_view_used_by_downstream_peer(self):
        class View:
            def __init__(self, count):
                self._nodes = (_Arena(_Row(i) for i in range(count)),)
                self._edges = (_Arena(_Row(i) for i in range(v862._EDGE_SLICE + 5)),)

            def node_records(self, *, level=None):
                raise AssertionError("active view must be sliced")

            def edge_records(self):
                raise AssertionError("active view must be sliced")

        active = View(v862._NODE_SLICE + 5)
        stale = View(v862._NODE_SLICE + 50)
        supervisor = types.SimpleNamespace(
            read_view=active,
            _v813_live_read_view=stale,
            _seen={},
            _cycles=1,
            _last_developmental_cut=None,
            _v862_edge_wrapped_since_cycle=False,
            current_watermark=lambda: 100,
        )
        original_base = v862._BASE_PEER_RUN_ONCE
        captured = {}
        try:
            def base(self):
                captured["nodes"] = len(self.read_view.node_records())

            v862._BASE_PEER_RUN_ONCE = base
            v862._peer_run_once_v862(supervisor)
        finally:
            v862._BASE_PEER_RUN_ONCE = original_base

        self.assertEqual(captured["nodes"], v862._NODE_SLICE)

    def test_arena_backed_cut_cannot_bypass_slice_accessors(self):
        class View:
            def __init__(self):
                self._nodes = (_Arena(_Row(i) for i in range(v862._NODE_SLICE + 7)),)
                self._edges = (_Arena(_Row(i) for i in range(v862._EDGE_SLICE + 7)),)

            def _stable_records_with_version(self, arena):
                return arena._rows, arena.sequence

            def node_records(self, *, level=None):
                raise AssertionError("full accessor must be replaced")

            def edge_records(self):
                raise AssertionError("full accessor must be replaced")

            def source_games(self, uid, *, max_depth=8):
                return frozenset()

        supervisor = types.SimpleNamespace(
            read_view=View(),
            _seen={},
            _cycles=1,
            _last_developmental_cut=None,
            _v862_edge_wrapped_since_cycle=False,
            current_watermark=lambda: 100,
        )
        original_base = v862._BASE_PEER_RUN_ONCE
        captured = {}
        try:
            def base(self):
                cut = capture_developmental_cut(
                    self.read_view,
                    generation=1,
                    watermark=100,
                )
                captured["nodes"] = len(cut.nodes)
                captured["edges"] = len(cut.edges)

            v862._BASE_PEER_RUN_ONCE = base
            v862._peer_run_once_v862(supervisor)
        finally:
            v862._BASE_PEER_RUN_ONCE = original_base

        self.assertEqual(captured["nodes"], v862._NODE_SLICE)
        # Only edges whose synthetic source is inside the bounded node slice remain.
        # An arena fast-path capture would have returned more than this slice.
        self.assertEqual(captured["edges"], v862._NODE_SLICE)

    def test_watermark_cadence_runs_full_checkpoint_before_sweep_wrap(self):
        class View:
            def __init__(self):
                self._nodes = (_Arena(_Row(i) for i in range(v862._NODE_SLICE * 4)),)
                self._edges = (_Arena(_Row(i) for i in range(v862._EDGE_SLICE * 4)),)

            def node_records(self, *, level=None):
                rows = tuple(row for arena in self._nodes for row in arena._rows)
                if level is None:
                    return rows
                return tuple(row for row in rows if int(row.level) == int(level))

            def edge_records(self):
                return tuple(row for arena in self._edges for row in arena._rows)

        view = View()
        supervisor = types.SimpleNamespace(
            read_view=view,
            _seen={},
            _cycles=4,
            _last_developmental_cut="before",
            _v862_edge_wrapped_since_cycle=False,
            _v82_stabilizing=False,
            _v862_last_coherent_checkpoint_time=0.0,
            _v862_last_coherent_checkpoint_watermark=1000,
            current_watermark=lambda: 11000,
        )
        original_base = v862._BASE_PEER_RUN_ONCE
        original_time = v862.time.monotonic
        calls = []

        def base(self):
            calls.append((
                bool(getattr(self, "_v82_stabilizing", False)),
                len(self.read_view.node_records()),
                len(self.read_view.edge_records()),
            ))
            self._cycles += 1
            self._last_developmental_cut = "after"

        try:
            v862._BASE_PEER_RUN_ONCE = base
            v862.time.monotonic = lambda: 1.0
            v862._peer_run_once_v862(supervisor)
        finally:
            v862._BASE_PEER_RUN_ONCE = original_base
            v862.time.monotonic = original_time

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0],
            (True, 2048, 2048),
        )
        self.assertEqual(supervisor._cycles, 5)
        self.assertEqual(supervisor._last_developmental_cut, "after")
        self.assertEqual(supervisor._v862_last_coherent_checkpoint_watermark, 11000)
        self.assertEqual(v862._saved_offset(supervisor, v862._NODE_OFFSET_KEY), 0)

    def test_time_cadence_requires_real_watermark_progress(self):
        supervisor = types.SimpleNamespace(
            _v862_last_coherent_checkpoint_time=10.0,
            _v862_last_coherent_checkpoint_watermark=5000,
        )
        self.assertFalse(
            v862._coherent_checkpoint_due(supervisor, watermark=6000, now=40.0)
        )
        self.assertTrue(
            v862._coherent_checkpoint_due(supervisor, watermark=7000, now=40.0)
        )

    def test_completed_large_sweep_runs_one_coherent_full_checkpoint(self):
        class View:
            def __init__(self):
                self._nodes = (_Arena(_Row(i) for i in range(v862._NODE_SLICE + 1)),)
                self._edges = (_Arena(_Row(i) for i in range(v862._EDGE_SLICE + 1)),)

            def node_records(self, *, level=None):
                rows = tuple(row for arena in self._nodes for row in arena._rows)
                if level is None:
                    return rows
                return tuple(row for row in rows if int(row.level) == int(level))

            def edge_records(self):
                return tuple(row for arena in self._edges for row in arena._rows)

        view = View()
        supervisor = types.SimpleNamespace(
            read_view=view,
            _seen={
                v862._NODE_OFFSET_KEY: 1,
                v862._EDGE_OFFSET_KEY: 1,
            },
            _cycles=4,
            _last_developmental_cut="before",
            _v862_edge_wrapped_since_cycle=False,
            _v82_stabilizing=False,
            current_watermark=lambda: 9000,
        )
        original_base = v862._BASE_PEER_RUN_ONCE
        calls = []

        def base(self):
            calls.append((
                bool(getattr(self, "_v82_stabilizing", False)),
                len(self.read_view.node_records()),
                len(self.read_view.edge_records()),
            ))
            self._cycles += 1
            self._last_developmental_cut = "after"

        try:
            v862._BASE_PEER_RUN_ONCE = base
            v862._peer_run_once_v862(supervisor)
        finally:
            v862._BASE_PEER_RUN_ONCE = original_base

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0],
            (True, v862._NODE_SLICE + 1, v862._NODE_SLICE + 1),
        )
        self.assertEqual(supervisor._cycles, 5)
        self.assertEqual(supervisor._last_developmental_cut, "after")
        self.assertFalse(supervisor._v82_stabilizing)

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

    def test_finalization_allows_third_bounded_window_for_full_noop_proof(self):
        original_base = v862._BASE_RUNTIME_WAIT

        class Peers:
            def __init__(self):
                self.reasons = iter(("max_cycles", "max_cycles", "stable"))
                self.calls = 0

            def pause(self):
                return None

            def run_until_stable(self, **kwargs):
                self.calls += 1
                kwargs["commit_proposals"]()
                return next(self.reasons)

        peers = Peers()
        runtime = types.SimpleNamespace(
            _sampling_complete=True,
            _accepting=False,
            peers=peers,
            generation=12,
        )
        try:
            v862._BASE_RUNTIME_WAIT = lambda self, **kwargs: None
            v862._runtime_wait_quiescent_v862(runtime, timeout=1.0)
        finally:
            v862._BASE_RUNTIME_WAIT = original_base
        self.assertEqual(peers.calls, 3)
        self.assertTrue(runtime._v82_developmental_finalized)
        self.assertEqual(runtime._v82_developmental_finalized_generation, 12)

    def test_finalization_timeout_defers_semantic_work_and_still_drains(self):
        original_base = v862._BASE_RUNTIME_WAIT
        original_time = v862.time.monotonic
        drains = []

        class Peers:
            def __init__(self):
                self.timeouts = []

            def pause(self):
                return None

            def run_until_stable(self, **kwargs):
                self.timeouts.append(kwargs["timeout"])
                return "max_cycles"

        peers = Peers()
        runtime = types.SimpleNamespace(
            _sampling_complete=True,
            _accepting=False,
            peers=peers,
            generation=12,
        )
        ticks = iter((10.0, 10.1, 10.6, 11.0))
        try:
            v862._BASE_RUNTIME_WAIT = lambda self, **kwargs: drains.append(kwargs)
            v862.time.monotonic = lambda: next(ticks)
            v862._runtime_wait_quiescent_v862(runtime, timeout=1.0)
        finally:
            v862._BASE_RUNTIME_WAIT = original_base
            v862.time.monotonic = original_time

        self.assertEqual(len(peers.timeouts), 2)
        self.assertAlmostEqual(peers.timeouts[0], 0.9)
        self.assertAlmostEqual(peers.timeouts[1], 0.4)
        self.assertEqual(runtime._v82_developmental_finalization_status, "timeout")
        self.assertFalse(getattr(runtime, "_v82_developmental_finalized", False))
        self.assertEqual(len(drains), 1)
        self.assertEqual(drains[0]["timeout"], 1.0)

    def test_incomplete_cut_does_not_abort_canonical_shutdown_drain(self):
        original_base = v862._BASE_RUNTIME_WAIT
        drains = []
        stabilization_calls = []

        def stabilize(**_kwargs):
            stabilization_calls.append(1)
            return "incomplete_cut"

        peers = types.SimpleNamespace(
            pause=lambda: None,
            run_until_stable=stabilize,
        )
        runtime = types.SimpleNamespace(
            _sampling_complete=True,
            _accepting=False,
            peers=peers,
            generation=17,
        )
        try:
            v862._BASE_RUNTIME_WAIT = lambda self, **kwargs: drains.append(kwargs)
            v862._runtime_wait_quiescent_v862(runtime, timeout=9.0)
        finally:
            v862._BASE_RUNTIME_WAIT = original_base

        self.assertEqual(runtime._v82_developmental_finalization_status, "incomplete_cut")
        self.assertFalse(getattr(runtime, "_v82_developmental_finalized", False))
        self.assertTrue(runtime._v82_developmental_finalization_deferred)
        self.assertEqual(runtime._v82_developmental_finalization_attempted_generation, 17)
        self.assertEqual(len(drains), 1)
        self.assertEqual(drains[0]["timeout"], 9.0)

        # Metrics, snapshot, and close paths may all request quiescence. Do not
        # repeat the known-incomplete semantic pass in this process, even when
        # draining its own proposals advances the graph generation.
        runtime.generation = 18
        try:
            v862._BASE_RUNTIME_WAIT = lambda self, **kwargs: drains.append(kwargs)
            v862._runtime_wait_quiescent_v862(runtime, timeout=9.0)
        finally:
            v862._BASE_RUNTIME_WAIT = original_base
        self.assertEqual(stabilization_calls, [1])
        self.assertEqual(len(drains), 2)

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
