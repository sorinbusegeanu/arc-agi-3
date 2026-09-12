from __future__ import annotations

import types
import threading
import unittest

import v8
from v8 import developmental_multigeneration_checkpoint_v882 as v882
from v8 import incremental_peer_drain_v862 as v862
from v8.model import MemoryUid


class _Cursor:
    def __init__(self, value=0):
        self.value = int(value)


class _Ring:
    def __init__(self, head=0, tail=0):
        self._head = _Cursor(head)
        self._tail = _Cursor(tail)


class _Arena:
    def __init__(self, sequence=0):
        self.sequence = int(sequence)


class _Runtime:
    def __init__(self):
        self._shard_rings = (_Ring(), _Ring())
        self._node_arenas = (_Arena(), _Arena())
        self._snapshot_freeze = threading.Event()
        self.worker_error_checks = 0
        self.wait_quiescent_called = False

    def submit_proposal(self, proposal):
        return None

    def raise_worker_errors(self):
        self.worker_error_checks += 1

    def wait_quiescent(self, **kwargs):
        self.wait_quiescent_called = True
        raise AssertionError("active checkpoint must not wait for global quiescence")


class DevelopmentalMultigenerationCheckpointV882Tests(unittest.TestCase):
    def test_active_checkpoint_view_bounds_m1_and_preserves_all_higher_nodes(self):
        rows = tuple(
            types.SimpleNamespace(uid=MemoryUid(1, index + 1), level=1)
            for index in range(v882._ACTIVE_M1_LIMIT + 9)
        ) + (
            types.SimpleNamespace(uid=MemoryUid(2, 1), level=2),
            types.SimpleNamespace(uid=MemoryUid(7, 1), level=7),
        )
        edges = tuple(
            types.SimpleNamespace(
                source_uid=row.uid,
                target_uid=MemoryUid.zero(),
            )
            for row in rows
        )

        class View:
            def node_records(self):
                return rows

            def edge_records(self):
                return edges

            def source_games(self, uid, *, max_depth=8):
                return (int(uid.hi),)

        (
            bounded,
            live_count,
            selected_count,
            edge_count,
            live_edge_count,
        ) = v882._active_checkpoint_view(View())
        selected = bounded.node_records()
        self.assertEqual(live_count, len(rows))
        self.assertEqual(selected_count, v882._ACTIVE_M1_LIMIT + 2)
        self.assertEqual(edge_count, selected_count)
        self.assertEqual(live_edge_count, len(edges))
        self.assertEqual(sum(int(row.level) == 1 for row in selected), v882._ACTIVE_M1_LIMIT)
        self.assertEqual(sum(int(row.level) >= 2 for row in selected), 2)
        self.assertEqual(bounded.source_games(MemoryUid(7, 1)), (7,))

    def test_arena_backed_active_view_reads_only_recent_edge_window(self):
        class Arena:
            def __init__(self, rows):
                self.rows = tuple(rows)
                self.sequence = 0

            @property
            def count(self):
                return len(self.rows)

            def read(self, index):
                return self.rows[index]

        nodes = tuple(
            types.SimpleNamespace(uid=MemoryUid(1, index + 1), level=1)
            for index in range(32)
        )
        edges = tuple(
            types.SimpleNamespace(
                source_uid=nodes[index % len(nodes)].uid,
                target_uid=MemoryUid.zero(),
            )
            for index in range(v882._ACTIVE_EDGE_LIMIT + 17)
        )
        view = types.SimpleNamespace(
            _nodes=(Arena(nodes),),
            _edges=(Arena(edges),),
        )
        materialized_nodes, materialized_edges, edge_count = (
            v882._materialize_active_rows(view)
        )
        self.assertEqual(materialized_nodes, nodes)
        self.assertEqual(len(materialized_edges), v882._ACTIVE_EDGE_LIMIT)
        self.assertEqual(materialized_edges[0], edges[-v882._ACTIVE_EDGE_LIMIT])
        self.assertEqual(edge_count, len(edges))

    def test_recent_edge_window_is_balanced_across_shards(self):
        class Arena:
            def __init__(self, rows):
                self.rows = tuple(rows)
                self.sequence = 0

            @property
            def count(self):
                return len(self.rows)

            def read(self, index):
                return self.rows[index]

        arenas = (
            Arena(("a0", "a1", "a2")),
            Arena(("b0", "b1", "b2")),
        )
        self.assertEqual(
            v882._recent_rows_per_arena(arenas, 4),
            ("a1", "a2", "b1", "b2"),
        )

    def test_active_view_prioritizes_m1_parents_referenced_by_higher_edges(self):
        m1 = tuple(
            types.SimpleNamespace(uid=MemoryUid(1, index + 1), level=1)
            for index in range(v882._ACTIVE_M1_LIMIT + 1)
        )
        higher = types.SimpleNamespace(uid=MemoryUid(2, 1), level=2)
        causal_edge = types.SimpleNamespace(
            source_uid=higher.uid,
            target_uid=m1[-1].uid,
        )

        class View:
            def node_records(self):
                return m1 + (higher,)

            def edge_records(self):
                return (causal_edge,)

            def source_games(self, uid, *, max_depth=8):
                return ()

        bounded, *_counts = v882._active_checkpoint_view(View())
        selected_uids = {row.uid for row in bounded.node_records()}
        self.assertIn(m1[-1].uid, selected_uids)
        self.assertNotIn(m1[-2].uid, selected_uids)
        self.assertEqual(bounded.edge_records(), (causal_edge,))

    def test_checkpoint_runs_six_formation_generations_then_one_full_cut(self):
        runtime = _Runtime()
        supervisor = types.SimpleNamespace(submit_proposal=runtime.submit_proposal)
        formation_calls = []
        full_calls = []
        original_formation = v882._formation_checkpoint
        original_full = v882._BASE_COHERENT_CHECKPOINT

        def advance(calls, current, before_cycles, before_cut):
            self.assertFalse(runtime._snapshot_freeze.is_set())
            calls.append((current, before_cycles, before_cut))
            for ring, arena in zip(runtime._shard_rings, runtime._node_arenas, strict=True):
                ring._tail.value += 1
                ring._head.value = ring._tail.value
                arena.sequence += 2

        try:
            v882._formation_checkpoint = lambda current, *, before_cycles, before_cut: advance(
                formation_calls, current, before_cycles, before_cut
            )
            v882._BASE_COHERENT_CHECKPOINT = lambda current, *, before_cycles, before_cut: advance(
                full_calls, current, before_cycles, before_cut
            )
            v882._coherent_checkpoint_v882(
                supervisor,
                before_cycles=7,
                before_cut="cut",
            )
        finally:
            v882._formation_checkpoint = original_formation
            v882._BASE_COHERENT_CHECKPOINT = original_full

        self.assertEqual(len(formation_calls), 6)
        self.assertEqual(len(full_calls), 1)
        self.assertFalse(runtime._snapshot_freeze.is_set())
        self.assertFalse(runtime.wait_quiescent_called)
        self.assertGreaterEqual(runtime.worker_error_checks, 5)

    def test_active_checkpoint_releases_ingress_freeze_before_analysis(self):
        runtime = _Runtime()
        drain_freeze_states = []
        analysis_freeze_states = []
        supervisor = types.SimpleNamespace(submit_proposal=runtime.submit_proposal)
        original_drain = v882._drain_published_ingress
        original_formation = v882._formation_checkpoint
        original_barrier = v882._commit_barrier
        try:
            v882._drain_published_ingress = lambda current: (
                drain_freeze_states.append(current._snapshot_freeze.is_set()) or True
            )
            v882._formation_checkpoint = lambda *args, **kwargs: (
                analysis_freeze_states.append(runtime._snapshot_freeze.is_set())
            )
            v882._commit_barrier = lambda *args, **kwargs: False
            v882._coherent_checkpoint_v882(
                supervisor,
                before_cycles=1,
                before_cut=None,
            )
        finally:
            v882._drain_published_ingress = original_drain
            v882._formation_checkpoint = original_formation
            v882._commit_barrier = original_barrier

        self.assertEqual(drain_freeze_states, [True])
        self.assertEqual(analysis_freeze_states, [False])
        self.assertFalse(runtime._snapshot_freeze.is_set())

    def test_later_unrelated_packets_do_not_block_checkpoint_fence(self):
        runtime = _Runtime()
        before = v882._capture_fence(runtime)
        runtime._shard_rings[0]._tail.value = 2
        after = v882._capture_fence(runtime)

        # Checkpoint packets are consumed and committed.
        runtime._shard_rings[0]._head.value = 2
        runtime._node_arenas[0].sequence = 2
        # Unrelated traffic arrives after the checkpoint fence.
        runtime._shard_rings[0]._tail.value = 50

        self.assertTrue(v882._fence_committed(runtime, before, after))

    def test_dequeued_fence_is_not_committed_until_arena_write_finishes(self):
        runtime = _Runtime()
        before = v882._capture_fence(runtime)
        runtime._shard_rings[0]._tail.value = 1
        after = v882._capture_fence(runtime)
        runtime._shard_rings[0]._head.value = 1

        runtime._node_arenas[0].sequence = 1
        self.assertFalse(v882._fence_committed(runtime, before, after))
        runtime._node_arenas[0].sequence = 2
        self.assertTrue(v882._fence_committed(runtime, before, after))

    def test_shards_without_checkpoint_proposals_need_no_sequence_advance(self):
        runtime = _Runtime()
        before = v882._capture_fence(runtime)
        after = v882._capture_fence(runtime)
        self.assertTrue(v882._fence_committed(runtime, before, after))

    def test_missing_runtime_falls_back_to_one_generation(self):
        supervisor = types.SimpleNamespace(submit_proposal=lambda proposal: None)
        calls = []
        original = v882._BASE_COHERENT_CHECKPOINT

        def base(current, *, before_cycles, before_cut):
            calls.append((before_cycles, before_cut))

        try:
            v882._BASE_COHERENT_CHECKPOINT = base
            v882._coherent_checkpoint_v882(
                supervisor,
                before_cycles=3,
                before_cut="before",
            )
        finally:
            v882._BASE_COHERENT_CHECKPOINT = original

        self.assertEqual(calls, [(3, "before")])

    def test_commit_failure_stops_before_full_evidence_cut(self):
        runtime = _Runtime()
        supervisor = types.SimpleNamespace(submit_proposal=runtime.submit_proposal)
        formation_calls = []
        full_calls = []
        original_formation = v882._formation_checkpoint
        original_full = v882._BASE_COHERENT_CHECKPOINT
        original_barrier = v882._commit_barrier
        try:
            v882._formation_checkpoint = lambda *args, **kwargs: formation_calls.append(1)
            v882._BASE_COHERENT_CHECKPOINT = lambda *args, **kwargs: full_calls.append(1)
            v882._commit_barrier = lambda *args, **kwargs: False
            v882._coherent_checkpoint_v882(
                supervisor,
                before_cycles=1,
                before_cut=None,
            )
        finally:
            v882._formation_checkpoint = original_formation
            v882._BASE_COHERENT_CHECKPOINT = original_full
            v882._commit_barrier = original_barrier

        self.assertEqual(formation_calls, [1])
        self.assertEqual(full_calls, [])

    def test_active_formation_defers_correspondence_to_full_cut(self):
        supervisor = types.SimpleNamespace(
            _cycles=9,
            _last_developmental_cut="newer",
            _v82_stabilizing=False,
        )
        calls = []
        from v8 import stabilization_noop_retry_v883 as v883

        original = v883._run_formation_cut_once
        try:
            v883._run_formation_cut_once = lambda current, **kwargs: calls.append(
                (current, kwargs)
            )
            v882._formation_checkpoint(
                supervisor,
                before_cycles=3,
                before_cut="cut",
            )
        finally:
            v883._run_formation_cut_once = original

        self.assertEqual(calls, [(supervisor, {"include_correspondence": False})])
        self.assertEqual(supervisor._cycles, 3)
        self.assertEqual(supervisor._last_developmental_cut, "cut")
        self.assertFalse(supervisor._v82_stabilizing)

    def test_ingress_drain_requires_two_stable_checks(self):
        states = iter((False, True, True))
        runtime = types.SimpleNamespace(
            _is_quiescent=lambda: next(states),
            raise_worker_errors=lambda: None,
        )
        self.assertTrue(v882._drain_published_ingress(runtime))

    def test_runtime_stack_installs_v882_as_v862_checkpoint_authority(self):
        self.assertIs(v862._coherent_checkpoint, v882._coherent_checkpoint_v882)


if __name__ == "__main__":
    unittest.main()
