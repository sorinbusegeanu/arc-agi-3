from __future__ import annotations

import types
import unittest

import v8
from v8 import developmental_multigeneration_checkpoint_v882 as v882
from v8 import incremental_peer_drain_v862 as v862


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
    def test_checkpoint_runs_four_generations_with_three_scoped_fences(self):
        runtime = _Runtime()
        supervisor = types.SimpleNamespace(submit_proposal=runtime.submit_proposal)
        calls = []
        original = v882._BASE_COHERENT_CHECKPOINT

        def base(current, *, before_cycles, before_cut):
            calls.append((current, before_cycles, before_cut))
            for ring, arena in zip(runtime._shard_rings, runtime._node_arenas, strict=True):
                ring._tail.value += 1
                ring._head.value = ring._tail.value
                arena.sequence += 2

        try:
            v882._BASE_COHERENT_CHECKPOINT = base
            v882._coherent_checkpoint_v882(
                supervisor,
                before_cycles=7,
                before_cut="cut",
            )
        finally:
            v882._BASE_COHERENT_CHECKPOINT = original

        self.assertEqual(len(calls), 4)
        self.assertFalse(runtime.wait_quiescent_called)
        self.assertGreaterEqual(runtime.worker_error_checks, 3)

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

    def test_runtime_stack_installs_v882_as_v862_checkpoint_authority(self):
        self.assertIs(v862._coherent_checkpoint, v882._coherent_checkpoint_v882)


if __name__ == "__main__":
    unittest.main()
