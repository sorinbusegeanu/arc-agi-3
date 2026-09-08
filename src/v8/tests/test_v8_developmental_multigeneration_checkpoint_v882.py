from __future__ import annotations

import types
import unittest

import v8
from v8 import developmental_multigeneration_checkpoint_v882 as v882
from v8 import incremental_peer_drain_v862 as v862


class _Ring:
    def __init__(self, empty=True):
        self.empty = bool(empty)


class _Value:
    def __init__(self, value=0):
        self.value = int(value)


class _Runtime:
    def __init__(self):
        self._shard_rings = (_Ring(True), _Ring(True))
        self._shard_inflight = (_Value(0), _Value(0))
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
    def test_checkpoint_runs_four_generations_with_three_shard_commit_barriers(self):
        runtime = _Runtime()
        supervisor = types.SimpleNamespace(submit_proposal=runtime.submit_proposal)
        calls = []
        original = v882._BASE_COHERENT_CHECKPOINT

        def base(current, *, before_cycles, before_cut):
            calls.append((current, before_cycles, before_cut))

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

    def test_peer_commit_barrier_ignores_active_stage_pipeline(self):
        runtime = _Runtime()
        runtime._stage_rings = (_Ring(False),)
        runtime._stage_inflight = (_Value(1),)
        supervisor = types.SimpleNamespace(submit_proposal=runtime.submit_proposal)
        self.assertTrue(v882._commit_barrier(supervisor))
        self.assertFalse(runtime.wait_quiescent_called)

    def test_busy_shard_state_is_not_committed(self):
        runtime = _Runtime()
        runtime._shard_rings = (_Ring(False),)
        runtime._shard_inflight = (_Value(0),)
        self.assertFalse(v882._peer_proposals_committed(runtime))
        runtime._shard_rings = (_Ring(True),)
        runtime._shard_inflight = (_Value(1),)
        self.assertFalse(v882._peer_proposals_committed(runtime))

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
