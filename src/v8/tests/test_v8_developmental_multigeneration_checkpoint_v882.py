from __future__ import annotations

import types
import unittest

import v8
from v8 import developmental_multigeneration_checkpoint_v882 as v882
from v8 import incremental_peer_drain_v862 as v862


class _Runtime:
    def __init__(self):
        self.waits = []

    def submit_proposal(self, proposal):
        return None

    def wait_quiescent(self, **kwargs):
        self.waits.append(dict(kwargs))


class DevelopmentalMultigenerationCheckpointV882Tests(unittest.TestCase):
    def test_checkpoint_runs_four_generations_with_three_commit_barriers(self):
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
        self.assertEqual(len(runtime.waits), 3)
        for wait in runtime.waits:
            self.assertEqual(wait["timeout"], v882._COMMIT_TIMEOUT_SECONDS)
            self.assertEqual(wait["stable_checks"], 2)
            self.assertTrue(wait["resume_peers"])
            self.assertFalse(wait["settle_peers"])

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
