from __future__ import annotations

import types
import unittest

import v8
from v8 import stabilization_noop_retry_v883 as v883
from v8 import peers_v82


class _Pause:
    def __init__(self):
        self.value = False

    def is_set(self):
        return self.value


class _Supervisor:
    def __init__(self):
        self._pause = _Pause()
        self._v841_peer_cancel = None
        self._v82_stabilizing = False
        self._v841_last_input_token = object()
        self._cycles = 4
        self._cut = types.SimpleNamespace(nodes=())
        self.calls = 0
        self.public_calls = 0
        self.submit_proposal = lambda proposal: None

    @property
    def last_developmental_cut(self):
        return self._cut

    def pause(self):
        self._pause.value = True

    def resume(self):
        self._pause.value = False

    def wait_idle(self, timeout):
        return True

    def run_once(self):
        self.public_calls += 1
        raise AssertionError("final stabilization must bypass public run_once wrappers")


class StabilizationNoopRetryV883Tests(unittest.TestCase):
    def test_final_stabilization_uses_direct_full_cut_authority(self):
        supervisor = _Supervisor()
        commits = []
        original = v883._BASE_FULL_CUT_RUN_ONCE

        def direct(current):
            current.calls += 1
            current._cycles += 1
            current._cut = types.SimpleNamespace(nodes=())

        try:
            v883._BASE_FULL_CUT_RUN_ONCE = direct
            result = v883._run_until_stable_v883(
                supervisor,
                max_cycles=2,
                commit_proposals=lambda: commits.append(1),
                timeout=1.0,
            )
        finally:
            v883._BASE_FULL_CUT_RUN_ONCE = original

        self.assertEqual(result, "stable")
        self.assertEqual(supervisor.calls, 1)
        self.assertEqual(supervisor.public_calls, 0)
        self.assertEqual(len(commits), 2)
        self.assertFalse(supervisor._pause.is_set())

    def test_true_direct_noop_is_retried(self):
        supervisor = _Supervisor()
        original = v883._BASE_FULL_CUT_RUN_ONCE

        def direct(current):
            current.calls += 1
            if current.calls == 1:
                return
            current._cycles += 1
            current._cut = types.SimpleNamespace(nodes=())

        try:
            v883._BASE_FULL_CUT_RUN_ONCE = direct
            result = v883._run_until_stable_v883(
                supervisor,
                max_cycles=1,
                commit_proposals=lambda: None,
                timeout=1.0,
            )
        finally:
            v883._BASE_FULL_CUT_RUN_ONCE = original

        self.assertEqual(result, "stable")
        self.assertEqual(supervisor.calls, 2)

    def test_partial_cut_still_fails(self):
        supervisor = _Supervisor()
        original = v883._BASE_FULL_CUT_RUN_ONCE

        def partial(current):
            current.calls += 1
            current._cycles += 1

        try:
            v883._BASE_FULL_CUT_RUN_ONCE = partial
            with self.assertRaisesRegex(RuntimeError, "partial developmental cut"):
                v883._run_until_stable_v883(
                    supervisor,
                    max_cycles=1,
                    commit_proposals=lambda: None,
                    timeout=1.0,
                )
        finally:
            v883._BASE_FULL_CUT_RUN_ONCE = original

    def test_runtime_stack_installs_v883_authority(self):
        self.assertIs(peers_v82.V82DevelopmentalPeerSupervisor.run_until_stable, v883._run_until_stable_v883)
        self.assertTrue(callable(v883._BASE_FULL_CUT_RUN_ONCE))


if __name__ == "__main__":
    unittest.main()
