from __future__ import annotations

import io
import threading
import time
import unittest
from contextlib import redirect_stdout

import v8
from v8.arena import NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState
from v8.peers_v82 import V82DevelopmentalPeerSupervisor


class FakeReadView:
    def __init__(self, nodes, edges=()):
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)

    def node_records(self, *, level=None):
        if level is None:
            return self.nodes
        return tuple(row for row in self.nodes if int(row.level) == int(level))

    def edge_records(self):
        return self.edges

    def source_games(self, uid, *, max_depth=8):
        del uid, max_depth
        return frozenset()


def node(key=(101,)) -> NodeRecord:
    uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
    return NodeRecord(
        uid,
        (uid.hi ^ uid.lo) & ((1 << 64) - 1),
        int(MemoryLevel.M1),
        int(MemoryType.CONTINGENCY),
        tuple(key),
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1,
        0,
        int(CognitiveState.ACTIVE),
        int(ValidationState.STRUCTURAL),
        0.0,
        0.0,
        0.0,
    )


class DedicatedLifecycleWorkerTests(unittest.TestCase):
    def test_lifecycle_default_start_delay_is_five_minutes(self) -> None:
        from v8 import runtime_repair_v822 as repair

        self.assertEqual(repair._LIFECYCLE_START_DELAY_SECONDS, 300.0)

    def test_lifecycle_completes_while_main_peer_cycle_is_blocked(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        submitted = []
        supervisor = V82DevelopmentalPeerSupervisor(
            read_view=FakeReadView((node(),)),
            submit_proposal=submitted.append,
            watermark=lambda: 1,
            generation=lambda: 64,
            interval_seconds=0.05,
        )
        supervisor.set_candidate_budget(512)
        # Unit tests exercise lifecycle behavior itself, not the production 5-minute delay.
        supervisor._v822_lifecycle_start_delay_seconds = 0.0

        def blocked_peer_cycle():
            entered.set()
            release.wait(timeout=5.0)

        supervisor.run_once = blocked_peer_cycle
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                supervisor.start()
                self.assertTrue(entered.wait(timeout=1.0))
                deadline = time.monotonic() + 4.0
                while (
                    int(getattr(supervisor.lifecycle, "_v812_last_completed_window", -1)) < 1
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                self.assertGreaterEqual(
                    int(getattr(supervisor.lifecycle, "_v812_last_completed_window", -1)),
                    1,
                )
                self.assertFalse(release.is_set())
                self.assertIsNotNone(supervisor._v813_lifecycle_thread)
                self.assertTrue(supervisor._v813_lifecycle_thread.is_alive())
                release.set()
                supervisor.close()
        finally:
            release.set()
            supervisor.close()

        lines = [line for line in output.getvalue().splitlines() if "lifecycle window=" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("lifecycle window=1 complete", lines[0])

    def test_peer_generation_dispatch_is_suppressed(self) -> None:
        from v8 import dedicated_lifecycle_v813 as dedicated
        from v8 import final_save_lifecycle_v812 as lifecycle_runtime

        supervisor = V82DevelopmentalPeerSupervisor(
            read_view=FakeReadView((node((202,)),)),
            submit_proposal=lambda proposal: None,
            watermark=lambda: 1,
            generation=lambda: 64,
            interval_seconds=100.0,
        )
        prior = bool(getattr(dedicated._CONTEXT, "peer_cycle", False))
        dedicated._CONTEXT.peer_cycle = True
        try:
            evaluated = lifecycle_runtime._run_generation_lifecycle(
                supervisor,
                supervisor.read_view.node_records(),
            )
        finally:
            dedicated._CONTEXT.peer_cycle = prior
        self.assertEqual(evaluated, 0)
        self.assertEqual(
            int(getattr(supervisor.lifecycle, "_v812_last_completed_window", -1)),
            -1,
        )


if __name__ == "__main__":
    unittest.main()
