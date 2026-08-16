from __future__ import annotations

import queue
import threading
import time
import unittest
from unittest.mock import Mock

import v8
from v8.arena import NodeRecord
from v8.final_save_lifecycle_v812 import _run_generation_lifecycle
from v8.intelligence_loop_v087 import process_replay_cognition
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState
from v8.peers_v82 import V82DevelopmentalPeerSupervisor
from v8.snapshot import SnapshotResult, SnapshotService


class _AliveProcess:
    def is_alive(self) -> bool:
        return True


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


def node(
    key,
    *,
    support=5,
    significance=1.0,
    learning=1.0,
    transfer=0.5,
    explanatory=1.0,
    cognitive=CognitiveState.ACTIVE,
):
    uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
    return NodeRecord(
        uid,
        (uid.hi ^ uid.lo) & ((1 << 64) - 1),
        int(MemoryLevel.M1),
        int(MemoryType.CONTINGENCY),
        tuple(key),
        int(support),
        float(significance),
        0.0,
        float(learning),
        float(transfer),
        float(explanatory),
        0.0,
        1.0,
        10,
        0,
        int(cognitive),
        int(ValidationState.STRUCTURAL),
        0.0,
        0.0,
        0.0,
    )


class SnapshotAckRegressionTests(unittest.TestCase):
    def test_consistent_capture_treats_empty_poll_as_wait_not_failure(self) -> None:
        service = object.__new__(SnapshotService)
        service._requests = queue.Queue(maxsize=1)
        service._acks = queue.Queue()
        service._process = _AliveProcess()

        def delayed_ack() -> None:
            time.sleep(1.05)
            service._acks.put(("captured", 7))

        thread = threading.Thread(target=delayed_ack)
        thread.start()
        service.request_consistent_capture(
            7,
            11,
            generation=3,
            auxiliary_state="{}",
            timeout=2.5,
        )
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())

    def test_final_save_preempts_background_and_tolerates_empty_poll(self) -> None:
        service = object.__new__(SnapshotService)
        service._requests = queue.Queue(maxsize=1)
        service._acks = queue.Queue()
        service._process = _AliveProcess()
        service._preempt = threading.Event()

        result = SnapshotResult(9, 22, "/tmp/final", "digest", True, 4)

        def delayed_ack() -> None:
            time.sleep(1.05)
            service._acks.put(("ok", result))

        thread = threading.Thread(target=delayed_ack)
        thread.start()
        observed = service.request_final(
            9,
            22,
            generation=4,
            auxiliary_state="{}",
            timeout=2.5,
        )
        thread.join(timeout=1.0)
        self.assertIs(observed, result)
        self.assertTrue(service._preempt.is_set())


class ReplayShutdownRegressionTests(unittest.TestCase):
    def test_full_graph_promotion_is_computed_once_for_replay_batch(self) -> None:
        rows = (node((101,)), node((202,)))
        supervisor = V82DevelopmentalPeerSupervisor(
            read_view=FakeReadView(rows),
            submit_proposal=lambda proposal: None,
            watermark=lambda: 50,
            generation=lambda: 130,
            interval_seconds=100.0,
        )
        supervisor.promotion.propose = Mock(return_value=())

        metrics = process_replay_cognition(supervisor)

        self.assertGreater(metrics["processed"], 0)
        self.assertEqual(supervisor.promotion.propose.call_count, 1)

    def test_pause_skips_replay_before_graph_materialization(self) -> None:
        view = FakeReadView((node((303,)),))
        view.node_records = Mock(side_effect=AssertionError("graph should not be read"))
        supervisor = V82DevelopmentalPeerSupervisor(
            read_view=view,
            submit_proposal=lambda proposal: None,
            watermark=lambda: 50,
            generation=lambda: 130,
            interval_seconds=100.0,
        )
        supervisor.pause()

        metrics = process_replay_cognition(supervisor)

        self.assertEqual(metrics["processed"], 0)
        view.node_records.assert_not_called()


class DeterministicLifecycleWindowTests(unittest.TestCase):
    def test_lifecycle_ages_by_generation_window_not_replay_selection(self) -> None:
        row = node(
            (404,),
            support=1,
            significance=0.0,
            learning=0.0,
            transfer=0.0,
            explanatory=0.0,
        )
        submitted = []
        generation = [64]
        supervisor = V82DevelopmentalPeerSupervisor(
            read_view=FakeReadView((row,)),
            submit_proposal=submitted.append,
            watermark=lambda: 1,
            generation=lambda: generation[0],
            interval_seconds=100.0,
        )

        self.assertIsNone(supervisor.lifecycle.decide(row))
        self.assertNotIn(row.uid, supervisor.lifecycle._low_windows)

        for _ in range(16):
            _run_generation_lifecycle(supervisor, (row,))
        self.assertEqual(supervisor.lifecycle._low_windows[row.uid], 1)

        for _ in range(16):
            _run_generation_lifecycle(supervisor, (row,))
        self.assertEqual(supervisor.lifecycle._low_windows[row.uid], 1)

        generation[0] = 128
        for _ in range(16):
            _run_generation_lifecycle(supervisor, (row,))
        self.assertEqual(supervisor.lifecycle._low_windows[row.uid], 2)

        generation[0] = 192
        for _ in range(16):
            _run_generation_lifecycle(supervisor, (row,))
        self.assertEqual(supervisor.lifecycle._low_windows[row.uid], 3)
        self.assertTrue(
            any(
                int(proposal.cognitive_state) == int(CognitiveState.QUARANTINED)
                for proposal in submitted
            )
        )


if __name__ == "__main__":
    unittest.main()
