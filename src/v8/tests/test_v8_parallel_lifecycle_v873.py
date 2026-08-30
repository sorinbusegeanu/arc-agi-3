from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import v8
from v8 import final_save_lifecycle_v812 as clock
from v8 import parallel_lifecycle_v873 as parallel
from v8.arena import NodeRecord
from v8.lifecycle import LifecycleController
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState
from v8.peers_v82 import V82DevelopmentalPeerSupervisor
from v8.pruning import PruningPlanner


def _node(uid: MemoryUid, *, state: CognitiveState = CognitiveState.ACTIVE) -> NodeRecord:
    return NodeRecord(
        uid,
        (int(uid.hi) ^ int(uid.lo)) & ((1 << 64) - 1),
        int(MemoryLevel.M1),
        int(MemoryType.CONTINGENCY),
        (int(uid.lo),),
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
        int(state),
        int(ValidationState.STRUCTURAL),
        0.0,
        0.0,
        0.0,
    )


class _CountingReadView:
    def __init__(self, nodes, edges=()) -> None:
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.node_reads = 0
        self.edge_reads = 0

    def node_records(self, *, level=None):
        self.node_reads += 1
        if level is None:
            return self.nodes
        return tuple(row for row in self.nodes if int(row.level) == int(level))

    def edge_records(self):
        self.edge_reads += 1
        return self.edges

    def source_games(self, uid, *, max_depth=8):
        del uid, max_depth
        return frozenset()


def _supervisor(nodes):
    view = _CountingReadView(nodes)
    submitted = []
    supervisor = V82DevelopmentalPeerSupervisor(
        read_view=view,
        submit_proposal=submitted.append,
        watermark=lambda: 10,
        generation=lambda: 64,
        interval_seconds=100.0,
    )
    supervisor.set_candidate_budget(512)
    return supervisor, view, submitted


class ParallelLifecycleV873Tests(unittest.TestCase):
    def tearDown(self) -> None:
        # Tests create lazy executors without starting the supervisor lifecycle.
        for supervisor in tuple(getattr(self, "_supervisors", ())):
            executor = getattr(supervisor, "_v873_lifecycle_executor", None)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                supervisor._v873_lifecycle_executor = None

    def _track(self, supervisor):
        supervisors = list(getattr(self, "_supervisors", ()))
        supervisors.append(supervisor)
        self._supervisors = supervisors
        return supervisor

    def test_window_uses_one_cut_and_resumable_bucket_slices(self) -> None:
        nodes = tuple(_node(MemoryUid(index + 1, index * 17 + 3)) for index in range(128))
        supervisor, view, _submitted = _supervisor(nodes)
        self._track(supervisor)

        iterations = 0
        while int(supervisor.lifecycle._v812_last_completed_window) < 1:
            parallel._run_parallel_lifecycle_iteration_v873(supervisor)
            iterations += 1
            self.assertLess(iterations, 25)

        self.assertEqual(view.node_reads, 1)
        self.assertEqual(view.edge_reads, 1)
        self.assertEqual(supervisor.lifecycle._v812_last_completed_window, 1)
        self.assertEqual(supervisor.lifecycle._v812_active_window, -1)
        self.assertEqual(supervisor.lifecycle._v812_next_bucket, 0)
        self.assertEqual(supervisor.lifecycle._v873_retirement_window, -1)

    def test_lifecycle_bucket_analysis_runs_on_multiple_workers(self) -> None:
        if parallel._worker_count(8) < 2:
            self.skipTest("parallel lifecycle worker count is one")
        nodes = tuple(_node(MemoryUid(index + 1, index + 1)) for index in range(64))
        supervisor, _view, _submitted = _supervisor(nodes)
        self._track(supervisor)
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        thread_ids = set()
        entrants = 0
        original = parallel._analyze_lifecycle_bucket

        def observed(*args, **kwargs):
            nonlocal entrants
            with lock:
                thread_ids.add(threading.get_ident())
                entrants += 1
                should_wait = entrants <= 2
            if should_wait:
                barrier.wait(timeout=2.0)
            return original(*args, **kwargs)

        with patch.object(parallel, "_analyze_lifecycle_bucket", observed):
            parallel._run_parallel_lifecycle_iteration_v873(supervisor)

        self.assertGreaterEqual(len(thread_ids), 2)
        self.assertEqual(supervisor.lifecycle._v812_next_bucket, 8)

    def test_cancelled_slice_does_not_advance_cursor_or_low_windows(self) -> None:
        nodes = tuple(_node(MemoryUid(index + 1, index + 9)) for index in range(16))
        supervisor, _view, _submitted = _supervisor(nodes)
        self._track(supervisor)
        buckets = tuple(
            tuple(row for row in nodes if parallel._bucket(row.uid, 64) == bucket)
            for bucket in range(64)
        )
        supervisor.lifecycle._v812_next_bucket = 0
        before = dict(supervisor.lifecycle._low_windows)
        supervisor._pause.set()

        evaluated = parallel._analyze_lifecycle_slice(
            supervisor,
            buckets,
            window=1,
            start=0,
            stop=8,
        )

        self.assertEqual(evaluated, 0)
        self.assertEqual(supervisor.lifecycle._v812_next_bucket, 0)
        self.assertEqual(supervisor.lifecycle._low_windows, before)

    def test_retirement_cursor_round_trips_through_lifecycle_state(self) -> None:
        controller = LifecycleController()
        controller._v812_active_window = 7
        controller._v812_next_bucket = int(clock._LIFECYCLE_BUCKETS)
        controller._v873_retirement_window = 7
        controller._v873_retirement_next_bucket = 19
        state = controller.state_dict()

        restored = LifecycleController()
        restored.load_state(state)

        self.assertEqual(restored._v873_retirement_window, 7)
        self.assertEqual(restored._v873_retirement_next_bucket, 19)
        self.assertEqual(restored._v812_active_window, 7)
        self.assertEqual(restored._v812_next_bucket, clock._LIFECYCLE_BUCKETS)

    def test_pruning_scan_honors_cancellation_without_partial_candidates(self) -> None:
        cancel = threading.Event()
        cancel.set()
        pending = _node(MemoryUid(1, 2), state=CognitiveState.RETIRE_PENDING)

        rows = PruningPlanner().candidates(
            (pending,),
            (),
            cancel_event=cancel,
        )

        self.assertEqual(rows, ())


if __name__ == "__main__":
    unittest.main()
