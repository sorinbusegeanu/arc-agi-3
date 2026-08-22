from __future__ import annotations

import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from v8 import runtime_scaling_v841 as v841
from v8.model import MemoryLevel, MemoryUid
from v8.peers import DevelopmentalPeerSupervisor


class FeedbackScalingTests(unittest.TestCase):
    def test_strategy_statistics_use_existing_index_without_graph_scan(self):
        uid = MemoryUid(1, 2)
        row = SimpleNamespace(uid=uid, level=int(MemoryLevel.M7))
        submitted = []
        evidence = []

        class ReadView:
            _node_by_uid = {uid: row}

            def node_records(self, **_kwargs):
                raise AssertionError("feedback must not rescan the live graph")

        supervisor = SimpleNamespace(
            read_view=ReadView(),
            _v845_state_lock=threading.RLock(),
            _existing_proposal=lambda value, **kwargs: (value, kwargs),
            _submit=submitted.append,
            _append_evidence=lambda *args, **kwargs: evidence.append((args, kwargs)),
        )

        accepted = DevelopmentalPeerSupervisor.record_strategy_statistics(
            supervisor,
            uid,
            attempts=1,
            successes=1,
            cost=3.0,
            source_game_hash=7,
        )

        self.assertTrue(accepted)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0][1]["provenance_games"], (7,))

    def test_feedback_worker_uses_disk_queue_and_preserves_order(self):
        release = threading.Event()
        seen = []
        calls = 0

        def callback(rows):
            nonlocal calls
            calls += 1
            if calls == 1:
                release.wait(2.0)
            seen.extend(rows)

        with tempfile.TemporaryDirectory() as root:
            worker = v841._SqliteBatchWorker(
                callback,
                name="v841-feedback-test",
                root=Path(root),
            )
            for value in range(12):
                worker.submit((value,))
            self.assertTrue((Path(root) / "feedback_queue.sqlite3").exists())
            release.set()
            worker.close(timeout=3.0)
            self.assertEqual(seen, list(range(12)))
            submitted, completed, pending, _maximum = worker.metrics()
            self.assertEqual((submitted, completed, pending), (12, 12, 0))


class DeferredScalingTests(unittest.TestCase):
    def test_deferred_worker_drains_more_than_old_four_item_fixed_batch(self):
        runtime = SimpleNamespace(
            _v819_deferred_sources=[object() for _ in range(40)],
            generation=1,
            watermark=1,
            _error_queue=queue.Queue(),
        )
        from v8 import lease_dispatch_continuity_v839 as v839

        original = v839._retry_deferred_batch

        limits = []

        def retry(target, *, limit):
            limits.append(int(limit))
            count = min(len(target._v819_deferred_sources), int(limit))
            del target._v819_deferred_sources[:count]
            return count, count

        v839._retry_deferred_batch = retry
        prior_budget = v841._DEFERRED_TIME_BUDGET_SECONDS
        try:
            v841._DEFERRED_TIME_BUDGET_SECONDS = 1.0
            worker = v841._AdaptiveDeferredRetryWorker(runtime)
            examined, resolved = worker._drain_slice()
            worker.close(timeout=1.0)
        finally:
            v841._DEFERRED_TIME_BUDGET_SECONDS = prior_budget
            v839._retry_deferred_batch = original
        self.assertEqual(examined, 40)
        self.assertEqual(resolved, 40)
        self.assertEqual(runtime._v819_deferred_sources, [])
        self.assertEqual(limits, [1] * 40)

    def test_deferred_worker_stops_between_single_item_retries(self):
        runtime = SimpleNamespace(
            _v819_deferred_sources=[object() for _ in range(3)],
            generation=1,
            watermark=1,
            _error_queue=queue.Queue(),
        )
        from v8 import lease_dispatch_continuity_v839 as v839

        original = v839._retry_deferred_batch
        worker = v841._AdaptiveDeferredRetryWorker(runtime)
        calls = []

        def retry(target, *, limit):
            calls.append(int(limit))
            del target._v819_deferred_sources[:1]
            worker._stop.set()
            return 1, 0

        v839._retry_deferred_batch = retry
        try:
            examined, resolved = worker._drain_slice()
        finally:
            v839._retry_deferred_batch = original
            worker.close(timeout=1.0)
        self.assertEqual((examined, resolved), (1, 0))
        self.assertEqual(calls, [1])
        self.assertEqual(len(runtime._v819_deferred_sources), 2)


class PeerScalingTests(unittest.TestCase):
    def test_cancelled_peer_cycle_does_not_read_the_graph(self):
        class ReadView:
            def node_records(self):
                raise AssertionError("cancelled peer cycle must not read nodes")

        peer = SimpleNamespace(
            _run_lock=threading.Lock(),
            _v841_peer_cancel=threading.Event(),
            read_view=ReadView(),
        )
        peer._v841_peer_cancel.set()

        v841._peer_run_once_v841(peer)

    def test_peer_cycle_stops_after_analysis_when_final_drain_is_requested(self):
        cancel = threading.Event()

        class ReadView:
            @staticmethod
            def node_records():
                return ()

            @staticmethod
            def edge_records():
                return ()

        def analyses(_nodes, _edges):
            cancel.set()
            return {"replay": ()}

        peer = SimpleNamespace(
            _run_lock=threading.Lock(),
            _v841_peer_cancel=cancel,
            read_view=ReadView(),
            _parallel_analyses=analyses,
        )

        from v8.peers_v82 import V82DevelopmentalPeerSupervisor

        base_peer_class = V82DevelopmentalPeerSupervisor.__mro__[1]
        base_peer_class.__dict__["run_once"](peer)

    def test_dirty_gate_skips_unchanged_peer_input(self):
        class Arena:
            sequence = 2

        class FakePeer:
            def __init__(self):
                self.read_view = SimpleNamespace(_nodes=(Arena(),), _edges=())
                self._cycles = 0
                self._last_developmental_cut = None
                self._v841_last_input_token = None

            @staticmethod
            def current_generation():
                return 1

            @staticmethod
            def current_watermark():
                return 1

        peer = FakePeer()
        original = v841._BASE_PEER_RUN_ONCE

        def base(target):
            target._cycles += 1

        v841._BASE_PEER_RUN_ONCE = base
        try:
            v841._peer_run_once_v841(peer)
            v841._peer_run_once_v841(peer)
            self.assertEqual(peer._cycles, 1)
            peer.read_view._nodes[0].sequence = 4
            v841._peer_run_once_v841(peer)
            self.assertEqual(peer._cycles, 2)
        finally:
            v841._BASE_PEER_RUN_ONCE = original

    def test_v845_wraps_v841_peer_authority(self):
        from v8 import snapshot_state_consistency_v845 as v845
        from v8.peers_v82 import V82DevelopmentalPeerSupervisor

        self.assertIs(V82DevelopmentalPeerSupervisor.run_once, v845._peer_run_once_v845)
        self.assertIs(v845._BASE_PEER_RUN_ONCE, v841._peer_run_once_v841)
        self.assertIs(
            V82DevelopmentalPeerSupervisor._parallel_analyses,
            v841._parallel_analyses_v841,
        )


class OptimizerScalingTests(unittest.TestCase):
    def test_full_validator_queue_routes_to_nonblocking_overflow(self):
        from v8 import trajectory_optimizer_v818 as v818

        game = "g1"
        validation_queue = queue.Queue(maxsize=1)
        validation_queue.put(object())
        service = SimpleNamespace(
            _v818_validator_lock=threading.Lock(),
            _v818_game_queues={game: validation_queue},
            _v818_waiting_games=set(),
            _v818_validator_threads={},
            _v818_max_validators=1,
            _stop=threading.Event(),
        )
        candidate = SimpleNamespace(
            candidate_id="c1",
            edit_kind="DELETE_ACTION",
            actions=(1,),
            cost=1,
            source=SimpleNamespace(anchor=SimpleNamespace(source_id=game)),
        )

        original_ensure = v818._ensure_validator
        original_start = v818._start_waiting_validators
        v818._ensure_validator = lambda *_args, **_kwargs: None
        v818._start_waiting_validators = lambda *_args, **_kwargs: None
        try:
            started = time.perf_counter()
            self.assertTrue(v841._route_candidate_base_v841(service, candidate))
            self.assertLess(time.perf_counter() - started, 0.1)
            dispatcher = service._v841_candidate_overflow
            self.assertEqual(dispatcher.pending_count(), 1)
            validation_queue.get_nowait()
            deadline = time.time() + 1.0
            while dispatcher.pending_count() and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(dispatcher.pending_count(), 0)
            self.assertIs(validation_queue.get_nowait(), candidate)
            dispatcher.close(drain=False, timeout=1.0)
        finally:
            v818._ensure_validator = original_ensure
            v818._start_waiting_validators = original_start

    def test_v830_and_v819_wrappers_remain_final_route_authorities(self):
        from v8 import adaptive_learning_allocation_v819 as v819
        from v8 import optimizer_budget_control_v830 as v830
        from v8 import trajectory_optimizer_v818 as v818

        self.assertIs(v818._route_candidate, v830._route_candidate_v830)
        self.assertIs(v830._BASE_ROUTE_CANDIDATE, v819._route_candidate_v819)
        self.assertIs(v819._BASE_ROUTE_CANDIDATE, v841._route_candidate_base_v841)

    def test_post_sampling_quiescence_includes_optimizer_overflow(self):
        pending = SimpleNamespace(value=1)
        worker = SimpleNamespace(
            pending_count=lambda: pending.value,
            _wake=threading.Event(),
        )
        service = SimpleNamespace(
            _v841_candidate_overflow=worker,
            _sources=SimpleNamespace(unfinished_tasks=0),
            _lock=threading.Lock(),
            _active_validations=0,
            _v818_validator_lock=threading.Lock(),
            _v818_game_queues={},
            inbox=Path("unused-v841-inbox"),
        )
        runtime = SimpleNamespace(
            _sampling_complete=True,
            _v841_optimizer_drain=False,
            _v814_trajectory_optimizer=service,
        )
        original = v841._BASE_RUNTIME_IS_QUIESCENT
        v841._BASE_RUNTIME_IS_QUIESCENT = lambda _runtime: True
        try:
            self.assertFalse(v841._runtime_is_quiescent_v841(runtime))
            self.assertTrue(worker._wake.is_set())
            pending.value = 0
            self.assertTrue(v841._runtime_is_quiescent_v841(runtime))
        finally:
            v841._BASE_RUNTIME_IS_QUIESCENT = original

    def test_post_sampling_preserves_durable_inbox_after_ram_work_drains(self):
        with tempfile.TemporaryDirectory() as root:
            inbox = Path(root)
            (inbox / "pending.json").write_text("{}", encoding="utf-8")
            service = SimpleNamespace(
                _v841_candidate_overflow=None,
                _v841_preserve_inbox_on_shutdown=True,
                _sources=SimpleNamespace(unfinished_tasks=0),
                _lock=threading.Lock(),
                _active_validations=0,
                _v818_validator_lock=threading.Lock(),
                _v818_game_queues={},
                inbox=inbox,
            )

            self.assertTrue(v841._optimizer_idle_v841(service))

            service._sources.unfinished_tasks = 1
            self.assertFalse(v841._optimizer_idle_v841(service))

    def test_shutdown_mode_stops_admitting_durable_inbox(self):
        calls = []
        service = SimpleNamespace(_v841_preserve_inbox_on_shutdown=True)
        original = v841._BASE_INGEST_INBOX
        v841._BASE_INGEST_INBOX = lambda target: calls.append(target)
        try:
            v841._ingest_inbox_v841(service)
            service._v841_preserve_inbox_on_shutdown = False
            v841._ingest_inbox_v841(service)
        finally:
            v841._BASE_INGEST_INBOX = original

        self.assertEqual(calls, [service])

    def test_normal_runtime_close_drains_optimizer_before_base_close(self):
        calls = []
        runtime = SimpleNamespace(
            _v814_trajectory_optimizer=object(),
            _v841_optimizer_drain=False,
        )

        def wait_quiescent(**kwargs):
            self.assertTrue(runtime._v841_optimizer_drain)
            calls.append(("drain", kwargs))

        runtime.wait_quiescent = wait_quiescent
        original = v841._BASE_RUNTIME_CLOSE
        v841._BASE_RUNTIME_CLOSE = lambda _runtime, **kwargs: calls.append(
            ("close", kwargs)
        )
        try:
            v841._runtime_close_v841(runtime, normal=True, timeout=300.0)
        finally:
            v841._BASE_RUNTIME_CLOSE = original

        self.assertEqual([name for name, _kwargs in calls], ["drain", "close"])
        self.assertEqual(calls[0][1]["timeout"], 300.0)
        self.assertFalse(calls[0][1]["resume_peers"])
        self.assertFalse(calls[0][1]["settle_peers"])
        self.assertFalse(runtime._v841_optimizer_drain)


if __name__ == "__main__":
    unittest.main()
