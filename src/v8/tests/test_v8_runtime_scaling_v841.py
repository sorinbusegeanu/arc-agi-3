from __future__ import annotations

import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from v8 import runtime_scaling_v841 as v841


class FeedbackScalingTests(unittest.TestCase):
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

        def retry(target, *, limit):
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


class PeerScalingTests(unittest.TestCase):
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

    def test_final_peer_authorities_are_v841(self):
        from v8.peers_v82 import V82DevelopmentalPeerSupervisor

        self.assertIs(V82DevelopmentalPeerSupervisor.run_once, v841._peer_run_once_v841)
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


if __name__ == "__main__":
    unittest.main()
