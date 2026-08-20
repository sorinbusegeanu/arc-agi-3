from __future__ import annotations

import threading
import unittest

import v8  # noqa: F401 - installs the complete runtime stack
from v8.model import MemoryUid
from v8.peers_v82 import V82DevelopmentalPeerSupervisor


class _EmptyReadView:
    def node_records(self, *, level=None):
        return ()

    def edge_records(self):
        return ()

    def source_games(self, uid, *, max_depth=8):
        return frozenset()


class _BlockingItemsDict(dict):
    def __init__(self, *args, entered: threading.Event, release: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = entered
        self.release = release

    def items(self):
        for item in super().items():
            self.entered.set()
            if not self.release.wait(2.0):
                raise TimeoutError("test did not release peer-state serialization")
            yield item


class SnapshotStateConsistencyV845Tests(unittest.TestCase):
    @staticmethod
    def _supervisor():
        return V82DevelopmentalPeerSupervisor(
            read_view=_EmptyReadView(),
            submit_proposal=lambda _proposal: None,
            watermark=lambda: 1,
            generation=lambda: 1,
        )

    def test_seen_dictionary_cannot_mutate_during_state_serialization(self):
        supervisor = self._supervisor()
        entered = threading.Event()
        release = threading.Event()
        mutation_done = threading.Event()
        errors = []
        supervisor._seen = _BlockingItemsDict(
            {("seed", 1, 2): 1},
            entered=entered,
            release=release,
        )

        def capture():
            try:
                supervisor.state_dict()
            except BaseException as exc:
                errors.append(exc)

        def mutate():
            try:
                supervisor._fresh("new", MemoryUid(3, 4), 2)
            except BaseException as exc:
                errors.append(exc)
            finally:
                mutation_done.set()

        capture_thread = threading.Thread(target=capture)
        capture_thread.start()
        self.assertTrue(entered.wait(1.0))

        mutation_thread = threading.Thread(target=mutate)
        mutation_thread.start()
        self.assertFalse(mutation_done.wait(0.05))

        release.set()
        capture_thread.join(2.0)
        mutation_thread.join(2.0)

        self.assertFalse(capture_thread.is_alive())
        self.assertFalse(mutation_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(mutation_done.is_set())
        self.assertIn(("new", 3, 4), supervisor._seen)

    def test_transfer_mutation_and_snapshot_share_one_state_lock(self):
        supervisor = self._supervisor()
        entered = threading.Event()
        release = threading.Event()
        mutation_done = threading.Event()
        snapshot_done = threading.Event()
        errors = []
        original_record_trial = supervisor.transfer.record_trial

        def blocked_record_trial(*args, **kwargs):
            entered.set()
            if not release.wait(2.0):
                raise TimeoutError("test did not release transfer mutation")
            return original_record_trial(*args, **kwargs)

        supervisor.transfer.record_trial = blocked_record_trial

        def mutate():
            try:
                supervisor.record_transfer_trial(
                    MemoryUid(5, 6),
                    target_game_hash=7,
                    metric_on=1.0,
                    metric_off=0.0,
                )
            except BaseException as exc:
                errors.append(exc)
            finally:
                mutation_done.set()

        def capture():
            try:
                supervisor.state_dict()
            except BaseException as exc:
                errors.append(exc)
            finally:
                snapshot_done.set()

        mutation_thread = threading.Thread(target=mutate)
        mutation_thread.start()
        self.assertTrue(entered.wait(1.0))

        capture_thread = threading.Thread(target=capture)
        capture_thread.start()
        self.assertFalse(snapshot_done.wait(0.05))

        release.set()
        mutation_thread.join(2.0)
        capture_thread.join(2.0)

        self.assertFalse(mutation_thread.is_alive())
        self.assertFalse(capture_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(mutation_done.is_set())
        self.assertTrue(snapshot_done.is_set())


if __name__ == "__main__":
    unittest.main()
