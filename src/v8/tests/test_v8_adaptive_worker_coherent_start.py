from __future__ import annotations

import queue
import threading
import unittest
from unittest.mock import patch

import v8  # noqa: F401 - install chronological runtime stack
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_learning_allocation_v819_worker_fix as worker_fix
from v8.model import MemoryUid


class AdaptiveWorkerCoherentStartTests(unittest.TestCase):
    def test_worker_warms_record_cuts_before_signalling_ready(self) -> None:
        assignments = queue.Queue()
        events = queue.Queue()
        ready = threading.Event()
        stop = threading.Event()
        seen = []

        lease = v819.ActorLease(
            1,
            22,
            "tt01",
            1,
            7,
            None,
            0.1,
            1000,
            v819.SamplingMode.DISCOVERY,
            MemoryUid.zero(),
        )
        assignments.put(lease)
        assignments.put(None)

        class FakeView:
            def __init__(self, _descriptors, *, refresh_interval_seconds, record_cuts):
                if ready.is_set():
                    raise AssertionError("worker signalled ready before warming its graph cut")
                record_cuts[("nodes", "warm")] = (("coherent",), 2)

            def close(self):
                return None

        def actor_worker(**kwargs):
            seen.append(
                (
                    ready.is_set(),
                    kwargs["record_cuts"].get(("nodes", "warm")),
                )
            )

        with (
            patch("v8.publication.LiveReadView", FakeView),
            patch("v8.actor.actor_worker", actor_worker),
        ):
            worker_fix._worker_until_completed_win(
                worker_id=22,
                assignment_queue=assignments,
                event_queue=events,
                ready_event=ready,
                experience_ring_args={},
                read_descriptors=(),
                watermark=None,
                stop_event=stop,
                actor_throttle=None,
                snapshot_freeze=None,
                trajectory_root="unused",
            )

        self.assertTrue(ready.is_set())
        self.assertEqual(seen, [(True, (("coherent",), 2))])


if __name__ == "__main__":
    unittest.main()
