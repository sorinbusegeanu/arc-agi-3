from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from v8.actor import ActorJob, ActorResult, run_actor_jobs


class _ThreadProcess:
    def __init__(self, *, target, kwargs, name, **_ignored) -> None:
        self.name = name
        self._target = target
        self._kwargs = kwargs
        self._thread: threading.Thread | None = None
        self._exitcode = None

    def start(self) -> None:
        def run() -> None:
            try:
                self._target(**self._kwargs)
            except BaseException:
                self._exitcode = 1
                return
            self._exitcode = 0

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    @property
    def exitcode(self):
        return self._exitcode

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def terminate(self) -> None:
        self._kwargs["stop_event"].set()

    def join(self, timeout=None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


class ActorGraphStartupTests(unittest.TestCase):
    def test_all_actor_views_are_ready_before_sampling_and_peers_resume(self) -> None:
        events: list[str] = []
        peers = Mock()
        peers.pause.side_effect = lambda: events.append("peers-pause")
        peers.wait_idle.side_effect = lambda _timeout: events.append("peers-idle") or True
        peers.resume.side_effect = lambda: events.append("peers-resume")
        ctx = SimpleNamespace(
            Event=threading.Event,
            Queue=lambda **kwargs: queue.Queue(**kwargs),
            Process=lambda **kwargs: _ThreadProcess(**kwargs),
        )
        stop = threading.Event()
        runtime = SimpleNamespace(
            peers=peers,
            start=lambda: events.append("runtime-start"),
            wait_quiescent=lambda **_kwargs: events.append("runtime-drained"),
            _mp_ctx=ctx,
            _stage_rings=(SimpleNamespace(attachment_args=lambda: {}),),
            shard_descriptors=(),
            _watermark=object(),
            _stop=stop,
            _actor_throttle=object(),
            _snapshot_freeze=threading.Event(),
            record_actor_results=Mock(),
        )
        jobs = (
            ActorJob(1, "tt01", 1, 0),
            ActorJob(2, "tt02", 1, 1),
        )
        ready_count = 0
        ready_lock = threading.Lock()

        def actor(**kwargs) -> None:
            nonlocal ready_count
            self.assertFalse(kwargs["startup_gate"].is_set())
            with ready_lock:
                ready_count += 1
                events.append(f"actor-{kwargs['job'].actor_id}-ready")
            kwargs["startup_ready"].set()
            self.assertTrue(kwargs["startup_gate"].wait(1.0))
            with ready_lock:
                self.assertEqual(ready_count, len(jobs))
            kwargs["result_queue"].put(
                ActorResult(kwargs["job"].actor_id, kwargs["job"].game_id, 1, 0, 0, 0, 0)
            )

        with patch("v8.actor.actor_worker", side_effect=actor):
            results = run_actor_jobs(runtime, jobs, timeout=2.0)

        self.assertEqual(tuple(row.actor_id for row in results), (1, 2))
        self.assertLess(events.index("peers-pause"), events.index("runtime-start"))
        self.assertLess(events.index("runtime-drained"), events.index("actor-1-ready"))
        self.assertLess(events.index("actor-2-ready"), events.index("peers-resume"))


if __name__ == "__main__":
    unittest.main()
