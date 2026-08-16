from __future__ import annotations

import multiprocessing as mp
import os
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from v8.actor import (
    ActorJob,
    ActorLearningBatch,
    ActorProgress,
    ActorResult,
    _publish_progress,
    run_actor_jobs,
)
from v8.evidence import EvidenceLedger, EvidenceRecord
from v8.model import MemoryLevel, MemoryUid, ValidationState
from v8.reporter import DedicatedReporter


def evidence(evidence_id: str, *, watermark: int = 1) -> EvidenceRecord:
    return EvidenceRecord.for_uid(
        evidence_id,
        MemoryUid(1, 2),
        evidence_kind="contingency_recurrence",
        watermark=watermark,
        raw_value=1.0,
        normalized_value=1.0,
        developmental_stage=int(MemoryLevel.M1),
        validation_state=int(ValidationState.VALIDATED),
    )


class EvidenceMirrorTests(unittest.TestCase):
    def test_listener_replays_existing_rows_and_receives_new_accepted_rows(self) -> None:
        ledger = EvidenceLedger()
        first = evidence("first")
        second = evidence("second")
        ledger.append(first)
        mirrored: list[EvidenceRecord] = []

        ledger.set_append_listener(mirrored.append, replay=True)
        self.assertEqual(mirrored, [first])
        self.assertTrue(ledger.append(second))
        self.assertFalse(ledger.append(second))
        self.assertEqual(mirrored, [first, second])

        ledger.set_append_listener(None)
        ledger.append(evidence("third"))
        self.assertEqual(mirrored, [first, second])


class ActorProgressFanoutTests(unittest.TestCase):
    def test_actor_progress_is_sent_to_parent_and_dedicated_reporter(self) -> None:
        parent: queue.Queue[ActorProgress] = queue.Queue()
        reporting: queue.Queue[ActorProgress] = queue.Queue()
        job = ActorJob(3, "tt01", 10, 0)

        _publish_progress(
            parent,
            reporting,
            job=job,
            steps=7,
            wins=1,
            failures=0,
            levels_completed=2,
            replans=3,
            planned_steps=4,
        )

        expected = ActorProgress(3, "tt01", 7, 1, 0, 2, 3, 4)
        self.assertEqual(parent.get_nowait(), expected)
        self.assertEqual(reporting.get_nowait(), expected)

    def test_parent_reporting_deadline_is_checked_before_busy_queue_is_empty(self) -> None:
        job = ActorJob(1, "tt01", 1, 0)
        pending = ActorLearningBatch(1, "tt01")
        result = ActorResult(1, "tt01", 1, 0, 0, 0, 0, pending_learning=pending)

        class BusyProgressQueue:
            def __init__(self) -> None:
                self.remaining = 300

            def get_nowait(self):
                if self.remaining <= 0:
                    raise queue.Empty
                self.remaining -= 1
                return ActorProgress(1, "tt01", 1, 0, 0, 0)

        class ResultQueue:
            def __init__(self) -> None:
                self.pending = [result]

            def get_nowait(self):
                if not self.pending:
                    raise queue.Empty
                return self.pending.pop()

        class Process:
            exitcode = 0

            def start(self) -> None:
                pass

            def is_alive(self) -> bool:
                return False

            def join(self, timeout=None) -> None:
                pass

        progress = BusyProgressQueue()
        queues = iter((ResultQueue(), progress))
        context = SimpleNamespace(
            Queue=lambda **_kwargs: next(queues),
            Process=lambda **_kwargs: Process(),
        )
        runtime = SimpleNamespace(
            start=Mock(),
            _mp_ctx=context,
            _stage_rings=(SimpleNamespace(attachment_args=lambda: {}),),
            shard_descriptors=(),
            _watermark=object(),
            _stop=object(),
            _actor_throttle=object(),
            _snapshot_freeze=object(),
            record_actor_results=Mock(),
        )
        callback_depths: list[int] = []

        with patch("v8.actor.time.monotonic", side_effect=(0.0, 61.0, 61.0)):
            rows = run_actor_jobs(
                runtime,
                (job,),
                progress_interval_seconds=60.0,
                progress_callback=lambda _rows: callback_depths.append(progress.remaining),
            )

        self.assertEqual(rows, (result,))
        self.assertGreater(callback_depths[0], 0)
        self.assertEqual(callback_depths[-1], 0)
        runtime.record_actor_results.assert_called_once_with((pending,))

    def test_failed_process_start_only_cleans_up_started_processes(self) -> None:
        jobs = (ActorJob(1, "tt01", 1, 0), ActorJob(2, "tt02", 1, 1))

        class EmptyQueue:
            def get_nowait(self):
                raise queue.Empty

        class Process:
            def __init__(self, *, fail_start: bool) -> None:
                self.fail_start = fail_start
                self.started = False
                self.terminated = False
                self.joined = False

            def start(self) -> None:
                if self.fail_start:
                    raise AttributeError("cannot pickle actor target")
                self.started = True

            def is_alive(self) -> bool:
                if not self.started:
                    raise AssertionError("queried unstarted process")
                return True

            def terminate(self) -> None:
                self.terminated = True

            def join(self, timeout=None) -> None:
                if not self.started:
                    raise AssertionError("joined unstarted process")
                self.joined = True

        created: list[Process] = []

        def make_process(**_kwargs):
            process = Process(fail_start=bool(created))
            created.append(process)
            return process

        context = SimpleNamespace(
            Queue=lambda **_kwargs: EmptyQueue(),
            Process=make_process,
        )
        runtime = SimpleNamespace(
            start=Mock(),
            _mp_ctx=context,
            _stage_rings=(SimpleNamespace(attachment_args=lambda: {}),),
            shard_descriptors=(),
            _watermark=object(),
            _stop=object(),
            _actor_throttle=object(),
            _snapshot_freeze=object(),
            record_actor_results=Mock(),
        )

        with self.assertRaisesRegex(AttributeError, "cannot pickle"):
            run_actor_jobs(runtime, jobs)

        self.assertTrue(created[0].terminated)
        self.assertTrue(created[0].joined)
        self.assertFalse(created[1].started)
        self.assertFalse(created[1].joined)


class DedicatedReporterProcessTests(unittest.TestCase):
    def test_process_reports_progress_only_without_hypothesis_stdout(self) -> None:
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method)
        watermark = ctx.Value("Q", 10)
        output = ctx.Queue()
        reporter = DedicatedReporter(
            ctx,
            watermark=watermark,
            actors=((1, "tt01"), (2, "tt02")),
            interval_seconds=0.3,
            output_queue=output,
        )
        try:
            reporter.start()
            self.assertNotEqual(reporter._process.pid, os.getpid())
            reporter.progress_queue.put(ActorProgress(1, "tt01", 20, 1, 0, 2))
            reporter.publish_evidence(evidence("h01-a"))
            reporter.publish_evidence(evidence("h01-b"))

            with self.assertRaises(queue.Empty):
                output.get(timeout=0.1)

            game_line = output.get(timeout=3.0)
            self.assertIn("current_run_wins=50.0%", game_line)
            self.assertIn("current_run_levels_solved=20.0%", game_line)
            self.assertIn("current_run_solved_games=1/2 (tt01:B=20,L=20)", game_line)
            self.assertNotIn("hypotheses", game_line)
            with self.assertRaises(queue.Empty):
                output.get(timeout=0.15)
        finally:
            reporter.close()
            output.close()
            output.join_thread()


if __name__ == "__main__":
    unittest.main()
