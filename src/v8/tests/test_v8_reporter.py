from __future__ import annotations

import multiprocessing as mp
import os
import queue
import tempfile
import unittest
from pathlib import Path
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
from v8.reporter import (
    SAMPLING_COMPLETE,
    ContinuousProgressBaseline,
    DedicatedReporter,
    format_budget_game_rate_line,
    load_continuous_progress_baseline,
)


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

        expected = ActorProgress(3, "tt01", 7, 1, 0, 2, 3, 4, first_win_step=7)
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
            def __init__(self):
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


class ContinuousProgressBaselineTests(unittest.TestCase):
    def test_log_high_water_and_durable_solutions_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "log.txt"
            games = tuple(f"game-{index}" for index in range(36))
            log_path.write_text(
                f"v8 continuous: games=36 game_ids={','.join(games)}\n"
                "[11:42] 97% - current_run_wins=38.9% "
                "current_run_levels_solved=56.1% current_run_solved_games=14/36\n"
                "[13:16] 48% - current_run_wins=5.6% "
                "current_run_levels_solved=16.7% current_run_solved_games=2/36\n"
                "[13:37] 0% - current_run_wins=0.0% "
                "current_run_levels_solved=0.0% current_run_solved_games=0/36\n",
                encoding="utf-8",
            )

            baseline = load_continuous_progress_baseline(
                log_path,
                games=games,
                durable_solved_games=(f"game-{index}" for index in range(14)),
            )

        self.assertEqual(baseline.solved_games, 14)
        self.assertEqual(baseline.games, 36)
        self.assertEqual(baseline.level_rate, 56.1)
        self.assertEqual(len(baseline.game_ids), 14)

    def test_same_size_different_game_set_does_not_reuse_log_high_water(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "log.txt"
            log_path.write_text(
                "v8 continuous: games=2 game_ids=ez01,ez02\n"
                "[21:04] 62% - current_run_wins=100.0% "
                "current_run_levels_solved=100.0% current_run_solved_games=2/2 "
                "(ez01:B=25,L=25; ez02:B=49,L=51)\n"
                "v8 continuous: games=2 game_ids=gp01,gp02\n",
                encoding="utf-8",
            )

            baseline = load_continuous_progress_baseline(
                log_path,
                games=("gp01", "gp02"),
            )

        self.assertEqual(baseline.game_ids, ())
        self.assertEqual(baseline.solved_games, 0)
        self.assertEqual(baseline.games, 2)
        self.assertEqual(baseline.level_rate, 0.0)

    def test_legacy_high_water_requires_matching_durable_game_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "log.txt"
            log_path.write_text(
                "v8 continuous: games=2\n"
                "[21:04] 62% - current_run_wins=100.0% "
                "current_run_levels_solved=100.0% current_run_solved_games=2/2 "
                "(ez01:B=25,L=25; ez02:B=49,L=51)\n",
                encoding="utf-8",
            )

            matching = load_continuous_progress_baseline(
                log_path,
                games=("ez01", "ez02"),
                durable_solved_games=("ez01", "ez02"),
            )
            different = load_continuous_progress_baseline(
                log_path,
                games=("gp01", "gp02"),
            )

        self.assertEqual(matching.solved_games, 2)
        self.assertEqual(matching.level_rate, 100.0)
        self.assertEqual(different.solved_games, 0)
        self.assertEqual(different.level_rate, 0.0)

    def test_first_report_retains_rates_while_new_budget_starts_at_zero(self) -> None:
        rows = tuple(
            ActorProgress(index + 1, f"game-{index}", 0, 0, 0, 0)
            for index in range(36)
        )
        baseline = ContinuousProgressBaseline(
            game_ids=tuple(f"game-{index}" for index in range(14)),
            solved_games=14,
            games=36,
            level_rate=56.1,
        )

        line = format_budget_game_rate_line(rows, 36_000, baseline)

        self.assertTrue(line.startswith("0% - current_run_wins=38.9%"))
        self.assertIn("current_run_levels_solved=56.1%", line)
        self.assertIn("current_run_solved_games=14/36", line)

    def test_new_win_is_merged_with_durable_solved_game_ids(self) -> None:
        rows = (
            ActorProgress(1, "old", 10, 0, 0, 0),
            ActorProgress(2, "new", 10, 1, 0, 5, first_win_step=10),
        )
        baseline = ContinuousProgressBaseline(
            game_ids=("old",),
            solved_games=1,
            games=2,
            level_rate=50.0,
        )

        line = format_budget_game_rate_line(rows, 200, baseline)

        self.assertIn("current_run_wins=100.0%", line)
        self.assertIn("current_run_levels_solved=100.0%", line)
        self.assertIn("current_run_solved_games=2/2", line)


class DedicatedReporterProcessTests(unittest.TestCase):
    def test_budget_percentage_uses_total_actor_steps(self) -> None:
        line = format_budget_game_rate_line(
            (
                ActorProgress(1, "tt01", 20, 0, 0, 0),
                ActorProgress(2, "tt01", 30, 0, 0, 0),
            ),
            200,
        )

        self.assertTrue(line.startswith("25% - current_run_wins="))

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
            total_steps=200,
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
            self.assertIn("10% - effectiveness", game_line)
            self.assertIn("L=20.0%", game_line)
            self.assertIn("G=50.0%", game_line)
            self.assertIn("M7=0.0%", game_line)
            self.assertNotIn("hypotheses", game_line)
            with self.assertRaises(queue.Empty):
                output.get(timeout=0.15)
        finally:
            reporter.close()
            output.close()
            output.join_thread()

    def test_sampling_complete_stops_periodic_reports_immediately(self) -> None:
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method)
        output = ctx.Queue()
        watermark = ctx.Value("Q", 0)
        reporter = DedicatedReporter(
            ctx,
            watermark=watermark,
            actors=((1, "tt01"),),
            interval_seconds=60.0,
            output_queue=output,
            total_steps=100,
        )
        try:
            reporter.start()
            reporter.progress_queue.put(SAMPLING_COMPLETE)
            self.assertIn("sampling done", output.get(timeout=3.0))
            reporter._process.join(timeout=3.0)
            self.assertEqual(reporter._process.exitcode, 0)
            with self.assertRaises(queue.Empty):
                output.get(timeout=0.1)
        finally:
            reporter.close()
            output.close()
            output.join_thread()


if __name__ == "__main__":
    unittest.main()
