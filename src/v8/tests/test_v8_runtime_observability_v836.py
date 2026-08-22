from __future__ import annotations

import io
import queue
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import v8  # noqa: F401 - installs the chronological runtime stack
from v8 import reporter
from v8 import runtime_observability_v836 as observability


class RuntimeObservabilityV836Tests(unittest.TestCase):
    def test_stdout_is_mirrored_to_root_log_txt(self):
        with tempfile.TemporaryDirectory() as root:
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                with observability.stdout_log_context(
                    ["continuous-run", "--root", root]
                ):
                    print("mirrored-line", flush=True)
            self.assertIn("mirrored-line", terminal.getvalue())
            self.assertEqual(
                (Path(root) / "log.txt").read_text(encoding="utf-8"),
                "mirrored-line\n",
            )

    def test_continuous_progress_is_file_only_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            terminal = io.StringIO()
            progress = (
                "[12:34] optimizer game=ez01 level=1 status=START",
                "[12:34] learning state game=ez01 UNSOLVED->SOLVED_OPTIMIZING",
                "[12:34] frontier game=ez01 level=1 source=SAMPLER",
                "[12:34] trajectory optimization validators=1 games=1",
                "[12:34] sampling allocation unsolved=0 optimizing=1",
                "[12:34] lifecycle window=1 complete",
                "[12:34] hypotheses H01=INSUFFICIENT_EVIDENCE",
            )
            with redirect_stdout(terminal):
                with observability.stdout_log_context(
                    ["continuous-run", "--root", root]
                ):
                    print("v8 continuous: games=1", flush=True)
                    print("[12:34] graph source=empty(no-snapshot) nodes=0", flush=True)
                    print(
                        "[12:34] 20% - current_run_wins=100.0% "
                        "current_run_levels_solved=100.0% "
                        "current_run_solved_games=1/1 (ez01:B=25,L=60)",
                        flush=True,
                    )
                    print(
                        "[12:34] 20% - effectiveness L=40.0% G=0.0% M7=1.0%",
                        flush=True,
                    )
                    for line in progress:
                        print(line, flush=True)
                    print("[12:34] sampling done", flush=True)

            visible = terminal.getvalue()
            self.assertIn("v8 continuous: games=1", visible)
            self.assertIn("graph source=empty(no-snapshot)", visible)
            self.assertIn("20% - current_run_wins=100.0%", visible)
            self.assertIn("current_run_wins=100.0%", visible)
            self.assertIn("current_run_levels_solved=100.0%", visible)
            self.assertIn("current_run_solved_games=1/1", visible)
            self.assertIn("20% - effectiveness L=40.0% G=0.0% M7=1.0%", visible)
            self.assertIn("sampling done", visible)
            for line in progress:
                self.assertNotIn(line, visible)

            logged = (Path(root) / "log.txt").read_text(encoding="utf-8")
            for line in progress:
                self.assertIn(line, logged)

    def test_verbose_progress_restores_terminal_output(self):
        with tempfile.TemporaryDirectory() as root:
            terminal = io.StringIO()
            line = "[12:34] optimizer game=ez01 level=1 status=START"
            with redirect_stdout(terminal):
                with observability.stdout_log_context(
                    [
                        "continuous-run",
                        "--root",
                        root,
                        "--verbose-progress",
                    ]
                ):
                    print(line, flush=True)
            self.assertIn(line, terminal.getvalue())
            self.assertIn(
                line,
                (Path(root) / "log.txt").read_text(encoding="utf-8"),
            )

    def test_hypothesis_reporting_interval_is_five_minutes(self):
        self.assertEqual(observability._HYPOTHESIS_INTERVAL_SECONDS, 300.0)
        self.assertIn(
            reporter.reporting_worker.__name__,
            {"_reporting_worker_v836", "_reporting_worker_v851_integrity"},
        )

    def test_reporter_emits_hypothesis_status_on_its_independent_schedule(self):
        events = queue.Queue()
        output = queue.Queue()
        stop = threading.Event()
        thread = threading.Thread(
            target=observability._reporting_worker_v836,
            kwargs={
                "event_queue": events,
                "stop_event": stop,
                "watermark": SimpleNamespace(value=0),
                "actors": ((1, "g"),),
                "interval_seconds": 10.0,
                "output_queue": output,
                "hypothesis_interval_seconds": 0.05,
            },
            daemon=True,
        )
        try:
            thread.start()
            line = output.get(timeout=1.0)
            self.assertIn("hypotheses H01=INSUFFICIENT_EVIDENCE", line)
            self.assertIn("H15=INSUFFICIENT_EVIDENCE", line)
            self.assertNotIn("current_run_wins", line)
        finally:
            stop.set()
            thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
