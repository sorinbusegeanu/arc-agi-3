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

    def test_hypothesis_reporting_interval_is_five_minutes(self):
        self.assertEqual(observability._HYPOTHESIS_INTERVAL_SECONDS, 300.0)
        self.assertIs(reporter.reporting_worker, observability._reporting_worker_v836)

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
