from __future__ import annotations

import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


_INSTALLED = False
_STDOUT_LOG_ENV = "ARC_AGI3_V8_STDOUT_LOG"
_STDOUT_PROGRESS_ENV = "ARC_AGI3_V8_STDOUT_PROGRESS"
_HYPOTHESIS_INTERVAL_SECONDS = 300.0
_PROCESS_MIRROR = None
_BASE_REPORTING_WORKER = None


class _AsyncHypothesisReporter:
    """Evaluate the slower scientific summary without delaying minute telemetry."""

    def __init__(self) -> None:
        self._thread = None
        self._results = queue.SimpleQueue()

    def start(self, evidence_rows, watermark_value: int) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False

        def evaluate() -> None:
            try:
                result = (True, _hypothesis_status_line(evidence_rows, watermark_value))
            except BaseException as exc:
                result = (False, exc)
            self._results.put(result)

        self._thread = threading.Thread(
            target=evaluate,
            name="v8-hypothesis-reporter",
            daemon=True,
        )
        self._thread.start()
        return True

    def emit_ready(self, output_queue=None) -> None:
        from v8 import reporter

        while True:
            try:
                succeeded, value = self._results.get_nowait()
            except queue.Empty:
                return
            if not succeeded:
                raise value
            reporter._emit_line(str(value), output_queue)


class _TeeStdout:
    def __init__(self, original, path: Path, *, suppress_progress: bool = False) -> None:
        self.original = original
        self.path = Path(path)
        self.suppress_progress = bool(suppress_progress)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._closed = False
        self._last_write_suppressed = False

    @staticmethod
    def _is_progress_line(value: str) -> bool:
        line = str(value).rstrip("\r\n")
        if not (
            len(line) >= 8
            and line[0] == "["
            and line[1:3].isdigit()
            and line[3] == ":"
            and line[4:6].isdigit()
            and line[6:8] == "] "
        ):
            return False
        payload = line[8:]
        budget_summary = False
        if "% - current_run_wins=" in payload:
            percentage, _separator, _summary = payload.partition(
                "% - current_run_wins="
            )
            try:
                budget_summary = 0.0 <= float(percentage) <= 100.0
            except ValueError:
                budget_summary = False
        effectiveness_summary = False
        if "% - effectiveness L=" in payload:
            percentage, _separator, _summary = payload.partition(
                "% - effectiveness L="
            )
            try:
                effectiveness_summary = 0.0 <= float(percentage) <= 100.0
            except ValueError:
                effectiveness_summary = False
        return not (
            payload.startswith("graph source=")
            or payload.startswith("current_run_wins=")
            or budget_summary
            or effectiveness_summary
            or payload == "sampling done"
        )

    def write(self, value) -> int:
        text = str(value)
        with self._lock:
            self.log.write(text)
            suppress = bool(
                self.suppress_progress
                and (
                    self._is_progress_line(text)
                    or (self._last_write_suppressed and text in {"\n", "\r\n"})
                )
            )
            self._last_write_suppressed = suppress and text not in {"\n", "\r\n"}
            written = len(text) if suppress else self.original.write(text)
        return len(text) if written is None else int(written)

    def flush(self) -> None:
        with self._lock:
            self.original.flush()
            self.log.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(self.original.fileno())

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.original, "errors", None)

    def close_log(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            self.log.close()

    def __getattr__(self, name: str):
        return getattr(self.original, name)


class _MirrorHandle:
    def __init__(self, tee: _TeeStdout, prior_stdout) -> None:
        self.tee = tee
        self.prior_stdout = prior_stdout
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if sys.stdout is self.tee:
            sys.stdout = self.prior_stdout
        self.tee.close_log()


class _NoopMirrorHandle:
    def close(self) -> None:
        return None


def _install_stdout_mirror(
    path: str | Path,
    *,
    suppress_progress: bool = False,
):
    target = Path(path)
    current = sys.stdout
    if (
        isinstance(current, _TeeStdout)
        and current.path == target
        and current.suppress_progress == bool(suppress_progress)
    ):
        return _NoopMirrorHandle()
    tee = _TeeStdout(current, target, suppress_progress=suppress_progress)
    sys.stdout = tee
    return _MirrorHandle(tee, current)


def _log_path_from_argv(argv) -> Path | None:
    values = tuple(str(value) for value in argv)
    if "continuous-run" not in values:
        return None
    root = "runs/v8/continuous"
    for index, value in enumerate(values):
        if value == "--root" and index + 1 < len(values):
            root = values[index + 1]
        elif value.startswith("--root="):
            root = value.split("=", 1)[1]
    return Path(root) / "log.txt"


@contextmanager
def stdout_log_context(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    path = _log_path_from_argv(values)
    if path is None:
        yield
        return

    prior_env = os.environ.get(_STDOUT_LOG_ENV)
    prior_progress = os.environ.get(_STDOUT_PROGRESS_ENV)
    show_progress = "--verbose-progress" in values
    os.environ[_STDOUT_LOG_ENV] = str(path)
    os.environ[_STDOUT_PROGRESS_ENV] = "1" if show_progress else "0"
    handle = _install_stdout_mirror(path, suppress_progress=not show_progress)
    try:
        yield
    finally:
        handle.close()
        if prior_env is None:
            os.environ.pop(_STDOUT_LOG_ENV, None)
        else:
            os.environ[_STDOUT_LOG_ENV] = prior_env
        if prior_progress is None:
            os.environ.pop(_STDOUT_PROGRESS_ENV, None)
        else:
            os.environ[_STDOUT_PROGRESS_ENV] = prior_progress


def _mirror_from_environment() -> None:
    global _PROCESS_MIRROR
    if _PROCESS_MIRROR is not None:
        return
    raw = str(os.environ.get(_STDOUT_LOG_ENV, "")).strip()
    if not raw:
        return
    show_progress = str(os.environ.get(_STDOUT_PROGRESS_ENV, "1")).strip() == "1"
    _PROCESS_MIRROR = _install_stdout_mirror(
        Path(raw),
        suppress_progress=not show_progress,
    )


def _hypothesis_status_line(evidence_rows, watermark_value: int) -> str:
    from v8.diagnostics import format_hypothesis_line
    from v8.evaluation import ScientificHypothesisEvaluator

    cut = tuple(
        row
        for row in evidence_rows
        if int(getattr(row, "watermark", 0)) <= int(watermark_value)
    )
    evaluator = ScientificHypothesisEvaluator()
    statuses = evaluator.status_map(evaluator.evaluate(cut))
    return format_hypothesis_line(statuses)


def _reporting_worker_v836(
    *,
    event_queue,
    stop_event,
    watermark,
    actors,
    interval_seconds: float,
    output_queue=None,
    hypothesis_interval_seconds: float = _HYPOTHESIS_INTERVAL_SECONDS,
    total_steps: int | None = None,
    baseline=None,
) -> None:
    from v8 import reporter
    from v8.actor import ActorProgress
    from v8.evidence import EvidenceRecord

    latest = {
        int(actor_id): ActorProgress(int(actor_id), str(game_id), 0, 0, 0, 0)
        for actor_id, game_id in actors
    }
    evidence_by_id = {}
    now = time.monotonic()
    next_report = now + float(interval_seconds)
    next_hypotheses = now + max(0.001, float(hypothesis_interval_seconds))
    hypotheses = _AsyncHypothesisReporter()

    while not stop_event.is_set():
        now = time.monotonic()
        timeout = max(
            0.0,
            min(0.25, next_report - now, next_hypotheses - now),
        )
        try:
            row = event_queue.get(timeout=timeout)
        except queue.Empty:
            row = None

        if isinstance(row, ActorProgress):
            latest[int(row.actor_id)] = row
        elif isinstance(row, EvidenceRecord):
            evidence_by_id[str(row.evidence_id)] = row
        elif row == reporter.SAMPLING_COMPLETE:
            reporter._emit_sampling_complete(output_queue)
            return

        now = time.monotonic()
        if now >= next_report:
            rows = tuple(latest[key] for key in sorted(latest))
            reporter._emit_line(
                reporter.format_periodic_progress_line(rows, total_steps, baseline),
                output_queue,
            )
            while next_report <= now:
                next_report += float(interval_seconds)

        hypotheses.emit_ready(output_queue)

        if now >= next_hypotheses:
            hypotheses.start(
                tuple(evidence_by_id.values()),
                int(getattr(watermark, "value", 0)),
            )
            while next_hypotheses <= now:
                next_hypotheses += max(0.001, float(hypothesis_interval_seconds))


def install_runtime_observability_v836() -> None:
    global _INSTALLED, _BASE_REPORTING_WORKER
    if _INSTALLED:
        _mirror_from_environment()
        return

    from v8 import reporter

    _BASE_REPORTING_WORKER = reporter.reporting_worker
    reporter.reporting_worker = _reporting_worker_v836
    _INSTALLED = True
    _mirror_from_environment()
