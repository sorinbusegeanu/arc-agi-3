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
_HYPOTHESIS_INTERVAL_SECONDS = 300.0
_PROCESS_MIRROR = None
_BASE_REPORTING_WORKER = None


class _TeeStdout:
    def __init__(self, original, path: Path) -> None:
        self.original = original
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._closed = False

    def write(self, value) -> int:
        text = str(value)
        with self._lock:
            written = self.original.write(text)
            self.log.write(text)
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


def _install_stdout_mirror(path: str | Path):
    target = Path(path)
    current = sys.stdout
    if isinstance(current, _TeeStdout) and current.path == target:
        return _NoopMirrorHandle()
    tee = _TeeStdout(current, target)
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
    os.environ[_STDOUT_LOG_ENV] = str(path)
    handle = _install_stdout_mirror(path)
    try:
        yield
    finally:
        handle.close()
        if prior_env is None:
            os.environ.pop(_STDOUT_LOG_ENV, None)
        else:
            os.environ[_STDOUT_LOG_ENV] = prior_env


def _mirror_from_environment() -> None:
    global _PROCESS_MIRROR
    if _PROCESS_MIRROR is not None:
        return
    raw = str(os.environ.get(_STDOUT_LOG_ENV, "")).strip()
    if not raw:
        return
    _PROCESS_MIRROR = _install_stdout_mirror(Path(raw))


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

        now = time.monotonic()
        if now >= next_report:
            rows = tuple(latest[key] for key in sorted(latest))
            reporter._emit_line(reporter.format_game_rate_line(rows), output_queue)
            while next_report <= now:
                next_report += float(interval_seconds)

        if now >= next_hypotheses:
            line = _hypothesis_status_line(
                tuple(evidence_by_id.values()),
                int(getattr(watermark, "value", 0)),
            )
            reporter._emit_line(line, output_queue)
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
