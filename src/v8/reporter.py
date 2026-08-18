from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Iterable

from v8.actor import ActorProgress
from v8.diagnostics import format_game_rate_line
from v8.evidence import EvidenceRecord


SAMPLING_COMPLETE = "v8:reporter:sampling-complete"


def _emit_line(message: str, output_queue: mp.Queue | None) -> None:
    line = f'[{time.strftime("%H:%M")}] {message}'
    if output_queue is None:
        print(line, flush=True)
    else:
        output_queue.put(line)


def _emit_sampling_complete(output_queue: mp.Queue | None) -> None:
    _emit_line("sampling done", output_queue)
    if output_queue is not None:
        output_queue.close()
        output_queue.join_thread()


def format_budget_game_rate_line(
    rows: Iterable[ActorProgress],
    total_steps: int | None,
) -> str:
    rows = tuple(rows)
    line = format_game_rate_line(rows)
    budget = 0 if total_steps is None else max(0, int(total_steps))
    if budget <= 0:
        return line
    used = min(budget, sum(max(0, int(row.steps)) for row in rows))
    percentage = 100.0 * used / budget
    return f"{percentage:.0f}% - {line}"


def reporting_worker(
    *,
    event_queue: mp.Queue,
    stop_event: mp.synchronize.Event,
    watermark: mp.sharedctypes.Synchronized,
    actors: tuple[tuple[int, str], ...],
    interval_seconds: float,
    output_queue: mp.Queue | None = None,
    total_steps: int | None = None,
) -> None:
    latest = {
        int(actor_id): ActorProgress(int(actor_id), str(game_id), 0, 0, 0, 0)
        for actor_id, game_id in actors
    }
    next_report = time.monotonic() + float(interval_seconds)

    while not stop_event.is_set():
        now = time.monotonic()
        timeout = max(0.0, min(0.25, next_report - now))
        try:
            row = event_queue.get(timeout=timeout)
        except queue.Empty:
            row = None

        if isinstance(row, ActorProgress):
            latest[int(row.actor_id)] = row
        elif isinstance(row, EvidenceRecord):
            # Evidence remains authoritative in the runtime ledger and final reports.
            # The dedicated stdout reporter intentionally ignores it for now.
            pass
        elif row == SAMPLING_COMPLETE:
            _emit_sampling_complete(output_queue)
            return

        now = time.monotonic()
        if now < next_report:
            continue

        rows = tuple(latest[key] for key in sorted(latest))
        _emit_line(format_budget_game_rate_line(rows, total_steps), output_queue)
        while next_report <= now:
            next_report += float(interval_seconds)


class DedicatedReporter:
    """Isolated periodic progress reporter fed by actor progress."""

    def __init__(
        self,
        mp_context,
        *,
        watermark: mp.sharedctypes.Synchronized,
        actors: Iterable[tuple[int, str]],
        interval_seconds: float = 60.0,
        output_queue: mp.Queue | None = None,
        total_steps: int | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("reporting interval must be positive")
        self._queue = mp_context.Queue()
        self._stop = mp_context.Event()
        self._process = mp_context.Process(
            target=reporting_worker,
            kwargs={
                "event_queue": self._queue,
                "stop_event": self._stop,
                "watermark": watermark,
                "actors": tuple((int(actor_id), str(game_id)) for actor_id, game_id in actors),
                "interval_seconds": float(interval_seconds),
                "output_queue": output_queue,
                "total_steps": total_steps,
            },
            name="v8-dedicated-reporter",
            daemon=True,
        )
        self._started = False
        self._closed = False

    @property
    def progress_queue(self) -> mp.Queue:
        return self._queue

    def start(self) -> None:
        if self._started:
            return
        self._process.start()
        self._started = True

    def publish_evidence(self, row: EvidenceRecord) -> None:
        if self._closed:
            return
        self._queue.put(row)

    def raise_if_failed(self) -> None:
        if self._started and self._process.exitcode not in (None, 0):
            raise RuntimeError(
                f"v8 dedicated reporter exited unexpectedly: {self._process.exitcode}"
            )

    def close(self, *, timeout: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._started:
            self._process.join(timeout=float(timeout))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
        self._queue.cancel_join_thread()
        self._queue.close()
