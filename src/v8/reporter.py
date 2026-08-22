from __future__ import annotations

import multiprocessing as mp
import queue
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from v8.actor import ActorProgress
from v8.diagnostics import format_game_rate_line, solved_game_ids
from v8.evidence import EvidenceRecord


SAMPLING_COMPLETE = "v8:reporter:sampling-complete"
_PROGRESS_PATTERN = re.compile(
    r"current_run_wins=(?P<wins>[0-9]+(?:\.[0-9]+)?)% "
    r"current_run_levels_solved=(?P<levels>[0-9]+(?:\.[0-9]+)?)% "
    r"current_run_solved_games=(?P<solved>[0-9]+)/(?P<games>[0-9]+)"
)
_RUN_HEADER_PATTERN = re.compile(r"^v8 continuous: games=(?P<games>[0-9]+)\b(?P<options>.*)$")
_GAME_IDS_PATTERN = re.compile(r"(?:^|\s)game_ids=(?P<game_ids>[^\s]+)")
_PROGRESS_GAME_PATTERN = re.compile(
    r"(?:\(|[;,]\s*)(?P<game_id>[A-Za-z0-9_.-]+):"
)


@dataclass(frozen=True)
class ContinuousProgressBaseline:
    """High-water competence observed under one continuous-run root."""

    game_ids: tuple[str, ...] = ()
    solved_games: int = 0
    games: int = 0
    level_rate: float = 0.0


def load_continuous_progress_baseline(
    log_path: str | Path,
    *,
    games: Iterable[str],
    durable_solved_games: Iterable[str] = (),
) -> ContinuousProgressBaseline:
    """Recover progress only from runs with the same selected game IDs."""
    selected = tuple(dict.fromkeys(str(game_id) for game_id in games))
    selected_set = set(selected)
    durable = tuple(
        game_id
        for game_id in dict.fromkeys(str(game_id) for game_id in durable_solved_games)
        if game_id in selected_set
    )
    durable_set = set(durable)
    retained_ids = set(durable)
    best_solved = len(durable)
    best_level_rate = 100.0 * best_solved / len(selected) if selected else 0.0

    path = Path(log_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""

    active_game_ids: frozenset[str] | None = None
    for line in text.splitlines():
        header = _RUN_HEADER_PATTERN.match(line)
        if header is not None:
            scoped = _GAME_IDS_PATTERN.search(header.group("options"))
            active_game_ids = (
                frozenset(
                    game_id
                    for game_id in scoped.group("game_ids").split(",")
                    if game_id
                )
                if scoped is not None
                else None
            )
            continue

        match = _PROGRESS_PATTERN.search(line)
        if match is None:
            continue
        if int(match.group("games")) != len(selected):
            continue
        solved = min(len(selected), max(0, int(match.group("solved"))))
        detail_ids = set(_PROGRESS_GAME_PATTERN.findall(line[match.end() :]))

        exact_scope = active_game_ids == frozenset(selected_set)
        legacy_scope = bool(
            active_game_ids is None
            and solved == len(durable)
            and detail_ids
            and detail_ids <= durable_set
        )
        if not (exact_scope or legacy_scope):
            continue

        level_rate = min(100.0, max(0.0, float(match.group("levels"))))
        best_solved = max(best_solved, solved)
        best_level_rate = max(best_level_rate, level_rate)
        retained_ids.update(detail_ids & selected_set)

    best_level_rate = max(
        best_level_rate,
        100.0 * best_solved / len(selected) if selected else 0.0,
    )
    return ContinuousProgressBaseline(
        game_ids=tuple(game_id for game_id in selected if game_id in retained_ids),
        solved_games=best_solved,
        games=len(selected),
        level_rate=best_level_rate,
    )


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
    baseline: ContinuousProgressBaseline | None = None,
) -> str:
    rows = tuple(rows)
    line = format_game_rate_line(rows)
    match = _PROGRESS_PATTERN.search(line)
    if (
        baseline is not None
        and match is not None
        and int(baseline.games) == int(match.group("games"))
    ):
        games = int(match.group("games"))
        level_rate = float(match.group("levels"))
        observed_ids = set(solved_game_ids(rows))
        retained_ids = set(str(game_id) for game_id in baseline.game_ids)
        solved_games = max(
            int(baseline.solved_games),
            len(observed_ids | retained_ids),
        )
        solved_games = min(games, solved_games)
        win_rate = 100.0 * solved_games / games if games else 0.0
        level_rate = max(float(baseline.level_rate), level_rate, win_rate)
        headline = (
            f"current_run_wins={win_rate:.1f}% "
            f"current_run_levels_solved={level_rate:.1f}% "
            f"current_run_solved_games={solved_games}/{games}"
        )
        line = line[: match.start()] + headline + line[match.end() :]
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
    baseline: ContinuousProgressBaseline | None = None,
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
        _emit_line(
            format_budget_game_rate_line(rows, total_steps, baseline),
            output_queue,
        )
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
        baseline: ContinuousProgressBaseline | None = None,
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
                "baseline": baseline,
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
