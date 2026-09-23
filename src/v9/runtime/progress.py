from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import sys
import time
from typing import TypeVar


T = TypeVar("T")


class InlineProgress:
    """Single-line terminal progress renderer using carriage-return updates."""

    def __init__(self, label: str) -> None:
        self.label = str(label)
        self._last_width = 0
        self._closed = False

    def update(self, detail: str = "") -> None:
        if self._closed:
            return
        text = self.label if not detail else f"{self.label} {detail}"
        padded = text.ljust(self._last_width)
        sys.stdout.write("\r" + padded)
        sys.stdout.flush()
        self._last_width = max(self._last_width, len(text))

    def finish(self, detail: str = "done") -> None:
        if self._closed:
            return
        self.update(detail)
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._closed = True


def progress_iter(
    iterable: Iterable[T],
    *,
    label: str,
    total: int | None = None,
    initial: int = 0,
    interval_seconds: float = 1.0,
    detail: Callable[[int, float], str] | None = None,
) -> Iterator[T]:
    """Yield items while updating one terminal line instead of printing rows."""
    progress = InlineProgress(label)
    count = int(initial)
    started = time.monotonic()
    next_refresh = started

    def render() -> None:
        elapsed = max(0.0, time.monotonic() - started)
        if total is None:
            counter = str(count)
        else:
            counter = f"{count}/{int(total)}"
        suffix = "" if detail is None else str(detail(count, elapsed)).strip()
        progress.update(f"{counter}" + (f" {suffix}" if suffix else ""))

    render()
    try:
        for item in iterable:
            yield item
            count += 1
            now = time.monotonic()
            if now >= next_refresh or (total is not None and count >= int(total)):
                render()
                next_refresh = now + max(0.05, float(interval_seconds))
    finally:
        render()
        progress.finish()
