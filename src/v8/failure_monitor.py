from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class PeerFailure:
    operator: str
    error_type: str
    message: str


class PeerFailureMonitor:
    def __init__(self) -> None:
        self._failures: list[PeerFailure] = []
        self._lock = Lock()

    def record(self, operator: str, exc: BaseException) -> None:
        with self._lock:
            self._failures.append(PeerFailure(str(operator), type(exc).__name__, str(exc)))

    def failures(self) -> tuple[PeerFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    def raise_if_failed(self) -> None:
        failures = self.failures()
        if failures:
            raise RuntimeError(f"v8 peer failure: {failures}")
