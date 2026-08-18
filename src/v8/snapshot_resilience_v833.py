from __future__ import annotations

"""v8.33 resilient periodic snapshots.

A periodic snapshot is opportunistic. If a long peer/lifecycle cycle cannot reach
an idle point before the cadence timeout, that single periodic snapshot is skipped
instead of poisoning the runtime and crashing shutdown later. Explicit consistent
snapshots and final snapshots remain strict and unchanged.
"""

import time

_INSTALLED = False
_BASE_SNAPSHOT_CADENCE = None


def _background_snapshot_attempt_v833(runtime, *, timeout: float) -> str:
    try:
        runtime.request_async_snapshot()
        return "saved"
    except TimeoutError:
        runtime._v833_background_snapshot_skips = int(
            getattr(runtime, "_v833_background_snapshot_skips", 0)
        ) + 1
        return "skipped"
    except BaseException as exc:
        runtime._snapshot_error = f"{type(exc).__name__}: {exc}"
        return "fatal"


def _snapshot_cadence_v833(self) -> None:
    interval = float(self.config.snapshot_interval_seconds)
    while not self._snapshot_thread_stop.wait(interval):
        if self._closed or self.snapshot_service is None:
            return
        status = _background_snapshot_attempt_v833(
            self,
            timeout=max(10.0, interval),
        )
        if status == "fatal":
            return


def install_snapshot_resilience_v833() -> None:
    global _INSTALLED, _BASE_SNAPSHOT_CADENCE
    if _INSTALLED:
        return

    from v8.runtime import ContinuousMemoryRuntime

    _BASE_SNAPSHOT_CADENCE = ContinuousMemoryRuntime._snapshot_cadence
    ContinuousMemoryRuntime._snapshot_cadence = _snapshot_cadence_v833
    _INSTALLED = True
