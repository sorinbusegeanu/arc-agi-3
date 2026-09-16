from __future__ import annotations

from functools import wraps
from typing import Any


def install_metrics_concurrency(runtime_cls: type) -> None:
    """Keep composed metrics/dashboard traversals on one authoritative runtime cut."""
    if getattr(runtime_cls, "_metrics_concurrency_installed", False):
        return

    original_metrics = runtime_cls.metrics
    original_dashboard_metrics = runtime_cls.dashboard_metrics

    @wraps(original_metrics)
    def metrics(self: Any) -> dict[str, Any]:
        with self._lock:
            return original_metrics(self)

    @wraps(original_dashboard_metrics)
    def dashboard_metrics(self: Any) -> dict[str, Any]:
        with self._lock:
            return original_dashboard_metrics(self)

    runtime_cls.metrics = metrics
    runtime_cls.dashboard_metrics = dashboard_metrics
    runtime_cls._metrics_concurrency_installed = True
