from __future__ import annotations

import json
import time
from types import SimpleNamespace

from v9.telemetry.http_server import MetricsHTTPServer


class _MetricsProvider:
    def __init__(self, root) -> None:
        self.config = SimpleNamespace(root=root)

    def metrics(self):
        return {"runtime_only": 99}

    def dashboard_metrics(self):
        return {
            "primary_dashboard": {"behavioral_success_rate": 0.25, "M0_count": 12},
            "telemetry_diagnostics": {"sampling_rate": 123.0, "sampling_backlog": 4},
        }


def test_dashboard_metrics_are_logged_from_same_provider(tmp_path) -> None:
    provider = _MetricsProvider(tmp_path)
    server = MetricsHTTPServer(provider.metrics, host="127.0.0.1", port=0, refresh_seconds=0.1)
    log_path = tmp_path / "telemetry" / "dashboard_metrics.jsonl"
    server.start()
    deadline = time.monotonic() + 2.0
    while (not log_path.exists() or not log_path.stat().st_size) and time.monotonic() < deadline:
        time.sleep(0.01)
    server.close()

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    latest = rows[-1]
    assert latest["primary_dashboard"] == provider.dashboard_metrics()["primary_dashboard"]
    assert latest["telemetry_diagnostics"] == provider.dashboard_metrics()["telemetry_diagnostics"]
    assert latest["timestamp_utc"]
