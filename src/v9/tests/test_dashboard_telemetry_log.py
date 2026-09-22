from __future__ import annotations

import json
import time
from types import SimpleNamespace
from urllib.request import urlopen

from v9 import ContinuousMemoryRuntime
from v9.runtime.config import RuntimeConfig
from v9.telemetry.http_server import DASHBOARD_REFRESH_SECONDS, MetricsHTTPServer


class _MetricsProvider:
    def __init__(self, root) -> None:
        self.config = SimpleNamespace(root=root)

    def metrics(self):
        return {"runtime_only": 99}

    def dashboard_metrics(self):
        return {
            "primary_dashboard": {"behavioral_success_rate": 0.25, "M0_count": 12},
            "telemetry_diagnostics": {
                "sampling_rate": 123.0,
                "sampling_backlog": 4,
                "very_large_debug_blob": "x" * 20_000,
            },
            "M0_resident": 11,
            "compaction_backlog": 3,
            "process_rss_bytes": 456,
        }


class _InitiallyFailingMetricsProvider(_MetricsProvider):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.calls = 0

    def dashboard_metrics(self):
        self.calls += 1
        if self.calls == 1:
            raise KeyError("transient-retirement-cut")
        return super().dashboard_metrics()


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
    assert latest["M0_resident"] == 11
    assert latest["compaction_backlog"] == 3
    assert latest["process_rss_bytes"] == 456
    assert "telemetry_diagnostics" not in latest
    assert "very_large_debug_blob" not in json.dumps(latest)
    assert latest["timestamp_utc"]
    assert len(json.dumps(latest)) < 2_000


def test_dashboard_logger_records_provider_failure_and_keeps_polling(tmp_path) -> None:
    provider = _InitiallyFailingMetricsProvider(tmp_path)
    server = MetricsHTTPServer(
        provider.metrics,
        host="127.0.0.1",
        port=0,
        refresh_seconds=0.1,
    )
    log_path = tmp_path / "telemetry" / "dashboard_metrics.jsonl"
    server.start()
    deadline = time.monotonic() + 2.0
    rows = []
    while time.monotonic() < deadline:
        if log_path.exists():
            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if len(rows) >= 2:
                break
        time.sleep(0.01)
    server.close()

    assert rows[0]["dashboard_metrics_error"]["type"] == "KeyError"
    assert rows[-1]["primary_dashboard"]["M0_count"] == 12
    assert provider.calls >= 2


def test_dashboard_html_hides_diagnostics_block(tmp_path) -> None:
    provider = _MetricsProvider(tmp_path)
    server = MetricsHTTPServer(provider.metrics, host="127.0.0.1", port=0, refresh_seconds=0.1)
    server.start()
    try:
        port = int(server._server.server_address[1])
        with urlopen(f"http://127.0.0.1:{port}/dashboard", timeout=2.0) as response:
            html = response.read().decode("utf-8")
        assert 'id="grid"' in html
        assert 'id="diag"' not in html
        assert "<h2>Diagnostics</h2>" not in html
    finally:
        server.close()


def test_runtime_dashboard_does_not_call_full_metrics(tmp_path, monkeypatch) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig(tmp_path / "runtime", enable_snapshots=False, restore=False)
    )
    try:
        def fail_full_metrics():
            raise AssertionError("dashboard polling called full metrics")

        monkeypatch.setattr(runtime, "metrics", fail_full_metrics)
        snapshot = runtime.dashboard_metrics()
        assert "primary_dashboard" in snapshot
        assert len(snapshot["primary_dashboard"]) <= 24
    finally:
        monkeypatch.undo()
        runtime.close()


def test_default_dashboard_and_jsonl_refresh_is_30_seconds() -> None:
    assert DASHBOARD_REFRESH_SECONDS == 30.0


def test_dashboard_log_is_per_run_not_append_forever(tmp_path) -> None:
    provider = _MetricsProvider(tmp_path)
    log_path = tmp_path / "telemetry" / "dashboard_metrics.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('{"stale": true}\n' * 50, encoding="utf-8")

    server = MetricsHTTPServer(
        provider.metrics,
        host="127.0.0.1",
        port=0,
        refresh_seconds=0.1,
    )
    server.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if log_path.exists() and "primary_dashboard" in log_path.read_text(encoding="utf-8"):
            break
        time.sleep(0.01)
    server.close()

    text = log_path.read_text(encoding="utf-8")
    assert '"stale"' not in text
    assert '"primary_dashboard"' in text
