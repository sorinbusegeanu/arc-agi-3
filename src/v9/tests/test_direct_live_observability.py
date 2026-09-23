from __future__ import annotations

import json
import time
from types import SimpleNamespace
from urllib.request import urlopen

from v9.runtime import concurrent_validation_startup as startup
from v9.telemetry.http_server import MetricsHTTPServer


def test_post_sampling_progress_writes_directly_to_tty_fd(monkeypatch) -> None:
    writes: list[tuple[int, bytes]] = []

    class TTY:
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 1

    def fake_write(fd: int, payload: bytes) -> int:
        writes.append((fd, payload))
        return len(payload)

    monkeypatch.setattr(startup.sys, "stdout", TTY())
    monkeypatch.setattr(startup.os, "write", fake_write)
    startup._terminal_print("heartbeat", "visible", flush=True)

    assert writes == [(1, b"heartbeat visible\n")]


def test_http_dashboard_serves_precomputed_live_snapshot(tmp_path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.config = SimpleNamespace(root=tmp_path)
            self.calls = 0

        def metrics(self):
            return {"unused": True}

        def dashboard_metrics(self):
            self.calls += 1
            return {
                "primary_dashboard": {
                    "run_phase": "sampling/pipeline",
                    "M0_count": 17,
                }
            }

    provider = Provider()
    server = MetricsHTTPServer(
        provider.metrics,
        host="127.0.0.1",
        port=0,
        refresh_seconds=30.0,
        live_refresh_seconds=0.05,
    )
    server.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if getattr(server, "_live_cache", {}).get("primary_dashboard", {}).get("M0_count") == 17:
                break
            time.sleep(0.01)

        assert server._live_cache["primary_dashboard"]["M0_count"] == 17
        calls_after_capture = provider.calls

        def blocked_provider():
            raise AssertionError("HTTP request entered runtime provider")

        server._dashboard_provider = blocked_provider
        port = int(server._server.server_address[1])
        with urlopen(f"http://127.0.0.1:{port}/api/metrics", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["primary_dashboard"]["run_phase"] == "sampling/pipeline"
        assert payload["primary_dashboard"]["M0_count"] == 17
        assert provider.calls == calls_after_capture
    finally:
        server.close()
