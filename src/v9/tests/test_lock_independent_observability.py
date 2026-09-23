from __future__ import annotations

import json
import threading
import time
from urllib.request import urlopen

import v9.runtime.post_sampling_progress as post_progress
from v9 import ContinuousMemoryRuntime
from v9.runtime.config import RuntimeConfig
from v9.telemetry.http_server import MetricsHTTPServer


def _hold_runtime_lock(runtime, locked: threading.Event, release: threading.Event) -> None:
    with runtime._lock:
        locked.set()
        release.wait(timeout=3.0)


def test_dashboard_is_populated_while_runtime_lock_is_owned_elsewhere(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig(tmp_path / "runtime", enable_snapshots=False, restore=False)
    )
    runtime.set_telemetry_gauge("sampled_steps", 123)
    runtime.set_telemetry_gauge("ingested_steps", 97)
    runtime.set_telemetry_gauge("sampling_backlog", 26)

    locked = threading.Event()
    release = threading.Event()
    holder = threading.Thread(
        target=_hold_runtime_lock,
        args=(runtime, locked, release),
        daemon=True,
    )
    holder.start()
    assert locked.wait(timeout=1.0)
    try:
        started = time.monotonic()
        snapshot = runtime.dashboard_metrics()
        elapsed = time.monotonic() - started
        primary = snapshot["primary_dashboard"]
        assert elapsed < 0.25
        assert primary
        assert primary["sampling_backlog"] == 26
    finally:
        release.set()
        holder.join(timeout=1.0)
        runtime.close()


def test_http_dashboard_is_populated_while_runtime_lock_is_owned_elsewhere(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig(tmp_path / "runtime-http", enable_snapshots=False, restore=False)
    )
    runtime.set_telemetry_gauge("sampled_steps", 321)
    runtime.set_telemetry_gauge("ingested_steps", 300)
    runtime.set_telemetry_gauge("sampling_backlog", 21)
    server = MetricsHTTPServer(
        runtime.metrics,
        host="127.0.0.1",
        port=0,
        refresh_seconds=30.0,
        live_refresh_seconds=0.05,
    )

    locked = threading.Event()
    release = threading.Event()
    holder = threading.Thread(
        target=_hold_runtime_lock,
        args=(runtime, locked, release),
        daemon=True,
    )
    holder.start()
    assert locked.wait(timeout=1.0)
    server.start()
    try:
        port = int(server._server.server_address[1])
        deadline = time.monotonic() + 1.0
        payload = {}
        while time.monotonic() < deadline:
            with urlopen(f"http://127.0.0.1:{port}/api/metrics", timeout=0.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("primary_dashboard", {}).get("sampling_backlog") == 21:
                break
            time.sleep(0.02)
        primary = payload["primary_dashboard"]
        assert primary
        assert primary["sampling_backlog"] == 21
    finally:
        server.close()
        release.set()
        holder.join(timeout=1.0)
        runtime.close()


def test_sampling_heartbeat_does_not_call_runtime_locking_setter(monkeypatch) -> None:
    class Telemetry:
        gauges = {
            "sampled_steps": 200,
            "causally_admitted_steps": 180,
            "ingested_steps": 150,
            "publication_backlog": 20,
            "canonical_ingest_backlog": 30,
        }

        def diagnostic_metrics(self):
            return dict(self.gauges)

    class Runtime:
        unified_telemetry = Telemetry()

        def set_telemetry_gauge(self, *_args, **_kwargs):
            raise AssertionError("heartbeat touched runtime setter")

    lines: list[str] = []
    monkeypatch.setattr(post_progress, "_current_runtime", Runtime())
    monkeypatch.setattr(post_progress, "_sampling_active", True)
    monkeypatch.setattr(post_progress, "_sampling_started", time.monotonic() - 60.0)
    monkeypatch.setattr(post_progress, "_sampling_label", "sampling")
    monkeypatch.setattr(post_progress, "_progress_interval_seconds", 0.1)
    monkeypatch.setattr(
        post_progress,
        "print",
        lambda *values, **_kwargs: lines.append(" ".join(str(value) for value in values)),
    )

    stop = threading.Event()
    worker = threading.Thread(target=post_progress._heartbeat_loop, args=(stop,), daemon=True)
    worker.start()
    deadline = time.monotonic() + 0.5
    while not lines and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    worker.join(timeout=1.0)

    assert lines
    assert "sampled=200" in lines[0]
    assert "published=180" in lines[0]
    assert "ingested=150" in lines[0]
