from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from v9.runtime import post_sampling_progress
from v9.telemetry.http_server import (
    DASHBOARD_REFRESH_SECONDS,
    MetricsHTTPServer,
)


def test_dashboard_refresh_uses_single_jsonl_cadence() -> None:
    server = MetricsHTTPServer(
        lambda: {"primary_dashboard": {}},
        host="127.0.0.1",
        port=0,
        log_path=None,
    )
    try:
        assert DASHBOARD_REFRESH_SECONDS == 30.0
        assert server.refresh_seconds == 30.0
        assert server.log_refresh_seconds == 30.0
        assert not hasattr(server, "live_refresh_seconds")
        assert not hasattr(server, "_live_thread")
    finally:
        server._server.server_close()


def test_parallel_wrapper_prints_sampling_completion_as_normal_line() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.gauges: dict[str, object] = {}
            self._lock = None
            self._metrics_cache = {}
            self.unified_telemetry = SimpleNamespace(
                diagnostic_metrics=lambda: dict(self.gauges)
            )

        def set_telemetry_gauge(self, key: str, value: object) -> None:
            self.gauges[key] = value

        def wait_quiescent(self, *_args, **_kwargs):
            return None

        def flush_deferred_memory_updates(self, *_args, **_kwargs):
            return None

        def replay_once(self, *_args, **_kwargs):
            return SimpleNamespace()

        def snapshot(self, *_args, **_kwargs):
            return None

        def dashboard_metrics(self):
            return {"primary_dashboard": {}}

    module = SimpleNamespace(
        run_epochs=lambda *args, **kwargs: None,
        run_parallel_memory_jobs=lambda runtime, jobs, *args, **kwargs: [
            SimpleNamespace(steps=int(job[2])) for job in jobs
        ],
        train_hgt_epoch=lambda *args, **kwargs: SimpleNamespace(
            model_version="hgt-test"
        ),
        run_lifecycle_maintenance=lambda *args, **kwargs: {},
    )
    post_sampling_progress.install(module, Runtime)
    runtime = Runtime()
    jobs = [(1, SimpleNamespace(), 7, 0), (2, SimpleNamespace(), 5, 0)]
    output = StringIO()
    with redirect_stdout(output):
        rows = module.run_parallel_memory_jobs(
            runtime, jobs, evaluation_only=False
        )
    assert len(rows) == 2
    text = output.getvalue()
    assert "100.0% sampled=12/12" in text
    assert "games_finished=2/2 sampling complete" in text
    assert text.endswith("\n")
    assert runtime.gauges["run_phase"] == "post-sampling"
