from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from v9.runtime import post_sampling_progress as progress


def test_dashboard_uses_cached_snapshot_when_runtime_lock_is_busy() -> None:
    class Runtime:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._metrics_cache = {"primary_dashboard": {"memories": 7}}

        def dashboard_metrics(self):
            with self._lock:
                return {"primary_dashboard": {"memories": 7}}

        def wait_quiescent(self, *_args, **_kwargs):
            return None

        def flush_deferred_memory_updates(self):
            return None

        def replay_once(self):
            return SimpleNamespace()

        def snapshot(self):
            return None

    epoch = SimpleNamespace(
        run_epochs=lambda *_args, **_kwargs: None,
        train_hgt_epoch=lambda *_args, **_kwargs: SimpleNamespace(model_version="test"),
        run_lifecycle_maintenance=lambda *_args, **_kwargs: {},
    )
    progress.install(epoch, Runtime)
    runtime = Runtime()
    assert runtime.dashboard_metrics()["primary_dashboard"]["memories"] == 7

    acquired = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with runtime._lock:
            acquired.set()
            release.wait(1.0)

    worker = threading.Thread(target=hold_lock)
    worker.start()
    assert acquired.wait(1.0)
    started = time.monotonic()
    snapshot = runtime.dashboard_metrics()
    elapsed = time.monotonic() - started
    release.set()
    worker.join(timeout=1.0)

    assert elapsed < 0.1
    assert snapshot["primary_dashboard"]["memories"] == 7
    assert snapshot["dashboard_snapshot"] == "cached"


def test_post_sampling_heartbeat_repeats_during_long_phase(capsys) -> None:
    class Runtime:
        def __init__(self) -> None:
            self._lock = threading.RLock()

        def dashboard_metrics(self):
            return {"primary_dashboard": {}}

        def wait_quiescent(self, *_args, **_kwargs):
            time.sleep(0.25)

        def flush_deferred_memory_updates(self):
            return None

        def replay_once(self):
            return SimpleNamespace()

        def snapshot(self):
            return None

    runtime = Runtime()

    def run_epochs(instance, _specs, _args):
        instance.wait_quiescent()
        return (), ()

    epoch = SimpleNamespace(
        run_epochs=run_epochs,
        train_hgt_epoch=lambda *_args, **_kwargs: SimpleNamespace(model_version="test"),
        run_lifecycle_maintenance=lambda *_args, **_kwargs: {},
    )
    progress.install(epoch, Runtime)
    args = SimpleNamespace(progress_interval_seconds=0.1)
    epoch.run_epochs(runtime, (), args)
    output = capsys.readouterr().out
    assert output.count("quiescent/drain") >= 2
    assert "elapsed=" in output
