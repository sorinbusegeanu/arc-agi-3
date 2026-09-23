from __future__ import annotations

import inspect
from types import SimpleNamespace

import v9
from v9.runtime import concurrent_transfer_validation as concurrent
from v9.runtime import concurrent_validation_startup as startup
from v9.runtime.multiprocess import ProcessTopology


def test_validation_server_is_initialized_before_supervisor_thread() -> None:
    source = inspect.getsource(concurrent.ConcurrentTransferValidationSession.start)
    assert source.index("ensure_process_server_ready") < source.index("original_session_start")


def test_validation_executor_reuses_runtime_process_context(monkeypatch) -> None:
    calls: list[object] = []

    class Pool:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    class Runtime:
        config = SimpleNamespace(multiprocessing_start_method=None)

        def set_telemetry_gauge(self, key, value):
            calls.append((key, value))

    monkeypatch.setattr(startup, "ensure_process_server_ready", lambda _method: "forkserver")
    monkeypatch.setattr(concurrent.mp, "get_context", lambda method: ("ctx", method))
    monkeypatch.setattr(concurrent, "ProcessPoolExecutor", Pool)

    fake = SimpleNamespace(
        _executor=None,
        adapter_factory=concurrent._base._eligible_target_specs,
        runtime=Runtime(),
        max_workers=30,
    )
    executor = concurrent.ConcurrentTransferValidationSession._ensure_executor(fake)
    assert isinstance(executor, Pool)
    kwargs = next(row for row in calls if isinstance(row, dict))
    assert kwargs["max_workers"] == 30
    assert kwargs["mp_context"] == ("ctx", "forkserver")
    assert ("transfer_validation_executor", "process:forkserver") in calls


def test_actor_prefill_uses_single_inline_progress_line() -> None:
    source = inspect.getsource(ProcessTopology.start_actor)
    assert "InlineProgress" in source
    assert "actor prefill" in source
    assert "progress.update" in source
    assert "progress.finish" in source
