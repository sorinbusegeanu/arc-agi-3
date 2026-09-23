from __future__ import annotations

import builtins
import json
import os
import sys
import time
from threading import Event, RLock, Thread
from typing import Any

from . import concurrent_transfer_validation as concurrent
from . import post_sampling_progress as post_progress
from .multiprocess import ProcessTopology, ensure_process_server_ready
from .progress import InlineProgress
from v9.telemetry import http_server as dashboard_http


def _terminal_print(*values: object, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:
    """Write live progress straight to fd 1 so renderers/wrappers cannot swallow it."""
    if file not in (None, sys.stdout, sys.__stdout__):
        builtins.print(*values, sep=sep, end=end, file=file, flush=flush)
        return
    text = sep.join(str(value) for value in values) + end
    try:
        os.write(1, text.encode("utf-8", errors="replace"))
    except (AttributeError, BrokenPipeError, OSError, ValueError):
        builtins.print(*values, sep=sep, end=end, file=sys.__stdout__, flush=True)


def _install_direct_live_observability() -> None:
    # post_sampling_progress runs from a heartbeat thread while the coordinator
    # owns the main thread. Bypass sys.stdout renderers so those lines are always
    # visible in the terminal, including during publication/ingestion drains.
    post_progress.print = _terminal_print

    server_cls = dashboard_http.MetricsHTTPServer
    if getattr(server_cls, "_direct_live_cache_installed", False):
        return
    server_cls._direct_live_cache_installed = True

    original_init = server_cls.__init__
    original_start = server_cls.start
    original_close = server_cls.close

    def init_with_live_cache(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._live_cache_lock = RLock()
        self._live_cache: dict[str, Any] = {}
        self._live_cache_error: dict[str, str] | None = None
        self._live_cache_stop = Event()
        self._live_cache_thread: Thread | None = None

    def capture_live_snapshot(self: Any) -> dict[str, Any]:
        snapshot = dict(self._dashboard_provider())
        self._record_model_history(snapshot)
        snapshot["model_history"] = self._model_history_snapshot()
        with self._live_cache_lock:
            self._live_cache = snapshot
            self._live_cache_error = None
        return snapshot

    def live_cache_loop(self: Any) -> None:
        while not self._live_cache_stop.is_set():
            try:
                capture_live_snapshot(self)
            except Exception as exc:
                with self._live_cache_lock:
                    self._live_cache_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
            if self._live_cache_stop.wait(max(0.1, float(self.live_refresh_seconds))):
                break

    def dashboard_snapshot_from_cache(self: Any) -> dict[str, Any]:
        # HTTP handlers never enter runtime.metrics()/dashboard_metrics(). They
        # only serve a snapshot prepared by the dedicated live sampler.
        with self._live_cache_lock:
            if self._live_cache:
                return dict(self._live_cache)
            error = None if self._live_cache_error is None else dict(self._live_cache_error)

        # The 30-second JSONL logger is an independent fallback source. It is
        # known-good even if a live provider call is temporarily blocked.
        log_path = getattr(self, "log_path", None)
        if log_path is not None:
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
                if lines:
                    snapshot = json.loads(lines[-1])
                    if isinstance(snapshot, dict) and "primary_dashboard" in snapshot:
                        snapshot["model_history"] = self._model_history_snapshot()
                        return snapshot
            except (OSError, ValueError, TypeError):
                pass

        result: dict[str, Any] = {
            "primary_dashboard": {"dashboard_status": "starting"},
            "model_history": self._model_history_snapshot(),
        }
        if error is not None:
            result["dashboard_metrics_error"] = error
        return result

    def start_with_live_cache(self: Any) -> None:
        original_start(self)
        if self._live_cache_thread is None or not self._live_cache_thread.is_alive():
            self._live_cache_stop.clear()
            self._live_cache_thread = Thread(
                target=live_cache_loop,
                args=(self,),
                name="v9-dashboard-live-cache",
                daemon=True,
            )
            self._live_cache_thread.start()

    def close_with_live_cache(self: Any) -> None:
        self._live_cache_stop.set()
        thread = self._live_cache_thread
        if thread is not None:
            thread.join(timeout=max(0.2, min(2.0, float(self.live_refresh_seconds) + 0.2)))
        original_close(self)

    server_cls.__init__ = init_with_live_cache
    server_cls._capture_live_snapshot = capture_live_snapshot
    server_cls._dashboard_snapshot = dashboard_snapshot_from_cache
    server_cls.start = start_with_live_cache
    server_cls.close = close_with_live_cache


def install() -> None:
    if getattr(concurrent, "_startup_safety_installed", False):
        _install_direct_live_observability()
        return
    concurrent._startup_safety_installed = True

    original_session_start = concurrent.ConcurrentTransferValidationSession.start

    def start_after_process_server(self: Any) -> None:
        # The forkserver must exist before any supervisor/background thread is
        # created. Starting the server from a multi-threaded parent can stall
        # the initial actor-process prefill before sampling telemetry begins.
        ensure_process_server_ready(
            getattr(self.runtime.config, "multiprocessing_start_method", None)
        )
        original_session_start(self)

    concurrent.ConcurrentTransferValidationSession.start = start_after_process_server

    def ensure_validation_executor(self: Any) -> Any:
        if self._executor is not None:
            return self._executor
        factory_qualname = str(getattr(self.adapter_factory, "__qualname__", ""))
        process_safe = bool(getattr(self.adapter_factory, "__module__", "")) and (
            "<locals>" not in factory_qualname
        )
        if process_safe:
            method = ensure_process_server_ready(
                getattr(self.runtime.config, "multiprocessing_start_method", None)
            )
            self._executor = concurrent.ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=concurrent.mp.get_context(method),
                initializer=concurrent._base._transfer_worker_init,
            )
            executor_name = f"process:{method}"
        else:
            self._executor = concurrent.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="v9-transfer-concurrent",
            )
            executor_name = "thread_fallback"
        self.runtime.set_telemetry_gauge(
            "transfer_validation_executor", executor_name
        )
        return self._executor

    concurrent.ConcurrentTransferValidationSession._ensure_executor = ensure_validation_executor

    original_start_actor = ProcessTopology.start_actor

    def start_actor_with_prefill_progress(self: ProcessTopology, *args: Any, **kwargs: Any) -> None:
        progress = self.__dict__.get("_actor_prefill_progress")
        if progress is None and len(self.actor_processes) == 0 and self.actors > 1:
            progress = InlineProgress(f"{time.strftime('[%H:%M]')} actor prefill")
            self.__dict__["_actor_prefill_progress"] = progress
            progress.update(f"0/{self.actors}")
        original_start_actor(self, *args, **kwargs)
        progress = self.__dict__.get("_actor_prefill_progress")
        if progress is not None:
            launched = min(len(self.actor_processes), self.actors)
            if launched >= self.actors:
                progress.finish(f"{launched}/{self.actors} ready")
                self.__dict__.pop("_actor_prefill_progress", None)
            else:
                progress.update(f"{launched}/{self.actors}")

    ProcessTopology.start_actor = start_actor_with_prefill_progress
    _install_direct_live_observability()
