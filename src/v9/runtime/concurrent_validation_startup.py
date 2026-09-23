from __future__ import annotations

import builtins
import os
import sys
import time
from typing import Any

from . import concurrent_transfer_validation as concurrent
from . import post_sampling_progress as post_progress
from .multiprocess import ProcessTopology, ensure_process_server_ready
from .progress import InlineProgress


def _terminal_print(*values: object, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:
    """Use fd 1 only for an actual interactive terminal; preserve normal capture elsewhere."""
    target = sys.stdout if file is None else file
    direct_terminal = False
    if target in (sys.stdout, sys.__stdout__):
        try:
            direct_terminal = bool(target.isatty() and target.fileno() == 1)
        except (AttributeError, OSError, ValueError):
            direct_terminal = False
    if direct_terminal:
        text = sep.join(str(value) for value in values) + end
        try:
            os.write(1, text.encode("utf-8", errors="replace"))
            return
        except (BrokenPipeError, OSError, ValueError):
            pass
    builtins.print(*values, sep=sep, end=end, file=target, flush=flush)


def _install_direct_live_observability() -> None:
    # Terminal output remains independent from carriage-return progress renderers.
    # Dashboard live caching is implemented natively by MetricsHTTPServer.
    post_progress.print = _terminal_print


def install() -> None:
    if getattr(concurrent, "_startup_safety_installed", False):
        _install_direct_live_observability()
        return
    concurrent._startup_safety_installed = True

    original_session_start = concurrent.ConcurrentTransferValidationSession.start

    def start_after_process_server(self: Any) -> None:
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
