from __future__ import annotations

import time
from typing import Any

from .memory_pipeline import derivation_batch_worker_main, ingest_batch_worker_main
from .multiprocess import WorkerStop


def _close_queue(queue_obj: Any, *, drain: bool) -> None:
    if queue_obj is None:
        return
    try:
        queue_obj.cancel_join_thread()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        queue_obj.close()
    except (AttributeError, OSError, ValueError):
        pass


def _close_process(process: Any) -> None:
    try:
        if process.is_alive():
            return
    except (AttributeError, ValueError):
        return
    try:
        process.close()
    except (AttributeError, OSError, ValueError):
        pass


class MemoryWorkerTopology:
    def __init__(
        self,
        ctx: Any,
        *,
        ingest_workers: int,
        derivation_workers: int,
        ingest_queue_capacity: int,
        derivation_queue_capacity: int,
        ingest_result_queue_capacity: int | None = None,
        derivation_result_queue_capacity: int | None = None,
        result_queue_capacity: int | None = None,
    ) -> None:
        self.ctx = ctx
        self.ingest_workers = int(ingest_workers)
        self.derivation_workers = int(derivation_workers)
        legacy_capacity = max(1, int(result_queue_capacity or 1024))
        ingest_results = max(1, int(ingest_result_queue_capacity or legacy_capacity))
        derivation_results = max(1, int(derivation_result_queue_capacity or legacy_capacity))
        self.ingest_queue = ctx.Queue(maxsize=int(ingest_queue_capacity))
        self.derivation_queue = ctx.Queue(maxsize=int(derivation_queue_capacity))
        self.ingest_result_queue = ctx.Queue(maxsize=ingest_results)
        self.derivation_result_queue = ctx.Queue(maxsize=derivation_results)
        self.result_queue = self.ingest_result_queue
        self.ingest_processes: list[Any] = []
        self.derivation_processes: list[Any] = []
        self._closed = False

    def start(self) -> None:
        for index in range(self.ingest_workers):
            process = self.ctx.Process(
                target=ingest_batch_worker_main,
                args=(self.ingest_queue, self.ingest_result_queue),
                name=f"v9-ingest-{index}",
            )
            process.start()
            self.ingest_processes.append(process)
        for index in range(self.derivation_workers):
            process = self.ctx.Process(
                target=derivation_batch_worker_main,
                args=(self.derivation_queue, self.derivation_result_queue),
                name=f"v9-derive-{index}",
            )
            process.start()
            self.derivation_processes.append(process)

    def signal_ingest_stop(self) -> None:
        for _ in self.ingest_processes:
            self.ingest_queue.put(WorkerStop())

    def signal_derivation_stop(self) -> None:
        for _ in self.derivation_processes:
            self.derivation_queue.put(WorkerStop())

    @staticmethod
    def _join_checked(processes: list[Any], *, role: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout)
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        stragglers = [process for process in processes if process.is_alive()]
        for process in stragglers:
            process.terminate()
        for process in stragglers:
            process.join(timeout=2.0)
        failed = [process for process in processes if process.exitcode not in (0, -15)]
        if failed:
            details = ", ".join(f"{process.name}:{process.exitcode}" for process in failed)
            raise RuntimeError(f"{role} worker shutdown failed: {details}")

    def join_ingest(self) -> None:
        self._join_checked(self.ingest_processes, role="ingest")

    def join_derivation(self) -> None:
        self._join_checked(self.derivation_processes, role="derivation")

    def terminate(self) -> None:
        for process in self.ingest_processes + self.derivation_processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    def close(self, *, drain: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        for queue_obj in (
            self.ingest_queue,
            self.derivation_queue,
            self.ingest_result_queue,
            self.derivation_result_queue,
        ):
            _close_queue(queue_obj, drain=drain)
        for process in self.ingest_processes + self.derivation_processes:
            _close_process(process)

    @staticmethod
    def _safe_qsize(queue_obj: Any) -> int:
        try:
            return int(queue_obj.qsize())
        except (NotImplementedError, AttributeError, OSError, ValueError):
            return -1

    def queue_depths(self) -> dict[str, int]:
        ingest_results = self._safe_qsize(self.ingest_result_queue)
        derivation_results = self._safe_qsize(self.derivation_result_queue)
        combined = -1 if ingest_results < 0 or derivation_results < 0 else ingest_results + derivation_results
        return {
            "ingest_queue_depth": self._safe_qsize(self.ingest_queue),
            "derivation_queue_depth": self._safe_qsize(self.derivation_queue),
            "ingest_result_queue_depth": ingest_results,
            "derivation_result_queue_depth": derivation_results,
            "memory_result_queue_depth": combined,
        }
