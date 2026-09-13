from __future__ import annotations

from typing import Any

from .memory_pipeline import derivation_worker_main, ingest_worker_main
from .multiprocess import WorkerStop


def _close_queue(queue_obj: Any, *, drain: bool) -> None:
    if queue_obj is None:
        return
    if not drain:
        try:
            queue_obj.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
    try:
        queue_obj.close()
    except (AttributeError, OSError, ValueError):
        pass
    if drain:
        try:
            queue_obj.join_thread()
        except (AttributeError, AssertionError, OSError, ValueError):
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
    def __init__(self, ctx: Any, *, ingest_workers: int, derivation_workers: int, ingest_queue_capacity: int, derivation_queue_capacity: int, result_queue_capacity: int) -> None:
        self.ctx = ctx
        self.ingest_workers = int(ingest_workers)
        self.derivation_workers = int(derivation_workers)
        self.ingest_queue = ctx.Queue(maxsize=int(ingest_queue_capacity))
        self.derivation_queue = ctx.Queue(maxsize=int(derivation_queue_capacity))
        self.result_queue = ctx.Queue(maxsize=int(result_queue_capacity))
        self.ingest_processes = []
        self.derivation_processes = []
        self._closed = False

    def start(self) -> None:
        for index in range(self.ingest_workers):
            process = self.ctx.Process(target=ingest_worker_main, args=(self.ingest_queue, self.result_queue), name=f"v9-ingest-{index}")
            process.start()
            self.ingest_processes.append(process)
        for index in range(self.derivation_workers):
            process = self.ctx.Process(target=derivation_worker_main, args=(self.derivation_queue, self.result_queue), name=f"v9-derive-{index}")
            process.start()
            self.derivation_processes.append(process)

    def signal_ingest_stop(self) -> None:
        for _ in self.ingest_processes:
            self.ingest_queue.put(WorkerStop())

    @staticmethod
    def _join_checked(processes: list[Any], *, role: str, timeout: float = 30.0) -> None:
        for process in processes:
            process.join(timeout=float(timeout))
            if process.is_alive():
                raise RuntimeError(f"{role} process {process.name} did not exit within {float(timeout):.1f}s")
            if process.exitcode != 0:
                raise RuntimeError(f"{role} process {process.name} exited with code {process.exitcode}")

    def join_ingest(self) -> None:
        self._join_checked(self.ingest_processes, role="ingest")

    def signal_derivation_stop(self) -> None:
        for _ in self.derivation_processes:
            self.derivation_queue.put(WorkerStop())

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
        for queue_obj in (self.ingest_queue, self.derivation_queue, self.result_queue):
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
        return {
            "ingest_queue_depth": self._safe_qsize(self.ingest_queue),
            "derivation_queue_depth": self._safe_qsize(self.derivation_queue),
            "memory_result_queue_depth": self._safe_qsize(self.result_queue),
        }
