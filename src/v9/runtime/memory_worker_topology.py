from __future__ import annotations

from typing import Any

from .memory_pipeline import derivation_worker_main, ingest_worker_main
from .multiprocess import WorkerStop


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

    def join_ingest(self) -> None:
        for process in self.ingest_processes:
            process.join(timeout=30)

    def signal_derivation_stop(self) -> None:
        for _ in self.derivation_processes:
            self.derivation_queue.put(WorkerStop())

    def join_derivation(self) -> None:
        for process in self.derivation_processes:
            process.join(timeout=30)

    @staticmethod
    def _safe_qsize(queue_obj: Any) -> int:
        try:
            return int(queue_obj.qsize())
        except (NotImplementedError, AttributeError):
            return -1

    def queue_depths(self) -> dict[str, int]:
        return {
            "ingest_queue_depth": self._safe_qsize(self.ingest_queue),
            "derivation_queue_depth": self._safe_qsize(self.derivation_queue),
            "memory_result_queue_depth": self._safe_qsize(self.result_queue),
        }
