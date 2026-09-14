from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import pickle
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class SharedBatchDescriptor:
    name: str
    size: int
    start_sequence: int
    end_sequence: int
    rows: int
    encode_ms: float


def publish_shared_batch(value: Any, *, start_sequence: int, end_sequence: int, rows: int) -> SharedBatchDescriptor:
    started = time.perf_counter()
    payload = pickle.dumps(value, protocol=5)
    segment = shared_memory.SharedMemory(create=True, size=max(1, len(payload)))
    try:
        segment.buf[: len(payload)] = payload
        name = segment.name
    finally:
        segment.close()
    return SharedBatchDescriptor(
        name=name,
        size=len(payload),
        start_sequence=int(start_sequence),
        end_sequence=int(end_sequence),
        rows=int(rows),
        encode_ms=1000.0 * (time.perf_counter() - started),
    )


def consume_shared_batch(descriptor: SharedBatchDescriptor) -> tuple[Any, float]:
    started = time.perf_counter()
    segment = shared_memory.SharedMemory(name=descriptor.name, create=False)
    try:
        payload = bytes(segment.buf[: int(descriptor.size)])
    finally:
        segment.close()
        segment.unlink()
    value = pickle.loads(payload)
    return value, 1000.0 * (time.perf_counter() - started)
