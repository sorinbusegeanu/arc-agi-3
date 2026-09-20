from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from multiprocessing import resource_tracker, shared_memory
import pickle
import hashlib
import struct
from threading import RLock
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


class SlabOwnership(str, Enum):
    GRANT_FREE = "grant_free"
    WORKER_OWNED = "worker_owned"
    COORDINATOR_OWNED = "coordinator_owned"


_DESCRIPTOR = struct.Struct(">64sIIIIQQQQ32s")


@dataclass(frozen=True, slots=True)
class TransportSlabDescriptor:
    slab_name: str
    slab_index: int
    offset: int
    length: int
    rows: int
    producer_id: int
    start_sequence: int
    end_sequence: int
    ownership_epoch: int
    checksum: str

    def __post_init__(self) -> None:
        if len(self.slab_name.encode("ascii")) > 63:
            raise ValueError("transport slab name exceeds fixed descriptor width")
        if min(
            self.slab_index,
            self.offset,
            self.length,
            self.rows,
            self.producer_id,
            self.start_sequence,
            self.end_sequence,
            self.ownership_epoch,
        ) < 0:
            raise ValueError("transport descriptor values must be non-negative")
        if len(self.checksum) != 64 or any(character not in "0123456789abcdef" for character in self.checksum):
            raise ValueError("transport descriptor checksum must be SHA-256")
        if self.rows and self.end_sequence < self.start_sequence:
            raise ValueError("transport descriptor sequence range is inverted")

    def pack(self) -> bytes:
        return _DESCRIPTOR.pack(
            self.slab_name.encode("ascii").ljust(64, b"\0"),
            self.slab_index,
            self.offset,
            self.length,
            self.rows,
            self.producer_id,
            self.start_sequence,
            self.end_sequence,
            self.ownership_epoch,
            bytes.fromhex(self.checksum),
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "TransportSlabDescriptor":
        if len(payload) != _DESCRIPTOR.size:
            raise ValueError("invalid fixed transport descriptor size")
        values = _DESCRIPTOR.unpack(payload)
        return cls(
            values[0].rstrip(b"\0").decode("ascii"),
            *values[1:-1],
            values[-1].hex(),
        )

    @classmethod
    def binary_size(cls) -> int:
        return _DESCRIPTOR.size


@dataclass(frozen=True, slots=True)
class TransportBatchBundle:
    descriptors: tuple[TransportSlabDescriptor, ...]

    @property
    def rows(self) -> int:
        return sum(row.rows for row in self.descriptors)

    @property
    def payload_bytes(self) -> int:
        return sum(row.length for row in self.descriptors)


class TransportSlabPool:
    """Fixed-size write-once shared-memory slabs with explicit credit ownership."""

    def __init__(
        self,
        *,
        slab_count: int = 8,
        slab_bytes: int = 16 * 1024 * 1024,
        global_byte_ceiling: int = 128 * 1024 * 1024,
        ownership_epoch: int = 0,
    ) -> None:
        if min(slab_count, slab_bytes, global_byte_ceiling) <= 0:
            raise ValueError("transport slab bounds must be positive")
        if slab_count * slab_bytes > global_byte_ceiling:
            raise ValueError("transport slabs exceed the global transport ceiling")
        self.slab_count = int(slab_count)
        self.slab_bytes = int(slab_bytes)
        self.global_byte_ceiling = int(global_byte_ceiling)
        self.ownership_epoch = int(ownership_epoch)
        self._lock = RLock()
        self._slabs = tuple(shared_memory.SharedMemory(create=True, size=self.slab_bytes) for _ in range(self.slab_count))
        self._ownership = [SlabOwnership.GRANT_FREE for _ in self._slabs]
        self._lengths = [0 for _ in self._slabs]
        self._closed = False

    @property
    def tracked_bytes(self) -> int:
        return self.slab_count * self.slab_bytes

    def ownership_bytes(self) -> dict[str, int]:
        with self._lock:
            return {
                state.value: sum(self.slab_bytes for value in self._ownership if value is state)
                for state in SlabOwnership
            }

    def write(
        self,
        payload: bytes,
        *,
        producer_id: int,
        start_sequence: int,
        end_sequence: int,
        rows: int,
    ) -> TransportSlabDescriptor:
        if not payload or len(payload) > self.slab_bytes:
            raise OverflowError("transport payload is empty or exceeds one slab")
        with self._lock:
            if self._closed:
                raise RuntimeError("transport slab pool is closed")
            try:
                index = self._ownership.index(SlabOwnership.GRANT_FREE)
            except ValueError as exc:
                raise BufferError("transport slab pool has no free credit") from exc
            slab = self._slabs[index]
            slab.buf[: len(payload)] = payload
            self._lengths[index] = len(payload)
            self._ownership[index] = SlabOwnership.WORKER_OWNED
            return TransportSlabDescriptor(
                slab.name,
                index,
                0,
                len(payload),
                int(rows),
                int(producer_id),
                int(start_sequence),
                int(end_sequence),
                self.ownership_epoch,
                hashlib.sha256(payload).hexdigest(),
            )

    def transfer_to_coordinator(self, descriptor: TransportSlabDescriptor) -> None:
        with self._lock:
            self._validate_local(descriptor)
            if self._ownership[descriptor.slab_index] is not SlabOwnership.WORKER_OWNED:
                raise RuntimeError("only worker-owned slabs can transfer to coordinator")
            self._ownership[descriptor.slab_index] = SlabOwnership.COORDINATOR_OWNED

    def read(self, descriptor: TransportSlabDescriptor) -> bytes:
        with self._lock:
            self._validate_local(descriptor)
            if self._ownership[descriptor.slab_index] is SlabOwnership.GRANT_FREE:
                raise RuntimeError("cannot read a free transport slab")
            payload = bytes(
                self._slabs[descriptor.slab_index].buf[
                    descriptor.offset : descriptor.offset + descriptor.length
                ]
            )
            if hashlib.sha256(payload).hexdigest() != descriptor.checksum:
                raise RuntimeError("transport slab checksum mismatch")
            return payload

    def release(self, descriptor: TransportSlabDescriptor, *, owner: SlabOwnership) -> None:
        with self._lock:
            self._validate_local(descriptor)
            if self._ownership[descriptor.slab_index] is not owner:
                raise RuntimeError("transport slab release ownership mismatch")
            self._ownership[descriptor.slab_index] = SlabOwnership.GRANT_FREE
            self._lengths[descriptor.slab_index] = 0

    def _validate_local(self, descriptor: TransportSlabDescriptor) -> None:
        if descriptor.ownership_epoch != self.ownership_epoch:
            raise RuntimeError("stale transport ownership epoch")
        if not 0 <= descriptor.slab_index < self.slab_count:
            raise ValueError("transport slab index is outside the pool")
        if descriptor.slab_name != self._slabs[descriptor.slab_index].name:
            raise ValueError("transport descriptor does not belong to this pool")
        if descriptor.offset + descriptor.length > self.slab_bytes:
            raise ValueError("transport descriptor range exceeds slab")
        if descriptor.length != self._lengths[descriptor.slab_index]:
            raise ValueError("transport descriptor length is stale")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for slab in self._slabs:
                slab.close()
                try:
                    slab.unlink()
                except FileNotFoundError:
                    pass
            self._closed = True

    def recover_worker_death(self) -> int:
        """Return only worker-owned credits; coordinator-owned data stays live."""
        with self._lock:
            recovered = 0
            for index, owner in enumerate(self._ownership):
                if owner is SlabOwnership.WORKER_OWNED:
                    self._ownership[index] = SlabOwnership.GRANT_FREE
                    self._lengths[index] = 0
                    recovered += self.slab_bytes
            return recovered

    def __enter__(self) -> "TransportSlabPool":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(slots=True)
class _ProducerAdmissionState:
    next_sequence: int = 1
    held: dict[int, TransportSlabDescriptor] = None  # type: ignore[assignment]
    held_bytes: int = 0
    first_gap_at: float | None = None

    def __post_init__(self) -> None:
        if self.held is None:
            self.held = {}


class ProducerCausalAdmission:
    def __init__(self, *, gap_rows_limit: int = 4096, gap_bytes_limit: int = 64 * 1024 * 1024, gap_timeout_seconds: float = 30.0, producer_limit: int = 4096) -> None:
        if min(gap_rows_limit, gap_bytes_limit, gap_timeout_seconds, producer_limit) <= 0:
            raise ValueError("producer gap bounds must be positive")
        self.gap_rows_limit = int(gap_rows_limit)
        self.gap_bytes_limit = int(gap_bytes_limit)
        self.gap_timeout_seconds = float(gap_timeout_seconds)
        self.producer_limit = int(producer_limit)
        self._states: dict[int, _ProducerAdmissionState] = {}
        self._held_rows = 0
        self._held_bytes = 0

    def admit(self, descriptor: TransportSlabDescriptor, *, now: float | None = None) -> tuple[TransportSlabDescriptor, ...]:
        timestamp = time.monotonic() if now is None else float(now)
        state = self._states.get(descriptor.producer_id)
        if state is None:
            if len(self._states) >= self.producer_limit:
                raise OverflowError("producer causal-admission identity ceiling exceeded")
            state = _ProducerAdmissionState()
            self._states[descriptor.producer_id] = state
        if descriptor.end_sequence < state.next_sequence:
            raise RuntimeError("duplicate/stale producer range")
        if descriptor.start_sequence < state.next_sequence or descriptor.start_sequence in state.held:
            raise RuntimeError("overlapping producer range")
        for held in state.held.values():
            if not (descriptor.end_sequence < held.start_sequence or descriptor.start_sequence > held.end_sequence):
                raise RuntimeError("overlapping producer range")
        if descriptor.start_sequence > state.next_sequence:
            state.held[descriptor.start_sequence] = descriptor
            state.held_bytes += descriptor.length
            self._held_rows += descriptor.rows
            self._held_bytes += descriptor.length
            state.first_gap_at = timestamp if state.first_gap_at is None else state.first_gap_at
            if self._held_rows > self.gap_rows_limit or self._held_bytes > self.gap_bytes_limit:
                state.held.pop(descriptor.start_sequence, None)
                state.held_bytes -= descriptor.length
                self._held_rows -= descriptor.rows
                self._held_bytes -= descriptor.length
                if not state.held:
                    state.first_gap_at = None
                    if state.next_sequence == 1:
                        self._states.pop(descriptor.producer_id, None)
                raise OverflowError("producer gap window ceiling exceeded")
            return ()
        ready = [descriptor]
        state.next_sequence = descriptor.end_sequence + 1
        while state.next_sequence in state.held:
            following = state.held.pop(state.next_sequence)
            state.held_bytes -= following.length
            self._held_rows -= following.rows
            self._held_bytes -= following.length
            ready.append(following)
            state.next_sequence = following.end_sequence + 1
        state.first_gap_at = None if not state.held else state.first_gap_at
        return tuple(ready)

    def check_gap_timeouts(self, *, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        expired = [producer for producer, state in self._states.items() if state.first_gap_at is not None and timestamp - state.first_gap_at > self.gap_timeout_seconds]
        if expired:
            raise TimeoutError(f"producer transport gap timeout: {min(expired)}")


class ProducerAffinityRouter:
    def __init__(self, shards: int) -> None:
        if shards <= 0:
            raise ValueError("shards must be positive")
        self.shards = int(shards)

    def shard_for(self, producer_id: int) -> int:
        raw = int(producer_id).to_bytes(16, "big", signed=False)
        return int.from_bytes(hashlib.blake2b(raw, digest_size=8, person=b"v9-affinity").digest(), "big") % self.shards


def publish_shared_batch(value: Any, *, start_sequence: int, end_sequence: int, rows: int) -> SharedBatchDescriptor:
    started = time.perf_counter()
    payload = pickle.dumps(value, protocol=5)
    segment = shared_memory.SharedMemory(create=True, size=max(1, len(payload)))
    try:
        segment.buf[: len(payload)] = payload
        name = segment.name
    finally:
        segment.close()
    try:
        resource_tracker.unregister(segment._name, "shared_memory")
    except (AttributeError, KeyError):
        pass
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
        value = pickle.loads(segment.buf[: int(descriptor.size)])
    finally:
        segment.close()
        segment.unlink()
    return value, 1000.0 * (time.perf_counter() - started)
