from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory


@dataclass(frozen=True, slots=True)
class RingDescriptor:
    name: str
    capacity: int
    slot_size: int


class SharedRingBuffer:
    """Bounded fixed-slot shared-memory ring.

    The hot path copies compact binary packets directly into shared memory. Put and
    get use independent locks so producers and consumers do not serialize behind one
    global mutex. Semaphores provide bounded backpressure without disk spill.
    """

    def __init__(
        self,
        *,
        capacity: int,
        slot_size: int,
        create: bool = True,
        name: str | None = None,
        free=None,
        used=None,
        put_lock=None,
        get_lock=None,
        count_lock=None,
        head=None,
        tail=None,
        count=None,
    ) -> None:
        if capacity <= 0 or slot_size <= 0:
            raise ValueError("capacity and slot_size must be positive")
        self.capacity = int(capacity)
        self.slot_size = int(slot_size)
        self._record_size = 4 + self.slot_size
        self._owner = bool(create)
        self._shm = SharedMemory(
            create=create,
            size=self.capacity * self._record_size if create else 0,
            name=name,
        )
        if create:
            self._free = mp.Semaphore(self.capacity)
            self._used = mp.Semaphore(0)
            self._put_lock = mp.Lock()
            self._get_lock = mp.Lock()
            self._count_lock = mp.Lock()
            self._head = mp.Value("Q", 0, lock=False)
            self._tail = mp.Value("Q", 0, lock=False)
            self._count = mp.Value("Q", 0, lock=False)
        else:
            required = (free, used, put_lock, get_lock, count_lock, head, tail, count)
            if any(value is None for value in required):
                raise ValueError("attached rings require synchronization primitives")
            self._free = free
            self._used = used
            self._put_lock = put_lock
            self._get_lock = get_lock
            self._count_lock = count_lock
            self._head = head
            self._tail = tail
            self._count = count

    @property
    def descriptor(self) -> RingDescriptor:
        return RingDescriptor(self._shm.name, self.capacity, self.slot_size)

    def attachment_args(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "slot_size": self.slot_size,
            "create": False,
            "name": self._shm.name,
            "free": self._free,
            "used": self._used,
            "put_lock": self._put_lock,
            "get_lock": self._get_lock,
            "count_lock": self._count_lock,
            "head": self._head,
            "tail": self._tail,
            "count": self._count,
        }

    def put(self, payload: bytes, *, timeout: float | None = None) -> bool:
        if len(payload) > self.slot_size:
            raise ValueError(f"payload size {len(payload)} exceeds slot size {self.slot_size}")
        acquired = self._free.acquire(timeout=timeout) if timeout is not None else self._free.acquire()
        if not acquired:
            return False
        try:
            with self._put_lock:
                index = int(self._tail.value % self.capacity)
                offset = index * self._record_size
                self._shm.buf[offset : offset + 4] = len(payload).to_bytes(4, "little")
                body = offset + 4
                self._shm.buf[body : body + len(payload)] = payload
                if len(payload) < self.slot_size:
                    self._shm.buf[body + len(payload) : body + self.slot_size] = b"\0" * (
                        self.slot_size - len(payload)
                    )
                self._tail.value += 1
            with self._count_lock:
                self._count.value += 1
        except Exception:
            self._free.release()
            raise
        self._used.release()
        return True

    def get(self, *, timeout: float | None = None) -> bytes | None:
        acquired = self._used.acquire(timeout=timeout) if timeout is not None else self._used.acquire()
        if not acquired:
            return None
        try:
            with self._get_lock:
                index = int(self._head.value % self.capacity)
                offset = index * self._record_size
                size = int.from_bytes(self._shm.buf[offset : offset + 4], "little")
                if size < 0 or size > self.slot_size:
                    raise RuntimeError("shared ring slot contains invalid payload length")
                body = offset + 4
                payload = bytes(self._shm.buf[body : body + size])
                self._head.value += 1
            with self._count_lock:
                self._count.value -= 1
        finally:
            self._free.release()
        return payload

    @property
    def qsize(self) -> int:
        with self._count_lock:
            return int(self._count.value)

    @property
    def empty(self) -> bool:
        return self.qsize == 0

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def dispose(self) -> None:
        self.close()
        self.unlink()
