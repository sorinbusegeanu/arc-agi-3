from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from struct import Struct
from typing import Iterator

from v8.model import MemoryUid, u64

_HEADER = Struct("<QQ")
_NODE = Struct("<QQQBHBQQQQqdddddddQQdQQBB")
_EDGE = Struct("<QQHQQqQ")
_ACTION = Struct("<BQiqddQ")


@dataclass(frozen=True, slots=True)
class ArenaDescriptor:
    name: str
    capacity: int
    kind: str


@dataclass(frozen=True, slots=True)
class NodeRecord:
    uid: MemoryUid
    fingerprint: int
    level: int
    memory_type: int
    key_parts: tuple[int, ...]
    support_count: int
    significance_sum: float
    prediction_error_sum: float
    learning_value_sum: float
    transfer_prior_sum: float
    explanatory_sum: float
    future_option_sum: float
    score_weight: float
    strategy_attempt_count: int = 0
    strategy_success_count: int = 0
    strategy_cost_sum: float = 0.0
    updated_watermark: int = 0
    game_mask: int = 0
    cognitive_state: int = 0
    validation_state: int = 0

    @property
    def significance(self) -> float:
        return 0.0 if self.score_weight <= 0 else self.significance_sum / self.score_weight

    @property
    def prediction_error(self) -> float:
        return 0.0 if self.score_weight <= 0 else self.prediction_error_sum / self.score_weight

    @property
    def learning_value(self) -> float:
        return 0.0 if self.score_weight <= 0 else self.learning_value_sum / self.score_weight

    @property
    def transfer_prior(self) -> float:
        return 0.0 if self.score_weight <= 0 else self.transfer_prior_sum / self.score_weight

    @property
    def explanatory_reach(self) -> float:
        return 0.0 if self.score_weight <= 0 else self.explanatory_sum / self.score_weight

    @property
    def future_option_delta(self) -> float:
        return 0.0 if self.score_weight <= 0 else self.future_option_sum / self.score_weight

    @property
    def game_evidence_count(self) -> int:
        return int(self.game_mask).bit_count()

    @property
    def strategy_reliability(self) -> float:
        return 0.0 if self.strategy_attempt_count <= 0 else self.strategy_success_count / self.strategy_attempt_count

    @property
    def strategy_mean_cost(self) -> float:
        return 1.0 if self.strategy_success_count <= 0 else self.strategy_cost_sum / self.strategy_success_count


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    source_uid: MemoryUid
    relation_type: int
    target_uid: MemoryUid
    support_count: int
    updated_watermark: int


@dataclass(frozen=True, slots=True)
class ActionRecord:
    context_signature: int
    action_id: int
    support_count: int
    score_sum: float
    score_weight: float
    updated_watermark: int

    @property
    def score(self) -> float:
        return 0.0 if self.score_weight <= 0 else self.score_sum / self.score_weight


class _SharedArena:
    record: Struct
    kind: str

    def __init__(self, *, capacity: int, create: bool = True, name: str | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._owner = bool(create)
        self._shm = SharedMemory(create=create, size=_HEADER.size + self.capacity * self.record.size if create else 0, name=name)
        if create:
            self._shm.buf[:] = b"\0" * len(self._shm.buf)

    @property
    def descriptor(self) -> ArenaDescriptor:
        return ArenaDescriptor(self._shm.name, self.capacity, self.kind)

    @classmethod
    def attach(cls, descriptor: ArenaDescriptor):
        if descriptor.kind != cls.kind:
            raise ValueError(f"expected {cls.kind} arena, got {descriptor.kind}")
        return cls(capacity=descriptor.capacity, create=False, name=descriptor.name)

    @property
    def count(self) -> int:
        return int(_HEADER.unpack_from(self._shm.buf, 0)[0])

    @property
    def sequence(self) -> int:
        return int(_HEADER.unpack_from(self._shm.buf, 0)[1])

    def _set_header(self, count: int, sequence: int) -> None:
        _HEADER.pack_into(self._shm.buf, 0, int(count), int(sequence))

    def begin_write(self) -> None:
        count, seq = _HEADER.unpack_from(self._shm.buf, 0)
        if int(seq) & 1:
            raise RuntimeError("arena write already active")
        self._set_header(int(count), int(seq) + 1)

    def end_write(self, *, count: int | None = None) -> None:
        current_count, seq = _HEADER.unpack_from(self._shm.buf, 0)
        if not int(seq) & 1:
            raise RuntimeError("arena write not active")
        self._set_header(int(current_count if count is None else count), int(seq) + 1)

    def _offset(self, row: int) -> int:
        if row < 0 or row >= self.capacity:
            raise IndexError(row)
        return _HEADER.size + row * self.record.size

    def snapshot_bytes(self, *, retries: int = 1000) -> bytes:
        for _ in range(retries):
            count1, seq1 = _HEADER.unpack_from(self._shm.buf, 0)
            if int(seq1) & 1:
                continue
            length = _HEADER.size + int(count1) * self.record.size
            payload = bytes(self._shm.buf[:length])
            count2, seq2 = _HEADER.unpack_from(self._shm.buf, 0)
            if int(seq1) == int(seq2) and int(count1) == int(count2) and not (int(seq2) & 1):
                return payload
        raise RuntimeError(f"could not obtain stable {self.kind} arena snapshot")

    def load_snapshot(self, payload: bytes) -> None:
        if len(payload) < _HEADER.size:
            raise ValueError("invalid arena snapshot")
        count, seq = _HEADER.unpack_from(payload, 0)
        expected = _HEADER.size + int(count) * self.record.size
        if len(payload) != expected:
            raise ValueError("arena snapshot size mismatch")
        if int(count) > self.capacity:
            raise ValueError("snapshot exceeds arena capacity")
        self._shm.buf[:] = b"\0" * len(self._shm.buf)
        self._shm.buf[:len(payload)] = payload
        if int(seq) & 1:
            self._set_header(int(count), int(seq) + 1)

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


class SharedNodeArena(_SharedArena):
    record = _NODE
    kind = "nodes"

    def write(self, row: int, value: NodeRecord) -> None:
        parts = tuple(u64(v) for v in value.key_parts)
        if len(parts) > 4:
            raise ValueError("node key has more than four hot-path parts")
        padded = parts + (0,) * (4 - len(parts))
        _NODE.pack_into(
            self._shm.buf, self._offset(row), u64(value.uid.hi), u64(value.uid.lo), u64(value.fingerprint),
            int(value.level), int(value.memory_type), len(parts), *padded, int(value.support_count),
            float(value.significance_sum), float(value.prediction_error_sum), float(value.learning_value_sum),
            float(value.transfer_prior_sum), float(value.explanatory_sum), float(value.future_option_sum),
            float(value.score_weight), u64(value.strategy_attempt_count), u64(value.strategy_success_count),
            float(value.strategy_cost_sum), u64(value.updated_watermark), u64(value.game_mask),
            int(value.cognitive_state) & 0xFF, int(value.validation_state) & 0xFF,
        )

    def read(self, row: int) -> NodeRecord:
        values = _NODE.unpack_from(self._shm.buf, self._offset(row))
        return NodeRecord(
            uid=MemoryUid(values[0], values[1]), fingerprint=int(values[2]), level=int(values[3]), memory_type=int(values[4]),
            key_parts=tuple(values[6:10][: int(values[5])]), support_count=int(values[10]), significance_sum=float(values[11]),
            prediction_error_sum=float(values[12]), learning_value_sum=float(values[13]), transfer_prior_sum=float(values[14]),
            explanatory_sum=float(values[15]), future_option_sum=float(values[16]), score_weight=float(values[17]),
            strategy_attempt_count=int(values[18]), strategy_success_count=int(values[19]), strategy_cost_sum=float(values[20]),
            updated_watermark=int(values[21]), game_mask=int(values[22]), cognitive_state=int(values[23]), validation_state=int(values[24]),
        )

    def records(self) -> Iterator[NodeRecord]:
        for row in range(self.count):
            yield self.read(row)


class SharedEdgeArena(_SharedArena):
    record = _EDGE
    kind = "edges"

    def write(self, row: int, value: EdgeRecord) -> None:
        _EDGE.pack_into(self._shm.buf, self._offset(row), u64(value.source_uid.hi), u64(value.source_uid.lo), int(value.relation_type), u64(value.target_uid.hi), u64(value.target_uid.lo), int(value.support_count), u64(value.updated_watermark))

    def read(self, row: int) -> EdgeRecord:
        source_hi, source_lo, relation, target_hi, target_lo, support, watermark = _EDGE.unpack_from(self._shm.buf, self._offset(row))
        return EdgeRecord(MemoryUid(source_hi, source_lo), int(relation), MemoryUid(target_hi, target_lo), int(support), int(watermark))

    def records(self) -> Iterator[EdgeRecord]:
        for row in range(self.count):
            yield self.read(row)


class SharedActionArena(_SharedArena):
    record = _ACTION
    kind = "actions"

    def __init__(self, *, capacity: int, create: bool = True, name: str | None = None) -> None:
        super().__init__(capacity=capacity, create=create, name=name)
        if create:
            self._set_header(self.capacity, 0)

    def write(self, row: int, value: ActionRecord, *, occupied: bool = True) -> None:
        _ACTION.pack_into(self._shm.buf, self._offset(row), 1 if occupied else 0, u64(value.context_signature), int(value.action_id), int(value.support_count), float(value.score_sum), float(value.score_weight), u64(value.updated_watermark))

    def read_slot(self, row: int) -> tuple[bool, ActionRecord]:
        occupied, context, action, support, score_sum, weight, watermark = _ACTION.unpack_from(self._shm.buf, self._offset(row))
        return bool(occupied), ActionRecord(int(context), int(action), int(support), float(score_sum), float(weight), int(watermark))

    def lookup(self, context_signature: int, action_id: int) -> ActionRecord | None:
        start = (u64(context_signature) ^ (int(action_id) * 0x9E3779B185EBCA87)) % self.capacity
        for _attempt in range(8):
            seq1 = self.sequence
            if seq1 & 1:
                continue
            found = None
            for offset in range(self.capacity):
                row = int((start + offset) % self.capacity)
                occupied, value = self.read_slot(row)
                if not occupied:
                    break
                if value.context_signature == u64(context_signature) and value.action_id == int(action_id):
                    found = value
                    break
            seq2 = self.sequence
            if seq1 == seq2 and not (seq2 & 1):
                return found
        return None
