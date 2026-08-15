from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from struct import Struct
from typing import Iterator

from v8.model import MemoryUid, u64

_HEADER = Struct("<QQ")  # count, seqlock version
_NODE = Struct("<QQQBHBQQQQqddddddddddQQBB")
_EDGE_V1 = Struct("<QQHQQqQ")
_EDGE = Struct("<QQHQQqQddQQ")
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
    updated_watermark: int
    game_mask: int = 0
    cognitive_state: int = 0
    validation_state: int = 0
    success_sum: float = 0.0
    cost_sum: float = 0.0
    attempt_weight: float = 0.0

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
        return 0.0 if self.attempt_weight <= 0 else self.success_sum / self.attempt_weight

    @property
    def strategy_mean_cost(self) -> float:
        return 1.0 if self.attempt_weight <= 0 else self.cost_sum / self.attempt_weight


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    source_uid: MemoryUid
    relation_type: int
    target_uid: MemoryUid
    support_count: int
    updated_watermark: int
    score_sum: float = 0.0
    score_weight: float = 0.0
    source_version: int = 0
    target_version: int = 0

    @property
    def score(self) -> float:
        return 0.0 if self.score_weight <= 0 else self.score_sum / self.score_weight


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
        self._shm = SharedMemory(
            create=create,
            size=_HEADER.size + self.capacity * self.record.size if create else 0,
            name=name,
        )
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
        seq = int(seq)
        if seq & 1:
            raise RuntimeError("arena write already active")
        self._set_header(int(count), seq + 1)

    def end_write(self, *, count: int | None = None) -> None:
        current_count, seq = _HEADER.unpack_from(self._shm.buf, 0)
        seq = int(seq)
        if not seq & 1:
            raise RuntimeError("arena write not active")
        self._set_header(int(current_count if count is None else count), seq + 1)

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
            self._shm.buf,
            self._offset(row),
            u64(value.uid.hi),
            u64(value.uid.lo),
            u64(value.fingerprint),
            int(value.level),
            int(value.memory_type),
            len(parts),
            *padded,
            int(value.support_count),
            float(value.significance_sum),
            float(value.prediction_error_sum),
            float(value.learning_value_sum),
            float(value.transfer_prior_sum),
            float(value.explanatory_sum),
            float(value.future_option_sum),
            float(value.score_weight),
            float(value.success_sum),
            float(value.cost_sum),
            float(value.attempt_weight),
            u64(value.updated_watermark),
            u64(value.game_mask),
            int(value.cognitive_state) & 0xFF,
            int(value.validation_state) & 0xFF,
        )

    def read(self, row: int) -> NodeRecord:
        values = _NODE.unpack_from(self._shm.buf, self._offset(row))
        (
            hi,
            lo,
            fingerprint,
            level,
            memory_type,
            key_count,
            k0,
            k1,
            k2,
            k3,
            support,
            significance,
            prediction_error,
            learning_value,
            transfer_prior,
            explanatory,
            future_option,
            weight,
            success_sum,
            cost_sum,
            attempt_weight,
            watermark,
            game_mask,
            cognitive_state,
            validation_state,
        ) = values
        return NodeRecord(
            uid=MemoryUid(hi, lo),
            fingerprint=int(fingerprint),
            level=int(level),
            memory_type=int(memory_type),
            key_parts=tuple((k0, k1, k2, k3)[: int(key_count)]),
            support_count=int(support),
            significance_sum=float(significance),
            prediction_error_sum=float(prediction_error),
            learning_value_sum=float(learning_value),
            transfer_prior_sum=float(transfer_prior),
            explanatory_sum=float(explanatory),
            future_option_sum=float(future_option),
            score_weight=float(weight),
            updated_watermark=int(watermark),
            game_mask=int(game_mask),
            cognitive_state=int(cognitive_state),
            validation_state=int(validation_state),
            success_sum=float(success_sum),
            cost_sum=float(cost_sum),
            attempt_weight=float(attempt_weight),
        )

    def records(self) -> Iterator[NodeRecord]:
        count = self.count
        for row in range(count):
            yield self.read(row)


class SharedEdgeArena(_SharedArena):
    record = _EDGE
    kind = "edges"

    def write(self, row: int, value: EdgeRecord) -> None:
        _EDGE.pack_into(
            self._shm.buf,
            self._offset(row),
            u64(value.source_uid.hi),
            u64(value.source_uid.lo),
            int(value.relation_type),
            u64(value.target_uid.hi),
            u64(value.target_uid.lo),
            int(value.support_count),
            u64(value.updated_watermark),
            float(value.score_sum),
            float(value.score_weight),
            u64(value.source_version),
            u64(value.target_version),
        )

    def read(self, row: int) -> EdgeRecord:
        (
            source_hi,
            source_lo,
            relation,
            target_hi,
            target_lo,
            support,
            watermark,
            score_sum,
            score_weight,
            source_version,
            target_version,
        ) = _EDGE.unpack_from(self._shm.buf, self._offset(row))
        return EdgeRecord(
            MemoryUid(source_hi, source_lo),
            int(relation),
            MemoryUid(target_hi, target_lo),
            int(support),
            int(watermark),
            float(score_sum),
            float(score_weight),
            int(source_version),
            int(target_version),
        )

    def load_snapshot(self, payload: bytes) -> None:
        if len(payload) < _HEADER.size:
            raise ValueError("invalid arena snapshot")
        count, seq = _HEADER.unpack_from(payload, 0)
        current_size = _HEADER.size + int(count) * _EDGE.size
        if len(payload) == current_size:
            return super().load_snapshot(payload)
        legacy_size = _HEADER.size + int(count) * _EDGE_V1.size
        if len(payload) != legacy_size:
            raise ValueError("edge arena snapshot size mismatch")
        if int(count) > self.capacity:
            raise ValueError("snapshot exceeds edge arena capacity")
        self._shm.buf[:] = b"\0" * len(self._shm.buf)
        self._set_header(int(count), int(seq) + 1 if int(seq) & 1 else int(seq))
        for row in range(int(count)):
            offset = _HEADER.size + row * _EDGE_V1.size
            source_hi, source_lo, relation, target_hi, target_lo, support, watermark = _EDGE_V1.unpack_from(
                payload, offset
            )
            self.write(
                row,
                EdgeRecord(
                    MemoryUid(source_hi, source_lo),
                    int(relation),
                    MemoryUid(target_hi, target_lo),
                    int(support),
                    int(watermark),
                ),
            )

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
        _ACTION.pack_into(
            self._shm.buf,
            self._offset(row),
            1 if occupied else 0,
            u64(value.context_signature),
            int(value.action_id),
            int(value.support_count),
            float(value.score_sum),
            float(value.score_weight),
            u64(value.updated_watermark),
        )

    def read_slot(self, row: int) -> tuple[bool, ActionRecord]:
        occupied, context, action, support, score_sum, weight, watermark = _ACTION.unpack_from(
            self._shm.buf, self._offset(row)
        )
        return bool(occupied), ActionRecord(
            int(context), int(action), int(support), float(score_sum), float(weight), int(watermark)
        )

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