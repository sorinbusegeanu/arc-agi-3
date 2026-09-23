from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable

_MASK64 = (1 << 64) - 1
_SCHEMA = b"arc-agi3-hydra-v9.5"


def stable_u64(*parts: object, person: bytes = b"v9-stable") -> int:
    """Return a deterministic unsigned identity without semantic interpretation."""
    digest = blake2b(digest_size=8, person=person[:16])
    for part in parts:
        if isinstance(part, bytes):
            raw = part
        elif isinstance(part, str):
            raw = part.encode("utf-8")
        else:
            raw = int(part).to_bytes(16, "little", signed=True)
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
    return int.from_bytes(digest.digest(), "little")


@dataclass(frozen=True, order=True, slots=True)
class Uid:
    hi: int
    lo: int

    def __post_init__(self) -> None:
        if not (0 <= int(self.hi) <= _MASK64 and 0 <= int(self.lo) <= _MASK64):
            raise ValueError("UID components must be uint64")

    @classmethod
    def derive(cls, domain: str, *parts: object) -> "Uid":
        digest = blake2b(digest_size=16, person=b"arc-hydra-v9")
        digest.update(_SCHEMA)
        raw_domain = domain.encode("utf-8")
        digest.update(len(raw_domain).to_bytes(2, "little"))
        digest.update(raw_domain)
        for part in parts:
            if isinstance(part, bytes):
                raw = part
            elif isinstance(part, str):
                raw = part.encode("utf-8")
            else:
                raw = int(part).to_bytes(16, "little", signed=True)
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
        value = digest.digest()
        return cls(int.from_bytes(value[:8], "little"), int.from_bytes(value[8:], "little"))

    @classmethod
    def zero(cls) -> "Uid":
        return cls(0, 0)

    @property
    def is_zero(self) -> bool:
        return not (self.hi or self.lo)

    def shard(self, count: int) -> int:
        if int(count) <= 0:
            raise ValueError("shard count must be positive")
        return int((self.hi ^ self.lo) % int(count))

    def hex(self) -> str:
        return f"{self.hi:016x}{self.lo:016x}"


class MemoryUid(Uid):
    @classmethod
    def from_key(cls, level: int, memory_type: int, key_parts: Iterable[int]) -> "MemoryUid":
        return cls.derive("memory", int(level), int(memory_type), *(int(v) for v in key_parts))


class EventUid(Uid):
    @classmethod
    def from_producer(cls, producer_id: int, sequence: int) -> "EventUid":
        if min(int(producer_id), int(sequence)) < 0:
            raise ValueError("producer identity and sequence must be non-negative")
        return cls.derive("event", int(producer_id), int(sequence))


@dataclass(frozen=True, order=True, slots=True)
class ScalarUid:
    value: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.value) <= _MASK64:
            raise ValueError("identity must be uint64")


class EnvironmentFamilyId(ScalarUid):
    pass


class EnvironmentTypeId(ScalarUid):
    pass


class EnvironmentConfigId(ScalarUid):
    pass


class EnvironmentInstanceId(ScalarUid):
    pass


class EpisodeId(ScalarUid):
    pass


class ModalityId(ScalarUid):
    pass


class SymbolVocabularyId(ScalarUid):
    pass


class SymbolStreamId(ScalarUid):
    pass


class SymbolId(ScalarUid):
    pass


class LineageUid(ScalarUid):
    pass


class ContextScopeId(ScalarUid):
    pass
