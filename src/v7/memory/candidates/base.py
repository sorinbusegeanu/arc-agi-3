from __future__ import annotations

from typing import Protocol, Sequence

from v7.memory.ids import MemoryId

CandidateKey = tuple[int, ...]


class CandidateProvider(Protocol):
    def role_candidates(self, keys: Sequence[CandidateKey], *, limit: int) -> tuple[tuple[MemoryId, ...], ...]: ...

    def concept_candidates(self, role_ids: Sequence[MemoryId], *, limit: int) -> tuple[tuple[MemoryId, ...], ...]: ...
