from __future__ import annotations

from collections.abc import Mapping, Sequence

from v7.memory.candidates.base import CandidateKey
from v7.memory.ids import MemoryId


class IndexedCpuCandidateProvider:
    """Bounded exact-index candidate provider; semantic validation remains external."""

    def __init__(
        self,
        *,
        role_index: Mapping[CandidateKey, Sequence[MemoryId]] | None = None,
        concept_index: Mapping[MemoryId, Sequence[MemoryId]] | None = None,
    ) -> None:
        self._role_index = role_index or {}
        self._concept_index = concept_index or {}

    def role_candidates(self, keys: Sequence[CandidateKey], *, limit: int) -> tuple[tuple[MemoryId, ...], ...]:
        limit = self._validate_limit(limit)
        return tuple(tuple(self._role_index.get(key, ()))[:limit] for key in keys)

    def concept_candidates(self, role_ids: Sequence[MemoryId], *, limit: int) -> tuple[tuple[MemoryId, ...], ...]:
        limit = self._validate_limit(limit)
        return tuple(tuple(self._concept_index.get(role_id, ()))[:limit] for role_id in role_ids)

    @staticmethod
    def _validate_limit(limit: int) -> int:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("candidate limit must be positive")
        return limit
