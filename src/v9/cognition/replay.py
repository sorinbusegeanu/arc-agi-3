from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from v9.memory.identity import MemoryUid


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    uid: MemoryUid
    fitness: float


def select_replay(candidates: tuple[ReplayCandidate, ...], *, limit: int) -> tuple[ReplayCandidate, ...]:
    if limit <= 0:
        raise ValueError("replay limit must be positive")
    return tuple(sorted(candidates, key=lambda row: (-row.fitness, row.uid))[:limit])


@dataclass(frozen=True, slots=True)
class ReplayResult:
    selected: int
    processed: int
    new_memories: int
    revisions: int
    correspondences: int


class ReplayScheduler:
    """Bounded active cognition over existing evidence only."""

    def __init__(self, *, candidate_limit: int) -> None:
        if candidate_limit <= 0:
            raise ValueError("replay candidate limit must be positive")
        self.candidate_limit = int(candidate_limit)
        self.selected = 0
        self.processed = 0
        self.new_memories = 0
        self.revisions = 0
        self.correspondences = 0

    def run(self, candidates: tuple[ReplayCandidate, ...], processor: Callable[[ReplayCandidate], tuple[int, int, int]]) -> ReplayResult:
        rows = select_replay(candidates, limit=self.candidate_limit)
        new_memories = revisions = correspondences = 0
        processed = 0
        for row in rows:
            created, revised, matched = processor(row)
            new_memories += int(created)
            revisions += int(revised)
            correspondences += int(matched)
            processed += 1
        self.selected += len(rows)
        self.processed += processed
        self.new_memories += new_memories
        self.revisions += revisions
        self.correspondences += correspondences
        return ReplayResult(len(rows), processed, new_memories, revisions, correspondences)

    def state_dict(self) -> dict[str, int]:
        return {
            "candidate_limit": self.candidate_limit,
            "selected": self.selected,
            "processed": self.processed,
            "new_memories": self.new_memories,
            "revisions": self.revisions,
            "correspondences": self.correspondences,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ReplayScheduler":
        result = cls(candidate_limit=int(state["candidate_limit"]))
        for name in ("selected", "processed", "new_memories", "revisions", "correspondences"):
            setattr(result, name, int(state.get(name, 0)))
        return result
