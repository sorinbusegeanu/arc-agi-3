from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock

from .proposals import MutationProposal


class MutationOutcome(str, Enum):
    ACCEPTED = "ACCEPTED"
    STALE_READ_SET = "STALE_READ_SET"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class MutationResult:
    proposal_uid: int
    outcome: MutationOutcome
    graph_generation: int


class TransactionCoordinator:
    """Coordinates deterministic atomic publication across one or more partitions."""

    def __init__(self, partition_count: int) -> None:
        if partition_count <= 0:
            raise ValueError("partition count must be positive")
        self._locks = tuple(RLock() for _ in range(partition_count))

    def locked(self, partitions: tuple[int, ...]):
        coordinator = self

        class _LockSet:
            def __enter__(self):
                for partition in partitions:
                    coordinator._locks[partition].acquire()

            def __exit__(self, *_args: object):
                for partition in reversed(partitions):
                    coordinator._locks[partition].release()

        return _LockSet()

