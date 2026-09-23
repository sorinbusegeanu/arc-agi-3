from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Iterable


@dataclass(frozen=True, order=True, slots=True)
class DerivationTaskIdentity:
    signature: int
    target_support: int
    derivation_schema_version: int

    def __post_init__(self) -> None:
        if min(self.signature, self.target_support, self.derivation_schema_version) < 0:
            raise ValueError("derivation identity values must be non-negative")


@dataclass(frozen=True, slots=True)
class DerivationLease:
    identity: DerivationTaskIdentity
    lease_epoch: int
    attempt: int
    deadline: float


class DerivationLeaseManager:
    def __init__(
        self,
        *,
        pending_limit: int = 4096,
        inflight_limit: int = 256,
        completed_limit: int = 2048,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min(pending_limit, inflight_limit, completed_limit) <= 0:
            raise ValueError("derivation lease bounds must be positive")
        self.pending_limit = int(pending_limit)
        self.inflight_limit = int(inflight_limit)
        self.completed_limit = int(completed_limit)
        self._clock = clock
        self._pending: OrderedDict[DerivationTaskIdentity, int] = OrderedDict()
        self._inflight: dict[DerivationTaskIdentity, DerivationLease] = {}
        self._completed: OrderedDict[DerivationTaskIdentity, object] = OrderedDict()
        self._lease_epoch = 0

    @property
    def counts(self) -> tuple[int, int, int]:
        return len(self._pending), len(self._inflight), len(self._completed)

    def enqueue(self, identity: DerivationTaskIdentity) -> bool:
        if identity in self._pending or identity in self._inflight or identity in self._completed:
            return False
        if len(self._pending) >= self.pending_limit:
            raise OverflowError("derivation pending ceiling exceeded")
        self._pending[identity] = 0
        return True

    def lease(self, *, count: int = 1, duration_seconds: float = 30.0) -> tuple[DerivationLease, ...]:
        if count <= 0 or duration_seconds <= 0:
            raise ValueError("lease count and duration must be positive")
        available = max(0, self.inflight_limit - len(self._inflight))
        selected = min(int(count), available, len(self._pending))
        result = []
        for _ in range(selected):
            identity, prior_attempt = self._pending.popitem(last=False)
            self._lease_epoch += 1
            lease = DerivationLease(identity, self._lease_epoch, prior_attempt + 1, self._clock() + duration_seconds)
            self._inflight[identity] = lease
            result.append(lease)
        return tuple(result)

    def complete(self, lease: DerivationLease, result: object) -> bool:
        current = self._inflight.get(lease.identity)
        if current != lease:
            return False
        self._inflight.pop(lease.identity)
        self._completed[lease.identity] = result
        while len(self._completed) > self.completed_limit:
            self._completed.popitem(last=False)
        return True

    def accepts(self, lease: DerivationLease) -> bool:
        return self._inflight.get(lease.identity) == lease

    def retry(self, lease: DerivationLease) -> bool:
        current = self._inflight.get(lease.identity)
        if current != lease:
            return False
        if len(self._pending) >= self.pending_limit:
            raise OverflowError("derivation retry would exceed pending ceiling")
        self._inflight.pop(lease.identity)
        self._pending[lease.identity] = lease.attempt
        return True

    def expire(self) -> tuple[DerivationTaskIdentity, ...]:
        now = self._clock()
        expired = tuple(sorted(identity for identity, lease in self._inflight.items() if lease.deadline <= now))
        for identity in expired:
            lease = self._inflight.pop(identity)
            if len(self._pending) >= self.pending_limit:
                raise OverflowError("derivation retry would exceed pending ceiling")
            self._pending[identity] = lease.attempt
        return expired


@dataclass(frozen=True, slots=True)
class DerivationAggregate:
    schema_version: int
    evidence_ids: frozenset[str]
    sufficient_statistics: tuple[tuple[str, int], ...]
    maturity: int
    representative: str
    abstraction_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version <= 0 or self.maturity < 0:
            raise ValueError("invalid derivation aggregate version or maturity")
        keys = tuple(key for key, _ in self.sufficient_statistics)
        if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
            raise ValueError("sufficient statistics must have unique sorted keys")
        expected = derive_abstraction_id(self.schema_version, self.evidence_ids)
        if self.abstraction_id and self.abstraction_id != expected:
            raise ValueError("derivation abstraction identity conflict")
        if not self.abstraction_id:
            object.__setattr__(self, "abstraction_id", expected)


def derive_abstraction_id(schema_version: int, evidence_ids: Iterable[str]) -> str:
    raw = json.dumps(
        {"schema_version": int(schema_version), "evidence_ids": sorted(set(str(value) for value in evidence_ids))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def merge_derivation_aggregates(
    left: DerivationAggregate, right: DerivationAggregate
) -> DerivationAggregate:
    if left.schema_version != right.schema_version:
        raise ValueError("cannot merge derivation aggregates from different schemas")
    evidence = left.evidence_ids | right.evidence_ids
    # Statistics describe evidence contributions. Shared evidence makes blindly
    # adding two aggregates ambiguous, so only disjoint/equal aggregates merge.
    overlap = left.evidence_ids & right.evidence_ids
    if overlap and left != right:
        raise ValueError("conflicting derivation aggregates overlap evidence")
    if left == right:
        return left
    statistics = dict(left.sufficient_statistics)
    for key, value in right.sufficient_statistics:
        statistics[key] = statistics.get(key, 0) + int(value)
    representatives = tuple(value for value in (left.representative, right.representative) if value)
    return DerivationAggregate(
        left.schema_version,
        evidence,
        tuple(sorted(statistics.items())),
        max(left.maturity, right.maturity),
        min(representatives) if representatives else "",
    )


__all__ = [
    "DerivationAggregate",
    "DerivationLease",
    "DerivationLeaseManager",
    "DerivationTaskIdentity",
    "derive_abstraction_id",
    "merge_derivation_aggregates",
]
