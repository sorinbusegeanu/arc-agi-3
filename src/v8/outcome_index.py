from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.model import MemoryUid


@dataclass(frozen=True, slots=True)
class OutcomeDescriptor:
    uid: MemoryUid
    future_bucket: int
    changed_bucket: int
    context_bucket: int = 0


class OutcomeCandidateIndex:
    """Bound M6 exact comparisons to descriptor buckets instead of all outcomes."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[int, int, int], list[OutcomeDescriptor]] = defaultdict(list)

    def add(self, descriptor: OutcomeDescriptor) -> None:
        key = (descriptor.future_bucket, descriptor.changed_bucket, descriptor.context_bucket)
        if all(existing.uid != descriptor.uid for existing in self._buckets[key]):
            self._buckets[key].append(descriptor)

    def candidates(
        self,
        *,
        future_bucket: int,
        changed_bucket: int,
        context_bucket: int = 0,
        limit: int = 32,
    ) -> tuple[OutcomeDescriptor, ...]:
        exact = list(self._buckets.get((int(future_bucket), int(changed_bucket), int(context_bucket)), ()))
        if len(exact) >= int(limit):
            return tuple(exact[: int(limit)])
        # Adjacent future-option buckets are the only relaxed candidates.
        for delta in (-1, 1):
            exact.extend(self._buckets.get((int(future_bucket) + delta, int(changed_bucket), int(context_bucket)), ()))
            if len(exact) >= int(limit):
                break
        return tuple(exact[: int(limit)])
