from __future__ import annotations

from v9.mutation.proposals import MutationProposal
from v9.mutation.transactions import MutationResult

from .publication import CanonicalGraph
from .rings import BoundedRing


class PartitionReducer:
    def __init__(self, partition_id: int, graph: CanonicalGraph, capacity: int) -> None:
        self.partition_id = int(partition_id)
        self.graph = graph
        self.queue: BoundedRing[MutationProposal] = BoundedRing(capacity)

    def submit(self, proposal: MutationProposal) -> bool:
        if self.partition_id not in proposal.target_partitions:
            raise ValueError("proposal is not owned by this reducer")
        return self.queue.put(proposal)

    def drain(self) -> tuple[MutationResult, ...]:
        rows = sorted(self.queue.drain(), key=lambda row: row.ordering_key)
        return tuple(self.graph.publish(row) for row in rows)

