from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .observedFacts import InferredLocalFactV4, ObservedCellFactV4
from .unknownFacts import UnknownCellFactV4


@dataclass(frozen=True)
class BeliefSnapshotReferenceV4:
    revision: int | None = None
    observed_cell_count: int = 0
    unknown_cell_count: int = 0
    frontier_cell_count: int = 0
    inferred_fact_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeliefStateV4:
    revision: int = 0
    state_key: str | None = None
    observed_cells: tuple[ObservedCellFactV4, ...] = ()
    unknown_cells: tuple[UnknownCellFactV4, ...] = ()
    inferred_facts: tuple[InferredLocalFactV4, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot_reference(self) -> BeliefSnapshotReferenceV4:
        return BeliefSnapshotReferenceV4(
            revision=self.revision,
            observed_cell_count=len(self.observed_cells),
            unknown_cell_count=len(self.unknown_cells),
            frontier_cell_count=sum(1 for cell in self.unknown_cells if cell.frontier),
            inferred_fact_count=len(self.inferred_facts),
        )


class BeliefStoreV4:
    def __init__(
        self,
        *,
        max_observed_cells: int = 256,
        max_unknown_cells: int = 256,
        max_inferred_facts: int = 64,
        max_evidence_refs: int = 64,
    ) -> None:
        self.max_observed_cells = int(max_observed_cells)
        self.max_unknown_cells = int(max_unknown_cells)
        self.max_inferred_facts = int(max_inferred_facts)
        self.max_evidence_refs = int(max_evidence_refs)
        self._state = BeliefStateV4()

    def reset(self) -> None:
        self._state = BeliefStateV4()

    def snapshot(self) -> BeliefStateV4:
        return self._state

    def replace(self, state: BeliefStateV4) -> BeliefStateV4:
        if not isinstance(state, BeliefStateV4):
            raise ValueError("state must be BeliefStateV4")
        self._state = BeliefStateV4(
            revision=state.revision,
            state_key=state.state_key,
            observed_cells=tuple(state.observed_cells[: self.max_observed_cells]),
            unknown_cells=tuple(state.unknown_cells[: self.max_unknown_cells]),
            inferred_facts=tuple(state.inferred_facts[: self.max_inferred_facts]),
            evidence_refs=tuple(state.evidence_refs[: self.max_evidence_refs]),
        )
        return self._state
