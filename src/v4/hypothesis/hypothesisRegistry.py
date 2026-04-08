from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .evidenceLedger import HypothesisEvidenceLedgerEntryV4, HypothesisEvidenceLedgerV4
from .hypothesisContracts import HypothesisSnapshotReferenceV4
from .hypothesisPruner import HypothesisPrunerV4
from .hypothesisTypes import HypothesisV4


@dataclass(frozen=True)
class HypothesisStateV4:
    revision: int = 0
    hypotheses: tuple[HypothesisV4, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot_reference(self) -> HypothesisSnapshotReferenceV4:
        return HypothesisSnapshotReferenceV4(
            revision=self.revision,
            hypothesis_count=len(self.hypotheses),
            low_count=sum(1 for item in self.hypotheses if item.confidence_band == "low"),
            medium_count=sum(1 for item in self.hypotheses if item.confidence_band == "medium"),
            high_count=sum(1 for item in self.hypotheses if item.confidence_band == "high"),
        )


class HypothesisRegistryV4:
    def __init__(
        self,
        pruner: HypothesisPrunerV4 | None = None,
        ledger: HypothesisEvidenceLedgerV4 | None = None,
    ) -> None:
        self.pruner = pruner if pruner is not None else HypothesisPrunerV4()
        self.ledger = ledger if ledger is not None else HypothesisEvidenceLedgerV4()
        self._state = HypothesisStateV4()

    def reset(self) -> None:
        self._state = HypothesisStateV4()
        self.ledger.reset()

    def snapshot(self) -> HypothesisStateV4:
        return self._state

    def replace(self, state: HypothesisStateV4) -> HypothesisStateV4:
        if not isinstance(state, HypothesisStateV4):
            raise ValueError("state must be HypothesisStateV4")
        self._state = state
        return self._state

    def ledger_entries(self) -> tuple[HypothesisEvidenceLedgerEntryV4, ...]:
        return self.ledger.entries()

    def update(self, revision: int, hypotheses: tuple[HypothesisV4, ...]) -> HypothesisStateV4:
        pruned = self.pruner.prune(revision, hypotheses)
        self._state = HypothesisStateV4(revision=int(revision), hypotheses=pruned)
        for hypothesis in pruned:
            self.ledger.append_for_hypothesis(revision, hypothesis)
        return self._state
