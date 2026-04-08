from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .hypothesisTypes import HypothesisV4


@dataclass(frozen=True)
class HypothesisEvidenceLedgerEntryV4:
    entry_id: str
    hypothesis_id: str
    ref_id: str
    ref_kind: str
    supports: bool
    revision: int

    def __post_init__(self) -> None:
        for field_name in ("entry_id", "hypothesis_id", "ref_id", "ref_kind"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HypothesisEvidenceLedgerV4:
    def __init__(self) -> None:
        self._entries: list[HypothesisEvidenceLedgerEntryV4] = []

    def append_for_hypothesis(self, revision: int, hypothesis: HypothesisV4) -> None:
        refs = list(hypothesis.supporting_evidence) + list(hypothesis.contradicting_evidence)
        for index, ref in enumerate(refs):
            self._entries.append(
                HypothesisEvidenceLedgerEntryV4(
                    entry_id=f"evidence:{revision}:{hypothesis.hypothesis_id}:{index}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    ref_id=ref.ref_id,
                    ref_kind=ref.ref_kind,
                    supports=ref.supports,
                    revision=int(revision),
                )
            )

    def entries(self) -> tuple[HypothesisEvidenceLedgerEntryV4, ...]:
        return tuple(self._entries)

    def reset(self) -> None:
        self._entries.clear()
