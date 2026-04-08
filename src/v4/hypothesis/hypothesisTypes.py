from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HypothesisEvidenceRefV4:
    ref_id: str
    ref_kind: str
    supports: bool = True

    def __post_init__(self) -> None:
        if not self.ref_id:
            raise ValueError("ref_id must be non-empty")
        if not self.ref_kind:
            raise ValueError("ref_kind must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisV4:
    hypothesis_id: str
    kind: str
    claimed_facts: tuple[str, ...]
    payload: dict[str, object]
    supporting_evidence: tuple[HypothesisEvidenceRefV4, ...]
    contradicting_evidence: tuple[HypothesisEvidenceRefV4, ...]
    confidence_band: str
    expiry_revision: int | None = None
    compatible_with: tuple[str, ...] = ()
    incompatible_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must be non-empty")
        if not self.kind:
            raise ValueError("kind must be non-empty")
        if not self.confidence_band:
            raise ValueError("confidence_band must be non-empty")
        if self.confidence_band not in {"low", "medium", "high"}:
            raise ValueError("confidence_band must be one of low, medium, high")
        for field_name in (
            "claimed_facts",
            "supporting_evidence",
            "contradicting_evidence",
            "compatible_with",
            "incompatible_with",
        ):
            if not isinstance(getattr(self, field_name), tuple):
                raise ValueError(f"{field_name} must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
