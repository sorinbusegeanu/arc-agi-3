from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DomainSliceV4:
    domain_name: str
    is_present: bool
    summary: dict[str, object]

    def __post_init__(self) -> None:
        if not self.domain_name:
            raise ValueError("domain_name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompositionSnapshotReferenceV4:
    revision: int | None = None
    domain_count: int = 0
    present_domain_names: tuple[str, ...] = ()
    cross_domain_effect_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComposedDomainStateV4:
    revision: int = 0
    state_key: str | None = None
    domain_slices: tuple[DomainSliceV4, ...] = ()
    cross_domain_effect_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot_reference(self) -> CompositionSnapshotReferenceV4:
        return CompositionSnapshotReferenceV4(
            revision=self.revision,
            domain_count=len(self.domain_slices),
            present_domain_names=tuple(slice_.domain_name for slice_ in self.domain_slices if slice_.is_present),
            cross_domain_effect_count=len(self.cross_domain_effect_codes),
        )
