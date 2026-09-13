from __future__ import annotations

from dataclasses import dataclass

from .versions import ObjectRef, VersionTable


@dataclass(frozen=True, slots=True, order=True)
class ReadDependency:
    ref: ObjectRef
    version: int


@dataclass(frozen=True, slots=True)
class ReadSet:
    dependencies: tuple[ReadDependency, ...]

    @classmethod
    def build(cls, dependencies: tuple[ReadDependency, ...], *, maximum_size: int) -> "ReadSet":
        rows = tuple(sorted(set(dependencies)))
        if len(rows) > int(maximum_size):
            raise ValueError("read set exceeds configured bound")
        return cls(rows)

    def valid(self, versions: VersionTable) -> bool:
        return all(versions.get(row.ref) == row.version for row in self.dependencies)

