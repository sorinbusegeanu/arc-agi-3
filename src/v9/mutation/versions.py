from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class ObjectRef:
    kind: str
    uid_hi: int
    uid_lo: int = 0


class VersionTable:
    def __init__(self) -> None:
        self._versions: dict[ObjectRef, int] = {}

    def get(self, ref: ObjectRef) -> int:
        return int(self._versions.get(ref, 0))

    def bump(self, ref: ObjectRef) -> int:
        value = self.get(ref) + 1
        self._versions[ref] = value
        return value

    def state_dict(self) -> list[dict[str, object]]:
        return [
            {"kind": ref.kind, "uid_hi": ref.uid_hi, "uid_lo": ref.uid_lo, "version": version}
            for ref, version in sorted(self._versions.items())
        ]

    @classmethod
    def from_state_dict(cls, rows: list[dict[str, object]]) -> "VersionTable":
        result = cls()
        for row in rows:
            ref = ObjectRef(str(row["kind"]), int(row["uid_hi"]), int(row.get("uid_lo", 0)))
            result._versions[ref] = int(row["version"])
        return result

