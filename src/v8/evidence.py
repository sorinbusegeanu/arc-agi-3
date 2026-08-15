from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

from v8.model import MemoryUid


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    memory_uid_hi: int
    memory_uid_lo: int
    evidence_kind: str
    evidence_available_watermark: int
    decision_watermark: int
    raw_value: float
    normalized_value: float
    developmental_stage: int
    validation_state: int
    source_game_hash: int = 0

    @classmethod
    def for_uid(
        cls,
        evidence_id: str,
        uid: MemoryUid,
        *,
        evidence_kind: str,
        watermark: int,
        raw_value: float,
        normalized_value: float,
        developmental_stage: int,
        validation_state: int,
        source_game_hash: int = 0,
    ) -> "EvidenceRecord":
        return cls(
            str(evidence_id),
            int(uid.hi),
            int(uid.lo),
            str(evidence_kind),
            int(watermark),
            int(watermark),
            float(raw_value),
            float(normalized_value),
            int(developmental_stage),
            int(validation_state),
            int(source_game_hash),
        )


class EvidenceLedger:
    """Append-only in-RAM scientific ledger; immutable cuts are used for reporting."""

    def __init__(self) -> None:
        self._rows: list[EvidenceRecord] = []
        self._ids: set[str] = set()
        self._lock = Lock()

    def append(self, row: EvidenceRecord) -> bool:
        if row.evidence_available_watermark > row.decision_watermark:
            raise ValueError("future evidence cannot influence an earlier decision")
        with self._lock:
            if row.evidence_id in self._ids:
                return False
            self._ids.add(row.evidence_id)
            self._rows.append(row)
            return True

    def cut(self, watermark: int) -> tuple[EvidenceRecord, ...]:
        with self._lock:
            return tuple(row for row in self._rows if row.evidence_available_watermark <= int(watermark))

    def export_jsonl(self, path: str | Path, *, watermark: int | None = None) -> None:
        rows = tuple(self._rows) if watermark is None else self.cut(int(watermark))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
